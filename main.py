import os
os.environ["HF_HOME"] = "/shared/data3/pk36/.cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

from prompts import PromptBuilder
from search import search_semantic_scholar, collect_snippets
from classes import Theme, Domain, Proposal, theme_json_schema, domain_json_schema, gaps_json_schema, bridge_idea_json_schema
import argparse
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import json_repair
import json
import time

if __name__ == "__main__":
    # argparse: dimensions -> default value is list: ["goal_motivation", "method", "dataset", "experiments"]
    parser = argparse.ArgumentParser(description="Extract themes from a research proposal.")
    parser.add_argument("--proposal_file", type=str, default="data/claimspect.txt", help="Path to the proposal text file.")
    # parser.add_argument(
    #     "--dimensions",
    #     nargs="+",
    #     default=["goal_motivation", "method", "dataset", "experiments"],
    #     help="List of research dimensions to extract themes for.",
    # )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="LLM model name or path.")
    parser.add_argument("--output_file", type=str, default="output/claimspect_proposal_analysis.json", help="Path to output JSON file.")
    args = parser.parse_args()
    args.dimensions = ["goal_motivation", "method", "dataset", "experiments"]

    # First read in a proposal (from data/ directory) as input
    with open(args.proposal_file, "r") as f:
        proposal_text = f.read()
    
    proposal_text = proposal_text.strip()
    start_time = time.perf_counter()

    # Next, instantiate a Proposal instance
    proposal = Proposal(text=proposal_text, themes={})
    proposal_text = proposal.normalize()

    # Batch offline inference with vllm with structured (json) output
    llm = LLM(model=args.model_name, tensor_parallel_size=2)

    # If the output file already exists, directly create/populate the Proposal instance from the JSON file
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            proposal_json = json.load(f)
        # Populate the Proposal instance from the JSON
        for theme_data in proposal_json.get("themes", {}).values():
            theme_instance = Theme(
                theme=theme_data["theme"],
                dimension=theme_data.get("dimension"),
                segments=theme_data.get("segments", []),
                domains=[
                    Domain(
                        theme=domain_data["theme"],
                        coarse_domain=domain_data.get("coarse_domain"),
                        fine_domain=domain_data.get("fine_domain"),
                        rationale=domain_data.get("rationale"),
                        queries=domain_data.get("queries", []),
                        snippets=domain_data.get("snippets", {})
                    )
                    for domain_data in theme_data.get("domains", [])
                ]
            )
            proposal.themes[theme_instance.theme] = theme_instance
        print(f"Loaded proposal analysis from {args.output_file}.")
    else:
        print(f"Processing proposal from {args.proposal_file}.")

        # Then pass into PromptBuilder.theme_extraction_prompt(proposal_text: str, dimensions: List[str]) -> this gets us the following JSON output:
        prompt = PromptBuilder.theme_extraction_prompt(proposal_text, args.dimensions)
        messages = [
            {"role": "user", "content": prompt}
        ]
        sampling_params = SamplingParams(
            max_tokens=2048,
            temperature=0.6,
            top_p=0.95,
            structured_outputs=StructuredOutputsParams(
                json=theme_json_schema
            ),
        )
        response = llm.chat([messages], sampling_params, chat_template_kwargs={"enable_thinking": False})
        themes_json = json_repair.loads(response[0].outputs[0].text)
        
        # For each one of the themes, instantiate a Theme instance and populate the theme, dimension, and segments fields.
        for theme_data in themes_json["themes"]:
            theme_instance = Theme(
                theme=theme_data["theme"],
                dimension=theme_data.get("dimension"),
                segments=theme_data.get("segments", []),
            )
            proposal.themes[theme_instance.theme] = theme_instance

        # Pass the themes JSON as input into PromptBuilder.domain_discovery_prompt(theme), which has the following format:
        domain_prompt = PromptBuilder.domain_discovery_prompt(json.dumps(themes_json, indent=4))
        messages = [
            {"role": "user", "content": domain_prompt}
        ]
        sampling_params = SamplingParams(
            max_tokens=4096,
            temperature=0.6,
            top_p=0.95,
            structured_outputs=StructuredOutputsParams(
                json=domain_json_schema
            ),
        )
        attempts = 0
        domains_json = {}
        while (attempts < 3) and (not domains_json):
            try:
                domain_response = llm.chat([messages], sampling_params, chat_template_kwargs={"enable_thinking": False})
                domains_json = json_repair.loads(domain_response[0].outputs[0].text)
                assert domains_json is not None
                print(domains_json)
            except Exception as e:
                print(f"Attempt {attempts+1} failed with error: {e}. Retrying...")
                attempts += 1
        
        # For each of the themes, in each of the returned domains, for each of the queries, call search_semantic_scholar(query, coarse_domain, year=None) to get results from Semantic Scholar API. Collect snippets using collect_snippets(response) and populate the snippets field in the Domain instance accordingly.
        for theme_text, domain_list in domains_json.items():
            theme_instance = proposal.themes.get(theme_text)
            if not theme_instance:
                continue
            for domain_data in domain_list:
                queries = domain_data.get("queries", [])
                domain_instance = Domain(
                    theme=theme_text,
                    coarse_domain=domain_data.get("coarse_domain"),
                    fine_domain=domain_data.get("fine_grained_area"),
                    rationale=domain_data.get("rationale"),
                    queries=queries
                )
                for query in queries:
                    response = search_semantic_scholar(query, domain_instance.coarse_domain)
                    snippets = collect_snippets(response)
                    domain_instance.snippets = snippets
                theme_instance.domains.append(domain_instance)
        
        # Serialize the Proposal instance (with all nested Themes and Domains) to a JSON file.
        with open(args.output_file, "w") as f:
            json.dump(proposal.to_dict(), f, indent=4)
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")

    # For each of the domains within each of the themes in the Proposal instance, have the respective persona generate (a) gaps in the proposal that their domain could help address, and (b) specific open research questions from their domain that the proposal has the potential to answer.

    ## Batch (a) gaps prompts
    gaps_messages = []
    gaps_domain_map = []
    for theme in proposal.themes.values():
        for domain in theme.domains:
            gaps_prompt = PromptBuilder.proposal_gaps_prompt(theme, domain, proposal.text)
            gaps_messages.append([{"role": "user", "content": gaps_prompt}])
            gaps_domain_map.append((theme, domain))
    
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        structured_outputs=StructuredOutputsParams(
            json=gaps_json_schema
        )
    )
    gaps_responses = llm.chat(gaps_messages, sampling_params, chat_template_kwargs={"enable_thinking": False})
    
    for i, (theme, domain) in enumerate(gaps_domain_map):
        gaps_json = json_repair.loads(gaps_responses[i].outputs[0].text)
        domain.gap = gaps_json

    ## Batch (b) bridge idea prompts
    bridge_messages = []
    bridge_domain_map = []
    for theme in proposal.themes.values():
        for domain in theme.domains:
            domain_prompt = PromptBuilder.proposal_domain_bridge_prompt(theme, domain, proposal.text)
            bridge_messages.append([{"role": "user", "content": domain_prompt}])
            bridge_domain_map.append((theme, domain))
    
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        structured_outputs=StructuredOutputsParams(
            json=bridge_idea_json_schema
        )
    )
    bridge_responses = llm.chat(bridge_messages, sampling_params, chat_template_kwargs={"enable_thinking": False})
    
    for i, (theme, domain) in enumerate(bridge_domain_map):
        bridge_idea_json = json_repair.loads(bridge_responses[i].outputs[0].text)
        domain.bridge_idea = bridge_idea_json

    # Update the Proposal instance (with all nested Themes and Domains and now gaps and research questions) to a JSON file.
    with open(args.output_file, "w") as f:
        json.dump(proposal.to_dict(), f, indent=4)
    

