import os
os.environ["HF_HOME"] = "/shared/data3/pk36/.cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

from prompts import PromptBuilder
from search import search_semantic_scholar, collect_snippets
from classes import Theme, Domain, Proposal, Collaboration, theme_json_schema, domain_json_schema, gaps_json_schema, bridge_idea_json_schema
import argparse
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import json_repair
import json
import time

def domain_specific_collaborator(proposal, themes_json):
    start_time = time.perf_counter()
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
        for domain_data in [domain_list]:
            queries = domain_data.get("queries", [])
            domain_instance = Domain(
                theme=theme_text,
                coarse_domain=domain_data.get("coarse_domain"),
                fine_domain=domain_data.get("fine_grained_area"),
                rationale=domain_data.get("rationale"),
                queries=queries
            )
            for query in queries:
                response = search_semantic_scholar(query + ' ' + domain_instance.fine_domain.lower(), domain_instance.coarse_domain)
                snippets = collect_snippets(response)
                domain_instance.snippets = snippets
            theme_instance.domains.append(domain_instance)
    
    # Serialize the Proposal instance (with all nested Themes and Domains) to a JSON file.
    with open(os.path.join(args.output_directory, args.proposal_file, f"{args.proposal_file}_metadata.json"), "w") as f:
        json.dump(proposal.to_dict(), f, indent=4)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    return proposal

def simulate_collaboration(llm, proposal, turns=10):
    # via vLLM, have two agents collaborate on improving the proposal
    curr_turn = 0
    conversation_history = []

    while curr_turn < turns:
        if curr_turn % 2 == 0:
            # Proposal agent's turn
            prompt = PromptBuilder.proposal_agent_prompt(proposal.text, conversation_history)
        else:
            # Collaborator agent's turn
            prompt = PromptBuilder.general_collab_agent_prompt(proposal.text, conversation_history)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        sampling_params = SamplingParams(
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
        )
        response = llm.chat([messages], sampling_params, chat_template_kwargs={"enable_thinking": False})
        agent_message = response[0].outputs[0].text.strip()
        conversation_history.append(agent_message)
        curr_turn += 1
    
    return conversation_history


def simulate_domain_specific_collaboration(proposal, turns=10):
    # via vLLM, have one agent per external domain collaborate with an agent who has written the proposal on improving it
    curr_turn = 0
    convo_histories = {str(domain): [] for theme in proposal.themes.values() for domain in theme.domains}
    
    while curr_turn < turns:
        # Collect all prompts and domain keys for batch processing
        batch_prompts = []
        batch_domain_keys = []
        
        for theme in proposal.themes.values():
            for domain in theme.domains:
                domain_key = str(domain)
                if curr_turn % 2 == 0:
                    # Proposal agent's turn
                    prompt = PromptBuilder.proposal_agent_prompt(proposal.text, convo_histories[domain_key])
                else:
                    # Domain-specific Collaborator agent's turn
                    prompt = PromptBuilder.external_domain_collab_agent_prompt(proposal.text, domain, convo_histories[domain_key])
                
                batch_prompts.append([{"role": "user", "content": prompt}])
                batch_domain_keys.append(domain_key)
        
        # Perform batch inference
        sampling_params = SamplingParams(
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
        )
        responses = llm.chat(batch_prompts, sampling_params, chat_template_kwargs={"enable_thinking": False})
        
        # Process all responses
        for domain_key, response in zip(batch_domain_keys, responses):
            agent_message = response.outputs[0].text.strip()
            convo_histories[domain_key].append(agent_message)
        
        curr_turn += 1

    return convo_histories


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description="Simulate a conversation between two agents on improving a proposal.")
    parser.add_argument("--proposal_directory", type=str, default="data/", help="Directory containing proposal text files.")
    parser.add_argument("--output_directory", type=str, default="output/", help="Directory to save output JSON files.")
    parser.add_argument("--proposal_file", type=str, default="claimspect", help="Path to the proposal text file.")
    parser.add_argument("--max_turns", type=int, default=10, help="Maximum number of turns in the collaboration.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Name of the LLM model to use.")
    args = parser.parse_args()
    args.dimensions = ["goal_motivation", "method", "dataset", "experiments"]

    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(args.output_directory, args.proposal_file), exist_ok=True)

    # First read in a proposal (from data/ directory) as input
    with open(os.path.join(args.proposal_directory, f"{args.proposal_file}.txt"), "r") as f:
        proposal_text = f.read()
    
    proposal_text = proposal_text.strip()

    # Batch offline inference with vllm with structured (json) output
    llm = LLM(model=args.model_name, tensor_parallel_size=8)
    
    start_time = time.perf_counter()

    # Check if the metadata JSON file already exists
    metadata_file_path = os.path.join(args.output_directory, args.proposal_file, f"{args.proposal_file}_metadata.json")
    if os.path.exists(metadata_file_path):
        with open(metadata_file_path, "r") as f:
            proposal_json = json.load(f)
        # Populate the Proposal instance from the JSON
        proposal = Proposal(text=proposal_text, themes={})
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
        print(f"Loaded proposal analysis from {metadata_file_path}.")

    else:
        # Next, instantiate a Proposal instance
        proposal = Proposal(text=proposal_text, themes={})
        proposal_text = proposal.normalize()

        # Extract multi-dimensional themes from the proposal
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
        
        proposal = domain_specific_collaborator(proposal, themes_json)

    # Given the proposal, have the two models work together on improving the proposal (where one agent has written the proposal and the other is coming in as a collaborator)

    ## SIMULATION #1: general collaborator
    convo_history = simulate_collaboration(llm, proposal, turns=args.max_turns)
    collab = Collaboration(proposal=proposal, collaborators={"general_collaborator": convo_history})

    # Serialize the Collaboration instance (with nested Proposal, Themes, Domains, and conversation histories) to a JSON file.
    with open(os.path.join(args.output_directory, args.proposal_file, f"{args.proposal_file}_collaboration_turns_{args.max_turns}.json"), "w") as f:
        json.dump({
            "proposal": collab.proposal.to_dict(),
            "collaborators": collab.collaborators
        }, f, indent=4)

    ## SIMULATION #2 (domain-specific): one of the collaborators is from an external domain
    
    collab.proposal = proposal
    convo_histories = simulate_domain_specific_collaboration(proposal, turns=args.max_turns)
    # Update collaborators
    collab.collaborators.update(convo_histories)
    
    # Serialize the Collaboration instance (with nested Proposal, Themes, Domains, and conversation histories) to a JSON file.
    with open(os.path.join(args.output_directory, args.proposal_file, f"{args.proposal_file}_collaboration_turns_{args.max_turns}.json"), "w") as f:
        json.dump({
            "proposal": collab.proposal.to_dict(),
            "collaborators": collab.collaborators
        }, f, indent=4)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
    