import os
os.environ["HF_HOME"] = "/shared/data3/pk36/.cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import argparse
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import json_repair
import re

from search import search_semantic_scholar, collect_snippets
from prompts import create_goal_decomposition_prompt, decompositon_schema
from classes import ResearchProblem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve & analyze interdisciplinary research for a given .")
    parser.add_argument("--problem_file", type=str, default="data/ai_ideas/adaptive_confidence-guided_prompting.txt", help="Path to the proposal text file.")
    parser.add_argument("--target_domain", type=str, default="Computer Science", help="The user's desired target domain (what field they are in).")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="LLM model name or path.")
    parser.add_argument("--output_file", type=str, default="output/claimspect_proposal_analysis.json", help="Path to output JSON file.")
    args = parser.parse_args()

    # Read in the "1. Problem Statement:" line from the problem file (can use regex). The entire problem statement value is on the same line.
    if os.path.exists(args.problem_file):
        with open(args.problem_file, "r") as f:
            problem_file_text = f.read()
            problem_statement = re.search(r"1\. Problem Statement:\s*(.*)", problem_file_text).group(1).strip()
        print(f"Problem Statement: {problem_statement}")
    else:
        print(f"File {args.problem_file} does not exist!")

    # Batch offline inference with vllm with structured (json) output
    llm = LLM(model=args.model_name, tensor_parallel_size=2)

    # 1. Generate sub-questions for goal
    # 2. For each sub-question → identify associated domains and search queries
    prompt = create_goal_decomposition_prompt(problem_statement)
    messages = [
        {"role": "user", "content": prompt}
    ]
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.7,
        top_p=0.95,
        structured_outputs=StructuredOutputsParams(
            json=decompositon_schema
        ),
    )
    response = llm.chat([messages], sampling_params, chat_template_kwargs={"enable_thinking": False})
    decomposition_output = json_repair.loads(response[0].outputs[0].text)

    research_problem = ResearchProblem(decomposition_json=decomposition_output)

    ## a. Retrieve papers and associated snippets
    ### For each of the subquestions, in each of the returned domains, for each of the queries, call search_semantic_scholar(query, coarse_domain, year=None) to get results from Semantic Scholar API. Collect snippets using collect_snippets(response) and populate them into the papers variable of Question2Domain.
    for sub_question in research_problem.sub_questions:
        # Prioritize same domain first
        ## Search
        papers = {}
        for query in research_problem.target_domain.question2queries[sub_question]:
            response = search_semantic_scholar(query, research_problem.target_domain.domain_name)
            snippets = collect_snippets(response)
            if len(snippets) > 0:
                papers.update(snippets)

        research_problem.target_domain.add_question_papers(sub_question, papers)

        ## Judge whether domain snippets are relevant and thorough for addressing the subquestion (if yes, then do not progress to external domains); by sufficiency, we mean that the 


    ## b. Judge whether each domain is relevant and adequate to the sub-question based on whether its papers address the sub-question; if they are relevant, then abstract out the high-level, domain-agnostic takeaways from those papers relevant to the sub-question.

    ## c. Frame the takeaways to the target domain (single prompt or a multi-persona collaboration)



    # 3. **Multiple routes: *Given the target domain-specific framing of each source domain’s takeaways:***
    #     1. Re-visit the existing sub-questions to see which are resolved → if progress is made, then this is a positive signal (retain the domain takeaways)
    #     2. Compare/rank the different source domains: If domain **A** is simply a subset of **B**, then discard **A**.
    #         1. Optimization problem: Find the minimal # of sets which cover all sub-questions.
    #     3. For the completely unaddressed sub-questions, decompose them further and continue from Step (2)