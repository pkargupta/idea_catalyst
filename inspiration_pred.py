import os
os.environ["HF_HOME"] = "/shared/data3/pk36/.cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import argparse
from vllm import LLM
import json
from tqdm import tqdm
import time
from collections import defaultdict

from prompts import (
    create_initial_decomposition_prompt,
    create_target_domain_analysis_prompt,
    create_cross_domain_query_prompt,
    create_cross_domain_analysis_prompt,
    initial_decomposition_schema,
    target_domain_analysis_schema,
    cross_domain_queries_schema,
    cross_domain_analysis_schema
)
from classes import ResearchProblem
from utils import batch_llm_inference, retrieve_papers_for_question

def decompose(args, llm, problem_statement):
    prompt = create_initial_decomposition_prompt(problem_statement, args.target_domain)
    messages = [{"role": "user", "content": prompt}]

    decomposition_outputs = batch_llm_inference(
        llm, 
        [messages], 
        initial_decomposition_schema,
        temperature=args.temp
    )
    decomposition_output = decomposition_outputs[0]

    if decomposition_output is None:
        print("Failed to get decomposition output!")
        return
    else:
        # Create ResearchProblem object
        research_problem = ResearchProblem.from_initial_decomposition(
            decomposition_output, 
            args.target_domain
        )

        print(f"Generated {len(research_problem.research_questions)} research questions:")
        for q in research_problem.research_questions:
            print(f"  - {q.id}:\n\t\t-{q.domain_specific_question}\n\t\t-{q.domain_agnostic_question}")
    
    return research_problem

def explore_target_domain(args, llm, research_problem):
    # Step 2a: Retrieve papers for all questions in target domain
    print("\n2a. Retrieving papers from target domain...")
    for question in research_problem.research_questions:
        print(f"  Retrieving for {question.id}...")
        papers = retrieve_papers_for_question(
            question,
            research_problem.target_domain,
            max_papers=args.max_papers_per_query
        )
        research_problem.target_domain.add_question_papers(question, papers)
        print(f"    -Retrieved {len(papers)} papers")

    # Step 2b: Batch analyze all questions in target domain
    print("\n2b. Analyzing target domain papers (batch inference)...")

    # Prepare batch of analysis prompts
    analysis_messages_list = []
    for question in research_problem.research_questions:
        papers = research_problem.target_domain.fetch_question_papers(question)
        
        if not papers:
            print(f"  Warning: No papers for {question.id}, skipping analysis")
            continue
        
        prompt = create_target_domain_analysis_prompt(
            research_problem=research_problem.problem_statement,
            domain_specific_question=question.domain_specific_question,
            domain_agnostic_question=question.domain_agnostic_question,
            question_rationale=question.rationale,
            papers_with_snippets=papers,
            target_domain=args.target_domain,
            fine_grained_domain=research_problem.fine_grained_domain
        )
        messages = [{"role": "user", "content": prompt}]
        analysis_messages_list.append(messages)

    # Batch inference for all analyses
    if analysis_messages_list:
        analysis_outputs = batch_llm_inference(
            llm,
            analysis_messages_list,
            target_domain_analysis_schema,
            temperature=args.temp,  # Lower temperature for analysis,
            max_tokens=4096
        )
    else:
        return

    # Process analysis results
    for i, (question, analysis_output) in enumerate(zip(research_problem.research_questions, analysis_outputs)):
        if analysis_output is None:
            print(f"  Failed to analyze {question.id}")
            continue

        paper_relevance = {p["paper_title"]: p["is_relevant"] for p in analysis_output.get("paper_relevance", [])}
        paper_titles = list(research_problem.target_domain.fetch_question_papers(question).keys())
        
        question.target_domain_analysis = analysis_output
        research_problem.target_domain.add_question_analysis(question, analysis_output)
        # delete irrelevant papers (determined from analysis output)
        for paper in paper_titles:
            if paper in paper_relevance and not paper_relevance[paper]:
                research_problem.target_domain.del_question_paper(question, paper)
        
        # Determine if addressed
        assessment = analysis_output.get("overall_assessment", "largely unaddressed").lower()
        is_addressed = "substantially" in assessment or "partial" in assessment
        question.mark_as_addressed(is_addressed)
        
        print(f"  {question.id}: {assessment} ({question.domain_specific_question})")
        remaining_challenges = question.remaining_challenges
        for challenge in remaining_challenges:
            research_problem.add_remaining_challenge(question, challenge)
            print(f"\t-> New challenge: {challenge.domain_specific_question}")
            # print(f"\t\t-> Rationale: {challenge.rationale}")

def explore_external_domains(args, llm, research_problem):
    # Get all questions needing cross-domain search
    questions_needing_cross_domain = research_problem.get_questions_needing_cross_domain()

    print(f"\nFound {len(questions_needing_cross_domain)} questions needing cross-domain search:")
    for q in questions_needing_cross_domain:
        print(f"  - {q.id}: {q.domain_agnostic_question}")

    if not questions_needing_cross_domain:
        print("\nAll questions addressed in target domain! No cross-domain search needed.")
        return
    
    # Step 3a: Generate cross-domain queries (batch)
    print("\n3a. Generating cross-domain queries (batch inference)...")
    
    cross_domain_messages_list = []
    for question in questions_needing_cross_domain:
        # Get target domain assessment if available
        target_assessment = None
        if question.parent_question and question.parent_question.target_domain_analysis:
            # This is a remaining challenge (includes )
            target_assessment = question.rationale
        elif question.target_domain_analysis:
            # This is an original question (iterate over all challenges in target domain analysis)
            target_assessment = ""
            for challenge in question.remaining_challenges:
                target_assessment += f"- {challenge.rationale}\n"
        
        prompt = create_cross_domain_query_prompt(
            problem_statement=research_problem.problem_statement,
            domain_specific_question=question.domain_specific_question,
            domain_agnostic_question=question.domain_agnostic_question,
            question_rationale=question.rationale,
            target_domain=args.target_domain,
            fine_grained_domain=research_problem.fine_grained_domain,
            target_domain_assessment=target_assessment
        )
        messages = [{"role": "user", "content": prompt}]
        cross_domain_messages_list.append(messages)
    
    # Batch inference for cross-domain queries
    cross_domain_outputs = batch_llm_inference(
        llm,
        cross_domain_messages_list,
        cross_domain_queries_schema,
        temperature=args.temp
    )

    # Process cross-domain query results
    cross_domain_analysis_prompts = []
    cross_domain_analysis_keys = []
    for question, cross_domain_output in tqdm(zip(questions_needing_cross_domain, cross_domain_outputs), total=len(questions_needing_cross_domain)):
        if cross_domain_output is None:
            print(f"  Failed to generate cross-domain queries for {question.id}")
            continue
        
        question.cross_domain_queries = cross_domain_output
        
        print(f"\n  {question.domain_agnostic_question}:")
        for domain_search in cross_domain_output.get("cross_domain_searches", []):
            domain_name = domain_search["domain"]
            queries = domain_search["queries"]
            
            # Get or create domain
            domain = research_problem.get_or_create_domain(domain_name)
            domain.add_question_queries(question, queries)
            question.add_external_domain(domain)
            
            print(f"    - {domain_name}: {len(queries)} queries")
            papers = retrieve_papers_for_question(
                question,
                domain,
                max_papers=args.max_papers_per_query
            )
            
            # Conduct cross-domain analysis on domain papers
            cross_domain_analysis_prompt = create_cross_domain_analysis_prompt(
                problem_statement=research_problem.problem_statement,
                domain_agnostic_question=question.domain_agnostic_question,
                question_challenge=question.rationale,
                source_domain=domain_name,
                papers_with_snippets=papers,
                target_domain=research_problem.target_domain,
                fine_grained_domain=research_problem.fine_grained_domain
            )
            cross_domain_analysis_messages = [{"role": "user", "content": cross_domain_analysis_prompt}]
            cross_domain_analysis_prompts.append(cross_domain_analysis_messages)
            cross_domain_analysis_keys.append((question, domain))
            
            domain.add_question_papers(question, papers)
            domain_search["retrieved_papers"] = papers
            print(f"      -Retrieved {len(papers)} papers")
    
    print("\n3b. Analyzing cross-domain papers (batch inference)...")
    cross_domain_analysis_outputs = batch_llm_inference(
        llm,
        cross_domain_analysis_prompts,
        cross_domain_analysis_schema,
        temperature=args.temp,
        max_tokens=4096
    )

    return cross_domain_analysis_keys, cross_domain_analysis_outputs


def save_results(args, research_problem, cross_domain_analysis_keys, cross_domain_analysis_outputs):
    options = []
    questions2domains = defaultdict(dict)
    questions2domains["research_problem"] = research_problem.problem_statement
    questions2domains["domain"] = research_problem.target_domain.domain_name
    questions2domains["fine_grained_domain"] = research_problem.fine_grained_domain

    for idx, ((q, domain), out) in enumerate(zip(cross_domain_analysis_keys, cross_domain_analysis_outputs)):
        if_relevant = [p["paper_title"] if p["directly_addresses_challenge"] else None for p in out["paper_relevance"]]
        num_relevant = sum([1 if p is not None else 0 for p in if_relevant])
        prop_relevant = num_relevant/len(if_relevant)
        options.append((idx, q.domain_specific_question, out["source_domain"], f": {num_relevant}/{len(if_relevant)}", prop_relevant))

        if (prop_relevant > 0.5) and (out["challenge_sufficiency_assessment"]["is_challenge_addressed"]):
            if q.domain_specific_question not in questions2domains:
                if q.parent_question is not None:
                    questions2domains[q.domain_specific_question]["parent_question"] = q.parent_question.domain_specific_question
                    target_paper_info = {p:snippets for p, snippets in research_problem.target_domain.fetch_question_papers(q.parent_question).items()}
                    questions2domains[q.domain_specific_question]["target_domain_papers"] = target_paper_info

                questions2domains[q.domain_specific_question]["rationale"] = q.rationale

            paper_info = {p.lower():snippets for p, snippets in domain.fetch_question_papers(q).items()}
            questions2domains[q.domain_specific_question][out["source_domain"]] = {'papers': {p:paper_info[p.lower()] for p in if_relevant if ((p is not None) and (p.lower() in paper_info))}, 'takeaways': out["solution_takeaways"], "remaining_challenge": out["challenge_sufficiency_assessment"]}
    ranked_options = sorted(options, key=lambda x: x[-1], reverse=True)
    for o in ranked_options:
        print(o)
    
    with open(f"output_debug/{args.problem_id}_recommendations.json", "w") as f:
        json.dump(questions2domains, fp=f, indent=2)
    


def main():
    parser = argparse.ArgumentParser(description="Retrieve & analyze interdisciplinary research.")
    parser.add_argument("--problem_file", type=str, 
                       default="data/cross-domain-inspiration-relations.json",
                       help="Path to the proposal text file.")
    parser.add_argument("--target_domain", type=str, default="Computer Science",
                       help="The user's desired target domain.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B",
                       help="LLM model name or path.")
    parser.add_argument("--output_dir", type=str, 
                       default="output_debug",
                       help="Path to output directory.")
    parser.add_argument("--max_papers_per_query", type=int, default=20,
                       help="Maximum papers to retrieve per question.")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Temperature for all LLM generation.")
    args = parser.parse_args()


    # Construct all problem statements
    if os.path.exists(args.problem_file):
        with open(args.problem_file, "r") as f:
            problems = json.load(f)
            problems = {sample["source_text"].lower().replace(" ", "_"): {"source_domain": sample["source_domain"],
                                                                          "target_domain": sample["target_domain"],
                                                                          "source_text": sample["source_text"],
                                                                          "target_text": sample["target_text"],
                                                                          "publication_year": sample["publication_year"],
                                                                          "abstract": sample["abstract"],
                                                                          "context": sample["context"]} 
                                                                          for sample in problems}
    else:
        print(f"File {args.problem_file} does not exist!")

    # Initialize vLLM model
    print("Loading model...")
    llm = LLM(model=args.model_name, tensor_parallel_size=2)
    print("Model loaded.\n")


    for problem_id, problem_info in tqdm(problems.items()):
        args.problem_id = problem_id
        args.problem_statement = problem_info["abstract"]
        print(f"Problem Statement: {args.problem_statement}\n")

        # Create output file path
        output_file_name = problem_id + f"_{args.max_papers_per_query}_results.json"
        condensed_output_file_name = problem_id + f"_{args.max_papers_per_query}_condensed.json"

        args.output_file = os.path.join(args.output_dir, output_file_name)
        args.condensed_output_file = os.path.join(args.output_dir, condensed_output_file_name)

        # Create output directory if needed
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

        print("Decomposing...")
        research_problem = decompose(args, llm, args.problem_statement)

        print("Exploring target domain...")
        explore_target_domain(args, llm, research_problem)

        print("Exploring external domains...")
        cross_domain_analysis_keys, cross_domain_analysis_outputs = explore_external_domains(args, llm, research_problem)

        print("Saving results...")
        save_results(args, research_problem, cross_domain_analysis_keys, cross_domain_analysis_outputs)

