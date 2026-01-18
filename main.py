import os
os.environ["HF_HOME"] = "/shared/data3/pk36/.cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import argparse
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import json_repair
import re
import json
from typing import List, Dict
import time

from search import search_semantic_scholar, collect_snippets
from prompts import (
    create_initial_decomposition_prompt,
    create_target_domain_analysis_prompt,
    create_cross_domain_query_prompt,
    initial_decomposition_schema,
    target_domain_analysis_schema,
    cross_domain_queries_schema
)
from classes import ResearchProblem, Question, Domain


def batch_llm_inference(llm, messages_list: List[List[Dict]], schema: dict, temperature: float = 0.7) -> List[dict]:
    """
    Perform batch inference with structured output.
    
    Args:
        llm: vLLM model
        messages_list: List of message sequences (each is a list of message dicts)
        schema: JSON schema for structured output
        temperature: Sampling temperature
        
    Returns:
        List of parsed JSON responses
    """
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=temperature,
        top_p=0.95,
        structured_outputs=StructuredOutputsParams(json=schema),
    )
    
    responses = llm.chat(messages_list, sampling_params, chat_template_kwargs={"enable_thinking": False})
    
    # Parse all responses
    parsed_responses = []
    for response in responses:
        try:
            parsed = json_repair.loads(response.outputs[0].text)
            parsed_responses.append(parsed)
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response text: {response.outputs[0].text}")
            parsed_responses.append(None)
    
    return parsed_responses


def retrieve_papers_for_question(question: Question, domain: Domain, max_papers: int = 10) -> Dict[str, List[str]]:
    """
    Retrieve papers and snippets for a question in a specific domain.
    
    Args:
        question: The research question
        domain: The domain to search in
        max_papers: Maximum number of unique papers to retrieve
        
    Returns:
        Dictionary mapping paper titles to lists of snippets
    """
    queries = domain.fetch_question_queries(question)
    papers = {}
    
    for query in queries:
        if len(papers) >= max_papers:
            break
        
        try:
            response = search_semantic_scholar(query, domain.domain_name)
            snippets = collect_snippets(response)
            
            if len(snippets) > 0:
                # Add new papers up to the limit
                for paper_title, snippet_list in snippets.items():
                    if paper_title not in papers:
                        papers[paper_title] = snippet_list
                    else:
                        papers[paper_title].extend(snippet_list)
                    
                    if len(papers) >= max_papers:
                        break
        except Exception as e:
            print(f"Error searching for query '{query}': {e}")
    
    return papers


def main():
    parser = argparse.ArgumentParser(description="Retrieve & analyze interdisciplinary research.")
    parser.add_argument("--problem_file", type=str, 
                       default="data/tree_of_debate.txt",
                       help="Path to the proposal text file.")
    parser.add_argument("--target_domain", type=str, default="Computer Science",
                       help="The user's desired target domain.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                       help="LLM model name or path.")
    parser.add_argument("--output_dir", type=str, 
                       default="output",
                       help="Path to output directory.")
    parser.add_argument("--max_papers_per_query", type=int, default=20,
                       help="Maximum papers to retrieve per question.")
    args = parser.parse_args()

    # Read problem statement
    if os.path.exists(args.problem_file):
        with open(args.problem_file, "r") as f:
            problem_file_text = f.read()
            match = re.search(r"Problem Statement:\s*(.*)", problem_file_text)
            if match:
                problem_statement = match.group(1).strip()
            else:
                print("Could not find problem statement in file!")
                return
        print(f"Problem Statement: {problem_statement}\n")
    else:
        print(f"File {args.problem_file} does not exist!")
        return
    
    # Create output file path
    output_file_name = os.path.splitext(os.path.basename(args.problem_file))[0] + "_results.json"
    args.output_file = os.path.join(args.output_dir, output_file_name)

    # Initialize vLLM model
    print("Loading model...")
    llm = LLM(model=args.model_name, tensor_parallel_size=2)
    print("Model loaded.\n")

    start_time = time.perf_counter()

    # =========================================================================
    # STEP 1: Initial Decomposition
    # =========================================================================
    print("=" * 80)
    print("STEP 1: Initial Decomposition")
    print("=" * 80)
    
    prompt = create_initial_decomposition_prompt(problem_statement, args.target_domain)
    messages = [{"role": "user", "content": prompt}]
    
    decomposition_outputs = batch_llm_inference(
        llm, 
        [messages], 
        initial_decomposition_schema,
        temperature=0.7
    )
    decomposition_output = decomposition_outputs[0]
    
    if decomposition_output is None:
        print("Failed to get decomposition output!")
        return
    
    # Create ResearchProblem object
    research_problem = ResearchProblem.from_initial_decomposition(
        decomposition_output, 
        args.target_domain
    )
    
    print(f"Generated {len(research_problem.research_questions)} research questions:")
    for q in research_problem.research_questions:
        print(f"  - {q.id}: {q.question}")
    print()

    # =========================================================================
    # STEP 2: Target Domain Analysis
    # =========================================================================
    print("=" * 80)
    print("STEP 2: Target Domain Analysis")
    print("=" * 80)
    
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
        print(f"    Retrieved {len(papers)} papers")
    
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
            question=question.question,
            question_rationale=question.rationale,
            papers_with_snippets=papers,
            target_domain=args.target_domain
        )
        messages = [{"role": "user", "content": prompt}]
        analysis_messages_list.append(messages)
    
    # Batch inference for all analyses
    if analysis_messages_list:
        analysis_outputs = batch_llm_inference(
            llm,
            analysis_messages_list,
            target_domain_analysis_schema,
            temperature=0.5  # Lower temperature for analysis
        )
        
        # Process analysis results
        for i, (question, analysis_output) in enumerate(zip(research_problem.research_questions, analysis_outputs)):
            if analysis_output is None:
                print(f"  Failed to analyze {question.id}")
                continue
            
            question.target_domain_analysis = analysis_output
            research_problem.target_domain.add_question_analysis(question, analysis_output)
            
            # Determine if addressed
            assessment = analysis_output.get("overall_assessment", "largely unaddressed").lower()
            is_addressed = "substantially" in assessment or "partial" in assessment
            question.mark_as_addressed(is_addressed)
            
            print(f"  {question.id}: {assessment}")
            
            # Create sub-questions for remaining challenges
            remaining_challenges = analysis_output.get("remaining_challenges", [])
            for challenge_data in remaining_challenges:
                challenge = research_problem.add_remaining_challenge(question, challenge_data)
                print(f"    -> New challenge: {challenge.id}")
    
    print()

    # =========================================================================
    # STEP 3: Cross-Domain Query Generation
    # =========================================================================
    print("=" * 80)
    print("STEP 3: Cross-Domain Query Generation")
    print("=" * 80)
    
    # Get all questions needing cross-domain search
    questions_needing_cross_domain = research_problem.get_questions_needing_cross_domain()
    
    print(f"\nFound {len(questions_needing_cross_domain)} questions needing cross-domain search:")
    for q in questions_needing_cross_domain:
        print(f"  - {q.id}: {q.question}")
    
    if not questions_needing_cross_domain:
        print("\nAll questions addressed in target domain! No cross-domain search needed.")
    else:
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
                question=question.question,
                question_rationale=question.rationale,
                target_domain=args.target_domain,
                target_domain_assessment=target_assessment
            )
            messages = [{"role": "user", "content": prompt}]
            cross_domain_messages_list.append(messages)
        
        # Batch inference for cross-domain queries
        cross_domain_outputs = batch_llm_inference(
            llm,
            cross_domain_messages_list,
            cross_domain_queries_schema,
            temperature=0.7
        )
        
        # Process cross-domain query results
        for question, cross_domain_output in zip(questions_needing_cross_domain, cross_domain_outputs):
            if cross_domain_output is None:
                print(f"  Failed to generate cross-domain queries for {question.id}")
                continue
            
            question.cross_domain_queries = cross_domain_output
            
            print(f"\n  {question.id}:")
            for domain_search in cross_domain_output.get("cross_domain_searches", []):
                domain_name = domain_search["domain"]
                queries = domain_search["queries"]
                
                # Get or create domain
                domain = research_problem.get_or_create_domain(domain_name)
                domain.add_question_queries(question, queries)
                question.add_external_domain(domain)
                
                print(f"    - {domain_name}: {len(queries)} queries")
        
        # Step 3b: Retrieve papers from external domains
        print("\n3b. Retrieving papers from external domains...")
        
        for question in questions_needing_cross_domain:
            if not question.external_domains:
                continue
            
            print(f"\n  {question.id}:")
            for domain_name, domain in question.external_domains.items():
                print(f"    Retrieving from {domain_name}...")
                papers = retrieve_papers_for_question(
                    question,
                    domain,
                    max_papers=args.max_papers_per_query
                )
                domain.add_question_papers(question, papers)
                print(f"      Retrieved {len(papers)} papers")

    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Prepare output structure
    output = {
        "problem_statement": research_problem.problem_statement,
        "target_domain": research_problem.target_domain.domain_name,
        "fine_grained_domain": research_problem.fine_grained_domain,
        "core_challenge": research_problem.core_challenge,
        "research_questions": []
    }
    
    for question in research_problem.research_questions:

        q_data = {
            "id": question.id,
            "question": question.question,
            "rationale": question.rationale,
            "target_domain_analysis": {"paper_relevance": question.target_domain_analysis["paper_relevance"],
                                       "addressed_aspects": question.target_domain_analysis["addressed_aspects"],
                                       "overall_assessment": question.target_domain_analysis["overall_assessment"]},
            "is_addressed_in_target": question.is_addressed_in_target,
            "remaining_challenges": [
                {
                    "id": c.id,
                    "question": c.question,
                    "rationale": c.rationale,
                    "cross_domain_queries": c.cross_domain_queries,
                    "external_domains_searched": list(c.external_domains.keys()),
                    "external_papers": {domain_name: domain.fetch_question_papers(question) 
                               for domain_name, domain in question.external_domains.items()} 
                               if c.external_domains else {}
                }
                for c in question.remaining_challenges
            ]
        }

        if not question.is_addressed_in_target:
            q_data["cross_domain_queries"] = question.cross_domain_queries if not question.is_addressed_in_target else None
            q_data["external_domains_searched"] = list(question.external_domains.keys()) if question.external_domains else []
            q_data["external_papers"] = {domain_name: domain.fetch_question_papers(question) 
                                         for domain_name, domain in question.external_domains.items()} if question.external_domains else {}

        output["research_questions"].append(q_data)
    
    # Save to file
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
    print(f"\nResults saved to: {args.output_file}")
    print("\nSummary:")
    print(f"  Total research questions: {len(research_problem.research_questions)}")
    print(f"  Questions addressed in target domain: {sum(q.is_addressed_in_target for q in research_problem.research_questions)}")
    print(f"  Remaining challenges identified: {sum(len(q.remaining_challenges) for q in research_problem.research_questions)}")
    print(f"  Questions needing cross-domain search: {len(questions_needing_cross_domain)}")
    print(f"  External domains explored: {len([d for d in research_problem.domains.keys() if d != args.target_domain])}")


if __name__ == "__main__":
    main()