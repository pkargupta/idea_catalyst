"""
Interdisciplinary Research Analysis Pipeline

This module retrieves and analyzes cross-domain research papers to identify
potential interdisciplinary connections and solutions.
"""

import os
import json
import argparse
from collections import defaultdict

from tqdm import tqdm
from vllm import LLM

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

# Environment configuration
os.environ["HF_HOME"] = "/shared/data3/pk36/.cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


def decompose(args, llm, problem_statement):
    """
    Decompose the research problem into specific research questions.
    
    Args:
        args: Command-line arguments
        llm: Language model instance
        problem_statement: The research problem to decompose
        
    Returns:
        ResearchProblem object containing decomposed questions
    """
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
        return None

    # Create ResearchProblem object
    research_problem = ResearchProblem.from_initial_decomposition(
        decomposition_output, 
        args.target_domain
    )

    print(f"Generated {len(research_problem.research_questions)} research questions:")
    for question in research_problem.research_questions:
        print(f"  - {question.id}:")
        print(f"\t\t- {question.domain_specific_question}")
        print(f"\t\t- {question.domain_agnostic_question}")
    
    return research_problem


def explore_target_domain(args, llm, research_problem):
    """
    Retrieve and analyze papers from the target domain.
    
    Args:
        args: Command-line arguments
        llm: Language model instance
        research_problem: ResearchProblem object to populate
    """
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
        print(f"    - Retrieved {len(papers)} papers")

    # Step 2b: Batch analyze all questions in target domain
    print("\n2b. Analyzing target domain papers (batch inference)...")
    analysis_messages_list = _prepare_target_domain_analysis_prompts(
        research_problem, 
        args.target_domain
    )

    if not analysis_messages_list:
        print("  Warning: No papers to analyze")
        return

    # Batch inference for all analyses
    analysis_outputs = batch_llm_inference(
        llm,
        analysis_messages_list,
        target_domain_analysis_schema,
        temperature=args.temp,
        max_tokens=4096
    )

    # Process analysis results
    _process_target_domain_analysis(research_problem, analysis_outputs)


def _prepare_target_domain_analysis_prompts(research_problem, target_domain):
    """
    Prepare batch of analysis prompts for target domain papers.
    
    Args:
        research_problem: ResearchProblem object
        target_domain: Target domain name
        
    Returns:
        List of message prompts for batch inference
    """
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
            target_domain=target_domain,
            fine_grained_domain=research_problem.fine_grained_domain
        )
        messages = [{"role": "user", "content": prompt}]
        analysis_messages_list.append(messages)
    
    return analysis_messages_list


def _process_target_domain_analysis(research_problem, analysis_outputs):
    """
    Process and store target domain analysis results.
    
    Args:
        research_problem: ResearchProblem object
        analysis_outputs: List of analysis outputs from LLM
    """
    for question, analysis_output in zip(research_problem.research_questions, analysis_outputs):
        if analysis_output is None:
            print(f"  Failed to analyze {question.id}")
            continue

        # Extract paper relevance and remove irrelevant papers
        paper_relevance = {
            paper["paper_title"]: paper["is_relevant"] 
            for paper in analysis_output.get("paper_relevance", [])
        }
        paper_titles = list(
            research_problem.target_domain.fetch_question_papers(question).keys()
        )
        
        question.target_domain_analysis = analysis_output
        research_problem.target_domain.add_question_analysis(question, analysis_output)
        
        # Delete irrelevant papers
        for paper_title in paper_titles:
            if paper_title in paper_relevance and not paper_relevance[paper_title]:
                research_problem.target_domain.del_question_paper(question, paper_title)
        
        # Determine if question is addressed
        assessment = analysis_output.get("overall_assessment", "largely unaddressed").lower()
        is_addressed = "substantially" in assessment or "partial" in assessment
        question.mark_as_addressed(is_addressed)
        
        print(f"  {question.id}: {assessment} ({question.domain_specific_question})")
        
        # Log remaining challenges
        for challenge in question.remaining_challenges:
            research_problem.add_remaining_challenge(question, challenge)
            print(f"\t-> New challenge: {challenge.domain_specific_question}")


def explore_external_domains(args, llm, research_problem):
    """
    Generate cross-domain queries and retrieve papers from external domains.
    
    Args:
        args: Command-line arguments
        llm: Language model instance
        research_problem: ResearchProblem object
        
    Returns:
        Tuple of (analysis_keys, analysis_outputs) for cross-domain analyses
    """
    questions_needing_cross_domain = research_problem.get_questions_needing_cross_domain()

    print(f"\nFound {len(questions_needing_cross_domain)} questions needing cross-domain search:")
    for question in questions_needing_cross_domain:
        print(f"  - {question.id}: {question.domain_agnostic_question}")

    if not questions_needing_cross_domain:
        print("\nAll questions addressed in target domain! No cross-domain search needed.")
        return [], []
    
    # Step 3a: Generate cross-domain queries (batch)
    print("\n3a. Generating cross-domain queries (batch inference)...")
    cross_domain_messages_list = _prepare_cross_domain_query_prompts(
        research_problem,
        questions_needing_cross_domain,
        args.target_domain
    )
    
    # Batch inference for cross-domain queries
    cross_domain_outputs = batch_llm_inference(
        llm,
        cross_domain_messages_list,
        cross_domain_queries_schema,
        temperature=args.temp
    )

    # Process cross-domain query results and retrieve papers
    cross_domain_analysis_prompts, cross_domain_analysis_keys = (
        _process_cross_domain_queries(
            research_problem,
            questions_needing_cross_domain,
            cross_domain_outputs,
            args.max_papers_per_query
        )
    )
    
    # Step 3b: Analyze cross-domain papers
    print("\n3b. Analyzing cross-domain papers (batch inference)...")
    cross_domain_analysis_outputs = batch_llm_inference(
        llm,
        cross_domain_analysis_prompts,
        cross_domain_analysis_schema,
        temperature=args.temp,
        max_tokens=4096
    )

    return cross_domain_analysis_keys, cross_domain_analysis_outputs


def _prepare_cross_domain_query_prompts(research_problem, questions, target_domain):
    """
    Prepare prompts for generating cross-domain queries.
    
    Args:
        research_problem: ResearchProblem object
        questions: List of questions needing cross-domain search
        target_domain: Target domain name
        
    Returns:
        List of message prompts for batch inference
    """
    cross_domain_messages_list = []
    
    for question in questions:
        # Get target domain assessment if available
        target_assessment = _get_target_assessment(question)
        
        prompt = create_cross_domain_query_prompt(
            problem_statement=research_problem.problem_statement,
            domain_specific_question=question.domain_specific_question,
            domain_agnostic_question=question.domain_agnostic_question,
            question_rationale=question.rationale,
            target_domain=target_domain,
            fine_grained_domain=research_problem.fine_grained_domain,
            target_domain_assessment=target_assessment
        )
        messages = [{"role": "user", "content": prompt}]
        cross_domain_messages_list.append(messages)
    
    return cross_domain_messages_list


def _get_target_assessment(question):
    """
    Extract target domain assessment for a question.
    
    Args:
        question: Research question object
        
    Returns:
        String containing target domain assessment
    """
    if question.parent_question and question.parent_question.target_domain_analysis:
        # This is a remaining challenge
        return question.rationale
    elif question.target_domain_analysis:
        # This is an original question
        target_assessment = ""
        for challenge in question.remaining_challenges:
            target_assessment += f"- {challenge.rationale}\n"
        return target_assessment
    
    return None


def _process_cross_domain_queries(research_problem, questions, outputs, max_papers):
    """
    Process cross-domain query outputs and retrieve papers.
    
    Args:
        research_problem: ResearchProblem object
        questions: List of questions
        outputs: Cross-domain query outputs from LLM
        max_papers: Maximum papers to retrieve per query
        
    Returns:
        Tuple of (analysis_prompts, analysis_keys)
    """
    cross_domain_analysis_prompts = []
    cross_domain_analysis_keys = []
    
    for question, cross_domain_output in tqdm(
        zip(questions, outputs), 
        total=len(questions)
    ):
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
            
            # Retrieve papers
            papers = retrieve_papers_for_question(
                question,
                domain,
                max_papers=max_papers
            )
            
            # Prepare cross-domain analysis prompt
            analysis_prompt = create_cross_domain_analysis_prompt(
                problem_statement=research_problem.problem_statement,
                domain_agnostic_question=question.domain_agnostic_question,
                question_challenge=question.rationale,
                source_domain=domain_name,
                papers_with_snippets=papers,
                target_domain=research_problem.target_domain,
                fine_grained_domain=research_problem.fine_grained_domain
            )
            analysis_messages = [{"role": "user", "content": analysis_prompt}]
            cross_domain_analysis_prompts.append(analysis_messages)
            cross_domain_analysis_keys.append((question, domain))
            
            domain.add_question_papers(question, papers)
            domain_search["retrieved_papers"] = papers
            print(f"      - Retrieved {len(papers)} papers")
    
    return cross_domain_analysis_prompts, cross_domain_analysis_keys


def save_results(problem_id, research_problem, cross_domain_analysis_keys, cross_domain_analysis_outputs):
    """
    Process and display cross-domain analysis results.
    
    Args:
        research_problem: ResearchProblem object
        cross_domain_analysis_keys: List of (question, domain) tuples
        cross_domain_analysis_outputs: List of analysis outputs
    """
    options = []
    questions_to_domains = defaultdict(dict)
    
    # Store metadata
    questions_to_domains["research_problem"] = research_problem.problem_statement
    questions_to_domains["domain"] = research_problem.target_domain.domain_name
    questions_to_domains["fine_grained_domain"] = research_problem.fine_grained_domain

    for idx, ((question, domain), output) in enumerate(
        zip(cross_domain_analysis_keys, cross_domain_analysis_outputs)
    ):
        # Calculate relevance metrics
        relevant_papers = [
            paper["paper_title"] 
            for paper in output["paper_relevance"] 
            if paper["directly_addresses_challenge"]
        ]
        num_relevant = len(relevant_papers)
        total_papers = len(output["paper_relevance"])
        prop_relevant = num_relevant / total_papers if total_papers > 0 else 0
        
        options.append((
            idx, 
            question.domain_specific_question, 
            output["source_domain"], 
            f": {num_relevant}/{total_papers}", 
            prop_relevant
        ))

        # Store results for sufficiently relevant and addressed questions
        if (prop_relevant > 0.5 and 
            output["challenge_sufficiency_assessment"]["is_challenge_addressed"]):
            
            question_key = question.domain_specific_question
            
            if question_key not in questions_to_domains:
                # Add parent question info if applicable
                if question.parent_question is not None:
                    questions_to_domains[question_key]["parent_question"] = (
                        question.parent_question.domain_specific_question
                    )
                    target_papers = research_problem.target_domain.fetch_question_papers(
                        question.parent_question
                    )
                    questions_to_domains[question_key]["target_domain_papers"] = {
                        paper: snippets 
                        for paper, snippets in target_papers.items()
                    }
                
                questions_to_domains[question_key]["rationale"] = question.rationale

            # Store cross-domain paper information
            paper_info = {
                paper.lower(): snippets 
                for paper, snippets in domain.fetch_question_papers(question).items()
            }
            
            questions_to_domains[question_key][output["source_domain"]] = {
                'papers': {
                    paper: paper_info[paper.lower()] 
                    for paper in relevant_papers 
                    if paper.lower() in paper_info
                },
                'takeaways': output["solution_takeaways"],
                "remaining_challenge": output["challenge_sufficiency_assessment"]
            }
    
    # Display ranked results
    ranked_options = sorted(options, key=lambda x: x[-1], reverse=True)
    for option in ranked_options:
        print(option)
    # Save to file
    with open(f"output_debug/{problem_id}_recommendations.json", "w") as f:
        json.dump(questions_to_domains, fp=f, indent=2)


def load_problems(problem_file):
    """
    Load research problems from file.
    
    Args:
        problem_file: Path to JSON file containing problems
        
    Returns:
        Dictionary of problems keyed by normalized source text
    """
    if not os.path.exists(problem_file):
        print(f"File {problem_file} does not exist!")
        return {}
    
    with open(problem_file, "r") as f:
        problems_list = json.load(f)
    
    problems = {
        sample["source_text"].lower().replace(" ", "_"): {
            "source_domain": sample["source_domain"],
            "target_domain": sample["target_domain"],
            "source_text": sample["source_text"],
            "target_text": sample["target_text"],
            "publication_year": sample["publication_year"],
            "abstract": sample["abstract"],
            "context": sample["context"]
        }
        for sample in problems_list
    }
    
    return problems


def process_single_problem(args, llm, problem_id, problem_info):
    """
    Process a single research problem through the full pipeline.
    
    Args:
        args: Command-line arguments
        llm: Language model instance
        problem_id: Unique identifier for the problem
        problem_info: Dictionary containing problem details
    """
    problem_statement = problem_info["abstract"]
    print(f"Problem Statement: {problem_statement}\n")

    # Create output file paths
    output_file_name = f"{problem_id}_{args.max_papers_per_query}_results.json"
    condensed_output_file_name = f"{problem_id}_{args.max_papers_per_query}_condensed.json"
    
    args.output_file = os.path.join(args.output_dir, output_file_name)
    args.condensed_output_file = os.path.join(args.output_dir, condensed_output_file_name)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Execute pipeline
    print("Decomposing...")
    research_problem = decompose(args, llm, problem_statement)
    
    if research_problem is None:
        print(f"Skipping {problem_id} due to decomposition failure")
        return

    print("Exploring target domain...")
    explore_target_domain(args, llm, research_problem)

    print("Exploring external domains...")
    cross_domain_analysis_keys, cross_domain_analysis_outputs = explore_external_domains(
        args, llm, research_problem
    )

    print("Saving results...")
    save_results(problem_id, research_problem, cross_domain_analysis_keys, cross_domain_analysis_outputs)


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Retrieve & analyze interdisciplinary research."
    )
    parser.add_argument(
        "--problem_file",
        type=str,
        default="data/cross-domain-inspiration-relations.json",
        help="Path to the proposal text file."
    )
    parser.add_argument(
        "--target_domain",
        type=str,
        default="Computer Science",
        help="The user's desired target domain."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-14B",
        help="LLM model name or path."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_debug",
        help="Path to output directory."
    )
    parser.add_argument(
        "--max_papers_per_query",
        type=int,
        default=20,
        help="Maximum papers to retrieve per question."
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Temperature for all LLM generation."
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Load problems
    problems = load_problems(args.problem_file)
    if not problems:
        return

    # Initialize vLLM model
    print("Loading model...")
    llm = LLM(model=args.model_name, tensor_parallel_size=2)
    print("Model loaded.\n")

    # Process each problem
    for problem_id, problem_info in tqdm(problems.items()):
        process_single_problem(args, llm, problem_id, problem_info)


if __name__ == "__main__":
    main()