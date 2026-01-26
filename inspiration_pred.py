"""
Interdisciplinary Research Analysis Pipeline

This module retrieves and analyzes cross-domain research papers to identify
potential interdisciplinary connections and solutions.
"""

import os
# Environment configuration
os.environ["HF_HOME"] = "/shared/data3/pk36/.cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

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
    create_target_domain_integration_prompt,
    create_interdisciplinary_comparison_prompt,
    initial_decomposition_schema,
    target_domain_analysis_schema,
    cross_domain_queries_schema,
    cross_domain_analysis_schema,
    target_domain_integration_schema,
    interdisciplinary_comparison_schema
)
from classes import ResearchProblem
from utils import batch_llm_inference, retrieve_papers_for_question, convert_domain


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
    prompt = create_initial_decomposition_prompt(problem_statement, args.fine_grained_domain)
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
        args.fine_grained_domain
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
            max_papers=args.max_papers_per_query,
            year=args.publication_year
        )
        research_problem.target_domain.add_question_papers(question, papers)
        print(f"    - Retrieved {len(papers)} papers")

    # Step 2b: Batch analyze all questions in target domain
    print("\n2b. Analyzing target domain papers (batch inference)...")
    analysis_messages_list = _prepare_target_domain_analysis_prompts(
        research_problem, 
        research_problem.target_domain
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
        remaining_challenges = analysis_output.get("remaining_challenges", [])
        for challenge_data in remaining_challenges:
            challenge = research_problem.add_remaining_challenge(question, challenge_data)
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
        research_problem.target_domain
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
            args,
            research_problem,
            questions_needing_cross_domain,
            cross_domain_outputs
        )
    )
    
    # Step 3b: Analyze cross-domain papers
    print("\n3b. Analyzing cross-domain papers (batch inference)...")
    cross_domain_analysis_outputs = batch_llm_inference(
        llm,
        cross_domain_analysis_prompts,
        cross_domain_analysis_schema,
        temperature=args.temp,
        max_tokens=8192
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
            target_domain=target_domain.domain_name,
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


def _process_cross_domain_queries(args, research_problem, questions, outputs):
    """
    Process cross-domain query outputs and retrieve papers.
    
    Args:
        args: For hyperparams (year + max_papers_per_query)
        research_problem: ResearchProblem object
        questions: List of questions
        outputs: Cross-domain query outputs from LLM
        
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
                max_papers=args.max_papers_per_query,
                year=args.publication_year
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

def integrate_cross_domain_insights(args, llm, research_problem, cross_domain_analysis_keys, cross_domain_analysis_outputs):
    """
    Generate integrated idea fragments by combining external domain takeaways
    with target domain state-of-the-art.
    
    Args:
        args: Command-line arguments
        llm: Language model instance
        research_problem: ResearchProblem object
        cross_domain_analysis_keys: List of (question, domain) tuples
        cross_domain_analysis_outputs: List of analysis outputs
        
    Returns:
        Dict mapping (question, domain) to integrated idea
    """
    print("\n4. Integrating cross-domain insights with target domain...")
    
    integration_prompts = []
    integration_keys = []
    
    # Group by question to process each question's domains together
    question_to_domains = defaultdict(list)
    for (question, domain), output in zip(cross_domain_analysis_keys, cross_domain_analysis_outputs):
        # Only process if challenge is addressed, the papers are sufficiently relevant, and there are takeaways
        # Calculate relevance metrics
        relevant_papers = [
            paper["paper_title"] 
            for paper in output["paper_relevance"] 
            if paper["directly_addresses_challenge"]
        ]
        num_relevant = len(relevant_papers)
        total_papers = len(output["paper_relevance"])
        prop_relevant = num_relevant / total_papers if total_papers > 0 else 0
        
        if (output["challenge_sufficiency_assessment"]["is_challenge_addressed"] and 
            prop_relevant >= args.min_rel_threshold and
            output.get("solution_takeaways")):
            question_to_domains[question].append((domain, output))
    
    # Generate integration prompts for each question-domain pair
    for question, domain_outputs in question_to_domains.items():
        # Get parent question's target domain papers
        if question.parent_question:
            target_papers = research_problem.target_domain.fetch_question_papers(
                question.parent_question
            )
        else:
            target_papers = research_problem.target_domain.fetch_question_papers(question)
        
        if not target_papers:
            print(f"  Warning: No target domain papers for {question.id}, skipping integration")
            continue
        
        for domain, output in domain_outputs:
            source_papers = domain.fetch_question_papers(question)
            
            # Filter to only relevant papers
            relevant_paper_titles = [
                p["paper_title"] 
                for p in output["paper_relevance"] 
                if p["directly_addresses_challenge"]
            ]
            relevant_source_papers = {
                title: source_papers[title]
                for title in relevant_paper_titles
                if title in source_papers
            }
            
            if not relevant_source_papers:
                print(f"  Warning: No relevant source papers for {question.id} from {domain.domain_name}")
                continue
            
            # Create integration prompt
            prompt = create_target_domain_integration_prompt(
                problem_statement=research_problem.problem_statement,
                domain_specific_question=question.domain_specific_question,
                domain_agnostic_question=question.domain_agnostic_question,
                question_challenge=question.rationale,
                target_domain=research_problem.target_domain.domain_name,
                fine_grained_domain=research_problem.fine_grained_domain,
                target_domain_papers=target_papers,
                source_domain=domain.domain_name,
                source_domain_papers=relevant_source_papers,
                source_domain_takeaways=output["solution_takeaways"]
            )
            
            messages = [{"role": "user", "content": prompt}]
            integration_prompts.append(messages)
            integration_keys.append((question, domain))
            
            print(f"  Prepared integration for {question.id} + {domain.domain_name}")
    
    if not integration_prompts:
        print("  No integrations to generate")
        return {}
    
    # Batch inference for integrations
    integration_outputs = batch_llm_inference(
        llm,
        integration_prompts,
        target_domain_integration_schema,
        temperature=args.temp,
        max_tokens=4096
    )
    
    # Store integrated ideas
    integrated_ideas = {}
    for (question, domain), output in zip(integration_keys, integration_outputs):
        if output is None:
            print(f"  Failed to generate integration for {question.id} + {domain.domain_name}")
            continue
        
        question.add_integrated_idea(domain.domain_name, output["idea_fragment"])
        integrated_ideas[(question, domain)] = output["idea_fragment"]
        print(f"  Generated: {output['idea_fragment'].get('title', 'Untitled')}")
    
    return integrated_ideas


def rank_interdisciplinary_potential(args, llm, research_problem, integrated_ideas):
    """
    Rank all integrated ideas by their interdisciplinary potential using pairwise comparisons.
    
    Args:
        args: Command-line arguments
        llm: Language model instance
        research_problem: ResearchProblem object
        integrated_ideas: Dict mapping (question, domain) to integrated idea
        
    Returns:
        Overall rankings dict
    """
    print("\n5. Ranking interdisciplinary potential (pairwise comparison)...")
    
    if len(integrated_ideas) < 2:
        print(f"  Insufficient ideas for ranking: {len(integrated_ideas)} idea(s), need at least 2")
        return None
    
    # Prepare all ideas for comparison
    all_ideas = []
    idea_to_key = {}  # Maps index to (question, domain) key
    
    for idx, ((question, domain), idea) in enumerate(integrated_ideas.items()):
        all_ideas.append({
            'id': idx,
            'question': question.domain_specific_question,
            'source_domain': domain.domain_name,
            'idea_fragment': idea
        })
        idea_to_key[idx] = (question, domain)
    
    print(f"  Comparing {len(all_ideas)} integrated ideas pairwise...")

    # Prepare pairwise comparison prompts
    idea_pair_keys = []
    idea_pair_prompts = []

    for idea_one_idx in range(len(all_ideas)):
        for idea_two_idx in range(idea_one_idx + 1, len(all_ideas)):
            idea_one = all_ideas[idea_one_idx]
            idea_two = all_ideas[idea_two_idx]
            idea_pair = [idea_one, idea_two]
    
            # Create comparison prompt
            prompt = create_interdisciplinary_comparison_prompt(
                problem_statement=research_problem.problem_statement,
                target_domain=research_problem.target_domain.domain_name,
                fine_grained_domain=research_problem.fine_grained_domain,
                integrated_ideas=idea_pair
            )
            idea_pair_keys.append((idea_one_idx, idea_two_idx))
            idea_pair_prompts.append([{"role": "user", "content": prompt}])
    
    # Single inference for all comparisons
    ranking_outputs = batch_llm_inference(
        llm,
        idea_pair_prompts,
        interdisciplinary_comparison_schema,
        temperature=args.temp,
        max_tokens=4096
    )
    
    if ranking_outputs == []:
        print("  Failed to generate rankings")
        return None

    # Tally wins/losses for each idea and metric
    idea_rankings = defaultdict(lambda: {
        'wins': 0,
        'losses': 0,
        'total_comparisons': 0,
        'criteria_wins': defaultdict(int)
    })
    for (idea_one_idx, idea_two_idx), comparison_output in zip(idea_pair_keys, ranking_outputs):
        comparison = comparison_output["pairwise_comparisons"]
        winner_idx = comparison["overall_winner"]
        if winner_idx == 1:
            winning_idea = idea_one_idx
            losing_idea = idea_two_idx
        else:
            winning_idea = idea_two_idx
            losing_idea = idea_one_idx
        
        # Update win/loss counts
        idea_rankings[winning_idea]['wins'] += 1
        idea_rankings[losing_idea]['losses'] += 1
        idea_rankings[winning_idea]['total_comparisons'] += 1
        idea_rankings[losing_idea]['total_comparisons'] += 1

        # Update criteria wins
        for criterion, preferred_idx in comparison["criteria_preferences"].items():
            if preferred_idx == winner_idx:
                idea_rankings[winning_idea]['criteria_wins'][criterion] += 1
            else:
                idea_rankings[losing_idea]['criteria_wins'][criterion] += 1
    
    # Compile all ideas' info (provide idea fragments for overall and criteria-specific rankings), sorted based on overall rank
    ranked_ideas = sorted(
        idea_rankings.items(),
        key=lambda x: x[1]['wins'] / x[1]['total_comparisons'] if x[1]['total_comparisons'] > 0 else 0,
        reverse=True
    )
    overall_rankings = []
    for rank, (idea_idx, ranking_info) in enumerate(ranked_ideas):
        question, domain = idea_to_key[idea_idx]
        overall_rankings.append({
            'rank': rank + 1,
            'question': question.domain_specific_question,
            'source_domain': domain.domain_name,
            'idea_fragment': all_ideas[idea_idx]['idea_fragment'],
            'ranking_info': ranking_info
        })

    return overall_rankings

def save_results(args, research_problem, cross_domain_analysis_keys, cross_domain_analysis_outputs,
                integrated_ideas, question_rankings):
    """
    Process and display cross-domain analysis results.
    
    Args:
        research_problem: ResearchProblem object
        cross_domain_analysis_keys: List of (question, domain) tuples
        cross_domain_analysis_outputs: List of analysis outputs
        integrated_ideas: Dict mapping (question, domain) to integrated idea
        question_rankings: Dict mapping question to rankings
    """
    options = []
    questions_to_domains = defaultdict(dict)
    
    # Store metadata
    questions_to_domains["research_problem"] = research_problem.problem_statement
    questions_to_domains["target_domain"] = research_problem.target_domain.domain_name
    questions_to_domains["fine_grained_domain"] = research_problem.fine_grained_domain
    questions_to_domains["source_groundtruth"] = args.ground_truth

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
        if (prop_relevant > args.min_rel_threshold and 
            output["challenge_sufficiency_assessment"]["is_challenge_addressed"]):
            
            question_key = question.domain_specific_question
            
            if question_key not in questions_to_domains:
                # Add parent question info if applicable
                if question.parent_question is not None:
                    questions_to_domains[question_key]["parent_question"] = (
                        question.parent_question.domain_specific_question
                    )
                    questions_to_domains[question_key]["parent_assessment"] = (
                        question.parent_question.target_domain_analysis.get("overall_assessment", "")
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
                'relevant_paper_prop': prop_relevant,
                'papers': {
                    paper: paper_info[paper.lower()] 
                    for paper in relevant_papers 
                    if paper.lower() in paper_info
                },
                'takeaways': output["solution_takeaways"],
                "remaining_challenge": output["challenge_sufficiency_assessment"]
            }
    
    # Add rankings to output
    questions_to_domains["idea_rankings"] = question_rankings
    
    # Display ranked results
    ranked_options = sorted(options, key=lambda x: x[-1], reverse=True)
    for option in ranked_options:
        print(option)
    # Save to file
    with open(args.output_file, "w") as f:
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
        f'{sample["source_id"]}_{sample["target_id"]}_{sample["source_text"].lower().replace(" ", "_")}': {
            "source_id": sample["source_id"],
            "source_domain": sample["source_domain"],
            "target_id": sample["target_id"],
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
    problem_statement = problem_info["context"]
    args.fine_grained_domain = convert_domain(problem_info["source_domain"])
    args.ground_truth = {"gt_domain": convert_domain(problem_info["target_domain"]),
                         "gt_domain_insight": problem_info["target_text"],
                         "gt_abstract": problem_info["abstract"]}
    args.publication_year = problem_info["publication_year"]
    
    print(f"Problem Statement: {problem_statement}\n")
    print(f"Fine-grained Domain: {args.fine_grained_domain}\n")
    print(f"Ground Truth Domain: {args.ground_truth['gt_domain']}\n")

    # Create output file paths
    output_file_name = f"{problem_id}_{args.max_papers_per_query}_predictions.json"
    
    args.output_file = os.path.join(args.output_dir, output_file_name)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # If the file already exists and skip is on:
    if args.skip_if_exists and os.path.exists(args.output_file):
        print(f"Output file {args.output_file} already exists, skipping...")
        return

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

    print("Integrating insights...")
    integrated_ideas = integrate_cross_domain_insights(
        args, llm, research_problem, 
        cross_domain_analysis_keys, cross_domain_analysis_outputs
    )

    print("Ranking interdisciplinary potential...")
    question_rankings = rank_interdisciplinary_potential(
        args, llm, research_problem, integrated_ideas
    )

    print("Saving results...")
    save_results(args, research_problem, cross_domain_analysis_keys, 
                cross_domain_analysis_outputs, integrated_ideas, question_rankings)


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
        "--model_name",
        type=str,
        default="Qwen/Qwen3-14B",
        help="LLM model name or path."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inspiration_pred_output",
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
    parser.add_argument(
        "--min_rel_threshold",
        type=float,
        default=0.5,
        help="Minimum proportion of applicable papers for the domain to be considered relevant."
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip processing if output file already exists."
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
    for problem_id, problem_info in tqdm(problems.items(), total=len(problems)):
        process_single_problem(args, llm, problem_id, problem_info)


if __name__ == "__main__":
    main()