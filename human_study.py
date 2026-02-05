import os
# Environment configuration
os.environ["HF_HOME"] = "/shared/data3/pk36/.cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import json
import argparse
from collections import defaultdict
import glob

from tqdm import tqdm
from vllm import LLM
from inspiration_pred import decompose, explore_external_domains, explore_target_domain, integrate_cross_domain_insights, rank_interdisciplinary_potential
from utils import convert_domain

from openai import AzureOpenAI
from config import OPENAI_KEY

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

    for idx, ((question, domain), output) in enumerate(
        zip(cross_domain_analysis_keys, cross_domain_analysis_outputs)
    ):
        # Calculate relevance metrics
        relevant_papers = [
            paper["paper_title"] 
            for paper in output["paper_relevance"] 
            if (paper["directly_addresses_challenge"]) or (len(questions_to_domains) == 1)
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
        if (prop_relevant > args.min_rel_threshold):
            
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
    if question_rankings is not None:
        questions_to_domains["idea_rankings"] = question_rankings
    else:
        questions_to_domains["idea_rankings"] = [{'rank': idx+1,
                                                  'question': question.domain_specific_question,
                                                  'source_domain': domain.domain_name,
                                                  'idea_fragment': fragment} for idx, ((question, domain), fragment) in enumerate(integrated_ideas.items())]

    for idea in questions_to_domains["idea_rankings"]:
        question_text = idea["question"]
        source_domain = idea["source_domain"]
        if (question_text in questions_to_domains) and (source_domain in questions_to_domains[question_text]) and ("takeaways" in questions_to_domains[question_text][source_domain]):
            selected_takeaways = [s["takeaway_id"] for s in idea["idea_fragment"]["integration_mechanism"]["selected_takeaways"]]
            takeaway_info = {takeaway["takeaway_id"]: takeaway for takeaway in questions_to_domains[question_text][source_domain]["takeaways"] if takeaway["takeaway_id"] in selected_takeaways}
            # Only save the selected takeaways in idea_fragment
            for selected_takeaway in idea["idea_fragment"]["integration_mechanism"]["selected_takeaways"]:
                takeaway_id = selected_takeaway["takeaway_id"]
                selected_takeaway["source_domain_formulation"] = takeaway_info[takeaway_id]["source_domain_formulation"]
                selected_takeaway["mechanism_explanation"] = takeaway_info[takeaway_id]["mechanism_explanation"]
    
    # Display ranked results
    ranked_options = sorted(options, key=lambda x: x[-1], reverse=True)
    for option in ranked_options:
        print(option)
    # Save to file
    with open(args.output_file, "w") as f:
        json.dump(questions_to_domains, fp=f, indent=2)


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
    args.fine_grained_domain = convert_domain(problem_info["target_domain"])
    args.publication_year = problem_info["publication_year"]
    
    print(f"Problem Statement: {problem_statement}\n")
    print(f"Fine-grained Domain: {args.fine_grained_domain}\n")

    # Create output file paths
    output_file_name = f"{problem_id[:30]}_{args.max_papers_per_query}_gpt_simple_predictions.json"
    
    args.output_file = os.path.join(args.output_dir, output_file_name)
    
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
        "--model_name",
        type=str,
        default="gpt-5-mini",
        help="LLM model name or path."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="human_study_output",
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

    return parser.parse_args()

def main():
    args = parse_arguments()

    if 'Qwen' in args.model_name:
        # Initialize vLLM model
        print("Loading model...")
        llm = LLM(model=args.model_name, tensor_parallel_size=2)
        print("Model loaded.\n")
    else:
        llm = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint="https://pk36-mfft96mz-eastus2.cognitiveservices.azure.com/",
            api_key=OPENAI_KEY
        )

    problems = {
        "ishika_lsk":{
            "target_id": "ishika_lsk",
            "target_domain": "Natural Language Processing",
            "publication_year": 2027,
            "context": "Language models have been shown to change their answer to a query when the query changes languages (even for high-resource languages, where the same information is readily available in multiple languages). This could indicate either a a mismatch in the knowledge that each language contains within a language model, or a mismatch between the semantics of languages."
        }
        # "beyza_persuasion": {
        #             "target_id": "beyza_persuasion",
        #             "target_domain": "Natural Language Processing",
        #             "publication_year": 2027,
        #             "context": "Persuasion has emerged as a powerful capability in interactions with LLMs. At the same time, LLMs themselves are susceptible to persuasion, allowing them not only to adapt to new information or correct prior outputs, but also to accept harmful, misleading, or adversarial influences."
        #             },
        # "ishika_influence": {
        #             "target_id": "ishika_influence",
        #             "target_domain": "Natural Language Processing",
        #             "publication_year": 2027,
        #             "context": "Influence functions are used to determine the importance of data samples with respect to a model. While influence values have been effective in areas like data selection, they are ultimately static -- the same data point cannot have the same influence for a different model."
        #             },
        # "shuhaib_user_simulation": {
        #     "target_id": "shuhaib_user_simulation",
        #     "target_domain": "Natural Language Processing",
        #     "publication_year": 2027,
        #     "context": "LLMs can be used to simulate user behaviour. However there are many issues with current LLM-based user simulators: (1) they cannot consistently adhere to the user profile throughout multi-turn interactions, (2) there are no good ways to evaluate how realistic the user simulators are, (3) the user simulators need to reflect diverse user simulator behavior to reflect real-world users."
        #     }
        # "priyanka_education": {
        #     "target_id": "priyanka_education",
        #     "target_domain": "Natural Language Processing",
        #     "publication_year": 2027,
        #     "context": "Large language models are increasingly being used by students, which is significantly reducing their ability to develop abstract problem solving schemas and thereby critically reason. This will significantly harm their ability to tackle both well and ill-structured, real-world problems. How can we develop a personalized learning framework which truly engages & excites students in learning and developing their critical reasoning/problem solving abilities, without gamification."
        #     }
        # "mihir_imc": {
        #     "target_id": "mihir_imc",
        #     "target_domain": "In-Memory Computing",
        #     "publication_year": 2027,
        #     "context": "How do we boost the accuracy of in-memory computing while preserving high energy efficiency, specifically for Edge-AI."
        #     }
        }
    
    for problem_id, problem_info in problems.items():
        process_single_problem(args, llm, problem_id, problem_info)

if __name__ == "__main__":
    main()