"""
Process ground-truth abstracts to match the format of the framework's generated idea fragments.
This extracts and reformats information from the original abstracts without introducing new concepts.
"""

import os
# Environment configuration
os.environ["HF_HOME"] = "/shared/data3/pk36/.cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import json
import argparse
from tqdm import tqdm
from vllm import LLM
from pydantic import BaseModel, Field
from typing import List

# Assuming utils is in parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import batch_llm_inference, convert_domain


def create_abstract_processing_prompt(
    abstract: str,
    source_domain: str,
    target_domain: str,
    source_text: str,
    target_text: str,
    context: str
) -> str:
    """
    Creates a prompt to extract and reformat information from a ground-truth abstract.
    
    Args:
        abstract: The ground-truth abstract to process
        source_domain: The source/inspiration domain
        target_domain: The target domain where idea was applied
        source_text: Brief description of source concept
        target_text: Brief description of target application
        context: Additional context about the research problem
        
    Returns:
        Formatted prompt string
    """
    
    prompt = f"""You are an expert at analyzing and reformatting research abstracts. Your task is to extract and reformat information from a ground-truth abstract that describes an interdisciplinary research contribution. The abstract shows how concepts from a source domain were integrated with the target domain to create a novel approach.

# IMPORTANT CONSTRAINTS
- **EXTRACT ONLY**: All information must come directly from the provided abstract
- **NO NEW CONCEPTS**: Do not introduce ideas, methods, or concepts not mentioned in the abstract
- **PRESERVE MEANING**: Maintain the original intent and contributions described in the abstract
- **REFORMAT ONLY**: You are restructuring existing information, not generating new research ideas

# GROUND-TRUTH ABSTRACT
{abstract}

# METADATA
- **Source Domain**: {source_domain}
- **Target Domain**: {target_domain}
- **Source Concept**: {source_text}
- **Target Application**: {target_text}
- **Context**: {context}

# YOUR TASK

Extract and reformat the abstract's content into a structured idea fragment format. Identify:

1. **Title**: Create a brief, descriptive title (max 15 words) that captures the main contribution described in the abstract

2. **Core Insight**: Extract the 2-3 sentence summary of the key integrated insight from the abstract

3. **Integration Mechanism**:
   - **Target Domain Elements**: Identify specific concepts, methods, or approaches from {target_domain} mentioned in the abstract
   - **Source Domain Takeaways**: Identify specific concepts, methods, or principles from {source_domain} mentioned in the abstract
   - **Synthesis Approach**: Extract how the abstract describes combining these elements (1-2 sentences from the abstract)

4. **Concrete Realization**:
   - **Proposed Approach**: Extract the specific technical approach, algorithm, or framework described (3-4 sentences from the abstract)
   - **Key Innovations**: Identify novel aspects mentioned in the abstract that emerge from the integration (max 2-3)

# OUTPUT FORMAT

Return a JSON object:

{{
    "idea_fragment": {{
        "title": "Brief, descriptive title for the integrated idea (max 15 words) - extracted from abstract",
        "core_insight": "2-3 sentence summary of the key integrated insight - extracted from abstract",
        "integration_mechanism": {{
            "target_domain_elements": [
                "Specific concept/method from {target_domain} mentioned in abstract",
                "Another specific concept/method from {target_domain} mentioned in abstract"
            ],
            "source_domain_takeaways": [
                {{
                    "takeaway_id": "t1",
                    "source_domain_formulation": "Description of the {source_domain} concept/method as described in the abstract",
                    "mechanism_explanation": "How this concept works based on the abstract's description",
                    "selection_rationale": "Why this was relevant for integration (based on abstract's explanation)"
                }}
            ],
            "synthesis_approach": "1-2 sentence explanation of how these elements are combined - extracted from abstract"
        }},
        "concrete_realization": {{
            "proposed_approach": "Specific technical approach, algorithm, or framework described in the abstract (3-4 sentences)",
            "key_innovations": [
                "Novel aspect 1 mentioned in the abstract",
                "Novel aspect 2 mentioned in the abstract"
            ]
        }}
    }}
}}

# GUIDELINES

- **Quote or closely paraphrase** from the abstract - do not invent
- If the abstract doesn't explicitly mention something (e.g., specific target domain methods), make reasonable inferences based on context, but mark them clearly
- If information is missing from the abstract, you may note this with "[Inferred from context]" but do NOT fabricate details
- Maintain technical accuracy to the original abstract
- Focus on what the abstract actually describes, not what could have been done

# EXAMPLE STRUCTURE

If an abstract says: "We propose a recurrent neural network approach inspired by Kalman filtering to handle missing observations in time series data. Kalman filtering provides a probabilistic framework for state estimation under uncertainty, while RNNs excel at sequential data processing. By integrating these concepts, we develop an RNN architecture that incorporates Kalman filtering principles to effectively manage missing data points."

Good extraction:
- Source domain takeaway: "Kalman filtering provides a probabilistic framework for state estimation under uncertainty"
- Target domain element: "Recurrent neural networks for sequential data processing"
- Synthesis: "Incorporating Kalman filtering principles into RNN architecture to handle missing observations in time series data"

Bad extraction (introduces new concepts):
- "Use transformer attention mechanisms" (not in abstract)
- "Apply Bayesian deep learning" (not in abstract)

Now extract and reformat the provided abstract.
"""
    
    return prompt


class SourceDomainTakeaway(BaseModel):
    takeaway_id: str
    source_domain_formulation: str = Field(
        description="Description of source domain concept as mentioned in abstract"
    )
    mechanism_explanation: str = Field(
        description="How this concept works based on abstract's description"
    )
    selection_rationale: str = Field(
        description="Why this was relevant for integration based on abstract"
    )


class IntegrationMechanism(BaseModel):
    target_domain_elements: List[str] = Field(
        min_items=1,
        description="Specific concepts/methods from target domain mentioned in abstract"
    )
    source_domain_takeaways: List[SourceDomainTakeaway] = Field(
        min_items=1,
        description="Concepts/methods from source domain mentioned in abstract"
    )
    synthesis_approach: str = Field(
        description="How elements are combined according to the abstract"
    )


class ConcreteRealization(BaseModel):
    proposed_approach: str = Field(
        description="Specific approach described in abstract"
    )
    key_innovations: List[str] = Field(
        min_items=1,
        max_items=5,
        description="Novel aspects mentioned in abstract"
    )


class IdeaFragment(BaseModel):
    title: str
    core_insight: str
    integration_mechanism: IntegrationMechanism
    concrete_realization: ConcreteRealization


class ProcessedAbstract(BaseModel):
    idea_fragment: IdeaFragment


processed_abstract_schema = ProcessedAbstract.model_json_schema()


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
            "source_id": sample["target_id"],
            "source_domain": convert_domain(sample["target_domain"]),
            "target_id": sample["source_id"],
            "target_domain": convert_domain(sample["source_domain"]),
            "source_text": sample["target_text"],
            "target_text": sample["source_text"],
            "publication_year": sample["publication_year"],
            "abstract": sample["abstract"],
            "context": sample["context"]
        }
        for sample in problems_list
    }
    
    return problems


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Process ground-truth abstracts into structured format."
    )
    parser.add_argument(
        "--problem_file",
        type=str,
        default="data/cross-domain-inspiration-relations.json",
        help="Path to the input JSON file with abstracts."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-14B",
        help="LLM model name or path."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation/processed_abstracts.json",
        help="Path to output JSON file."
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.3,
        help="Temperature for LLM generation (lower for more faithful extraction)."
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Load problems
    print("Loading problems...")
    problems = load_problems(args.problem_file)
    if not problems:
        print("No problems loaded. Exiting.")
        return
    
    print(f"Loaded {len(problems)} abstracts to process")

    # Initialize vLLM model
    print("Loading model...")
    llm = LLM(model=args.model_name, tensor_parallel_size=2)
    print("Model loaded.\n")

    # Prepare all prompts for batch processing
    print("Preparing prompts...")
    all_prompts = []
    all_keys = []
    
    for problem_id, problem_info in tqdm(problems.items(), desc="Creating prompts"):
        prompt = create_abstract_processing_prompt(
            abstract=problem_info["abstract"],
            source_domain=problem_info["source_domain"],
            target_domain=problem_info["target_domain"],
            source_text=problem_info["source_text"],
            target_text=problem_info["target_text"],
            context=problem_info["context"]
        )
        
        all_prompts.append([{"role": "user", "content": prompt}])
        all_keys.append(problem_id)
    
    print(f"Prepared {len(all_prompts)} prompts")
    
    # Batch process all abstracts
    print("\nProcessing abstracts with LLM...")
    processed_outputs = batch_llm_inference(
        llm,
        all_prompts,
        processed_abstract_schema,
        temperature=args.temp,
        max_tokens=2048
    )
    
    # Compile results
    print("\nCompiling results...")
    processed_abstracts = {}
    successful = 0
    failed = 0
    
    for problem_id, output, problem_info in zip(all_keys, processed_outputs, problems.values()):
        if output is None:
            print(f"  Failed to process: {problem_id}")
            failed += 1
            processed_abstracts[problem_id] = {
                "status": "failed",
                "original_data": problem_info,
                "processed_abstract": None
            }
        else:
            successful += 1
            processed_abstracts[problem_id] = {
                "status": "success",
                "original_data": {
                    "source_id": problem_info["source_id"],
                    "target_id": problem_info["target_id"],
                    "source_domain": problem_info["source_domain"],
                    "target_domain": problem_info["target_domain"],
                    "source_text": problem_info["source_text"],
                    "target_text": problem_info["target_text"],
                    "publication_year": problem_info["publication_year"],
                    "original_abstract": problem_info["abstract"],
                    "context": problem_info["context"]
                },
                "processed_abstract": output["idea_fragment"]
            }
    
    # Save results
    print(f"\nSaving results to: {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, "w") as f:
        json.dump(processed_abstracts, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total abstracts: {len(problems)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(problems)*100:.1f}%")
    print("\n✓ Processing complete!")
    
    # Show a sample processed abstract
    if successful > 0:
        print("\n" + "="*80)
        print("SAMPLE PROCESSED ABSTRACT")
        print("="*80)
        
        sample_id = all_keys[0]
        if processed_abstracts[sample_id]["status"] == "success":
            sample = processed_abstracts[sample_id]
            print(f"\nProblem ID: {sample_id}")
            print(f"Source Domain: {sample['original_data']['source_domain']}")
            print(f"Target Domain: {sample['original_data']['target_domain']}")
            print(f"\nTitle: {sample['processed_abstract']['title']}")
            print(f"\nCore Insight: {sample['processed_abstract']['core_insight']}")
            print(f"\nKey Innovations:")
            for innovation in sample['processed_abstract']['concrete_realization']['key_innovations']:
                print(f"  - {innovation}")


if __name__ == "__main__":
    main()