import os
# os.environ["HF_HOME"] = "/shared/data3/pk36/.cache"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import argparse
import json
from json_repair import load
from pydantic import BaseModel, Field
from typing import List, Literal
from vllm import LLM

# Assuming utils is in parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import batch_llm_inference


def create_takeaway_evaluation_prompt(
    research_problem: str,
    target_domain: str,
    method_1_takeaway: List[dict],
    method_2_takeaway: List[dict],
) -> str:
    """
    Creates a prompt for evaluating and comparing cross-domain takeaways from two methods
    relative to ground-truth takeaway quality.
    
    Args:
        research_problem: The research problem being addressed
        target_domain: The target domain (e.g., "Computer Science")
        method_1_takeaways: List of takeaway dicts from method 1 (e.g., main method)
        method_2_takeaways: List of takeaway dicts from method 2 (e.g., baseline)

    Returns:
        Formatted prompt string
    """
    
    def format_takeaway(takeaway, set_name):
        """Format a list of takeaways for display."""
        if not takeaway:
            return f"{set_name}: No takeaways provided."
        
        formatted_parts = [f"\n## {set_name}\n"]
        
        formatted_parts.append(f"**Source Domain**: {takeaway.get('source_domain', 'N/A')}")
        
        integration = takeaway.get('integration_mechanism', {})
        
        # Target domain elements
        target_elements = integration.get('target_domain_elements', [])
        if target_elements:
            formatted_parts.append(f"\n**Target Domain Elements**:")
            for elem in target_elements:
                formatted_parts.append(f"  - {elem}")
        
        # Source domain takeaways
        source_takeaways = integration.get('source_domain_takeaways', [])
        if source_takeaways:
            formatted_parts.append(f"\n**Source Domain Insights**:")
            for i, st in enumerate(source_takeaways, 1):
                formatted_parts.append(f"\n  Insight {i}:")
                formatted_parts.append(f"  - Rationale: {st.get('selection_rationale', 'N/A')}")
                source_form = st.get('source_domain_formulation', 'N/A')
                formatted_parts.append(f"  - Source Formulation: {source_form}")
                mech = st.get('mechanism_explanation', 'N/A')
                formatted_parts.append(f"  - Mechanism: {mech}")
        
        # Synthesis
        synthesis = integration.get('synthesis_approach', 'N/A')
        formatted_parts.append(f"\n**Synthesis Approach**: {synthesis}")
        
        formatted_parts.append("\n" + "-"*60)
        
        return "\n".join(formatted_parts)
    
    # Format all three sets of takeaways
    method_1_text = format_takeaway(method_1_takeaway, "METHOD 1 TAKEAWAYS")
    method_2_text = format_takeaway(method_2_takeaway, "METHOD 2 TAKEAWAYS")
    
    prompt = f"""You are an expert evaluator assessing the quality of cross-domain research takeaways.
Your task is to compare takeaways from two different methods that attempt to address the same research problem by drawing insights from domains outside the target domain.

--------------------------------------------------
RESEARCH PROBLEM
{research_problem}

TARGET DOMAIN
{target_domain}

--------------------------------------------------
METHOD 1 TAKEAWAYS
{method_1_text}

--------------------------------------------------
METHOD 2 TAKEAWAYS
{method_2_text}

--------------------------------------------------
EVALUATION CRITERIA

When evaluating Method 1 and Method 2, explicitly ground your judgment in the relevant fields of each takeaway, as described below.

### 1. INTERDISCIPLINARY INSIGHTFULNESS
Assess whether the method's takeaways provide insightful perspectives on the research problem.
  - Perspectives should introduce specific concepts/frameworks from their respective source domain
  - Insightful perspectives should be intellectually interesting, non-obvious, and thought-provoking to researchers in the target domain ({target_domain})
    - Non-obvious perspectives typically come from source domains that are meaningfully distinct from the target domain ({target_domain})

### 2. INTERDISCIPLINARY RELEVANCE
Assess whether the method's takeaways are relevant to the research problem and have strong potential for integration in the target domain ({target_domain}).
  - Ideal takeaways should:
    - Inspire new approaches/solutions to the research problem in the target domain ({target_domain})
    - Address a gap/challenge for the research problem in the target domain ({target_domain})
  - The complexity, simplicity, or practicality of the takeaway should not factor into your decision (e.g., a "clear, immediately applicable" solution does not mean more relevant). Relevance is defined based on the potential impact of the source domain being introduced to the target domain for the research problem.
  - Keep in mind that if the distance between the source and target domain is larger (e.g., Computer Science & Engineering are closer than Computer Science & Philosophy), the idea may inherently be less practical. This does not mean that it is less relevant. Focus on the degree of the potential impact to the research problem instead.

IGNORE:
- Length of explanations
- Narrative polish
- Missing implementation details

CONSIDER:
- **Consistency**: Are the method’s takeaways consistently meaningful, or uneven?
- **Groundedness**: Are claims supported by real conceptual alignment?
- **Scope appropriateness**: Are takeaways neither trivial nor wildly speculative?

--------------------------------------------------
OUTPUT FORMAT

Return a JSON object:

{{
  "takeaway_comparison": {{
    "interdisciplinary_insightfulness": {{
      "preferred_method": 1 | 2,
      "reasoning": "1–2 sentences explaining your reasoning for the preferred method"
    }},
    "interdisciplinary_relevance": {{
      "preferred_method": 1 | 2,
      "reasoning": "1–2 sentences explaining your reasoning for the preferred method based on the evaluation criteria"
    }},
  }},
  "overall_assessment": {{
    "preferred_method": 1 | 2,
    "summary": "2–3 sentences explaining which method’s takeaways are higher quality in terms of interdisciplinary insightfulness and interdisciplinary relevance"
  }}
}}
"""
    
    return prompt

class MetricComparison(BaseModel):
    preferred_method: Literal[1, 2]
    reasoning: str

class TakeawayComparison(BaseModel):
    interdisciplinary_insightfulness: MetricComparison
    interdisciplinary_relevance: MetricComparison

class OverallAssessment(BaseModel):
    preferred_method: Literal[1, 2]
    summary: str

class TakeawayEvaluation(BaseModel):
    takeaway_comparison: TakeawayComparison
    overall_assessment: OverallAssessment

takeaway_evaluation_schema = TakeawayEvaluation.model_json_schema()

def create_idea_evaluation_prompt(
    research_problem: str,
    target_domain: str,
    method_1_takeaway: list,
    method_2_takeaway: list,
    method_1_idea: dict,
    method_2_idea: dict,
) -> str:
    """
    Creates a prompt for comparing the overall cross-domain ideas from two methods
    relative to a ground-truth idea from a reference paper.
    """

    def format_takeaway(takeaway, set_name):
        """Format a list of takeaways for display."""
        if not takeaway:
            return f"{set_name}: No takeaways provided."
        
        formatted_parts = [f"\n## {set_name}\n"]
        
        formatted_parts.append(f"**Source Domain**: {takeaway.get('source_domain', 'N/A')}")
        
        integration = takeaway.get('integration_mechanism', {})
        
        # Target domain elements
        target_elements = integration.get('target_domain_elements', [])
        if target_elements:
            formatted_parts.append(f"\n**Target Domain Elements**:")
            for elem in target_elements:
                formatted_parts.append(f"  - {elem}")
        
        # Source domain takeaways
        source_takeaways = integration.get('source_domain_takeaways', [])
        if source_takeaways:
            formatted_parts.append(f"\n**Source Domain Insights**:")
            for i, st in enumerate(source_takeaways, 1):
                formatted_parts.append(f"\n  Insight {i}:")
                formatted_parts.append(f"  - Rationale: {st.get('selection_rationale', 'N/A')}")
                source_form = st.get('source_domain_formulation', 'N/A')
                formatted_parts.append(f"  - Source Formulation: {source_form}")
                mech = st.get('mechanism_explanation', 'N/A')
                formatted_parts.append(f"  - Mechanism: {mech}")
        
        # Synthesis
        synthesis = integration.get('synthesis_approach', 'N/A')
        formatted_parts.append(f"\n**Synthesis Approach**: {synthesis}")
        
        formatted_parts.append("\n" + "-"*60)
        
        return "\n".join(formatted_parts)
    
    # Format all three sets of takeaways
    method_1_text = format_takeaway(method_1_takeaway, "METHOD 1 TAKEAWAYS")
    method_2_text = format_takeaway(method_2_takeaway, "METHOD 2 TAKEAWAYS")

    prompt = f"""You are an expert evaluator assessing the quality of cross-domain RESEARCH IDEAS.
Your task is to compare two proposed ideas that integrate insights from an external domain
to address the same research problem.

--------------------------------------------------
RESEARCH PROBLEM
{research_problem}

TARGET DOMAIN
{target_domain}

--------------------------------------------------
METHOD 1 IDEA

Source Domain:
{method_1_idea.get("source_domain", "N/A")}

Proposed Approach:
{method_1_idea.get("idea", {}).get("proposed_approach", "N/A")}

Key Innovations:
{method_1_idea.get("idea", {}).get("key_innovations", [])}

Supporting Takeaways:
{method_1_text}

--------------------------------------------------
METHOD 2 IDEA

Source Domain:
{method_2_idea.get("source_domain", "N/A")}

Proposed Approach:
{method_2_idea.get("idea", {}).get("proposed_approach", "N/A")}

Key Innovations:
{method_2_idea.get("idea", {}).get("key_innovations", [])}

Supporting Takeaways:
{method_2_text}

--------------------------------------------------
EVALUATION CRITERIA

### 1.INTERDISCIPLINARY NOVELTY
Which idea is more novel?
  - The **source domain** chosen and its conceptual distance from the target domain
  - The **proposed_approach**: Is the idea non-obvious to target-domain experts?
  - The **key_innovations**: Do they reflect insights unlikely to arise within the target domain alone?
  - Whether the supporting takeaways draw on **less common or underexplored external insights**

Higher novelty means:
  - The idea is surprising but still credible
  - The cross-domain move feels inventive rather than expected

### 2.INTERDISCIPLINARY USEFULNESS
Which idea has greater interdisciplinary potential for addressing the research problem in the target domain ({target_domain})?
  - Ideas with greater interdisciplinary potential should:
    - Present new approaches/solutions to the research problem in the target domain ({target_domain})
    - Address a gap/challenge for the research problem in the target domain ({target_domain})
    - The idea integrates the concepts from both the target domain and source domain into a well-formed idea that addresses the research problem
  - The complexity, simplicity, or practicality of the proposed idea should not factor into your decision (e.g., a more "clear, immediately applicable"/"direct"/"concrete" solution does not make it more useful).Usefulness is defined based on the potential impact of the source domain being introduced to the target domain. Specifically, a more useful interdisciplinary idea integrates the source and target domains in a way that allows for a more significant problem/challenge to be solved or a significant gap in existing ideas to be addressed.
  - Keep in mind that if the distance between the source and target domain is larger (e.g., Computer Science & Engineering are closer than Computer Science & Philosophy), the idea may inherently be less practical. This does not mean that it is less useful. Focus on the degree of the potential impact instead.

--------------------------------------------------
OUTPUT FORMAT

Return a JSON object:

{{
  "idea_comparison": {{
    "interdisciplinary_novelty": {{
      "preferred_method": 1 | 2,
      "reasoning": "1–2 sentences explaining which idea is more novel"
    }},
    "interdisciplinary_usefulness": {{
      "preferred_method": 1 | 2,
      "reasoning": "1-2 sentences explaining why the preferred idea is more useful than the other idea for the research problem based on the evaluation criteria"
    }}
  }},
  "overall_assessment": {{
    "preferred_method": 1 | 2,
    "summary": "2–3 sentences summarizing which idea is overall more interdisciplinary novel, interdisciplinary useful, and integrates the two domains better"
  }}
}}
"""

    return prompt

class IdeaComparison(BaseModel):
    interdisciplinary_novelty: MetricComparison
    interdisciplinary_usefulness: MetricComparison

class IdeaEvaluation(BaseModel):
    idea_comparison: IdeaComparison
    overall_assessment: OverallAssessment

idea_evaluation_schema = IdeaEvaluation.model_json_schema()

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
        "--input_dir",
        type=str,
        default="analysis",
        help="Path to directory with all results."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-14B",
        #default="openai/gpt-oss-120b",
        help="LLM model name or path."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/comparison",
        help="Path to output directory."
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Temperature for all LLM generation."
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip processing if output file already exists."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=400,
        help="Max number of samples to run."
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Initialize vLLM model
    print("Loading model...")
    llm = LLM(model=args.model_name, tensor_parallel_size=8, max_model_len=16384, gpu_memory_utilization=0.8, max_num_seqs=400)
    print("Model loaded.\n")

    overall_takeaway_stats = {"baseline_one": {"wins": 0,
                                      "losses": 0,
                                      "ties": 0},
                    "baseline_two": {"wins": 0,
                                     "losses": 0,
                                     "ties": 0},
                    "mainmethod": {"wins": 0,
                                   "losses": 0,
                                   "ties": 0}
                    }
    
    overall_idea_stats = {"baseline_one": {"wins": 0,
                                      "losses": 0,
                                      "ties": 0},
                    "baseline_two": {"wins": 0,
                                     "losses": 0,
                                     "ties": 0},
                    "mainmethod": {"wins": 0,
                                   "losses": 0,
                                   "ties": 0}
                    }

    # Iterate through each of the baselines and run the eval
    for model_id in ["baseline_one", "baseline_two", "mainmethod"]:
        print(f"Running evaluation for method: {model_id}...")

        # Setup output dir
        takeaway_output_file_name = f"{model_id}_takeaway_eval.json"
        idea_output_file_name = f"{model_id}_idea_eval.json"
        args.takeaway_output_file = os.path.join(args.output_dir, takeaway_output_file_name)
        args.idea_output_file = os.path.join(args.output_dir, idea_output_file_name)

        # Create output directory if needed
        os.makedirs(os.path.dirname(args.takeaway_output_file), exist_ok=True)
        takeaway_eval_output_dict = {}
        idea_eval_output_dict = {}

        # Read in data file
        fname = os.path.exists(os.path.join(args.input_dir, f"final_{model_id}_results.json"))
        if not fname:
            print(f"Path to data does not exist! {fname}")
            continue
        else:
            with open(os.path.join(args.input_dir, f"final_{model_id}_results.json")) as f:
                baseline_data = load(f)

        takeaway_eval_prompts = []
        idea_eval_prompts = []
        eval_keys = []

        # Collect all samples
        for i, (sample_id, sample_info) in enumerate(baseline_data.items()):
            if i == args.max_samples:
                break

            research_problem = sample_info["research_problem"]
            target_domain = sample_info["target_domain"]
            method_takeaways = sample_info["predicted_takeaways"]
            gt_takeaways = sample_info["gt_takeaways"]
            method_ideas = sample_info["proposed_ideas"]
            gt_idea = sample_info["gt_idea"]


            # Aggregate takeaways and ideas
            for idx, (takeaway, proposed_idea) in enumerate(zip(method_takeaways, method_ideas)):

                takeaway_prompt = create_takeaway_evaluation_prompt(research_problem=research_problem,
                                                                    target_domain=target_domain,
                                                                    method_1_takeaway=takeaway,
                                                                    method_2_takeaway=gt_takeaways)
                takeaway_msg = [{"role": "user", "content": takeaway_prompt}]
                takeaway_eval_prompts.append(takeaway_msg)

                idea_prompt = create_idea_evaluation_prompt(research_problem=research_problem,
                                                            target_domain=target_domain,
                                                            method_1_takeaway=takeaway,
                                                            method_2_takeaway=gt_takeaways,
                                                            method_1_idea=proposed_idea,
                                                            method_2_idea=gt_idea)
                
                idea_msg = [{"role": "user", "content": idea_prompt}]
                idea_eval_prompts.append(idea_msg)
                eval_keys.append((sample_id, idx))

        
        # Batch inference for all takeaway evals
        takeaway_eval_outputs = batch_llm_inference(
            llm,
            takeaway_eval_prompts,
            takeaway_evaluation_schema,
            temperature=args.temp,
            max_tokens=8192
        )

        # Batch inference for all idea evals
        idea_eval_outputs = batch_llm_inference(
            llm,
            idea_eval_prompts,
            idea_evaluation_schema,
            temperature=args.temp,
            max_tokens=8192
        )

        for (sample_id, idx), t_output, i_output in zip(eval_keys, takeaway_eval_outputs, idea_eval_outputs):
            takeaway_eval_output_dict[str((sample_id, idx))] = t_output
            idea_eval_output_dict[str((sample_id, idx))] = i_output
            
            try:
                if t_output["overall_assessment"]["preferred_method"] == 1:
                    overall_takeaway_stats[model_id]["wins"] += 1
                elif t_output["overall_assessment"]["preferred_method"] == 2:
                    overall_takeaway_stats[model_id]["losses"] += 1
            except (KeyError, TypeError) as e:
                print(f"Warning: Failed to parse takeaway output for ({sample_id}, {idx}): {e}")
                print(f"  t_output: {t_output}")
            
            try:
                if i_output["overall_assessment"]["preferred_method"] == 1:
                    overall_idea_stats[model_id]["wins"] += 1
                elif i_output["overall_assessment"]["preferred_method"] == 2:
                    overall_idea_stats[model_id]["losses"] += 1
            except (KeyError, TypeError) as e:
                print(f"Warning: Failed to parse idea output for ({sample_id}, {idx}): {e}")
                print(f"  i_output: {i_output}")
        
        takeaway_eval_output_dict["final_stats"] = overall_takeaway_stats[model_id]
        idea_eval_output_dict["final_stats"] = overall_idea_stats[model_id]
        
        # Save output_dict
        with open(args.takeaway_output_file, "w") as f:
            json.dump(takeaway_eval_output_dict, f, indent=2)
        with open(args.idea_output_file, "w") as f:
            json.dump(idea_eval_output_dict, f, indent=2)

        # Print out takeaway stats for model_id
        print(f"Takeaway Evaluation Stats for {model_id}")
        stats = overall_takeaway_stats[model_id]
        win_rate = stats["wins"] / (stats["wins"] + stats["losses"] + stats["ties"]) if (stats["wins"] + stats["losses"] + stats["ties"]) > 0 else 0
        tie_rate = stats["ties"] / (stats["wins"] + stats["losses"] + stats["ties"]) if (stats["wins"] + stats["losses"] + stats["ties"]) > 0 else 0
        loss_rate = stats["losses"] / (stats["wins"] + stats["losses"] + stats["ties"]) if (stats["wins"] + stats["losses"] + stats["ties"]) > 0 else 0
        print(f"{model_id} Stats: {json.dumps(stats, indent=2)}")
        print(f"{model_id} Win Rate: {win_rate:.2%}\nTie Rate: {tie_rate:.2%}\nLoss Rate: {loss_rate:.2%}\n")

        # Print out idea stats for model_id
        print(f"Idea Evaluation Stats for {model_id}")
        stats = overall_idea_stats[model_id]
        win_rate = stats["wins"] / (stats["wins"] + stats["losses"] + stats["ties"]) if (stats["wins"] + stats["losses"] + stats["ties"]) > 0 else 0
        tie_rate = stats["ties"] / (stats["wins"] + stats["losses"] + stats["ties"]) if (stats["wins"] + stats["losses"] + stats["ties"]) > 0 else 0
        loss_rate = stats["losses"] / (stats["wins"] + stats["losses"] + stats["ties"]) if (stats["wins"] + stats["losses"] + stats["ties"]) > 0 else 0
        print(f"{model_id} Stats: {json.dumps(stats, indent=2)}")
        print(f"{model_id} Win Rate: {win_rate:.2%}\nTie Rate: {tie_rate:.2%}\nLoss Rate: {loss_rate:.2%}\n")


if __name__ == "__main__":
    main()
