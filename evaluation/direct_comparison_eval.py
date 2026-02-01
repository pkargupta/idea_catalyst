import os
# os.environ["HF_HOME"] = "/shared/data3/pk36/.cache"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import argparse
import json
from json_repair import load
from pydantic import BaseModel, Field
from typing import List, Literal
from itertools import combinations
from collections import defaultdict
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
      "reasoning": "1–2 sentences explaining your reasoning for the preferred method"
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
  - The complexity, simplicity, or practicality of the proposed idea should not factor into your decision. Usefulness is defined based on the potential impact of the source domain being introduced to the target domain. Specifically, a more useful interdisciplinary idea integrates the source and target domains in a way that allows for a more significant problem/challenge to be solved or a significant gap in existing ideas to be addressed.
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
      "reasoning": "1–2 sentences explaining which idea is more useful for the research problem"
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
    llm = LLM(model=args.model_name, tensor_parallel_size=8, max_model_len=16384, gpu_memory_utilization=0.8)
    print("Model loaded.\n")

    model_ids = ["baseline_one", "baseline_two", "mainmethod"]

    pairwise_stats_template = {
        "wins": 0,   # method_1 wins
        "losses": 0, # method_2 wins
    }

    overall_takeaway_stats = defaultdict(lambda: pairwise_stats_template.copy())
    overall_idea_stats = defaultdict(lambda: pairwise_stats_template.copy())

    all_model_data = {}

    for model_id in model_ids:
        path = os.path.join(args.input_dir, f"final_{model_id}_results.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing data for {model_id}: {path}")
        with open(path) as f:
            all_model_data[model_id] = load(f)
    

    for model_a, model_b in combinations(model_ids, 2):
        pair_id = f"{model_a}_vs_{model_b}"
        print(f"Running pairwise evaluation: {pair_id}")

        takeaway_output_file = os.path.join(
            args.output_dir, f"{pair_id}_takeaway_eval.json"
        )
        idea_output_file = os.path.join(
            args.output_dir, f"{pair_id}_idea_eval.json"
        )

        os.makedirs(args.output_dir, exist_ok=True)

        takeaway_eval_output_dict = {}
        idea_eval_output_dict = {}

        takeaway_eval_prompts = []
        idea_eval_prompts = []
        eval_keys = []

        data_a = all_model_data[model_a]
        data_b = all_model_data[model_b]

        for i, sample_id in enumerate(data_a.keys()):
            if i == args.max_samples:
                break

            sample_a = data_a[sample_id]
            sample_b = data_b[sample_id]

            research_problem = sample_a["research_problem"]
            target_domain = sample_a["target_domain"]

            takeaways_a = sample_a["predicted_takeaways"]
            takeaways_b = sample_b["predicted_takeaways"]

            ideas_a = sample_a["proposed_ideas"]
            ideas_b = sample_b["proposed_ideas"]

            for idx, (ta, tb, ia, ib) in enumerate(
                zip(takeaways_a, takeaways_b, ideas_a, ideas_b)
            ):
                takeaway_prompt = create_takeaway_evaluation_prompt(
                    research_problem=research_problem,
                    target_domain=target_domain,
                    method_1_takeaway=ta,
                    method_2_takeaway=tb,
                )

                idea_prompt = create_idea_evaluation_prompt(
                    research_problem=research_problem,
                    target_domain=target_domain,
                    method_1_takeaway=ta,
                    method_2_takeaway=tb,
                    method_1_idea=ia,
                    method_2_idea=ib,
                )

                takeaway_eval_prompts.append(
                    [{"role": "user", "content": takeaway_prompt}]
                )
                idea_eval_prompts.append(
                    [{"role": "user", "content": idea_prompt}]
                )

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
                pref = t_output["overall_assessment"]["preferred_method"]
                if pref == 1:
                    overall_takeaway_stats[pair_id]["wins"] += 1
                elif pref == 2:
                    overall_takeaway_stats[pair_id]["losses"] += 1
            except (KeyError, TypeError):
                pass

            try:
                pref = i_output["overall_assessment"]["preferred_method"]
                if pref == 1:
                    overall_idea_stats[pair_id]["wins"] += 1
                elif pref == 2:
                    overall_idea_stats[pair_id]["losses"] += 1
            except (KeyError, TypeError):
                pass

        
        takeaway_eval_output_dict["final_stats"] = overall_takeaway_stats[pair_id]
        idea_eval_output_dict["final_stats"] = overall_idea_stats[pair_id]

        with open(takeaway_output_file, "w") as f:
            json.dump(takeaway_eval_output_dict, f, indent=2)

        with open(idea_output_file, "w") as f:
            json.dump(idea_eval_output_dict, f, indent=2)

        def print_stats(title, stats):
            total = stats["wins"] + stats["losses"]
            win_rate = stats["wins"] / total if total else 0
            loss_rate = stats["losses"] / total if total else 0

            print(title)
            print(json.dumps(stats, indent=2))
            print(
                f"Win Rate: {win_rate:.2%}\n"
                f"Loss Rate: {loss_rate:.2%}\n"
            )

        print_stats(
            f"Takeaway Evaluation Stats ({pair_id}, {model_a}=method_1)",
            overall_takeaway_stats[pair_id],
        )
        print_stats(
            f"Idea Evaluation Stats ({pair_id}, {model_a}=method_1)",
            overall_idea_stats[pair_id],
        )



if __name__ == "__main__":
    main()
