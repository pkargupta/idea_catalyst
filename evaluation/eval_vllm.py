import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
    method_1_takeaways: List[dict],
    method_2_takeaways: List[dict],
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
    
    def format_takeaway_set(takeaways_list, set_name):
        """Format a list of takeaways for display."""
        if not takeaways_list:
            return f"{set_name}: No takeaways provided."
        
        formatted_parts = [f"\n## {set_name}\n"]
        for idx, takeaway in enumerate(takeaways_list, 1):
            formatted_parts.append(f"\n### Takeaway {idx}")
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
    method_1_text = format_takeaway_set(method_1_takeaways, "METHOD 1 TAKEAWAYS")
    method_2_text = format_takeaway_set(method_2_takeaways, "METHOD 2 TAKEAWAYS")
    
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
EVALUATION GOAL

Determine which method produces takeaways that are higher quality as a cross-domain research insight**, focusing on:
1. Whether the insight is genuinely meaningful for the research problem
2. Whether it has strong potential to integrate with core target-domain elements
3. Whether it is intellectually interesting and non-obvious, without being forced

--------------------------------------------------
EVALUATION CRITERIA

When evaluating Method 1 and Method 2, explicitly ground your judgment in the relevant fields of each takeaway, as described below.

### 1. RATIONALE QUALITY ALIGNMENT
Assess whether the method’s takeaways provide principled and meaningful justification
for selecting the external-domain insight.

Primarily assess using:
- **Rationale**: Why this source-domain insight was chosen
- **Source Formulation**: How the insight is framed in source-domain terms
- **Mechanism**: Why and how the insight addresses the research challenge

Evaluate whether:
- The rationale reflects a non-trivial, principled connection (not a surface analogy)
- The source-domain concept is meaningfully articulated, even if briefly
- The mechanism explanation supports transfer *in principle*

IGNORE:
- Length of explanations
- Degree of elaboration
- Narrative polish

### 2. INTEGRATION POTENTIAL ALIGNMENT
Assess whether the method’s takeaways have strong potential to integrate with
core target-domain elements.

Primarily assess using:
- **Target Domain Elements**: Which concrete target-domain components are engaged
- **Synthesis Approach**: How the elements and takeaways are combined
- **Mechanism**: Whether the integration logic is technically coherent
- **Source Formulation**: Whether the source insight aligns with target mechanisms

Evaluate whether:
- Integration is plausible and useful *in principle*
- Target-domain elements are core rather than peripheral
- The synthesis forms a coherent research direction rather than a loose pairing

IGNORE:
- Missing implementation details

### 3. NOVELTY–RELEVANCE ALIGNMENT
Assess whether the method’s takeaways are intellectually interesting and non-obvious
while remaining substantively grounded.

Primarily assess using:
- **Source Domain**: Conceptual distance from the target domain
- **Source Formulation**: Whether the insight offers a genuinely new perspective
- **Mechanism**: Whether novelty is grounded in real, feasible conceptual alignment

Evaluate whether:
- The source domain is meaningfully distinct from the target domain
- The insight would be surprising or thought-provoking to a target-domain expert
- Novelty is earned through substance, not metaphor alone

NOTE:
Greater domain distance is ONLY positive if relevance and integration remain strong.

--------------------------------------------------
QUALITY PATTERNS TO CONSIDER

- **Consistency**: Are the method’s takeaways consistently meaningful, or uneven?
- **Groundedness**: Are claims supported by real conceptual alignment?
- **Integration coherence**: Do the takeaways form a coherent integration story?
- **Scope appropriateness**: Are takeaways neither trivial nor wildly speculative?

--------------------------------------------------
OUTPUT FORMAT

Return a JSON object:

{{
  "method_1_evaluation": {{
    "rationale_alignment": {{
      "reasoning": "1–2 sentences explaining rationale alignment",
      "score": score from 1-5 (1 being the lowest quality and 5 being the highest quality)
    }},
    "integration_alignment": {{
      "reasoning": "1–2 sentences explaining integration alignment",
      "score": score from 1-5 (1 being the lowest quality and 5 being the highest quality)
    }},
    "novelty_alignment": {{
      "reasoning": "1–2 sentences explaining novelty alignment",
      "score": score from 1-5 (1 being the lowest quality and 5 being the highest quality)
    }},
    "consistency_assessment": "Brief assessment of quality consistency across takeaways"
  }},
  "method_2_evaluation": {{
    "rationale_alignment": {{
      "reasoning": "1–2 sentences explaining rationale alignment",
      "score": score from 1-5 (1 being the lowest quality and 5 being the highest quality)
    }},
    "integration_alignment": {{
      "reasoning": "1–2 sentences explaining integration alignment",
      "score": score from 1-5 (1 being the lowest quality and 5 being the highest quality)
    }},
    "novelty_alignment": {{
      "reasoning": "1–2 sentences explaining novelty alignment",
      "score": score from 1-5 (1 being the lowest quality and 5 being the highest quality)
    }},
    "consistency_assessment": "Brief assessment of quality consistency across takeaways"
  }},
  "comparative_analysis": {{
    "preferred_method": "1" | "2" | "tie",
    "summary": "2–3 sentences explaining which method’s takeaways are higher quality in terms of meaningfulness, usefulness, and intellectual interest"
  }},
}}
"""
    
    return prompt

class AlignmentScore(BaseModel):
    reasoning: str
    score: Literal[1, 2, 3, 4, 5]

class MethodEvaluation(BaseModel):
    rationale_alignment: AlignmentScore
    integration_alignment: AlignmentScore
    novelty_balance_alignment: AlignmentScore
    consistency_assessment: str

class ComparativeAnalysis(BaseModel):
    preferred_method: Literal[1, 2, "tie"]
    summary: str

class TakeawayEvaluation(BaseModel):
    method_1_evaluation: MethodEvaluation
    method_2_evaluation: MethodEvaluation
    comparative_analysis: ComparativeAnalysis

takeaway_evaluation_schema = TakeawayEvaluation.model_json_schema()

def create_idea_evaluation_prompt(
    research_problem: str,
    target_domain: str,
    method_1_takeaways: list,
    method_2_takeaways: list,
    method_1_idea: dict,
    method_2_idea: dict,
) -> str:
    """
    Creates a prompt for comparing the overall cross-domain ideas from two methods
    relative to a ground-truth idea from a reference paper.
    """

    def format_takeaway_set(takeaways_list, set_name):
        """Format a list of takeaways for display."""
        if not takeaways_list:
            return f"{set_name}: No takeaways provided."
        
        formatted_parts = [f"\n## {set_name}\n"]
        for idx, takeaway in enumerate(takeaways_list, 1):
            formatted_parts.append(f"\n### Takeaway {idx}")
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
    method_1_text = format_takeaway_set(method_1_takeaways, "METHOD 1 TAKEAWAYS")
    method_2_text = format_takeaway_set(method_2_takeaways, "METHOD 2 TAKEAWAYS")

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
EVALUATION GOAL

Determine which method proposes the **stronger overall cross-domain idea**, focusing on:
1. Which idea is more **novel**
2. Which idea is more **useful** for addressing the research problem
3. Which idea demonstrates **better integration of the two domains**

--------------------------------------------------
EVALUATION CRITERIA

### 1. NOVELTY
Which idea is more novel?

Assess using:
- The **source domain** chosen and its conceptual distance from the target domain
- The **proposed_approach**: Is the idea non-obvious to target-domain experts?
- The **key_innovations**: Do they reflect insights unlikely to arise within the target domain alone?
- Whether the supporting takeaways draw on **less common or underexplored external insights**

Higher novelty means:
- The idea is surprising but still credible
- The cross-domain move feels inventive rather than expected

### 2. USEFULNESS
Which idea has greater potential to meaningfully advance solutions to the research problem?

Assess using:
- The **key_innovations**: Do they directly address gaps or limitations in the target domain?
- The **proposed_approach**: Does it plausibly improve performance, robustness, efficiency, or understanding?
- Whether the **source domain** offers capabilities the target domain currently lacks
- How well the supporting takeaways justify the idea’s relevance

Higher usefulness means:
- The idea has clear research or practical payoff
- The integration targets real shortcomings of existing approaches

### 3. QUALITY OF INTEGRATION

Evaluate integration depth using the following lenses:

#### (a) Depth of Integration
Which idea better combines elements from both domains into a genuinely unified framework?

Strong integration:
- Components from both domains are co-designed and mutually constraining
- Removing either domain would fundamentally weaken or break the idea

Weak integration:
- One domain’s methods are simply applied to the other’s problem
- Domains are combined sequentially or superficially

#### (b) Multi-Stage Disciplinary Engagement
Which idea requires expertise from both domains across multiple research stages?

Strong engagement:
- Both domains inform problem formulation, method design, and interpretation
- Sustained cross-domain reasoning is required

Weak engagement:
- One domain dominates; the other is used for a single conceptual step

#### (c) Innovation Payoff
Which idea has a clearer, more plausible path to outcomes that would NOT emerge
from either domain alone?

Strong payoff:
- The core capability exists only because of the integration
- The cross-domain synthesis is necessary, not ornamental

--------------------------------------------------
OUTPUT FORMAT

Return a JSON object:

{{
  "idea_comparison": {{
    "novelty": {{
      "preferred_method": 1 | 2 | "tie",
      "reasoning": "1–2 sentences explaining which idea is more novel"
    }},
    "usefulness": {{
      "preferred_method": 1 | 2 | "tie",
      "reasoning": "1–2 sentences explaining which idea is more useful for the research problem"
    }},
    "integration_quality": {{
      "preferred_method": 1 | 2 | "tie",
      "reasoning": "1–2 sentences explaining which idea shows deeper and more meaningful integration"
    }}
  }},
  "overall_assessment": {{
    "preferred_method": 1 | 2 | "tie",
    "summary": "2–3 sentences summarizing which idea is overall more novel, useful, and integrates the two domains better"
  }}
}}
"""

    return prompt

class MetricComparison(BaseModel):
    preferred_method: Literal[1, 2, "tie"]
    reasoning: str

class IdeaComparison(BaseModel):
    novelty: MetricComparison
    usefulness: MetricComparison
    integration_quality: MetricComparison

class OverallAssessment(BaseModel):
    preferred_method: Literal[1, 2, "tie"]
    summary: str

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
        default="openai/gpt-oss-120b",
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
        default=280,
        help="Number of samples to evaluate."
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Initialize vLLM model
    print("Loading model...")
    llm = LLM(model=args.model_name, tensor_parallel_size=2)
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

        # Read in baseline file
        if not os.path.exists(os.path.join(args.input_dir, f"final_{model_id}_results.json")):
            print(f"Path to baseline data does not exist! {os.path.join(args.input_dir, f"final_{model_id}_results.json")}")
            continue
        else:
            with open(os.path.join(args.input_dir, f"final_{model_id}_results.json")) as f:
                baseline_data = load(f)

        takeaway_eval_prompts = []
        idea_eval_prompts = []
        eval_keys = []

        # Collect all samples
        for sample_id, sample_info in baseline_data.items():

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
                                                                    method_takeaways=takeaway,
                                                                    ground_truth_takeaways=gt_takeaways)
                takeaway_msg = [{"role": "user", "content": takeaway_prompt}]
                takeaway_eval_prompts.append(takeaway_msg)

                idea_prompt = create_idea_evaluation_prompt(research_problem=research_problem, 
                                                            target_domain=target_domain, 
                                                            method_takeaways=method_takeaways,
                                                            method_ideas=proposed_idea,
                                                            ground_truth_takeaways=gt_takeaways,
                                                            ground_truth_idea=gt_idea)
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
            takeaway_eval_output_dict[(sample_id, idx)] = t_output
            idea_eval_output_dict[(sample_id, idx)] = i_output
            
            # @SHUHAIB: Can modify this to whatever matches up with your updated schema!
            if t_output["comparative_analysis"]["preferred_method"] == "1":
                overall_takeaway_stats[model_id]["win"] += 1
            elif t_output["comparative_analysis"]["preferred_method"] == "2":
                overall_takeaway_stats[model_id]["losses"] += 1
            elif t_output["comparative_analysis"]["preferred_method"] == "tie":
                overall_takeaway_stats[model_id]["ties"] += 1
            
            # @SHUHAIB: Can modify this to whatever matches up with your updated schema!
            if i_output["comparative_analysis"]["preferred_method"] == "1":
                overall_idea_stats[model_id]["win"] += 1
            elif i_output["comparative_analysis"]["preferred_method"] == "2":
                overall_idea_stats[model_id]["losses"] += 1
            elif i_output["comparative_analysis"]["preferred_method"] == "tie":
                overall_idea_stats[model_id]["ties"] += 1
        
        takeaway_eval_output_dict["final_stats"] = overall_takeaway_stats[model_id]
        idea_eval_output_dict["final_stats"] = overall_idea_stats[model_id]
        
        # Save output_dict
        with open(args.takeaway_output_file) as f:
            json.dump(takeaway_eval_output_dict, f, indent=2)
        with open(args.idea_output_file) as f:
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
