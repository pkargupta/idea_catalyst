import traceback
from litellm import completion
from json_repair import repair_json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import List, Literal


def create_takeaway_evaluation_prompt(
    research_problem: str,
    target_domain: str,
    method_1_takeaways: List[dict],
    method_2_takeaways: List[dict],
    ground_truth_takeaways: dict
) -> str:
    """
    Creates a prompt for evaluating and comparing cross-domain takeaways from two methods
    relative to ground-truth takeaway quality.
    
    Args:
        research_problem: The research problem being addressed
        target_domain: The target domain (e.g., "Computer Science")
        method_1_takeaways: List of takeaway dicts from method 1 (e.g., main method)
        method_2_takeaways: List of takeaway dicts from method 2 (e.g., baseline)
        ground_truth_takeaways: Ground truth takeaway dict from reference paper
        
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
    ground_truth_text = format_takeaway_set([ground_truth_takeaways], "GROUND TRUTH TAKEAWAYS (REFERENCE)")
    
    prompt = f"""You are an expert evaluator assessing the quality of cross-domain research takeaways.
Your task is to compare takeaways from two different methods that attempt to address the same research problem by drawing insights from domains outside the target domain.

You will evaluate these methods relative to a **ground-truth reference takeaway** extracted from a published paper that successfully addressed the same research problem.

CRITICAL FRAMING:
The ground truth takeaway is extracted from a paper abstract and is therefore intentionally brief and underspecified.
It should be treated as a **minimal but authoritative exemplar of a high-quality cross-domain insight**, NOT as a fully elaborated solution.

You MUST NOT:
- Penalize the ground truth for brevity or lack of implementation detail
- Reward a method simply for being more verbose, detailed, or stylistically polished
- Treat length, jargon density, or elaboration as indicators of higher quality

You MUST:
- Focus on **conceptual meaningfulness**, **integration potential in principle**, and **intellectual interest**
- Judge whether a method’s takeaways are **comparable to or better than** the ground truth along these dimensions
- Assume all takeaways could be expanded further in a full paper

--------------------------------------------------
RESEARCH PROBLEM
{research_problem}

TARGET DOMAIN
{target_domain}

--------------------------------------------------
GROUND TRUTH TAKEAWAYS (REFERENCE)
{ground_truth_text}

--------------------------------------------------
METHOD 1 TAKEAWAYS
{method_1_text}

--------------------------------------------------
METHOD 2 TAKEAWAYS
{method_2_text}

--------------------------------------------------
EVALUATION GOAL

Determine which method produces takeaways that are most comparable to — or exceed — the **ground truth’s quality as a cross-domain research insight**, focusing on:

1. Whether the insight is genuinely meaningful for the research problem
2. Whether it has strong potential to integrate with core target-domain elements
3. Whether it is intellectually interesting and non-obvious, without being forced

The goal is NOT content matching. The goal is **quality alignment**.

--------------------------------------------------
GROUND TRUTH QUALITY BENCHMARKS (CONCEPTUAL)

Use the ground truth to establish a conceptual quality bar along the following dimensions:

### 1. Conceptual Rationale Quality
The ground truth demonstrates:
- A principled reason why an external-domain idea is relevant
- A non-trivial, non-generic connection to the research problem
- Conceptual legitimacy even when expressed briefly

This concerns **meaning**, not explanation length.

### 2. Integration Potential Quality
The ground truth demonstrates:
- Plausible integration into the target domain *in principle*
- Alignment with core target-domain mechanisms
- Research usefulness (i.e., the insight could inform method design or training strategy)

This concerns **viability**, not implementation detail.

### 3. Novelty–Relevance Balance
The ground truth demonstrates:
- A source domain that is meaningfully distinct from the target domain
- An intellectually interesting or surprising connection
- Novelty that is grounded rather than speculative

--------------------------------------------------
EVALUATION CRITERIA

Evaluate Method 1 and Method 2 relative to the ground truth benchmarks.

### 1. RATIONALE QUALITY ALIGNMENT
Assess whether the method’s takeaways:
- Provide principled, meaningful reasons for selecting the source-domain insight
- Identify genuinely useful external ideas rather than surface analogies
- Would remain compelling if summarized at an abstract-level, like the ground truth

IGNORE:
- Length of rationale
- Degree of elaboration
- Narrative polish

### 2. INTEGRATION POTENTIAL ALIGNMENT
Assess whether the method’s takeaways:
- Identify integration pathways that are conceptually viable
- Engage with core elements of the target domain
- Offer research leverage beyond generic inspiration

IGNORE:
- Whether full implementation details are provided
- Whether integration is more detailed than the ground truth

### 3. NOVELTY–RELEVANCE ALIGNMENT
Assess whether the method’s takeaways:
- Use source domains at an appropriate conceptual distance
- Provide intellectually interesting connections
- Balance surprise with substantive grounding

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
      "score": "matches_or_exceeds" | "partially_matches" | "falls_short",
      "reasoning": "1–2 sentences explaining alignment with ground truth quality"
    }},
    "integration_alignment": {{
      "score": "matches_or_exceeds" | "partially_matches" | "falls_short",
      "reasoning": "1–2 sentences explaining alignment with ground truth quality"
    }},
    "novelty_alignment": {{
      "score": "matches_or_exceeds" | "partially_matches" | "falls_short",
      "reasoning": "1–2 sentences explaining alignment with ground truth quality"
    }},
    "consistency_assessment": "Brief assessment of quality consistency across takeaways"
  }},
  "method_2_evaluation": {{
    "rationale_alignment": {{
      "score": "matches_or_exceeds" | "partially_matches" | "falls_short",
      "reasoning": "1–2 sentences explaining alignment with ground truth quality"
    }},
    "integration_alignment": {{
      "score": "matches_or_exceeds" | "partially_matches" | "falls_short",
      "reasoning": "1–2 sentences explaining alignment with ground truth quality"
    }},
    "novelty_alignment": {{
      "score": "matches_or_exceeds" | "partially_matches" | "falls_short",
      "reasoning": "1–2 sentences explaining alignment with ground truth quality"
    }},
    "consistency_assessment": "Brief assessment of quality consistency across takeaways"
  }},
  "comparative_analysis": {{
    "preferred_method": 1 | 2 | "tie",
    "summary": "2–3 sentences explaining which method’s takeaways are most comparable to or better than the ground truth in terms of meaningfulness, usefulness, and intellectual interest"
  }},
  "ranking": [1, 2] | [2, 1]
}}

"""
    
    return prompt

class AlignmentScore(BaseModel):
    score: Literal["matches_well", "partially_matches", "does_not_match"]
    reasoning: str

class MethodEvaluation(BaseModel):
    rationale_alignment: AlignmentScore
    integration_alignment: AlignmentScore
    novelty_balance_alignment: AlignmentScore
    consistency_assessment: str

class ComparativeAnalysis(BaseModel):
    preferred_method: Literal[1, 2]
    summary: str

class TakeawayEvaluation(BaseModel):
    method_1_evaluation: MethodEvaluation
    method_2_evaluation: MethodEvaluation
    comparative_analysis: ComparativeAnalysis
    ranking: List[int] = Field(min_items=2, max_items=2)

takeaway_evaluation_schema = TakeawayEvaluation.model_json_schema()


takeaway_eval_prompt = """You are an expert evaluator, evaluating how effective a research assistant is. The research assistant is tasked with identifying insightful takeaways from different domains that are meaningful for addressing the research problem. You will assess multiple of these takeaways based on the provided criteria, and rank them in order of quality.

# Research Problem:
{research_problem}

# Takeaway 1:
{takeaway_1}

# Takeaway 2:
{takeaway_2}

# Criteria for Evaluation:
- Best rational
- Best potential for integrating well with target domains
- Most surprising and interesting

# Output Format:
{{
   "reasoning": str, # Brief reasoning (2-3 sentences max). Explain your reasoning for evaluating the takeaway.
   "ranking": list[int], # List of integers representing the order of prefered takeaways. It should be [1, 2] if the first takeaway is preferred over the second takeaway, and [2, 1] if the second takeaway is preferred over the first takeaway.
}}"""

idea_eval_prompt = """You are an expert evaluator, evaluating how effective a research assistant is. The research assistant is proposing an idea for a research problem. You will assess multiple of these ideas based on the provided criteria, and rank them in order of quality.

# Research Problem:
{research_problem}

# Idea 1:
{idea_1}

# Idea 2:
{idea_2}

# Criteria for Ideas:
- Best rational
- Best potential for integrating well with target domains
- Most surprising and interesting

# Output Format:
{{
   "reasoning": str, # Brief reasoning (2-3 sentences max). Explain your reasoning for evaluating the takeaway.
   "ranking": list[int], # List of integers representing the order of prefered takeaways. It should be [1, 2] if the first takeaway is preferred over the second takeaway, and [2, 1] if the second takeaway is preferred over the first takeaway.
}}"""


class Evaluator:
    def __init__(
        self,
        num_retries=10,
        model_name=None,
        api_base=None,
        api_key=None
    ):
        self.num_retries = num_retries

        self.model_name = model_name
        self.kwargs = {"temperature": 0.0, "max_tokens": 1024}
        if api_base and api_key:
            self.kwargs["api_base"] = api_base
            self.kwargs["api_key"] = api_key

    def completion(self, messages):
        return completion(model=self.model_name, messages=messages, num_retries=self.num_retries, **self.kwargs).choices[0].message.content

    def evaluate_sample(self, sample):
        eval_results = {}
        research_problem,selected_takeaways,gt_takeaways,proposed_idea,gt_idea = sample,sample,sample,sample,sample

        # Part 1: Evaluate takeaways
        for _ in self.num_retries:
            processed_takeaway_eval_prompt = takeaway_eval_prompt.format(
                research_problem=research_problem,
                takeaway_1=selected_takeaways,
                takeaway_2=gt_takeaways
            )

            messages = [{"role": "user", "content": processed_takeaway_eval_prompt}]
            takeaway_eval_response = self.completion(messages)
            takeaway_eval_response = repair_json(takeaway_eval_response, return_objects=True)

            if "reasoning" in takeaway_eval_response and "ranking" in takeaway_eval_response:
                if takeaway_eval_response["ranking"] == [1, 2]:
                    eval_results["takeaways"] = {
                        "judge_response": takeaway_eval_response,
                        "selected_takeaway_win": True
                    }
                    break
                elif takeaway_eval_response["ranking"] == [2, 1]:
                    eval_results["takeaways"] = {
                        "judge_response": takeaway_eval_response,
                        "selected_takeaway_win": False
                    }
                    break

        # Part 2: Evaluate overall proposed idea
        for _ in self.num_retries:
            processed_idea_eval_prompt = idea_eval_prompt.format(
                research_problem=research_problem,
                idea_1=proposed_idea,
                idea_2=gt_idea
            )

            messages = [{"role": "user", "content": processed_idea_eval_prompt}]
            idea_eval_response = self.completion(messages)
            idea_eval_response = repair_json(idea_eval_response, return_objects=True)

            if "reasoning" in idea_eval_response and "ranking" in idea_eval_response:
                if idea_eval_response["ranking"] == [1, 2]:
                    eval_results["idea"] = {
                        "judge_response": idea_eval_response,
                        "selected_idea_win": True
                    }
                    break
                elif takeaway_eval_response["ranking"] == [2, 1]:
                    eval_results["idea"] = {
                        "judge_response": idea_eval_response,
                        "selected_idea_win": False
                    }
                    break

        return eval_results

    def evaluate_samples(self, samples):
        evaluated_samples = []
        batch_size = 300
        total_successes,total_failures = len(evaluated_samples),0

        with tqdm(total=len(samples), desc="Evaluating samples") as progress_bar:
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]

                with ThreadPoolExecutor(max_workers=min(batch_size, len(batch))) as executor:
                    futures_to_index = {executor.submit(self.evaluate_sample, sample): sample for sample in batch}

                    for future in as_completed(futures_to_index):
                        curr_result = future.result()

                        if curr_result == None:
                            total_failures += 1
                            continue                        
                        evaluated_samples.append(curr_result)

                        total_successes += 1
                        progress_bar.update(1)




        print(f"\n\n\nEvaluation complete!")
        print(f"    # succeeded conversations: {total_successes}")
        print(f"    # failed conversations: {total_failures}")

        # Print summary statistics
        takeaway_win_rate = sum([1 for sample in evaluated_samples if sample['takeaways']['selected_takeaway_win']]) / len(evaluated_samples)
        idea_win_rate = sum([1 for sample in evaluated_samples if sample['idea']['selected_idea_win']]) / len(evaluated_samples)
        print(f"    # takeaway win rate: {takeaway_win_rate}")
        print(f"    # idea win rate: {idea_win_rate}")

        return evaluated_samples