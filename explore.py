"""
Goal Decomposition Module for Interdisciplinary Research Discovery

This module implements the first step of an iterative decomposition-retrieval-synthesis
process for helping researchers discover interdisciplinary insights, inspired by how
reinforcement learning evolved through cross-domain knowledge integration.
"""

import json
from typing import Dict, List, Any


def create_goal_decomposition_prompt(research_goal: str) -> str:
    """
    Creates a detailed prompt for decomposing a research goal into fundamental sub-questions.
    
    Args:
        research_goal: The high-level research goal to decompose
        
    Returns:
        A formatted prompt string for LLM input
    """
    
    prompt = f"""You are an expert research strategist specializing in interdisciplinary problem decomposition. Your task is to break down complex research goals into fundamental sub-questions that can be addressed through cross-domain literature search and synthesis.

# RESEARCH GOAL
{research_goal}

# YOUR TASK
Decompose this research goal into a set of fundamental sub-questions that represent the core conceptual bottlenecks preventing progress on this goal. These sub-questions should guide an interdisciplinary literature search across domains beyond the primary field.

# DECOMPOSITION PRINCIPLES

## 1. Conceptual Atomicity
Each sub-question must be:
- **Irreducible**: Cannot be meaningfully answered without addressing a distinct underlying phenomenon or mechanism
- **Focused**: Addresses ONE specific conceptual challenge, not multiple bundled issues
- **Self-contained**: Can be understood and investigated independently, though it contributes to the larger goal

## 2. Bottleneck Identification
Each sub-question must represent a genuine bottleneck by being:
- **Critical**: Addressing it is necessary (not just helpful) for making progress on the overall goal
- **Currently unresolved**: Either no satisfactory answer exists in the primary domain, or existing answers are incomplete/inadequate
- **Enabling**: Answering it would unlock or enable progress on the broader research goal

## 3. Cross-Domain Potential
Each sub-question should:
- **Transcend domain boundaries**: Be formulated in a way that could have been addressed (explicitly or implicitly) in other disciplines
- **Avoid domain-specific jargon**: Use conceptual language that allows mapping to analogous problems in other fields
- **Focus on mechanisms/principles**: Ask about "how" or "why" something works, not just "what" the current approach is

# REASONING PROCESS

Before generating sub-questions, explicitly work through:

1. **Core Challenge Identification**: What is the fundamental problem or gap this research goal aims to address?

2. **Assumption Surfacing**: What assumptions does the current approach make? Which assumptions might be limiting progress?

3. **Mechanistic Requirements**: What underlying mechanisms, processes, or principles need to be understood to achieve this goal?

4. **Historical Parallels**: Are there analogous challenges that other fields have faced? What made those challenges difficult?

5. **Capability Gaps**: What capabilities, representations, or methods are needed but currently missing or inadequate?

# OUTPUT FORMAT

Provide your response as a JSON object with the following structure:

{{
    "research_goal": "The original research goal",
    "domain": "The primary domain/field of this research goal",
    "core_challenge_summary": "A 2-3 sentence summary of the fundamental challenge",
    "decomposition_reasoning": "Your step-by-step reasoning process for how you identified the sub-questions",
    "sub_questions": [
        {{
            "id": "q1",
            "question": "The sub-question in clear, conceptual terms",
            "rationale": "Why this is a critical bottleneck and conceptually atomic",
            "cross_domain_keywords": ["keyword1", "keyword2", "keyword3"],
            "potential_source_domains": ["domain1", "domain2", "domain3"],
            "current_gaps": "What is missing or inadequate in current approaches to this sub-question"
        }},
        ...
    ],
    "interdependencies": [
        {{
            "question_ids": ["q1", "q2"],
            "relationship": "Description of how these sub-questions relate or depend on each other"
        }},
        ...
    ]
}}

# QUALITY CRITERIA

Your decomposition should result in:
- **3-7 sub-questions** (fewer if the goal is narrow, more if genuinely complex)
- Sub-questions that are **conceptually distinct** from each other
- Sub-questions that **collectively cover** the critical bottlenecks for the research goal
- Sub-questions formulated to **facilitate discovery** of relevant work in other domains

# EXAMPLES OF GOOD VS. BAD SUB-QUESTIONS

**Poor sub-question (too broad):**
"How can we improve neural network performance?"
→ Not atomic, not a specific bottleneck, too vague

**Good sub-question:**
"How can a system identify which earlier decisions contributed to a delayed outcome?"
→ Atomic, clear bottleneck (credit assignment), cross-domain potential (animal learning, economics)

**Poor sub-question (domain-specific):**
"How can we optimize the hyperparameters of transformer architectures?"
→ Too tied to specific technical approach, limited cross-domain potential

**Good sub-question:**
"How can a system effectively allocate limited computational resources when different tasks have varying information value?"
→ General principle, applicable across domains (economics, biology, cognitive science)

# IMPORTANT GUIDELINES

- **Avoid solution-space questions**: Don't ask "Should we use method X or Y?" Instead ask about the underlying challenge.
- **Focus on principles, not implementations**: Ask about mechanisms and phenomena, not specific algorithms or techniques.
- **Think beyond the obvious**: The most valuable sub-questions often reveal non-obvious bottlenecks that cross-domain insights can address.
- **Be specific enough to search**: Each sub-question should be concrete enough that relevant literature could be identified.

Now, please decompose the research goal into fundamental sub-questions following these principles.
"""
    
    return prompt


def parse_decomposition_response(llm_response: str) -> Dict[str, Any]:
    """
    Parses the LLM's JSON response into a structured format.
    
    Args:
        llm_response: The JSON string response from the LLM
        
    Returns:
        Parsed dictionary containing the decomposition
    """
    try:
        # Handle potential markdown code blocks
        if "```json" in llm_response:
            llm_response = llm_response.split("```json")[1].split("```")[0].strip()
        elif "```" in llm_response:
            llm_response = llm_response.split("```")[1].split("```")[0].strip()
        
        decomposition = json.loads(llm_response)
        
        # Validate required fields
        required_fields = ["research_goal", "sub_questions"]
        for field in required_fields:
            if field not in decomposition:
                raise ValueError(f"Missing required field: {field}")
        
        return decomposition
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from LLM: {e}")


def validate_decomposition(decomposition: Dict[str, Any]) -> List[str]:
    """
    Validates the quality of the decomposition and returns warnings/suggestions.
    
    Args:
        decomposition: The parsed decomposition dictionary
        
    Returns:
        List of validation warnings or suggestions
    """
    warnings = []
    
    sub_questions = decomposition.get("sub_questions", [])
    
    # Check number of sub-questions
    if len(sub_questions) < 3:
        warnings.append("Very few sub-questions (< 3). Consider if the goal needs more comprehensive decomposition.")
    elif len(sub_questions) > 10:
        warnings.append("Many sub-questions (> 10). Consider if some can be merged or if they're truly atomic.")
    
    # Check for required fields in each sub-question
    for i, sq in enumerate(sub_questions):
        required_sq_fields = ["question", "rationale", "cross_domain_keywords"]
        for field in required_sq_fields:
            if field not in sq:
                warnings.append(f"Sub-question {i+1} missing field: {field}")
        
        # Check if keywords are provided
        if "cross_domain_keywords" in sq and len(sq["cross_domain_keywords"]) < 2:
            warnings.append(f"Sub-question {i+1} has too few cross-domain keywords (< 2)")
    
    return warnings


# Example usage
if __name__ == "__main__":
    # Example research goal
    research_goal = """
    How can we develop AI agents that can effectively learn from sparse, delayed rewards 
    in complex environments where the relationship between actions and outcomes is not 
    immediately apparent?
    """
    
    # Generate the prompt
    prompt = create_goal_decomposition_prompt(research_goal)
    
    print("=" * 80)
    print("GENERATED PROMPT FOR LLM")
    print("=" * 80)
    print(prompt)
    print("\n" + "=" * 80)
    print("\nThis prompt should be sent to an LLM (like Claude) to get the decomposition.")
    print("\nThe LLM response should be a JSON object that can be parsed using:")
    print("parse_decomposition_response(llm_response)")
    
    # Example of what the response structure should look like
    example_response = {
        "research_goal": research_goal.strip(),
        "domain": "Machine Learning / Reinforcement Learning",
        "core_challenge_summary": "The fundamental challenge is enabling agents to learn effective behaviors when feedback is sparse, delayed, and the causal chain between actions and outcomes is long or obscured.",
        "decomposition_reasoning": "The goal involves multiple conceptual bottlenecks: temporal credit assignment, exploration under uncertainty, handling sparse feedback, and learning causal relationships. Each of these represents a distinct mechanism that other fields may have addressed.",
        "sub_questions": [
            {
                "id": "q1",
                "question": "How can a system identify which earlier actions in a sequence contributed to a delayed outcome?",
                "rationale": "This is the temporal credit assignment problem - a fundamental bottleneck in learning from delayed rewards. It's atomic because it focuses specifically on the mechanism of attribution across time.",
                "cross_domain_keywords": ["credit assignment", "delayed feedback", "temporal causality", "attribution"],
                "potential_source_domains": ["Psychology (animal learning)", "Economics (intertemporal choice)", "Neuroscience (dopamine signaling)"],
                "current_gaps": "Current methods struggle with very long delays and need dense intermediate signals"
            },
            {
                "id": "q2",
                "question": "How should a system balance exploring new possibilities versus exploiting known successful strategies when feedback is infrequent?",
                "rationale": "This is the exploration-exploitation dilemma under scarcity. It's a distinct bottleneck because it addresses resource allocation under uncertainty, separate from credit assignment.",
                "cross_domain_keywords": ["exploration-exploitation", "uncertainty", "resource allocation", "sampling"],
                "potential_source_domains": ["Ecology (foraging theory)", "Economics (multi-armed bandits)", "Cognitive Science (information seeking)"],
                "current_gaps": "Existing approaches often require many samples and struggle with truly sparse reward settings"
            }
        ],
        "interdependencies": [
            {
                "question_ids": ["q1", "q2"],
                "relationship": "Credit assignment affects exploration: if an agent cannot properly attribute outcomes to actions, it cannot learn which strategies to exploit vs. explore."
            }
        ]
    }
    
    print("\n" + "=" * 80)
    print("EXAMPLE OUTPUT STRUCTURE")
    print("=" * 80)
    print(json.dumps(example_response, indent=2))