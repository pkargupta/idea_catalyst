
from pydantic import BaseModel
from typing import List

def create_goal_decomposition_prompt(problem_statement: str, target_domain: str) -> str:
    """
    Creates a detailed prompt for decomposing a research goal into fundamental sub-questions.
    
    Args:
        research_goal: The high-level research goal to decompose
        
    Returns:
        A formatted prompt string for LLM input
    """
    
    prompt = f"""You are an expert research strategist specializing in interdisciplinary problem decomposition. Your task is to break down complex research goals into fundamental sub-questions that can be addressed through cross-domain literature search and synthesis.

# RESEARCH GOAL/PROBLEM STATEMENT
{problem_statement}

# RESEARCH DOMAIN
The problem statement primarily falls within the domain of {target_domain}.

# YOUR TASK
Decompose this research goal of addressing this problem statement into a set of fundamental sub-questions that represent the core conceptual bottlenecks preventing progress on this goal. These sub-questions should guide an interdisciplinary literature search across domains beyond the primary field.

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

# SEMANTIC SCHOLAR DOMAIN SPECIFICATION

For literature search, you must specify domains from the following list. These are the ONLY valid domains:
- Computer Science
- Medicine
- Chemistry
- Biology
- Materials Science
- Physics
- Geology
- Psychology
- Art
- History
- Geography
- Sociology
- Business
- Political Science
- Economics
- Philosophy
- Mathematics
- Engineering
- Environmental Science
- Agricultural and Food Sciences
- Education
- Law
- Linguistics

For each sub-question, you will provide:
1. **same_domain_keywords**: Keywords/concepts for searching within the target domain (if relevant work exists there)
2. **search_queries**: Structured search queries for Semantic Scholar API, organized by domain

Each search query must be:
- **Concise**: Maximum 5 words
- **Specific**: Focused on the core concept, not generic terms
- **Searchable**: Uses terminology likely to appear in paper titles/abstracts

# OUTPUT FORMAT

Provide your response as a JSON object with the following structure:

{{
    "research_goal": "The original research goal",
    "fine_grained_domain": "The specific sub-domain/sub-field of this research goal within the broader target domain",
    "core_challenge_summary": "A 2-3 sentence summary of the fundamental challenge",
    "decomposition_reasoning": "Your step-by-step reasoning process for how you identified the sub-questions",
    "sub_questions": [
        {{
            "id": "q1",
            "question": "The sub-question in clear, conceptual terms",
            "rationale": "Why this is a critical bottleneck and conceptually atomic",
            "cross_domain_search_queries": [
                {{
                    "domain": "A valid Semantic Scholar domain that could have relevant work",
                    "queries": [
                        "first 3-4 word long search query",
                        "second search query"
                    ]
                }},
                {{
                    "domain": "example cross-domain: Biology",
                    "queries": [
                        "example query: dopamine prediction error timing"
                    ]
                }}
            ],
            "same_domain_search_queries": [
                "reinforcement learning credit assignment",
                "temporal difference learning",
                "policy gradient methods"
            ],
            "current_gaps": "What is missing or inadequate in current approaches to this sub-question?"
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
- **2-4 search queries per domain** for each sub-question (not too many, not too few)
- **2-5 domains per sub-question** (focus on the most relevant ones)

# EXAMPLES OF GOOD VS. BAD SUB-QUESTIONS AND SEARCH QUERIES

**Poor sub-question (too broad):**
"How can we improve neural network performance?"
→ Not atomic, not a specific bottleneck, too vague

**Good sub-question:**
"How can a system identify which earlier decisions contributed to a delayed outcome?"
→ Atomic, clear bottleneck (credit assignment), cross-domain potential (animal learning, economics)

**Poor search queries:**
- "machine learning" (too generic, >5 words unlikely to be useful)
- "how to assign credit to actions over time in reinforcement learning" (too long, >5 words)

**Good search queries:**
- "temporal credit assignment" (concise, specific, 3 words)
- "delayed reward attribution" (concise, specific, 3 words)
- "dopamine prediction error timing" (specific to neuroscience context, 4 words)

**Poor sub-question (domain-specific):**
"How can we optimize the hyperparameters of transformer architectures?"
→ Too tied to specific technical approach, limited cross-domain potential

**Good sub-question:**
"How can a system effectively allocate limited computational resources when different tasks have varying information value?"
→ General principle, applicable across domains (economics, biology, cognitive science)

# SEARCH QUERY GUIDELINES

For each sub-question, create queries that:
1. **Use domain-appropriate terminology**: "foraging theory" for Biology/Ecology, "intertemporal choice" for Economics
2. **Are maximally informative**: Balance generality (to find relevant work) with specificity (to avoid noise)
3. **Cover different aspects**: If a sub-question has multiple facets, create queries for each
4. **Include same-domain keywords**: If the target domain has relevant work, provide 2-3 keywords for searching within it

# IMPORTANT GUIDELINES

- **Avoid solution-space questions**: Don't ask "Should we use method X or Y?" Instead ask about the underlying challenge.
- **Focus on principles, not implementations**: Ask about mechanisms and phenomena, not specific algorithms or techniques.
- **Think beyond the obvious**: The most valuable sub-questions often reveal non-obvious bottlenecks that cross-domain insights can address.
- **Be specific enough to search**: Each sub-question should be concrete enough that relevant literature could be identified.
- **Map to Semantic Scholar domains**: Only use domains from the provided list. If unsure, pick the closest match.
- **Prioritize domains**: List domains in order of likely relevance, with the most promising first.

Now, please decompose the research goal into fundamental sub-questions following these principles.
"""
    
    return prompt

class CrossDomainQuery(BaseModel):
    domain: str
    queries: List[str]

class SubQuestion(BaseModel):
    id: str
    question: str
    rationale: str
    cross_domain_search_queries: List[CrossDomainQuery]
    same_domain_search_queries: List[str]
    current_gaps: str

class Decomposition(BaseModel):
    research_goal: str
    fine_grained_domain: str
    core_challenge_summary: str
    decomposition_reasoning: str
    sub_questions: List[SubQuestion]

decompositon_schema = Decomposition.model_json_schema()
