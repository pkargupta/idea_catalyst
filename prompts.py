from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json


# =============================================================================
# PROMPT 1: Initial Goal Decomposition
# =============================================================================

def create_initial_decomposition_prompt(problem_statement: str, fine_grained_domain: str) -> str:
    """
    Creates a prompt for decomposing a research problem into fundamental questions,
    grounded in a *LLM-selected coarse-grained domain* which encompasses the fine-grained domain, and
    producing BOTH:
      (1) a domain-specific version (for target-domain search), and
      (2) a domain-agnostic version (for external-domain search).
    """

    prompt = f"""You are an expert research strategist. Your task is to:
(1) identify the most relevant **coarse-grained domain** (the target domain) that encompasses the input fine-grained domain for the given research problem,
(2) decompose the problem into fundamental bottleneck questions that are **grounded in that domain and subfield**,
and (3) produce two parallel formulations for each question:
   - a **domain-specific** version (optimized for searching within the identified subfield of the coarse-grained domain)
   - a **domain-agnostic** version (optimized for discovering relevant work in external domains)

# RESEARCH PROBLEM
{problem_statement}

# FINE-GRAINED SUBFIELD
{fine_grained_domain}

# STEP 1: SUBFIELD IDENTIFICATION (MANDATORY)
First, determine the most relevant **coarse-grained domain** (the target domain which encompasses the subfield, {fine_grained_domain}) that the problem best fits out of the following options (ONLY SELECT THE TARGET DOMAIN FROM THIS LIST):

Computer Science, Medicine, Chemistry, Biology, Materials Science, Physics, Geology, Psychology, Art, History, Geography, Sociology, Business, Political Science, Economics, Philosophy, Mathematics, Engineering, Environmental Science, Agricultural and Food Sciences, Education, Law, Linguistics

Then, all subsequent questions and queries MUST be aligned to the domain and subfields' typical concepts, bottlenecks, and vocabulary.

Examples of "coarse_grained_domain" (illustrative):
- "Probabilistic graphical models" → Computer Science, "Distributed systems" → Computer Science, "Program analysis" → Computer Science, "Human-computer interaction" → Computer Science
- "Gene regulatory networks" → Biology, "Protein folding" → Biology, "Microbial ecology" → Biology
- "Market design" → Economics, "Behavioral decision theory" → Economics, "Causal inference" → Economics

# STEP 2: CORE CHALLENGE
Write a 2–3 sentence **core_challenge** that frames the fundamental difficulty *as researchers in the fine_grained_domain would*.

# STEP 3: QUESTION DECOMPOSITION (SUBFIELD-GROUNDED)
Produce 2-4 underlying bottleneck questions. These should be:
- **Atomic**: one distinct conceptual challenge
- **More Fine-Grained**: specific enough to be useful in the identified subfield (avoid generic questions like “How do we improve performance?”)
- **Critical**: necessary to make progress
- **Mechanistic**: focuses on “how/why” (causal/process/constraints)
- **Not solution-prescriptive**: avoid naming a single algorithm/tool as the question (no “tune X hyperparameter”)

IMPORTANT: The chosen questions should be at the *right altitude*:
- Not so broad they apply to every subfield (“How do we model uncertainty?”)
- Not so narrow they pre-commit to a specific technique (“How do we tune Adam’s beta2?”)
Aim for subfield-relevant bottlenecks that recur in the fine_grained_domain literature.

# STEP 4: QUESTION PAIRING (TWO VERSIONS OF THE SAME BOTTLENECK)
For EACH research question, produce:
- **domain_specific_question**: uses terminology typical of the **fine_grained_domain** (subfield-level terms are encouraged)
- **domain_agnostic_question**: same bottleneck, but phrased in cross-disciplinary mechanistic language, avoiding coarse-grained domain / subfield ({fine_grained_domain}) jargon

Constraint: Both questions must refer to the *same* bottleneck.

# STEP 5: SEARCH QUERIES (SUBFIELD-ALIGNED)
For EACH research question, provide:
- 3–5 **target_domain_queries** (max 5 words each)
  - MUST strongly reflect **fine_grained_domain** vocabulary and canonical phrases
  - SHOULD be specific enough to appear in titles/abstracts in that subfield
  - SHOULD vary across theoretical/methodological/evaluative angles

# OUTPUT FORMAT

Return a JSON object:

{{
  "problem_statement": "{problem_statement}",
  "fine_grained_domain": "{fine_grained_domain}",
  "coarse_grained_domain": "Domain selected from the provided options",
  "core_challenge": "2-3 sentence subfield-grounded summary of the fundamental challenge",
  "research_questions": [
    {{
      "id": "q1",
      "domain_specific_question": "Conceptual bottleneck question using fine_grained_domain terminology",
      "domain_agnostic_question": "Same bottleneck, jargon-free mechanistic phrasing",
      "rationale": "Why this is a critical bottleneck in the fine_grained_domain",
      "target_domain_queries": [
        "max five words",
        "subfield canonical phrase",
        "title/abstract likely terms"
      ]
    }}
  ]
}}

# EXAMPLE (BRIEF)

If fine_grained_domain = "Probabilistic graphical models":
- domain_specific_question might reference "identifiability", "latent variables", "structure learning"
- target_domain_queries might include those phrases (<=5 words)
- domain_agnostic_question would translate to general terms like "recover hidden causes from observations"

Now perform the decomposition following these instructions. Ensure every question and query is clearly aligned with the chosen fine_grained_domain.
"""
    return prompt

# =============================================================================
# SCHEMA
# =============================================================================

class ResearchQuestion(BaseModel):
    id: str

    domain_specific_question: str = Field(
        description=(
            "Same bottleneck phrased using terminology typical of the identified fine_grained_domain "
            "(subfield-level vocabulary encouraged)."
        )
    )
    domain_agnostic_question: str = Field(
        description=(
            "Same bottleneck phrased in cross-disciplinary, jargon-free mechanistic language; "
            "must avoid target-domain and fine_grained_domain jargon."
        )
    )

    rationale: str = Field(
        description="Why this bottleneck is critical specifically within the identified fine_grained_domain."
    )

    target_domain_queries: List[str] = Field(
        min_items=3,
        max_items=5,
        description=(
            "Search queries (<=5 words) for within-target-domain search. Must strongly reflect "
            "fine_grained_domain vocabulary and canonical phrases likely in titles/abstracts."
        ),
    )


class InitialDecomposition(BaseModel):
    problem_statement: str

    fine_grained_domain: str = Field(
        description="Most relevant specific subfield for this problem."
    )

    coarse_grained_domain: str = Field(
        description="Most relevant high-level domain which encompoasses the fine-grained domain for this problem."
    )

    core_challenge: str = Field(
        description="2–3 sentence summary of the fundamental challenge, framed in fine_grained_domain terms."
    )

    research_questions: List[ResearchQuestion] = Field(
        min_items=2,
        max_items=4,
        description="2–4 subfield-grounded bottleneck questions, each with domain-specific and domain-agnostic versions.",
    )



# =============================================================================
# PROMPT 2: Fine-Grained Target Subfield Analysis (Explicit Assessment Rubric)
# =============================================================================

def create_target_domain_analysis_prompt(
    research_problem: str,
    domain_specific_question: str,
    domain_agnostic_question: str,
    question_rationale: str,
    papers_with_snippets: Dict[str, List[str]],
    target_domain: str,
    fine_grained_domain: str,
) -> str:
    """
    Creates a prompt for analyzing retrieved papers from a fine-grained subfield
    of the target domain. Remaining challenges and overall assessment are reported
    in both domain-specific and domain-agnostic forms, with an explicit rubric for
    determining coverage.
    """

    # Format papers for the prompt
    papers_formatted: List[str] = []
    for i, (title, snippets) in enumerate(papers_with_snippets.items(), 1):
        papers_formatted.append(f"\n## Paper {i}: {title}")
        for j, snippet in enumerate(snippets, 1):
            papers_formatted.append(f"   Snippet {j}: {snippet}")

    papers_text = "\n".join(papers_formatted)

    prompt = f"""You are an expert research analyst. Analyze retrieved papers from a specific subfield of the target domain to assess progress on a research question that supports the overall research problem.

# RESEARCH PROBLEM
{research_problem}

# TARGET DOMAIN
{target_domain}

# FINE-GRAINED DOMAIN (SUBFIELD)
{fine_grained_domain}

# RESEARCH QUESTION (PAIRED FORMULATIONS)
- **Domain-specific question** (use for relevance and coverage assessment):
{domain_specific_question}

- **Domain-agnostic question** (use for conceptual gap articulation):
{domain_agnostic_question}

**Rationale**: {question_rationale}

# RETRIEVED PAPERS AND SNIPPETS (FROM THE TARGET DOMAIN)
{papers_text}

# YOUR TASK

0. **Subfield grounding**
   - Treat {fine_grained_domain} as the authoritative lens for interpretation.
   - Judge relevance, coverage, and gaps strictly at the *subfield-appropriate level of specificity*.

1. **Assess Paper Relevance**
   For each paper, determine whether it directly addresses the **domain-specific question**
   as understood in {fine_grained_domain}.
   - Mark papers that only provide background, tangential tools, or adjacent applications as *not relevant*.

2. **Identify Addressed Aspects**
   Across all *relevant* papers, identify which specific sub-aspects of the
   domain-specific question are convincingly addressed.
   - Each aspect should correspond to a concrete conceptual or methodological component.
   - If no papers are relevant, output an empty list.

3. **Identify Remaining Challenges (Dual Form)**
   Identify what remains unsolved *given the evidence above*.
   - Each challenge must be:
     - **Atomic** (one bottleneck)
     - **Major** (blocking full resolution of the question)
     - **Non-iterative** (not just “improve X”)
   - For each challenge, output:
     - a **domain_specific_challenge_question** (fine_grained_domain terminology)
     - a **domain_agnostic_challenge_question** (jargon-free, mechanistic framing)
   - If *no papers address the question at all*, the first remaining challenge should
     restate the research question itself (in both forms).
   - If *nothing meaningful remains*, output an empty list.

4. **Determine Overall Assessment (FOLLOW THIS RUBRIC EXACTLY)**

You MUST choose **one** of the following labels using the criteria below.
Do NOT default to "partially addressed".

- **"largely unaddressed"** if:
  - Zero or nearly zero papers are relevant, OR
  - Relevant papers exist but fail to address the *core mechanism* of the question, OR
  - The main question itself appears verbatim (or nearly so) in remaining_challenges.

- **"partially addressed"** if:
  - There is clear, non-trivial progress on *some* core aspects, BUT
  - At least one **major conceptual bottleneck** remains that would prevent a full solution.

- **"substantially addressed"** if:
  - Most core aspects are addressed by multiple papers, AND
  - Remaining challenges (if any) are minor, edge-case, or refinement-level rather than foundational.

Before selecting "partially addressed", explicitly check:
> “Is there decisive evidence that *some* core bottlenecks are solved, but *others fundamentally remain*?”
If the answer is **no**, choose either "largely unaddressed" or "substantially addressed".

# OUTPUT FORMAT

Return a JSON object:

{{
  "domain_specific_question": "{domain_specific_question}",
  "domain_agnostic_question": "{domain_agnostic_question}",
  "target_domain": "{target_domain}",
  "fine_grained_domain": "{fine_grained_domain}",
  "paper_relevance": [
    {{
      "paper_title": "Exact title from above",
      "is_relevant": bool,
      "relevance_explanation": "Why this paper does or does not address the domain-specific question"
    }}
  ],
  "addressed_aspects": [
    {{
      "sub_question": "Which specific aspect of the domain-specific question was addressed?",
      "evidence": "Which papers address this and how?"
    }}
  ],
  "remaining_challenges": [
    {{
      "challenge_id": "c1",
      "domain_specific_challenge_question": "Unsolved challenge in fine_grained_domain terminology",
      "domain_agnostic_challenge_question": "Same challenge in cross-disciplinary language without any target or fine-grained domain jargon.",
      "why_unaddressed": "Why existing work fails to solve this? Write 3-4 sentences on what makes the challenge difficult to solve and what would be necessary to solve it.",
      "importance": "Why this challenge blocks full resolution of the research question"
    }}
  ],
  "overall_assessment": "substantially addressed | partially addressed | largely unaddressed"
}}

# GUIDELINES

- Be evidence-driven; do not infer unstated paper contributions
- Prefer "largely unaddressed" over "partially addressed" when in doubt
- The assessment must be logically consistent with addressed_aspects and remaining_challenges

Now analyze the papers.
"""
    return prompt

# =============================================================================
# SCHEMA
# =============================================================================

class PaperRelevance(BaseModel):
    paper_title: str
    is_relevant: bool
    relevance_explanation: str


class AddressedAspect(BaseModel):
    sub_question: str
    evidence: str


class RemainingChallenge(BaseModel):
    challenge_id: str
    domain_specific_challenge_question: str = Field(
        description="Unsolved challenge phrased using fine_grained_domain terminology."
    )
    domain_agnostic_challenge_question: str = Field(
        description="Same challenge phrased in cross-disciplinary, jargon-free language."
    )
    why_unaddressed: str
    importance: str


class TargetDomainAnalysis(BaseModel):
    domain_specific_question: str
    domain_agnostic_question: str
    target_domain: str
    fine_grained_domain: str
    paper_relevance: List[PaperRelevance]
    addressed_aspects: List[AddressedAspect]
    remaining_challenges: List[RemainingChallenge]
    overall_assessment: str = Field(
        description='One of: "substantially addressed", "partially addressed", "largely unaddressed".'
    )


# =============================================================================
# PROMPT 3: Cross-Domain Query Generation (Subfield-Aware, Domain-Agnostic)
# =============================================================================

def create_cross_domain_query_prompt(
    problem_statement: str,
    domain_specific_question: str,
    domain_agnostic_question: str,
    question_rationale: str,
    target_domain: str,
    fine_grained_domain: str,
    target_domain_assessment: Optional[str] = None,
) -> str:
    """
    Creates a prompt for identifying cross-domain search queries using the
    domain-agnostic version of a research question, while grounding context
    in the target domain and fine-grained subfield.
    """


    prompt = f"""You are an expert at identifying cross-disciplinary research connections. Your task is to identify **external domains** and **search queries** that could address a research challenge by analogy, shared mechanisms, or transferred principles.

# RESEARCH PROBLEM
{problem_statement}

# RESEARCH QUESTION (PAIRED FORMULATIONS)

- **Domain-specific version** (for context only — do NOT reuse terminology):
{domain_specific_question}

- **Domain-agnostic version** (PRIMARY DRIVER for cross-domain search):
{domain_agnostic_question}

# ORIGINAL DOMAIN CONTEXT
- Broad domain: {target_domain}
- Fine-grained subfield: {fine_grained_domain}

# CHALLENGES: WHY CURRENT {target_domain.upper()} RESEARCH (IN {fine_grained_domain.upper()}) IS INSUFFICIENT
{question_rationale}

# VALID SEMANTIC SCHOLAR DOMAINS
Computer Science, Medicine, Chemistry, Biology, Materials Science, Physics, Geology, Psychology, Art, History, Geography, Sociology, Business, Political Science, Economics, Philosophy, Mathematics, Engineering, Environmental Science, Agricultural and Food Sciences, Education, Law, Linguistics

# YOUR TASK

Using the **domain-agnostic version of the question**, identify **1–3 external domains**
(from the list above, excluding {target_domain}) that are likely to contain relevant insights for directly addressing the challenges that {target_domain.upper()} research has in addressing the research question..

These domains should have studied:
- analogous mechanisms to solve the challenges,
- structurally similar problems,
- or transferable principles

even if the surface application differs from {fine_grained_domain}.

For EACH selected domain:
- Explain *why* this domain is a good match for addressing the challenges in solving the domain-agnostic question
- Provide **2–4 search queries** suitable for that domain

# QUERY DESIGN PRINCIPLES

Each query must be:
- **Domain-appropriate**: use terminology natural to the selected external field
- **Mechanistic**: targets underlying processes, constraints, or principles
- **Concise**: maximum 5 words
- **Specific**: likely to appear in paper titles or abstracts
- **Non-redundant**: queries should reflect different angles within the same domain

STRICT CONSTRAINTS:
- Do NOT reuse terminology specific to {target_domain} or {fine_grained_domain}
- Do NOT restate the domain-specific question
- Queries must be directly derived from the **domain-agnostic question**

# OUTPUT FORMAT

Return a JSON object:

{{
  "domain_specific_question": "{domain_specific_question}",
  "domain_agnostic_question": "{domain_agnostic_question}",
  "cross_domain_searches": [
    {{
      "domain": "Valid domain from the list",
      "domain_rationale": "Why this domain likely has relevant insights for the domain-agnostic question",
      "queries": [
        "domain-specific query #1",
        "domain-specific query #2",
        "domain-specific query #3"
      ]
    }}
  ]
}}

# EXAMPLE

Domain-agnostic question:
"How can a system link delayed outcomes to earlier decisions?"

Good output:
{{
  "cross_domain_searches": [
    {{
      "domain": "Psychology",
      "domain_rationale": "Psychology studies how agents associate actions with delayed feedback through learning and conditioning",
      "queries": ["delayed reinforcement learning", "trace conditioning", "temporal contiguity effects"]
    }},
    {{
      "domain": "Economics",
      "domain_rationale": "Economic models analyze how agents make decisions under delayed or deferred consequences",
      "queries": ["intertemporal choice", "delayed incentives", "dynamic decision making"]
    }}
  ]
}}

Bad examples:
- Using target-domain terms in queries
- Selecting domains not in the provided list
- Generic queries like ["learning", "decision making"]
- Queries longer than five words

Now identify cross-domain searches for this research question.
"""
    return prompt


# =============================================================================
# SCHEMA
# =============================================================================

class CrossDomainSearch(BaseModel):
    domain: str
    domain_rationale: str
    queries: List[str] = Field(min_items=2, max_items=4)


class CrossDomainQueries(BaseModel):
    domain_specific_question: str
    domain_agnostic_question: str
    cross_domain_searches: List[CrossDomainSearch] = Field(min_items=1, max_items=3)


# =============================================================================
# PROMPT 4: Cross-Domain Relevance and Takeaways
# =============================================================================

def create_cross_domain_analysis_prompt(
    problem_statement: str,
    domain_agnostic_question: str,
    question_challenge: str,
    source_domain: str,
    papers_with_snippets: Dict[str, List[str]],
    target_domain: str,
    fine_grained_domain: str
) -> str:
    """
    Creates a prompt for analyzing cross-domain papers to find solutions to challenges.
    
    Args:
        problem_statement: The overall research problem
        domain_specific_question: Question phrased in target domain terminology
        domain_agnostic_question: Question phrased in general terms
        question_challenge: The specific challenge to be solved
        source_domain: The external domain being analyzed
        papers_with_snippets: Dict mapping paper titles to lists of snippets
        target_domain: The broad target domain (e.g., "Computer Science")
        fine_grained_domain: The specific subfield (e.g., "Reinforcement Learning")
        
    Returns:
        Formatted prompt string for LLM
    """
    
    # Format papers for the prompt
    papers_formatted = []
    for i, (title, snippets) in enumerate(papers_with_snippets.items(), 1):
        papers_formatted.append(f"\n## Paper {i}: {title}")
        for j, snippet in enumerate(snippets, 1):
            papers_formatted.append(f"   Snippet {j}: {snippet}")
    
    papers_text = "\n".join(papers_formatted) if papers_formatted else "No papers retrieved."
    
    prompt = f"""You are an expert at identifying cross-disciplinary solutions to research challenges. Your goal is to analyze papers from an external domain to determine if they solve a specific challenge and extract actionable solution approaches.

# RESEARCH PROBLEM
{problem_statement}

# TARGET DOMAIN CONTEXT
- **Broad Domain**: {target_domain}
- **Specific Subfield**: {fine_grained_domain}

# THE CHALLENGE TO SOLVE

**Conceptual Challenge Formulation**:
{domain_agnostic_question}

**What makes this challenging? What is the bottleneck to solving it**:
{question_challenge}

# SOURCE DOMAIN TO ANALYZE
{source_domain}

# RETRIEVED PAPERS AND SNIPPETS FROM {source_domain.upper()}
{papers_text}

# YOUR TASK

## 1. Assess Paper Relevance
For each paper, determine if it **attempts to solve** the conceptual challenge (not just tangentially related).

## 2. Extract Solution Takeaways
For papers that address the challenge, identify how they solve it. Each takeaway should:
- Capture a **concrete solution approach or mechanism**
- Be presented in the following **formulation**:
  - **{source_domain}-specific**: Using natural {source_domain} terminology and concepts. Be detailed in your explanation.
- Be **evidence-based**: Clearly grounded in the provided papers
- Focus on **how the solution works**, not just what it achieves

## 3. Synthesize Overall Assessment
Based on all takeaways, judge whether the conceptual challenge is sufficiently addressed by this source domain.

# OUTPUT FORMAT

Return a JSON object:

{{
    "conceptual_challenge": "{domain_agnostic_question}",
    "source_domain": "{source_domain}",
    "target_domain": "{target_domain}",
    "fine_grained_domain": "{fine_grained_domain}",
    
    "paper_relevance": [
        {{
            "paper_title": "Exact title from above",
            "directly_addresses_challenge": true,
            "relevance_explanation": "Explain how this paper does/doesn't directly attempt to solve the challenge"
        }}
    ],
    
    "solution_takeaways": [
        {{
            "takeaway_id": "t1",
            "source_domain_formulation": "Description of the solution using {source_domain} terminology and concepts",
            "mechanism_explanation": "How this solution approach works and why it addresses the challenge. If applicable, provide examples to make it intuitive.",
            "supporting_papers": ["Paper Title 1", "Paper Title 2"]
        }}
    ],
    
    "challenge_sufficiency_assessment": {{
        "is_challenge_addressed": true,
        "assessment_explanation": "Explain whether the conceptual crux of the challenge is sufficiently solved by the source domain's approaches",
        "key_solutions_summary": "Brief summary of the main solution approaches found",
        "remaining_gaps": "What aspects of the challenge remain unaddressed (if any)"
    }}
}}

# GUIDELINES

- **Direct relevance only**: Only mark papers as relevant if they actively try to solve the challenge, not just mention related concepts
- **Conservative assessment**: Only include takeaways you're confident are well-supported by evidence
- **Dual formulations are critical**: Each takeaway must have both source-domain and target-domain versions
  - Source-domain version should use natural terminology from that field
- **Focus on mechanisms**: Explain *how* solutions work, not just *what* they achieve
- **Challenge sufficiency**: Judge based on whether the core problem is solved, not whether implementation details are provided
- **Honest assessment**: If the source domain doesn't adequately address the challenge, say so

# EXAMPLE

Challenge: "How can a system attribute delayed outcomes to earlier actions?"
Source Domain: "Psychology"
Fine-Grained Domain: "Reinforcement Learning"

Good takeaway:
{{
    "takeaway_id": "t1",
    "source_domain_formulation": "Animals maintain decaying memory traces of conditioned stimuli that persist after stimulus offset. When an unconditioned stimulus (reward) arrives later, learning occurs proportionally to the remaining trace strength, enabling associations across temporal gaps.",
    "mechanism_explanation": "By preserving a gradually fading representation of past events, the system maintains a 'bridge' that connects earlier decisions to later outcomes. The decay rate controls how far back credit propagates, balancing recency bias with long-term dependencies.",
    "supporting_papers": ["Temporal Conditioning in Animal Learning", "Trace Decay Mechanisms in Associative Learning"]
}}

Bad takeaway (too vague and unrelated to challenge):
- "Systems should learn from past experiences" → Doesn't explain delayed attribution

Bad takeaway (too shallow):
- "Use memory to connect past and present" → Needs to explain *how* memory is structured and used

Now analyze whether {source_domain} papers solve the challenge.
"""
    
    return prompt


class SolutionTakeaway(BaseModel):
    takeaway_id: str
    source_domain_formulation: str
    mechanism_explanation: str
    supporting_papers: List[str]


class ChallengeSufficiencyAssessment(BaseModel):
    is_challenge_addressed: bool
    assessment_explanation: str
    key_solutions_summary: str
    remaining_gaps: str


class PaperChallengeRelevance(BaseModel):
    paper_title: str
    directly_addresses_challenge: bool
    relevance_explanation: str


class CrossDomainAnalysis(BaseModel):
    conceptual_challenge: str
    source_domain: str
    target_domain: str
    fine_grained_domain: str
    paper_relevance: List[PaperChallengeRelevance]
    solution_takeaways: List[SolutionTakeaway] = Field(min_items=1, max_items=10)
    challenge_sufficiency_assessment: ChallengeSufficiencyAssessment


# =============================================================================
# PROMPT 5: Target Domain Framing
# =============================================================================

def create_target_domain_framing_prompt(
    problem_statement: str,
    question: str,
    source_domain: str,
    target_domain: str,
    high_level_takeaways: List[dict]
) -> str:
    """
    Creates a prompt for framing cross-domain takeaways in target domain context.
    
    Args:
        problem_statement: The research problem
        question: The research question
        source_domain: The external domain the takeaways came from
        target_domain: The target domain to frame takeaways for
        high_level_takeaways: List of takeaway dicts with principle, explanation, evidence
        
    Returns:
        Formatted prompt string for LLM
    """
    
    # Format takeaways
    takeaways_formatted = []
    for i, takeaway in enumerate(high_level_takeaways, 1):
        takeaways_formatted.append(f"\n## Takeaway {i} (ID: {takeaway.get('takeaway_id', f't{i}')})")
        takeaways_formatted.append(f"**Principle**: {takeaway.get('principle', 'N/A')}")
        takeaways_formatted.append(f"**Explanation**: {takeaway.get('explanation', 'N/A')}")
        takeaways_formatted.append(f"**Evidence**: {takeaway.get('supporting_evidence', 'N/A')}")
    
    takeaways_text = "\n".join(takeaways_formatted) if takeaways_formatted else "No takeaways provided."
    
    prompt = f"""You are an expert at translating cross-disciplinary insights into domain-specific applications. Your task is to frame general principles from one domain in the language and context of the target domain.

# RESEARCH PROBLEM
{problem_statement}

# RESEARCH QUESTION
{question}

# SOURCE DOMAIN
{source_domain}

# TARGET DOMAIN
{target_domain}

# HIGH-LEVEL TAKEAWAYS FROM {source_domain.upper()}
{takeaways_text}

# YOUR TASK

For each takeaway, translate it into {target_domain}-specific language and highlight concrete implications. Each framed takeaway should:

1. **Restate the principle** using {target_domain} terminology and concepts
2. **Identify concrete applications** in {target_domain} that this principle suggests
3. **Highlight specific challenges** in {target_domain} that this principle could address
4. **Suggest potential approaches** that could implement this principle in {target_domain}

# OUTPUT FORMAT

Return a JSON object:

{{
    "question": "{question}",
    "source_domain": "{source_domain}",
    "target_domain": "{target_domain}",
    "framed_takeaways": [
        {{
            "takeaway_id": "t1",
            "original_principle": "The general principle from source domain",
            "target_domain_framing": "How this principle translates to {target_domain} language and concepts",
            "concrete_applications": [
                "Specific application 1 in {target_domain}",
                "Specific application 2 in {target_domain}"
            ],
            "addresses_challenges": [
                "Specific {target_domain} challenge this could help with"
            ],
            "potential_approaches": [
                "Concrete approach or method that could implement this principle in {target_domain}"
            ]
        }}
    ],
    "overall_synthesis": "2-3 sentence summary of how these {source_domain} insights collectively inform the {target_domain} question"
}}

# GUIDELINES

- Use terminology, concepts, and examples natural to {target_domain}
- Be specific - avoid generic statements that could apply to any domain
- Focus on actionable insights that could inform actual {target_domain} work
- Identify concrete connections to existing {target_domain} concepts or methods
- If a principle doesn't translate well, explain why and what's missing
- Prioritize insights that are novel or under-explored in {target_domain}

# EXAMPLE

Source Domain: Psychology
Target Domain: Computer Science
Original Principle: "Temporal associations can be strengthened by maintaining eligibility traces that decay over time"

Good framing:
{{
    "target_domain_framing": "In reinforcement learning, agents can bridge temporal credit assignment gaps by maintaining exponentially decaying eligibility traces for state-action pairs, allowing delayed rewards to update earlier decisions proportionally to trace strength",
    "concrete_applications": [
        "Implementing eligibility traces (e.g., TD(λ)) in RL agents to handle delayed rewards",
        "Using trace decay parameters to control how far back credit propagates"
    ],
    "addresses_challenges": [
        "The temporal credit assignment problem when rewards are delayed",
        "Sample efficiency in sparse reward environments"
    ],
    "potential_approaches": [
        "TD(λ) algorithm with tunable trace decay rates",
        "Neural network architectures with built-in temporal trace mechanisms"
    ]
}}

Bad framing (too generic):
"Systems should remember past events to connect them with future outcomes"
→ Not specific to {target_domain}, no concrete applications

Now frame the {source_domain} takeaways for {target_domain}.
"""
    
    return prompt


class FramedTakeaway(BaseModel):
    takeaway_id: str
    original_principle: str
    target_domain_framing: str
    concrete_applications: List[str] = Field(min_items=1, max_items=5)
    addresses_challenges: List[str] = Field(min_items=1, max_items=5)
    potential_approaches: List[str] = Field(min_items=1, max_items=5)


class TargetDomainFraming(BaseModel):
    question: str
    source_domain: str
    target_domain: str
    framed_takeaways: List[FramedTakeaway]
    overall_synthesis: str

# =============================================================================
# PROMPT 6: Target Domain Integration
# =============================================================================

def create_target_domain_integration_prompt(
    problem_statement: str,
    domain_specific_question: str,
    domain_agnostic_question: str,
    question_challenge: str,
    target_domain: str,
    fine_grained_domain: str,
    target_domain_papers: Dict[str, List[str]],
    source_domain: str,
    source_domain_papers: Dict[str, List[str]],
    source_domain_takeaways: List[dict]
) -> str:
    """
    Creates a prompt for integrating cross-domain takeaways with target domain 
    state-of-the-art to generate a novel insight/idea fragment.
    
    Args:
        problem_statement: The overall research problem
        domain_specific_question: Question in target domain terminology
        domain_agnostic_question: Question in general terms
        question_challenge: The specific challenge to be solved
        target_domain: The broad target domain
        fine_grained_domain: The specific subfield
        target_domain_papers: Papers from target domain with snippets
        source_domain: The external domain providing insights
        source_domain_papers: Relevant papers from source domain with snippets
        source_domain_takeaways: List of takeaway dicts from source domain
        
    Returns:
        Formatted prompt string for LLM
    """
    
    # Format target domain papers
    target_papers_formatted = []
    for i, (title, snippets) in enumerate(target_domain_papers.items(), 1):
        target_papers_formatted.append(f"\n## Target Domain Paper {i}: {title}")
        for j, snippet in enumerate(snippets, 1):
            target_papers_formatted.append(f"   Snippet {j}: {snippet}")
    target_papers_text = "\n".join(target_papers_formatted) if target_papers_formatted else "No papers available."
    
    # Format source domain papers
    source_papers_formatted = []
    for i, (title, snippets) in enumerate(source_domain_papers.items(), 1):
        source_papers_formatted.append(f"\n## Source Domain Paper {i}: {title}")
        for j, snippet in enumerate(snippets, 1):
            source_papers_formatted.append(f"   Snippet {j}: {snippet}")
    source_papers_text = "\n".join(source_papers_formatted) if source_papers_formatted else "No papers available."
    
    # Format takeaways
    takeaways_formatted = []
    for i, takeaway in enumerate(source_domain_takeaways, 1):
        takeaways_formatted.append(f"\n## Takeaway {i} (ID: {takeaway.get('takeaway_id', f't{i}')})")
        takeaways_formatted.append(f"**Source Domain Formulation**: {takeaway.get('source_domain_formulation', 'N/A')}")
        takeaways_formatted.append(f"**Mechanism**: {takeaway.get('mechanism_explanation', 'N/A')}")
    takeaways_text = "\n".join(takeaways_formatted) if takeaways_formatted else "No takeaways provided."
    
    prompt = f"""You are an expert at interdisciplinary research integration. Your task is to synthesize insights from an external domain with the current state-of-the-art in the target domain to generate a novel, actionable research idea fragment.

# RESEARCH PROBLEM
{problem_statement}

# TARGET DOMAIN CONTEXT
- **Broad Domain**: {target_domain}
- **Fine-Grained Subfield**: {fine_grained_domain}

# THE CHALLENGE TO ADDRESS
**Domain-Specific Question**: {domain_specific_question}

**Domain-Agnostic Question**: {domain_agnostic_question}

**Challenge Details**: {question_challenge}

# CURRENT STATE-OF-THE-ART IN {fine_grained_domain.upper()}
{target_papers_text}

# INSIGHTS FROM {source_domain.upper()}
## Relevant Papers from {source_domain}
{source_papers_text}

## Key Takeaways from {source_domain}
{takeaways_text}

# YOUR TASK

Generate a single, coherent **idea fragment** that integrates the most promising takeaways from {source_domain} with the current state-of-the-art in {fine_grained_domain}. This idea fragment should:

1. **Select the most promising takeaways**: Not all takeaways need to be used. Focus on those that offer the strongest potential for integration with {fine_grained_domain}.

2. **Address three key integration questions**:
   a. **How can you integrate the takeaways from {source_domain} with current state-of-the-art ideas in {fine_grained_domain}?**
      - Identify specific concepts, methods, or frameworks from both domains that can be combined
      - Explain how these elements complement each other
   
   b. **How does this integration address the research problem and the target domain challenge?**
      - Show how the integrated approach solves limitations in current {fine_grained_domain} methods
      - Demonstrate how it advances progress on the domain-specific question
   
   c. **What are the challenges with the external domain's takeaways, and how does integration address them?**
      - Identify potential limitations or gaps in the {source_domain} takeaways
      - Explain how combining with {fine_grained_domain} approaches mitigates these challenges

3. **Provide a concrete integration pathway**: Describe specific steps or mechanisms for combining the approaches, not just high-level aspirations.

# OUTPUT FORMAT

Return a JSON object:

{{
    "idea_fragment": {{
        "title": "Brief, descriptive title for the integrated idea (max 15 words)",
        "core_insight": "2-3 sentence summary of the key integrated insight",
        "integration_mechanism": {{
            "target_domain_elements": [
                "Specific concept/method from {fine_grained_domain} paper 1",
                "Specific concept/method from {fine_grained_domain} paper 2"
            ],
            "selected_takeaways": [
                {{
                    "takeaway_id": "t1",
                    "selection_rationale": "Why this takeaway was selected for integration"
                }}
            ],
            "synthesis_approach": "Brief explanation (1-2 sentences) of how these elements combine with the selected_takeaways into a unified framework or method"
        }},
        "challenge_resolution": {{
            "addresses_target_challenge": "How this addresses the specific {fine_grained_domain} challenge",
            "addresses_source_limitations": "How integration with {fine_grained_domain} overcomes limitations in {source_domain} takeaways",
            "addresses_research_problem": "How this contributes to solving the overall research problem"
        }},
        "concrete_realization": {{
            "proposed_approach": "Specific technical approach, algorithm, or framework that realizes this integration (3-4 sentences)",
            "key_innovations": [
                "Novel aspect 1 that wouldn't exist in either domain alone",
                "Novel aspect 2 that emerges from the integration"
            ]
        }}
    }}
}}

# GUIDELINES

- **Be specific**: Avoid vague statements like "combine X and Y". Explain exactly HOW they combine.
- **Be evidence-based**: Ground your integration in the specific papers and takeaways provided.
- **Focus on actionability**: The idea fragment should suggest a clear research direction, not just a philosophical observation.
- **Prioritize depth over breadth**: Better to deeply integrate 1-2 takeaways than superficially mention many.
- **Highlight emergence**: Emphasize what's novel about the integration that wouldn't exist in either domain alone.

# EXAMPLE STRUCTURE (NOT TO BE COPIED)

If integrating Psychology's "uncertainty estimation through probabilistic inference" with Computer Science's "RNN-based sequence modeling":

Good integration:
- Proposes specific RNN architecture modifications inspired by cognitive mechanisms
- Explains how uncertainty estimates from psychology inform loss functions or attention mechanisms
- Describes concrete algorithm that combines both insights

Bad integration:
- "Use psychological principles to improve RNNs" (too vague)
- Lists psychology concepts and CS concepts side-by-side without connection
- Proposes using one domain's method without actual integration

Now generate the integrated idea fragment.
"""
    
    return prompt


class SelectedTakeaway(BaseModel):
    takeaway_id: str
    selection_rationale: str


class IntegrationMechanism(BaseModel):
    target_domain_elements: List[str] = Field(
        min_items=1,
        description="Specific concepts/methods from target domain papers"
    )
    selected_takeaways: List[SelectedTakeaway] = Field(min_items=1)
    synthesis_approach: str = Field(
        description="Detailed explanation of how elements combine"
    )


class ChallengeResolution(BaseModel):
    addresses_target_challenge: str
    addresses_source_limitations: str
    addresses_research_problem: str


class ConcreteRealization(BaseModel):
    proposed_approach: str
    key_innovations: List[str] = Field(min_items=1, max_items=5)


class IdeaFragment(BaseModel):
    title: str
    core_insight: str
    integration_mechanism: IntegrationMechanism
    challenge_resolution: ChallengeResolution
    concrete_realization: ConcreteRealization


class TargetDomainIntegration(BaseModel):
    idea_fragment: IdeaFragment


# =============================================================================
# PROMPT 7: Interdisciplinary Potential Pairwise Comparison
# =============================================================================

def create_interdisciplinary_comparison_prompt(
    problem_statement: str,
    target_domain: str,
    fine_grained_domain: str,
    integrated_ideas: List[dict]  # List of {question, source_domain, idea_fragment} dicts
) -> str:
    """
    Creates a prompt for pairwise comparison of integrated ideas by their interdisciplinary potential.
    
    Args:
        problem_statement: The overall research problem
        target_domain: The broad target domain
        fine_grained_domain: The specific subfield
        integrated_ideas: List of dicts with 'question', 'source_domain', and 'idea_fragment'
        
    Returns:
        Formatted prompt string for LLM
    """
    
    # Format integrated ideas
    ideas_formatted = []
    for i, idea_dict in enumerate(integrated_ideas, 1):
        question = idea_dict['question']
        source_domain = idea_dict['source_domain']
        idea = idea_dict['idea_fragment']
        
        ideas_formatted.append(f"\n## Idea {i}")
        ideas_formatted.append(f"**Source Domain**: {source_domain}")
        ideas_formatted.append(f"**Addresses Challenge**: {question}")
        ideas_formatted.append(json.dumps(idea, indent=4))
    
    ideas_text = "\n".join(ideas_formatted) if ideas_formatted else "No ideas provided."
    
    prompt = f"""You are an expert in evaluating interdisciplinary research potential. Your task is to perform pairwise comparisons of integrated research ideas based on their interdisciplinary potential and rank them accordingly.

# RESEARCH PROBLEM
{problem_statement}

# TARGET DOMAIN CONTEXT
- **Broad Domain**: {target_domain}
- **Fine-Grained Subfield**: {fine_grained_domain}

# INTEGRATED IDEAS TO COMPARE
{ideas_text}

# EVALUATION CRITERIA

Between the two ideas, determine which idea has stronger interdisciplinary potential based on these criteria:

## 1. DEPTH OF INTEGRATION
Which idea better combines theories, models, or methods into a genuinely shared framework where components from both domains are co-designed and mutually constrain each other?

**Strong integration**: Elements are truly unified in a single framework, not just sequentially applied. Domains inform each other's design. Removing one domain would fundamentally break the approach.

**Weak integration**: Methods placed side-by-side or one domain's method applied to another's problem without deep synthesis.

## 2. MULTI-STAGE DISCIPLINARY ENGAGEMENT
Which idea better requires perspectives and skills from both disciplines across multiple research stages (problem definition, study design, data collection, analysis, interpretation)?

**Strong engagement**: Expertise from both domains needed throughout. Both domains required to properly formulate the problem. Sustained collaboration necessary.

**Weak engagement**: One discipline defines problem/approach; other consulted for single step.

## 3. INNOVATION PAYOFF
Which idea has a clearer, more plausible path to new concepts, tools, or solutions that would NOT emerge within either discipline alone?

**Strong payoff**: Specific novel capability emerges ONLY from integration. Clear articulation of why cross-domain synthesis is necessary. Not achievable by incremental advances in one domain.

**Weak payoff**: Outcome could likely be achieved within one discipline or novelty is unclear.

## 4. NOVELTY + FEASIBILITY
Which idea better balances groundbreaking novelty with practical feasibility?

**Strong balance**: Proposes genuinely novel approach with clear, concrete implementation pathway. Innovation is ambitious but achievable.

**Weak balance**: Either too incremental (low novelty) or too speculative (low feasibility).

# TASK

1. **Perform all pairwise comparisons**: For each pair of ideas, determine which has stronger interdisciplinary potential overall, considering all four criteria.

2. **Provide justification**: For each comparison, explain your preference with specific evidence from the ideas.

3. **Aggregate to rankings**: Based on pairwise preferences, produce a final ranking.

# OUTPUT FORMAT

Return a JSON object:

{{
    "pairwise_comparisons":
        {{
            "idea_1_index": 1,
            "idea_2_index": 2,
            "preferred_idea_index": 1,
            "preference_strength": "strong",
            "justification": "Detailed explanation of why idea 1 is preferred over idea 2, addressing specific criteria",
            "criteria_preferences": {{
                "depth_of_integration": "index of idea with stronger integration",
                "multi_stage_engagement": "index of idea with stronger engagement",
                "innovation_payoff": "index of idea with stronger payoff",
                "novelty_feasibility": "index of idea with better balance"
            }},
            "overall_winner": "index of idea with stronger interdisciplinary potential",
            "overall_assessment": "2-3 sentence summary of why this idea has the strongest interdisciplinary potential",
            "key_strengths": [
                "Specific strength across criteria",
                "Another key strength"
            ],
            "key_limitations": [
                "Specific limitation",
                "Another limitation"
            ]
        }}
}}

# GUIDELINES

- **Be thorough**: Compare every pair of ideas (n choose 2 comparisons for n ideas)
- **Be specific**: Justify preferences with concrete evidence from idea descriptions
- **Consider all criteria**: Don't over-weight one dimension; balance all four
- **Be discriminating**: Use "strong" preference when clear winner; "moderate" when closer; "weak" when very similar
- **Focus on emergence**: Prioritize ideas where integration creates something fundamentally new
- **Balance ambition and realism**: Novel ideas should also be feasible

# PREFERENCE STRENGTH GUIDE

- **strong**: Clear winner across most/all criteria
- **moderate**: Winner on balance, but closer call
- **weak**: Very similar; slight edge to one

Now perform the pairwise comparisons and generate final rankings.
"""
    
    return prompt


class CriteriaPreferences(BaseModel):
    depth_of_integration: int
    multi_stage_engagement: int
    innovation_payoff: int
    novelty_feasibility: int


class PairwiseComparison(BaseModel):
    idea_1_index: int
    idea_2_index: int
    preferred_idea_index: int
    preference_strength: str = Field(description="strong, moderate, or weak")
    justification: str
    criteria_preferences: CriteriaPreferences
    overall_winner: int
    overall_assessment: str
    key_strengths: List[str] = Field(min_items=1, max_items=3)
    key_limitations: List[str] = Field(min_items=1, max_items=3)


class InterdisciplinaryComparison(BaseModel):
    pairwise_comparisons: PairwiseComparison

# =============================================================================
# Schema exports for easy access
# =============================================================================

initial_decomposition_schema = InitialDecomposition.model_json_schema()
target_domain_analysis_schema = TargetDomainAnalysis.model_json_schema()
cross_domain_queries_schema = CrossDomainQueries.model_json_schema()
cross_domain_analysis_schema = CrossDomainAnalysis.model_json_schema()
target_domain_framing_schema = TargetDomainFraming.model_json_schema()
target_domain_integration_schema = TargetDomainIntegration.model_json_schema()
interdisciplinary_comparison_schema = InterdisciplinaryComparison.model_json_schema()
