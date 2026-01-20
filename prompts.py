from pydantic import BaseModel, Field
from typing import List, Optional


# =============================================================================
# PROMPT 1: Initial Goal Decomposition
# =============================================================================

def create_initial_decomposition_prompt(problem_statement: str, target_domain: str) -> str:
    """
    Creates a prompt for decomposing a research problem into fundamental questions
    with target-domain search queries.
    
    Args:
        problem_statement: The research problem to decompose
        target_domain: The primary domain (e.g., "Computer Science")
        
    Returns:
        Formatted prompt string for LLM
    """
    
    prompt = f"""You are an expert research strategist. Your task is to decompose a research problem into subfield-specific fundamental questions and identify fine-grained-domain search queries.

# RESEARCH PROBLEM
{problem_statement}

# TARGET DOMAIN
{target_domain}

# YOUR TASK
Break down this problem into 3-5 fundamental research questions that represent core conceptual bottlenecks. For each question, provide 3-5 concise search queries (max 5 words each) to find relevant work in the target domain.

# DECOMPOSITION PRINCIPLES

Each research question must be:
- **Atomic**: Addresses one distinct conceptual challenge for solving the research problem
- **More Fine-Grained**: Tackles a more specific question (fine-grained) within the research problem (coarse-grained)
- **Critical**: Necessary for making progress on the problem
- **Cross-disciplinary**: Formulated to allow mapping to other fields (avoid domain-specific jargon)
- **Mechanistic**: Focuses on "how" or "why", not just "what"

# SEARCH QUERY REQUIREMENTS

Each query must be:
- **Concise**: Maximum 5 words
- **Specific**: Uses precise terminology likely in paper titles/abstracts and reflects critical term(s) from the research problem + question
- **Diverse**: Cover different aspects or approaches to the research problem

# OUTPUT FORMAT

Return a JSON object:

{{
    "problem_statement": "{problem_statement}",
    "target_domain": "{target_domain}",
    "fine_grained_domain": "Specific subfield within {target_domain}",
    "core_challenge": "2-3 sentence summary of the fundamental challenge",
    "research_questions": [
        {{
            "id": "q1",
            "question": "Clear, conceptual question avoiding domain jargon",
            "rationale": "Why this is a critical bottleneck",
            "target_domain_queries": [
                "query one max five words",
                "query two concise specific",
                "query three focused searchable"
            ]
        }}
    ]
}}

# EXAMPLE

For problem: "Develop AI agents that learn from sparse, delayed rewards"
Good question: "How can a system attribute delayed outcomes to earlier actions?"
Good queries: ["temporal credit assignment", "delayed reward learning", "eligibility trace methods"]

Bad question: "How do we optimize TD-lambda hyperparameters?" (too specific, solution-focused)
Bad queries: ["machine learning", "how to do reinforcement learning with delays"] (too generic/long)

Now decompose the research problem following these principles.
"""
    
    return prompt


class ResearchQuestion(BaseModel):
    id: str
    question: str
    rationale: str
    target_domain_queries: List[str] = Field(min_items=3, max_items=5)


class InitialDecomposition(BaseModel):
    problem_statement: str
    target_domain: str
    fine_grained_domain: str
    core_challenge: str
    research_questions: List[ResearchQuestion]


# =============================================================================
# PROMPT 2: Target Domain Analysis
# =============================================================================

def create_target_domain_analysis_prompt(
    research_problem: str,
    question: str,
    question_rationale: str,
    papers_with_snippets: dict,
    target_domain: str
) -> str:
    """
    Creates a prompt for analyzing retrieved papers from the target domain.
    
    Args:
        question: The research question being analyzed
        question_rationale: Why this question is important
        papers_with_snippets: Dict mapping paper titles to lists of snippets
        target_domain: The domain papers were retrieved from
        
    Returns:
        Formatted prompt string for LLM
    """
    
    # Format papers for the prompt
    papers_formatted = []
    for i, (title, snippets) in enumerate(papers_with_snippets.items(), 1):
        papers_formatted.append(f"\n## Paper {i}: {title}")
        for j, snippet in enumerate(snippets, 1):
            papers_formatted.append(f"   Snippet {j}: {snippet}")
    
    papers_text = "\n".join(papers_formatted)
    
    prompt = f"""You are an expert research analyst. Analyze retrieved papers from the target domain to assess progress on a research question that should ultimately target the research problem.

# RESEARCH PROBLEM
{research_problem}

# RESEARCH QUESTION
{question}

**Rationale**: {question_rationale}

# TARGET DOMAIN
{target_domain}

# RETRIEVED PAPERS AND SNIPPETS
{papers_text}

# YOUR TASK

1. **Assess Relevance**: For each paper, determine if it genuinely addresses the research question (not just tangentially related)

2. **Identify Addressed Sub-questions**: Across all relevant papers, what specific sub-questions or aspects of the main question have been adequately addressed? If no papers are relevant, output an empty list.

3. **Identify Remaining Challenges**: What meaningful challenges for solving the research question ({question}) remain unaddressed? These should be:
   - **Non-iterative**: Not just "do X better" but fundamentally different problems
   - **Major**: Represent significant conceptual or practical bottlenecks for solving the research question that are not addressed by the papers or the target domain to the best of your knowledge.
   - **Atomic**: Each challenge is a distinct issue
   - **Formulated as a question**: If the original research question was not addressed at all, just repeat it here. Otherwise, word the remaining challenges as clear questions.
   If no challenges remain, output an empty list.

4. **Overall Assessment**: Based on the above, is the research question "substantially addressed", "partially addressed", or "largely unaddressed" in this domain?

# OUTPUT FORMAT

Return a JSON object:

{{
    "question": "{question}",
    "paper_relevance": [
        {{
            "paper_title": "Exact title from above",
            "is_relevant": bool,
            "relevance_explanation": "Brief explanation of why relevant/irrelevant"
        }}
    ],
    "addressed_aspects": [
        {{
            "sub_question": "What specific aspect was addressed?",
            "evidence": "Which papers address this and how?"
        }}
    ],
    "remaining_challenges": [
        {{
            "challenge_id": "c1",
            "challenge_question": "Clear question about what remains unsolved for research question ({question})",
            "why_unaddressed": "Why current work doesn't solve this",
            "importance": "Why solving this matters for the main research problem and research question."
        }}
    ],
    "overall_assessment": "Is the research question \"substantially addressed\", \"partially addressed\", or \"largely unaddressed\"?"
}}

# GUIDELINES

- Be critical but fair in assessing relevance
- "Addressed" means substantial progress exists, not perfection
- Focus on conceptual gaps, not just performance improvements
- If all aspects are addressed, remaining_challenges can be empty
- Base judgments only on provided evidence

Now analyze the papers for this research question.
"""
    
    return prompt


class PaperRelevance(BaseModel):
    paper_title: str
    is_relevant: bool
    relevance_explanation: str


class AddressedAspect(BaseModel):
    sub_question: str
    evidence: str


class RemainingChallenge(BaseModel):
    challenge_id: str
    challenge_question: str
    why_unaddressed: str
    importance: str


class TargetDomainAnalysis(BaseModel):
    question: str
    paper_relevance: List[PaperRelevance]
    addressed_aspects: List[AddressedAspect]
    remaining_challenges: List[RemainingChallenge]
    overall_assessment: str


# =============================================================================
# PROMPT 3: Cross-Domain Query Generation
# =============================================================================

def create_cross_domain_query_prompt(
    problem_statement: str,
    question: str,
    question_rationale: str,
    target_domain: str,
    target_domain_assessment: Optional[str] = None
) -> str:
    """
    Creates a prompt for identifying cross-domain search queries.
    
    Args:
        question: The research question or challenge
        question_rationale: Why this question matters
        target_domain: The original research domain
        target_domain_assessment: Summary of what was/wasn't addressed in target domain
        is_new_challenge: Whether this is a newly identified challenge
        
    Returns:
        Formatted prompt string for LLM
    """
    
    context_section = ""
    if target_domain_assessment:
        context_section = f"""
# WHY CURRENT {target_domain.upper()} RESEARCH IS INSUFFICIENT:
{target_domain_assessment}
"""
    
    prompt = f"""You are an expert at identifying cross-disciplinary research connections. Your task is to identify external domains and search queries that could address a research question under the given research problem.

# RESEARCH PROBLEM
{problem_statement}

# RESEARCH QUESTION
{question}

**Rationale**: {question_rationale}

# ORIGINAL DOMAIN
{target_domain}
{context_section}

# VALID SEMANTIC SCHOLAR DOMAINS
Computer Science, Medicine, Chemistry, Biology, Materials Science, Physics, Geology, Psychology, Art, History, Geography, Sociology, Business, Political Science, Economics, Philosophy, Mathematics, Engineering, Environmental Science, Agricultural and Food Sciences, Education, Law, Linguistics

# YOUR TASK

Identify 1-3 external domains (from the list above different from the original domain) that may have addressed this question through analogous problems, mechanisms, or principles. For each domain, provide 2-4 specific search queries.

# QUERY DESIGN PRINCIPLES

- **Domain-appropriate terminology**: Use terms natural to that field (e.g., "foraging theory" for Biology, "intertemporal choice" for Economics)
- **Mechanistic focus**: Target underlying principles, not surface similarities
- **Concise**: Maximum 5 words per query
- **Varied**: Cover different angles or approaches within the domain

# OUTPUT FORMAT

Return a JSON object:

{{
    "question": "{question}",
    "cross_domain_searches": [
        {{
            "domain": "Valid domain from the list",
            "domain_rationale": "Why this domain likely has relevant insights",
            "queries": [
                "concise query max five",
                "another specific query",
                "third targeted query"
            ]
        }}
    ]
}}

# EXAMPLE

Question: "How can a system attribute delayed outcomes to earlier actions?"
Good output:
{{
    "cross_domain_searches": [
        {{
            "domain": "Psychology",
            "domain_rationale": "Animal learning research extensively studied delayed reinforcement and temporal credit assignment",
            "queries": ["delayed reinforcement learning", "trace conditioning mechanisms", "temporal contiguity association"]
        }},
        {{
            "domain": "Neuroscience", 
            "domain_rationale": "Dopamine systems solve credit assignment for rewards occurring after actions",
            "queries": ["dopamine prediction error", "reward timing neurons", "temporal difference brain"]
        }}
    ]
}}

Bad example:
- Domain not in list: "Cognitive Science" (not a valid Semantic Scholar domain)
- Too generic of a query: ["learning", "psychology of rewards"]
- Too long of a query: ["how do animals learn from delayed rewards"]

Now identify cross-domain searches for this question.
"""
    
    return prompt


class CrossDomainSearch(BaseModel):
    domain: str
    domain_rationale: str
    queries: List[str] = Field(min_items=2, max_items=4)


class CrossDomainQueries(BaseModel):
    question: str
    cross_domain_searches: List[CrossDomainSearch] = Field(min_items=1, max_items=3)


# =============================================================================
# PROMPT 4: Cross-Domain Relevance and Takeaways
# =============================================================================

def create_cross_domain_analysis_prompt(
    problem_statement: str,
    question: str,
    question_rationale: str,
    source_domain: str,
    papers_with_snippets: dict,
    target_domain: str
) -> str:
    """
    Creates a prompt for analyzing cross-domain papers and extracting takeaways.
    
    Args:
        problem_statement: The research problem
        question: The research question being analyzed
        question_rationale: Why this question matters
        source_domain: The external domain being analyzed
        papers_with_snippets: Dict mapping paper titles to lists of snippets
        target_domain: The original target domain
        
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
    
    prompt = f"""You are an expert at extracting cross-disciplinary insights. Analyze papers from an external domain to assess their relevance to a research question and extract high-level takeaways.

# RESEARCH PROBLEM
{problem_statement}

# RESEARCH QUESTION
{question}

**Rationale**: {question_rationale}

# SOURCE DOMAIN
{source_domain}

# TARGET DOMAIN (for context)
{target_domain}

# RETRIEVED PAPERS AND SNIPPETS FROM {source_domain.upper()}
{papers_text}

# YOUR TASK

1. **Assess Domain Relevance**: Is this domain ({source_domain}) relevant to the research question? Consider:
   - Do the papers actually address analogous problems or mechanisms at a deeper level?
   - Are there genuine conceptual connections, or only superficial similarities?
   - Would insights from this domain likely transfer to the target domain?

2. **Assess Domain Adequacy**: Based on the papers' relevance, do these papers provide sufficient insights? Consider:
   - Do the papers offer substantive solutions, frameworks, or principles?
   - Is there enough depth to extract actionable takeaways?
   - Are there clear mechanisms or approaches that could inform the target domain?

3. **Extract High-Level Takeaways**: If the domain is relevant and adequate, identify 2-5 key insights that are:
   - **Domain-agnostic**: Formulated as general principles, not {source_domain}-specific details
   - **Mechanistic**: Focused on "how" and "why", not just "what"
   - **Actionable**: Could potentially inform approaches in {target_domain}
   - **Evidence-based**: Clearly supported by the provided papers
   - **Be Conservative**: Only include takeaways you are confident are well-supported

# OUTPUT FORMAT

Return a JSON object:

{{
    "question": "{question}",
    "source_domain": "{source_domain}",
    "paper_relevance": [
        {{
            "paper_title": "Exact title from above",
            "is_relevant": bool,
            "relevance_explanation": "Brief explanation of why relevant/irrelevant"
        }}
    ],
    "is_relevant": bool,
    "relevance_explanation": "Brief explanation of why this domain is/isn't relevant to the question",
    "is_adequate": bool,
    "adequacy_explanation": "Explain whether papers provide sufficient depth and insights",
    "high_level_takeaways": [
        {{
            "takeaway_id": "t1",
            "principle": "Clear, domain-agnostic statement of the key insight or mechanism",
            "explanation": "How this principle works and why it matters",
            "supporting_evidence": "Which papers demonstrate this and how"
        }}
    ]
}}

# GUIDELINES

- Be honest about relevance - superficial keyword matches don't mean true relevance
- "Relevant" means the domain addresses analogous problems/mechanisms, not just similar terms
- "Adequate" means sufficient depth for extracting actionable insights
- If not relevant, set is_relevant=false and leave high_level_takeaways as empty list
- If relevant but not adequate (e.g., too shallow), set is_adequate=false
- Takeaways should be abstracted to their core principles, stripped of domain-specific jargon
- Focus on mechanisms and approaches that could generalize, not specific implementations

# EXAMPLE

Question: "How can a system attribute delayed outcomes to earlier actions?"
Source Domain: "Psychology"

Good takeaway:
{{
    "takeaway_id": "t1",
    "principle": "Delayed outcomes can be attributed to earlier actions by preserving a fading influence of past events until feedback arrives",
    "explanation": "When outcomes do not immediately follow actions, systems can still assign credit by retaining a temporary, gradually weakening influence of prior actions or states. When feedback eventually occurs, earlier elements that remain influential receive proportionate credit, enabling learning across time gaps.",
    "supporting_evidence": "Studies on delayed learning and conditioning show that organisms can learn associations even when outcomes occur later, as long as internal representations of earlier events persist long enough to overlap with feedback."
}}

Bad takeaway (too domain-specific):
"Dopamine neurons in the ventral tegmental area fire in response to reward prediction errors"
→ Should be: "Prediction error signals can drive learning by indicating discrepancies between expected and actual outcomes"

Now analyze the {source_domain} papers for this research question.
"""
    
    return prompt


class HighLevelTakeaway(BaseModel):
    takeaway_id: str
    principle: str
    explanation: str
    supporting_evidence: str


class CrossDomainAnalysis(BaseModel):
    question: str
    source_domain: str
    paper_relevance: List[PaperRelevance]
    is_relevant: bool
    relevance_explanation: str
    is_adequate: bool
    adequacy_explanation: str
    high_level_takeaways: List[HighLevelTakeaway]


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
# Schema exports for easy access
# =============================================================================

initial_decomposition_schema = InitialDecomposition.model_json_schema()
target_domain_analysis_schema = TargetDomainAnalysis.model_json_schema()
cross_domain_queries_schema = CrossDomainQueries.model_json_schema()
cross_domain_analysis_schema = CrossDomainAnalysis.model_json_schema()
target_domain_framing_schema = TargetDomainFraming.model_json_schema()