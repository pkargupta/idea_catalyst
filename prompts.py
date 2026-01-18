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
    
    prompt = f"""You are an expert research strategist. Your task is to decompose a research problem into fundamental questions and identify target-domain search queries.

# RESEARCH PROBLEM
{problem_statement}

# TARGET DOMAIN
{target_domain}

# YOUR TASK
Break down this problem into 3-5 fundamental research questions that represent core conceptual bottlenecks. For each question, provide 3-5 concise search queries (max 5 words each) to find relevant work in the target domain.

# DECOMPOSITION PRINCIPLES

Each research question must be:
- **Atomic**: Addresses one distinct conceptual challenge
- **Critical**: Necessary for making progress on the problem
- **Cross-disciplinary**: Formulated to allow mapping to other fields (avoid domain-specific jargon)
- **Mechanistic**: Focuses on "how" or "why", not just "what"

# SEARCH QUERY REQUIREMENTS

Each query must be:
- **Concise**: Maximum 5 words
- **Specific**: Uses precise terminology likely in paper titles/abstracts
- **Diverse**: Cover different aspects or approaches to the question

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

3. **Identify Remaining Challenges**: What meaningful challenges remain unaddressed? These should be:
   - **Non-iterative**: Not just "do X better" but fundamentally different problems
   - **Major**: Represent significant conceptual or practical bottlenecks
   - **Atomic**: Each challenge is a distinct issue
   - **Formulated as a question**: If the original research question was not addressed at all, just repeat it here. Otherwise, word the remaining challenges as clear questions.
   If no challenges remain, output an empty list.

4. **Overall Assessment**: Based on the above, is the original research question "substantially addressed", "partially addressed", or "largely unaddressed" in this domain?

# OUTPUT FORMAT

Return a JSON object:

{{
    "question": "{question}",
    "paper_relevance": [
        {{
            "paper_title": "Exact title from above",
            "is_relevant": true,
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
            "challenge_question": "Clear question about what remains unsolved",
            "why_unaddressed": "Why current work doesn't solve this",
            "importance": "Why solving this matters for the main question"
        }}
    ],
    "overall_assessment": "Is the original question \"substantially addressed\", \"partially addressed\", or \"largely unaddressed\"?"
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

Identify 2-5 external domains (from the list above different from the original domain) that may have addressed this question through analogous problems, mechanisms, or principles. For each domain, provide 2-4 specific search queries.

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
    cross_domain_searches: List[CrossDomainSearch] = Field(min_items=2, max_items=5)


# =============================================================================
# Schema exports for easy access
# =============================================================================

initial_decomposition_schema = InitialDecomposition.model_json_schema()
target_domain_analysis_schema = TargetDomainAnalysis.model_json_schema()
cross_domain_queries_schema = CrossDomainQueries.model_json_schema()