"""
Baseline Methods for Interdisciplinary Research Analysis

This module implements simplified baseline approaches to compare against
the full pipeline in inspiration_pred.py.
"""

import os
# Environment configuration
os.environ["HF_HOME"] = "/shared/data3/pk36/.cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import argparse
from collections import defaultdict
from tqdm import tqdm
from vllm import LLM
from pydantic import BaseModel, Field
from typing import List, Dict

# Assuming utils is in parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import batch_llm_inference, retrieve_papers_for_queries, convert_domain
from classes import Domain


# =============================================================================
# BASELINE 1: Direct Domain Identification and Ideation
# =============================================================================

def create_direct_source_queries_prompt(
    problem_statement: str,
    target_domain: str
) -> str:
    """
    Creates a prompt that asks the model to directly identify relevant domains
    and generate queries, then ideate without explicit cross-domain exploration.
    
    Args:
        problem_statement: The research problem
        target_domain: The target/fine-grained domain
        
    Returns:
        Formatted prompt string
    """
    
    prompt = f"""You are an expert research strategist. Your task is to determine other potential domains that addresses the following research problem and suggest search queries for retrieving papers from those domains.

# RESEARCH PROBLEM
{problem_statement}

# TARGET DOMAIN
{target_domain}

# YOUR TASK

Identify relevant domains and generate search queries for retrieving papers from those domains that addresses this problem. Your approach should:

1. **Identify Relevant Domains**: Consider which research domains might contain relevant insights or methods for this problem. You may explore connections to other fields if you believe they would be helpful.

2. **Generate Search Queries**: For each domain you identify, provide 3-5 search queries (max 5 words each) that would help find relevant research.

# OUTPUT FORMAT

Return a JSON object:

{{
    "identified_domains": [
        {{
            "domain": "Domain name",
            "rationale": "Why this domain is relevant",
            "queries": ["query 1", "query 2", "query 3"]
        }},
        ...
    ]
}}

# GUIDELINES

- You may identify and explore connections to any research domains you believe are relevant
- Be specific about how different domains' insights integrate
- Focus on concrete, actionable research directions
- Explain the mechanisms behind your proposed approach

Now develop your research idea.
"""
    
    return prompt

class IdentifiedDomain(BaseModel):
    domain: str
    rationale: str
    queries: List[str] = Field(min_items=3, max_items=5)

class SourceDomains(BaseModel):
    identified_domains: List[IdentifiedDomain]= Field(min_items=1, max_items=3)


source_domain_schema = SourceDomains.model_json_schema()

def convert_domain_to_semantic_scholar(domain: str) -> str:
    prompt = f"""Convert the following fine-grained research domain into a valid Semantic Scholar coarse domain.
Fine-grained domain: {domain}

Valid Semantic Scholar coarse domains include:
Computer Science, Medicine, Chemistry, Biology, Materials Science, Physics, Geology, Psychology, Art, History, Geography, Sociology, Business, Political Science, Economics, Philosophy, Mathematics, Engineering, Environmental Science, Agricultural and Food Sciences, Education, Law, Linguistics

Output the corresponding coarse domain in JSON format:
{{
    "coarse_domain": "Corresponding coarse domain"
}}
"""
    
    return prompt

class CoarseDomain(BaseModel):
    coarse_domain: str

coarse_domain_schema = CoarseDomain.model_json_schema()

def create_direct_ideation_prompt(
    problem_statement: str,
    target_domain: str,
    source_domain: str=None,
    source_papers: Dict[str, List[str]]=None
) -> str:
    """
    Creates a prompt for direct ideation without explicit cross-domain retrieval.
    
    Args:
        problem_statement: The research problem
        target_domain: The target/fine-grained domain
        source_domains: Optional list of identified source domains
        source_papers: Optional dict mapping domain name to papers
    Returns:
        Formatted prompt string
    """
    
    prompt = f"""You are an expert at interdisciplinary research synthesis. Your task is to develop a novel research idea by integrating insights from multiple research domains.

# RESEARCH PROBLEM
{problem_statement}

# TARGET DOMAIN
{target_domain}

# SOURCE DOMAIN
{source_domain}

# SOURCE DOMAIN PAPERS
"""
    if source_papers:
        for i, (title, snippets) in enumerate(source_papers.items(), 1):
            prompt += f"\n## Paper {i}: {title}\n"
            for j, snippet in enumerate(snippets[:3], 1):  # Limit to 3 snippets per paper
                prompt += f"   Snippet {j}: {snippet}\n"
    else:
        prompt += "No papers retrieved.\n"
    
    prompt += f"""
# YOUR TASK
**Develop Integrated Idea**: Based on your knowledge of {target_domain} and the papers you've retrieved from other domains, propose a concrete research idea.

Your idea should:
1. Integrate insights from the {target_domain} with relevant insights from {source_domain}
2. Explain how the two different domains' concepts/methods combine
3. Propose a specific, actionable research approach

{{
    "idea_fragment": {{
        "title": "Brief, descriptive title for the integrated idea (max 15 words)",
        "core_insight": "2-3 sentence summary of the key integrated insight",
        "integration_mechanism": {{
            "target_domain_elements": [
                "Specific concept/method from {target_domain}",
                "Another specific concept/method from {target_domain}"
            ],
            "source_domain_takeaways": [
                {{
                    "source_domain_formulation": "Description of concept/method from identified domain",
                    "mechanism_explanation": "How this approach works and why it addresses the challenge",
                    "selection_rationale": "Why this was selected for integration"
                }}
            ],
            "synthesis_approach": "1-2 sentence explanation of how these elements combine"
        }},
        "concrete_realization": {{
            "proposed_approach": "Specific technical approach, algorithm, or framework (3-4 sentences)",
            "key_innovations": [
                "Novel aspect 1",
                "Novel aspect 2"
            ]
        }}
    }}
}}
# GUIDELINES
- Ground your idea in the specific papers provided
- Be concrete about how insights from different domains integrate
- Focus on actionable research directions
- Highlight what's novel about the integration

Now develop your research idea.
"""
    
    return prompt

class SourceDomainTakeaway(BaseModel):
    source_domain_formulation: str
    mechanism_explanation: str
    selection_rationale: str


class IntegrationMechanism(BaseModel):
    target_domain_elements: List[str] = Field(min_items=1)
    source_domain_takeaways: List[SourceDomainTakeaway] = Field(min_items=1)
    synthesis_approach: str


class ConcreteRealization(BaseModel):
    proposed_approach: str
    key_innovations: List[str] = Field(min_items=1, max_items=5)


class IdeaFragment(BaseModel):
    title: str
    core_insight: str
    integration_mechanism: IntegrationMechanism
    concrete_realization: ConcreteRealization


class DirectIdeation(BaseModel):
    idea_fragment: IdeaFragment


direct_ideation_schema = DirectIdeation.model_json_schema()


def baseline_direct_ideation(args, llm, problem_id, problem_info):
    """
    Baseline 1: No explicit target domain retrieval & without explicitly requesting cross-domains (can or cannot request retrieval from different coarse-domains).
    Model identifies domains and generates idea based on its own knowledge.
    
    Args:
        args: Command-line arguments
        llm: Language model instance
        problem_id: Unique identifier for the problem
        problem_info: Dictionary containing problem details
        
    Returns:
        Dict with generated idea
    """
    problem_statement = problem_info["context"]
    target_domain = convert_domain(problem_info["target_domain"])
    
    print(f"\nBaseline 1: Direct Ideation")
    print(f"Problem: {problem_id}")
    print(f"Target Domain: {target_domain}")
    
    # Create prompt for domains and queries
    prompt = create_direct_source_queries_prompt(problem_statement, target_domain)
    messages = [{"role": "user", "content": prompt}]
    
    # Generate idea
    identified_domain_outputs = batch_llm_inference(
        llm,
        [messages],
        source_domain_schema,
        temperature=args.temp,
        max_tokens=4096
    )
    
    identified_domains = identified_domain_outputs[0]["identified_domains"]
    
    if (identified_domains is None) or (len(identified_domains) == 0):
        print(f"  Failed to generate potential domains for {problem_id}")
        return None
    
    # Convert identified domains to Semantic Scholar coarse domains
    all_coarse_prompts = []
    for domain_info in identified_domains:
        domain_name = domain_info['domain']
        coarse_prompt = convert_domain_to_semantic_scholar(domain_name)
        coarse_messages = [{"role": "user", "content": coarse_prompt}]
        all_coarse_prompts.append(coarse_messages)
        
    coarse_outputs = batch_llm_inference(
        llm,
        all_coarse_prompts,
        coarse_domain_schema,
        temperature=0.0,
        max_tokens=256
    )

    for domain_info, coarse_output in zip(identified_domains, coarse_outputs):
        domain_info['coarse_domain'] = coarse_output['coarse_domain']

    # Retrieve papers from identified domains
    source_ideation_prompts = []
    domain2papers = {}
    for domain_info in identified_domains:
        domain_name = domain_info['domain']
        coarse_domain = domain_info['coarse_domain']
        queries = domain_info['queries']
        
        papers = retrieve_papers_for_queries(
            queries=queries,
            domain_name=coarse_domain,
            max_papers=args.max_papers_per_query,
            year=problem_info["publication_year"],
            baseline=True
        )

        domain2papers[domain_name] = papers
        
        prompt = create_direct_ideation_prompt(problem_statement=problem_statement, target_domain=target_domain, source_domain=domain_name, source_papers=papers)
        messages = [{"role": "user", "content": prompt}]
        source_ideation_prompts.append(messages)
    
    # Generate ideas
    ideation_outputs = batch_llm_inference(
        llm,
        source_ideation_prompts,
        direct_ideation_schema,
        temperature=args.temp,
        max_tokens=4096
    )

    final_outputs = []
    for domain_info, ideation_output in zip(identified_domains, ideation_outputs):
        final_outputs.append({
            "source_domain": domain_info["coarse_domain"],
            "fine_grained_source_domain": domain_info["domain"],
            "queries": domain_info["queries"],
            "idea_fragment": ideation_output['idea_fragment']})
    
    return {
        "research_problem": problem_statement,
        "target_domain": target_domain,
        "papers": domain2papers,
        "source_ground_truth": {
            "gt_domain": convert_domain(problem_info["source_domain"]),
            "gt_domain_insight": problem_info["source_text"],
            "gt_abstract": problem_info["abstract"]
        },
        "idea_rankings": final_outputs
    }


# =============================================================================
# BASELINE 2: Target-Retrieve-Source-Retrieve-Ideate
# =============================================================================

# =============================================================================
# PROMPT 1: Initial Goal Decomposition
# =============================================================================

def create_target_queries(problem_statement: str, fine_grained_domain: str) -> str:
    """
    Creates a prompt for decomposing a research problem into a coarse-grained target domain and queries.
    """

    prompt = f"""You are an expert research strategist. Your task is to:
(1) identify the most relevant **coarse-grained domain** (the target domain) that encompasses the input fine-grained domain for the given research problem,
(2) identify 3-5 word queries to search for papers in the coarse-grained/fine-grained domains.

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

# STEP 2: SEARCH QUERIES (SUBFIELD-ALIGNED)
Provide:
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
  "target_domain_queries": [
    "max five words",
    "subfield canonical phrase",
    "title/abstract likely terms"
    ]
}}

# EXAMPLE (BRIEF)

If fine_grained_domain = "Probabilistic graphical models":
- target_domain_queries might include "identifiability", "latent variables", or "structure learning" (<=5 words)

Ensure every query is clearly aligned with the chosen fine_grained_domain. Now output your JSON object following the above instructions.
"""
    return prompt

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

    target_domain_queries: List[str] = Field(
        min_items=3,
        max_items=5,
        description=(
            "Search queries (<=5 words) for within-target-domain search. Must strongly reflect "
            "fine_grained_domain vocabulary and canonical phrases likely in titles/abstracts."
        ),
    )

target_domain_queries_schema = InitialDecomposition.model_json_schema()


def create_source_domain_suggestion_prompt(
    problem_statement: str,
    target_domain: str,
    target_domain_subfield: str,
    target_domain_papers: Dict[str, List[str]]
) -> str:
    """
    Creates a prompt to suggest external domains after seeing target domain papers.
    
    Args:
        problem_statement: The research problem
        target_domain: The target/fine-grained domain
        target_domain_papers: Papers retrieved from target domain
        
    Returns:
        Formatted prompt string
    """
    
    prompt = f"""You are an expert at identifying interdisciplinary research connections. Your task is to suggest external research domains that might provide valuable insights for addressing a research problem.

# RESEARCH PROBLEM
{problem_statement}

# TARGET DOMAIN
{target_domain}

# TARGET DOMAIN SUBFIELD
{target_domain_subfield}

# PAPERS FROM {target_domain.upper()}
"""
    if target_domain_papers:
        for i, (title, snippets) in enumerate(target_domain_papers.items(), 1):
            prompt += f"\n## Paper {i}: {title}\n"
            for j, snippet in enumerate(snippets[:3], 1):  # Limit to 3 snippets per paper
                prompt += f"   Snippet {j}: {snippet}\n"
    else:
        prompt += "No papers retrieved.\n"
    
    prompt += f"""

# YOUR TASK

Based on the research problem and the current state of research in {target_domain} and {target_domain_subfield} (as shown by the papers above), suggest 1-3 external research domains that might provide relevant insights, methods, or analogies.

For each suggested domain:
1. Explain why this domain is relevant
2. Provide 3-5 search queries (max 5 words each) to find relevant work in that domain

# VALID DOMAINS
Computer Science, Medicine, Chemistry, Biology, Materials Science, Physics, Geology, Psychology, Art, History, Geography, Sociology, Business, Political Science, Economics, Philosophy, Mathematics, Engineering, Environmental Science, Agricultural and Food Sciences, Education, Law, Linguistics

# OUTPUT FORMAT

Return a JSON object:

{{
    "suggested_domains": [
        {{
            "domain": "Domain name from the list above",
            "rationale": "Why this domain likely has relevant insights",
            "queries": [
                "search query 1",
                "search query 2",
                "search query 3"
            ]
        }}
    ]
}}

Now suggest external domains.
"""
    
    return prompt

class SuggestedDomain(BaseModel):
    domain: str
    rationale: str
    queries: List[str] = Field(min_items=3, max_items=5)


class DomainSuggestions(BaseModel):
    suggested_domains: List[SuggestedDomain] = Field(min_items=1, max_items=3)

domain_suggestions_schema = DomainSuggestions.model_json_schema()


def create_target_source_ideation_prompt(
    problem_statement: str,
    target_domain: str,
    target_domain_subfield: str,
    source_domain: str,
    target_domain_papers: Dict[str, List[str]],
    source_domain_papers: Dict[str, Dict[str, List[str]]]
) -> str:
    """
    Creates a prompt for ideation after retrieving from both target and external domains.
    
    Args:
        problem_statement: The research problem
        source_domain: The source/fine-grained domain
        target_domain_papers: Papers from target domain
        external_domains_papers: Dict mapping domain name to papers
        
    Returns:
        Formatted prompt string
    """
    
    prompt = f"""You are an expert at interdisciplinary research synthesis. Your task is to develop a novel research idea by integrating insights from multiple research domains.

# RESEARCH PROBLEM
{problem_statement}

# TARGET DOMAIN
{target_domain}

## TARGET DOMAIN SUBFIELD
{target_domain_subfield}

# PAPERS FROM {target_domain.upper()}
"""
    if target_domain_papers:
        for i, (title, snippets) in enumerate(target_domain_papers.items(), 1):
            prompt += f"\n## Paper {i}: {title}\n"
            for j, snippet in enumerate(snippets[:3], 1):  # Limit to 3 snippets per paper
                prompt += f"   Snippet {j}: {snippet}\n"
    else:
        prompt += "No papers retrieved.\n"
    
    prompt += f"""

# SOURCE DOMAIN
{source_domain}

# PAPERS FROM {source_domain.upper()}
"""
    if source_domain_papers:
        for i, (title, snippets) in enumerate(source_domain_papers.items(), 1):
            prompt += f"\n## Paper {i}: {title}\n"
            for j, snippet in enumerate(snippets[:3], 1):  # Limit to 3 snippets per paper
                prompt += f"   Snippet {j}: {snippet}\n"
    else:
        prompt += "No papers retrieved.\n"
    
    prompt += f"""

# YOUR TASK

Based on the research problem and the papers from both the target and source domains, develop a concrete, novel research idea that addresses the problem.

Your idea should:
1. Integrate insights from the {target_domain} with relevant insights from {source_domain}
2. Explain how the two different domains' concepts/methods combine
3. Propose a specific, actionable research approach

{{
    "idea_fragment": {{
        "title": "Brief, descriptive title for the integrated idea (max 15 words)",
        "core_insight": "2-3 sentence summary of the key integrated insight",
        "integration_mechanism": {{
            "target_domain_elements": [
                "Specific concept/method from {target_domain}",
                "Another specific concept/method from {target_domain}"
            ],
            "source_domain_takeaways": [
                {{
                    "source_domain_formulation": "Description of concept/method from identified domain",
                    "mechanism_explanation": "How this approach works and why it addresses the challenge",
                    "selection_rationale": "Why this was selected for integration"
                }}
            ],
            "synthesis_approach": "1-2 sentence explanation of how these elements combine"
        }},
        "concrete_realization": {{
            "proposed_approach": "Specific technical approach, algorithm, or framework (3-4 sentences)",
            "key_innovations": [
                "Novel aspect 1",
                "Novel aspect 2"
            ]
        }}
    }}
}}
# GUIDELINES
- Ground your idea in the specific papers provided
- Be concrete about how insights from different domains integrate
- Focus on actionable research directions
- Highlight what's novel about the integration

Now develop your research idea.
"""
    
    return prompt


class RetrieveRetrieveIdeation(BaseModel):
    idea_fragment: IdeaFragment

target_source_ideation_schema = RetrieveRetrieveIdeation.model_json_schema()

def baseline_dual_retrieval(args, llm, problem_id, problem_info):
    """
    Baseline 2: Retrieve in target → suggest external domains → retrieve in external → ideate.
    
    Args:
        args: Command-line arguments
        llm: Language model instance
        problem_id: Unique identifier for the problem
        problem_info: Dictionary containing problem details
        
    Returns:
        Dict with generated idea
    """
    problem_statement = problem_info["context"]
    target_domain_subfield = convert_domain(problem_info["target_domain"])
    
    print(f"Problem: {problem_id}")
    print(f"Target Domain Subfield: {target_domain_subfield}")
    
    # Create prompt for target domain and queries
    prompt = create_target_queries(problem_statement, target_domain_subfield)
    messages = [{"role": "user", "content": prompt}]
    
    # Generate target domain/queries
    target_domain_outputs = batch_llm_inference(
        llm,
        [messages],
        target_domain_queries_schema,
        temperature=args.temp,
        max_tokens=1024
    )
    
    target_domain = target_domain_outputs[0]["coarse_grained_domain"]
    target_domain_queries = target_domain_outputs[0]["target_domain_queries"]
    
    if (target_domain is None) or (len(target_domain_queries) == 0):
        print(f"  Failed to generate target domain and queries for {problem_id}")
        return None
    
    # Retrieve papers from target domain
    target_domain_papers = retrieve_papers_for_queries(
            queries=target_domain_queries,
            domain_name=target_domain,
            max_papers=args.max_papers_per_query,
            year=problem_info["publication_year"],
            baseline=True
        )
    
    # Identify potential source domains and relevant queries
    suggest_source_prompt = create_source_domain_suggestion_prompt(problem_statement, target_domain, target_domain_subfield, target_domain_papers)
    source_suggestion_msg = [{"role": "user", "content": suggest_source_prompt}]

    # Generate source domain/queries
    source_domain_outputs = batch_llm_inference(
        llm,
        [source_suggestion_msg],
        domain_suggestions_schema,
        temperature=args.temp,
        max_tokens=2048
    )

    suggested_source_domains = source_domain_outputs[0]["suggested_domains"]
    
    if (suggested_source_domains is None) or (len(suggested_source_domains) == 0):
        print(f"  Failed to generate potential domains for {problem_id}")
        return None

    # Retrieve papers from identified domains
    source_ideation_prompts = []
    domain2papers = {}
    for domain_info in suggested_source_domains:
        domain_name = domain_info['domain']
        queries = domain_info['queries']
        
        papers = retrieve_papers_for_queries(
            queries=queries,
            domain_name=domain_name,
            max_papers=args.max_papers_per_query,
            year=problem_info["publication_year"],
            baseline=True
        )

        domain2papers[domain_name] = papers
        
        prompt = create_target_source_ideation_prompt(problem_statement=problem_statement, target_domain=target_domain, 
                                                      target_domain_subfield=target_domain_subfield, source_domain=domain_name, 
                                                      target_domain_papers=target_domain_papers, source_domain_papers=papers)
        messages = [{"role": "user", "content": prompt}]
        source_ideation_prompts.append(messages)
    
    # Generate ideas
    ideation_outputs = batch_llm_inference(
        llm,
        source_ideation_prompts,
        target_source_ideation_schema,
        temperature=args.temp,
        max_tokens=8192
    )

    final_outputs = []
    for domain_info, ideation_output in zip(suggested_source_domains, ideation_outputs):
        final_outputs.append({
            "source_domain": domain_info["domain"],
            "queries": domain_info["queries"],
            "idea_fragment": ideation_output['idea_fragment']})
    
    # "target_papers": target_domain_papers,
    # "source_papers": domain2papers,
    return {
        "research_problem": problem_statement,
        "target_domain": target_domain,
        "source_ground_truth": {
            "gt_domain": convert_domain(problem_info["source_domain"]),
            "gt_domain_insight": problem_info["source_text"],
            "gt_abstract": problem_info["abstract"]
        },
        "idea_rankings": final_outputs
    }

# =============================================================================
# Main Execution
# =============================================================================

def load_problems(problem_file):
    """Load research problems from file."""
    if not os.path.exists(problem_file):
        print(f"File {problem_file} does not exist!")
        return {}
    
    with open(problem_file, "r") as f:
        problems_list = json.load(f)
    
    problems = {
        f'{sample["source_id"]}_{sample["target_id"]}_{sample["source_text"].lower().replace(" ", "_")}': {
            "source_id": sample["target_id"],
            "source_domain": sample["target_domain"],
            "target_id": sample["source_id"],
            "target_domain": sample["source_domain"],
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline methods for interdisciplinary research analysis."
    )
    parser.add_argument(
        "--problem_file",
        type=str,
        default="data/cross-domain-inspiration-relations.json",
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-14B",
        help="LLM model name or path."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/baseline_output",
        help="Path to output directory."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["1", "2", "both"],
        default="2",
        help="Which baseline to run: 1 (direct), 2 (retrieve-suggest-retrieve), or both."
    )
    parser.add_argument(
        "--max_papers_per_query",
        type=int,
        default=20,
        help="Maximum papers to retrieve per query."
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Temperature for LLM generation."
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip processing if output file already exists."
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=5,
        help="Frequency of saving outputs."
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
    
    print(f"Loaded {len(problems)} problems")
    
    # Initialize vLLM model
    print("Loading model...")
    llm = LLM(model=args.model_name, tensor_parallel_size=1)
    print("Model loaded.\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    baseline_one_outputs = {}
    baseline_two_outputs = {}
    # Process each problem
    run_baseline_1 = args.baseline in ["1", "both"]
    run_baseline_2 = args.baseline in ["2", "both"]

    save_freq = args.save_freq  # Save outputs every 20 problems
    curr_problem = 0

    for problem_id, problem_info in tqdm(problems.items(), desc="Processing problems", total=len(problems)):
        # Baseline 1: Direct Ideation
        if run_baseline_1:
            result_1 = baseline_direct_ideation(args, llm, problem_id, problem_info)
            baseline_one_outputs[problem_id] = result_1

            if len(baseline_one_outputs) and curr_problem % save_freq == 0:
                output_file_1 = os.path.join(
                        args.output_dir,
                        f"baseline1_direct_new.json"
                    )
                
                with open(output_file_1, "w") as f:
                    json.dump(baseline_one_outputs, f, indent=2)
                print(f"  Saved to {output_file_1}")
        
        # Baseline 2: Retrieve-Suggest-Retrieve-Ideate
        if run_baseline_2:
            result_2 = baseline_dual_retrieval(args, llm, problem_id, problem_info)
            baseline_two_outputs[problem_id] = result_2
            
            if len(baseline_two_outputs) and curr_problem % save_freq == 0:
                output_file_2 = os.path.join(
                        args.output_dir,
                        f"baseline2_dual_retrieval.json"
                    )
                
                with open(output_file_2, "w") as f:
                    json.dump(baseline_two_outputs, f, indent=2)
                print(f"  Saved to {output_file_2}")

    
    print("\n" + "="*80)
    print("BASELINE PROCESSING COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()