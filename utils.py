from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import json_repair
from typing import List, Dict

def batch_llm_inference(llm, messages_list: List[List[Dict]], schema: dict, temperature: float = 0.7, max_tokens: int = 2048) -> List[dict]:
    """
    Perform batch inference with structured output.
    
    Args:
        llm: vLLM model
        messages_list: List of message sequences (each is a list of message dicts)
        schema: JSON schema for structured output
        temperature: Sampling temperature
        
    Returns:
        List of parsed JSON responses
    """
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        structured_outputs=StructuredOutputsParams(json=schema),
    )
    
    responses = llm.chat(messages_list, sampling_params, chat_template_kwargs={"enable_thinking": False})
    
    # Parse all responses
    parsed_responses = []
    for response in responses:
        try:
            parsed = json_repair.loads(response.outputs[0].text)
            parsed_responses.append(parsed)
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response text: {response.outputs[0].text}")
            parsed_responses.append(None)
    
    return parsed_responses


def retrieve_papers_for_question(question: Question, domain: Domain, max_papers: int = 10) -> Dict[str, List[str]]:
    """
    Retrieve papers and snippets for a question in a specific domain.
    
    Args:
        question: The research question
        domain: The domain to search in
        max_papers: Maximum number of unique papers to retrieve
        
    Returns:
        Dictionary mapping paper titles to lists of snippets
    """
    queries = domain.fetch_question_queries(question)
    papers = {}
    
    for query in queries:
        if len(papers) >= max_papers:
            break
        
        try:
            response = search_semantic_scholar(query, domain.domain_name)
            snippets = collect_snippets(response)
            
            if len(snippets) > 0:
                # Add new papers up to the limit
                for paper_title, snippet_list in snippets.items():
                    if paper_title not in papers:
                        papers[paper_title] = snippet_list
                    else:
                        papers[paper_title].extend(snippet_list)
                        papers[paper_title] = list(set(papers[paper_title]))  # Ensure uniqueness
                    
                    if len(papers) >= max_papers:
                        break
        except Exception as e:
            print(f"Error searching for query '{query}': {e}")
    
    return papers

def prepare_output(research_problem):
    # Prepare output structure
    output = {
        "problem_statement": research_problem.problem_statement,
        "target_domain": research_problem.target_domain.domain_name,
        "fine_grained_domain": research_problem.fine_grained_domain,
        "core_challenge": research_problem.core_challenge,
        "research_questions": []
    }

    condensed_output = {
        "problem_statement": research_problem.problem_statement,
        "target_domain": research_problem.target_domain.domain_name,
        "fine_grained_domain": research_problem.fine_grained_domain,
        "core_challenge": research_problem.core_challenge,
        "research_questions": []
    }
    
    for question in research_problem.research_questions:

        q_data = {
            "id": question.id,
            "question": question.question,
            "rationale": question.rationale,
            "target_domain_analysis": {"paper_relevance": question.target_domain_analysis["paper_relevance"],
                                       "addressed_aspects": question.target_domain_analysis["addressed_aspects"],
                                       "overall_assessment": question.target_domain_analysis["overall_assessment"]},
            "is_addressed_in_target": question.is_addressed_in_target,
            "remaining_challenges": [
                {
                    "id": c.id,
                    "question": c.question,
                    "rationale": c.rationale,
                    "cross_domain_queries": c.cross_domain_queries["cross_domain_searches"],
                    "external_domains_searched": list(c.external_domains.keys())
                }
                for c in question.remaining_challenges
            ]
        }

        condensed_q_data = {"question": question.question,
                            "overall_assessment": question.target_domain_analysis["overall_assessment"],
                            "remaining_challenges": [
                                {"question": c.question,
                                 "rationale": c.rationale,
                                 "cross_domain_queries": c.cross_domain_queries["cross_domain_searches"]
                }
                for c in question.remaining_challenges
            ]
        }


        if not question.is_addressed_in_target:
            q_data["cross_domain_queries"] = question.cross_domain_queries["cross_domain_searches"] if not question.is_addressed_in_target else None
            q_data["external_domains_searched"] = list(question.external_domains.keys()) if question.external_domains else []
            
            condensed_q_data["cross_domain_queries"] = question.cross_domain_queries["cross_domain_searches"] if not question.is_addressed_in_target else None

        output["research_questions"].append(q_data)
        condensed_output["research_questions"].append(condensed_q_data)
    
    return output, condensed_output