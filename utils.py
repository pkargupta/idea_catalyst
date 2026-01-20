
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