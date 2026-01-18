from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class Domain:
    def __init__(self, domain_name, question2queries=None):
        self.domain_name = domain_name
        self.question2queries: Optional[Dict[Question, List[str]]] = question2queries if question2queries else None

        self.question2papers: Optional[Dict[Question, Dict[str, List[str]]]] = {}  # Question : {paper_title: [snippets: strings]}
        self.question2abstract_takeaways: Optional[Dict[Question, str]] = {}
        self.question2sufficiency: Optional[Dict[Question, bool]] = {}  # populated later

    def fetch_question_queries(self, question):
        return self.question2queries[question]
    
    def add_question_queries(self, question, queries):
        self.question2queries[question] = queries
    
    def add_abstract_takeaway(self, question, takeaway):
        self.question2abstract_takeaways[question] = takeaway
    
    def mark_sufficiency(self, question, is_sufficient):
        self.question2sufficiency[question] = is_sufficient
    
    def add_question_papers(self, question, papers):
        self.question2papers[question] = papers
    
    def fetch_question_papers(self, question):
        return self.question2papers[question]
    
    def format_question_papers(self, question):
        papers = self.fetch_question_papers(question)
        formatted = []
        for paper_title, snippets in papers.items():
            formatted.append(f"\nPaper: {paper_title}")
            for snippet in snippets:
                formatted.append(f"  - {snippet}")
        return "\n".join(formatted)

    
    def __str__(self):
        return f"Domain(domain_name={self.domain_name})"

@dataclass
class Question:
    def __init__(self, id, question, rationale, current_gaps):
        self.id = id
        self.question = question
        self.rationale = rationale
        self.current_gaps = current_gaps

        # Populate later
        self.external_domains: Optional[Dict[str, Domain]] = None # domain name: Domain obj
    
    def add_external_domain(self, name, domain_obj):
        self.external_domains[name] = domain_obj
    
    def __str__(self):
        return f"Question(question={self.question})"

@dataclass
class ResearchProblem:
    def __init__(self, decomposition_json: dict):
        self.problem_statement = decomposition_json["research_goal"]
        self.core_challenge = decomposition_json["core_challenge_summary"]

        self.target_domain = Domain(domain_name=decomposition_json["domain"])
        self.domains = {self.target_domain.domain_name: self.target_domain}

        self.sub_questions = []
        for sub_question_data in decomposition_json["sub_questions"]:
            id = len(self.sub_questions)
            question = sub_question_data["question"]
            rationale = sub_question_data["rationale"]
            current_gaps = sub_question_data["current_gaps"]

            sub_question = Question(id=id, question=question, rationale=rationale, current_gaps=current_gaps)

            # Same domain
            same_domain_queries = sub_question_data["same_domain_search_queries"]
            self.target_domain.add_question_queries(sub_question, same_domain_queries)

            # Cross domains
            for domain in sub_question_data["cross_domain_search_queries"]:
                domain_name = domain["domain"]
                queries = domain["queries"]

                if domain_name not in self.domains:
                    self.domains[domain_name] = Domain(domain_name=domain_name)
                self.domains[domain_name].add_question_queries(sub_question, queries)

            self.sub_questions.append(sub_question)