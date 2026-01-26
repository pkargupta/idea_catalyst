from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class Domain:
    """Represents a research domain with associated queries and papers for questions."""
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.question2queries: Dict['Question', List[str]] = {}
        self.question2papers: Dict['Question', Dict[str, List[str]]] = {}  # Question : {paper_title: [snippets]}
        self.question2analysis: Dict['Question', 'TargetDomainAnalysis'] = {}  # Only for target domain
    
    def add_question_queries(self, question: 'Question', queries: List[str]):
        """Add search queries for a specific question."""
        if question not in self.question2queries:
            self.question2queries[question] = []
        self.question2queries[question].extend(queries)
    
    def add_question_papers(self, question: 'Question', papers: Dict[str, List[str]]):
        """Add retrieved papers and snippets for a question."""
        self.question2papers[question] = papers
    
    def del_question_paper(self, question, papers):
        """Delete specific papers for a question."""
        if question in self.question2papers:
            if type(papers) is str:
                if papers in self.question2papers[question]:
                    del self.question2papers[question][papers]
            else:
                for paper in papers:
                    if paper in self.question2papers[question]:
                        del self.question2papers[question][paper]
        
    def add_question_analysis(self, question: 'Question', analysis):
        """Add target domain analysis for a question (only applicable to target domain)."""
        self.question2analysis[question] = analysis
    
    def fetch_question_queries(self, question: 'Question') -> List[str]:
        """Retrieve queries for a question."""
        return self.question2queries.get(question, [])
    
    def fetch_question_papers(self, question: 'Question') -> Dict[str, List[str]]:
        """Retrieve papers for a question."""
        return self.question2papers.get(question, {})
    
    def format_question_papers(self, question: 'Question') -> str:
        """Format papers and snippets as a readable string."""
        papers = self.fetch_question_papers(question)
        if not papers:
            return "No papers retrieved."
        
        formatted = []
        for paper_title, snippets in papers.items():
            formatted.append(f"\nPaper: {paper_title}")
            for snippet in snippets:
                formatted.append(f"  - {snippet}")
        return "\n".join(formatted)
    
    def __str__(self):
        return f"Domain(domain_name={self.domain_name})"
    
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash(self.domain_name)
    
    def __eq__(self, other):
        if isinstance(other, Domain):
            return self.domain_name == other.domain_name
        return False


@dataclass
class Question:
    """Represents a research question with associated metadata."""
    
    def __init__(self, id: str, domain_specific_question: str, domain_agnostic_question: str, rationale: str, parent_question: Optional['Question'] = None):
        self.id = id
        self.domain_specific_question = domain_specific_question
        self.domain_agnostic_question = domain_agnostic_question
        self.rationale = rationale
        self.parent_question = parent_question  # For tracking sub-question hierarchy
        
        # Analysis results
        self.target_domain_analysis = None  # TargetDomainAnalysis object
        self.is_addressed_in_target = False  # Whether substantially/partially addressed
        
        # External domains to explore
        self.external_domains: Dict[str, Domain] = {}  # domain_name: Domain object
        self.cross_domain_queries = None  # CrossDomainQueries object
        
        # Sub-questions generated from target domain analysis
        self.remaining_challenges: List['Question'] = []  # Sub-questions that need cross-domain search

        self.integrated_ideas: Dict[str, dict] = {}  # domain_name: integrated_idea
        self.interdisciplinary_rankings: Optional[dict] = None
    
    def add_external_domain(self, domain: Domain):
        """Add an external domain for cross-domain search."""
        self.external_domains[domain.domain_name] = domain
    
    def mark_as_addressed(self, addressed: bool):
        """Mark whether this question was addressed in the target domain."""
        self.is_addressed_in_target = addressed
    
    def add_remaining_challenge(self, challenge: 'Question'):
        """Add a remaining challenge (sub-question) from target domain analysis."""
        self.remaining_challenges.append(challenge)
    
    def needs_cross_domain_search(self) -> bool:
        """Determine if this question needs cross-domain exploration."""
        # Need cross-domain if not addressed in target OR if there are remaining challenges
        return not self.is_addressed_in_target or len(self.remaining_challenges) > 0
    
    def add_integrated_idea(self, domain_name: str, integrated_idea: dict):
        """Add an integrated idea for a specific external domain."""
        self.integrated_ideas[domain_name] = integrated_idea

    def set_interdisciplinary_rankings(self, rankings: dict):
        """Set the interdisciplinary potential rankings for this question."""
        self.interdisciplinary_rankings = rankings
    
    def __str__(self):
        return f"Question(id={self.id}, domain_specific={self.domain_specific_question}, domain_agnostic={self.domain_agnostic_question})"
    
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash(self.__str__())
    
    def __eq__(self, other):
        if isinstance(other, Question):
            return self.__str__() == other.__str__()
        return False


@dataclass
class ResearchProblem:
    """Represents the overall research problem with decomposed questions."""
    
    def __init__(self, problem_statement: str, target_domain_name: str, 
                 fine_grained_domain: str = "", core_challenge: str = ""):
        self.problem_statement = problem_statement
        self.target_domain = Domain(domain_name=target_domain_name)
        self.fine_grained_domain = fine_grained_domain
        self.core_challenge = core_challenge
        
        # All domains involved (target + external)
        self.domains: Dict[str, Domain] = {target_domain_name: self.target_domain}
        
        # Top-level research questions
        self.research_questions: List[Question] = []
        
        # All questions including sub-questions (for easy lookup)
        self.all_questions: Dict[str, Question] = {}
    
    @classmethod
    def from_initial_decomposition(cls, decomposition_json: dict, fine_grained_domain: str):
        """Create ResearchProblem from initial decomposition output."""
        problem = cls(
            problem_statement=decomposition_json["problem_statement"],
            target_domain_name=decomposition_json.get("coarse_grained_domain", ""),
            fine_grained_domain=fine_grained_domain,
            core_challenge=decomposition_json.get("core_challenge", "")
        )
        
        # Create questions from decomposition
        for q_data in decomposition_json["research_questions"]:
            question = Question(
                id=q_data["id"],
                domain_specific_question=q_data["domain_specific_question"],
                domain_agnostic_question=q_data["domain_agnostic_question"],
                rationale=q_data["rationale"]
            )
            
            # Add target domain queries
            queries = q_data.get("target_domain_queries", [])
            if queries:
                problem.target_domain.add_question_queries(question, queries)
            
            problem.add_research_question(question)
        
        return problem
    
    def add_research_question(self, question: Question):
        """Add a top-level research question."""
        self.research_questions.append(question)
        self.all_questions[str(question)] = question
    
    def add_remaining_challenge(self, parent_question: Question, challenge_data: dict):
        """Create and add a remaining challenge as a sub-question."""
        challenge = Question(
            id=challenge_data["challenge_id"],
            domain_specific_question=challenge_data["domain_specific_challenge_question"],
            domain_agnostic_question=challenge_data["domain_agnostic_challenge_question"],
            rationale=f'{challenge_data.get("importance", "")} {challenge_data.get("why_unaddressed", "")}',
            parent_question=parent_question
        )
        
        parent_question.add_remaining_challenge(challenge)
        self.all_questions[challenge.id] = challenge
        
        return challenge
    
    def get_or_create_domain(self, domain_name: str) -> Domain:
        """Get existing domain or create new one."""
        if domain_name not in self.domains:
            self.domains[domain_name] = Domain(domain_name=domain_name)
        return self.domains[domain_name]
    
    def get_questions_needing_cross_domain(self) -> List[Question]:
        """Get all questions (including challenges) that need cross-domain search."""
        questions_needing_search = []
        
        # Check all top-level questions
        for question in self.research_questions:
            if question.needs_cross_domain_search():
                if not question.is_addressed_in_target:
                    # If the question itself needs search (is "substantially unaddressed (don't do the decomposition yet)")
                    questions_needing_search.append(question)
                else:
                    # Add any remaining challenges
                    questions_needing_search.extend(question.remaining_challenges)
        
        return questions_needing_search
    
    def __str__(self):
        return (f"ResearchProblem(problem={self.problem_statement[:50]}..., "
                f"domain={self.target_domain.domain_name}, "
                f"questions={len(self.research_questions)})")
    
    def __repr__(self):
        return self.__str__()