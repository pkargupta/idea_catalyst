from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import json_repair
from typing import List, Dict
from classes import Question, Domain
from search import search_semantic_scholar, collect_snippets

arxiv_domains = {
    "cs.ai": "Artificial Intelligence",
    "cs.ar": "Hardware Architecture",
    "cs.cc": "Computational Complexity",
    "cs.ce": "Computational Engineering, Finance, and Science",
    "cs.cg": "Computational Geometry",
    "cs.cl": "Computation and Language",
    "cs.cr": "Cryptography and Security",
    "cs.cv": "Computer Vision and Pattern Recognition",
    "cs.cy": "Computers and Society",
    "cs.db": "Databases",
    "cs.dc": "Distributed, Parallel, and Cluster Computing",
    "cs.dl": "Digital Libraries",
    "cs.dm": "Discrete Mathematics",
    "cs.ds": "Data Structures and Algorithms",
    "cs.et": "Emerging Technologies",
    "cs.fl": "Formal Languages and Automata Theory",
    "cs.gl": "General Literature",
    "cs.gr": "Graphics",
    "cs.gt": "Computer Science and Game Theory",
    "cs.hc": "Human-Computer Interaction",
    "cs.ir": "Information Retrieval",
    "cs.it": "Information Theory",
    "cs.lg": "Machine Learning",
    "cs.lo": "Logic in Computer Science",
    "cs.ma": "Multiagent Systems",
    "cs.mm": "Multimedia",
    "cs.ms": "Mathematical Software",
    "cs.na": "Numerical Analysis",
    "cs.ne": "Neural and Evolutionary Computing",
    "cs.ni": "Networking and Internet Architecture",
    "cs.oh": "Other Computer Science",
    "cs.os": "Operating Systems",
    "cs.pf": "Performance",
    "cs.pl": "Programming Languages",
    "cs.ro": "Robotics",
    "cs.sc": "Symbolic Computation",
    "cs.sd": "Sound",
    "cs.se": "Software Engineering",
    "cs.si": "Social and Information Networks",
    "cs.sy": "Systems and Control",

    "econ.em": "Econometrics",
    "econ.gn": "General Economics",
    "econ.th": "Theoretical Economics",

    "eess.as": "Audio and Speech Processing",
    "eess.iv": "Image and Video Processing",
    "eess.sp": "Signal Processing",
    "eess.sy": "Systems and Control",

    "math.ac": "Commutative Algebra",
    "math.ag": "Algebraic Geometry",
    "math.ap": "Analysis of PDEs",
    "math.at": "Algebraic Topology",
    "math.ca": "Classical Analysis and ODEs",
    "math.co": "Combinatorics",
    "math.ct": "Category Theory",
    "math.cv": "Complex Variables",
    "math.dg": "Differential Geometry",
    "math.ds": "Dynamical Systems",
    "math.fa": "Functional Analysis",
    "math.gm": "General Mathematics",
    "math.gn": "General Topology",
    "math.gr": "Group Theory",
    "math.gt": "Geometric Topology",
    "math.ho": "History and Overview",
    "math.it": "Information Theory",
    "math.kt": "K-Theory and Homology",
    "math.lo": "Logic",
    "math.mg": "Metric Geometry",
    "math.mp": "Mathematical Physics",
    "math.na": "Numerical Analysis",
    "math.nt": "Number Theory",
    "math.oa": "Operator Algebras",
    "math.oc": "Optimization and Control",
    "math.pr": "Probability",
    "math.qa": "Quantum Algebra",
    "math.ra": "Rings and Algebras",
    "math.rt": "Representation Theory",
    "math.sg": "Symplectic Geometry",
    "math.sp": "Spectral Theory",
    "math.st": "Statistics Theory",

    "astro-ph.co": "Cosmology and Nongalactic Astrophysics",
    "astro-ph.ep": "Earth and Planetary Astrophysics",
    "astro-ph.ga": "Astrophysics of Galaxies",
    "astro-ph.he": "High Energy Astrophysical Phenomena",
    "astro-ph.im": "Instrumentation and Methods for Astrophysics",
    "astro-ph.sr": "Solar and Stellar Astrophysics",

    "cond-mat.dis-nn": "Disordered Systems and Neural Networks",
    "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics",
    "cond-mat.mtrl-sci": "Materials Science",
    "cond-mat.other": "Other Condensed Matter",
    "cond-mat.quant-gas": "Quantum Gases",
    "cond-mat.soft": "Soft Condensed Matter",
    "cond-mat.stat-mech": "Statistical Mechanics",
    "cond-mat.str-el": "Strongly Correlated Electrons",
    "cond-mat.supr-con": "Superconductivity",

    "gr-qc": "General Relativity and Quantum Cosmology",
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "High Energy Physics - Lattice",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",
    "math-ph": "Mathematical Physics",

    "nlin.ao": "Adaptation and Self-Organizing Systems",
    "nlin.cd": "Chaotic Dynamics",
    "nlin.cg": "Cellular Automata and Lattice Gases",
    "nlin.ps": "Pattern Formation and Solitons",
    "nlin.si": "Exactly Solvable and Integrable Systems",

    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",

    "physics.acc-ph": "Accelerator Physics",
    "physics.ao-ph": "Atmospheric and Oceanic Physics",
    "physics.app-ph": "Applied Physics",
    "physics.atm-clus": "Atomic and Molecular Clusters",
    "physics.atom-ph": "Atomic Physics",
    "physics.bio-ph": "Biological Physics",
    "physics.chem-ph": "Chemical Physics",
    "physics.class-ph": "Classical Physics",
    "physics.comp-ph": "Computational Physics",
    "physics.data-an": "Data Analysis, Statistics and Probability",
    "physics.ed-ph": "Physics Education",
    "physics.flu-dyn": "Fluid Dynamics",
    "physics.gen-ph": "General Physics",
    "physics.geo-ph": "Geophysics",
    "physics.hist-ph": "History and Philosophy of Physics",
    "physics.ins-det": "Instrumentation and Detectors",
    "physics.med-ph": "Medical Physics",
    "physics.optics": "Optics",
    "physics.plasm-ph": "Plasma Physics",
    "physics.pop-ph": "Popular Physics",
    "physics.soc-ph": "Physics and Society",
    "physics.space-ph": "Space Physics",

    "quant-ph": "Quantum Physics",

    "q-bio.bm": "Biomolecules",
    "q-bio.cb": "Cell Behavior",
    "q-bio.gn": "Genomics",
    "q-bio.mn": "Molecular Networks",
    "q-bio.nc": "Neurons and Cognition",
    "q-bio.ot": "Other Quantitative Biology",
    "q-bio.pe": "Populations and Evolution",
    "q-bio.qm": "Quantitative Methods",
    "q-bio.sc": "Subcellular Processes",
    "q-bio.to": "Tissues and Organs",

    "q-fin.cp": "Computational Finance",
    "q-fin.ec": "Economics",
    "q-fin.gn": "General Finance",
    "q-fin.mf": "Mathematical Finance",
    "q-fin.pm": "Portfolio Management",
    "q-fin.pr": "Pricing of Securities",
    "q-fin.rm": "Risk Management",
    "q-fin.st": "Statistical Finance",
    "q-fin.tr": "Trading and Market Microstructure",

    "stat.ap": "Applications",
    "stat.co": "Computation",
    "stat.me": "Methodology",
    "stat.ml": "Machine Learning",
    "stat.ot": "Other Statistics",
    "stat.th": "Statistics Theory"
}

def convert_domain(input_name):
    if input_name.lower() in arxiv_domains:
        return arxiv_domains[input_name.lower()]
    else:
        return input_name


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


def retrieve_papers_for_question(question: Question, domain: Domain, max_papers: int = 10, year=None) -> Dict[str, List[str]]:
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
            response = search_semantic_scholar(query, domain.domain_name, year=year)
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