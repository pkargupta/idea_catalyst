import json
from classes import Domain, Theme, Paper
from typing import Dict, List, Optional

class PromptBuilder:
    @staticmethod
    def theme_extraction_prompt(proposal_text: str, dimensions: List[str]) -> str:
        """Returns a strict JSON-output prompt."""
        dims = ", ".join(dimensions)
    
        prompt = f"""You are an expert research analyst specializing in cross-disciplinary knowledge synthesis and thematic extraction from scientific proposals

**TASK**
Systematically identify and extract the key themes that characterize the research proposal. Your analysis should reveal both dimension-specific themes (tied to particular aspects of the research) and cross-cutting themes (spanning multiple dimensions).

**CRITICAL REQUIREMENTS FOR THEME FORMULATION**
- **Domain-agnostic abstraction:** Rewrite each theme to focus on the underlying problem, goal, methodological principle, or conceptual contribution—NOT field-specific terminology or application-specific details
- **Transferability:** Themes should be recognizable and applicable across different domains, disciplines, and application contexts
- **Clarity:** Each theme should be self-contained and immediately understandable without requiring knowledge of the specific proposal domain
- **Precision:** Avoid vague generalities; capture the specific intellectual contribution or challenge being addressed

**RESEARCH DIMENSIONS** (assign only when clearly applicable)

1. **goal_motivation** — The fundamental problem being addressed, its significance, and the research motivation
Example themes:
- "Early detection of rare but high-impact failure events in complex systems"
- "Improving decision-making under limited, noisy, or contradictory information"
- "Enabling personalized interventions at scale while preserving privacy"
- "Bridging the gap between theoretical guarantees and practical performance"

2. **method** — The general approach, technical strategy, algorithmic principle, or methodological innovation
Example themes:
- "Learning robust representations from heterogeneous, multi-modal time-series data"
- "Combining mechanistic domain knowledge with data-driven statistical models"
- "Decomposing complex problems into modular, independently solvable subproblems"
- "Iteratively refining predictions through feedback loops and active learning"

3. **dataset** — The nature, role, characteristics, or strategic use of data in the research
Example themes:
- "Leveraging weakly labeled, real-world observational data with inherent biases"
- "Achieving cross-domain generalization despite distribution shift and covariate mismatch"
- "Synthesizing insights from multiple heterogeneous data sources with varying quality"
- "Learning from limited labeled examples through transfer or few-shot learning"

4. **experiments** — The evaluation strategy, validation approach, or methodology for assessing research claims
Example themes:
- "Stress-testing models under controlled perturbations and adversarial conditions"
- "Comparing performance across simulated, semi-synthetic, and real-world settings"
- "Evaluating robustness through systematic ablation of components and assumptions"
- "Using expert judgment and qualitative analysis alongside quantitative metrics"

**DETAILED INSTRUCTIONS**

1. **Theme Identification:** Identify between 3–8 distinct themes that capture the proposal's core intellectual contributions
- Look for recurring concepts, emphasized challenges, novel approaches, and evaluation strategies
- Consider both explicit statements and implicit methodological choices
- Ensure themes are at an appropriate level of abstraction—neither too generic nor too specific

2. **Domain-agnostic Rewriting:** For each identified theme:
- Strip away domain-specific jargon, technical acronyms, and application-specific terminology
- Focus on the transferable problem, principle, or strategy
- Ask: "What would this look like in a completely different field?"
- Ensure the theme could appear in proposals from multiple disciplines

3. **Evidence Grounding:** For each theme:
- Extract 1–3 verbatim text segments from the proposal that directly support or exemplify the theme
- Segments should be **exact substrings** from the proposal text
- Select the most representative and clear evidence
- Segments can be phrases, sentences, or short passages
- Prioritize segments that explicitly articulate the theme

4. **Dimension Assignment:**
- Assign a dimension ({' | '.join(dims)}) ONLY if the theme clearly and predominantly relates to that dimension
- Use `null` for cross-cutting themes or when no single dimension dominates
- When uncertain, prefer `null` over forcing an incorrect categorization

**OUTPUT FORMAT** (Valid JSON only—no markdown, no commentary)

{{
"themes": [
    {{
    "dimension": "<dimension_name>" | null,
    "theme": "<concise, domain-agnostic thematic description>",
    "segments": [
        "<verbatim substring from proposal text>",
        "<verbatim substring from proposal text>",
        ...
    ]
    }},
    ...
]
}}

**QUALITY CHECKLIST**
Before finalizing, verify:
- [ ] Each theme is written without domain-specific jargon
- [ ] Themes capture distinct aspects (minimal redundancy)
- [ ] All segments are exact substrings from the proposal
- [ ] Dimension assignments are accurate or appropriately null
- [ ] 3–8 themes total, covering major proposal contributions
- [ ] Output is valid JSON

**PROPOSAL TEXT**

{proposal_text}

Analyze the proposal above and output your thematic analysis as JSON."""

        return prompt

    @staticmethod
    def domain_discovery_prompt(themes) -> str:
        return f"""**Role:** You are a research strategist specializing in cross-disciplinary knowledge synthesis and scientific literature analysis.

**Task:** For each theme provided in the input JSON, conduct a systematic analysis to identify parallel research domains and formulate targeted search queries for discovering relevant academic papers.

**Detailed Instructions:**

For each theme, you must:

1. **Identify the Core Problem:** Extract and articulate the fundamental challenge or research question that the theme addresses, abstracting away domain-specific terminology.

2. **Map to Related Coarse-Grained Research Domains:** Identify 1 distinct and diverse coarse-grained research domain from the following options: Computer Science, Medicine, Chemistry, Biology, Materials Science, Physics, Geology, Psychology, Art, History, Geography, Sociology, Business, Political Science, Economics, Philosophy, Mathematics, Engineering, Environmental Science, Agricultural and Food Sciences, Education, Law, Linguistics. Consider:
    - Broader research domains, fields, or communities that tackle the same or highly analogous core problem.
    - Non-obvious research domains that may offer unique perspectives or methodologies relevant to the theme.
    - Preferably select domains that differ significantly from the original proposal's field to maximize cross-disciplinary insights.

3. **Identify Fine-Grained Research Areas Within Each Domain:** For each identified coarse-grained domain, specify the fine-grained research areas, subfields, or specialties that are particularly relevant to the theme. Consider:
   - Alternative methodological traditions (e.g., symbolic AI vs. neural approaches, qualitative vs. quantitative methods)
   - Specific application areas where similar challenges arise (e.g., legal reasoning, medical diagnosis, education)
   - Both established and emerging research areas

3. **Provide Clear Rationale:** For each identified domain, explain:
   - Why this domain addresses the same core problem
   - What perspective or approach this domain brings
   - How insights from this domain could inform or complement the original theme
   - Specific parallels or analogies between the domains

4. **Formulate Targeted Search Queries:** For each identified domain, create 3-5 search queries optimized for academic databases (Google Scholar, Semantic Scholar, ACL Anthology, PubMed, etc.). Queries should:
   - Have only a max 4 words in the query
   - Use domain-appropriate terminology and keywords
   - Balance specificity (to retrieve relevant papers) with breadth (to avoid over-narrowing)
   - Include methodological terms, problem descriptors, and application contexts
   - Consider both foundational work and recent advances
   - Vary in scope (some broader surveys, some specific techniques)

**Input Format:**
```json
{{
  "themes": [
    {{
      "dimension": "<string>",
      "theme": "<concise theme description>",
      "segments": ["<supporting evidence/context>", ...]
    }}
  ]
}}
```

**Output Format:**
```json
{{
  "<theme_text>":
    {{
      "coarse_domain": "<coarse-grained research domain or field name from the provided list: Computer Science, Medicine, Chemistry, Biology, Materials Science, Physics, Geology, Psychology, Art, History, Geography, Sociology, Business, Political Science, Economics, Philosophy, Mathematics, Engineering, Environmental Science, Agricultural and Food Sciences, Education, Law, Linguistics>",
      "fine_grained_area": "<specific research area, subfield, or specialty within the coarse domain>",
      "rationale": "<detailed explanation of why this domain tackles the same core problem, what unique perspective it offers, and how it relates to the original theme>",
      "queries": [
        "<search query 1>",
        "<search query 2>",
        "<search query 3>",
        "..."
      ]
    }},
    ...
    ,
  "<next_theme_text>": [...]
}}
```

**Quality Criteria:**

- **Diversity:** Ensure identified domains span different levels of abstraction, methodological approaches, and disciplinary boundaries
- **Relevance:** Each domain should genuinely address the core problem, not just share superficial keywords
- **Actionability:** Search queries should be immediately usable and likely to retrieve high-quality, relevant academic papers
- **Insight:** Rationales should demonstrate deep understanding of both the original theme and the identified parallel domain
- **Completeness:** Cover both theoretical foundations and practical implementations/applications

**Example Considerations:**

- For themes about "comparison of contributions," consider: bibliometrics, science of science, argument mining, contrastive learning, systematic review methodologies
- For themes about "multi-perspective reasoning," consider: ensemble methods, wisdom of crowds, deliberative democracy, perspective-taking in cognitive science
- For themes about "hierarchical decomposition," consider: hierarchical planning, divide-and-conquer algorithms, ontology engineering, modular neural architectures
- For themes about "evidence grounding," consider: fact verification, evidence-based reasoning, source attribution, epistemic logic

**Note:** Use the "segments" array to understand context and nuance, but extract the core transferable problem that transcends the specific implementation details.

THEMES:

{themes}
"""
    # For each of the domains within each of the themes in the Proposal instance, have the respective persona generate: (a) gaps in the proposal that their domain could help address

    @staticmethod
    def proposal_gaps_prompt(theme: Theme, domain: Domain, proposal_text: str) -> str:
        dimension_context = {
            "goal_motivation": "the fundamental problem being addressed, its significance, and the research motivation",
            "method": "the general approach, technical strategy, algorithmic principle, or methodological innovation",
            "experiments": "the evaluation strategy, validation approach, or methodology for assessing research claims",
            "dataset": "the nature, role, characteristics, or strategic use of data in the research"
        }
        dim_description = dimension_context.get(theme.dimension, theme.dimension.replace('_', ' '))

        literature = ""
        for paper_title, snippets in domain.snippets.items():
            literature += f"\t- Paper Title: {paper_title}\n"
            for snip in snippets:
                literature += f"\t\t-Snippet: {snip[:400]}\n"
            literature += "\n"
        
        return f"""You are an expert in "{domain.coarse_domain}" with specialized knowledge in "{domain.fine_domain}". 

    CONTEXT:
    The proposal addresses the theme: "{theme.theme}"
    This theme relates to {dim_description} of the proposed work.
    Relevance to your domain: {domain.rationale}

    Relevant literature and findings from your field:
    {literature}

    PROPOSAL TEXT:
    {proposal_text}

    YOUR TASK:
    Conduct a critical analysis to identify a gap in how the proposal's {dim_description} address the theme of {theme.theme.lower()}. Evaluate this specifically through the lens of your domain expertise.

    Consider these gap categories:

    1. METHODOLOGICAL GAPS: Are there established methods, tools, frameworks, or techniques from your domain that are missing but would be valuable for {theme.theme.lower()}?

    2. THEORETICAL GAPS: Does the proposal overlook key theoretical frameworks, models, principles, or conceptual foundations from your field that are relevant to {theme.theme.lower()}?

    3. EMPIRICAL GAPS: Are there important findings, experimental results, empirical evidence, or benchmark comparisons from your domain that the proposal should address but doesn't?

    4. PRACTICAL GAPS: Does the proposal fail to consider implementation challenges, scalability issues, real-world constraints, or practical limitations that your field has documented for approaches related to {theme.theme.lower()}?

    5. ASSUMPTION GAPS: Does the proposal make implicit assumptions about {theme.theme.lower()} that your domain's research has shown to be problematic, oversimplified, or require careful qualification?

    6. INTEGRATION GAPS: Are there opportunities to integrate insights, paradigms, or best practices from your domain that would enhance how the proposal handles {theme.theme.lower()}?

    For the gap you identify:
    - Describe what's missing and why it's significant for {theme.theme.lower()}
    - Quote exact passages from the proposal that demonstrate this gap (use verbatim text)
    - Explain how your domain's research/methods relate to this gap (reference the provided snippets when applicable)
    - Provide concrete, actionable suggestions for addressing the gap using insights from your domain
    - Describe the expected benefit of incorporating your suggested improvement

    Be specific and constructive. Prioritize gaps that would meaningfully strengthen the proposal's treatment of {theme.theme.lower()}.

    OUTPUT FORMAT (JSON):
    {{
        "gap": "<clear description of what's missing>",
        "gap_type": "<methodological|theoretical|empirical|practical|assumption|integration>",
        "evidence": ["<verbatim quote from proposal>", "<additional quotes if needed>"],
        "domain_connection": "<how your domain's research relates to this gap, cite snippets if applicable>",
        "improvement": "<specific, actionable suggestion from your domain>",
        "expected_benefit": "<how this would strengthen the proposal's handling of {theme.theme.lower()}>"
    }}"""

    # (b) specific open research questions from their domain that the proposal has the potential to answer.
    @staticmethod
    def proposal_domain_bridge_prompt(theme: Theme, domain: Domain, proposal_text: str) -> str:
      dimension_context = {
        "goal_motivation": "the fundamental problem being addressed, its significance, and the research motivation",
        "method": "the general approach, technical strategy, algorithmic principle, or methodological innovation",
        "experiments": "the evaluation strategy, validation approach, or methodology for assessing research claims",
        "dataset": "the nature, role, characteristics, or strategic use of data in the research"
      }
      dim_description = dimension_context.get(theme.dimension, theme.dimension.replace('_', ' '))

      literature = ""
      for paper_title, snippets in domain.snippets.items():
        literature += f"\t- Paper Title: {paper_title}\n"
        for snip in snippets:
          literature += f"\t\t-Snippet: {snip[:400]}\n"
        literature += "\n"
      
      return f"""You are an expert in "{domain.coarse_domain}" with specialized knowledge in "{domain.fine_domain}".

    CONTEXT:
    The proposal addresses the theme: "{theme.theme}"
    This theme relates to {dim_description} of the proposed work.
    Relevance to your domain: {domain.rationale}

    Relevant literature and findings from your field:
    {literature}

    PROPOSAL TEXT:
    {proposal_text}

    YOUR TASK:
    Generate a novel idea that bridges the proposal's contributions toward {theme.theme.lower()} with an open or challenging problem in your domain. Describe how the proposal's specific ideas, methods, or approaches could be adapted or applied to tackle this external problem.

    Consider these bridging opportunities:

    1. METHODOLOGICAL TRANSFER: How could the proposal's technical approach or algorithmic strategy be adapted to solve a different but analogous problem in your domain?

    2. CONCEPTUAL ANALOGY: What conceptual insights or principles from the proposal could be reframed and applied to address an unsolved challenge in your field?

    3. FRAMEWORK EXTENSION: Could the proposal's framework, model, or system architecture be extended or modified to handle domain-specific requirements or constraints?

    4. HYBRID APPROACH: How could the proposal's approach be combined with established methods in your domain to create a novel solution to an open problem?

    5. VALIDATION OPPORTUNITY: Could the proposal's methodology serve as a testbed or validation mechanism for theories, assumptions, or predictions in your domain?

    For the bridging idea:
    - Clearly identify the open/challenging problem in your domain that the proposal could help address
    - Describe the specific contribution(s) from the proposal that are relevant (reference the {dim_description})
    - Explain how the proposal's ideas would need to be adapted for your domain's context
    - Outline the potential new idea or solution pathway that emerges from this bridge
    - Discuss feasibility and anticipated challenges in the adaptation
    - Explain the potential impact if this bridging idea were pursued

    OUTPUT FORMAT (JSON):
    {{
      "domain_problem": "<the open or challenging problem in your domain>",
      "proposal_contribution": "<the specific idea/method/approach from the proposal that's relevant>",
      "bridge_concept": "<how the proposal's contribution maps to your domain's problem>",
      "adaptation_needed": "<what modifications or adaptations would be required>",
      "novel_idea": "<the concrete bridging idea that emerges>",
      "feasibility": "<assessment of how realistic this adaptation is>",
      "anticipated_challenges": "<key obstacles or considerations>",
      "potential_impact": "<what this could achieve in your field if successful>"
    }}"""

    @staticmethod
    def proposal_agent_prompt(proposal_text, context_history=None):
      prompt = f"""You are an expert research proposal writer and the author of the following proposal:

    {proposal_text}

    You are now in a collaborative discussion with expert agents who will review and provide feedback on your work. Your role is to:
    - Listen carefully to their suggestions and concerns
    - Explain your design choices and reasoning when questioned
    - Ask clarifying questions to understand their perspective
    - Engage in substantive dialogue about potential improvements
    - Consider how their feedback might strengthen specific aspects of the proposal

    Reference specific sections, methods, or claims when discussing improvements, but focus on exploring ideas and rationales rather than implementing edits directly.
    """
      if context_history:
        prompt += "\nConversation history:\n"
        for message in context_history:
          prompt += f"{message}\n"
      return prompt

    @staticmethod
    def general_collab_agent_prompt(proposal_text, context_history=None):
      prompt = f"""You are an expert research collaborator engaged in a critical discussion about the following proposal:

    {proposal_text}

    Your role is to:
    - Identify strengths and potential weaknesses in the proposal's reasoning, methodology, and presentation
    - Ask probing questions about design choices and assumptions
    - Suggest conceptual improvements or alternative framings (not direct text edits)
    - Point to specific passages when discussing concerns or opportunities
    - Engage in dialogue with the proposal author to refine ideas collaboratively

    Focus on substantive discussion about how the proposal could be strengthened rather than proposing direct edits.
    """
      if context_history:
        prompt += "\nConversation history:\n"
        for message in context_history:
          prompt += f"{message}\n"
      return prompt

    @staticmethod
    def external_domain_collab_agent_prompt(proposal_text, domain, context_history=None):
      literature = ""
      for paper_title, snippets in domain.snippets.items():
        literature += f"\t- Paper Title: {paper_title}\n"
        for snip in snippets:
          literature += f"\t\t- Snippet: {snip[:400]}\n"
        literature += "\n"

      prompt = f"""You are an expert in "{domain.coarse_domain}" with specialized knowledge in "{domain.fine_domain}". You are engaged in a critical discussion about the following proposal:

    {proposal_text}

    Your domain is relevant because: {domain.rationale}

    Your role is to:
    - Identify gaps or opportunities where your domain's perspectives could strengthen the proposal
    - Reference specific sections when discussing how domain-specific insights might apply
    - Ask questions that encourage the proposal author to consider cross-disciplinary implications
    - Suggest conceptual improvements grounded in your field's established practices and findings
    - Engage in dialogue to explore how the proposal could benefit from your domain's approaches

    Ground your discussion in the relevant literature from your field:

    {literature}

    Focus on collaborative exploration of improvements rather than proposing direct edits to the proposal text.
    """
      if context_history:
        prompt += "\nConversation history:\n"
        for message in context_history:
          prompt += f"{message}\n"
      return prompt

    # @staticmethod
    # def conversation_turn_prompt(
    #     *,
    #     role: str,
    #     theme: Theme,
    #     proposal_text: str,
    #     relevant_segments: List[str],
    #     domain_name: Optional[str] = None,
    #     domain_papers: Optional[List[Paper]] = None,
    #     history: List[Dict[str, str]],
    #     instruction: str,
    # ) -> str:
    #     papers_blob = ""
    #     if domain_name and domain_papers:
    #         # Keep context bounded.
    #         lines: List[str] = []
    #         for p in domain_papers[:3]:
    #             lines.append(f"- {p.title} ({p.year or 'n/a'})")
    #             if p.abstract:
    #                 lines.append(f"  Abstract: {p.abstract[:900]}")
    #             for snip in p.context_snippets[:2]:
    #                 lines.append(f"  Snippet: {snip[:400]}")
    #         papers_blob = "\n".join(lines)

    #     history_blob = "\n".join([f"{m['speaker']}: {m['text']}" for m in history])

    #     return (
    #         "You are participating in a structured, multi-turn research critique discussion.\n"
    #         "Your job is to identify gaps, missing considerations, and cross-domain analogies that improve the proposal.\n\n"
    #         f"ROLE: {role}\n"
    #         + (f"DOMAIN: {domain_name}\n" if domain_name else "")
    #         + f"THEME: {theme.theme}\n"
    #         + (f"DIMENSION: {theme.dimension.value}\n" if theme.dimension else "DIMENSION: null\n")
    #         + "\nRELEVANT PROPOSAL SEGMENTS:\n"
    #         + "\n".join([f"- {s}" for s in relevant_segments])
    #         + "\n\nFULL PROPOSAL (for reference):\n"
    #         + proposal_text[:8000]
    #         + ("\n\nDOMAIN CONTEXT (papers/snippets):\n" + papers_blob if papers_blob else "")
    #         + "\n\nDISCUSSION SO FAR:\n"
    #         + (history_blob if history_blob else "<none>")
    #         + "\n\nINSTRUCTION:\n"
    #         + instruction
    #         + "\n\nReturn plain text only. Be specific and actionable.\n"
    #     )

    # @staticmethod
    # def synthesis_prompt(theme, history: List[Dict[str, str]]) -> str:
    #     history_blob = "\n".join([f"{m['speaker']}: {m['text']}" for m in history])
    #     return (
    #         "You are an editor.\n"
    #         "Task: Convert the multi-agent discussion into a single feedback comment addressed to the proposal author.\n"
    #         "Output should be structured with: Summary, Key Gaps, Cross-domain Suggestions, and Concrete Next Steps.\n\n"
    #         f"THEME: {theme}\n"
    #         f"DIMENSION: {theme.dimension.value if theme.dimension else 'null'}\n\n"
    #         "DISCUSSION:\n"
    #         f"{history_blob}\n\n"
    #         "Return plain text only. Keep it to ~200-350 words.\n"
    #     )