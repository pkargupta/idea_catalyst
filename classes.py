from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, RootModel, Field
from typing import Any, Dict, List, Optional
import re

# -----------------------------
# Core domain model
# -----------------------------

@dataclass(frozen=True)
class Span:
    """Half-open character span [start, end) into the proposal text."""
    start: int
    end: int

    def extract(self, text: str) -> str:
        return text[self.start : self.end]

@dataclass
class Domain:
    """A research domain."""
    theme: str
    coarse_domain: Optional[str] = None
    fine_domain: Optional[str] = None
    rationale: Optional[str] = None
    queries: List[str] = field(default_factory=list)
    snippets: Dict[str, List[str]] = field(default_factory=dict)  # paper title -> snippets
    bridge_idea: Optional[str] = None  # populated later
    gap: Optional[str] = None  # populated later

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theme": self.theme,
            "coarse_domain": self.coarse_domain,
            "fine_domain": self.fine_domain,
            "rationale": self.rationale,
            "queries": list(self.queries),
            "snippets": dict(self.snippets),
            "bridge_idea": self.bridge_idea,
            "gap": self.gap,
        }
    
    def __str__(self):
        return f"Domain(theme={self.theme}, coarse_domain={self.coarse_domain}, fine_domain={self.fine_domain})"


@dataclass
class Theme:
    """A theme extracted from the proposal."""
    theme: str
    dimension: Optional[str] = None
    segments: List[str] = field(default_factory=list)
    domains: List[Domain] = field(default_factory=list)  # populated later

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theme": self.theme,
            "dimension": self.dimension,
            "segments": list(self.segments),
            "domains": [domain.to_dict() for domain in self.domains],
        }
    
    def __str__(self):
        return f"Theme(theme={self.theme}, dimension={self.dimension})"

@dataclass
class Proposal:
    text: str
    themes: Dict[str, Theme]

    def normalize(self) -> str:
        # Light cleanup; avoid anything lossy.
        t = self.text.replace("\r\n", "\n").strip()
        # Collapse excessive whitespace
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal": self.text,
            "themes": {k: v.to_dict() for k, v in self.themes.items()},
        }


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    url: Optional[str] = None
    citation_count: Optional[int] = None

    # Optionally attach lightweight retrieved context for prompting
    context_snippets: List[str] = field(default_factory=list)

class ThemeData(BaseModel):
    theme: str
    dimension: Optional[str]
    segments: List[str]

class ThemeSchema(BaseModel):
    themes: List[ThemeData] = Field(..., max_length=5)

theme_json_schema = ThemeSchema.model_json_schema()

class DomainData(BaseModel):
    coarse_domain: Optional[str]
    fine_grained_area: Optional[str]
    rationale: Optional[str]
    queries: List[str]

class DomainSchema(RootModel[Dict[str, DomainData]]):
    pass

domain_json_schema = DomainSchema.model_json_schema()

class GapData(BaseModel):
    gap: str
    gap_type: Optional[str]
    evidence: List[str]
    domain_connection: Optional[str]
    improvement: Optional[str]
    expected_benefit: Optional[str]

gaps_json_schema = GapData.model_json_schema()

class BridgeIdeaData(BaseModel):
    domain_problem: str
    proposal_contribution: str
    bridge_concept: str
    adaptation_needed: str
    novel_idea: str
    feasibility: str
    anticipated_challenges: str
    potential_impact: str

bridge_idea_json_schema = BridgeIdeaData.model_json_schema()

@dataclass
class Collaboration:
    proposal: Proposal
    collaborators: Dict[Domain, List[str]]  # collaborator domain -> list of messages (convo_history)