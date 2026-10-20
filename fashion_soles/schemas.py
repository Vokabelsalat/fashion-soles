from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SafeModel(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class Overall(SafeModel):
    dominant_colors: list[str] = Field(default_factory=list)
    silhouette: list[str] = Field(default_factory=list)
    contrast: str = "unknown"


class Garment(SafeModel):
    id: str
    category: str = "unknown"
    name: str = "unknown"
    colors: list[str] = Field(default_factory=list)
    material: str = "unknown"
    texture: list[str] = Field(default_factory=list)
    silhouette: list[str] = Field(default_factory=list)
    outline: str = "unknown"
    condition: list[str] = Field(default_factory=list)
    visible_details: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)


class Relationship(SafeModel):
    kind: str
    description: str
    garment_ids: list[str] = Field(default_factory=list)


class OutfitAnalysis(SafeModel):
    image_suitability: Literal["suitable", "partial", "multiple_subjects", "no_outfit", "unclear"]
    warnings: list[str] = Field(default_factory=list)
    overall: Overall
    garments: list[Garment]
    relationships: list[Relationship] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ids(self):
        ids = [g.id for g in self.garments]
        if len(ids) != len(set(ids)):
            raise ValueError("Garment IDs must be unique")
        if any(not set(r.garment_ids).issubset(ids) for r in self.relationships):
            raise ValueError("Relationship refers to an unknown garment")
        return self


class MatchedRule(SafeModel):
    rule_id: str
    source: str
    motifs: list[str]
    narrative_function: str
    rationale: str


class Tension(SafeModel):
    rule_id: str
    between: list[str]
    description: str


class MappingTrace(SafeModel):
    matched_rules: list[MatchedRule]
    dominant_motifs: list[str]
    tensions: list[Tension]


class Poem(SafeModel):
    title: str = Field(min_length=1)
    lines: list[str] = Field(min_length=8, max_length=16)
    used_rule_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def nonempty_lines(self):
        if any(not line.strip() for line in self.lines):
            raise ValueError("Poem lines must not be empty")
        return self
