from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
import yaml
from .schemas import MappingTrace, MatchedRule, OutfitAnalysis, Tension

DEFAULT_RULES = Path(__file__).resolve().parents[1] / "config" / "mappings.yaml"


@lru_cache(maxsize=8)
def load_rules(path: str = str(DEFAULT_RULES)) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    ids = [r["id"] for r in data["rules"]] + [r["id"] for r in data["combinations"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Mapping rule IDs must be unique")
    return data


def canonical(value: str, synonyms: dict[str, str]) -> str:
    clean = value.strip().lower()
    return synonyms.get(clean, clean)


def observations(analysis: OutfitAnalysis, synonyms: dict[str, str]) -> list[tuple[str, str, str]]:
    result = []
    for garment in analysis.garments:
        for feature, values in (("color", garment.colors), ("surface", garment.colors + [garment.material] + garment.texture),
                                ("material", [garment.material]), ("texture", garment.texture),
                                ("silhouette", garment.silhouette), ("condition", garment.condition),
                                ("detail", garment.visible_details), ("outline", [garment.outline]),
                                ("category", [garment.category])):
            for raw in values:
                value = canonical(raw, synonyms)
                if value and value != "unknown":
                    result.append((feature, value, f"garment:{garment.id}.{feature}:{raw}"))
    for raw in analysis.overall.silhouette:
        value = canonical(raw, synonyms)
        if value != "unknown":
            result.append(("silhouette", value, f"overall.silhouette:{raw}"))
    relations = analysis.overall.color_relations
    if relations is not None:
        for feature, values in (("color_scheme", [relations.scheme]), ("color_harmony", [relations.harmony]),
                                ("color_temperature", [relations.temperature]), ("color_mood", relations.moods)):
            for raw in values:
                result.append((feature, canonical(raw, synonyms), f"overall.color_relations.{feature}:{raw}"))
    value = canonical(analysis.overall.contrast, synonyms)
    if value != "unknown":
        result.append(("contrast", value, f"overall.contrast:{analysis.overall.contrast}"))
    for relationship in analysis.relationships:
        value = canonical(relationship.kind, synonyms)
        if value != "unknown":
            result.append(("relationship", value,
                           f"relationship:{','.join(relationship.garment_ids)}:{relationship.description}"))
    return result


def map_analysis(analysis: OutfitAnalysis, rules: dict | None = None) -> MappingTrace:
    rules = rules or load_rules()
    found = []
    seen = set()
    for feature, value, source in observations(analysis, rules.get("synonyms", {})):
        for rule in rules["rules"]:
            if feature == rule["feature"] and value in rule["values"] and (rule["id"], source) not in seen:
                seen.add((rule["id"], source))
                found.append(MatchedRule(rule_id=rule["id"], source=source, motifs=rule["motifs"],
                                         narrative_function=rule["function"], rationale=rule["rationale"]))
    ids = {match.rule_id for match in found}
    tensions = [Tension(rule_id=rule["id"], between=rule["between"], description=rule["description"])
                for rule in rules["combinations"] if set(rule["all"]).issubset(ids)]
    weights = {rule["id"]: rule.get("weight", 1) for rule in rules["rules"]}
    scores = Counter()
    scored = set()
    for match in found:
        origin = match.source.split(":", 2)[1].split(".", 1)[0] if match.source.startswith("garment:") else match.source
        key = (match.rule_id, origin)
        if key in scored:
            continue
        scored.add(key)
        for motif in match.motifs:
            scores[motif] += weights[match.rule_id]
    for tension in tensions:
        for motif in tension.between:
            scores[motif] += 2
    dominant = [motif for motif, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:4]]
    return MappingTrace(matched_rules=found, dominant_motifs=dominant, tensions=tensions)
