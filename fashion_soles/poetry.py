"""Write poems from the outfit analysis with a Tracery-style grammar (config/poem_grammar.yaml).

A poem follows a fixed arc of slots: opening image, observation, the second
piece, color imagery, their relation, the palette and its mood, a turn, echoes,
and a closing. Each slot has a priority; shorter poems drop the least important
slots first, so the opening, the turn, and the closing always stay.

Every slot expands a grammar symbol narrowed by the tone and by the narrative
function of its motif (from the mapping rules), so the same structure yields
different imagery for, say, an outfit read as "absence" or as "rupture". The
random choices are seeded by the analysis, the tone, and the variation: the same
outfit gives the same poem, and "Generate poem" asks for the next variation.

Forms use syllables and rhymes from the CMU dictionary (prosody.py): "closing
rhyme" ends the free-verse arc on a rhymed pair, "couplets" writes rhymed pairs
throughout, and "haiku" samples short phrases until the lines count 5-7-5.
Repeated motifs swap to WordNet synonyms (synonyms.py).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import prosody
from .color_names import fashion_color_name
from .grammar import Grammar, load_grammar
from .schemas import MappingTrace, OutfitAnalysis, Poem
from .synonyms import alternatives

GRAMMAR = Path(__file__).resolve().parents[1] / "config" / "poem_grammar.yaml"
TONES = ("lyrical", "minimal", "fragmentary")
FORMS = ("free verse", "closing rhyme", "couplets", "haiku")

# (slot, priority) in reading order. Sixteen slots for the sixteen-line maximum.
ARC = [
    ("opening", 10), ("observe", 6), ("other", 8), ("imagery", 7), ("relation", 7), ("develop", 3),
    ("palette", 9), ("mood", 8), ("turn", 9), ("imagery", 6), ("develop", 2), ("echo", 4),
    ("develop", 1), ("echo", 3), ("quiet", 2), ("closing", 10),
]

PATTERN_WORDS = {"striped": "striped", "checked": "checked", "dotted": "dotted", "floral print": "flowered",
                 "geometric print": "geometric", "graphic print": "printed"}


def validate_poem(poem: Poem, trace: MappingTrace, length: int) -> Poem:
    allowed = {r.rule_id for r in trace.matched_rules} | {t.rule_id for t in trace.tensions}
    if len(poem.lines) != length:
        raise ValueError(f"Expected {length} lines, received {len(poem.lines)}.")
    if not set(poem.used_rule_ids).issubset(allowed):
        raise ValueError("Poem cites an inactive mapping rule.")
    return poem


def _patterned(garment) -> str:
    """Poem name for a piece, e.g. "striped top".

    Detector labels list alternatives ("bag, wallet"); only the first reads well in a sentence.
    Plain and merely textured pieces get no pattern word.
    """
    name = garment.name.split(",", 1)[0].strip() or garment.name
    word = next((PATTERN_WORDS[value] for value in garment.texture if value in PATTERN_WORDS), None)
    return f"{word} {name}" if word else name


def _imagery(garments, variation: int) -> list[tuple[str, str]]:
    """Up to two (piece, whimsical color name) pairs; plain color words ("black") are skipped."""
    found, used = [], set()
    for offset in range(len(garments)):
        garment = garments[(variation + offset) % len(garments)]
        if garment.swatch is None:
            continue
        name = fashion_color_name(garment.swatch, "whimsical").lower()
        if name in used or name == fashion_color_name(garment.swatch, "basic").lower():
            continue
        used.add(name)
        found.append((garment.name, name))
    return found[:2]


def _plan(length: int, available: dict[str, int]) -> list[str]:
    """Pick slots for the poem's length; slots whose data is missing become "develop" lines."""
    slots, imagery_seen = [], 0
    for slot, priority in ARC:
        if slot == "imagery":
            imagery_seen += 1
            slot = slot if imagery_seen <= available["imagery"] else "develop"
        elif not available.get(slot, 1):
            slot = "develop"
        slots.append((slot, priority))
    # Drop the lowest priority first; among equals, the later slot goes first.
    order = sorted(range(len(slots)), key=lambda index: (slots[index][1], -index))
    dropped = set(order[:len(slots) - length])
    return [slot for index, (slot, _) in enumerate(slots) if index not in dropped]


def _seed(analysis: OutfitAnalysis, *parts) -> int:
    digest = hashlib.sha256("|".join([analysis.model_dump_json(), *map(str, parts)]).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _rhymed_pairs(grammar: Grammar, count: int, keys: list[list[str]]) -> list[str]:
    """``count`` rhymed pairs, each from a different rhyme group, keys cycling per pair."""
    groups = list(grammar.rules["rhymes"])
    grammar.rng.shuffle(groups)
    lines = []
    for index in range(count):
        first, second = grammar.rng.sample(groups[index % len(groups)], 2)
        pair_keys = keys[index % len(keys)]
        lines += [grammar.expand("rhymed", pair_keys, {"end": first}),
                  grammar.expand("rhymed", pair_keys, {"end": second})]
    return lines


def _haiku_line(grammar: Grammar, symbol: str, target: int, keys: list[str]) -> str:
    """Sample phrases until one has ``target`` syllables, padding a line that is one or two short."""
    pads = grammar.rules["haiku_pad"]
    best = None
    for _ in range(300):
        text = grammar.expand(symbol, keys)
        gap = target - prosody.line_syllables(text)
        if gap == 0:
            return text
        if gap in pads:
            return f"{grammar.rng.choice(pads[gap])} {text}"
        if best is None or abs(gap) < abs(best[0]):
            best = (gap, text)
    return best[1]


def generate_poem(analysis: OutfitAnalysis, trace: MappingTrace, *, length: int = 12,
                  tone: str = "lyrical", variation: int = 0, color_mode: str = "basic",
                  form: str = "free verse") -> Poem:
    """Render a deterministic poem from selected observations and symbolic rules, locally.

    Couplets round the length down to an even number; a haiku always has three lines.
    """
    if tone not in TONES or form not in FORMS or not 8 <= length <= 16 or variation < 0:
        raise ValueError("Invalid poem controls.")
    if form != "free verse" and not prosody.available():
        raise RuntimeError("Rhymes and syllables need the CMU dictionary; run python scripts/download_language_data.py.")
    if not analysis.garments:
        raise ValueError("Select at least one clothing piece to generate a poem.")

    garments = [garment.model_copy(update={"name": _patterned(garment)}) for garment in analysis.garments]
    item = garments[variation % len(garments)]
    other = garments[(variation + 1) % len(garments)]
    motifs = trace.dominant_motifs or ["shape", "space"]
    motif = motifs[variation % len(motifs)]
    second = motifs[(variation + 1) % len(motifs)] if len(motifs) > 1 else "space"
    function_of = {}
    for rule in trace.matched_rules:
        for name in rule.motifs:
            function_of.setdefault(name, rule.narrative_function)
    palette = [fashion_color_name(entry.hex, color_mode).lower() for entry in analysis.overall.palette]
    palette = palette or analysis.overall.dominant_colors[:3]
    color = item.colors[0] if item.colors else "uncolored"
    relations = analysis.overall.color_relations
    imagery = _imagery(garments, variation)
    tension = trace.tensions[variation % len(trace.tensions)] if trace.tensions else None

    rules = load_grammar(GRAMMAR)
    blocked, extra = rules["synonyms"]["block"], rules["synonyms"]["extra"]
    grammar = Grammar(rules, _seed(analysis, tone, variation, color_mode, form), {
        "item": item.name, "other": other.name, "color": color,
        "other_color": other.colors[0] if other.colors else "uncolored",
        "outline": item.outline if item.outline != "unknown" else "visible",
        "motif": alternatives(motif, blocked, extra), "second": alternatives(second, blocked, extra),
        "name": imagery[0][1] if imagery else color,
        "lead": palette[0] if palette else color,
        "follow": palette[1] if len(palette) > 1 else "",
        "edge": palette[2] if len(palette) > 2 else "",
        "moods": " and ".join(relations.moods[:2]) if relations and relations.moods else "",
    })
    available = {"other": len(garments) > 1, "imagery": len(imagery), "palette": bool(palette),
                 "mood": bool(relations and relations.moods)}
    base_keys = [tone, function_of.get(motif, "any")]
    second_keys = [tone, function_of.get(second, "any")]
    title = grammar.expand("title", base_keys)  # Before the lines, so the title keeps the motif's own word.
    lines = []
    if form == "haiku":
        lines = [_haiku_line(grammar, "haiku_five", 5, base_keys), _haiku_line(grammar, "haiku_seven", 7, second_keys),
                 _haiku_line(grammar, "haiku_close", 5, base_keys)]
    elif form == "couplets":
        length -= length % 2
        lines = _rhymed_pairs(grammar, length // 2, [base_keys, second_keys])
    imagery_index = 0
    for slot in _plan(length, available) if form in ("free verse", "closing rhyme") else []:
        keys = base_keys
        if slot == "relation":
            if tension is not None:
                lines.append(tension.description)
                continue
            slot = "pairing" if len(garments) > 1 else "alone"
            keys = [tone, function_of.get(second, "any")]
        elif slot == "turn":
            keys = [tone, function_of.get(second, "any")]
        elif slot == "imagery":
            piece, name = imagery[imagery_index]
            imagery_index += 1
            lines.append(grammar.expand("imagery", keys, {"item": piece, "name": name}))
            continue
        elif slot == "palette":
            keys = [tone, ("one", "two", "three")[min(len(palette), 3) - 1]]
        elif slot == "mood":
            keys = [tone, relations.scheme, relations.harmony]
        lines.append(grammar.expand(slot, keys))

    if form == "closing rhyme":
        lines[-2:] = _rhymed_pairs(grammar, 1, [base_keys])

    used = list(dict.fromkeys(rule.rule_id for rule in trace.matched_rules
                              if motif in rule.motifs or second in rule.motifs))
    if tension is not None and form in ("free verse", "closing rhyme"):
        used.append(tension.rule_id)
    poem = Poem(title=title, lines=lines, used_rule_ids=list(dict.fromkeys(used)))
    return validate_poem(poem, trace, len(lines) if form == "haiku" else length)
