from __future__ import annotations

from .color_names import fashion_color_name
from .schemas import MappingTrace, OutfitAnalysis, Poem

# Each tone has sixteen lines so the line control can select any length from 8 to 16.
# The two versions change the imagery and cadence when Generate poem is pressed again.
TEMPLATES = {
    "lyrical": (
        (
            "The {color} {item} enters the frame.",
            "Its {outline} outline holds a line of {motif}.",
            "Nearby, the {other} answers in {other_color}.",
            "Two visible shapes make room for {second}.",
            "{relation}",
            "The edge of one shape meets the space of the other.",
            "{color_title} stays with the eye a moment longer.",
            "A {outline} mark crosses the imagined page.",
            "The {item} returns as a small measure of {motif}.",
            "Nothing beyond the photograph needs a name.",
            "Only the colors keep their quiet distance.",
            "The {other} gives {second} a place to rest.",
            "One contour opens; another comes into view.",
            "The picture gathers them without a story.",
            "{motif_title} remains at the visible edge.",
            "Then the image lets the line go.",
        ),
        (
            "First, a {outline} line around the {item}.",
            "Then {color}, held briefly by the light.",
            "The {other} appears beside it in {other_color}.",
            "Between their edges, {second} takes shape.",
            "{relation}",
            "The photograph leaves a little space between them.",
            "In that space, {motif} finds a rhythm.",
            "The {item} is still only what can be seen.",
            "Its outline gives the image a direction.",
            "{other_color_title} answers without explanation.",
            "A second contour shifts the line again.",
            "The {other} holds its own visible place.",
            "Here {second} is a word for the distance.",
            "Here {color} is simply color.",
            "The edges remain after the words move on.",
            "The page ends where the picture continues.",
        ),
    ),
    "minimal": (
        (
            "{color_title} {item}.", "A {outline} outline.",
            "{other_color_title} {other} nearby.", "Two shapes in one frame.",
            "{relation}", "A gap between the edges.",
            "Call that gap {motif}.", "Call the next line {second}.",
            "The {item} stays visible.", "The {other} stays visible.",
            "No further detail is needed.", "Only {color} beside {other_color}.",
            "Only the {outline} mark.", "A small pause in the image.",
            "{motif_title} at the edge.", "The frame holds still.",
        ),
        (
            "A {outline} shape.", "The {item}, in {color}.",
            "Beside it: {other_color}.", "Beside it: the {other}.",
            "{relation}", "One edge, then another.",
            "{second_title} between them.", "{motif_title} across the frame.",
            "A color returns.", "A contour remains.",
            "Nothing is added to the picture.", "The {item} meets the eye.",
            "The {other} keeps its place.", "A line of {motif}.",
            "The visible edge.", "Then a pause.",
        ),
    ),
    "fragmentary": (
        (
            "{item}: {color}.", "A {outline} edge, half a line.",
            "{other}, {other_color}; another mark.", "The frame interrupts itself.",
            "{relation}", "Between edges: {motif}.",
            "A color, a contour, a pause.", "{second_title}, or its outline.",
            "The {item} returns in pieces.", "The {other} holds the next fragment.",
            "{color_title} again, without a story.", "The eye crosses the space between.",
            "A {outline} trace on the page.", "{motif_title} breaks the line.",
            "The picture gathers its fragments.", "The final edge stays open.",
        ),
        (
            "{color_title}. The {item}.", "A line around it: {outline}.",
            "{other_color_title}. The {other}.", "Another shape enters the frame.",
            "{relation}", "A pause, then {second}.",
            "The visible things stay visible.", "The imagined word is {motif}.",
            "{item}: a different edge now.", "{other}: a second interval.",
            "A contour interrupts the color.", "{other_color_title} crosses the page.",
            "The photograph stops here.", "The line continues in fragments.",
            "{motif_title}, briefly.", "The rest belongs to the frame.",
        ),
    ),
}



SINGLE_TEMPLATES = {
    "lyrical": (
        "The {color} {item} enters the frame.",
        "Its {outline} outline holds a line of {motif}.",
        "Around it, the photograph leaves open space.",
        "That space gives {second} a place to begin.",
        "{relation}",
        "The visible edge holds its own direction.",
        "{color_title} stays with the eye a moment longer.",
        "A {outline} mark crosses the imagined page.",
        "The {item} returns as a small measure of {motif}.",
        "Nothing beyond the photograph needs a name.",
        "Only the color keeps its quiet place.",
        "The outline gives {second} a place to rest.",
        "One contour opens and comes into view.",
        "The picture holds it without a story.",
        "{motif_title} remains at the visible edge.",
        "Then the image lets the line go.",
    ),
    "minimal": (
        "{color_title} {item}.", "A {outline} outline.",
        "One shape in the frame.", "Its color has a place.",
        "{relation}", "Space around the edge.",
        "Call that space {motif}.", "Call the next line {second}.",
        "The {item} stays visible.", "The outline stays visible.",
        "No further detail is needed.", "Only {color} in the frame.",
        "Only the {outline} mark.", "A small pause in the image.",
        "{motif_title} at the edge.", "The frame holds still.",
    ),
    "fragmentary": (
        "{item}: {color}.", "A {outline} edge, half a line.",
        "One mark in the photograph.", "The frame interrupts itself.",
        "{relation}", "Around the edge: {motif}.",
        "A color, a contour, a pause.", "{second_title}, or its outline.",
        "The {item} returns in pieces.", "Its color holds the next fragment.",
        "{color_title} again, without a story.", "The eye crosses the open space.",
        "A {outline} trace on the page.", "{motif_title} breaks the line.",
        "The picture gathers its fragments.", "The final edge stays open.",
    ),
}

def validate_poem(poem: Poem, trace: MappingTrace, length: int) -> Poem:
    allowed = {r.rule_id for r in trace.matched_rules} | {t.rule_id for t in trace.tensions}
    if len(poem.lines) != length:
        raise ValueError(f"Expected {length} lines, received {len(poem.lines)}.")
    if not set(poem.used_rule_ids).issubset(allowed):
        raise ValueError("Poem cites an inactive mapping rule.")
    return poem


def _palette_line(analysis: OutfitAnalysis, color_mode: str) -> tuple[str, str] | None:
    colors = [fashion_color_name(entry.hex, color_mode).lower() for entry in analysis.overall.palette]
    colors = colors or analysis.overall.dominant_colors[:3]
    if not colors:
        return None
    primary = colors[0]
    if len(colors) >= 3:
        return primary, f"Across the outfit, {primary} leads; {colors[1]} follows, with {colors[2]} at the edge."
    if len(colors) == 2:
        return primary, f"Across the outfit, {primary} leads and {colors[1]} follows."
    return primary, f"Across the outfit, {primary} holds the visible pieces."


def generate_poem(analysis: OutfitAnalysis, trace: MappingTrace, *, length: int = 12,
                  tone: str = "lyrical", variation: int = 0, color_mode: str = "basic") -> Poem:
    """Render a deterministic poem from selected observations and symbolic rules, locally."""
    if tone not in TEMPLATES or not 8 <= length <= 16 or variation < 0:
        raise ValueError("Invalid poem controls.")
    if not analysis.garments:
        raise ValueError("Select at least one clothing piece to generate a poem.")

    garments = analysis.garments
    item = garments[variation % len(garments)]
    other = garments[(variation + 1) % len(garments)]
    motifs = trace.dominant_motifs or ["shape", "space"]
    motif = motifs[variation % len(motifs)]
    second = motifs[(variation + 1) % len(motifs)] if len(motifs) > 1 else "space"
    color = item.colors[0] if item.colors else "uncolored"
    other_color = other.colors[0] if other.colors else "uncolored"
    outline = item.outline if item.outline != "unknown" else "visible"
    relation = (trace.tensions[variation % len(trace.tensions)].description if trace.tensions else
                f"The {item.name} and {other.name} hold separate places in the frame.")
    slots = {
        "item": item.name, "other": other.name, "color": color, "other_color": other_color,
        "color_title": color.capitalize(), "other_color_title": other_color.capitalize(),
        "outline": outline, "motif": motif, "motif_title": motif.capitalize(),
        "second": second, "second_title": second.capitalize(), "relation": relation,
    }
    if len(garments) == 1:
        slots["relation"] = f"The {item.name} holds one visible place in the frame."
        templates = SINGLE_TEMPLATES[tone]
        # Shift later lines on alternate versions while keeping concrete observations first.
        if variation % 2:
            templates = templates[:5] + templates[9:] + templates[5:9]
    else:
        templates = TEMPLATES[tone][variation % 2]
    lines = [template.format_map(slots) for template in templates[:length]]
    palette = _palette_line(analysis, color_mode)
    if palette is not None:
        lines[6] = palette[1]
    used = list(dict.fromkeys(rule.rule_id for rule in trace.matched_rules
                              if motif in rule.motifs or second in rule.motifs))
    if trace.tensions:
        used.append(trace.tensions[variation % len(trace.tensions)].rule_id)
    poem = Poem(title=f"{motif.capitalize()} in {palette[0] if palette else color}", lines=lines,
                used_rule_ids=list(dict.fromkeys(used)))
    return validate_poem(poem, trace, length)
