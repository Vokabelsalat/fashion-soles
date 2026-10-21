import pytest

from fashion_soles.mapping import map_analysis
from fashion_soles.pipeline import demo_analysis
from fashion_soles.poetry import generate_poem, validate_poem
from fashion_soles.schemas import Poem


def test_local_templates_are_repeatable_and_all_controls_work(monkeypatch):
    # The generator must work even with a bogus API key and no imported API client.
    monkeypatch.setenv("OPENAI_API_KEY", "unusable")
    analysis = demo_analysis()
    trace = map_analysis(analysis)
    for tone in ("lyrical", "minimal", "fragmentary"):
        for length in (8, 12, 16):
            poem = generate_poem(analysis, trace, tone=tone, length=length)
            assert len(poem.lines) == length
            assert poem == generate_poem(analysis, trace, tone=tone, length=length)
            assert validate_poem(poem, trace, length) == poem
    assert generate_poem(analysis, trace, variation=0).lines != generate_poem(analysis, trace, variation=1).lines
    assert generate_poem(analysis, trace, tone="minimal").lines != generate_poem(analysis, trace, tone="lyrical").lines


def test_invalid_controls_and_inactive_rule_are_rejected():
    analysis = demo_analysis()
    trace = map_analysis(analysis)
    with pytest.raises(ValueError):
        generate_poem(analysis, trace, length=7)
    with pytest.raises(ValueError):
        generate_poem(analysis, trace, tone="mysterious")
    with pytest.raises(ValueError):
        generate_poem(analysis, trace, variation=-1)
    with pytest.raises(ValueError):
        validate_poem(Poem(title="A line", lines=["line"] * 8, used_rule_ids=["invented"]), trace, 8)


def test_whimsical_color_names_become_imagery_lines():
    from fashion_soles.mapping import map_analysis
    from fashion_soles.poetry import generate_poem
    from fashion_soles.schemas import Garment, OutfitAnalysis, Overall

    garments = [Garment(id="piece-1", category="jacket", name="jacket", colors=["black"], swatch="#1A1C22"),
                Garment(id="piece-2", category="bag", name="bag, wallet", colors=["black"], swatch="#15161B"),
                Garment(id="piece-3", category="shoe", name="shoe", colors=["black"], swatch="#000000")]
    analysis = OutfitAnalysis(image_suitability="partial", overall=Overall(), garments=garments)
    poem = generate_poem(analysis, map_analysis(analysis), length=12, variation=0)
    assert any("inkwell inception" in line.lower() and "jacket" in line for line in poem.lines)
    assert any("piano black" in line.lower() and "bag" in line for line in poem.lines)
    # Pure black is just "Black" in both lists, so it adds no imagery; detector alternatives are trimmed.
    assert not any("shade black" in line or "bag, wallet" in line for line in poem.lines)


def test_grammar_poems_are_seeded_complete_and_varied():
    import itertools
    from fashion_soles.color_relations import color_relations
    from fashion_soles.mapping import map_analysis
    from fashion_soles.poetry import TONES, generate_poem
    from fashion_soles.schemas import Garment, OutfitAnalysis, Overall, PaletteColor

    palette = [PaletteColor(role="primary", hex="#8B4A2B", share=.6), PaletteColor(role="secondary", hex="#EDE3CF", share=.4)]
    pieces = [Garment(id="piece-1", category="coat", name="coat", colors=["brown"], swatch="#8B4A2B", texture=["striped"]),
              Garment(id="piece-2", category="shoe", name="shoe", colors=["white"], swatch="#EDE3CF")]
    outfits = [
        OutfitAnalysis(image_suitability="partial", garments=pieces,
                       overall=Overall(palette=palette, color_relations=color_relations(palette))),
        OutfitAnalysis(image_suitability="partial", garments=pieces[:1], overall=Overall()),  # No palette or mood.
    ]
    for analysis, tone, length in itertools.product(outfits, TONES, range(8, 17)):
        trace = map_analysis(analysis)
        poem = generate_poem(analysis, trace, length=length, tone=tone)
        assert len(poem.lines) == length and all(line.strip() for line in poem.lines)
        assert "#" not in poem.title + " ".join(poem.lines), poem.lines
        assert poem == generate_poem(analysis, trace, length=length, tone=tone)  # Same outfit, same poem.
    analysis = outfits[0]
    trace = map_analysis(analysis)
    variants = {tuple(generate_poem(analysis, trace, length=12, variation=n).lines) for n in range(6)}
    assert len(variants) == 6
    assert "striped coat" in " ".join(generate_poem(analysis, trace, length=16).lines)


def test_grammar_narrows_by_keys_and_applies_modifiers():
    from fashion_soles.grammar import Grammar

    rules = {"line": {"lyrical": {"rupture": ["#noun.a.cap# breaks."], "any": ["Calm."]}, "any": ["Plain."]},
             "noun": ["ember"], "pair": ["one", "two"]}
    grammar = Grammar(rules, seed=1)
    assert grammar.expand("line", ["lyrical", "rupture"]) == "An ember breaks."
    assert grammar.expand("line", ["lyrical", "memory"]) == "Calm."
    assert grammar.expand("line", ["minimal"]) == "Plain."
    assert {grammar.expand("pair", []), grammar.expand("pair", [])} == {"one", "two"}  # No reuse before exhaustion.
    assert Grammar(rules, 1, {"lead": "dim grey"}).fill("#lead.title#", []) == "Dim Grey"
