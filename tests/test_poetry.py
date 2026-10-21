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
    assert poem.lines[3] == "The jacket keeps a little inkwell inception about it."
    assert poem.lines[9] == "Call this shade piano black; the bag wears it without asking."
    # Pure black is just "Black" in both lists, so it adds no imagery; detector alternatives are trimmed.
    assert not any("shade black" in line or "bag, wallet" in line for line in poem.lines)
