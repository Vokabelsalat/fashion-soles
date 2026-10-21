import itertools

import pytest
import yaml

from fashion_soles import prosody
from fashion_soles.color_relations import color_relations
from fashion_soles.mapping import map_analysis
from fashion_soles.poetry import GRAMMAR, TONES, generate_poem
from fashion_soles.schemas import Garment, OutfitAnalysis, Overall, PaletteColor
from fashion_soles.synonyms import alternatives, wordnet_synonyms

needs_cmudict = pytest.mark.skipif(not prosody.available(), reason="run scripts/download_language_data.py")


def outfit():
    palette = [PaletteColor(role="primary", hex="#1F2A44", share=.6), PaletteColor(role="secondary", hex="#D8C8A8", share=.4)]
    garments = [Garment(id="piece-1", category="coat", name="coat", colors=["blue"], swatch="#1F2A44", outline="vertical"),
                Garment(id="piece-2", category="shoe", name="shoe", colors=["white"], swatch="#D8C8A8")]
    return OutfitAnalysis(image_suitability="partial", garments=garments,
                          overall=Overall(palette=palette, color_relations=color_relations(palette)))


@needs_cmudict
def test_syllables_and_rhymes_come_from_the_dictionary():
    assert [prosody.syllables(word) for word in ("light", "concealment", "eigengrau")] == [1, 3, 3]
    assert prosody.line_syllables("The grey coat waits.") == 4
    assert prosody.rhymes("light", "night") and not prosody.rhymes("light", "light")
    assert not prosody.rhymes("light", "line")


@needs_cmudict
def test_every_rhyme_group_rhymes_throughout():
    for group in yaml.safe_load(GRAMMAR.read_text())["rhymes"]:
        for first, second in itertools.combinations(group, 2):
            assert prosody.rhymes(first, second), (first, second)


@needs_cmudict
def test_haiku_counts_five_seven_five():
    analysis = outfit()
    trace = map_analysis(analysis)
    for tone, variation in itertools.product(TONES, range(5)):
        poem = generate_poem(analysis, trace, tone=tone, variation=variation, form="haiku")
        assert [prosody.line_syllables(line) for line in poem.lines] == [5, 7, 5], poem.lines


@needs_cmudict
def test_couplets_and_closing_rhyme_rhyme():
    analysis = outfit()
    trace = map_analysis(analysis)
    couplets = generate_poem(analysis, trace, length=11, form="couplets")
    assert len(couplets.lines) == 10  # Rounded down to whole pairs.
    for first, second in zip(couplets.lines[::2], couplets.lines[1::2]):
        assert prosody.rhymes(prosody.last_word(first), prosody.last_word(second)), (first, second)
    closing = generate_poem(analysis, trace, length=12, form="closing rhyme")
    assert len(closing.lines) == 12
    assert prosody.rhymes(prosody.last_word(closing.lines[-2]), prosody.last_word(closing.lines[-1]))
    assert closing == generate_poem(analysis, trace, length=12, form="closing rhyme")


def test_synonym_lists_respect_block_and_extra():
    assert alternatives("threshold", {}, {"threshold": ["doorway"]})[:2] == ["threshold", "doorway"]
    assert "globe" not in alternatives("earth", {"earth": ["globe"]}, {})
    assert alternatives("borrowed space", {}, {}) == ["borrowed space"]


@pytest.mark.skipif(not wordnet_synonyms("dusk"), reason="run scripts/download_language_data.py")
def test_repeated_motif_swaps_to_a_synonym():
    from fashion_soles.grammar import Grammar
    grammar = Grammar({"line": ["#motif#, #motif#"]}, 1, {"motif": alternatives("dusk", {}, {})})
    assert grammar.expand("line", []) == "dusk, nightfall"
