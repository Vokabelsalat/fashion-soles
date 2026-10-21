import math

import numpy as np
import pytest

from fashion_soles.color_relations import colour_emotions, color_relations, pair_harmony
from fashion_soles.mapping import map_analysis
from fashion_soles.poetry import generate_poem
from fashion_soles.schemas import Garment, OutfitAnalysis, Overall, PaletteColor


def palette(*items):
    roles = ["primary", "secondary", "accent"]
    return [PaletteColor(role=roles[index], hex=value, share=share) for index, (value, share) in enumerate(items)]


def test_ou_colour_emotions_match_hand_calculation_for_black():
    activity, weight, heat = colour_emotions(np.array([0.0, 0.0, 0.0]))
    assert activity == pytest.approx(-2.1 + .06 * math.sqrt(2500 + 9 + (17 / 1.4) ** 2))
    assert weight == pytest.approx(2.2)
    assert heat == pytest.approx(-.5)


def test_ou_luo_harmony_matches_hand_calculation_for_identical_greys():
    grey = np.array([50.0, 0.0, 0.0])
    expected = ((.04 + .53 * math.tanh(.8)) + (.28 + .54 * math.tanh(-3.88 + .029 * 100))
                + (.14 + .15 * math.tanh(-2)) + 2 * (.5 + .5 * math.tanh(-2)) * (-.08 - .14 * math.sin(math.radians(50)) - .07))
    assert pair_harmony(grey, grey) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("colors, scheme, harmony, moods", [
    ((("#111111", .6), ("#F2F2F2", .4)), "neutral", "harmonious", ["formal", "sharp"]),
    ((("#1F2A44", .6), ("#D8C8A8", .4)), "complementary", "balanced", ["formal", "sharp"]),
    ((("#D62828", .5), ("#2A9D3F", .5)), "contrasting", "clashing", ["dramatic", "bold", "vibrant", "restless"]),
    ((("#8B4A2B", .5), ("#6B6B3A", .3), ("#EDE3CF", .2)), "analogous", "balanced", ["earthy", "classic"]),
    ((("#5E6364", .7), ("#8A8A8A", .3)), "neutral", "harmonious", ["chic", "understated"]),
])
def test_reference_outfits_get_plausible_relations(colors, scheme, harmony, moods):
    relations = color_relations(palette(*colors))
    assert (relations.scheme, relations.harmony, relations.moods) == (scheme, harmony, moods)


def test_single_color_and_empty_palettes():
    single = color_relations(palette(("#C19A6B", 1.0)))
    assert (single.scheme, single.harmony, single.contrast, single.temperature) == ("monochrome", "single", "low", "warm")
    assert color_relations([]) is None


def test_relations_feed_mapping_rules_and_the_poem():
    relations = color_relations(palette(("#D62828", .5), ("#2A9D3F", .5)))
    analysis = OutfitAnalysis(image_suitability="partial", overall=Overall(
        palette=palette(("#D62828", .5), ("#2A9D3F", .5)), color_relations=relations),
        garments=[Garment(id="piece-1", category="dress", name="dress", colors=["red"])])
    trace = map_analysis(analysis)
    assert {"scheme_opposed", "harmony_clashing", "temperature_warm", "mood_dramatic", "mood_restless"} <= {
        rule.rule_id for rule in trace.matched_rules}
    poem = generate_poem(analysis, trace, length=10)
    assert any("dramatic and bold" in line.lower() for line in poem.lines)
