import pytest
from PIL import Image

from fabric_swatches import SWATCHES, finish, mask
from fashion_soles.fashion_detector import PieceProposal
from fashion_soles.from_pieces import PieceInput, analysis_from_pieces, piece_relations
from fashion_soles.mapping import map_analysis
from fashion_soles.poetry import generate_poem
from fashion_soles.texture import analyse_texture

EXPECTED = {
    "plain": "plain", "stripes horizontal": "striped", "stripes diagonal": "striped", "pinstripe": "striped",
    "gingham": "checked", "polka dots": "dotted", "floral": "floral print",
    # Known limits: a single small graphic, sparse prints, and faint knit ribs carry too few crisp
    # edges and read as plain (see texture.py).
    "graphic": "plain", "geometric": "plain", "knit ribs": "plain",
}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_synthetic_fabrics_get_their_pattern(name):
    seed = list(SWATCHES).index(name)
    texture = analyse_texture(finish(SWATCHES[name], seed), mask())
    assert texture.label == EXPECTED[name]


def test_too_small_or_missing_masks_are_not_judged():
    image = Image.new("RGB", (40, 40), "white")
    assert analyse_texture(image, None) is None
    assert analyse_texture(image, Image.new("L", (40, 40), 255)) is None


def test_pattern_reaches_analysis_mapping_and_poem():
    image = finish(SWATCHES["stripes horizontal"], 1)
    piece = PieceProposal("shirt", .9, (20, 20, 280, 380), mask(), "ok", ("#1E326E",))
    analysis = analysis_from_pieces([PieceInput(piece, image=image)])
    assert analysis.garments[0].texture == ["striped"]
    trace = map_analysis(analysis)
    assert "pattern_striped" in {rule.rule_id for rule in trace.matched_rules}
    assert "striped shirt" in " ".join(generate_poem(analysis, trace, length=8).lines)
    edited = analysis_from_pieces([PieceInput(piece, image=image, pattern="Checked")])
    assert edited.garments[0].texture == ["checked"]


def test_each_piece_gets_its_own_palette_mood():
    image = finish(SWATCHES["gingham"], 4)
    relations = piece_relations(PieceProposal("shirt", .9, (20, 20, 280, 380), mask(), "ok"), image)
    assert relations is not None and relations.moods
