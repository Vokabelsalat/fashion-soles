import numpy as np
import pytest
from PIL import Image, ImageDraw

from fabric_swatches import SWATCHES, finish, mask
from fashion_soles import fashion_clip
from fashion_soles.fashion_clip import Attributes, noun, other_pieces, piece_crop, resolve
from fashion_soles.fashion_detector import PieceProposal
from fashion_soles.from_pieces import PieceInput, analysis_from_pieces, piece_material, piece_pattern
from fashion_soles.mapping import map_analysis
from fashion_soles.poetry import generate_poem
from fashion_soles.texture import Texture


def test_synonyms_add_up_and_unsure_answers_are_dropped():
    animal = resolve({"leopard print": .3, "zebra print": .25, "solid": .45}, {"denim": .9, "leather": .1})
    assert animal.pattern == "animal print" and animal.material == "denim"
    unsure = resolve({"striped": .4, "solid": .35, "plaid": .25}, {"denim": .55, "leather": .45})
    assert unsure.pattern is None and unsure.material is None
    close = resolve({"solid": 1.0}, {"denim": .62, "leather": .38})
    assert close.pattern == "none" and close.material is None  # Confident but without a clear lead.


def test_nouns_read_as_prompts():
    assert noun("top, t-shirt, sweatshirt") == "top"
    assert noun("shoe") == "shoes"


def test_crop_is_square_with_context():
    image = Image.new("RGB", (100, 200), "red")
    region = Image.new("L", (100, 200), 0)
    ImageDraw.Draw(region).rectangle((40, 20, 59, 179), fill=255)
    crop = piece_crop(image, region)
    assert crop.width == crop.height == 192  # 160 tall plus 10% context above and below.
    assert piece_crop(image, Image.new("L", (100, 200), 0)) is None


def test_other_pieces_are_hidden_from_the_crop():
    jacket = np.zeros((20, 20), bool); jacket[2:18, 2:18] = True
    top = np.zeros((20, 20), bool); top[6:14, 6:14] = True
    shoe = np.zeros((20, 20), bool); shoe[18:, :4] = True
    hidden = other_pieces([jacket, top, shoe], 0)
    assert hidden[8, 8] and hidden[19, 0] and not hidden[3, 3]
    assert not other_pieces([jacket, top, shoe], 1).any()  # Nothing of the top would remain.


def proposal(attributes, texture=None):
    return PieceProposal("pants", .9, (20, 20, 280, 380), mask(), "ok", ("#3A5A8C",), texture, attributes)


def test_fashion_clip_wins_and_texture_decides_when_it_is_unsure():
    stripes = Texture("textured", "striped", .1, .5, .1, 3.0, 2, .4, .2, 0)
    assert piece_pattern(proposal(Attributes("plaid", .8, None, .3), stripes)) == "plaid"
    assert piece_pattern(proposal(Attributes(None, .3, None, .3), stripes)) == "striped"
    assert piece_pattern(proposal(Attributes("none", .9, None, .3), stripes)) == "textured"
    assert piece_pattern(proposal(None, stripes)) == "striped"
    assert piece_material(proposal(Attributes("none", .9, "denim", .95))) == "denim"
    assert piece_material(proposal(None)) == "unknown"


def test_material_reaches_analysis_mapping_and_poem():
    piece = proposal(Attributes("graphic print", .9, "denim", .95))
    analysis = analysis_from_pieces([PieceInput(piece)])
    assert analysis.garments[0].material == "denim"
    trace = map_analysis(analysis)
    assert {"material_denim", "pattern_graphic"} <= {rule.rule_id for rule in trace.matched_rules}
    assert "printed denim pants" in " ".join(generate_poem(analysis, trace, length=8).lines)
    edited = analysis_from_pieces([PieceInput(piece, material="Corduroy")])
    assert edited.garments[0].material == "corduroy"
    assert "material_knit" in {rule.rule_id for rule in map_analysis(edited).matched_rules}


def _model_cached() -> bool:
    return any(fashion_clip.CACHE_DIR.glob("models--patrickjohncyh--fashion-clip/snapshots/*/model.safetensors"))


EXPECTED = {"plain": "none", "stripes horizontal": "striped", "stripes diagonal": "striped",
            "pinstripe": "striped", "gingham": "plaid", "polka dots": "dotted", "floral": "floral print",
            "geometric": "geometric print", "graphic": "graphic print", "knit ribs": "none"}


@pytest.mark.fashion_clip
@pytest.mark.skipif(not _model_cached(), reason="FashionCLIP weights are not downloaded")
def test_model_reads_synthetic_fabrics():
    found = {}
    for seed, name in enumerate(SWATCHES):
        found[name] = fashion_clip.describe_pieces(finish(SWATCHES[name], seed), [("shirt", mask())])[0].pattern
    assert found == EXPECTED


@pytest.mark.fashion_clip
@pytest.mark.skipif(not _model_cached(), reason="FashionCLIP weights are not downloaded")
def test_labelled_photos():
    """Rows marked "pass" in tests/fixtures/photos/labels.csv must keep passing; see scripts/evaluate_photos.py."""
    from scripts.evaluate_photos import evaluate
    regressions = [f"{r['file']} {r['piece']}: {r['found']} / {r['found_pattern']} / {r['found_material']}"
                   for r in evaluate() if r["status"] == "pass" and not (r["pattern_ok"] and r["material_ok"])]
    assert not regressions


def test_unsure_model_keeps_only_regular_repeats_from_texture():
    print_guess = Texture("busy", "geometric print", .15, .2, .1, 2.0, 3, .3, .2, 0)
    assert piece_pattern(proposal(Attributes(None, .4, None, .3), print_guess)) == "busy"
    assert piece_pattern(proposal(None, print_guess)) == "geometric print"  # Without the model, texture decides.
