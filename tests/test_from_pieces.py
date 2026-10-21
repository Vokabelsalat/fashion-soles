from PIL import Image, ImageDraw

from fashion_soles.fashion_detector import PieceProposal
from fashion_soles.from_pieces import PieceInput, analysis_from_pieces, color_name, mask_outline
from fashion_soles.mapping import map_analysis
from fashion_soles.poetry import generate_poem
from fashion_soles.palette import outfit_palette


def proposal(label, color, mask):
    return PieceProposal(label, .91, mask.getbbox(), mask, "Provisional intersection", (color,))


def test_selected_pieces_drive_color_category_and_outline_rules():
    vertical = Image.new("L", (40, 40), 0)
    broad = Image.new("L", (40, 40), 0)
    ImageDraw.Draw(vertical).rectangle((2, 2, 10, 32), fill=255)
    ImageDraw.Draw(broad).rectangle((12, 20, 37, 28), fill=255)
    jacket = proposal("jacket", "#FF0000", vertical)
    shoe = proposal("shoe", "#0000FF", broad)
    analysis = analysis_from_pieces([PieceInput(jacket), PieceInput(shoe)])
    assert [garment.colors for garment in analysis.garments] == [["red"], ["blue"]]
    assert [garment.outline for garment in analysis.garments] == ["vertical", "broad"]
    ids = {rule.rule_id for rule in map_analysis(analysis).matched_rules}
    assert {"color_red", "color_blue", "category_outerwear", "category_footwear",
            "outline_vertical", "outline_broad"} <= ids
    lines = generate_poem(analysis, map_analysis(analysis)).lines
    assert any("red" in line for line in lines)
    assert any("vertical" in line for line in lines)
    single = analysis_from_pieces([PieceInput(shoe)])
    assert "category_outerwear" not in {rule.rule_id for rule in map_analysis(single).matched_rules}


def test_edits_override_detector_observations():
    mask = Image.new("L", (20, 20), 255)
    item = PieceInput(proposal("bag, wallet", "#FFFFFF", mask), label="coat",
                      colors=("black",), outline="broad")
    analysis = analysis_from_pieces([item])
    garment = analysis.garments[0]
    assert (garment.category, garment.name, garment.colors, garment.outline) == (
        "coat", "coat", ["black"], "broad")
    ids = {rule.rule_id for rule in map_analysis(analysis).matched_rules}
    assert {"category_outerwear", "color_dark", "outline_broad"} <= ids


def test_color_names_and_mask_outline_are_approximate_observations():
    assert [color_name(value) for value in ("#000000", "#FFFFFF", "#00FF00", "#FFFF00")] == [
        "black", "white", "green", "yellow"]
    assert color_name("not a hex") == "unknown"
    assert color_name("#9C8E8A") == "white"  # Lit shirt in the reference photo.
    assert color_name("#C8BDB3") == "white"  # Warm white shoes in the reference photo.
    assert color_name("#5E6364") == "grey"   # The jeans remain grey.
    assert mask_outline(None) == "unknown"


def test_local_poem_uses_selected_observations_and_changes_with_edits():
    mask = Image.new("L", (20, 20), 255)
    piece = proposal("jacket", "#FF0000", mask)
    first = analysis_from_pieces([PieceInput(piece)])
    revised = analysis_from_pieces([PieceInput(piece, label="coat", colors=("blue",), outline="vertical")])
    first_poem = generate_poem(first, map_analysis(first))
    revised_poem = generate_poem(revised, map_analysis(revised))
    assert "jacket" in " ".join(first_poem.lines)
    assert "red" in " ".join(first_poem.lines)
    assert "coat" in " ".join(revised_poem.lines)
    assert "blue" in " ".join(revised_poem.lines)
    assert "vertical" in " ".join(revised_poem.lines)
    assert first_poem.lines != revised_poem.lines
    assert "Two visible shapes" not in " ".join(first_poem.lines)
    assert "Nearby" not in " ".join(first_poem.lines)
    assert generate_poem(first, map_analysis(first), variation=1).lines != first_poem.lines


def in_order(line, *words):
    """True when all words appear in the line in this order (the palette line, whatever its wording)."""
    positions = [line.lower().find(word) for word in words]
    return -1 not in positions and positions == sorted(positions)


def test_outfit_palette_uses_masked_pixel_frequency_and_distant_accent():
    image = Image.new("RGB", (100, 100), "#555555")
    ImageDraw.Draw(image).rectangle((60, 0, 89, 99), fill="#FFFFFF")
    ImageDraw.Draw(image).rectangle((90, 0, 99, 99), fill="#E52222")
    mask = Image.new("L", image.size, 255)
    item = PieceInput(proposal("coat", "#555555", mask), image=image)
    palette = outfit_palette([item])
    assert [(entry.role, entry.hex) for entry in palette] == [
        ("primary", "#555555"), ("secondary", "#FFFFFF"), ("accent", "#E52222")]
    assert [round(entry.share, 2) for entry in palette] == [.6, .3, .1]
    analysis = analysis_from_pieces([item])
    assert analysis.overall.palette == palette
    # Each poem's palette line uses one list, picked by the seed; across variations both appear.
    poems = [generate_poem(analysis, map_analysis(analysis), length=8, variation=n).lines for n in range(8)]
    assert any(in_order(line, "dim grey", "white", "red") for lines in poems for line in lines)
    assert any(in_order(line, "stone cold", "white", "lust") for lines in poems for line in lines)


def test_outfit_palette_ignores_unselected_background():
    image = Image.new("RGB", (100, 100), "#0000FF")
    ImageDraw.Draw(image).rectangle((0, 0, 49, 49), fill="#FF0000")
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((0, 0, 49, 49), fill=255)
    item = PieceInput(proposal("shirt", "#FF0000", mask), image=image)
    assert [(entry.role, entry.hex) for entry in outfit_palette([item])] == [("primary", "#FF0000")]


def test_descriptive_color_names_keep_their_mapping_family():
    from fashion_soles.color_names import fashion_color_name

    assert fashion_color_name("#800020") == "Maroon"
    assert fashion_color_name("#1F2A44", "basic") == "Midnight Blue"
    assert fashion_color_name("#800020", "whimsical") == "Oxblood"
    assert fashion_color_name("#000000", "whimsical") == "Black"
    assert fashion_color_name("nope") == "unknown"
    mask = Image.new("L", (20, 20), 255)
    garment = analysis_from_pieces([PieceInput(proposal("coat", "#800020", mask))]).garments[0]
    assert garment.colors == ["maroon", "oxblood", "red"]  # Both lists, then the family for the mapping rules.
    typed = analysis_from_pieces([PieceInput(proposal("coat", "#800020", mask), colors=("blue",))]).garments[0]
    assert typed.colors == ["blue"]
