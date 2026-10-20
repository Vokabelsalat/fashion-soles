from PIL import Image, ImageDraw
import numpy as np
from fashion_soles.fashion_detector import (Detection, PieceProposal, available_pieces, extract_piece_colors,
    intersect_detection, piece_thumbnail, preview_selected_pieces, propose_pieces,
    remove_duplicate_pieces)


def test_clothes_intersect_box_only():
    clothes = Image.new("L", (20, 20), 0)
    ImageDraw.Draw(clothes).rectangle((4, 4, 15, 15), fill=255)
    proposal = intersect_detection(Detection("jacket", .8, (0, 0, 10, 10)), clothes, None, (20, 20))
    assert proposal.mask_status == "Provisional intersection"
    assert proposal.mask.getbbox() == (4, 4, 10, 10)
    assert proposal.mask.getpixel((15, 15)) == 0


def test_accessory_uses_other_category_and_missing_mask_is_honest():
    clothes = Image.new("L", (20, 20), 255)
    accessory = Image.new("L", (20, 20), 0)
    ImageDraw.Draw(accessory).rectangle((2, 2, 5, 5), fill=255)
    proposal = intersect_detection(Detection("shoe", .7, (0, 0, 10, 10)), clothes, accessory, (20, 20))
    assert proposal.mask.getbbox() == (2, 2, 6, 6)
    missing = intersect_detection(Detection("jacket", .7, (0, 0, 10, 10)), None, None, (20, 20))
    assert missing.mask is None
    assert missing.mask_status == "Coarse mask unavailable; box only"


def piece(label, score, box, mask=None):
    return PieceProposal(label, score, box, mask, "Provisional intersection" if mask else "Box only")


def test_duplicate_bag_boxes_keep_confident_large_region():
    proposals = [piece("bag, wallet", .89, (0, 0, 9, 9)),
                 piece("bag, wallet", .87, (0, 0, 10, 10)),
                 piece("shoe", .8, (0, 0, 10, 10))]
    kept = remove_duplicate_pieces(proposals)
    assert len(kept) == 2
    assert next(p for p in kept if p.label == "bag, wallet").box == (0, 0, 10, 10)


def test_clearer_score_wins_and_separate_same_label_items_remain():
    proposals = [piece("bag, wallet", .96, (0, 0, 9, 9)),
                 piece("bag, wallet", .75, (0, 0, 10, 10)),
                 piece("bag, wallet", .81, (30, 0, 40, 10))]
    kept = remove_duplicate_pieces(proposals)
    assert {p.box for p in kept} == {(0, 0, 9, 9), (30, 0, 40, 10)}


def test_pixel_overlap_controls_duplicates_when_masks_exist():
    first = Image.new("L", (30, 30), 0)
    second = Image.new("L", (30, 30), 0)
    ImageDraw.Draw(first).rectangle((2, 2, 11, 11), fill=255)
    ImageDraw.Draw(second).rectangle((3, 3, 12, 12), fill=255)
    kept = remove_duplicate_pieces([piece("jacket", .9, (0, 0, 20, 20), first),
                                    piece("jacket", .8, (0, 0, 20, 20), second)])
    assert len(kept) == 1
    assert kept[0].score == .9


def test_overlapping_boxes_with_distinct_masks_are_not_duplicates():
    first = Image.new("L", (30, 30), 0)
    second = Image.new("L", (30, 30), 0)
    ImageDraw.Draw(first).rectangle((0, 0, 7, 7), fill=255)
    ImageDraw.Draw(second).rectangle((12, 12, 19, 19), fill=255)
    kept = remove_duplicate_pieces([piece("shoe", .9, (0, 0, 20, 20), first),
                                    piece("shoe", .85, (0, 0, 20, 20), second)])
    assert len(kept) == 2


def test_three_colors_come_only_from_masked_pixels():
    image = Image.new("RGB", (10, 10), (255, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 3, 9), fill=(0, 0, 255))
    draw.rectangle((4, 0, 6, 9), fill=(0, 255, 0))
    draw.rectangle((7, 0, 8, 9), fill=(255, 255, 0))
    mask = Image.new("L", image.size, 255)
    ImageDraw.Draw(mask).rectangle((9, 0, 9, 9), fill=0)
    assert extract_piece_colors(image, mask) == ("#0000FF", "#00FF00", "#FFFF00")
    assert extract_piece_colors(image, Image.new("L", image.size, 0)) == ()


def test_proposals_include_colors_and_mask_thumbnail(monkeypatch):
    monkeypatch.setattr("fashion_soles.fashion_detector.detect_pieces",
                        lambda image, threshold: [Detection("jacket", .9, (0, 0, 8, 8))])
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    mask = Image.new("L", (8, 8), 255)
    piece = propose_pieces(image, mask, None)[0]
    assert piece.colors == ("#0A141E",)
    thumbnail = piece_thumbnail(image, piece.mask)
    assert thumbnail.size == (112, 112)
    assert thumbnail.getpixel((0, 0)) == (0, 0, 0, 0)
    assert thumbnail.getpixel((56, 56)) == (10, 20, 30, 255)
    assert piece_thumbnail(image, None) is None


def test_thumbnail_keeps_photo_colors_only_inside_mask_with_padding():
    image = Image.new("RGB", (20, 20), "blue")
    ImageDraw.Draw(image).rectangle((5, 5, 14, 14), fill="red")
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((5, 5, 14, 14), fill=255)
    ImageDraw.Draw(mask).rectangle((9, 9, 10, 10), fill=0)
    thumbnail = piece_thumbnail(image, mask, size=60, padding=6)
    assert thumbnail.getpixel((0, 0))[3] == 0
    assert thumbnail.getpixel((10, 10)) == (255, 0, 0, 255)
    assert thumbnail.getpixel((30, 30))[3] == 0
    pixels = np.asarray(thumbnail).reshape(-1, 4)
    assert np.all(pixels[pixels[:, 3] == 255, 2] == 0)


def test_maskless_detections_are_filtered_from_piece_list(monkeypatch):
    monkeypatch.setattr("fashion_soles.fashion_detector.detect_pieces",
                        lambda image, threshold: [Detection("jacket", .9, (0, 0, 8, 8))])
    assert propose_pieces(Image.new("RGB", (8, 8), "red"), None, None) == []
    empty_mask = Image.new("L", (8, 8), 0)
    assert available_pieces([piece("jacket", .9, (0, 0, 8, 8), empty_mask),
                             piece("shoe", .8, (0, 0, 8, 8))]) == []


def test_valid_piece_remains_when_another_detection_has_no_mask(monkeypatch):
    monkeypatch.setattr("fashion_soles.fashion_detector.detect_pieces",
                        lambda image, threshold: [Detection("jacket", .9, (0, 0, 8, 8)),
                                                  Detection("shoe", .8, (0, 0, 8, 8))])
    pieces = propose_pieces(Image.new("RGB", (8, 8), "red"), Image.new("L", (8, 8), 255), None)
    assert [piece.label for piece in pieces] == ["jacket"]
    assert pieces[0].colors == ("#FF0000",)


def test_preview_contains_only_selected_piece_masks_and_boxes():
    image = Image.new("RGB", (40, 20), (100, 100, 100))
    first_mask = Image.new("L", image.size, 0)
    second_mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(first_mask).rectangle((5, 5, 14, 14), fill=255)
    ImageDraw.Draw(second_mask).rectangle((25, 5, 34, 14), fill=255)
    first = piece("jacket", .9, (3, 3, 17, 17), first_mask)
    second = piece("shoe", .8, (23, 3, 37, 17), second_mask)
    preview = preview_selected_pieces(image, [first])
    assert preview.getpixel((10, 10)) != image.getpixel((10, 10))
    assert preview.getpixel((30, 10)) == image.getpixel((30, 10))
    assert preview_selected_pieces(image, []).tobytes() == image.tobytes()
