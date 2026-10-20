"""Fashionpedia-trained box detection and coarse MediaPipe intersection.

The intersection is a provisional region, not an instance mask. Overlapping
garment boxes may claim the same pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import base64
import io
from PIL import Image, ImageDraw
import numpy as np

MODEL_ID = "valentinafevu/yolos-fashionpedia"
CACHE_DIR = Path(__file__).resolve().parents[1] / "models" / "hf"
ACCESSORY_LABELS = {"glasses", "hat", "headband, head covering, hair accessory", "tie", "glove",
                    "watch", "belt", "leg warmer", "tights, stockings", "sock", "shoe",
                    "bag, wallet", "scarf", "umbrella"}
PART_LABELS = {"hood", "collar", "lapel", "epaulette", "sleeve", "pocket", "neckline", "buckle",
               "zipper", "applique", "bead", "bow", "flower", "fringe", "ribbon", "rivet",
               "ruffle", "sequin", "tassel"}


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]


@dataclass
class PieceProposal:
    label: str
    score: float
    box: tuple[int, int, int, int]
    mask: Image.Image | None
    mask_status: str
    colors: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def _load_model():
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR),
                                                  trust_remote_code=False, use_fast=False)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR),
                                                      trust_remote_code=False, use_safetensors=False)
    model.eval()
    return torch, processor, model


def detect_pieces(image: Image.Image, threshold: float = .45) -> list[Detection]:
    """The first call downloads model weights; no image leaves the machine."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    torch, processor, model = _load_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=torch.tensor([image.size[::-1]]))[0]
    detections = []
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        label = model.config.id2label[int(label_id)].lower()
        if label in PART_LABELS:
            continue
        coords = tuple(round(float(value)) for value in box.tolist())
        detections.append(Detection(label, round(float(score), 3), coords))
    return sorted(detections, key=lambda detection: -detection.score)


def intersect_detection(detection: Detection, clothes_mask: Image.Image | None,
                        accessory_mask: Image.Image | None, image_size: tuple[int, int]) -> PieceProposal:
    x1, y1, x2, y2 = detection.box
    width, height = image_size
    box = (max(0, min(width, x1)), max(0, min(height, y1)),
           max(0, min(width, x2)), max(0, min(height, y2)))
    if box[2] <= box[0] or box[3] <= box[1]:
        return PieceProposal(detection.label, detection.score, box, None, "Invalid box")
    source = accessory_mask if detection.label in ACCESSORY_LABELS else clothes_mask
    if source is None:
        return PieceProposal(detection.label, detection.score, box, None, "Coarse mask unavailable; box only")
    if source.size != image_size:
        source = source.resize(image_size, Image.Resampling.NEAREST)
    rectangle = Image.new("L", image_size, 0)
    ImageDraw.Draw(rectangle).rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=255)
    from PIL import ImageChops
    result = ImageChops.multiply(source.convert("L"), rectangle)
    if not result.getbbox():
        return PieceProposal(detection.label, detection.score, box, None, "No matching MediaPipe pixels in box")
    return PieceProposal(detection.label, detection.score, box, result, "Provisional intersection")


def propose_pieces(image: Image.Image, clothes_mask: Image.Image | None,
                   accessory_mask: Image.Image | None, threshold: float = .45) -> list[PieceProposal]:
    proposals = [intersect_detection(d, clothes_mask, accessory_mask, image.size)
                 for d in detect_pieces(image, threshold)]
    pieces = remove_duplicate_pieces(proposals)
    for piece in pieces:
        if piece.mask is not None:
            piece.colors = extract_piece_colors(image, piece.mask)
    return pieces


def extract_piece_colors(image: Image.Image, mask: Image.Image, count: int = 3) -> tuple[str, ...]:
    """Return dominant RGB colors from masked pixels, in frequency order."""
    if count < 1:
        raise ValueError("Color count must be positive")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    pixels = np.asarray(image.convert("RGB"))
    selected = pixels[np.asarray(mask.convert("L")) > 0]
    if len(selected) == 0:
        return ()
    # Limit quantization work on large uploads while sampling the whole region.
    selected = selected[np.linspace(0, len(selected) - 1, min(len(selected), 12000), dtype=int)]
    strip = Image.fromarray(selected.reshape(1, -1, 3), "RGB")
    quantized = strip.quantize(colors=count, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()
    colors = []
    for _, index in sorted(quantized.getcolors() or [], reverse=True):
        rgb = palette[index * 3:index * 3 + 3]
        value = "#{:02X}{:02X}{:02X}".format(*rgb)
        if value not in colors:
            colors.append(value)
    return tuple(colors)


def mask_thumbnail_data_url(mask: Image.Image | None) -> str | None:
    """Make a compact preview of a piece's actual mask for the results table."""
    if mask is None or not mask.getbbox():
        return None
    thumbnail = mask.convert("L").crop(mask.getbbox())
    thumbnail.thumbnail((128, 128), Image.Resampling.NEAREST)
    output = io.BytesIO()
    thumbnail.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _piece_area(piece: PieceProposal) -> int:
    return int(np.count_nonzero(np.asarray(piece.mask))) if piece.mask is not None else _box_area(piece.box)


def _overlap(a: PieceProposal, b: PieceProposal) -> tuple[float, float]:
    """Return IoU and fraction of the smaller region covered by the other."""
    if a.mask is not None and b.mask is not None:
        pixels_a = np.asarray(a.mask) > 0
        pixels_b = np.asarray(b.mask) > 0
        if pixels_a.shape != pixels_b.shape:
            return 0.0, 0.0
        intersection = int(np.count_nonzero(pixels_a & pixels_b))
        area_a = int(np.count_nonzero(pixels_a))
        area_b = int(np.count_nonzero(pixels_b))
    else:
        x1 = max(a.box[0], b.box[0])
        y1 = max(a.box[1], b.box[1])
        x2 = min(a.box[2], b.box[2])
        y2 = min(a.box[3], b.box[3])
        intersection = _box_area((x1, y1, x2, y2))
        area_a, area_b = _box_area(a.box), _box_area(b.box)
    if not area_a or not area_b:
        return 0.0, 0.0
    return intersection / (area_a + area_b - intersection), intersection / min(area_a, area_b)


def remove_duplicate_pieces(proposals: list[PieceProposal], overlap_threshold: float = .5,
                            similar_area_ratio: float = .6) -> list[PieceProposal]:
    """Suppress near-identical same-label regions, keeping confidence and size in balance.

    Pixel masks are compared when both exist; otherwise boxes are compared.
    The 15% area term lets a slightly larger region beat a near-tied score,
    while a clearly stronger score still wins. This is class-aware NMS.
    """
    if not 0 <= overlap_threshold <= 1 or not 0 <= similar_area_ratio <= 1:
        raise ValueError("Overlap thresholds must be between 0 and 1")
    areas = [_piece_area(piece) for piece in proposals]
    max_by_label = {}
    for piece, area in zip(proposals, areas):
        max_by_label[piece.label] = max(max_by_label.get(piece.label, 0), area)
    ordered = sorted(range(len(proposals)), key=lambda i: (
        proposals[i].score * (.85 + .15 * areas[i] / max(1, max_by_label[proposals[i].label])),
        proposals[i].score, areas[i]), reverse=True)
    kept = []
    for index in ordered:
        candidate = proposals[index]
        duplicate = False
        for chosen in kept:
            if candidate.label != chosen.label:
                continue
            area_ratio = min(areas[index], _piece_area(chosen)) / max(1, max(areas[index], _piece_area(chosen)))
            if area_ratio < similar_area_ratio:
                continue
            iou, smaller_covered = _overlap(candidate, chosen)
            if iou >= overlap_threshold or smaller_covered >= .8:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def draw_boxes(image: Image.Image, proposals: list[PieceProposal]) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for proposal in proposals:
        draw.rectangle(proposal.box, outline=(225, 96, 61), width=3)
        draw.text((proposal.box[0] + 4, proposal.box[1] + 4), proposal.label, fill=(225, 96, 61),
                  stroke_width=2, stroke_fill=(245, 242, 235))
    return result
