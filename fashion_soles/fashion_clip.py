"""Zero-shot pattern and material for detected pieces with FashionCLIP.

FashionCLIP is CLIP fine-tuned on fashion product images and their titles, so
it knows words like "plaid", "paisley", or "denim" without any training here.
Each piece is compared with a few prompt templates per candidate word, and the
probabilities of words that mean the same thing ("leopard print", "zebra
print") are added up.

The crop is the piece's mask bounds with a little context, padded to a square:
CLIP's processor otherwise center-crops, which cuts most of a tall piece away.
The other pieces in the crop are greyed out (the shirt beside a bag, the top
under an open jacket), so their print is not read as this piece's.
Answers below a confidence floor are dropped, and the hand-tuned measures in
texture.py decide instead. On real photos material is the weaker of the two:
an open jacket's crop also shows the top and the trousers underneath.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

MODEL_ID = "patrickjohncyh/fashion-clip"
CACHE_DIR = Path(__file__).resolve().parents[1] / "models" / "hf"

# Prompt word -> pattern name used by the mapping rules and the poem; "none" means a solid piece.
PATTERNS = {"solid": "none", "striped": "striped", "plaid": "plaid", "gingham": "plaid",
            "houndstooth": "houndstooth", "polka dot": "dotted", "floral": "floral print", "paisley": "paisley",
            "leopard print": "animal print", "zebra print": "animal print", "snakeskin": "animal print",
            "camouflage": "camouflage", "tie-dye": "tie-dye", "graphic print": "graphic print",
            "logo": "graphic print", "geometric print": "geometric print"}
MATERIALS = ("denim", "leather", "knit", "wool", "cotton", "silk", "satin", "corduroy", "lace", "sequin",
             "fur", "nylon", "linen", "suede", "canvas", "tweed", "velvet", "mesh")
# Fewer, likelier candidates where the general list invites mistakes (fur sneakers, velvet bags).
CATEGORY_MATERIALS = {
    "shoe": ("leather", "suede", "canvas", "mesh", "knit", "patent leather"),
    "bag": ("leather", "nylon", "canvas", "suede", "straw", "denim", "velvet", "sequin"),
    "belt": ("leather", "suede", "canvas", "chain"),
    "glasses": ("plastic", "metal"),
    "watch": ("metal", "leather", "plastic"),
}
PATTERN_TEMPLATES = ("a photo of a {word} {noun}", "a {noun} with a {word} pattern", "{word} {noun}")
MATERIAL_TEMPLATES = ("a photo of a {word} {noun}", "a {noun} made of {word}", "{word} {noun}")
PATTERN_FLOOR = .5  # Summed probability a pattern needs; solid also needs it to overrule texture.py.
MATERIAL_FLOOR = .6  # Material also needs a clear lead over the runner-up.
MATERIAL_MARGIN = .3
CONTEXT = .1  # Share of the mask bounds added on each side.
NO_PATTERN = {"glasses", "watch"}  # Small hard accessories; a pattern word would only describe glare.
PLURAL = {"shoe": "shoes", "glove": "gloves", "sock": "socks", "glasses": "glasses"}


@dataclass(frozen=True)
class Attributes:
    pattern: str | None  # None when unsure; "none" for a confidently solid piece.
    pattern_score: float
    material: str | None
    material_score: float


@lru_cache(maxsize=1)
def _load_model():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    processor = CLIPProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR), use_fast=False)
    model = CLIPModel.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR), use_safetensors=True)
    model.eval()
    return torch, processor, model


def noun(label: str) -> str:
    """The detector label as a prompt noun: "top, t-shirt, sweatshirt" -> "top", "shoe" -> "shoes"."""
    first = label.split(",", 1)[0].strip().lower()
    return PLURAL.get(first, first)


def _binary(mask: Image.Image, size: tuple[int, int]) -> np.ndarray:
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask.convert("L")) > 0


def other_pieces(masks: Sequence[np.ndarray], index: int) -> np.ndarray:
    """Pixels of the other pieces that would show in this piece's crop.

    Coarse masks overlap: an open jacket's mask also holds the top and the jeans. Pixels another
    piece claims too are hidden as well, like those beside it (the shirt next to a tote bag).
    """
    own = masks[index]
    hidden = np.zeros_like(own)
    for other_index, other in enumerate(masks):
        if other_index != index:
            hidden |= other
    # Keep everything if nothing of its own would remain.
    return hidden if (own & ~hidden).any() else np.zeros_like(own)


def piece_crop(image: Image.Image, mask: Image.Image, hidden: np.ndarray | None = None) -> Image.Image | None:
    """The mask bounds plus some context, centered on a grey square; hidden pixels are greyed out."""
    own = _binary(mask, image.size)
    if not own.any():
        return None
    rows, cols = np.nonzero(own)
    x1, y1, x2, y2 = int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1
    dx, dy = round((x2 - x1) * CONTEXT), round((y2 - y1) * CONTEXT)
    image = image.convert("RGB")
    if hidden is not None and hidden.any():
        image = image.copy()
        image.paste((128, 128, 128), mask=Image.fromarray(hidden.astype(np.uint8) * 255))
    crop = image.crop((max(0, x1 - dx), max(0, y1 - dy),
                                      min(image.width, x2 + dx), min(image.height, y2 + dy)))
    side = max(crop.size)
    square = Image.new("RGB", (side, side), (128, 128, 128))
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return square


@lru_cache(maxsize=64)
def _text_features(words: tuple[str, ...], piece_noun: str, templates: tuple[str, ...]):
    """One embedding per word, averaged over the prompt templates."""
    torch, processor, model = _load_model()
    prompts = [template.format(word=word, noun=piece_noun) for template in templates for word in words]
    with torch.inference_mode():
        features = model.get_text_features(**processor(text=prompts, return_tensors="pt", padding=True))
    features = features / features.norm(dim=-1, keepdim=True)
    features = features.reshape(len(templates), len(words), -1).mean(0)
    return features / features.norm(dim=-1, keepdim=True)


def _probabilities(image_features, words: tuple[str, ...], piece_noun: str,
                   templates: tuple[str, ...]) -> dict[str, float]:
    torch, _, model = _load_model()
    logits = model.logit_scale.exp() * image_features @ _text_features(words, piece_noun, templates).T
    return dict(zip(words, (float(value) for value in logits.softmax(-1)[0])))


def resolve(pattern_probabilities: dict[str, float], material_probabilities: dict[str, float]) -> Attributes:
    """Merge synonymous prompt words and keep only confident answers."""
    patterns: dict[str, float] = {}
    for word, probability in pattern_probabilities.items():
        patterns[PATTERNS[word]] = patterns.get(PATTERNS[word], 0.0) + probability
    pattern, pattern_score = max(patterns.items(), key=lambda item: item[1])
    ranked = sorted(material_probabilities.items(), key=lambda item: -item[1])
    material, material_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    confident = material_score >= MATERIAL_FLOOR and material_score - runner_up >= MATERIAL_MARGIN
    return Attributes(pattern if pattern_score >= PATTERN_FLOOR else None, round(pattern_score, 3),
                      material if confident else None, round(material_score, 3))


def describe_pieces(image: Image.Image, pieces: Sequence[tuple[str, Image.Image | None]]) -> list[Attributes | None]:
    """Attributes for (label, mask) pairs, in one image batch; None per piece the model cannot judge.

    The first call downloads the model weights; no image leaves the machine.
    """
    masks = [_binary(mask, image.size) if mask is not None else np.zeros(image.size[::-1], bool)
             for _, mask in pieces]
    crops = [piece_crop(image, mask, other_pieces(masks, index)) if mask is not None else None
             for index, (_, mask) in enumerate(pieces)]
    usable = [index for index, crop in enumerate(crops) if crop is not None]
    results: list[Attributes | None] = [None] * len(pieces)
    if not usable:
        return results
    torch, processor, model = _load_model()
    with torch.inference_mode():
        features = model.get_image_features(**processor(images=[crops[i] for i in usable], return_tensors="pt"))
        features = features / features.norm(dim=-1, keepdim=True)
        for row, index in enumerate(usable):
            label = pieces[index][0].split(",", 1)[0].strip().lower()
            piece_noun = noun(label)
            materials = CATEGORY_MATERIALS.get(label, MATERIALS)
            image_features = features[row:row + 1]
            patterns = ({"solid": 1.0} if label in NO_PATTERN else
                        _probabilities(image_features, tuple(PATTERNS), piece_noun, PATTERN_TEMPLATES))
            results[index] = resolve(patterns,
                                     _probabilities(image_features, materials, piece_noun, MATERIAL_TEMPLATES))
    return results
