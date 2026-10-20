"""Turn selected detector proposals into grounded poem observations."""
from __future__ import annotations

from dataclasses import dataclass
import colorsys
import re
from typing import Sequence

from PIL import Image

from .fashion_detector import PieceProposal
from .schemas import Garment, OutfitAnalysis, Overall, Relationship


@dataclass(frozen=True)
class PieceInput:
    proposal: PieceProposal
    label: str | None = None
    colors: tuple[str, ...] | None = None
    outline: str | None = None


def color_name(value: str) -> str:
    """Give an approximate common name to a sampled RGB color."""
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return "unknown"
    red, green, blue = (int(value[index:index + 2], 16) / 255 for index in (1, 3, 5))
    hue, saturation, brightness = colorsys.rgb_to_hsv(red, green, blue)
    degrees = hue * 360
    if brightness < .20:
        return "black"
    if saturation < .20:
        return "white" if brightness >= .55 else "grey"
    if saturation < .32 and brightness > .70:
        return "white"
    if 15 <= degrees < 48 and brightness < .72:
        return "brown"
    if degrees < 15 or degrees >= 345:
        return "red"
    if degrees < 48:
        return "orange"
    if degrees < 70:
        return "yellow"
    if degrees < 170:
        return "green"
    if degrees < 255:
        return "blue"
    if degrees < 345:
        return "purple"
    return "red"


def mask_outline(mask: Image.Image | None) -> str:
    """Describe only the aspect ratio of the visible mask region."""
    bounds = mask.getbbox() if mask is not None else None
    if bounds is None:
        return "unknown"
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    if height > width * 1.5:
        return "vertical"
    if width > height * 1.5:
        return "broad"
    return "balanced"


def analysis_from_pieces(items: Sequence[PieceInput]) -> OutfitAnalysis:
    """Build an analysis from selected pieces; no whole-image vision inference."""
    if not items:
        raise ValueError("Select at least one detected piece first.")
    garments = []
    all_colors = []
    luminances = []
    for index, item in enumerate(items, start=1):
        piece = item.proposal
        label = (item.label or piece.label).strip() or piece.label
        colors = item.colors if item.colors is not None else tuple(color_name(value) for value in piece.colors)
        colors = list(dict.fromkeys(color.strip().lower() for color in colors if color.strip() and color != "unknown"))
        outline = (item.outline or mask_outline(piece.mask)).strip().lower()
        garments.append(Garment(id=f"piece-{index}", category=label.split(",", 1)[0].strip().lower(),
                                name=label, colors=colors, outline=outline, confidence=piece.score))
        all_colors.extend(colors)
        if piece.colors and re.fullmatch(r"#[0-9a-fA-F]{6}", piece.colors[0]):
            rgb = tuple(int(piece.colors[0][i:i + 2], 16) / 255 for i in (1, 3, 5))
            luminances.append(sum(channel * weight for channel, weight in zip(rgb, (.2126, .7152, .0722))))
    contrast = "high" if len(luminances) > 1 and max(luminances) - min(luminances) > .45 else "unknown"
    relationships = []
    if contrast == "high":
        relationships.append(Relationship(kind="contrast", description="The selected pieces have contrasting sampled colors.",
                                          garment_ids=[garment.id for garment in garments]))
    return OutfitAnalysis(image_suitability="partial", warnings=[
        "Detector labels, sampled colors, and mask outlines are provisional observations."],
        overall=Overall(dominant_colors=list(dict.fromkeys(all_colors))[:4], contrast=contrast),
        garments=garments, relationships=relationships)
