"""Turn selected detector proposals into grounded poem observations."""
from __future__ import annotations

from dataclasses import dataclass
import colorsys
import re
from typing import Sequence

from PIL import Image

from .color_names import fashion_color_name
from .color_relations import color_relations
from .fashion_detector import PieceProposal
from .palette import outfit_palette
from .schemas import ColorRelations, Garment, OutfitAnalysis, Overall, Relationship
from .texture import analyse_texture


@dataclass(frozen=True)
class PieceInput:
    proposal: PieceProposal
    label: str | None = None
    colors: tuple[str, ...] | None = None
    outline: str | None = None
    image: Image.Image | None = None
    pattern: str | None = None


def piece_pattern(piece: PieceProposal, image: Image.Image | None = None) -> str:
    """The piece's pattern, or its surface when no pattern was found; analysed now for older proposals."""
    texture = piece.texture
    if texture is None and image is not None:
        texture = analyse_texture(image, piece.mask)
    return texture.label if texture is not None else "unknown"


def piece_relations(piece: PieceProposal, image: Image.Image | None) -> ColorRelations | None:
    """Colour relations and mood of one piece's own palette."""
    return color_relations(outfit_palette([PieceInput(piece, image=image)]))


COLOR_FAMILIES = {"black", "grey", "white", "brown", "red", "orange", "yellow", "green", "blue", "purple"}


def color_name(value: str) -> str:
    """Give the basic color family that the mapping rules key on."""
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
        # Both lists by default ("maroon", "oxblood"); a typed color name replaces them.
        colors = item.colors if item.colors is not None else tuple(
            fashion_color_name(value, mode) for value in piece.colors[:1] for mode in ("basic", "whimsical"))
        colors = [color.strip().lower() for color in colors if color.strip() and color != "unknown"]
        # Descriptive names rarely match a mapping rule, so add the sampled family unless one was typed.
        if piece.colors and not COLOR_FAMILIES & set(colors):
            colors.append(color_name(piece.colors[0]))
        colors = list(dict.fromkeys(color for color in colors if color != "unknown"))
        outline = (item.outline or mask_outline(piece.mask)).strip().lower()
        pattern = (item.pattern or piece_pattern(piece, item.image)).strip().lower()
        garments.append(Garment(id=f"piece-{index}", category=label.split(",", 1)[0].strip().lower(),
                                name=label, colors=colors, outline=outline, confidence=piece.score,
                                swatch=piece.colors[0] if piece.colors and re.fullmatch(
                                    r"#[0-9a-fA-F]{6}", piece.colors[0]) else None,
                                texture=[pattern] if pattern and pattern != "unknown" else []))
        all_colors.extend(colors)
        if piece.colors and re.fullmatch(r"#[0-9a-fA-F]{6}", piece.colors[0]):
            rgb = tuple(int(piece.colors[0][i:i + 2], 16) / 255 for i in (1, 3, 5))
            luminances.append(sum(channel * weight for channel, weight in zip(rgb, (.2126, .7152, .0722))))
    contrast = "high" if len(luminances) > 1 and max(luminances) - min(luminances) > .45 else "unknown"
    relationships = []
    if contrast == "high":
        relationships.append(Relationship(kind="contrast", description="The selected pieces have contrasting sampled colors.",
                                          garment_ids=[garment.id for garment in garments]))
    palette = outfit_palette(items)
    palette_names = list(dict.fromkeys(color_name(color.hex) for color in palette))
    return OutfitAnalysis(image_suitability="partial", warnings=[
        "Detector labels, sampled colors, and mask outlines are provisional observations."],
        overall=Overall(dominant_colors=palette_names or list(dict.fromkeys(all_colors))[:4],
                        palette=palette, contrast=contrast, color_relations=color_relations(palette)),
        garments=garments, relationships=relationships)
