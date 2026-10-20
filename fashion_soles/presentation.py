"""Small HTML fragments for the detected-piece list."""
from __future__ import annotations

import re
import html

from .from_pieces import color_name
from .schemas import PaletteColor


def color_grid_html(colors: tuple[str, ...]) -> str:
    """Show the single dominant color as a validated swatch."""
    valid = [color for color in colors[:1] if re.fullmatch(r"#[0-9A-Fa-f]{6}", color)]
    if not valid:
        return "Unavailable"
    swatches = "".join(
        f'<div class="piece-swatch" style="background-color:{color}" title="{color}" aria-label="{color}"></div>'
        for color in valid
    )
    return f'<div class="piece-color-grid">{swatches}</div>'


def outfit_palette_html(colors: list[PaletteColor]) -> str:
    """Display the selected outfit's pixel palette in a compact row."""
    items = "".join(
        '<div class="outfit-palette-item">'
        f'<span class="outfit-palette-swatch" style="background-color:{color.hex}"></span>'
        f'<span>{html.escape(color.role.capitalize())}: {html.escape(color_name(color.hex))}</span>'
        '</div>'
        for color in colors
    )
    return f'<div class="outfit-palette">{items}</div>'
