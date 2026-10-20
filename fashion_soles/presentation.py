"""Small HTML fragments for the detected-piece list."""
from __future__ import annotations

import re


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
