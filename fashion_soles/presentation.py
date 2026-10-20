"""Small HTML fragments for the detected-piece list."""
from __future__ import annotations

import base64
import io
import re
import html

from PIL import Image

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


def _data_uri(image: Image.Image, fmt: str) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **({"quality": 88} if fmt == "JPEG" else {"optimize": True}))
    mime = "jpeg" if fmt == "JPEG" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def highlight_preview_html(preview: Image.Image, highlights: dict[int, Image.Image],
                           max_side: int = 1200) -> str:
    """Stack hidden per-piece layers over the preview; hovering list row N reveals layer N.

    The rows are Streamlit containers keyed ``piece_row_N``, so CSS ``:has()``
    connects them to the layers without a custom component.
    """
    scale = min(1.0, max_side / max(preview.size))
    size = (max(1, round(preview.width * scale)), max(1, round(preview.height * scale)))

    def fit(image: Image.Image) -> Image.Image:
        return image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)

    layers = "".join(
        f'<img class="piece-highlight piece-highlight-{index}" src="{_data_uri(fit(layer), "PNG")}" alt="">'
        for index, layer in highlights.items()
    )
    rules = "".join(
        f".stApp:has(.st-key-piece_row_{index}:hover) .piece-highlight-{index} {{opacity:1;}}"
        f".stApp:has(.st-key-piece_row_{index}:hover) .piece-preview-base {{opacity:.35;}}"
        for index in highlights
    )
    return (f"<style>{rules}</style>"
            f'<div class="piece-preview"><img class="piece-preview-base" '
            f'src="{_data_uri(fit(preview.convert("RGB")), "JPEG")}" alt="Analyzed frame with detected pieces">'
            f"{layers}</div>")


def app_header_html() -> str:
    """Pure-CSS logo (a shoe sole carrying a glowing soul) beside the app introduction."""
    return (
        '<div class="app-header">'
        '<div class="app-logo" role="img" aria-label="Fashion Soles logo">'
        '<span class="app-logo-mark"><span class="app-logo-sole"></span><span class="app-logo-soul"></span></span>'
        '<span class="app-logo-word">Fashion<br>Soles</span>'
        '</div>'
        '<p class="app-intro"><strong>You\'ve heard of Fashion Souls, enter the realm of your outfit\'s story.</strong> '
        'Upload a photo or short clip, choose the detected pieces, and read a poem woven from their shapes and colors. '
        '<span class="app-privacy">No footage is saved: everything is analysed locally on this machine. '
        'Faces and identity are of no interest, only the clothes.</span></p>'
        '</div>'
    )
