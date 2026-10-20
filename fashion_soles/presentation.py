"""Small HTML fragments for the detected-piece list."""
from __future__ import annotations

import base64
import io
import re
import html

from PIL import Image

from .color_names import fashion_color_name
from .from_pieces import color_name
from .schemas import PaletteColor


def color_grid_html(colors: tuple[str, ...]) -> str:
    """Show the single dominant color as a validated swatch."""
    valid = [color for color in colors[:1]
             if re.fullmatch(r"#[0-9A-Fa-f]{6}", color)]
    if not valid:
        return "Unavailable"
    swatches = "".join(
        f'<div class="piece-swatch" style="background-color:{color}" title="{color}" aria-label="{color}"></div>'
        for color in valid
    )
    return f'<div class="piece-color-grid">{swatches}</div>'


def outfit_palette_html(colors: list[PaletteColor], color_mode: str = "basic") -> str:
    """Display the selected outfit's pixel palette in a compact row."""
    items = "".join(
        f'<div class="outfit-palette-item" title="{color.hex} · {html.escape(color_name(color.hex))}">'
        f'<span class="outfit-palette-swatch" style="background-color:{color.hex}"></span>'
        f'<span>{html.escape(color.role.capitalize())}: {html.escape(fashion_color_name(color.hex, color_mode))}</span>'
        '</div>'
        for color in colors
    )
    return f'<div class="outfit-palette">{items}</div>'


def image_data_uri(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **
               ({"quality": 88} if fmt == "JPEG" else {"optimize": True}))
    mime = "jpeg" if fmt == "JPEG" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def highlight_rules_css(indices) -> str:
    """CSS that reveals ``.piece-highlight-N`` and dims ``.piece-preview-base`` while list row N is hovered."""
    return "".join(
        f".stApp:has(.st-key-piece_row_{index}:hover) .piece-highlight-{index} {{opacity:1;}}"
        f".stApp:has(.st-key-piece_row_{index}:hover) .piece-preview-base {{opacity:.35;}}"
        for index in indices
    )


def highlight_preview_html(preview: Image.Image, highlights: dict[int, Image.Image],
                           max_side: int = 1200) -> str:
    """Stack hidden per-piece layers over the preview; hovering list row N reveals layer N.

    The rows are Streamlit containers keyed ``piece_row_N``, so CSS ``:has()``
    connects them to the layers without a custom component.
    """
    scale = min(1.0, max_side / max(preview.size))
    size = (max(1, round(preview.width * scale)),
            max(1, round(preview.height * scale)))

    def fit(image: Image.Image) -> Image.Image:
        return image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)

    layers = "".join(
        f'<img class="piece-highlight piece-highlight-{index}" src="{image_data_uri(fit(layer), "PNG")}" alt="">'
        for index, layer in highlights.items()
    )
    return (f"<style>{highlight_rules_css(highlights)}</style>"
            f'<div class="piece-preview"><img class="piece-preview-base" '
            f'src="{image_data_uri(fit(preview.convert("RGB")), "JPEG")}" alt="Analyzed frame with detected pieces">'
            f"{layers}</div>")


def app_header_html() -> str:
    """Pure-CSS logo (a treaded shoe print) beside the app introduction."""
    return (
        '<div class="app-header">'
        '<div class="app-logo" role="img" aria-label="Fashion Soles logo">'
        '<span class="app-logo-mark"><span class="app-logo-print">'
        '<span class="app-logo-forefoot"></span><span class="app-logo-heel"></span></span>'
        '</span>'
        '<span class="app-logo-word">Fashion Soles</span>'
        '</div>'
        '<p class="app-intro"><strong>You\'ve heard of Fashion Souls, enter the realm of your outfit\'s story.</strong> '
        'Upload a photo or short clip, choose the detected pieces, and read a poem woven from their shapes and colors. '
        '</br><span class="app-privacy">No footage is saved: everything is analysed locally on this machine. '
        'Faces and identity are of no interest, only the clothes.</span></p>'
        '</div>'
    )


def color_histogram_html(title: str, bins: list[tuple[str, float]], color_mode: str = "basic",
                         labelled: int = 1) -> str:
    """Bars filled with each bin's own color, tallest first; hover shows hex, name, and share."""
    if not bins:
        return (f'<figure class="color-hist"><figcaption>{html.escape(title)}</figcaption>'
                '<p class="color-hist-empty">No pixels to count.</p></figure>')
    peak = max(share for _, share in bins)
    bars = "".join(
        f'<div class="color-hist-bar" style="height:{max(2.0, share / peak * 100):.1f}%; background:{value}" '
        f'data-tip="{value} · {html.escape(fashion_color_name(value, color_mode))} · {share:.1%}">'
        + (f'<span class="color-hist-value">{share:.0%}</span>' if rank < labelled else "")
        + "</div>"
        for rank, (value, share) in enumerate(bins)
        if re.fullmatch(r"#[0-9A-Fa-f]{6}", value)
    )
    return (f'<figure class="color-hist"><figcaption>{html.escape(title)}'
            f'<span>top {len(bins)} bins · peak {peak:.1%}</span></figcaption>'
            f'<div class="color-hist-plot">{bars}</div></figure>')
