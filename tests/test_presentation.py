from fashion_soles.presentation import color_grid_html


def test_only_dominant_color_renders_as_one_swatch():
    html = color_grid_html(("#AABBCC", "#102030", "#FFFFFF"))
    assert html.count('class="piece-color-grid"') == 1
    assert html.count('class="piece-swatch"') == 1
    assert 'background-color:#AABBCC' in html


def test_invalid_colors_are_not_added_to_html():
    assert color_grid_html(("red; color:red",)) == "Unavailable"


def test_highlight_preview_links_each_layer_to_its_list_row():
    from PIL import Image
    from fashion_soles.presentation import highlight_preview_html

    preview = Image.new("RGB", (2400, 1200), "white")
    layer = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    html = highlight_preview_html(preview, {0: layer, 3: layer})
    assert html.count('class="piece-highlight ') == 2
    assert "piece-highlight-3" in html and "piece-highlight-1" not in html
    assert ".stApp:has(.st-key-piece_row_3:hover) .piece-highlight-3" in html
    assert 'class="piece-preview-base"' in html
