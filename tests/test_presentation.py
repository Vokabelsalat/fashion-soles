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


def test_color_histogram_counts_whole_image_and_masked_pixels():
    from PIL import Image, ImageDraw
    from fashion_soles.palette import color_histogram
    from fashion_soles.presentation import color_histogram_html

    image = Image.new("RGB", (100, 100), "#102030")
    ImageDraw.Draw(image).rectangle((0, 0, 24, 99), fill="#F0F0F0")
    whole = color_histogram(image)
    assert [(value, round(share, 2)) for value, share in whole] == [("#102030", .75), ("#F0F0F0", .25)]
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((0, 0, 49, 99), fill=255)
    assert [round(share, 2) for _, share in color_histogram(image, mask)] == [.5, .5]
    markup = color_histogram_html("Whole image", whole)
    assert markup.count('class="color-hist-bar"') == 2
    assert 'data-tip="#102030 · ' in markup and "75.0%" in markup
    assert "No pixels" in color_histogram_html("Clothing pieces (0)", [])
