from fashion_soles.presentation import color_grid_html


def test_only_dominant_color_renders_as_one_swatch():
    html = color_grid_html(("#AABBCC", "#102030", "#FFFFFF"))
    assert html.count('class="piece-color-grid"') == 1
    assert html.count('class="piece-swatch"') == 1
    assert 'background-color:#AABBCC' in html


def test_invalid_colors_are_not_added_to_html():
    assert color_grid_html(("red; color:red",)) == "Unavailable"
