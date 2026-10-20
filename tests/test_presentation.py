from fashion_soles.presentation import color_grid_html


def test_colors_render_as_one_row_of_div_swatches():
    html = color_grid_html(("#AABBCC", "#102030", "#FFFFFF"))
    assert html.count('class="piece-color-grid"') == 1
    assert html.count('class="piece-swatch"') == 3
    assert 'background-color:#AABBCC' in html
    assert 'background-color:#102030' in html
    assert 'background-color:#FFFFFF' in html


def test_invalid_colors_are_not_added_to_html():
    assert color_grid_html(("red; color:red",)) == "Unavailable"
