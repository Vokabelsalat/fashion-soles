from types import SimpleNamespace
from PIL import Image
from fashion_soles.mapping import map_analysis
from fashion_soles.pipeline import demo_analysis
from fashion_soles.poetry import generate_poem
from fashion_soles.vision import analyze_image


class FakeResponses:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.result)


def test_separate_api_inputs_and_no_storage():
    analysis = demo_analysis()
    vision = FakeResponses(analysis)
    assert analyze_image(Image.new("RGB", (32, 32)), client=SimpleNamespace(responses=vision)) == analysis
    assert vision.calls[0]["store"] is False
    assert vision.calls[0]["input"][1]["content"][1]["type"] == "input_image"

    trace = map_analysis(analysis)
    poem = {"title": "A seam", "lines": [f"Line {i}" for i in range(12)], "used_rule_ids": ["color_dark"]}
    text = FakeResponses(poem)
    generate_poem(analysis, trace, client=SimpleNamespace(responses=text))
    assert text.calls[0]["store"] is False
    assert "input_image" not in str(text.calls[0]["input"])
    assert "data:image" not in str(text.calls[0]["input"])


def test_poem_retries_invalid_length_once():
    class RetryingResponses(FakeResponses):
        def parse(self, **kwargs):
            self.calls.append(kwargs)
            lines = 7 if len(self.calls) == 1 else 12
            return SimpleNamespace(output_parsed={"title": "A seam", "lines": [f"Line {i}" for i in range(lines)],
                                                  "used_rule_ids": []})

    responses = RetryingResponses(None)
    analysis = demo_analysis()
    poem = generate_poem(analysis, map_analysis(analysis), client=SimpleNamespace(responses=responses))
    assert len(poem.lines) == 12
    assert len(responses.calls) == 2
