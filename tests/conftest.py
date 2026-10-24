import pytest


@pytest.fixture(autouse=True)
def no_fashion_clip(request, monkeypatch):
    """Keep FashionCLIP out of unit tests; tests marked fashion_clip run the real local model."""
    if "fashion_clip" not in request.keywords:
        monkeypatch.setattr("fashion_soles.fashion_detector.describe_pieces",
                            lambda image, pieces: [None] * len(pieces))
