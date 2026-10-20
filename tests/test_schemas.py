import pytest
from fashion_soles.pipeline import demo_analysis
from fashion_soles.schemas import OutfitAnalysis


def test_unknown_fields_ignored_and_unknown_remains_unknown():
    data = demo_analysis().model_dump()
    data["garments"][0]["material"] = "unknown"
    data["garments"][0]["invented_field"] = "private identity"
    data["face"] = "ignored"
    parsed = OutfitAnalysis.model_validate(data)
    assert parsed.garments[0].material == "unknown"
    assert "face" not in parsed.model_dump()
    assert "invented_field" not in parsed.garments[0].model_dump()


def test_bad_relationship_rejected():
    data = demo_analysis().model_dump()
    data["relationships"][0]["garment_ids"] = ["missing"]
    with pytest.raises(ValueError):
        OutfitAnalysis.model_validate(data)
