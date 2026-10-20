from fashion_soles.mapping import load_rules, map_analysis
from fashion_soles.pipeline import demo_analysis
from fashion_soles.schemas import OutfitAnalysis


def test_expected_rules_and_tensions():
    trace = map_analysis(demo_analysis())
    ids = {r.rule_id for r in trace.matched_rules}
    assert {"color_dark", "surface_metallic", "condition_worn", "material_leather"}.issubset(ids)
    assert {"tension_structure_wear", "tension_dark_reflection"}.issubset({t.rule_id for t in trace.tensions})


def test_color_change_changes_mapping():
    data = demo_analysis().model_dump()
    data["garments"] = [data["garments"][0]]
    data["relationships"] = []
    data["overall"]["contrast"] = "unknown"
    before = map_analysis(OutfitAnalysis.model_validate(data))
    data["garments"][0]["colors"] = ["white"]
    after = map_analysis(OutfitAnalysis.model_validate(data))
    assert "color_dark" in {r.rule_id for r in before.matched_rules}
    assert "color_dark" not in {r.rule_id for r in after.matched_rules}
    assert "color_pale" in {r.rule_id for r in after.matched_rules}


def test_personal_attributes_cannot_trigger_rules():
    data = demo_analysis().model_dump()
    data["garments"] = []
    data["relationships"] = []
    data["overall"] = {"dominant_colors": [], "silhouette": [], "contrast": "unknown"}
    data["personality"] = "dark"
    trace = map_analysis(OutfitAnalysis.model_validate(data))
    assert not trace.matched_rules
    allowed = {r["id"] for r in load_rules()["rules"]}
    assert {r.rule_id for r in map_analysis(demo_analysis()).matched_rules} <= allowed
