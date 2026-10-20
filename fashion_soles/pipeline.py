from __future__ import annotations

import hashlib
import json
from pathlib import Path
from .mapping import map_analysis
from .poetry import demo_poem
from .schemas import OutfitAnalysis

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "example_analysis.json"


def demo_analysis() -> OutfitAnalysis:
    return OutfitAnalysis.model_validate_json(FIXTURE.read_text())


def image_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reset_for_new_image(state: dict, fingerprint: str | None) -> bool:
    if state.get("image_fingerprint") == fingerprint:
        return False
    for key in ("analysis", "trace", "poem", "segmentation", "piece_proposals", "variation"):
        state.pop(key, None)
    for key in list(state):
        if key.startswith(("cat_", "colors_", "mat_", "sil_", "cond_")):
            state.pop(key, None)
    state["image_fingerprint"] = fingerprint
    return True


def demo_run(length: int = 12, tone: str = "lyrical", variation: int = 0) -> dict:
    analysis = demo_analysis()
    trace = map_analysis(analysis)
    poem = demo_poem(analysis, trace, length=length, tone=tone, variation=variation)
    return {"mode": "demo", "analysis": analysis.model_dump(), "mapping": trace.model_dump(), "poem": poem.model_dump()}


def export_trace(analysis, trace, poem, *, mode: str, segmentation_status: str, pieces=None) -> str:
    return json.dumps({"mode": mode, "segmentation_status": segmentation_status,
                       "piece_proposals": [{"label": p.label, "score": p.score, "box": p.box,
                                            "mask_status": p.mask_status} for p in (pieces or [])],
                       "analysis": analysis.model_dump(), "mapping": trace.model_dump(),
                       "poem": poem.model_dump() if poem else None}, indent=2, ensure_ascii=False)
