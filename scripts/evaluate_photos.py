"""Run detection on the labelled photos in tests/fixtures/photos and compare with labels.csv.

Each CSV row names one expected piece. "piece", "pattern", and "material" may list
alternatives separated by "|"; an empty pattern or material is not checked. "status"
is "pass" for rows that must keep passing (tests/test_fashion_clip.py checks them)
and "known miss" for documented failures. Add a photo by saving it in that folder
and adding one row per piece worth checking.

    python scripts/evaluate_photos.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PHOTOS = ROOT / "tests" / "fixtures" / "photos"

from fashion_soles.fashion_detector import propose_pieces  # noqa: E402
from fashion_soles.from_pieces import piece_material, piece_pattern  # noqa: E402
from fashion_soles.segmentation import segment_clothes  # noqa: E402


def options(value: str) -> set[str]:
    return {option.strip().lower() for option in value.split("|") if option.strip()}


def evaluate() -> list[dict]:
    """One result per CSV row: what was found and whether pattern and material match."""
    rows = list(csv.DictReader((PHOTOS / "labels.csv").open(newline="")))
    pieces = {}
    for name in dict.fromkeys(row["file"] for row in rows):
        image = Image.open(PHOTOS / name).convert("RGB")
        segmentation = segment_clothes(image)
        pieces[name] = (image, propose_pieces(image, segmentation.mask, segmentation.accessory_mask))
    results = []
    for row in rows:
        image, found = pieces[row["file"]]
        wanted = options(row["piece"])
        match = next((piece for piece in found
                      if wanted & options(piece.label.replace(",", "|"))), None)
        pattern = piece_pattern(match, image) if match else "missing"
        material = piece_material(match) if match else "missing"
        results.append({**row, "found": match.label if match else "missing", "found_pattern": pattern,
                        "found_material": material,
                        "pattern_ok": not row["pattern"] or pattern in options(row["pattern"]),
                        "material_ok": not row["material"] or material in options(row["material"])})
    return results


def main():
    results = evaluate()
    for result in results:
        marks = ("ok " if result["pattern_ok"] else "XX ") + ("ok " if result["material_ok"] else "XX ")
        print(f"{marks} {result['status']:10} {result['file']:16} {result['found']:28.28} pattern {result['found_pattern']:15} "
              f"(want {result['pattern'] or '-'})  material {result['found_material']:9} "
              f"(want {result['material'] or '-'})")
    for key in ("pattern_ok", "material_ok"):
        print(f"{key.split('_')[0]}: {sum(r[key] for r in results)}/{len(results)}")


if __name__ == "__main__":
    main()
