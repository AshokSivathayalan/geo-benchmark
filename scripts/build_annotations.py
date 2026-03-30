"""
build_annotations.py — Generate annotations.csv from images.txt.

Parses the structured images.txt format and produces the annotations CSV
expected by the evaluation pipeline.

Usage:
    python scripts/build_annotations.py
"""

import csv
from pathlib import Path

COUNTRY_FOLDER_MAP = {
    "UK": "United Kingdom",
}

REGION_MAP = {
    "Japan": "East Asia",
    "South Korea": "East Asia",
    "Canada": "North America",
    "Brazil": "South America",
    "Turkey": "Middle East",
    "Russia": "Eastern Europe",
    "Germany": "Western Europe",
    "United Kingdom": "Western Europe",
    "India": "South Asia",
    "Indonesia": "Southeast Asia",
    "Mexico": "North America",
    "Finland": "Northern Europe",
    "Thailand": "Southeast Asia",
}

CUE_TYPE_MAP = {
    "Linguistic": "linguistic",
    "Environmental": "environmental",
    "Infrastructural": "infrastructure",
    "Multi-Cue": "multi_cue",
}

FIELDNAMES = ["id", "filepath", "country", "cue_type", "multi_cue", "cue_notes", "region"]


def parse_images_txt(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    rows = []
    current_country = None
    current_cue = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if it's a cue type header
        if line in CUE_TYPE_MAP:
            current_cue = line
            continue

        # Check if it's a country name (no digits)
        if not any(c.isdigit() for c in line):
            current_country = line
            current_cue = None
            continue

        # Otherwise it's an image ID line
        if current_country is None or current_cue is None:
            continue

        parts = line.split()
        image_id = parts[0]
        multi_cue_notes = " ".join(parts[1:]) if len(parts) > 1 else ""

        country = COUNTRY_FOLDER_MAP.get(current_country, current_country)
        folder = country
        is_multi = current_cue == "Multi-Cue"

        cue_type = ""
        if is_multi and multi_cue_notes:
            # Use the first cue letter as primary: L=linguistic, E=environmental, I=infrastructure
            cue_map = {"L": "linguistic", "E": "environmental", "I": "infrastructure"}
            first_letter = multi_cue_notes.strip().split("/")[0]
            cue_type = cue_map.get(first_letter, "")
        else:
            cue_type = CUE_TYPE_MAP.get(current_cue, "")

        rows.append({
            "id": image_id,
            "filepath": f"data/images/{folder}/{image_id}.jpg",
            "country": country,
            "cue_type": cue_type,
            "multi_cue": is_multi,
            "cue_notes": multi_cue_notes if is_multi else "",
            "region": REGION_MAP.get(country, ""),
        })

    return rows


def main() -> None:
    project_root = Path(__file__).parent.parent
    images_txt = project_root / "images.txt"
    annotations_path = project_root / "data" / "annotations.csv"

    rows = parse_images_txt(images_txt)

    with open(annotations_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {annotations_path}")


if __name__ == "__main__":
    main()
