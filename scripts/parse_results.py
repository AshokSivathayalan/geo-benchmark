"""
parse_results.py — Extract and normalize country predictions from raw model responses.

Usage:
    python scripts/parse_results.py --input results/results_claude.csv --output results/results_claude_parsed.csv
    # Or import and use parse_country() directly from evaluate.py
"""

import re
import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Normalization map for common country name variants
COUNTRY_TO_REGION: dict[str, str] = {
    # Ground truth countries
    "Brazil": "South America",
    "Canada": "North America",
    "Germany": "Western Europe",
    "India": "South Asia",
    "Indonesia": "Southeast Asia",
    "Japan": "East Asia",
    "Russia": "Eastern Europe",
    "South Korea": "East Asia",
    "Turkey": "Middle East",
    "United Kingdom": "Western Europe",
    # Countries predicted by models
    "Albania": "Southern Europe",
    "Algeria": "North Africa",
    "Australia": "Oceania",
    "Austria": "Western Europe",
    "Belarus": "Eastern Europe",
    "Bulgaria": "Eastern Europe",
    "China": "East Asia",
    "Colombia": "South America",
    "Costa Rica": "Central America",
    "Democratic Republic Of The Congo": "Sub-Saharan Africa",
    "Denmark": "Northern Europe",
    "Dominican Republic": "Caribbean",
    "El Salvador": "Central America",
    "Estonia": "Northern Europe",
    "Finland": "Northern Europe",
    "Ghana": "Sub-Saharan Africa",
    "Greece": "Southern Europe",
    "Guatemala": "Central America",
    "Guinea": "Sub-Saharan Africa",
    "Honduras": "Central America",
    "Iran": "Middle East",
    "Israel": "Middle East",
    "Ivory Coast": "Sub-Saharan Africa",
    "Kazakhstan": "Central Asia",
    "Kyrgyzstan": "Central Asia",
    "Laos": "Southeast Asia",
    "Lithuania": "Northern Europe",
    "Malawi": "Sub-Saharan Africa",
    "Malaysia": "Southeast Asia",
    "Mexico": "North America",
    "Myanmar": "Southeast Asia",
    "Netherlands": "Western Europe",
    "New Zealand": "Oceania",
    "Nicaragua": "Central America",
    "Nigeria": "Sub-Saharan Africa",
    "Norway": "Northern Europe",
    "Pakistan": "South Asia",
    "Palestine": "Middle East",
    "Peru": "South America",
    "Philippines": "Southeast Asia",
    "Poland": "Eastern Europe",
    "Romania": "Eastern Europe",
    "Rwanda": "Sub-Saharan Africa",
    "Saudi Arabia": "Middle East",
    "Serbia": "Southern Europe",
    "South Africa": "Sub-Saharan Africa",
    "Spain": "Southern Europe",
    "Sweden": "Northern Europe",
    "Switzerland": "Western Europe",
    "Syria": "Middle East",
    "Taiwan": "East Asia",
    "Thailand": "Southeast Asia",
    "Uganda": "Sub-Saharan Africa",
    "Ukraine": "Eastern Europe",
    "United States": "North America",
    "Vanuatu": "Oceania",
    "Venezuela": "South America",
    "Vietnam": "Southeast Asia",
    "Zambia": "Sub-Saharan Africa",
}

COUNTRY_ALIASES: dict[str, str] = {
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "britain": "United Kingdom",
    "usa": "United States",
    "u.s.a.": "United States",
    "us": "United States",
    "u.s.": "United States",
    "america": "United States",
    "united states of america": "United States",
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "south korea": "South Korea",
    "republic of korea": "South Korea",
    "korea": "South Korea",
    "north korea": "North Korea",
    "democratic people's republic of korea": "North Korea",
    "czech republic": "Czechia",
    "the netherlands": "Netherlands",
    "holland": "Netherlands",
    "taiwan": "Taiwan",
    "republic of china": "Taiwan",
    "people's republic of china": "China",
    "prc": "China",
    "russia": "Russia",
    "russian federation": "Russia",
    "iran": "Iran",
    "islamic republic of iran": "Iran",
    "syria": "Syria",
    "syrian arab republic": "Syria",
    "bolivia": "Bolivia",
    "plurinational state of bolivia": "Bolivia",
    "venezuela": "Venezuela",
    "bolivarian republic of venezuela": "Venezuela",
    "tanzania": "Tanzania",
    "united republic of tanzania": "Tanzania",
    "laos": "Laos",
    "lao pdr": "Laos",
    "lao people's democratic republic": "Laos",
    "vietnam": "Vietnam",
    "viet nam": "Vietnam",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "isle of man": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
    "united states (alaska)": "United States",
}


def parse_country(raw_response: str) -> str:
    """
    Extract and normalize the predicted country from a raw model response.

    Looks for a line starting with 'COUNTRY:' and extracts everything after the colon.
    Normalizes to title case and applies alias mapping.

    Args:
        raw_response: Full text response from the model.

    Returns:
        Normalized country name string, or 'PARSE_ERROR' if not found.
    """
    if not raw_response or not isinstance(raw_response, str):
        logger.warning("Empty or non-string response received.")
        return "PARSE_ERROR"

    lines = raw_response.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("COUNTRY:"):
            country_raw = stripped[len("COUNTRY:"):].strip()
            # If the country is on the next line (e.g. "COUNTRY:\nThailand")
            if not country_raw:
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    if next_line:
                        country_raw = next_line
                        break
            if not country_raw:
                logger.warning("COUNTRY: line found but value is empty.")
                return "PARSE_ERROR"
            return _normalize_country(country_raw)

    logger.warning("No COUNTRY: line found in response.")
    return "PARSE_ERROR"


def _normalize_country(raw: str) -> str:
    """
    Normalize a raw country string to a canonical title-cased name.

    Applies alias lookup (case-insensitive), strips trailing punctuation,
    and falls back to title case if no alias matches.

    Args:
        raw: Raw country string extracted from model output.

    Returns:
        Canonical country name.
    """
    # Strip trailing punctuation and extra whitespace
    cleaned = re.sub(r"[.\-,;]+$", "", raw.strip())
    cleaned = cleaned.strip()

    # Alias lookup (case-insensitive)
    lower = cleaned.lower()
    if lower in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[lower]

    # Default: title case
    return cleaned.title()


def get_region(country: str) -> str:
    """
    Look up the geographic region for a country name.

    Args:
        country: Normalized country name.

    Returns:
        Region string, or empty string if unknown.
    """
    if not country or country == "PARSE_ERROR":
        return ""
    return COUNTRY_TO_REGION.get(country, "")


def is_region_correct(predicted: str, true_region: str) -> bool:
    """
    Check whether the predicted country falls in the same region as the true country.

    Args:
        predicted: Normalized predicted country name.
        true_region: Ground truth region.

    Returns:
        True if the predicted country's region matches, False otherwise.
    """
    if predicted == "PARSE_ERROR" or not true_region:
        return False
    return get_region(predicted).lower() == true_region.strip().lower()


def is_correct(predicted: str, true_country: str) -> bool:
    """
    Compare predicted and true country names case-insensitively.

    Args:
        predicted: Normalized predicted country name.
        true_country: Ground truth country name.

    Returns:
        True if they match (case-insensitive), False otherwise.
    """
    if predicted == "PARSE_ERROR":
        return False
    return predicted.strip().lower() == true_country.strip().lower()


def parse_results_file(input_path: Path, output_path: Path, annotations_path: Path | None = None) -> None:
    """
    Re-parse the raw_response column of a results CSV and recompute predicted_country and correct.

    Useful for re-running parsing logic without re-querying the API.
    If annotations_path is provided, joins true_region and computes region_correct.

    Args:
        input_path: Path to results CSV with a raw_response column.
        output_path: Path to write the updated results CSV.
        annotations_path: Optional path to annotations CSV for region data.
    """
    df = pd.read_csv(input_path)

    if "raw_response" not in df.columns:
        raise ValueError(f"'raw_response' column not found in {input_path}")

    df["predicted_country"] = df["raw_response"].apply(parse_country)
    df["correct"] = df.apply(
        lambda row: is_correct(row["predicted_country"], row["true_country"]), axis=1
    )

    # Add region correctness
    if annotations_path is not None:
        annotations = pd.read_csv(annotations_path)
        region_map = annotations.set_index("id")["region"].to_dict()
        df["true_region"] = df["id"].map(region_map).fillna("")
    elif "true_region" not in df.columns:
        df["true_region"] = df["true_country"].apply(lambda c: COUNTRY_TO_REGION.get(c, ""))

    df["region_correct"] = df.apply(
        lambda row: is_region_correct(row["predicted_country"], row["true_region"]), axis=1
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Parsed results written to {output_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Re-parse raw model responses to extract country predictions."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input results CSV path.")
    parser.add_argument("--output", required=True, type=Path, help="Output parsed CSV path.")
    parser.add_argument("--annotations", type=Path, default=None, help="Annotations CSV for region data.")
    args = parser.parse_args()

    parse_results_file(args.input, args.output, annotations_path=args.annotations)


if __name__ == "__main__":
    main()
