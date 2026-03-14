"""
download_images.py — Fetch street-level images from the Mapillary API.

Downloads images for a specified set of geographic coordinates or image IDs,
saves them to data/images/, and appends rows to data/annotations.csv.

Usage:
    python scripts/download_images.py --coords coords.csv --output data/images/ --limit 50
    python scripts/download_images.py --image-ids ids.txt --output data/images/
"""

import argparse
import csv
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

MAPILLARY_API_BASE = "https://graph.mapillary.com"

ANNOTATIONS_FIELDNAMES = [
    "id", "filepath", "country", "cue_type", "multi_cue", "cue_notes", "region"
]


# ---------------------------------------------------------------------------
# Mapillary API helpers
# ---------------------------------------------------------------------------

def get_mapillary_token() -> str:
    """
    Retrieve the Mapillary API token from the environment.

    Returns:
        API token string.

    Raises:
        EnvironmentError: If MAPILLARY_API_KEY is not set.
    """
    token = os.environ.get("MAPILLARY_API_KEY")
    if not token:
        raise EnvironmentError("MAPILLARY_API_KEY environment variable not set.")
    return token


def search_images_near(
    lat: float,
    lon: float,
    radius_m: int = 100,
    limit: int = 5,
    token: str = "",
) -> list[dict]:
    """
    Search for Mapillary images near a given coordinate.

    Args:
        lat: Latitude of the search center.
        lon: Longitude of the search center.
        radius_m: Search radius in meters.
        limit: Maximum number of results to return.
        token: Mapillary API token.

    Returns:
        List of image metadata dicts from the Mapillary API.
    """
    url = f"{MAPILLARY_API_BASE}/images"
    params = {
        "access_token": token,
        "fields": "id,thumb_2048_url,geometry,captured_at",
        "closeto": f"{lon},{lat}",
        "radius": radius_m,
        "limit": limit,
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    return data.get("data", [])


def fetch_image_url(image_id: str, token: str) -> Optional[str]:
    """
    Fetch the thumbnail URL for a specific Mapillary image ID.

    Args:
        image_id: Mapillary image identifier.
        token: Mapillary API token.

    Returns:
        URL string for a 2048px thumbnail, or None if unavailable.
    """
    url = f"{MAPILLARY_API_BASE}/{image_id}"
    params = {
        "access_token": token,
        "fields": "id,thumb_2048_url",
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    return data.get("thumb_2048_url")


def download_image(image_url: str, dest_path: Path, max_retries: int = 3) -> bool:
    """
    Download an image from a URL and save it to disk.

    Args:
        image_url: URL of the image to download.
        dest_path: File path to save the downloaded image.
        max_retries: Number of retry attempts on transient failures.

    Returns:
        True if download succeeded, False otherwise.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(image_url, timeout=30, stream=True)
            response.raise_for_status()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"Download error (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s...")
            if attempt == max_retries:
                logger.error(f"Failed to download {image_url}: {e}")
                return False
            time.sleep(wait)
    return False


# ---------------------------------------------------------------------------
# Annotation writing
# ---------------------------------------------------------------------------

def append_annotation(
    annotations_path: Path,
    image_id: str,
    filepath: str,
    country: str = "",
    cue_type: str = "",
    multi_cue: bool = False,
    cue_notes: str = "",
    region: str = "",
) -> None:
    """
    Append a new row to the annotations CSV.

    Creates the file with headers if it does not exist.

    Args:
        annotations_path: Path to annotations.csv.
        image_id: Unique image ID.
        filepath: Relative path to the saved image.
        country: Ground truth country (fill in manually after download).
        cue_type: Cue classification (fill in manually after download).
        multi_cue: Whether the image has multiple cue types.
        cue_notes: Free-text annotation notes.
        region: Geographic region.
    """
    file_exists = annotations_path.exists()
    with open(annotations_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATIONS_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "id": image_id,
            "filepath": filepath,
            "country": country,
            "cue_type": cue_type,
            "multi_cue": multi_cue,
            "cue_notes": cue_notes,
            "region": region,
        })


# ---------------------------------------------------------------------------
# Batch download from coordinate list
# ---------------------------------------------------------------------------

def download_from_coords(
    coords_path: Path,
    output_dir: Path,
    annotations_path: Path,
    limit_per_coord: int,
    token: str,
) -> None:
    """
    Download images for each (lat, lon, country, region) row in a CSV file.

    The coords CSV must have columns: lat, lon, country, region.
    Cue type and notes must be filled in manually after download.

    Args:
        coords_path: Path to the coordinates CSV.
        output_dir: Directory to save downloaded images.
        annotations_path: Path to the annotations CSV to append to.
        limit_per_coord: Maximum images to download per coordinate.
        token: Mapillary API token.
    """
    import pandas as pd

    coords = pd.read_csv(coords_path)
    required = {"lat", "lon", "country", "region"}
    if not required.issubset(coords.columns):
        raise ValueError(f"coords CSV must have columns: {required}. Found: {set(coords.columns)}")

    downloaded = 0
    for _, row in coords.iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        country = str(row["country"])
        region = str(row["region"])
        logger.info(f"Searching near ({lat:.4f}, {lon:.4f}) — {country}...")

        try:
            images = search_images_near(lat, lon, limit=limit_per_coord, token=token)
        except Exception as e:
            logger.error(f"Search failed for ({lat}, {lon}): {e}")
            continue

        for img in images:
            image_id = img.get("id")
            img_url = img.get("thumb_2048_url")

            if not image_id or not img_url:
                continue

            dest_filename = f"{image_id}.jpg"
            dest_path = output_dir / dest_filename
            rel_path = f"data/images/{dest_filename}"

            if dest_path.exists():
                logger.info(f"Already downloaded: {dest_filename}")
            else:
                success = download_image(img_url, dest_path)
                if not success:
                    continue

            append_annotation(
                annotations_path,
                image_id=image_id,
                filepath=rel_path,
                country=country,
                region=region,
            )
            downloaded += 1
            logger.info(f"Saved: {dest_filename} ({downloaded} total)")

    logger.info(f"Done. {downloaded} images downloaded.")


# ---------------------------------------------------------------------------
# Batch download from explicit image ID list
# ---------------------------------------------------------------------------

def download_from_ids(
    ids_path: Path,
    output_dir: Path,
    annotations_path: Path,
    token: str,
) -> None:
    """
    Download specific Mapillary images by ID from a text file (one ID per line).

    Country, cue_type, and other fields must be filled in manually afterward.

    Args:
        ids_path: Path to a text file with one Mapillary image ID per line.
        output_dir: Directory to save downloaded images.
        annotations_path: Path to annotations CSV to append to.
        token: Mapillary API token.
    """
    image_ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    logger.info(f"Downloading {len(image_ids)} images by ID...")

    for image_id in image_ids:
        dest_path = output_dir / f"{image_id}.jpg"
        rel_path = f"data/images/{image_id}.jpg"

        if dest_path.exists():
            logger.info(f"Already exists: {image_id}.jpg")
            continue

        try:
            img_url = fetch_image_url(image_id, token)
        except Exception as e:
            logger.error(f"Failed to fetch URL for {image_id}: {e}")
            continue

        if not img_url:
            logger.warning(f"No thumbnail URL found for {image_id}.")
            continue

        success = download_image(img_url, dest_path)
        if success:
            append_annotation(annotations_path, image_id=image_id, filepath=rel_path)
            logger.info(f"Saved: {image_id}.jpg")

    logger.info("Download from IDs complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Download street-level images from Mapillary.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--coords",
        type=Path,
        help="CSV with columns lat, lon, country, region. Images are searched near each coordinate.",
    )
    source_group.add_argument(
        "--image-ids",
        type=Path,
        help="Text file with one Mapillary image ID per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/images"),
        help="Directory to save downloaded images (default: data/images/).",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations.csv"),
        help="Annotations CSV to append to (default: data/annotations.csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Max images per coordinate when using --coords (default: 3).",
    )
    args = parser.parse_args()

    token = get_mapillary_token()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.coords:
        download_from_coords(
            coords_path=args.coords,
            output_dir=args.output,
            annotations_path=args.annotations,
            limit_per_coord=args.limit,
            token=token,
        )
    else:
        download_from_ids(
            ids_path=args.image_ids,
            output_dir=args.output,
            annotations_path=args.annotations,
            token=token,
        )


if __name__ == "__main__":
    main()
