"""
download_images.py — Fetch street-level images from the Mapillary API.

Downloads images for a specified set of geographic coordinates or image IDs,
saves them to data/images/, and appends rows to data/annotations.csv.

The Graph API images search endpoint requires OAuth user tokens and returns
empty results with client tokens. This script uses the Vector Tiles API
(which works with client tokens) to find image IDs near coordinates, then
fetches thumbnail URLs via the Graph API.

Usage:
    python scripts/download_images.py --coords coords.csv --output data/images/ --limit 3
    python scripts/download_images.py --image-ids ids.txt --output data/images/
"""

import argparse
import csv
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import mapbox_vector_tile
import requests
from dotenv import load_dotenv

# Load .env from project root (one level up from scripts/)
load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

MAPILLARY_API_BASE = "https://graph.mapillary.com"
MAPILLARY_TILE_BASE = "https://tiles.mapillary.com/maps/vtp/mly1_public/2"
TILE_ZOOM = 14
TILE_EXTENT = 4096  # MVT default extent

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


# ---------------------------------------------------------------------------
# Tile coordinate math
# ---------------------------------------------------------------------------

def lat_lon_to_tile(lat: float, lon: float, zoom: int = TILE_ZOOM) -> tuple[int, int]:
    """
    Convert a lat/lon coordinate to slippy map tile indices at a given zoom level.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        zoom: Tile zoom level.

    Returns:
        (tile_x, tile_y) integer tile indices.
    """
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    tile_x = int((lon + 180) / 360 * n)
    tile_y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    return tile_x, tile_y


def tile_pixel_to_lat_lon(
    tile_x: int, tile_y: int, px: float, py: float, zoom: int = TILE_ZOOM
) -> tuple[float, float]:
    """
    Convert tile-local pixel coordinates back to lat/lon.

    Args:
        tile_x: Tile column index.
        tile_y: Tile row index.
        px: Pixel x within tile (0–TILE_EXTENT).
        py: Pixel y within tile (0–TILE_EXTENT).
        zoom: Tile zoom level.

    Returns:
        (lat, lon) in degrees.
    """
    n = 2 ** zoom
    lon = (tile_x + px / TILE_EXTENT) / n * 360 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (tile_y + py / TILE_EXTENT) / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the great-circle distance in metres between two lat/lon points.

    Args:
        lat1, lon1: First point in degrees.
        lat2, lon2: Second point in degrees.

    Returns:
        Distance in metres.
    """
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Vector tile image search
# ---------------------------------------------------------------------------

def fetch_tile(tile_x: int, tile_y: int, token: str) -> bytes:
    """
    Download a Mapillary vector tile as raw protobuf bytes.

    Args:
        tile_x: Tile column index at TILE_ZOOM.
        tile_y: Tile row index at TILE_ZOOM.
        token: Mapillary API token.

    Returns:
        Raw protobuf bytes of the tile.

    Raises:
        requests.HTTPError: On non-200 response.
    """
    url = f"{MAPILLARY_TILE_BASE}/{TILE_ZOOM}/{tile_x}/{tile_y}"
    response = requests.get(url, params={"access_token": token}, timeout=30)
    response.raise_for_status()
    return response.content


def search_images_near(
    lat: float,
    lon: float,
    radius_m: int = 500,
    limit: int = 5,
    token: str = "",
) -> list[dict]:
    """
    Find Mapillary image IDs near a coordinate using the vector tiles API.

    Fetches the tile covering the target coordinate, parses image features,
    filters by haversine distance, and returns the closest ones.

    Args:
        lat: Target latitude.
        lon: Target longitude.
        radius_m: Search radius in metres.
        limit: Maximum number of images to return.
        token: Mapillary API token.

    Returns:
        List of dicts with keys 'id', 'lat', 'lon', 'distance_m'.
    """
    tile_x, tile_y = lat_lon_to_tile(lat, lon)
    logger.debug(f"Fetching tile z={TILE_ZOOM} x={tile_x} y={tile_y}")

    tile_bytes = fetch_tile(tile_x, tile_y, token)
    tile_data = mapbox_vector_tile.decode(tile_bytes)

    image_layer = tile_data.get("image", {})
    features = image_layer.get("features", [])
    logger.debug(f"Tile contains {len(features)} image features")

    candidates = []
    for feature in features:
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates")
        props = feature.get("properties", {})
        image_id = props.get("id")

        if not coords or not image_id:
            continue

        # Point geometry: coordinates is [px, py]
        if geom.get("type") == "Point":
            px, py = coords
        else:
            continue

        img_lat, img_lon = tile_pixel_to_lat_lon(tile_x, tile_y, px, py)
        dist = haversine_m(lat, lon, img_lat, img_lon)

        if dist <= radius_m:
            candidates.append({
                "id": image_id,
                "lat": img_lat,
                "lon": img_lon,
                "distance_m": dist,
            })

    candidates.sort(key=lambda c: c["distance_m"])
    return candidates[:limit]


# ---------------------------------------------------------------------------
# Graph API: fetch thumbnail URL for a known image ID
# ---------------------------------------------------------------------------

def fetch_image_url(image_id: str | int, token: str, skip_pano: bool = True) -> Optional[str]:
    """
    Fetch the 2048px thumbnail URL for a specific Mapillary image ID.

    Args:
        image_id: Mapillary image identifier.
        token: Mapillary API token.
        skip_pano: If True, return None for panoramic/360° images.

    Returns:
        Thumbnail URL string, or None if unavailable or panoramic.
    """
    url = f"{MAPILLARY_API_BASE}/{image_id}"
    params = {"access_token": token, "fields": "id,thumb_2048_url,is_pano"}
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    if skip_pano and data.get("is_pano", False):
        logger.info(f"Skipping panoramic image: {image_id}")
        return None
    return data.get("thumb_2048_url")


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

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

    Uses the vector tiles API to find nearby image IDs, then fetches their
    thumbnail URLs via the Graph API.

    Args:
        coords_path: Path to the coordinates CSV (columns: lat, lon, country, region).
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
            # Fetch extra candidates to compensate for panoramas being skipped
            candidates = search_images_near(lat, lon, radius_m=500, limit=limit_per_coord * 2, token=token)
        except Exception as e:
            logger.error(f"Tile search failed for ({lat}, {lon}): {e}")
            continue

        if not candidates:
            logger.warning(f"No images found within 500m of ({lat:.4f}, {lon:.4f}). Skipping.")
            continue

        logger.info(f"Found {len(candidates)} candidate(s) — fetching thumbnail URLs...")

        saved_this_coord = 0
        for candidate in candidates:
            if saved_this_coord >= limit_per_coord:
                break
            image_id = str(candidate["id"])
            dest_filename = f"{image_id}.jpg"
            country_dir = output_dir / country
            country_dir.mkdir(parents=True, exist_ok=True)
            dest_path = country_dir / dest_filename
            rel_path = f"data/images/{country}/{dest_filename}"

            if dest_path.exists():
                logger.info(f"Already downloaded: {dest_filename}")
                downloaded += 1
                continue

            try:
                img_url = fetch_image_url(image_id, token)
            except Exception as e:
                logger.error(f"Failed to fetch URL for {image_id}: {e}")
                continue

            if not img_url:
                logger.warning(f"No thumbnail URL for {image_id}.")
                continue

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
            saved_this_coord += 1
            logger.info(
                f"Saved: {dest_filename} | {country} | {candidate['distance_m']:.0f}m from target "
                f"({downloaded} total)"
            )

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
# Batch download from images.txt (structured format with countries/cue types)
# ---------------------------------------------------------------------------

COUNTRY_FOLDER_MAP = {
    "UK": "United Kingdom",
}


def parse_images_txt(path: Path) -> list[tuple[str, str]]:
    """
    Parse images.txt and return a list of (image_id, country_folder_name) tuples.
    """
    lines = path.read_text().splitlines()
    entries = []
    current_country = None

    cue_headers = {"Linguistic", "Environmental", "Infrastructural", "Multi-Cue"}

    for line in lines:
        line = line.strip()
        if not line or line in cue_headers:
            continue

        # If no digits, it's a country name
        if not any(c.isdigit() for c in line):
            current_country = COUNTRY_FOLDER_MAP.get(line, line)
            continue

        if current_country is None:
            continue

        image_id = line.split()[0]
        entries.append((image_id, current_country))

    return entries


def download_from_images_txt(
    txt_path: Path,
    output_dir: Path,
    token: str,
) -> None:
    """
    Download images listed in images.txt into country subfolders.

    Args:
        txt_path: Path to images.txt.
        output_dir: Base directory for images (e.g. data/images/).
        token: Mapillary API token.
    """
    entries = parse_images_txt(txt_path)
    logger.info(f"Found {len(entries)} images in {txt_path.name}")

    downloaded = 0
    skipped = 0
    for image_id, country in entries:
        country_dir = output_dir / country
        country_dir.mkdir(parents=True, exist_ok=True)
        dest_path = country_dir / f"{image_id}.jpg"

        if dest_path.exists():
            skipped += 1
            continue

        try:
            img_url = fetch_image_url(image_id, token, skip_pano=False)
        except Exception as e:
            logger.error(f"Failed to fetch URL for {image_id}: {e}")
            continue

        if not img_url:
            logger.warning(f"No thumbnail URL for {image_id}.")
            continue

        success = download_image(img_url, dest_path)
        if success:
            downloaded += 1
            logger.info(f"Saved: {country}/{image_id}.jpg ({downloaded} new)")

    logger.info(f"Done. {downloaded} new, {skipped} already existed.")


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
        help="CSV with columns lat, lon, country, region.",
    )
    source_group.add_argument(
        "--image-ids",
        type=Path,
        help="Text file with one Mapillary image ID per line.",
    )
    source_group.add_argument(
        "--images-txt",
        type=Path,
        help="Structured images.txt file with countries and cue categories.",
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
    elif args.images_txt:
        download_from_images_txt(
            txt_path=args.images_txt,
            output_dir=args.output,
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
