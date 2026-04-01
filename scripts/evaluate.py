"""
evaluate.py — Core evaluation pipeline for GeoVLM-Bench.

Sends street-level images to a VLM and records country predictions.

Usage:
    python scripts/evaluate.py --model claude --input data/annotations.csv --output results/results_claude.csv
    python scripts/evaluate.py --model gpt4o --input data/annotations.csv --output results/results_gpt4o.csv
    python scripts/evaluate.py --model gemini --input data/annotations.csv --output results/results_gemini.csv
    python scripts/evaluate.py --model claude --input data/annotations.csv --output results/results_claude.csv --pilot 10
"""

import argparse
import base64
import csv
import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from parse_results import parse_country, is_correct, is_region_correct

logger = logging.getLogger(__name__)

EVALUATION_PROMPT = """You are analyzing a street-level image to determine where in the world it was taken.

First, carefully describe the geographic cues you can identify in the image. Consider:
- Any visible text, signs, or written language
- Vegetation, terrain, and climate indicators
- Road markings, infrastructure, and vehicle types
- Architectural styles and cultural indicators

Then, based on your reasoning, state which country you believe this image was taken in.

Format your response exactly as follows:
REASONING: <your step-by-step analysis>
COUNTRY: <single country name only>"""

SUPPORTED_MODELS = {
    "claude": "claude-haiku-4-5",
    "gpt4o": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "opus": "claude-opus-4-6"
}

# Minimum seconds between requests per model (to respect rate limits)
RATE_LIMITS = {
    "gemini": 12.0,  # 5 RPM = 12s between requests
    "gpt4o": 5.0,    # avoid exceeding TPM limit
}

RESULTS_FIELDNAMES = [
    "id", "cue_type", "true_country", "predicted_country", "correct",
    "true_region", "region_correct", "model", "raw_response",
]


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def load_image_base64(image_path: Path) -> str:
    """
    Load an image file and return its base64-encoded string.

    Args:
        image_path: Absolute or relative path to a JPEG image.

    Returns:
        Base64-encoded string of the image bytes.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Model-specific API callers
# ---------------------------------------------------------------------------

def call_claude(image_b64: str, model_id: str, max_retries: int = 3) -> str:
    """
    Query the Anthropic Claude API with a base64-encoded image.

    Args:
        image_b64: Base64-encoded JPEG image string.
        model_id: Anthropic model identifier.
        max_retries: Maximum number of retry attempts on transient errors.

    Returns:
        Raw text response from the model.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key, timeout=120.0)

    for attempt in range(1, max_retries + 1):
        try:
            message = client.messages.create(
                model=model_id,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": EVALUATION_PROMPT},
                        ],
                    }
                ],
            )
            return message.content[0].text
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"Claude API error (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s...")
            if attempt == max_retries:
                raise RuntimeError(f"Claude API failed after {max_retries} attempts: {e}") from e
            time.sleep(wait)

    raise RuntimeError("Unreachable")  # pragma: no cover


def call_gpt4o(image_b64: str, model_id: str, max_retries: int = 3) -> str:
    """
    Query the OpenAI GPT-4o API with a base64-encoded image.

    Args:
        image_b64: Base64-encoded JPEG image string.
        model_id: OpenAI model identifier.
        max_retries: Maximum number of retry attempts on transient errors.

    Returns:
        Raw text response from the model.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

    client = openai.OpenAI(api_key=api_key)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": EVALUATION_PROMPT},
                        ],
                    }
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"GPT-4o API error (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s...")
            if attempt == max_retries:
                raise RuntimeError(f"GPT-4o API failed after {max_retries} attempts: {e}") from e
            time.sleep(wait)

    raise RuntimeError("Unreachable")  # pragma: no cover


def call_gemini(image_b64: str, model_id: str, max_retries: int = 3) -> str:
    """
    Query the Google Gemini API with a base64-encoded image.

    Args:
        image_b64: Base64-encoded JPEG image string.
        model_id: Gemini model identifier.
        max_retries: Maximum number of retry attempts on transient errors.

    Returns:
        Raw text response from the model.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)
    image_bytes = base64.b64decode(image_b64)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    EVALUATION_PROMPT,
                ],
            )
            return response.text
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"Gemini API error (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s...")
            if attempt == max_retries:
                raise RuntimeError(f"Gemini API failed after {max_retries} attempts: {e}") from e
            time.sleep(wait)

    raise RuntimeError("Unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def query_model(model_key: str, image_b64: str) -> str:
    """
    Dispatch a model query to the appropriate API caller.

    Args:
        model_key: One of 'claude', 'gpt4o', or 'gemini'.
        image_b64: Base64-encoded JPEG image string.

    Returns:
        Raw text response from the model.
    """
    model_id = SUPPORTED_MODELS[model_key]
    if model_key in ("claude", "opus"):
        return call_claude(image_b64, model_id)
    elif model_key == "gpt4o":
        return call_gpt4o(image_b64, model_id)
    elif model_key == "gemini":
        return call_gemini(image_b64, model_id)
    else:
        raise ValueError(f"Unsupported model key: {model_key}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    model_key: str,
    annotations_path: Path,
    output_path: Path,
    pilot: Optional[int],
    project_root: Path,
) -> None:
    """
    Run the full evaluation pipeline for a given model.

    Loads annotations, queries the model for each image, parses predictions,
    and writes results incrementally to a CSV file.

    Args:
        model_key: Model to evaluate ('claude', 'gpt4o', or 'llava').
        annotations_path: Path to annotations.csv.
        output_path: Path for the output results CSV.
        pilot: If set, limit evaluation to this many images.
        project_root: Root directory of the project (for resolving image paths).
    """
    annotations = pd.read_csv(annotations_path)

    if pilot is not None:
        logger.info(f"Pilot mode: running on first {pilot} images.")
        annotations = annotations.head(pilot)

    total = len(annotations)
    logger.info(f"Evaluating {total} images with model '{model_key}' ({SUPPORTED_MODELS[model_key]}).")

    # Determine already-completed IDs to allow resuming interrupted runs
    completed_ids: set[str] = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        completed_ids = set(existing["id"].astype(str))
        logger.info(f"Resuming: {len(completed_ids)} images already done.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or len(completed_ids) == 0

    with open(output_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=RESULTS_FIELDNAMES)
        if write_header:
            writer.writeheader()

        correct_count = 0
        processed = 0

        for _, row in annotations.iterrows():
            image_id = str(row["id"])

            if image_id in completed_ids:
                logger.info(f"[{image_id}] Already processed, skipping.")
                continue

            image_path = project_root / row["filepath"]

            logger.info(f"[{image_id}] Processing ({processed + 1}/{total})...")

            try:
                image_b64 = load_image_base64(image_path)
            except FileNotFoundError:
                logger.warning(f"[{image_id}] Image file not found: {image_path}. Logging as PARSE_ERROR.")
                raw_response = "IMAGE_NOT_FOUND"
                predicted_country = "PARSE_ERROR"
                correct = False
            else:
                try:
                    request_start = time.time()
                    raw_response = query_model(model_key, image_b64)
                    predicted_country = parse_country(raw_response)
                    correct = is_correct(predicted_country, row["country"])

                    # Respect per-model rate limits
                    cooldown = RATE_LIMITS.get(model_key, 0)
                    if cooldown > 0:
                        elapsed = time.time() - request_start
                        wait_time = cooldown - elapsed
                        if wait_time > 0:
                            logger.info(f"Rate limit: waiting {wait_time:.1f}s...")
                            time.sleep(wait_time)
                except Exception as e:
                    logger.error(f"[{image_id}] Model query failed: {e}")
                    continue

            true_region = row.get("region", "")
            region_correct = is_region_correct(predicted_country, true_region)

            result_row = {
                "id": image_id,
                "cue_type": row["cue_type"],
                "true_country": row["country"],
                "predicted_country": predicted_country,
                "correct": correct,
                "true_region": true_region,
                "region_correct": region_correct,
                "model": SUPPORTED_MODELS[model_key],
                "raw_response": raw_response,
            }

            writer.writerow(result_row)
            csvfile.flush()  # Incremental save — prevents data loss on interruption

            if correct:
                correct_count += 1
            processed += 1

            logger.info(
                f"[{image_id}] Predicted: '{predicted_country}' | True: '{row['country']}' | "
                f"Correct: {correct} | Running accuracy: {correct_count}/{processed} "
                f"({100 * correct_count / processed:.1f}%)"
            )

    logger.info(
        f"Evaluation complete. Final accuracy: {correct_count}/{processed} "
        f"({100 * correct_count / processed:.1f}% correct)" if processed > 0
        else "No images processed."
    )


def estimate_cost(model_key: str, num_images: int) -> None:
    """
    Print a rough cost estimate before a full run.

    Args:
        model_key: Model being evaluated.
        num_images: Total number of images to be processed.
    """
    cost_per_image = {"claude": 0.02, "gpt4o": 0.02, "gemini": 0.001}
    estimate = cost_per_image.get(model_key, 0.02) * num_images
    print(f"Estimated cost for {num_images} images with '{model_key}': ~${estimate:.2f}")
    print("(Run with --pilot 10 first to verify before a full run.)")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="GeoVLM-Bench evaluation pipeline.")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model to evaluate: claude, gpt4o, or gemini.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to annotations CSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for output results CSV.",
    )
    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        metavar="N",
        help="Run on only the first N images (pilot mode).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    # Show cost estimate before any full run
    annotations = pd.read_csv(args.input)
    num_images = args.pilot if args.pilot is not None else len(annotations)
    estimate_cost(args.model, num_images)

    if args.pilot is None:
        confirm = input(f"\nAbout to run full evaluation on {num_images} images. Continue? [y/N]: ")
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return

    run_evaluation(
        model_key=args.model,
        annotations_path=args.input,
        output_path=args.output,
        pilot=args.pilot,
        project_root=project_root,
    )


if __name__ == "__main__":
    main()
