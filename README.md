# GeoVLM-Bench

A benchmark for evaluating geographic visual reasoning in vision-language models (VLMs), structured around cue type annotations (linguistic, environmental, infrastructure).

## Setup

```bash
pip install -r requirements.txt
```

Set API keys as environment variables (or in a `.env` file in the project root):
```bash
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
export GEMINI_API_KEY=your_key
export MAPILLARY_API_KEY=your_key     # for image downloading
```

## Workflow

### 1. Download images (optional — can collect manually)
```bash
# From a coordinates CSV (lat, lon, country, region columns required)
python scripts/download_images.py --coords my_coords.csv --output data/images/ --limit 3

# From explicit Mapillary image IDs
python scripts/download_images.py --image-ids my_ids.txt --output data/images/
```

### 2. Run evaluation (pilot first!)
```bash
# Always pilot first to estimate cost
python scripts/evaluate.py --model gemini --input data/annotations.csv --output results/results_gemini.csv --pilot 10

# Full run (will prompt for confirmation)
python scripts/evaluate.py --model gemini --input data/annotations.csv --output results/results_gemini.csv
python scripts/evaluate.py --model claude --input data/annotations.csv --output results/results_claude.csv
python scripts/evaluate.py --model gpt4o  --input data/annotations.csv --output results/results_gpt4o.csv
```

Supported models: `gemini` (Gemini 2.5 Flash), `claude` (Claude Haiku 4.5), `gpt4o` (GPT-4o-mini).

Runs are resumable — if interrupted, re-running the same command skips already-completed images.

### 3. Re-parse results (if needed)

Re-extract country predictions from saved raw responses without re-querying APIs:
```bash
python scripts/parse_results.py --input results/results_gemini.csv --output results/results_gemini.csv --annotations data/annotations.csv
```

The `--annotations` flag adds region correctness data to the results.

### 4. Analyze results
```bash
python scripts/analyze.py --results results/ --annotations data/annotations.csv --output figures/
```

Pass `--annotations` to exclude multi-cue images from per-cue breakdowns (recommended). Use `--no-plots` for tables only.

## Project Structure

```
geovlm-bench/
├── data/
│   ├── images/            ← street-level images (JPEG), organized by country
│   └── annotations.csv    ← ground truth labels and cue categories
├── figures/               ← generated plots (PNG)
├── report/                ← JMLR-format report (LaTeX)
├── results/               ← model evaluation outputs (CSV)
├── scripts/
│   ├── build_annotations.py
│   ├── download_images.py
│   ├── evaluate.py
│   ├── parse_results.py
│   └── analyze.py
├── images.txt             ← image IDs grouped by country and cue type
├── requirements.txt
└── README.md
```

## Dataset

246 images across 13 countries and 9 regions:

| Region | Countries | Images |
|--------|-----------|--------|
| East Asia | Japan, South Korea | 39 |
| Southeast Asia | Thailand, Indonesia | 39 |
| South Asia | India | 20 |
| Western Europe | Germany, United Kingdom | 38 |
| Northern Europe | Finland | 19 |
| Eastern Europe | Russia | 19 |
| North America | Canada, Mexico | 37 |
| South America | Brazil | 18 |
| Middle East | Turkey | 17 |

## Cue Types

| Type | Description | Images |
|------|-------------|--------|
| `linguistic` | Visible text, road signs, language-specific signage | 80 |
| `environmental` | Vegetation, terrain, climate indicators | 73 |
| `infrastructure` | Road markings, driving side, utility poles, vehicle types | 93 |

42 images are annotated as multi-cue (containing multiple cue types). These are excluded from per-cue accuracy breakdowns and are analyzed separately.

## Output Metrics

- Overall country-level accuracy (per model)
- Per-cue-type accuracy (single-cue images only)
- Model x cue type breakdown (main result table)
- Region-level accuracy (partial credit for correct region)
- Multi-cue vs single-cue comparison
