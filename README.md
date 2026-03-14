# GeoVLM-Bench

A benchmark for evaluating geographic visual reasoning in vision-language models (VLMs), structured around cue type annotations (linguistic, environmental, infrastructure).

## Setup

```bash
pip install -r requirements.txt
```

Set API keys as environment variables:
```bash
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
export TOGETHER_API_KEY=your_key      # for open-weight model
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
python scripts/evaluate.py --model claude --input data/annotations.csv --output results/results_claude.csv --pilot 10

# Full run (will prompt for confirmation)
python scripts/evaluate.py --model claude --input data/annotations.csv --output results/results_claude.csv
python scripts/evaluate.py --model gpt4o  --input data/annotations.csv --output results/results_gpt4o.csv
python scripts/evaluate.py --model llava  --input data/annotations.csv --output results/results_openweight.csv
```

### 3. Re-parse results (if needed)
```bash
python scripts/parse_results.py --input results/results_claude.csv --output results/results_claude.csv
```

### 4. Analyze results
```bash
python scripts/analyze.py --results results/ --output figures/
```

## Project Structure

```
geovlm-bench/
├── data/
│   ├── images/            ← street-level images (JPEG)
│   └── annotations.csv    ← ground truth labels and cue categories
├── figures/               ← generated plots (PNG)
├── results/               ← model evaluation outputs (CSV)
├── scripts/
│   ├── download_images.py
│   ├── evaluate.py
│   ├── parse_results.py
│   └── analyze.py
├── requirements.txt
└── README.md
```

## Cue Types

| Type | Description |
|------|-------------|
| `linguistic` | Visible text, road signs, language-specific signage |
| `environmental` | Vegetation, terrain, climate indicators |
| `infrastructure` | Road markings, driving side, utility poles, vehicle types |

## Output Metrics

- Overall country-level accuracy
- Per-model accuracy
- Per-cue-type accuracy
- Model × cue type breakdown (main result table)
- Multi-cue vs single-cue comparison
