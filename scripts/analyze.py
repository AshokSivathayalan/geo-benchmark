"""
analyze.py — Compute accuracy tables and generate plots for GeoVLM-Bench results.

Usage:
    python scripts/analyze.py --results results/ --output figures/
    python scripts/analyze.py --results results/ --output figures/ --no-plots
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

CUE_TYPES = ["linguistic", "environmental", "infrastructure"]
CUE_PALETTE = {
    "linguistic": "#4C72B0",
    "environmental": "#55A868",
    "infrastructure": "#C44E52",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate all results CSVs from a directory.

    Expects files matching the pattern results_{model}.csv.

    Args:
        results_dir: Directory containing results CSV files.

    Returns:
        Combined DataFrame with all model results.

    Raises:
        FileNotFoundError: If no results files are found.
    """
    files = list(results_dir.glob("results_*.csv"))
    if not files:
        raise FileNotFoundError(f"No results_*.csv files found in {results_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        logger.info(f"Loaded {len(df)} rows from {f.name}")

    combined = pd.concat(dfs, ignore_index=True)
    combined["correct"] = combined["correct"].astype(bool)
    logger.info(f"Total rows loaded: {len(combined)}")
    return combined


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

def overall_accuracy(df: pd.DataFrame) -> float:
    """Return overall accuracy across all models and images."""
    return df["correct"].mean()


def accuracy_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-model accuracy.

    Args:
        df: Combined results DataFrame.

    Returns:
        DataFrame with columns [model, accuracy, n_images].
    """
    grouped = df.groupby("model")["correct"].agg(["mean", "count"]).reset_index()
    grouped.columns = ["model", "accuracy", "n_images"]
    return grouped.sort_values("accuracy", ascending=False)


def accuracy_by_cue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-cue-type accuracy (aggregated across all models).

    Args:
        df: Combined results DataFrame.

    Returns:
        DataFrame with columns [cue_type, accuracy, n_images].
    """
    grouped = df.groupby("cue_type")["correct"].agg(["mean", "count"]).reset_index()
    grouped.columns = ["cue_type", "accuracy", "n_images"]
    return grouped


def accuracy_by_model_and_cue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy broken down by model × cue_type (the main result table).

    Args:
        df: Combined results DataFrame.

    Returns:
        Pivot table with models as rows and cue types as columns.
    """
    grouped = df.groupby(["model", "cue_type"])["correct"].mean().reset_index()
    grouped.columns = ["model", "cue_type", "accuracy"]
    pivot = grouped.pivot(index="model", columns="cue_type", values="accuracy")

    # Ensure all cue types present even if missing in data
    for cue in CUE_TYPES:
        if cue not in pivot.columns:
            pivot[cue] = float("nan")

    # Add overall column
    pivot["overall"] = df.groupby("model")["correct"].mean()
    return pivot[CUE_TYPES + ["overall"]].sort_values("overall", ascending=False)


def accuracy_multicue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy for multi-cue vs single-cue images, per model.

    Requires a 'multi_cue' column. If absent, returns an empty DataFrame.

    Args:
        df: Combined results DataFrame.

    Returns:
        DataFrame comparing multi-cue and single-cue accuracy per model.
    """
    if "multi_cue" not in df.columns:
        logger.warning("'multi_cue' column not found — skipping multi-cue analysis.")
        return pd.DataFrame()

    df = df.copy()
    df["multi_cue"] = df["multi_cue"].astype(str).str.lower().isin(["true", "1", "yes"])
    grouped = df.groupby(["model", "multi_cue"])["correct"].agg(["mean", "count"]).reset_index()
    grouped.columns = ["model", "multi_cue", "accuracy", "n_images"]
    grouped["subset"] = grouped["multi_cue"].map({True: "multi-cue", False: "single-cue"})
    return grouped[["model", "subset", "accuracy", "n_images"]]


# ---------------------------------------------------------------------------
# Printing tables
# ---------------------------------------------------------------------------

def print_accuracy_tables(df: pd.DataFrame) -> None:
    """Print all accuracy breakdowns to stdout."""
    print("\n" + "=" * 60)
    print(f"OVERALL ACCURACY: {overall_accuracy(df):.1%}  (n={len(df)})")
    print("=" * 60)

    print("\n--- Per-Model Accuracy ---")
    print(accuracy_by_model(df).to_string(index=False, float_format="{:.1%}".format))

    print("\n--- Per-Cue-Type Accuracy (all models) ---")
    print(accuracy_by_cue(df).to_string(index=False, float_format="{:.1%}".format))

    print("\n--- Main Result Table: Model × Cue Type ---")
    pivot = accuracy_by_model_and_cue(df)
    print(pivot.applymap(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A").to_string())

    multicue = accuracy_multicue(df)
    if not multicue.empty:
        print("\n--- Multi-Cue vs Single-Cue Accuracy ---")
        print(multicue.to_string(index=False, float_format="{:.1%}".format))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_accuracy_by_model(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Bar chart of overall accuracy per model.

    Args:
        df: Combined results DataFrame.
        output_dir: Directory to save the PNG figure.
    """
    data = accuracy_by_model(df)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(data["model"], data["accuracy"], color="#4C72B0", edgecolor="white")
    ax.bar_label(bars, fmt="{:.1%}", padding=3, fontsize=9)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("Country-Level Accuracy by Model")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, min(1.0, data["accuracy"].max() + 0.15))
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()

    out = output_dir / "accuracy_by_model.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")


def plot_accuracy_by_cue(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Bar chart of accuracy per cue type, aggregated across models.

    Args:
        df: Combined results DataFrame.
        output_dir: Directory to save the PNG figure.
    """
    data = accuracy_by_cue(df)
    colors = [CUE_PALETTE.get(c, "#888888") for c in data["cue_type"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(data["cue_type"], data["accuracy"], color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="{:.1%}", padding=3, fontsize=9)
    ax.set_xlabel("Cue Type")
    ax.set_ylabel("Accuracy")
    ax.set_title("Country-Level Accuracy by Cue Type (All Models)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, min(1.0, data["accuracy"].max() + 0.15))
    plt.tight_layout()

    out = output_dir / "accuracy_by_cue.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")


def plot_model_by_cue_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Heatmap of accuracy for each model × cue type combination.

    Args:
        df: Combined results DataFrame.
        output_dir: Directory to save the PNG figure.
    """
    pivot = accuracy_by_model_and_cue(df)
    pivot_display = pivot[CUE_TYPES]  # Exclude 'overall' column from heatmap

    fig, ax = plt.subplots(figsize=(7, max(3, len(pivot) * 0.8 + 1.5)))
    sns.heatmap(
        pivot_display,
        annot=True,
        fmt=".1%",
        cmap="YlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"format": mticker.PercentFormatter(xmax=1)},
    )
    ax.set_title("Accuracy Heatmap: Model × Cue Type")
    ax.set_xlabel("Cue Type")
    ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()

    out = output_dir / "heatmap_model_by_cue.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")


def plot_grouped_by_model_and_cue(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Grouped bar chart: one group per model, bars per cue type.

    Args:
        df: Combined results DataFrame.
        output_dir: Directory to save the PNG figure.
    """
    grouped = df.groupby(["model", "cue_type"])["correct"].mean().reset_index()
    grouped.columns = ["model", "cue_type", "accuracy"]

    fig, ax = plt.subplots(figsize=(9, 5))
    models = grouped["model"].unique()
    x = range(len(models))
    width = 0.25
    offsets = [-width, 0, width]

    for i, cue in enumerate(CUE_TYPES):
        cue_data = grouped[grouped["cue_type"] == cue]
        cue_data = cue_data.set_index("model").reindex(models)
        bars = ax.bar(
            [xi + offsets[i] for xi in x],
            cue_data["accuracy"].fillna(0),
            width=width,
            label=cue.capitalize(),
            color=CUE_PALETTE[cue],
            edgecolor="white",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Model and Cue Type")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.0)
    ax.legend(title="Cue Type")
    plt.tight_layout()

    out = output_dir / "accuracy_grouped_model_cue.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")


def plot_multicue(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Bar chart comparing multi-cue vs single-cue accuracy per model.

    Args:
        df: Combined results DataFrame.
        output_dir: Directory to save the PNG figure.
    """
    data = accuracy_multicue(df)
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = data.pivot(index="model", columns="subset", values="accuracy")
    pivot.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"], edgecolor="white")
    ax.set_ylabel("Accuracy")
    ax.set_title("Multi-Cue vs Single-Cue Accuracy by Model")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Subset")
    plt.tight_layout()

    out = output_dir / "accuracy_multicue.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")


def generate_all_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate all analysis plots and save to output_dir.

    Args:
        df: Combined results DataFrame.
        output_dir: Directory to save PNG figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy_by_model(df, output_dir)
    plot_accuracy_by_cue(df, output_dir)
    plot_model_by_cue_heatmap(df, output_dir)
    plot_grouped_by_model_and_cue(df, output_dir)
    plot_multicue(df, output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="GeoVLM-Bench analysis: accuracy tables and plots.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Directory containing results_*.csv files (default: results/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures"),
        help="Directory to save generated plots (default: figures/).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation; print tables only.",
    )
    args = parser.parse_args()

    df = load_all_results(args.results)
    print_accuracy_tables(df)

    if not args.no_plots:
        generate_all_plots(df, args.output)
        print(f"\nPlots saved to: {args.output}/")


if __name__ == "__main__":
    main()
