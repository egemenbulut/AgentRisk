import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEVERITY_KEY = "risk_severity"
DOMAINS_KEY = "risk_domains"
DRIVERS_KEY = "risk_drivers"
SEVERITY_LEVELS = [1, 2, 3, 4, 5]

# Human-readable metric names used in the charts and the summary
METRIC_LABELS = {
    "severity_mae": "risk_severity MAE (lower is better)",
    "domains_jaccard": "risk_domains Jaccard (higher is better)",
    "drivers_jaccard": "risk_drivers Jaccard (higher is better)",
}


def load_jsonl(path, id_key):
    """Return the records of a jsonl file as a dict keyed by id_key."""
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records[record[id_key]] = record
    return records


def to_label_set(labels):
    """Return a list of labels (or None / a single string) as a set of labels."""
    if labels is None:
        return set()
    if isinstance(labels, str):
        return {labels}
    return set(labels)


def jaccard(predicted, gold):
    """Jaccard similarity |P ∩ G| / |P ∪ G| of two label collections (1.0 if both are empty)."""
    predicted, gold = to_label_set(predicted), to_label_set(gold)
    union = predicted | gold
    if not union:
        return 1.0
    return len(predicted & gold) / len(union)


def to_severity(value):
    """Return a valid severity level 1-5 as int, or None if it is missing / invalid."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value in SEVERITY_LEVELS else None


def compare_predictions(pred_file, gold_file, pred_id_key="id", gold_id_key="example_id"):
    """Compare one prediction file with the gold file on the shared ids.

    Returns a per-example DataFrame with gold/predicted severity, absolute error
    and the risk_domains / risk_drivers Jaccard scores.
    """
    pred_file = Path(pred_file)
    gold_file = Path(gold_file)
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    if not gold_file.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_file}")

    predictions = load_jsonl(pred_file, pred_id_key)
    gold = load_jsonl(gold_file, gold_id_key)

    rows = []
    # Only the ids that exist in both files are compared
    for example_id in sorted(predictions.keys() & gold.keys()):
        pred, ref = predictions[example_id], gold[example_id]
        pred_severity = to_severity(pred.get(SEVERITY_KEY))
        gold_severity = to_severity(ref.get(SEVERITY_KEY))
        abs_error = (
            abs(pred_severity - gold_severity)
            if pred_severity is not None and gold_severity is not None
            else np.nan
        )
        rows.append(
            {
                "id": example_id,
                "gold_severity": gold_severity,
                "pred_severity": pred_severity,
                "severity_abs_error": abs_error,
                "domains_jaccard": jaccard(pred.get(DOMAINS_KEY), ref.get(DOMAINS_KEY)),
                "drivers_jaccard": jaccard(pred.get(DRIVERS_KEY), ref.get(DRIVERS_KEY)),
            }
        )
    return pd.DataFrame(rows)


def summarize(details):
    """Return the aggregated metrics of a per-example comparison DataFrame.

    MAE is computed over the examples with a valid predicted severity; the
    number of missing / invalid severities is reported separately.
    """
    return {
        "n_compared": len(details),
        "n_valid_severity": int(details["severity_abs_error"].notna().sum()),
        "n_missing_severity": int(details["severity_abs_error"].isna().sum()),
        "severity_mae": details["severity_abs_error"].mean(),
        "domains_jaccard": details["domains_jaccard"].mean(),
        "drivers_jaccard": details["drivers_jaccard"].mean(),
    }


def build_summary(details_by_model):
    """Build the aggregated summary DataFrame (one row per model) from per-model details."""
    summary = pd.DataFrame(
        {model: summarize(details) for model, details in details_by_model.items()}
    ).T
    count_columns = ["n_compared", "n_valid_severity", "n_missing_severity"]
    summary[count_columns] = summary[count_columns].astype(int)
    return summary


def compare_models(pred_files, gold_file, pred_id_key="id", gold_id_key="example_id"):
    """Compare each given prediction jsonl file with gold_file and print the summary.

    pred_files can be:
      - an iterable of file paths (model name is taken from each file's stem), or
      - a dict mapping custom model names to file paths.

    Returns (details_by_model, summary).
    """
    if isinstance(pred_files, dict):
        items = [(name, Path(path)) for name, path in pred_files.items()]
    else:
        items = [(Path(path).stem, Path(path)) for path in pred_files]

    if not items:
        raise ValueError("pred_files must contain at least one prediction file")

    details_by_model = {
        model_name: compare_predictions(pred_file, gold_file, pred_id_key, gold_id_key)
        for model_name, pred_file in items
    }
    summary = build_summary(details_by_model)
    print("Comparison against gold labels:")
    print(summary.round(4).to_string())
    return details_by_model, summary


def plot_metrics(summary, output_dir, file_name):
    """Save a bar chart of the three metrics per model as output_dir/<file_name>.png."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = summary.index.tolist()
    colors = plt.cm.tab10(np.arange(len(models)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (metric, title) in zip(axes, METRIC_LABELS.items()):
        values = summary[metric].values
        ax.bar(models, values, color=colors)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom")
        ax.set_title(title)
        # MAE ranges 0-4 for 1-5 labels, Jaccard ranges 0-1
        ax.set_ylim(0, (max(values) * 1.2 if metric == "severity_mae" else 1.1))
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.suptitle("Model predictions vs. gold (dev) labels")
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_name}.png", dpi=150)
    plt.close(fig)


def plot_jaccard_distribution(details_by_model, output_dir, file_name):
    """Save bubble plots of the per-example Jaccard scores per model as output_dir/<file_name>.png.

    Jaccard scores take only a few discrete values (0, 1/3, 0.5, 1, ...), so each
    distinct value is drawn as a bubble whose area and color show the share of
    examples with that score; the count and share are written next to it and the
    model mean is marked with a red diamond.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = list(details_by_model)
    max_bubble_area = 2500  # marker area (pt^2) of a value held by 100% of the examples
    fig, axes = plt.subplots(1, 2, figsize=(7 * 2, 7))
    for ax, metric in zip(axes, ["domains_jaccard", "drivers_jaccard"]):
        for x, model in enumerate(models):
            scores = details_by_model[model][metric].round(3)
            counts = scores.value_counts().sort_index()
            shares = counts / counts.sum()
            scatter = ax.scatter(
                np.full(len(counts), x),
                counts.index,
                s=shares * max_bubble_area,
                c=shares,
                cmap="viridis_r",
                vmin=0,
                vmax=1,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
                zorder=2,
            )
            for value, count, share in zip(counts.index, counts.values, shares.values):
                ax.annotate(
                    f"{count} ({share:.0%})",
                    (x, value),
                    xytext=(np.sqrt(share * max_bubble_area) / 2 + 4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                )
            ax.scatter(x, scores.mean(), marker="D", color="red", s=40, zorder=3,
                       label="mean" if x == 0 else None)
        ax.set_xticks(range(len(models)), models, rotation=20, ha="right")
        ax.set_xlim(-0.5, len(models) - 0.3)
        ax.set_ylim(-0.08, 1.08)
        ax.set_ylabel("Jaccard score")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"Per-example {METRIC_LABELS[metric]}")
    fig.colorbar(scatter, ax=axes, label="Share of examples", shrink=0.8)
    fig.suptitle("Distribution of per-example Jaccard scores (bubble size = share of examples)")
    fig.savefig(output_dir / f"{file_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_severity_confusion(details_by_model, output_dir, file_name):
    """Save gold vs. predicted risk_severity confusion matrices per model as output_dir/<file_name>.png."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = list(details_by_model)
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5), squeeze=False)
    for ax, model in zip(axes[0], models):
        valid = details_by_model[model].dropna(subset=["severity_abs_error"])
        matrix = pd.crosstab(valid["gold_severity"], valid["pred_severity"]).reindex(
            index=SEVERITY_LEVELS, columns=SEVERITY_LEVELS, fill_value=0
        )
        ax.imshow(matrix.values, cmap="Blues")
        for i in range(len(SEVERITY_LEVELS)):
            for j in range(len(SEVERITY_LEVELS)):
                ax.text(j, i, matrix.values[i, j], ha="center", va="center")
        ax.set_xticks(range(len(SEVERITY_LEVELS)), SEVERITY_LEVELS)
        ax.set_yticks(range(len(SEVERITY_LEVELS)), SEVERITY_LEVELS)
        ax.set_xlabel("Predicted risk_severity")
        ax.set_ylabel("Gold risk_severity")
        ax.set_title(model)
    fig.suptitle("risk_severity confusion matrices")
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_name}.png", dpi=150)
    plt.close(fig)


def write_summary(summary, output_dir, file_name):
    """Save the metric table as output_dir/<file_name>.txt."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Base vs. finetuned comparison against gold (dev) labels",
        "risk_severity: MAE over examples with a valid 1-5 prediction (lower is better)",
        "risk_domains / risk_drivers: mean Jaccard similarity (higher is better)",
        "",
        summary.round(4).to_string(),
        "",
        f"Best risk_severity MAE: {summary['severity_mae'].idxmin()}",
        f"Best risk_domains Jaccard: {summary['domains_jaccard'].idxmax()}",
        f"Best risk_drivers Jaccard: {summary['drivers_jaccard'].idxmax()}",
    ]
    (output_dir / f"{file_name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_details(details_by_model, output_dir, file_name):
    """Save per-example scores of every compared model as output_dir/<file_name>.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(details_by_model, names=["model"]).reset_index(level=0).to_csv(
        output_dir / f"{file_name}.csv", index=False
    )


def run_comparison(
    pred_files,
    gold_file,
    output_dir,
    metrics_chart_name,
    distribution_chart_name,
    confusion_chart_name,
    summary_name,
    details_name,
    pred_id_key="id",
    gold_id_key="example_id",
):
    """Compare the given prediction jsonl files with gold_file and save all outputs.

    pred_files can be a list of file paths or a dict mapping model names to file paths.

    Outputs written to output_dir:
      - <metrics_chart_name>.png:      bar chart of MAE / Jaccard per model
      - <distribution_chart_name>.png: per-example Jaccard bubble plots
      - <confusion_chart_name>.png:    risk_severity confusion matrices
      - <summary_name>.txt:            metric table
      - <details_name>.csv:            per-example scores of every model

    Returns (details_by_model, summary).
    """
    details_by_model, summary = compare_models(
        pred_files, gold_file, pred_id_key=pred_id_key, gold_id_key=gold_id_key
    )
    plot_metrics(summary, output_dir, metrics_chart_name)
    plot_jaccard_distribution(details_by_model, output_dir, distribution_chart_name)
    plot_severity_confusion(details_by_model, output_dir, confusion_chart_name)
    write_summary(summary, output_dir, summary_name)
    write_details(details_by_model, output_dir, details_name)
    return details_by_model, summary
