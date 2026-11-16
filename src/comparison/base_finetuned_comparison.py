import json

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


def plot_metrics(summary, output_dir, file_name):
    """Save a bar chart of the three metrics per model as output_dir/<file_name>.png."""
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
    """Save box plots of the per-example Jaccard scores per model as output_dir/<file_name>.png."""
    models = list(details_by_model)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric in zip(axes, ["domains_jaccard", "drivers_jaccard"]):
        data = [details_by_model[m][metric].values for m in models]
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(models) + 1), models, rotation=20, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Per-example {METRIC_LABELS[metric]}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_name}.png", dpi=150)
    plt.close(fig)


def plot_severity_confusion(details_by_model, output_dir, file_name):
    """Save gold vs. predicted risk_severity confusion matrices per model as output_dir/<file_name>.png."""
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


def run_comparison(
    pred_dir,
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
    """Compare every jsonl file in pred_dir with gold_file, print the metrics and save the outputs.

    Outputs written to output_dir:
      - <metrics_chart_name>.png:      bar chart of MAE / Jaccard per model
      - <distribution_chart_name>.png: per-example Jaccard box plots
      - <confusion_chart_name>.png:    risk_severity confusion matrices
      - <summary_name>.txt:            metric table
      - <details_name>.csv:            per-example scores of every model

    Returns the summary DataFrame (one row per prediction file).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    details_by_model = {}
    for pred_file in sorted(pred_dir.glob("*.jsonl")):
        details_by_model[pred_file.stem] = compare_predictions(
            pred_file, gold_file, pred_id_key, gold_id_key
        )
    if not details_by_model:
        raise FileNotFoundError(f"No .jsonl prediction files found in {pred_dir}")

    summary = pd.DataFrame(
        {model: summarize(details) for model, details in details_by_model.items()}
    ).T
    count_columns = ["n_compared", "n_valid_severity", "n_missing_severity"]
    summary[count_columns] = summary[count_columns].astype(int)
    print("Comparison against gold labels:")
    print(summary.round(4).to_string())

    plot_metrics(summary, output_dir, metrics_chart_name)
    plot_jaccard_distribution(details_by_model, output_dir, distribution_chart_name)
    plot_severity_confusion(details_by_model, output_dir, confusion_chart_name)
    write_summary(summary, output_dir, summary_name)
    pd.concat(details_by_model, names=["model"]).reset_index(level=0).to_csv(
        output_dir / f"{details_name}.csv", index=False
    )
    return summary
