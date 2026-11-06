from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

# Paths are resolved from the project root, so the script works from any directory
ROOT_DIR = Path(__file__).resolve().parents[2]
FACTORS_FILE = ROOT_DIR / "data" / "processed" / "factors_train.jsonl"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Feature subsets to compare
FEATURE_SUBSETS = {
    # Forward selection: factors added one by one in order of their Spearman
    # correlation with risk_level (f2 > f1 > f7 > f6 > f3 > f5 > f4)
    "Single Top": ["f2"],
    "Top-2": ["f2", "f1"],
    "Top-3": ["f2", "f1", "f7"],
    "Top-4": ["f2", "f1", "f7", "f6"],
    "Top-5": ["f2", "f1", "f7", "f6", "f3"],
    "Top-6": ["f2", "f1", "f7", "f6", "f3", "f5"],
    "All Factors": ["f1", "f2", "f3", "f4", "f5", "f6", "f7"],
    # Weakly correlated factors that carry information not shared by f1/f2/f6/f7
    "Weak Only": ["f3", "f4", "f5"],
    "f2 + Weak": ["f2", "f3", "f4", "f5"],
    # Ablation: how well the remaining factors do without the strongest one
    "All except f2": ["f1", "f3", "f4", "f5", "f6", "f7"],
}


def build_models():
    """Return (family, model name, params, unfitted model) for every scenario."""
    models = []

    # Linear models whose coefficients are constrained to be non-negative
    models.append(("Positive Linear", "NNLS", {}, LinearRegression(positive=True)))
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        models.append(("Positive Linear", "Ridge", {"alpha": alpha},
                       Ridge(alpha=alpha, positive=True)))
    for alpha in [0.001, 0.01, 0.1]:
        models.append(("Positive Linear", "Lasso", {"alpha": alpha},
                       Lasso(alpha=alpha, positive=True)))

    # Shallow decision trees
    for depth in [2, 3, 4]:
        models.append(("Tree", "DecisionTree", {"max_depth": depth},
                       DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE)))

    # Gradient boosting grid over learning rate, number of trees and tree depth
    for lr in [0.01, 0.03, 0.05, 0.1]:
        for n_est in [100, 300, 500]:
            for depth in [2, 3, 4]:
                params = {"learning_rate": lr, "n_estimators": n_est, "max_depth": depth}
                models.append(("Gradient Boosting", "GradientBoosting", params,
                               GradientBoostingRegressor(**params, random_state=RANDOM_STATE)))
    return models


def to_risk_level(pred):
    """Clip continuous predictions to 1-5 and round half up to the nearest integer."""
    return np.floor(np.clip(pred, 1, 5) + 0.5).astype(int)


# Load the factor scores
df = pd.read_json(FACTORS_FILE, lines=True)

# Stratified 80/20 split that preserves the risk_level distribution
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["risk_level"], random_state=RANDOM_STATE
)
y_train = train_df["risk_level"]
y_test = test_df["risk_level"]

# Train and evaluate every model on every feature subset
results = []
for subset_name, features in FEATURE_SUBSETS.items():
    X_train = train_df[features]
    X_test = test_df[features]
    for family, model_name, params, model in build_models():
        model.fit(X_train, y_train)
        y_pred = to_risk_level(model.predict(X_test))
        # Row label for the chart; the boosting grid is collapsed into a single row
        if family == "Gradient Boosting":
            config = "GradientBoosting (best of grid)"
        else:
            config = model_name + "".join(f" {k}={v}" for k, v in params.items())
        results.append({
            "model": model_name,
            "family": family,
            "feature_subset": subset_name,
            "params": str(params) if params else "-",
            "test_mae": np.mean(np.abs(y_pred - y_test.to_numpy())),
            # Kept only for the chart and the champion summary, not written to the leaderboard
            "_config": config,
            "_model": model,
            "_features": features,
        })

# Leaderboard sorted from best (lowest MAE) to worst; stable sort keeps the
# simpler models first when scores are tied
leaderboard = pd.DataFrame(results).sort_values("test_mae", kind="stable").reset_index(drop=True)
leaderboard.index += 1
table = leaderboard.drop(columns=["_config", "_model", "_features"])

# Heatmap of Test MAE: model configuration (rows) x feature subset (columns)
row_order = list(dict.fromkeys(r["_config"] for r in results))
heat = leaderboard.pivot_table(
    index="_config", columns="feature_subset", values="test_mae", aggfunc="min"
).loc[row_order, list(FEATURE_SUBSETS)]
best_mae = heat.values.min()

fig, ax = plt.subplots(figsize=(15, 8))
im = ax.imshow(heat.values, cmap="RdYlGn_r", aspect="auto")
# Rotate the subset names so the wider set of columns stays readable
ax.set_xticks(range(heat.shape[1]), heat.columns, rotation=30, ha="right")
ax.set_yticks(range(heat.shape[0]), heat.index)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        value = heat.values[i, j]
        # Cells that reach the best score are shown in bold
        ax.text(j, i, f"{value:.4f}", ha="center", va="center",
                fontweight="bold" if value == best_mae else "normal")
fig.colorbar(im, ax=ax, label="Test MAE (lower is better)")
ax.set_title(f"Test MAE by model and feature subset (test size = {len(test_df)})\n"
             f"Bold = best score ({best_mae:.4f})")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "benchmark_results.png", dpi=150)
plt.close(fig)

# Champion model (first row of the leaderboard)
champion = leaderboard.iloc[0]
model = champion["_model"]
features = champion["_features"]

lines = [
    f"Train size: {len(train_df)}, Test size: {len(test_df)}",
    "",
    "=== Champion Model ===",
    f"Model         : {champion['model']} ({champion['family']})",
    f"Feature subset: {champion['feature_subset']} -> {features}",
    f"Params        : {champion['params']}",
    f"Test MAE      : {champion['test_mae']:.4f}",
]
if champion["family"] == "Positive Linear":
    # Learned non-negative weights and the intercept
    lines.append("Weights:")
    lines += [f"  {name}: {coef:.4f}" for name, coef in zip(features, model.coef_)]
    lines.append(f"  intercept: {model.intercept_:.4f}")
elif champion["model"] == "DecisionTree":
    # Human-readable split rules of the tree
    lines.append("Tree rules:")
    lines.append(export_text(model, feature_names=features))
else:
    # Boosted ensembles are too large to list, so show impurity-based importances
    lines.append("Feature importances:")
    lines += [f"  {name}: {imp:.4f}" for name, imp in zip(features, model.feature_importances_)]

lines += [
    "",
    "=== Leaderboard ===",
    table.to_string(float_format=lambda x: f"{x:.4f}"),
]

# Save the champion summary and the full leaderboard to a text file
(OUTPUT_DIR / "benchmark_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
