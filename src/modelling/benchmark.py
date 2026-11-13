import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

RANDOM_STATE = 42

def build_feature_subsets(ranked_factors, n_weak=3):
    """Derive the feature subsets to compare from the correlation ranking.

    ranked_factors is the list returned by correlation.analyze_correlations,
    ordered from the strongest to the weakest correlation with risk_level.
    """
    top = ranked_factors[0]
    weak = sorted(ranked_factors[-n_weak:])
    subsets = {}
    # Forward selection: factors added one by one in order of their correlation
    subsets["Single Top"] = ranked_factors[:1]
    for k in range(2, len(ranked_factors)):
        subsets[f"Top-{k}"] = ranked_factors[:k]
    subsets["All Factors"] = sorted(ranked_factors)
    # Weakly correlated factors that may carry information not shared by the strong ones
    subsets["Weak Only"] = weak
    if top not in weak:
        subsets[f"{top} + Weak"] = [top] + weak
    # Ablation: how well the remaining factors do without the strongest one
    subsets[f"All except {top}"] = sorted(f for f in ranked_factors if f != top)
    return subsets


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


def evaluate_models(train_df, test_df, feature_subsets):
    """Train and evaluate every model on every feature subset and return one result per scenario."""
    y_train = train_df["risk_level"]
    y_test = test_df["risk_level"]

    results = []
    for subset_name, features in feature_subsets.items():
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
    return results


def build_leaderboard(results):
    """Return all scenarios sorted from best (lowest MAE) to worst, ranked from 1.

    The stable sort keeps the simpler models first when scores are tied.
    """
    leaderboard = pd.DataFrame(results).sort_values("test_mae", kind="stable").reset_index(drop=True)
    leaderboard.index += 1
    return leaderboard


def plot_heatmap(results, feature_subsets, test_size, output_dir, file_name):
    """Save a heatmap of Test MAE (model configuration x feature subset) as output_dir/<file_name>.png."""
    # Keep the rows in the order the models are defined in build_models
    row_order = list(dict.fromkeys(r["_config"] for r in results))
    heat = pd.DataFrame(results).pivot_table(
        index="_config", columns="feature_subset", values="test_mae", aggfunc="min"
    ).loc[row_order, list(feature_subsets)]
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
    ax.set_title(f"Test MAE by model and feature subset (test size = {test_size})\n"
                 f"Bold = best score ({best_mae:.4f})")
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_name}.png", dpi=150)
    plt.close(fig)


def format_champion(champion):
    """Format a champion row with its learned weights, tree rules, or feature importances."""
    model = champion["_model"]
    features = champion["_features"]
    lines = [
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
        lines.append(export_text(model, feature_names=features).rstrip())
    else:
        # Boosted ensembles are too large to list, so show impurity-based importances
        lines.append("Feature importances:")
        lines += [f"  {name}: {imp:.4f}" for name, imp in zip(features, model.feature_importances_)]
    return lines


def write_summary(leaderboard, train_size, test_size, output_dir, file_name):
    """Save the champion model summary and the full leaderboard to output_dir/<file_name>.txt."""
    # Overall champion (first row) and best linear model on all factors
    overall_champion = leaderboard.iloc[0]
    linear_all_champion = leaderboard[
        (leaderboard["family"] == "Positive Linear")
        & (leaderboard["feature_subset"] == "All Factors")
    ].iloc[0]

    lines = [
        f"Train size: {train_size}, Test size: {test_size}",
        "",
        "=== Champion Model ===",
        *format_champion(overall_champion),
        "",
        *format_champion(linear_all_champion),
        "",
    ]

    table = leaderboard.drop(columns=["_config", "_model", "_features"])
    lines += [
        "=== Leaderboard ===",
        table.to_string(float_format=lambda x: f"{x:.4f}"),
    ]
    (output_dir / f"{file_name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(factors_file, ranked_factors, output_dir, heatmap_name, summary_name):
    """Run the full benchmark on factors_file and save the heatmap and summary to output_dir.

    The feature subsets are derived from ranked_factors (see build_feature_subsets).
    """
    feature_subsets = build_feature_subsets(ranked_factors)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the factor scores
    df = pd.read_json(factors_file, lines=True)

    # Stratified 80/20 split that preserves the risk_level distribution
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["risk_level"], random_state=RANDOM_STATE
    )

    results = evaluate_models(train_df, test_df, feature_subsets)
    leaderboard = build_leaderboard(results)
    plot_heatmap(results, feature_subsets, len(test_df), output_dir, heatmap_name)
    write_summary(leaderboard, len(train_df), len(test_df), output_dir, summary_name)
    return leaderboard
