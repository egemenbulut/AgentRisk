import matplotlib.pyplot as plt
import pandas as pd

FACTORS = ["f1", "f2", "f3", "f4", "f5", "f6", "f7"]

# Human-readable factor names used as axis labels in the charts
LABELS = [
    "Likelihood / Uncertainty",
    "Harm Intensity",
    "Scale",
    "Target Susceptibility",
    "System Dependency",
    "Oversight Deficiency",
    "Reversibility Difficulty",
]


def compute_correlations(factors_file):
    """Return the Spearman correlations of the factors with risk_level and with each other."""
    # Load the factor scores and the risk level of every example
    df = pd.read_json(factors_file, lines=True)

    # Spearman correlation matrix over the factors and risk_level
    corr = df[FACTORS + ["risk_level"]].corr(method="spearman")

    risk_corr = corr.loc[FACTORS, "risk_level"]
    factor_corr = corr.loc[FACTORS, FACTORS]
    return risk_corr, factor_corr


def plot_factor_risk_correlation(risk_corr, output_dir, file_name):
    """Save a bar chart of each factor's correlation with risk_level as output_dir/<file_name>.png."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(LABELS, risk_corr.values, color="steelblue")
    for i, value in enumerate(risk_corr.values):
        ax.text(i, value, f"{value:.2f}", ha="center", va="bottom")
    ax.set_ylim(-1, 1.1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Spearman correlation of factors with risk_level")
    # Rotate the long factor names so they do not overlap
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_name}.png", dpi=150)
    plt.close(fig)


def plot_factor_factor_correlation(factor_corr, output_dir, file_name):
    """Save a heatmap of the factor-to-factor correlations as output_dir/<file_name>.png."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(factor_corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FACTORS)), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(FACTORS)), LABELS)
    for i in range(len(FACTORS)):
        for j in range(len(FACTORS)):
            ax.text(j, i, f"{factor_corr.values[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Spearman correlation between factors")
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_name}.png", dpi=150)
    plt.close(fig)


def analyze_correlations(factors_file, output_dir, risk_chart_name, factor_chart_name):
    """Compute the Spearman correlations, print them and save both charts to output_dir.

    Returns risk_corr, factor_corr and ranked_factors: the factor names sorted
    from the strongest to the weakest correlation with risk_level.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    risk_corr, factor_corr = compute_correlations(factors_file)

    print("Spearman correlation with risk_level:")
    print(risk_corr.round(3).to_string())
    print("\nSpearman correlation between factors:")
    print(factor_corr.round(3).to_string())

    plot_factor_risk_correlation(risk_corr, output_dir, risk_chart_name)
    plot_factor_factor_correlation(factor_corr, output_dir, factor_chart_name)

    # Factors ordered by their correlation with risk_level (strongest first)
    ranked_factors = risk_corr.sort_values(ascending=False, kind="stable").index.tolist()
    print("\nFactors ranked by correlation with risk_level:", ranked_factors)
    return risk_corr, factor_corr, ranked_factors
