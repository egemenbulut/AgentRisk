from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Paths are resolved from the project root, so the script works from any directory
ROOT_DIR = Path(__file__).resolve().parents[2]
FACTORS_FILE = ROOT_DIR / "data" / "processed" / "factors_train.jsonl"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# Load the factor scores and the risk level of every example
df = pd.read_json(FACTORS_FILE, lines=True)

# Spearman correlation matrix over the factors and risk_level
corr = df[FACTORS + ["risk_level"]].corr(method="spearman")

# Correlation of each factor with risk_level
risk_corr = corr.loc[FACTORS, "risk_level"]
print("Spearman correlation with risk_level:")
print(risk_corr.round(3).to_string())

# Correlations between the factors themselves
factor_corr = corr.loc[FACTORS, FACTORS]
print("\nSpearman correlation between factors:")
print(factor_corr.round(3).to_string())

# Bar chart: each factor vs risk_level
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
fig.savefig(OUTPUT_DIR / "factor_risk_correlation.png", dpi=150)
plt.close(fig)

# Heatmap: factor-to-factor correlations
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
fig.savefig(OUTPUT_DIR / "factor_factor_correlation.png", dpi=150)
plt.close(fig)
