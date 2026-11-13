from pathlib import Path

from analysis import correlation
from data_preb import dataset
from modelling import benchmark

# Project paths, resolved from this file so main.py works from any directory
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT_DIR / ".env"
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "output"

# File names without extension (the modules add .jsonl / .png / .txt)
TRAIN_NAME = "train"
DEV_NAME = "dev"
FACTORS_NAME = "factors_train"
MERGED_NAME = "merged_train"
RISK_CHART_NAME = "factor_risk_correlation"
FACTOR_CHART_NAME = "factor_factor_correlation"
HEATMAP_NAME = "benchmark_results"
SUMMARY_NAME = "benchmark_summary"


def main():
    """Run the whole pipeline step by step."""
    # 1. Download the raw train/dev splits from Hugging Face (overwrites RAW_DIR)
    train_file, dev_file = dataset.download_dataset(RAW_DIR, TRAIN_NAME, DEV_NAME)

    # 2. Query the model for factor scores; already scored ids are skipped
    factors_file = dataset.generate_factors(train_file, PROCESSED_DIR, FACTORS_NAME, ENV_FILE)

    # 3. Check that every example has a factor output with a matching risk level
    dataset.validate_factors(train_file, factors_file)

    # 4. Merge the train examples with their factor scores into a single file
    dataset.merge_factors(train_file, factors_file, PROCESSED_DIR, MERGED_NAME)

    # 5. Spearman correlation analysis and charts; returns the factors ranked by correlation
    _, _, ranked_factors = correlation.analyze_correlations(
        factors_file, OUTPUT_DIR, RISK_CHART_NAME, FACTOR_CHART_NAME
    )

    # 6. Model benchmark on feature subsets derived from the ranking: heatmap and summary
    benchmark.run_benchmark(factors_file, ranked_factors, OUTPUT_DIR, HEATMAP_NAME, SUMMARY_NAME)


if __name__ == "__main__":
    main()
