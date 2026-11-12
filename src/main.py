from pathlib import Path

from analysis import correlation
from data_preb import dataset
from modelling import benchmark

# Project paths, resolved from this file so main.py works from any directory
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT_DIR / ".env"
RAW_DIR = ROOT_DIR / "data" / "raw"
TRAIN_FILE = RAW_DIR / "train.jsonl"
FACTORS_FILE = ROOT_DIR / "data" / "processed" / "factors_train.jsonl"
MERGED_FILE = ROOT_DIR / "data" / "processed" / "merged_train.jsonl"
OUTPUT_DIR = ROOT_DIR / "output"


def main():
    """Run the whole pipeline step by step."""
    # 1. Download the raw train/dev splits from Hugging Face (overwrites data/raw)
    dataset.download_dataset(RAW_DIR)

    # 2. Query the model for factor scores; already scored ids are skipped
    dataset.generate_factors(TRAIN_FILE, FACTORS_FILE, ENV_FILE)

    # 3. Check that every example has a factor output with a matching risk level
    dataset.validate_factors(TRAIN_FILE, FACTORS_FILE)

    # 4. Merge the train examples with their factor scores into a single file
    dataset.merge_factors(TRAIN_FILE, FACTORS_FILE, MERGED_FILE)

    # 5. Spearman correlation analysis and charts
    correlation.analyze_correlations(FACTORS_FILE, OUTPUT_DIR)

    # 6. Model benchmark: heatmap and summary
    benchmark.run_benchmark(FACTORS_FILE, OUTPUT_DIR)


if __name__ == "__main__":
    main()

