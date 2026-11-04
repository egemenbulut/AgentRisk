import json
from pathlib import Path

# Paths are resolved from the project root, so the script works from any directory
ROOT_DIR = Path(__file__).resolve().parents[2]
TRAIN_FILE = ROOT_DIR / "data" / "raw" / "train.jsonl"
FACTORS_FILE = ROOT_DIR / "data" / "processed" / "factors_train.jsonl"

# Map each id in factors_train.jsonl to its risk_level
risk_levels = {}
with open(FACTORS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            risk_levels[row["id"]] = row["risk_level"]

# Lists that collect the problematic example ids
missing = []
mismatch = []

# Check every example in train.jsonl against factors_train.jsonl
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = row["example_id"]
        if example_id not in risk_levels:
            # No output exists for this example
            missing.append(example_id)
        elif risk_levels[example_id] != row["risk_severity"]:
            # risk_level does not match the ground-truth risk_severity
            mismatch.append(example_id)

# Print each list separately after the check is finished
print("MISSING:", missing)
print("MISMATCH:", mismatch)

