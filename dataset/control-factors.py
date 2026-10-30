import json

# Map each id in factors.jsonl to its risk_level
risk_levels = {}
with open("factors.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            risk_levels[row["id"]] = row["risk_level"]

# Lists that collect the problematic example ids
missing = []
mismatch = []

# Check every example in train.jsonl against factors.jsonl
with open("train.jsonl", "r", encoding="utf-8") as f:
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

