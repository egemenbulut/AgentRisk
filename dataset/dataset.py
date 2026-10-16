from pathlib import Path
from datasets import load_dataset

# Paths for local dataset files
CURRENT_DIR = Path(__file__).parent
TRAIN_FILE = CURRENT_DIR / "train.jsonl"
DEV_FILE = CURRENT_DIR / "dev.jsonl"

# Fetch latest dataset from Hugging Face Hub
print("Fetching 'AgentRisk/agentrisk-data-2027' dataset from Hugging Face...")
ds = load_dataset("AgentRisk/agentrisk-data-2027")

# Save/overwrite splits directly as JSONL files in this directory
print("Saving and overwriting local dataset files...")
ds["train"].to_pandas().to_json(TRAIN_FILE, orient="records", lines=True)
ds["dev"].to_pandas().to_json(DEV_FILE, orient="records", lines=True)

print(f"Done!\n- {TRAIN_FILE.name}\n- {DEV_FILE.name}")