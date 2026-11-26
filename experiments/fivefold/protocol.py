"""Shared inputs, metrics and frozen voting rules for the five-fold experiment."""
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "qwen_reference"
sys.path.insert(0, str(HERE.parent / "qwen25"))
from data_utils import ROOT, TARGET_KEYS, read_jsonl, validate_target
from evaluate import jaccard

SYSTEM_PROMPT = (HERE / "system_prompt.txt").read_text().rstrip("\n")


def indexed(rows):
    result = {r["example_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate example_id")
    return result


def dataset():
    train, dev = (indexed(read_jsonl(ROOT / "data/raw" / f"{s}.jsonl")) for s in ("train", "dev"))
    folds = json.loads((REFERENCE / "folds_train_only.json").read_text())
    manifest = indexed(read_jsonl(REFERENCE / "split_manifest.jsonl"))
    if set(folds) != set(train) or train.keys() & dev.keys() or set(manifest) != train.keys() | dev.keys():
        raise ValueError("Fold, split or manifest IDs differ")
    if Counter(folds.values()) != dict(enumerate([117, 126, 96, 109, 120])):
        raise ValueError("Unexpected fold sizes")
    seen = {}
    for key, row in manifest.items():
        split = "train" if key in train else "dev"
        partition = folds.get(key, "dev")
        if row["split"] != split or seen.setdefault(row["scenario_group"], partition) != partition:
            raise ValueError(f"Split mismatch or scenario leakage at {key}")
    return train, dev, folds


def messages(row, answer=False):
    user = f"### Interaction Trajectory:\n{row['formatted_interaction']}\n\n### Proposed Action:\n{row['proposed_action']}"
    chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    if answer:
        target = {k: row[k] for k in TARGET_KEYS}
        chat.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False)})
    return chat


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as out:
        out.write(text)
    os.replace(out.name, path)


def save_json(path, obj):
    atomic_text(path, json.dumps(obj, indent=2, allow_nan=False) + "\n")


def save_rows(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def parse(text):
    # Retain raw text separately. A usable vote needs all three fields.
    try:
        obj = json.loads(text.strip())
        if type(obj["risk_severity"]) is not int or not 1 <= obj["risk_severity"] <= 5:
            return None
        for k in ("risk_domains", "risk_drivers"):
            if not isinstance(obj[k], list) or not all(isinstance(s, str) for s in obj[k]):
                return None
        return {k: obj[k] for k in TARGET_KEYS}
    except (ValueError, KeyError, TypeError):
        return None


def vote_labels(items, key, sentinel, minimum, majority):
    counts = Counter(label for p in items for label in set(p[key]))
    ranked = sorted(counts, key=lambda label: (-counts[label], label))
    if counts[sentinel] >= majority:
        return [sentinel]
    selected = [label for label in ranked if label != sentinel and counts[label] >= minimum]
    return selected[:2] or ranked[:1] or [sentinel]


def vote(records):
    # Use the original thresholds as proportions; ten adapters get 6/4/6 votes.
    n = len(records)
    if n not in (5, 10):
        raise ValueError("Expected five or ten adapter records per example")
    usable = [r for r in records if r["parsed"] is not None]
    if not usable:
        return None
    mass = [0.] * 5
    for record in usable:
        severity = record["parsed"]["risk_severity"]
        probs = record["severity_probs"]
        if probs is None:
            probs = [float(i == severity) for i in range(1, 6)]
        if len(probs) != 5 or any(not math.isfinite(x) or x < 0 for x in probs) or not math.isclose(sum(probs), 1., abs_tol=1e-5):
            raise ValueError(f"Invalid severity probabilities at {record['example_id']}")
        mass = [a + b for a, b in zip(mass, probs)]
    cumulative = 0.
    for severity, p in enumerate(mass, start=1):
        cumulative += p
        if cumulative >= len(usable) / 2:
            break
    parsed = [r["parsed"] for r in usable]
    return dict(risk_domains=vote_labels(parsed, "risk_domains", "No Material Harm", 3*n//5, n//2+1),
                risk_drivers=vote_labels(parsed, "risk_drivers", "No Material Risk", 2*n//5, n//2+1),
                risk_severity=severity)


def scores(predictions, gold):
    if predictions.keys() != gold.keys():
        raise ValueError("Prediction IDs must match the complete evaluation split")
    errors, domains, drivers = [], [], []
    malformed = schema_errors = exact = 0
    for key, row in gold.items():
        p = predictions[key]
        if p is None:
            malformed += 1
            errors.append(4)
            domains.append(0.)
            drivers.append(0.)
            continue
        try:
            validate_target(p)
        except ValueError:
            schema_errors += 1
        errors.append(abs(p["risk_severity"] - row["risk_severity"]))
        exact += int(p["risk_severity"] == row["risk_severity"])
        domains.append(jaccard(p["risk_domains"], row["risk_domains"]))
        drivers.append(jaccard(p["risk_drivers"], row["risk_drivers"]))
    n = len(gold)
    return dict(n=n, malformed=malformed, parsed_with_schema_errors=schema_errors,
                severity_mae=sum(errors)/n, severity_accuracy=exact/n,
                domains_jaccard=sum(domains)/n, drivers_jaccard=sum(drivers)/n)
