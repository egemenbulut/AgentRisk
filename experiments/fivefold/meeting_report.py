"""Save and display matched interim scores from a live or completed run. CPU only."""
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from protocol import REFERENCE, ROOT, atomic_text, dataset, indexed, read_jsonl, scores


def make_report(root, log, data):
    train, dev, folds = data
    baseline = json.loads((REFERENCE / "metrics.json").read_text())["qwen_dev"]
    lines = ["AgentRisk meeting update",
             f"Existing Qwen five-adapter dev result (143 cases): MAE {baseline['severity_mae']:.3f}, "
             f"domain J {baseline['domains_jaccard']:.3f}, driver J {baseline['drivers_jaccard']:.3f}."]
    if log.exists():
        # The smoke test uses an isolated one-update run; exclude its progress.
        pattern = r"^Fold [0-4] update \d+/\d+ loss="
        active = False
        status = ""
        with log.open(errors="replace") as src:
            for line in src:
                if re.match(pattern, line):
                    active = not re.match(r"^Fold [0-4] update 1/1 loss=", line)
                if active and line.startswith("Fold ") and ("update " in line or " complete:" in line):
                    status = line.strip()
        if status:
            lines.append("Latest training log: " + status)
    sections = []
    for fold in range(5):
        folder = root / f"fold_{fold}"
        if not folder.is_dir():
            continue
        for name in ["preview_8", "preview_20", "val", "dev"]:
            split = "dev" if name == "dev" else "val"
            pool = dev if split == "dev" else {k:r for k,r in train.items() if folds[k] == fold}
            total = 24 if name.startswith("preview") else len(pool)
            # Writer renames the final result before deleting its partial snapshot.
            partial = folder / f"{name}.partial.jsonl"
            final = folder / f"{name}.jsonl"
            try:
                records = indexed(read_jsonl(final if final.exists() else partial))
            except FileNotFoundError:
                if not final.exists():
                    continue
                records = indexed(read_jsonl(final))
            if not records or not records.keys() <= pool.keys():
                raise ValueError(f"Unexpected predictions in {folder}/{name}")
            truth = {k:pool[k] for k in records}
            ref = indexed(read_jsonl(REFERENCE / "predictions" / f"qwen25_7b_train_only_fold_{fold}_{split}.jsonl"))
            label = (f"early update {name.split('_')[-1]}, fixed holdout sample" if name.startswith("preview")
                     else ("held-out training rows" if split == "val" else "development rows"))
            sections.append(f"Fold {fold}, {label} ({len(records)}/{total} scored)")
            for title, pred in [("Llama", records), ("Qwen", ref)]:
                res = scores({k:pred[k]["parsed"] for k in truth}, truth)
                sections.append(f"  {title:5} severity MAE {res['severity_mae']:.3f} | domain Jaccard "
                                f"{res['domains_jaccard']:.3f} | driver Jaccard {res['drivers_jaccard']:.3f} "
                                f"| malformed {res['malformed']}")
    if sections:
        lines.extend(["", "Matched single-adapter comparisons on identical cases.", *sections,
                      "Early previews are preliminary. No checkpoint or vote rule is chosen from these previews."])
    else:
        lines.append("Llama prediction scores pending; training/validation loss is a progress measure.")
    comparison = root / "comparison/metrics.json"
    if comparison.exists():
        lines.extend(["", "Final 143-case development comparison"])
        metrics = json.loads(comparison.read_text())
        for key in ("qwen_dev", "llama_dev", "qwen_llama_dev"):
            row = metrics[key]
            lines.append(f"  {key}: severity MAE {row['severity_mae']:.3f} | domain Jaccard "
                         f"{row['domains_jaccard']:.3f} | driver Jaccard {row['drivers_jaccard']:.3f}")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "output/llama-3b-fivefold")
    p.add_argument("--output", type=Path, help="Destination for the meeting summary")
    p.add_argument("--watch", action="store_true", help="Refresh until the ensemble completes or Ctrl-C")
    args = p.parse_args()
    data = dataset()
    report = args.output or args.root.parent / "meeting-summary.txt"
    previous = None
    while True:
        body = make_report(args.root, args.root.parent / "overnight.log", data)
        if previous != body:
            updated = datetime.now().astimezone().isoformat(timespec="minutes")
            text = f"Updated {updated}\n\n{body}"
            atomic_text(report, text)
            print(text, flush=True)
            previous = body
        if not args.watch or (args.root / "comparison/metrics.json").exists():
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
