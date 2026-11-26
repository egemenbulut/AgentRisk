"""Replay Egemen's five-fold result, then compare Llama and the ten-adapter vote."""
import argparse
import json
from pathlib import Path

from protocol import REFERENCE, ROOT, dataset, indexed, read_jsonl, save_json, save_rows, scores, vote


def family(root, reference, train, dev, folds):
    oof, voters, per_fold = {}, [], {}
    for fold in range(5):
        val_ids = {k for k in train if folds[k] == fold}
        for split, expected in [("val", val_ids), ("dev", set(dev))]:
            if reference:
                path = root / "predictions" / f"qwen25_7b_train_only_fold_{fold}_{split}.jsonl"
            else:
                path = root / f"fold_{fold}" / f"{split}.jsonl"
            items = indexed(read_jsonl(path))
            if set(items) != expected:
                raise ValueError(f"{path}: IDs differ from the saved split")
            parsed = {k: row["parsed"] for k, row in items.items()}
            per_fold[f"fold_{fold}_{split}"] = scores(parsed, {k: (dev if split == "dev" else train)[k] for k in items})
            if split == "val":
                oof.update(parsed)
            else:
                voters.append(items)
    return scores(oof, train), voters, per_fold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama", type=Path, help="Run directory containing five completed Llama folds")
    parser.add_argument("--output", type=Path, default=ROOT / "output/fivefold-comparison")
    args = parser.parse_args()
    train, dev, folds = dataset()
    q_oof, q_votes, q_folds = family(REFERENCE, True, train, dev, folds)
    families = {"qwen": q_votes}
    report = {"qwen_oof": q_oof, "per_fold_qwen": q_folds}
    if args.llama:
        configs = [json.loads((args.llama / f"fold_{f}" / "done.json").read_text())["config"] for f in range(5)]
        if any(c != configs[0] for c in configs) or configs[0]["smoke"]:
            raise ValueError("Five completed production folds with matching settings are required")
        l_oof, l_votes, l_folds = family(args.llama, False, train, dev, folds)
        report.update(llama_config=configs[0], llama_oof=l_oof, per_fold_llama=l_folds)
        families.update(llama=l_votes, qwen_llama=q_votes + l_votes)
    for name, voters in families.items():
        preds = {key: vote([v[key] for v in voters]) for key in dev}
        if name == "qwen":
            saved = {r["id"]: {k: v for k, v in r.items() if k != "id"}
                     for r in read_jsonl(ROOT / "data/test/qwen25_7b_5fold_ensemble_dev.jsonl")}
            if preds != saved or len(saved) != len(dev):
                raise ValueError("Reconstructed predictions differ from Egemen's submission file")
        report[f"{name}_dev"] = scores(preds, dev)
        save_rows(args.output / f"{name}_dev.jsonl", [
            {"id": key, **(p or {"risk_severity": None, "risk_domains": None, "risk_drivers": None})}
            for key, p in preds.items()
        ])
    save_json(args.output / "metrics.json", report)
    for name, row in report.items():
        if "severity_mae" in row:
            print(f"{name:17} n={row['n']:3}  MAE={row['severity_mae']:.4f}  "
                  f"domain J={row['domains_jaccard']:.4f}  driver J={row['drivers_jaccard']:.4f}  "
                  f"malformed={row['malformed']}")
    print(f"Full results: {args.output / 'metrics.json'}")


if __name__ == "__main__":
    main()
