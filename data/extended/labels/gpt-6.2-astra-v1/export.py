"""Export unique AgentRisk cases as blind model inputs or raw-schema training rows."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = {
    "agent_auditor": ("agent_auditor_dataset.jsonl",
                     "f6073be130d2df390152a296df578f8e788b40bfc3444291980b5d832ed57896"),
    "r_judge": ("r_judge_dataset.jsonl",
                "277c690723a76954d3372df059c00da58be074f3d162699475884383a8a111c2"),
}
TARGETS = ("risk_domains", "risk_drivers", "risk_severity")


def case_id(trace, action):
    data = json.dumps([trace, action], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_inputs(source_dir):
    cases = {}
    for dataset, (name, checksum) in SOURCES.items():
        data = (source_dir / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(f"{name}: source checksum differs from the labeled snapshot.")
        for number, line in enumerate(data.splitlines(), 1):
            row = json.loads(line)
            trace, action = row["formatted_interaction"], row["proposed_action"]
            key = case_id(trace, action)
            if key not in cases:
                cases[key] = ({"example_id": key, "formatted_interaction": trace,
                               "proposed_action": action}, {})
            refs = cases[key][1]
            refs.setdefault(dataset, []).append(number)
    return cases


def export(mode, cases, annotations, include_review):
    for key in sorted(cases):
        case, sources = cases[key]
        if mode == "blind":
            if not case["proposed_action"].strip():
                continue
            yield case
            continue
        anno = annotations[key]
        if sources != anno["source_rows"]:
            raise ValueError(f"Source row mapping changed for {key}.")
        if anno["status"] != "labeled" or (anno["needs_review"] and not include_review):
            continue
        yield {**case, **{k: anno[k] for k in TARGETS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["blind", "training"])
    parser.add_argument("output", type=Path, help="New output JSONL file")
    parser.add_argument("--source-dir", type=Path, default=HERE.parents[1],
                        help="Directory containing Egemen's two source JSONLs")
    parser.add_argument("--include-review", action="store_true",
                        help="Include available but flagged Astra judgments in training output")
    args = parser.parse_args()
    if args.mode == "blind" and args.include_review:
        parser.error("--include-review applies only to training")
    cases = load_inputs(args.source_dir)
    annotations = {}
    if args.mode == "training":
        for line in (HERE / "annotations.jsonl").read_bytes().splitlines():
            row = json.loads(line)
            key = row["case_id"]
            if key in annotations:
                raise ValueError(f"Duplicate annotation for {key}")
            annotations[key] = row
        if cases.keys() != annotations.keys():
            raise ValueError("Source/annotation case IDs differ.")
    rows = list(export(args.mode, cases, annotations, args.include_review))
    with args.output.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} unique {args.mode} rows written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
