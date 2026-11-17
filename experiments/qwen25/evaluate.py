"""Complete-denominator metrics; missing and invalid predictions are never dropped."""
import argparse
import json
from pathlib import Path

from data_utils import ROOT, TARGET_KEYS, read_jsonl, validate_target
from scenarios import DOMAINS, DRIVERS


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def macro_f1(pairs, labels):
    scores = []
    for label in labels:
        tp = sum(label in g and label in p for g, p in pairs)
        fp = sum(label not in g and label in p for g, p in pairs)
        fn = sum(label in g and label not in p for g, p in pairs)
        scores.append(2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return sum(scores) / len(scores)


def score(gold, predictions):
    if not gold:
        raise ValueError('Gold dataset is empty')
    pred = {}
    for r in predictions:
        if r['id'] in pred:
            raise ValueError('Duplicate prediction ID')
        pred[r['id']] = r
    gold_ids = {r['example_id'] for r in gold}
    if len(gold_ids) != len(gold):
        raise ValueError('Duplicate gold ID')
    if set(pred) - gold_ids:
        raise ValueError('Predictions contain IDs not present in the gold file')
    valid, invalid, missing, exact, error = 0, 0, 0, 0, 0
    domain_pairs, driver_pairs = [], []
    domain_j, driver_j = [], []
    confusion = [[0]*5 for _ in range(5)]
    for g in gold:
        p = pred.get(g['example_id'])
        if p is None:
            missing += 1
            ok = False
        else:
            try:
                validate_target({k: p[k] for k in TARGET_KEYS})
                ok = p.get('valid_output', True) is not False
            except (ValueError, TypeError, KeyError):
                ok = False
            if not ok:
                invalid += 1
        if ok:
            valid += 1
            error += abs(g['risk_severity']-p['risk_severity'])
            exact += int(g['risk_severity'] == p['risk_severity'])
            confusion[g['risk_severity']-1][p['risk_severity']-1] += 1
            dp, rp = p['risk_domains'], p['risk_drivers']
            domain_j.append(jaccard(g['risk_domains'], dp))
            driver_j.append(jaccard(g['risk_drivers'], rp))
        else:
            dp, rp = [], []
            domain_j.append(0.)
            driver_j.append(0.)
        domain_pairs.append((g['risk_domains'], dp))
        driver_pairs.append((g['risk_drivers'], rp))
    n = len(gold)
    return dict(n_gold=n, valid_outputs=valid, invalid_outputs=invalid, missing_outputs=missing,
                valid_output_rate=valid/n, severity_accuracy_all=exact/n,
                severity_mae_valid_only=error/valid if valid else None,
                severity_mae_penalized_all=(error+4*(invalid+missing))/n,
                penalty_note='Invalid/missing outputs receive maximum ordinal error 4; diagnostic, not an official task metric.',
                domains_jaccard_all=sum(domain_j)/n, drivers_jaccard_all=sum(driver_j)/n,
                domains_macro_f1=macro_f1(domain_pairs, list(DOMAINS.values())),
                drivers_macro_f1=macro_f1(driver_pairs, list(DRIVERS.values())),
                severity_confusion_valid_only=confusion)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--gold', type=Path, default=ROOT/'data/raw/dev.jsonl')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = score(read_jsonl(args.gold), read_jsonl(args.predictions))
    text = json.dumps(result, indent=2)+'\n'
    print(text, end='')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == '__main__':
    main()
