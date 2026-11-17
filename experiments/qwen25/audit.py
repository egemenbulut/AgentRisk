"""Recheck output integrity and report lexical similarity without changing any rows."""
import json
import re
from collections import Counter

from data_utils import DATA, ROOT, fingerprint, read_jsonl, sha256, validate_raw


def shingles(row):
    words = re.findall(r'\w+', row['formatted_interaction'].casefold())
    return set(zip(*(words[i:] for i in range(5))))


def nearest(left, right):
    rhs = [shingles(r) for r in right]
    result = []
    for row in left:
        a = shingles(row)
        values = [len(a & b)/len(a | b) if a | b else 1. for b in rhs]
        idx = max(range(len(values)), key=values.__getitem__)
        result.append(dict(id=row['example_id'], nearest_id=right[idx]['example_id'],
                           word_5gram_jaccard=round(values[idx], 6)))
    return sorted(result, key=lambda x: x['word_5gram_jaccard'], reverse=True)


def main():
    manifest = json.loads((DATA/'manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Source changed: {name}; rebuild before using the dataset')
    for name, digest in manifest['output_sha256'].items():
        if sha256(DATA/name) != digest:
            raise ValueError(f'Output changed: {name}; rebuild or explicitly version your edit')
    original = read_jsonl(ROOT/'data/raw/train.jsonl')
    dev = read_jsonl(ROOT/'data/raw/dev.jsonl')
    synthetic = read_jsonl(DATA/'synthetic_train.jsonl')
    extended = read_jsonl(DATA/'train_extended.jsonl')
    provenance = read_jsonl(DATA/'provenance.jsonl')
    for rows in (original, dev, synthetic, extended):
        validate_raw(rows)
    assert extended[:len(original)] == original
    assert extended[len(original):] == synthetic
    assert len({fingerprint(r) for r in extended}) == len(extended)
    assert not ({fingerprint(r) for r in extended} & {fingerprint(r) for r in dev})
    assert {p['example_id'] for p in provenance} == {r['example_id'] for r in synthetic}
    syn_near = nearest(synthetic, dev)
    original_near = nearest(original, dev)
    result = dict(schema_valid=True, originals_unchanged=True, exact_train_dev_overlap=0,
                  synthetic_families=len({p['family_id'] for p in provenance}),
                  synthetic_decision_cases=len({p['decision_case'] for p in provenance}),
                  synthetic_severity=dict(Counter(str(r['risk_severity']) for r in synthetic)),
                  similarity_method='Word five-gram Jaccard on full trajectories; lexical diagnostic only, not a semantic leakage guarantee.',
                  review_threshold=.6,
                  synthetic_dev_pairs_at_or_above_threshold=sum(r['word_5gram_jaccard'] >= .6 for r in syn_near),
                  original_train_rows_with_dev_similarity_at_or_above_threshold=sum(r['word_5gram_jaccard'] >= .6 for r in original_near),
                  nearest_synthetic_to_dev=syn_near[:10], nearest_original_to_dev=original_near[:10])
    (DATA/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('nearest_')}, indent=2))


if __name__ == '__main__':
    main()
