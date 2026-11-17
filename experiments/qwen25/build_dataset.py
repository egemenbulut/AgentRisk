"""Build a reproducible synthetic extension without reading dev examples as seeds."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from data_utils import DATA, ROOT, fingerprint, read_jsonl, sha256, to_chat, validate_raw, write_jsonl
from scenarios import CASES, DOMAINS, DRIVERS

VARIANTS = [
    dict(workspace='Alder', ticket='REQ-1842', host='alder.example'),
    dict(workspace='Cobalt', ticket='REQ-2957', host='cobalt.example'),
    dict(workspace='Juniper', ticket='REQ-3619', host='juniper.example'),
    dict(workspace='Willow', ticket='REQ-4726', host='willow.example'),
]


def generate():
    rows, provenance = [], []
    for c in CASES:
        for v, slots in enumerate(VARIANTS):
            request = c['request'].format(**slots)
            evidence = c['evidence'].format(**slots)
            context = f'Workspace: {slots["workspace"]}; tracking item: {slots["ticket"]}.'
            # Two presentations; evidence and role attribution remain the same.
            if v % 2 == 0:
                prefix = (f'Step 1. [USER]: {request} {context}\n'
                          'Step 2. [ACTION]: Inspect the task records and available controls without changing state.\n'
                          f'Step 3. [OBSERVATION]: {evidence}\n')
                last_step = 4
            else:
                prefix = (f'Step 1. [USER]: {context} {request}\n'
                          'Step 2. [THOUGHT]: I will check the records relevant to this request.\n'
                          'Step 3. [ACTION]: Read the task details and control preview.\n'
                          f'Step 4. [OBSERVATION]: {evidence}\n'
                          'Step 5. [THOUGHT]: The available details are shown above. My next action follows.\n')
                last_step = 6
            for oi, out in enumerate(c['outcomes']):
                action = out['action'].format(**slots)
                example_id = f'SYN-V1-{c["family"]}-{v+1}-{oi+1}'
                rows.append(dict(example_id=example_id,
                                 formatted_interaction=prefix + f'Step {last_step}. [ACTION]: {action}',
                                 proposed_action=action, **{k: out[k] for k in
                                 ('risk_severity', 'risk_domains', 'risk_drivers')}))
                provenance.append(dict(example_id=example_id, source='assistant_authored_synthetic',
                                       family_id=c['family'], decision_case=f'{c["family"]}:{oi+1}',
                                       entity_variant=v+1, human_reviewed=False,
                                       label_status='weak_synthetic', rationale=out['rationale'],
                                       intended_split='train'))
    return rows, provenance


def distribution(rows):
    return {k: dict(sorted(Counter(v for r in rows for v in
             (r[k] if isinstance(r[k], list) else [str(r[k])])).items()))
            for k in ('risk_severity', 'risk_domains', 'risk_drivers')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DATA)
    args = parser.parse_args()
    out = args.output_dir
    original = read_jsonl(ROOT / 'data/raw/train.jsonl')
    synthetic, provenance = generate()  # Does not consume dev content.
    validate_raw(original)
    validate_raw(synthetic)
    # Dev is read only here, after generation, for overlap checks and inference export.
    dev = read_jsonl(ROOT / 'data/raw/dev.jsonl')
    validate_raw(dev)
    extended = original + synthetic
    if len({fingerprint(r) for r in extended}) != len(extended):
        raise ValueError('Duplicate input content in training data')
    if {fingerprint(r) for r in extended} & {fingerprint(r) for r in dev}:
        raise ValueError('Exact input overlap with dev')
    if {r['example_id'] for r in extended} & {r['example_id'] for r in dev}:
        raise ValueError('ID overlap with dev')
    core_ids = {p['example_id'] for p in provenance if p['entity_variant'] == 1}
    core = original + [r for r in synthetic if r['example_id'] in core_ids]
    write_jsonl(out / 'synthetic_train.jsonl', synthetic)
    write_jsonl(out / 'provenance.jsonl', provenance)
    write_jsonl(out / 'train_extended.jsonl', extended)
    for name, rows in [('train_original_sft', original), ('train_core_sft', core),
                       ('train_extended_sft', extended)]:
        shuffled = list(rows)
        random.Random(20261117).shuffle(shuffled)
        write_jsonl(out / f'{name}.jsonl', [to_chat(r) for r in shuffled])
    write_jsonl(out / 'dev_inference.jsonl', [to_chat(r, with_target=False) for r in dev])
    # One decision case per family variant in the compact human-review worksheet.
    import csv
    with (out / 'review_cases.csv').open('w', encoding='utf-8', newline='') as f:
        fields = ['example_id', 'family_id', 'formatted_interaction', 'proposed_action',
                  'risk_domains', 'risk_drivers', 'risk_severity', 'rationale',
                  'review_status', 'reviewer_notes']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        by_id = {p['example_id']: p for p in provenance}
        for r in synthetic:
            if r['example_id'] not in core_ids:
                continue
            p = by_id[r['example_id']]
            writer.writerow({**r, 'family_id': p['family_id'], 'rationale': p['rationale'],
                             'risk_domains': json.dumps(r['risk_domains']),
                             'risk_drivers': json.dumps(r['risk_drivers']),
                             'review_status': 'unreviewed', 'reviewer_notes': ''})
    manifest = dict(version='synthetic-v1', seed=20261117, base_commit='c1fd551a13906322f796590f8126168fd7d779de',
                    counts=dict(original_train=len(original), synthetic=len(synthetic), extended=len(extended),
                                core=len(core), dev=len(dev), families=len(CASES), decision_cases=len(core_ids)),
                    allowed_domains=list(DOMAINS.values()), allowed_drivers=list(DRIVERS.values()),
                    label_fields=['risk_domains', 'risk_drivers', 'risk_severity'],
                    original_distribution=distribution(original), synthetic_distribution=distribution(synthetic),
                    extended_distribution=distribution(extended),
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                   [ROOT/'data/raw/train.jsonl', ROOT/'data/raw/dev.jsonl',
                                    Path(__file__).with_name('scenarios.py')]},
                    exact_train_dev_overlap=0,
                    limitations=['Synthetic labels are not independent human gold annotations.',
                                 '768 rows represent 192 decision cases in 64 scenario families, each rendered four times.',
                                 'Entity and presentation variants add repetition, not independent evidence.',
                                 'No new factor labels f1-f7 are generated.',
                                 'No performance improvement is claimed; compare original-only and extended training.',
                                 'Exact overlap checks do not prove absence of semantic similarity.'])
    manifest['output_sha256'] = {p.name: sha256(p) for p in sorted(out.iterdir())
                                 if p.suffix in ('.jsonl', '.csv')}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(manifest['counts'], indent=2))


if __name__ == '__main__':
    main()
