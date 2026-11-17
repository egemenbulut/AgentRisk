# Synthetic AgentRisk extension v1

**1,336 training rows = 568 original + 768 synthetic.**

Use `train_extended_sft.jsonl` for the supplied Qwen 2.5 recipe, or
`train_extended.jsonl` for the original raw schema. The synthetic labels use
only the existing `risk_domains`, `risk_drivers`, and `risk_severity` fields.
No `f1`–`f7` values were added. The original 143 dev rows remain separate.

The synthetic component contains **64 families, 192 distinct action decisions,
and four entity/presentation variants per decision**. All are weak synthetic
annotations, not independently human-reviewed gold labels. `train_core_sft.jsonl`
contains 760 rows, using one variant per new decision. `review_cases.csv` gives
the 192 representative decisions and annotation rationales for review.

| Severity | Original | Synthetic | Extended |
|---|---:|---:|---:|
| 1 | 265 | 256 | 521 |
| 2 | 33 | 56 | 89 |
| 3 | 127 | 220 | 347 |
| 4 | 136 | 224 | 360 |
| 5 | 7 | 12 | 19 |
| **Total** | **568** | **768** | **1,336** |

Domain counts are multi-label counts and therefore do not sum to the row count:

| Existing domain | Original | Synthetic | Extended |
|---|---:|---:|---:|
| No Material Harm | 262 | 256 | 518 |
| Privacy / Confidentiality | 180 | 88 | 268 |
| Security / System Integrity | 65 | 60 | 125 |
| Financial / Assets | 53 | 92 | 145 |
| Physical / Health | 52 | 60 | 112 |
| Operational / Service | 44 | 84 | 128 |
| Legal / Compliance | 15 | 60 | 75 |
| Reputational / Social / Psychological | 6 | 64 | 70 |
| Information Quality | 3 | 180 | 183 |
| Environmental | 2 | 32 | 34 |
| Discrimination / Fairness | 1 | 48 | 49 |

Provenance and review notes are stored separately and must not be fed into the
classifier. Synthetic examples have shorter, more explicit evidence than the
originals, and domain frequencies intentionally differ. Treat this as an
augmentation experiment; compare against the included original-only baseline.

See [the experiment guide](../../experiments/qwen25/README.md) for installation,
preflight, training, evaluation, reproducibility, and limitations. Review the
family definitions in `experiments/qwen25/scenarios.py`; rebuild after correcting
them. Do not randomly divide variants of the same family between train and eval.
