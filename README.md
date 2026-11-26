# AgentRisk

This repository contains the TUM practicum experiments for SemEval 2027 Task 10. Given an agent’s interaction history and its final proposed action, the task is to predict the risk domains, the risk drivers, and a severity score from 1 to 5.

## Where to work

| Goal | Branch | Start here |
| --- | --- | --- |
| Review the 112 cases left unlabeled in the Astra pass | `labels/gpt-6.2-astra-v1` | [Manual review workbook][workbook] · [Scoring rubric][rubric] |
| Use Astra’s labels or prepare an independent Claude pass | `labels/gpt-6.2-astra-v1` | [Annotation release and export instructions][astra] |
| Inspect Egemen’s AgentAuditor and R-Judge imports | `feature/extend-dataset` | [Source JSONL files][imports] |
| Reproduce Egemen’s Qwen ensemble or run five Llama folds on Apple silicon | `experiments/llama-five-fold` | [Mac quick start, verified results and TUM run](experiments/fivefold/README.md) |
| Fine-tune Qwen or compare training sets | `main` | [Experiment guide](experiments/qwen25/README.md) · [Synthetic dataset guide](data/synthetic_v1/README.md) |
| Reproduce the original factor analysis or model comparisons | `main` | [Pipeline entry point](src/main.py) · [Saved predictions](data/test/) |

## Data and experiments

| Location | Contents |
| --- | --- |
| [`data/raw/`](data/raw/) | The original 568 labeled training examples and 143 labeled development examples. |
| [`data/processed/`](data/processed/) | Factor annotations, merged training records, supervised fine-tuning chats, and development prompts with and without factors. |
| [`data/test/`](data/test/) | Saved base-model and fine-tuned predictions on the development set. The reference labels are in [`data/raw/dev.jsonl`](data/raw/dev.jsonl). |
| [`src/`](src/) | Data preparation, factor correlations, factor-to-severity benchmarks, and prediction comparisons. Running [`src/main.py`](src/main.py) with its current settings rebuilds the development prompts and writes comparison charts, a summary, and per-example scores under `output/comparison/`. |
| [`data/synthetic_v1/`](data/synthetic_v1/) | 768 generated examples covering 192 distinct action decisions. The extended training set combines all 768 with the 568 originals (1,336 rows). The core set uses one variant per decision (760 rows). |
| [`experiments/qwen25/`](experiments/qwen25/) | Scripts to build and audit that training data, fine-tune Qwen2.5, generate predictions, and score severity, domains, and drivers. |

## Reviewing the imported cases

The Astra release labels actions in the AgentAuditor and R-Judge imports. Their 6,556 source rows contain 2,583 distinct trajectory–action pairs. Astra assigned 2,125 provisional labels. The default training export includes 1,801 full-input, unflagged cases. The [release manifest][manifest] gives the counts for each outcome.

For manual review, download the [workbook][workbook], open a case’s source link, and score the **final proposed action** in context. The stable case ID joins the worksheet row to the [annotation record][annotations]. For a blind second opinion, use the [export instructions][astra] to produce the original inputs without Astra’s judgments.

[workbook]: https://github.com/egemenbulut/AgentRisk/blob/labels/gpt-6.2-astra-v1/data/extended/labels/gpt-6.2-astra-v1/AgentRisk-112-manual-review.xlsx
[rubric]: https://github.com/egemenbulut/AgentRisk/blob/labels/gpt-6.2-astra-v1/data/extended/labels/gpt-6.2-astra-v1/rubric.txt
[astra]: https://github.com/egemenbulut/AgentRisk/blob/labels/gpt-6.2-astra-v1/data/extended/labels/gpt-6.2-astra-v1/README.md
[imports]: https://github.com/egemenbulut/AgentRisk/tree/feature/extend-dataset/data/extended
[manifest]: https://github.com/egemenbulut/AgentRisk/blob/labels/gpt-6.2-astra-v1/data/extended/labels/gpt-6.2-astra-v1/manifest.json
[annotations]: https://github.com/egemenbulut/AgentRisk/blob/labels/gpt-6.2-astra-v1/data/extended/labels/gpt-6.2-astra-v1/annotations.jsonl
