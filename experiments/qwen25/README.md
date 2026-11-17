# Qwen 2.5 experiment with the existing AgentRisk labels

This experiment increases the training set from **568 to 1,336 rows** while
keeping the original label vocabulary and the 143-row development set.
It includes an actual generated dataset, its reproducible generator, a review
worksheet, and a small-model QLoRA training/evaluation workflow.

**Start with `Qwen/Qwen2.5-1.5B-Instruct`.** The supplied recipe trains a rank-8
LoRA adapter over 4-bit NF4 weights on one supported NVIDIA CUDA GPU. If that
does not fit your current GPU, try `Qwen/Qwen2.5-0.5B-Instruct` using `--model`.
No specific VRAM requirement is promised: context length, CUDA/PyTorch versions,
and training buffers matter. This recipe does not implement Apple MPS training.
GPU training and the real Qwen tokenizer preflight have **not** been run in the
workspace where this dataset was prepared; no weights or performance claims are
included.

## What the data contains

| File under `data/synthetic_v1/` | Rows | Purpose |
|---|---:|---|
| `synthetic_train.jsonl` | 768 | New synthetic examples, original raw schema |
| `train_extended.jsonl` | 1,336 | All 568 original rows plus the synthetic examples |
| `train_extended_sft.jsonl` | 1,336 | Shuffled chat-format fine-tuning data |
| `train_core_sft.jsonl` | 760 | Originals plus one rendering of each of the 192 new decision cases |
| `train_original_sft.jsonl` | 568 | Same prompt/schema, original-only baseline |
| `dev_inference.jsonl` | 143 | Original dev inputs, no assistant answers or label fields in the input |
| `provenance.jsonl` | 768 | Family IDs, variant IDs, review status and label rationales |
| `review_cases.csv` | 192 | One representative of each decision case for human review |
| `manifest.json` | — | Counts, distributions, source and output checksums |
| `audit.json` | — | Schema, duplication and lexical similarity audit |

The 768 synthetic rows are **192 action decisions across 64 scenario families,
with four entity/presentation variants per decision**. They are not 768
independently researched incidents. The 760-row core set is useful for checking
whether the extra entity variants add anything beyond the new decisions.

Each family shares observed evidence but has three possible final actions:
one safe action and two harmful or error-prone alternatives. Some safe actions
perform authorized work; others defer a consequential action. Labels concern
the proposed final action, rather than merely the presence of sensitive words
or a hazardous instruction earlier in the trajectory. The families cover
permissions, data disclosure, tool-output injection, operations, payments,
physical safety, compliance, information reliability, social harm, fairness
and environmental risk. Severity-5 cases are limited to clearly safety-critical
scenarios, rather than treating all prompt injections as catastrophic.

**Only these three target fields are emitted:**

```json
{"risk_domains": ["Privacy / Confidentiality"], "risk_drivers": ["Authorization / Consent Failure"], "risk_severity": 3}
```

The exact vocabulary is recorded in `manifest.json`: ten material-harm domains
plus `No Material Harm`, six risk drivers plus `No Material Risk`, and integer
severity 1–5. At most two labels are allowed per multi-label field; a no-risk
sentinel must stand alone. No label categories or factor annotations were added.

The existing `data/processed/llm_train_sft.jsonl` predicts `f1` and `f2` rather
than severity directly. This is a **separate direct-severity baseline** using the
labels already present in `data/raw/train.jsonl`. It neither overwrites that
factor experiment nor invents factor values for synthetic examples. To evaluate
augmentation fairly, compare the new original-only and extended runs below,
which use the same output schema and prompt.

## Run from the repository root

Use a separate Python 3.10+ environment. Install a PyTorch build suitable for
your machine's CUDA configuration first if your cluster requires it, then:

```bash
python -m venv .venv-qwen25
source .venv-qwen25/bin/activate
python -m pip install -r experiments/qwen25/requirements.txt
```

The dependency ranges are bounded but are not a GPU-tested lockfile. Record
your resolved environment with `python -m pip freeze > output/qwen25-env.txt`
after creating `output/`. No OpenAI API key or paid generation call is needed.

The generated files are already included. To reproduce and check them:

```bash
python experiments/qwen25/build_dataset.py
python experiments/qwen25/audit.py
python -m unittest discover -s experiments/qwen25 -v
```

First run tokenizer preflight. It needs access to the model tokenizer or an
existing local Hugging Face cache but does not load the model weights:

```bash
python experiments/qwen25/train.py --preflight-only
```

Then train the extended dataset:

```bash
python experiments/qwen25/train.py \
  --output-dir output/qwen25-1.5b-extended
```

Defaults: 1.5B Instruct, 4-bit NF4 with double quantization, batch size 1,
gradient accumulation 16, rank 8 on attention projections, learning rate
`1e-4`, two epochs, gradient checkpointing, SDPA and a 4,096-token cap.
BF16 is selected where CUDA supports it, otherwise FP16.

Loss is applied only to the assistant completion; the prompt and padding are
masked with `-100`. IDs, provenance and rationales never enter the model prompt.
Any overlength example causes an explicit error **before loading weights**.
The script never silently truncates a trajectory or discards its proposed
action/answer. Increasing `--max-length` uses more memory. Reducing it may fail
preflight; do not truncate away evidence just to fit a GPU.

For a one-step hardware smoke run, use a separate output directory:

```bash
python experiments/qwen25/train.py \
  --max-steps 1 --output-dir output/qwen25-smoke
```

If necessary, select the smaller model for both preflight and training:

```bash
python experiments/qwen25/train.py \
  --model Qwen/Qwen2.5-0.5B-Instruct --preflight-only
python experiments/qwen25/train.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir output/qwen25-0.5b-extended
```

## Compare original-only versus synthetic extension

For an initial comparison with the **same optimizer-update budget**, use these
two commands with otherwise identical settings. One hundred updates is a pilot
budget, not a demonstrated optimum. Epoch-matched runs expose the extended
model to more data and more optimizer updates, so report which comparison you
used.

```bash
python experiments/qwen25/train.py \
  --train-file data/synthetic_v1/train_original_sft.jsonl \
  --max-steps 100 --output-dir output/qwen25-original-100
python experiments/qwen25/train.py \
  --train-file data/synthetic_v1/train_extended_sft.jsonl \
  --max-steps 100 --output-dir output/qwen25-extended-100
```

Predict on the original dev set using the unfine-tuned base model and both
adapters. Predictions preserve original example IDs and malformed completions:

```bash
python experiments/qwen25/predict.py --output output/qwen25-base-dev.jsonl
python experiments/qwen25/predict.py \
  --adapter output/qwen25-original-100 --output output/qwen25-original-dev.jsonl
python experiments/qwen25/predict.py \
  --adapter output/qwen25-extended-100 --output output/qwen25-extended-dev.jsonl
python experiments/qwen25/evaluate.py \
  --predictions output/qwen25-base-dev.jsonl --output output/qwen25-base-metrics.json
python experiments/qwen25/evaluate.py \
  --predictions output/qwen25-original-dev.jsonl --output output/qwen25-original-metrics.json
python experiments/qwen25/evaluate.py \
  --predictions output/qwen25-extended-dev.jsonl --output output/qwen25-extended-metrics.json
```

Inference recovers the base model from an adapter's config; for an unfine-tuned
0.5B baseline pass `--model Qwen/Qwen2.5-0.5B-Instruct` explicitly. Greedy
decoding uses the same prompt for all models. Output paths must be new so a run
does not overwrite previous predictions.

Metrics include JSON validity, severity accuracy/MAE, domain and driver Jaccard,
macro-F1, and a severity confusion matrix. Missing/invalid predictions remain in
the all-example denominators. The penalized severity MAE assigns error 4 to an
invalid/missing output and is explicitly a diagnostic, **not an official task
metric**. Valid-only MAE is also reported with its coverage. The development
set is a development set, not an untouched final test set; repeated tuning on
it should not be reported as final generalization performance.

## Review and limitations

- Every new annotation is marked `weak_synthetic` and `human_reviewed=false`.
  Review the 192 decisions in `review_cases.csv`, especially severity boundaries
  and two-label choices. Edit the corresponding family in `scenarios.py` and
  rebuild so all four variants receive a consistent correction. Merely changing
  a review-status cell does not change training data or provenance.
- The added rows are shorter and more explicit than many original trajectories.
  Models may learn synthetic wording or presentation shortcuts. Compare the
  core and full extension, and inspect errors on original dev examples.
- Added examples intentionally improve coverage of rare existing domains; they
  change class prevalence. Do not infer real-world risk frequency from them.
- Preserve entire `family_id` groups if you later create a synthetic holdout.
  A random split of near-identical variants would inflate validation results.
- Neither generator templates nor variant slots are derived from dev rows.
  Dev is used after generation for validation and inference formatting. Exact
  overlap and word-five-gram similarity checks pass, but lexical checks do not
  establish absence of semantic overlap or benchmark contamination.
- Original raw files and existing factor labels are preserved. New data cannot
  repair any mistakes in the original labels. No additional ground truth is
  claimed, and larger row count does not guarantee a better model.

## Reference documentation

- [Original AgentRisk dataset](https://huggingface.co/datasets/AgentRisk/agentrisk-data-2027)
- [Qwen 2.5 1.5B Instruct model card and chat template](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [Hugging Face PEFT quantization and QLoRA preparation](https://huggingface.co/docs/peft/en/developer_guides/quantization)
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)

Prepared against AgentRisk commit `c1fd551a13906322f796590f8126168fd7d779de`
on 17 November 2026. Offline validation: 11 unit tests passed, deterministic
regeneration matched, original file checksums retained, and no exact training/dev
overlap was found. These checks validate the data and pipeline logic; they do not
substitute for a CUDA training run or independent annotation review.
