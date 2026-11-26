# Five-fold AgentRisk experiment

Train five independent Llama adapters on the same examples as Egemen Bulut’s Qwen2.5-7B experiment. The final command compares Qwen alone, Llama alone, and all ten adapters on the same 143 development examples. The existing Qwen predictions contain everything needed for the combined vote.

## Start on an M4 Mac with 24 GB

Use native Apple silicon Python 3.11 or 3.12 on macOS 14 or later. Open Terminal in the repository. The default model is four-bit Llama 3.2 3B Instruct. Keep the Mac connected to power for a full run.

```bash
git fetch origin
git switch experiments/llama-five-fold
python3 -m venv .venv-llama
source .venv-llama/bin/activate
python -m pip install -r experiments/fivefold/requirements.txt
python experiments/fivefold/llama.py smoke
```

The smoke run downloads the pinned model, checks every example for truncation, takes one optimizer step on the longest eligible fold-0 training example, validates, saves and reloads the adapter, and predicts one held-out training row. Read the peak memory in Terminal. Its prediction has no scientific significance. To check the tokenizer and data before downloading model weights, run `python experiments/fivefold/llama.py check`.

Once the smoke run finishes, run the full experiment in the same Terminal:

```bash
caffeinate -i python experiments/fivefold/llama.py run
```

To time a single production fold first, append `--fold 0`. The full command skips completed folds, writes results under `output/llama-3b-fivefold/comparison/`, and prints the metrics. A new invocation can reuse successfully trained adapters and complete prediction files. An interruption during training restarts that fold’s training. Smoke results have a separate directory. Each fold’s `training.json` records validation losses and the best update. Its `done.json` records peak memory and wall time for the completing invocation.

## What is held fixed

| Setting | Qwen reference and Llama run |
| --- | --- |
| Inputs | The 568 original training rows only. Dev inputs go to inference. |
| Splits | Saved assignments from seed 144. Folds 0–4 hold out 117, 126, 96, 109 and 120 rows. The runner checks example IDs and scenario groups against the official manifest. |
| Target | Same system prompt, trajectory, proposed action and three-field JSON answer. Each model uses its own chat template. Only assistant tokens, including the end of turn, contribute to loss. |
| Adaptation | Fresh four-bit base each fold, rank 64, alpha 128 (MLX multiplier 2), all seven attention/MLP projections in every block, zero dropout. |
| Training | Three passes, effective batch of eight (final partial batch per epoch), peak learning rate 0.0002, ten warmup updates, linear decay, weight decay 0.01, seed 3407 + fold. |
| Selection | Lowest held-out-fold validation loss, checked every 20 optimizer updates and at the end. Dev labels do not select the checkpoints. |
| Inference | Greedy generation, up to 128 new tokens. Save each JSON response, parsed output, and the probabilities assigned to severity 1–5. |

The Mac port uses MLX quantization and AdamW with one example per microbatch. Egemen used Unsloth, bitsandbytes, 8-bit AdamW and two examples per microbatch on the recorded Colab run. Model family, size and numerical backend all change. The Mac experiment tests whether the new predictions improve the ensemble. It cannot isolate which training change caused an improvement.

## Results on the original labels

| Run | Evaluated examples | Severity MAE ↓ | Domain Jaccard ↑ | Driver Jaccard ↑ |
| --- | ---: | ---: | ---: | ---: |
| Prior Qwen2.5-1.5B, original data | 143 dev | 0.385 | 0.828 | 0.704 |
| Prior Qwen2.5-1.5B, original + synthetic | 143 dev | 0.524 | 0.776 | 0.707 |
| Egemen’s Qwen2.5-7B, single held-out prediction per training row | 568 OOF | 0.2500 | 0.8542 | 0.7265 |
| Egemen’s Qwen2.5-7B, five votes per dev row | 143 dev | 0.1678 | 0.8578 | 0.7751 |

The 1.5B figures come from the [24 November comparison](../qwen25/synthetic_training_comparison.pdf). The Qwen 7B figures can be recomputed without ML libraries or model weights. The replay checks that all 143 ensemble predictions match Egemen’s saved export, including both label lists.

```bash
python3 experiments/fivefold/compare.py
```

For OOF (out-of-fold) scoring, each example receives a prediction from the one adapter whose training split excluded it. That fold also selected the checkpoint. For dev scoring, the notebook compared four severity aggregation methods and three label thresholds against the dev labels. The 0.1678 figure is a development score. Independent test performance and any improvement from adding Llama remain open.

## Voting and output files

For each example, the reference averages the five probability vectors over severity levels 1–5. It chooses the first severity whose cumulative probability reaches 0.5. Domains need three votes and drivers need two. Three votes for a no-risk label return that label alone. The rule keeps at most two labels, breaks ties alphabetically and falls back to the most frequent label when none reaches threshold.

The Llama-only run freezes those settings. For the ten-adapter run the thresholds are six domain votes, four driver votes and six votes for a no-risk majority. All ten probability vectors have equal weight. The Llama runner reads probabilities at the severity field itself. A parsed severity with no digit probabilities contributes a one-hot vote. Unparseable answers abstain. If every adapter abstains, the row receives severity error 4 and zero Jaccard in the diagnostic score. The report counts malformed answers and separately flags parseable predictions outside the label schema. Check those counts before using any JSONL as a submission.

After the full run, open `comparison/metrics.json` and compare `qwen_dev`, `llama_dev` and `qwen_llama_dev`. Each also has an ID-aligned JSONL export. Severities, domains and drivers may move in different directions. The per-fold scores and single-adapter OOF scores are included for diagnosis.

## Egemen’s source files

| Artifact | Location |
| --- | --- |
| Colab training and evaluation code | [Notebook with original source cells](qwen_reference/AgentRisk_training_25112026.ipynb). Execution output and Colab session metadata were cleared for Git. |
| Folds and groups | [Egemen’s exact ID-to-fold map](qwen_reference/folds_train_only.json) and [official split manifest](qwen_reference/split_manifest.jsonl) from [dataset revision 8aad17e](https://huggingface.co/datasets/AgentRisk/agentrisk-data-2027/blob/8aad17eb835bde22a8d093e0c8f12734d2f1881a/metadata/split_manifest.jsonl). |
| Raw predictions | [Five validation files, five dev files and the ensemble](qwen_reference/predictions/), including the individual severity probability vectors. |
| Formatted dev predictions | [Individual adapters](qwen_reference/formatted_dev_jsonl/) and [five-adapter ensemble](../../data/test/qwen25_7b_5fold_ensemble_dev.jsonl). |
| Earlier single-model dev result | [Qwen without factors](../../data/test/finetuned_without_factors.jsonl). |
| Recomputed metrics | [Per-fold, OOF and ensemble metrics](qwen_reference/metrics.json). |
| Original files and model weights | [25 November Drive experiment](https://drive.google.com/drive/folders/1jBnkeyMJBxh5HeFnY60dsV-oG1zyucgB) and [the five saved Qwen adapter directories](https://drive.google.com/drive/folders/1TfX0df3NLasA-q26fK200_-d2g3O0LDM). |

## Using the two-day TUM allocation

The same runner has an 8B option with a pinned Llama 3.1 Instruct checkpoint. On the allocated GPU node, use the MLX CUDA wheel compatible with the installed driver and toolkit. Install an isolated environment before timing. If the compute nodes have no internet, populate the shared Hugging Face cache on an internet-enabled node before the allocation starts.

```bash
# CUDA 12 example. Use the cuda13 extra on a compatible CUDA 13 node.
python3 -m venv .venv-llama-cuda
source .venv-llama-cuda/bin/activate
python -m pip install -r experiments/fivefold/requirements.txt 'mlx[cuda12]==0.32.5'
nvidia-smi
python -c 'import mlx.core as mx; print(mx.default_device(), mx.cuda.is_available())'
python experiments/fivefold/llama.py check --size 8b
python experiments/fivefold/llama.py smoke --size 8b
time python experiments/fivefold/llama.py run --size 8b --fold 0
```

Use fold 0’s complete training **and inference** time to decide whether four more folds fit in the remaining allocation. Allow for differences in fold size and scheduling. If time permits, the command below skips fold 0 and finishes the comparison. If time runs short, retain the completed adapters, predictions and fold IDs for a later run. Multiple allocated GPUs can run distinct `--fold` values in separate jobs sharing the same output directory.

```bash
python experiments/fivefold/llama.py run --size 8b
```

The pinned 3B and 8B tokenizers passed preflight for every example. Both encoded a maximum of 3,754 tokens for training and 3,166 for a dev prompt. A tiny four-bit Llama exercised training, validation, adapter reload and inference on Linux CPU. Metal performance, CUDA compatibility on the assigned node, full-size training times and Llama scores still require the respective hardware. [MLX installation and GPU requirements](https://ml-explore.github.io/mlx/build/html/install.html) · [3B model](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit) · [8B model](https://huggingface.co/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit).
