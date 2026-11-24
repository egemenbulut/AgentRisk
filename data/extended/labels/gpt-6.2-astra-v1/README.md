# AgentRisk synthetic labels from GPT-6.2 Astra

This is the first model annotation pass over Egemen’s November 23 AgentAuditor and R-Judge imports. GPT-6.2 Astra assigned weak labels for the risk of executing the final proposed action. The three targets are risk domains, risk drivers, and ordinal severity (1–5). Human and Claude adjudication are pending.

## Coverage

| Input or outcome | Count |
| --- | ---: |
| Source rows across the two files | 6,556 |
| Distinct trajectory–action pairs | 2,583 |
| Distinct pairs appearing in both sources | 561 |
| Distinct pairs with a recorded Astra judgment | 2,125 |
| Labeled pairs flagged for review | 324 |
| Full-input, unflagged judgments in the default training export | 1,801 |
| Model abstentions because the available evidence was insufficient | 201 |
| Remaining pairs without a usable annotation in this pass | 112 |
| Distinct pairs whose proposed action is empty | 145 |

AgentAuditor contributes 5,985 source rows and 2,576 distinct pairs. R-Judge contributes 571 source rows and 568 distinct pairs. The cross-source overlap makes a random split by source file unsuitable for evaluation. Training and validation splits should keep each case ID together. Similar but nonidentical cases may also need grouping.

## Manual review

Download the [review workbook for the 112 unlabeled cases](AgentRisk-112-manual-review.xlsx). Each row links to the original trajectory and proposed action and has editable fields for severity, domains, drivers, reviewer, and notes. Use the stable case ID to match completed reviews to `annotations.jsonl`. The workbook’s **Rubric** tab includes the scoring criteria and instructions for reading large source files locally.

## Annotation method

The labeling agents were explicitly configured as `gpt-6.2-astra` with low reasoning effort. Each assigned case was assessed using Egemen’s prompt saved in [rubric.txt](rubric.txt). Identical inputs across and within files share one judgment. Each judgment represents a single model assessment. Parallel workers used the same rubric and marked ambiguity for review. Source instructions were treated as data. No independent annotator scores were supplied to the workers.

Some source passages triggered a content filter. For a subset, Astra could assess a restricted view with technical or sensitive lines withheld, a named final tool and selected user statements, or excerpts of the final response. All 163 judgments from restricted views carry `input_view: "limited"` and `needs_review: true`. Other ambiguous full-input judgments were also flagged. Model abstentions, unavailable annotations and missing actions have no target values. Human adjudication is particularly important when the view was limited or severity lies near a category boundary.

## Files and identifiers

- [annotations.jsonl](annotations.jsonl) contains one record per distinct pair. `case_id` is SHA-256 of the UTF-8 compact JSON array `[formatted_interaction, proposed_action]`, with Unicode preserved. The hash uses the exact source strings. `source_rows` lists every matching 1-based JSONL line in each contributing file, including duplicates.
- [manifest.json](manifest.json) records the source commit, source and annotation hashes, model configuration, coverage, review counts and severity distribution. Annotation date is 2026-11-24.
- [export.py](export.py) joins labels to the two source files in this branch and checks their hashes. The original text stays in Egemen’s source files. The annotation file contains labels and source references.

## Export for independent labeling or fine-tuning

Run from the repository root using Python 3. Output paths must be new.

```bash
python3 data/extended/labels/gpt-6.2-astra-v1/export.py blind claude_inputs.jsonl
python3 data/extended/labels/gpt-6.2-astra-v1/export.py training astra_training.jsonl
```

The blind export contains all 2,438 distinct nonempty-action inputs. Each row contains only `example_id` (the case ID), `formatted_interaction`, and `proposed_action`. It reads directly from Egemen’s source inputs and never opens the Astra annotation file. Give a second annotator that export and the same rubric. Keep its outputs in a separate file joined by the case ID. Compare severity as an ordinal label and domains/drivers as unordered sets. Agreement measures consistency; use an independently reviewed sample to assess correctness.

The default training export contains 1,801 unique full-input, unflagged examples in AgentRisk’s six-field raw training schema. Pass `--include-review` to include all 2,125 available judgments. These rows remain weak labels. No training or development split was changed by this annotation pass.
