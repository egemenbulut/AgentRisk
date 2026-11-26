"""Run the frozen AgentRisk protocol on a four-bit Llama with MLX."""
import argparse
import fcntl
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load, stream_generate
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load_tokenizer

from protocol import ROOT, dataset, messages, parse, save_json, save_rows

MODELS = {
    "3b": ("mlx-community/Llama-3.2-3B-Instruct-4bit", "7f0dc925e0d0afb0322d96f9255cfddf2ba5636e"),
    "8b": ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "241a666dad6cb93c8ff213d39a7f34a36bf26db4"),
}
LORA = {"rank": 64, "scale": 2.0, "dropout": 0.0,
        "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]}
MAX_LENGTH = 4096
MAX_NEW = 128


def render(tokenizer, row, answer=False):
    return tokenizer.apply_chat_template(messages(row, answer), tokenize=True,
                                         date_string="25 Nov 2026", add_generation_prompt=not answer)


def encode(tokenizer, train, dev):
    encoded, prompts = {}, {}
    for key, row in train.items():
        prefix = render(tokenizer, row)
        full = render(tokenizer, row, True)
        if full[:len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError(f"{key}: assistant boundary mismatch")
        suffix = tokenizer.decode(full[len(prefix):], skip_special_tokens=True)
        if json.loads(suffix) != json.loads(messages(row, True)[-1]["content"]):
            raise ValueError(f"{key}: wrong unmasked answer")
        if len(full) > MAX_LENGTH or len(prefix) + MAX_NEW > MAX_LENGTH:
            raise ValueError(f"{key}: context exceeds {MAX_LENGTH}. No truncation applied.")
        encoded[key] = (full, len(prefix))
    for key, row in dev.items():
        prompts[key] = render(tokenizer, row)
        if len(prompts[key]) + MAX_NEW > MAX_LENGTH:
            raise ValueError(f"{key}: generation would exceed {MAX_LENGTH}")
    print(f"{len(encoded)} training rows, longest {max(len(x[0]) for x in encoded.values())} tokens. "
          f"{len(prompts)} dev prompts, longest {max(map(len, prompts.values()))} tokens.", flush=True)
    return encoded, prompts


def response_loss(model, tokens, start):
    # Causal shift: the final prompt token predicts the first answer token.
    # Only project the short answer span to vocabulary logits to save memory.
    ids = mx.array(tokens)[None]
    hidden = model.model(ids[:, :-1])[:, start-1:]
    logits = model.model.embed_tokens.as_linear(hidden) if model.args.tie_word_embeddings else model.lm_head(hidden)
    return nn.losses.cross_entropy(logits.astype(mx.float32), ids[:, start:], reduction="mean")


def validation(model, encoded, keys):
    model.eval()
    total, count = 0., 0
    for key in keys:
        tokens, start = encoded[key]
        n = len(tokens) - start
        total += n * response_loss(model, tokens, start).item()
        count += n
    model.train()
    return total / count


def train_adapter(model, encoded, train_keys, val_keys, fold, folder, smoke, preview=None):
    seed = 3407 + fold
    rng = random.Random(seed)
    mx.random.seed(seed)
    model.freeze()
    linear_to_lora_layers(model, len(model.layers), LORA)
    leaves = tree_flatten(model.trainable_parameters())
    if len(leaves) != 2 * len(LORA["keys"]) * len(model.layers):
        raise ValueError("LoRA projections did not match every Llama layer")
    print(f"{sum(x.size for _, x in leaves):,} trainable parameters; device={mx.default_device()}", flush=True)
    grad_checkpoint(model.layers[0])
    model.train()
    value_grad = nn.value_and_grad(model, response_loss)
    epochs = 1 if smoke else 3
    updates = epochs * math.ceil(len(train_keys) / 8)
    optimizer = optim.AdamW(learning_rate=2e-4, weight_decay=0.01, bias_correction=True)
    best, step, history = math.inf, 0, []
    for _ in range(epochs):
        rng.shuffle(train_keys)
        for i in range(0, len(train_keys), 8):
            group = train_keys[i:i+8]
            accum, total_loss = None, 0.
            for key in group:
                value, gradients = value_grad(model, *encoded[key])
                accum = gradients if accum is None else tree_map(lambda a, b: a+b, accum, gradients)
                mx.eval(value, accum)
                total_loss += value.item()
            accum = tree_map(lambda x: x / len(group), accum)
            clipped, _norm = optim.clip_grad_norm(accum, 1.0)
            # Schedule uses optimizer updates; the smoke test takes a nonzero step.
            scale = 1. if smoke else min(step / 10, (updates-step) / max(1, updates-10))
            optimizer.learning_rate = 2e-4 * scale
            optimizer.update(model, clipped)
            mx.eval(model.trainable_parameters(), optimizer.state)
            del accum, gradients, clipped
            step += 1
            if not math.isfinite(total_loss):
                raise FloatingPointError(f"Nonfinite loss at update {step}")
            if step == 1 or step % 10 == 0:
                print(f"Fold {fold} update {step}/{updates} loss={total_loss/len(group):.4f} "
                      f"peak={mx.get_peak_memory()/1e9:.1f} GB", flush=True)
            if preview is not None and step in (8, 20):
                model.eval()
                try:
                    preview(model, step)
                finally:
                    model.train()
            if step % 20 == 0 or step == updates:
                loss = validation(model, encoded, val_keys)
                if not math.isfinite(loss):
                    raise FloatingPointError("Nonfinite validation loss")
                history.append({"update": step, "validation_loss": loss})
                print(f"Fold {fold} update {step}: validation loss {loss:.4f}", flush=True)
                if loss < best:
                    best = loss
                    weights = dict(tree_flatten(model.trainable_parameters()))
                    mx.save_safetensors(str(folder / "next.safetensors"), weights)
                    os.replace(folder / "next.safetensors", folder / "adapters.safetensors")
    save_json(folder / "adapter_config.json", {"fine_tune_type": "lora", "num_layers": len(model.layers),
                                              "lora_parameters": LORA})
    save_json(folder / "training.json", {"seed": seed, "validations": history,
                                        "best": min(history, key=lambda h: h["validation_loss"])})


def digit_ids(tokenizer):
    digits = [tokenizer.encode(str(x), add_special_tokens=False) for x in range(1, 6)]
    if any(len(x) != 1 for x in digits) or len({x[0] for x in digits}) != 5:
        raise ValueError("Expected distinct single-token severity digits")
    return [x[0] for x in digits]


def predict(model, tokenizer, inputs):
    digits = digit_ids(tokenizer)
    indices = mx.array(digits)
    for key, prefix in inputs.items():
        generated, probabilities = [], None
        for result in stream_generate(model, tokenizer, prefix, max_tokens=MAX_NEW):
            if result.token in tokenizer.eos_token_ids:
                break
            if result.token in digits and re.search(r'"risk_severity"\s*:\s*$', tokenizer.decode(generated)):
                probabilities = mx.softmax(result.logprobs[indices].astype(mx.float32)).tolist()
            generated.append(result.token)
        raw = tokenizer.decode(generated, skip_special_tokens=True)
        yield {"example_id": key, "raw_text": raw, "parsed": parse(raw), "severity_probs": probabilities}


def save_predictions(path, predictions):
    """Publish snapshots with complete JSON lines while a split is being scored."""
    partial = path.with_suffix(".partial.jsonl")
    rows = []
    for row in predictions:
        rows.append(row)
        if len(rows) == 1 or len(rows) % 4 == 0:
            save_rows(partial, rows)
    save_rows(path, rows)
    partial.unlink(missing_ok=True)


def run_fold(args, path, model_id, revision, folder):
    config = {"model": model_id, "revision": revision, "smoke": args.action == "smoke",
              "mlx_lm": version("mlx-lm"), "mlx": version("mlx"), "transformers": version("transformers"),
              "max_length": MAX_LENGTH, "max_new_tokens": MAX_NEW, "lora": LORA}
    config_file = folder / "run_config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError(f"Different run settings already exist in {folder}. Select a new --output.")
    save_json(config_file, config)
    if (folder / "done.json").exists():
        print(f"Fold {args.fold} already completed at {folder}")
        return
    started = time.monotonic()
    train, dev, folds = dataset()
    tokenizer = load_tokenizer(path)
    encoded, dev_inputs = encode(tokenizer, train, dev)
    train_keys = [k for k in train if folds[k] != args.fold]
    val_keys = [k for k in train if folds[k] == args.fold]
    smoke = config["smoke"]
    if smoke:
        # Stress the longest eligible row and keep dev labels out of the smoke test.
        train_keys = [max(train_keys, key=lambda k: len(encoded[k][0]))]
        val_keys = val_keys[:1]
    if not (folder / "training.json").exists():
        model, _ = load(path)
        preview = None
        if args.fold == 0 and not smoke and getattr(args, "meeting", False):
            # Independent RNG and fixed held-out IDs. Preview scores never select checkpoints.
            sample = random.Random(144).sample(sorted(val_keys), 24)
            preview_inputs = {k: encoded[k][0][:encoded[k][1]] for k in sample}
            def preview(model, step):
                dest = folder / f"preview_{step}.jsonl"
                print(f"Meeting preview at update {step} on {len(sample)} held-out cases", flush=True)
                save_predictions(dest, predict(model, tokenizer, preview_inputs))
        train_adapter(model, encoded, train_keys, val_keys, args.fold, folder, smoke, preview)
        del model
        mx.clear_cache()
    model, tokenizer = load(path, adapter_path=str(folder))
    val_inputs = {k: encoded[k][0][:encoded[k][1]] for k in val_keys}
    splits = {"val": val_inputs}
    if not smoke:
        splits["dev"] = dev_inputs
    for split, inputs in splits.items():
        dest = folder / f"{split}.jsonl"
        if not dest.exists():
            print(f"Predicting {len(inputs)} {split} rows", flush=True)
            save_predictions(dest, predict(model, tokenizer, inputs))
    save_json(folder / "done.json", {"config": config, "fold": args.fold,
                                    "elapsed_this_invocation_seconds": time.monotonic()-started,
                                    "peak_gb": mx.get_peak_memory()/1e9})
    print(f"Fold {args.fold} complete: {folder}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["check", "smoke", "run"])
    p.add_argument("--meeting", action="store_true", help="Also score a fixed fold-0 sample after updates 8 and 20")
    p.add_argument("--size", choices=MODELS, default="3b")
    p.add_argument("--fold", type=int, choices=range(5))
    p.add_argument("--output", type=Path, help="Output root for this run")
    args = p.parse_args()
    root = args.output or ROOT / "output" / f"llama-{args.size}-{'smoke' if args.action == 'smoke' else 'fivefold'}"
    if args.action == "run" and args.fold is None:
        for fold in range(5):
            subprocess.run([sys.executable, __file__, *sys.argv[1:], "--fold", str(fold)], check=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name("compare.py")), "--llama", str(root),
                        "--output", str(root / "comparison")], check=True)
        return
    model_id, revision = MODELS[args.size]
    patterns = ["*.json", "*.jinja", "*.model"] if args.action == "check" else ["*.json", "*.jinja", "*.model", "*.safetensors"]
    path = snapshot_download(model_id, revision=revision, allow_patterns=patterns)
    if args.action == "check":
        train, dev, _ = dataset()
        tokenizer = load_tokenizer(path)
        digit_ids(tokenizer)
        encode(tokenizer, train, dev)
        return
    if mx.default_device() != mx.Device(mx.gpu):
        p.exit(2, "Training needs the Metal or CUDA GPU backend. Check your Python and MLX installation.\n")
    args.fold = 0 if args.fold is None else args.fold
    folder = root / f"fold_{args.fold}"
    folder.mkdir(exist_ok=True, parents=True)
    with (folder / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_fold(args, path, model_id, revision, folder)


if __name__ == "__main__":
    main()
