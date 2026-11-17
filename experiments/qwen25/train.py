"""Single-CUDA-GPU QLoRA; original severity labels; assistant-completion loss only."""
import argparse
import json
from pathlib import Path

from data_utils import DATA, encode_training, pad_features, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--train-file', type=Path, default=DATA/'train_extended_sft.jsonl')
    parser.add_argument('--output-dir', default='output/qwen25-1.5b-extended')
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--epochs', type=float, default=2)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--gradient-accumulation', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    rows = read_jsonl(args.train_file)
    if not rows:
        raise ValueError('Empty training file')
    encoded = []
    errors = []
    for row in rows:
        try:
            encoded.append(encode_training(row, tokenizer, args.max_length))
        except ValueError as error:
            errors.append(str(error))
    if errors:
        raise ValueError(f'{len(errors)} unusable rows; no training started.\n' + '\n'.join(errors[:20]))
    lengths = sorted(len(r['input_ids']) for r in encoded)
    print(json.dumps(dict(rows=len(rows), max_tokens=lengths[-1],
                          median_tokens=lengths[len(lengths)//2],
                          p95_tokens=lengths[int((len(lengths)-1)*.95)],
                          loss='assistant completion only', truncated_rows=0), indent=2))
    if args.preflight_only:
        return
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError('This QLoRA recipe requires a supported CUDA GPU; preflight works without GPU weights.')
    set_seed(args.seed)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map={'': torch.cuda.current_device()},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=dtype),
        attn_implementation='sdpa',
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                                          target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                                          bias='none', task_type='CAUSAL_LM'))
    model.print_trainable_parameters()

    def collate(features):
        return {k: torch.tensor(v, dtype=torch.long)
                for k, v in pad_features(features, tokenizer.pad_token_id).items()}

    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, warmup_ratio=.05, lr_scheduler_type='cosine',
        bf16=bf16, fp16=not bf16, optim='adamw_torch',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        save_strategy='epoch', save_total_limit=2, logging_steps=10, report_to='none',
        seed=args.seed, data_seed=args.seed, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=encoded,
                      data_collator=collate, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    Path(args.output_dir, 'run_config.json').write_text(json.dumps(vars(args), indent=2, default=str)+'\n')


if __name__ == '__main__':
    main()
