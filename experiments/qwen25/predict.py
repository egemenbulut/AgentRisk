"""Generate strict JSON risk predictions from either base Qwen or a saved adapter."""
import argparse
import json
from pathlib import Path

from data_utils import DATA, read_jsonl, validate_target


def parse_prediction(text):
    # Intentionally do not repair, coerce or silently discard invalid outputs.
    target = json.loads(text.strip())
    validate_target(target)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--adapter', default=None)
    p.add_argument('--input', type=Path, default=DATA/'dev_inference.jsonl')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--max-new-tokens', type=int, default=192)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} already exists; select a fresh output path')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError('This 4-bit inference recipe requires a supported CUDA GPU')
    model_id = args.model
    if args.adapter:
        from peft import PeftConfig
        model_id = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or model_id)
    tokenizer.pad_token = tokenizer.eos_token
    rows = read_jsonl(args.input)
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate input IDs')
    prompts = []
    for row in rows:
        if [m['role'] for m in row['messages']] != ['system', 'user']:
            raise ValueError('Inference input must contain only system and user messages')
        token_ids = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True)
        if len(token_ids) + args.max_new_tokens > args.max_length:
            raise ValueError(f'{row["id"]}: prompt plus output budget exceeds max length; no truncation applied')
        prompts.append(token_ids)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map={'': torch.cuda.current_device()},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=dtype),
        attn_implementation='sdpa')
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.config.use_cache = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        for row, ids in zip(rows, prompts):
            tokens = torch.tensor([ids], device=model.device)
            with torch.inference_mode():
                result = model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                        do_sample=False, max_new_tokens=args.max_new_tokens,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id)
            completion = tokenizer.decode(result[0, len(ids):], skip_special_tokens=True)
            record = {'id': row['id'], 'raw_output': completion}
            try:
                record.update(parse_prediction(completion))
                record['valid_output'] = True
            except (ValueError, TypeError, KeyError) as error:
                record.update(valid_output=False, error=str(error))
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
            f.flush()
    print(f'Wrote {len(rows)} predictions to {args.output}')


if __name__ == '__main__':
    main()
