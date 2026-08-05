from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from common import (
    ARTIFACT_ROOT,
    MODEL_ROOT,
    PROCESSED_ROOT,
    PROJECT_ROOT,
    configure_environment,
    ensure_project_dirs,
    now_utc,
    read_jsonl,
    write_json,
)


SYSTEM_PROMPT = (
    "You are a precise English-to-German medical-information translator for "
    "administrative use. Translate only. Do not give medical advice, explanations, "
    "or chain-of-thought. Preserve medicine names, numbers, units, dosage, warnings, "
    "negation, and formatting exactly where possible. Respond only with the German translation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "gemma4"], required=True)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--max-train-examples", type=int, default=50000)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def model_config(model_name: str) -> tuple[str, Path]:
    if model_name == "qwen3":
        return "Qwen/Qwen3-4B", MODEL_ROOT / "qwen3-4b-medical-lora"
    return "google/gemma-4-E2B-it", MODEL_ROOT / "gemma4-e2b-medical-lora"


def apply_chat(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def tokenize_example(example: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["source"]},
    ]
    prompt_text = apply_chat(tokenizer, messages, add_generation_prompt=True)
    full_messages = messages + [{"role": "assistant", "content": example["target"]}]
    full_text = apply_chat(tokenizer, full_messages, add_generation_prompt=False)

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = list(input_ids)
    prompt_length = min(len(prompt_ids), len(labels))
    labels[:prompt_length] = [-100] * prompt_length
    if all(value == -100 for value in labels):
        labels[-1] = input_ids[-1]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def load_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def make_training_args(training_args_cls: Any, output_dir: Path, args: argparse.Namespace) -> Any:
    import torch

    common = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "num_train_epochs": 1,
        "max_steps": args.max_steps,
        "logging_steps": 10,
        "save_steps": 500,
        "save_total_limit": 2,
        "eval_steps": 250,
        "logging_dir": str(PROJECT_ROOT / "logs" / output_dir.name),
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "optim": "adamw_torch",
        "seed": 42,
        "data_seed": 42,
        "dataloader_num_workers": 0,
        "ddp_find_unused_parameters": False,
    }
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        common["bf16"] = True
    elif torch.cuda.is_available():
        common["fp16"] = True

    try:
        return training_args_cls(
            eval_strategy="steps",
            save_strategy="steps",
            **common,
        )
    except TypeError:
        return training_args_cls(
            evaluation_strategy="steps",
            save_strategy="steps",
            **common,
        )



def json_safe(value):
    """Convert trainer metadata into values accepted by json.dump."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(json_safe(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value

def main() -> None:
    args = parse_args()
    configure_environment()
    ensure_project_dirs()

    model_id, configured_output = model_config(args.model)
    output_dir = Path(args.output_dir) if args.output_dir else configured_output
    output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    train_path = Path(args.train_file) if args.train_file else PROCESSED_ROOT / "ahazeemi_train.jsonl"
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path
    train_path = train_path.resolve()
    if PROJECT_ROOT not in train_path.parents:
        raise ValueError("--train-file must remain inside the project")
    train_rows = read_jsonl(train_path)[: args.max_train_examples]
    dev_rows = read_jsonl(PROCESSED_ROOT / "ahazeemi_dev.jsonl")
    if not train_rows:
        raise RuntimeError("Prepared training data is missing. Run download_data.py and prepare_data.py first.")

    tokenizer = load_tokenizer(model_id)
    train_dataset = Dataset.from_list(train_rows).map(
        lambda row: tokenize_example(row, tokenizer, args.max_length),
        remove_columns=Dataset.from_list(train_rows).column_names,
        desc="Tokenizing training examples",
    )
    dev_dataset = Dataset.from_list(dev_rows[: min(len(dev_rows), 256)]).map(
        lambda row: tokenize_example(row, tokenizer, args.max_length),
        remove_columns=Dataset.from_list(dev_rows[: min(len(dev_rows), 256)]).column_names,
        desc="Tokenizing validation examples",
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "cache_dir": os.environ["TRANSFORMERS_CACHE"],
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}

    print(f"Loading {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = False
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()

    default_target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    if args.model == "gemma4":
        # Gemma 4 is multimodal: vision/audio projections use custom
        # Gemma4ClippableLinear wrappers that this PEFT version cannot wrap.
        # Select only the native Linear projections in the language tower.
        target_modules = [
            name
            for name, module in model.named_modules()
            if name.startswith("model.language_model.layers.")
            and name.rsplit(".", 1)[-1] in set(default_target_modules)
            and isinstance(module, torch.nn.Linear)
        ]
        if not target_modules:
            raise RuntimeError(
                "Could not find native language-tower Linear targets for Gemma 4."
            )
    else:
        target_modules = default_target_modules

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    train_args = make_training_args(TrainingArguments, output_dir, args)

    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "data_collator": collator,
    }
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "created_at_utc": now_utc(),
        "model_name": args.model,
        "model_id": model_id,
        "output_dir": str(output_dir),
        "train_file": str(train_path),
        "train_examples": len(train_dataset),
        "validation_examples": len(dev_dataset),
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "seed": 42,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": lora_config.target_modules,
        },
        "training_result": {
            "global_step": getattr(result, "global_step", None),
            "training_loss": getattr(result, "training_loss", None),
            "metrics": json_safe(getattr(result, "metrics", {})),
        },
    }
    metadata = json_safe(metadata)
    write_json(output_dir / "training_metadata.json", metadata)
    artifact_name = f"{args.model}_training_metadata"
    if "100k" in output_dir.name:
        artifact_name += "_100k"
    write_json(ARTIFACT_ROOT / f"{artifact_name}.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
