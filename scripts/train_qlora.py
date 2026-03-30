import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_TRAIN_DATASET = "data/train.jsonl"
DEFAULT_VALIDATION_DATASET = "data/validation.jsonl"


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(model_name: str, use_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": False}

    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        if use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    return model, tokenizer


def format_training_example(example: dict) -> dict:
    metadata = example.get("metadata") or {}
    source = metadata.get("source", "unknown")
    title = metadata.get("title", "")

    prompt = (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Context:\n"
        f"{example['context']}\n\n"
        "### Response:\n"
        f"{example['response']}\n\n"
        "### Source:\n"
        f"{source}\n"
        "### Title:\n"
        f"{title}"
    )
    return {"text": prompt}


def load_formatted_dataset(dataset_path: str):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.filter(lambda row: bool(str(row.get("response", "")).strip()))
    if len(dataset) == 0:
        raise ValueError(f"Dataset has no annotated responses: {dataset_path}")
    return dataset.map(format_training_example, remove_columns=dataset.column_names)


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_training_args(
    output_dir: str,
    learning_rate: float,
    epochs: int,
    has_validation: bool,
):
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if has_validation else "no",
        load_best_model_at_end=has_validation,
        metric_for_best_model="eval_loss" if has_validation else None,
        greater_is_better=False if has_validation else None,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        report_to="none",
        fp16=torch.cuda.is_available() and not use_bf16,
        bf16=use_bf16,
    )


def train(
    model_name: str,
    train_dataset_path: str,
    validation_dataset_path: str | None,
    output_dir: str,
    learning_rate: float,
    epochs: int,
    max_seq_length: int,
    use_4bit: bool,
):
    device = detect_device()
    print(f"Training device: {device}")
    print(f"Base model: {model_name}")
    print(f"Train dataset: {train_dataset_path}")
    print(f"Validation dataset: {validation_dataset_path or 'none'}")

    model, tokenizer = load_model_and_tokenizer(model_name, use_4bit=use_4bit)
    train_dataset = load_formatted_dataset(train_dataset_path)
    eval_dataset = load_formatted_dataset(validation_dataset_path) if validation_dataset_path else None
    training_args = build_training_args(
        output_dir=output_dir,
        learning_rate=learning_rate,
        epochs=epochs,
        has_validation=eval_dataset is not None,
    )
    lora_config = build_lora_config()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        args=training_args,
        max_seq_length=max_seq_length,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved adapter and tokenizer to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train a QLoRA/SFT adapter for Finly on Linux + NVIDIA")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Base Hugging Face model id")
    parser.add_argument("--train-dataset-path", default=DEFAULT_TRAIN_DATASET, help="Annotated JSONL training split")
    parser.add_argument(
        "--validation-dataset-path",
        default=DEFAULT_VALIDATION_DATASET,
        help="Annotated JSONL validation split; pass an empty value to disable evaluation",
    )
    parser.add_argument("--output-dir", default="artifacts/finly-lora", help="Directory to save LoRA adapter files")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--no-4bit", action="store_true", help="Disable bitsandbytes 4-bit loading")

    args = parser.parse_args()

    train_dataset_path = Path(args.train_dataset_path)
    if not train_dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_dataset_path}")

    validation_dataset_path = None
    if args.validation_dataset_path.strip():
        candidate = Path(args.validation_dataset_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Validation dataset not found: {candidate}")
        validation_dataset_path = str(candidate)

    train(
        model_name=args.model_name,
        train_dataset_path=str(train_dataset_path),
        validation_dataset_path=validation_dataset_path,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        use_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()
