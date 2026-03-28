import os
import torch
import torch_directml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,  # Note: bitsandbytes may not work with DirectML; using as fallback
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# DirectML device setup
device = torch_directml.device()
print(f"Using DirectML device: {device}")

def load_model_and_tokenizer(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    # Note: bitsandbytes quantization may not be compatible with DirectML
    # Using 8-bit or half precision instead
    try:
        # Try 8-bit config (may not work with DirectML)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",  # Let transformers handle device placement
        )
    except Exception as e:
        print(f"8-bit quantization failed: {e}. Falling back to half precision.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device},  # Explicitly place on DirectML device
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def setup_lora(model):
    lora_config = LoraConfig(
        r=16,  # Low-rank dimension
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model

def train_model(model, tokenizer, dataset_path="data/curated_dataset_annotated.jsonl"):
    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Small batch for low VRAM
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        save_steps=500,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,  # Use mixed precision if supported
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none",  # Disable wandb/tensorboard for simplicity
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=None,  # Already applied LoRA
        dataset_text_field="context",  # Use context as input
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    trainer.train()
    return trainer

def save_model(trainer, output_dir="./fin_instruct_model"):
    trainer.model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    # Set environment for reproducibility
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # Not CUDA, but may help memory

    model, tokenizer = load_model_and_tokenizer()
    model = setup_lora(model)
    trainer = train_model(model, tokenizer)
    save_model(trainer)