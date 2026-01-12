!pip install -q transformers datasets peft accelerate trl

# Imports
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
import huggingface_hub
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)

wandb.login(key="...")
huggingface_hub.login(token="...")

# Constants
MODEL_NAME = "EleutherAI/pythia-160m"
DATASET_NAME = "bigcode/starcoderdata"

OUTPUT_DIR = "./outputs"
OUTPUT_MODEL_DIR = "./lora_pythia_160m"

SEED = 42

# Dataset
dataset = load_dataset(
    DATASET_NAME,
    data_dir="java",
    split="train",
    streaming=True
)

dataset = dataset.take(50000)
dataset = dataset.map(lambda x: {"text": x["content"]})

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

model.config.use_cache = False

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
)

# Training arguments
training_arguments = TrainingArguments(
    report_to="wandb",
    run_name="pythia-160m-lora",
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    max_steps=1000,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=3,
    seed=SEED,
    data_seed=SEED,
    optim="adamw_torch",
    gradient_checkpointing=True,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_arguments,
)

# Training
trainer.train()

# Saving the model
trainer.model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
wandb.finish()
