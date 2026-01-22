from transformers import TrainingArguments
from trl import SFTTrainer


def build_trainer(model, data, peft_cfg, cfg):
    t = cfg["training"]

    args = TrainingArguments(
        report_to="wandb",
        run_name="lora_run",
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["lr"],
        max_steps=t["max_steps"],
        warmup_steps=t["warmup"],
        lr_scheduler_type=t["scheduler"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        fp16=t["fp16"],
        save_steps=t["save_steps"],
        seed=cfg["seed"],
        optim="adamw_torch",
        gradient_checkpointing=t["gradient_checkpointing"],
        dataloader_num_workers=t["num_workers"],
        logging_steps=t["logging_steps"],
        logging_strategy=t["logging_strategy"],
        log_level=t["log_level"],
        disable_tqdm=t["disable_tqdm"],
    )

    return SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=peft_cfg,
        args=args,
        max_seq_length=t["max_seq_length"],
    )
