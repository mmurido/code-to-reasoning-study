from transformers import TrainingArguments
from trl import SFTTrainer


def build_trainer(
    cfg,
    model,
    dataset,
    peft_cfg,
    exp_dir,
):
    t = cfg.training
    args = TrainingArguments(
        output_dir=str(exp_dir / "train/checkpoints"),
        seed=t.seed,
        report_to="wandb",
        run_name=cfg.run.id,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        max_steps=t.max_steps,
        learning_rate=t.learning_rate,
        warmup_steps=t.warmup_steps,
        lr_scheduler_type=t.lr_scheduler_type,
        weight_decay=t.weight_decay,
        fp16=t.fp16,
        max_grad_norm=t.max_grad_norm,
        gradient_checkpointing=t.gradient_checkpointing,
        dataloader_num_workers=t.dataloader_num_workers,
        logging_steps=t.logging_steps,
        logging_strategy=t.logging_strategy,
        log_level=t.log_level,
        disable_tqdm=t.disable_tqdm,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
    )

    return SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_cfg,
        args=args,
        max_seq_length=t.max_seq_length,
    )
