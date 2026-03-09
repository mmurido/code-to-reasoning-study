from trl import SFTConfig, SFTTrainer


def build_trainer(
    cfg,
    model,
    dataset,
    peft_cfg,
    exp_dir,
):
    """Build the trainer used for fine-tuning."""

    training_cfg = cfg.training

    sft_config = SFTConfig(
        output_dir=str(exp_dir / "train/checkpoints"),
        seed=training_cfg.seed,
        report_to="wandb",
        run_name=cfg.run_id,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        max_steps=training_cfg.max_steps,
        learning_rate=training_cfg.learning_rate,
        warmup_steps=training_cfg.warmup_steps,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        weight_decay=training_cfg.weight_decay,
        fp16=training_cfg.fp16,
        max_grad_norm=training_cfg.max_grad_norm,
        gradient_checkpointing=training_cfg.gradient_checkpointing,
        dataloader_num_workers=training_cfg.dataloader_num_workers,
        logging_steps=training_cfg.logging_steps,
        logging_strategy=training_cfg.logging_strategy,
        log_level=training_cfg.log_level,
        disable_tqdm=training_cfg.disable_tqdm,
        save_steps=training_cfg.save_steps,
        save_total_limit=training_cfg.save_total_limit,
        max_seq_length=training_cfg.max_seq_length,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_cfg,
        args=sft_config,
    )

    trainer.can_return_loss = True
    return trainer
