import json
import wandb
import torch
import time
from datetime import datetime
from omegaconf import OmegaConf
from lm_eval.tasks import TaskManager


def log_timestamp(log_f):
    print(
        f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        file=log_f,
        flush=True,
    )


def log_hydra_info(log_f, cfg):
    print(f"Full Hydra config:\n{OmegaConf.to_yaml(cfg)}", file=log_f, flush=True)


def log_hardware_info(log_f):
    print("Hardware info:", file=log_f, flush=True)
    print(f"GPUs detected: {torch.cuda.device_count()}", file=log_f, flush=True)
    for i in range(torch.cuda.device_count()):
        print(
            f"  GPU {i}: {torch.cuda.get_device_name(i)} | Mem: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB",
            file=log_f,
            flush=True,
        )
    print(f"CUDA version: {torch.version.cuda}", file=log_f, flush=True)


def log_trainer_info(log_f, trainer):
    print("\nFull TrainingArguments / SFTConfig:", file=log_f, flush=True)
    print(json.dumps(trainer.args.to_dict(), indent=2), file=log_f, flush=True)


def log_peft_info(log_f, peft_cfg):
    print("\nPEFT config:", file=log_f, flush=True)
    if peft_cfg is not None:
        peft_dict = (
            OmegaConf.to_container(peft_cfg, resolve=True)
            if OmegaConf.is_config(peft_cfg)
            else dict(peft_cfg)
        )
        print(json.dumps(peft_dict, indent=2, default=str), file=log_f, flush=True)

    if wandb.run is not None:
        wandb.config.update({"peft_config": peft_dict})


def log_model_info(log_f, model):
    print("\nModel parameters:", file=log_f, flush=True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable: {trainable_params:,} ({trainable_params / all_params * 100:.2f}%)",
        file=log_f,
        flush=True,
    )
    print(f"Total:     {all_params:,}", file=log_f, flush=True)

    if wandb.run is not None:
        wandb.config.update(
            {
                "trainable_parameters": trainable_params,
                "total_parameters": all_params,
                "trainable_pct": trainable_params / all_params * 100,
            }
        )


def save_metadata(log_dir, cfg, trainer, model, peft_cfg):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    peft_dict = (
        OmegaConf.to_container(peft_cfg, resolve=True)
        if OmegaConf.is_config(peft_cfg)
        else dict(peft_cfg)
    )

    full_info = {
        "timestamp_start": datetime.now().isoformat(),
        "hydra_config": OmegaConf.to_container(cfg, resolve=True),
        "training_args": trainer.args.to_dict(),
        "peft_config": peft_dict,
        "model": {
            "hf_id": cfg.model.hf_id,
            "name": cfg.model.name,
            "trainable_parameters": trainable_params,
            "total_parameters": all_params,
            "trainable_percentage": trainable_params / all_params * 100,
        },
    }

    full_info_file = log_dir / "metadata.json"
    with open(full_info_file, "w", encoding="utf-8") as f:
        json.dump(full_info, f, indent=2, default=str)

    print(f"\nFull experiment info saved to: {full_info_file}", flush=True)
    return full_info_file


def log_eval_config(
    log_f,
    task: str,
    lm,
    num_fewshot: int,
    batch_size: int,
    start_time: float = None,
) -> dict:
    print(f"\nEvaluation started for task: {task}", file=log_f, flush=True)
    print(
        f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        file=log_f,
        flush=True,
    )

    # Model info
    print("\nModel:", file=log_f, flush=True)
    print(f"  Pretrained: {lm.model.config.name_or_path}", file=log_f, flush=True)
    if hasattr(lm, "peft_config") and lm.peft_config:
        print(f"  PEFT adapter: {lm.peft_config.peft_model_id}", file=log_f, flush=True)
    print(f"  Batch size (requested): {batch_size}", file=log_f, flush=True)
    print(f"  Effective batch size: {lm.batch_size}", file=log_f, flush=True)

    # Few-shot info
    effective_fewshot = num_fewshot
    try:
        task_manager = TaskManager()
        task_group = task_manager.load_task_or_group(task)
        # Try common keys or fall back
        if task in task_group:
            task_obj = task_group[task]
        elif hasattr(task_group, "tasks") and task in task_group.tasks:
            task_obj = task_group.tasks[task]
        else:
            task_obj = None

        if task_obj and hasattr(task_obj, "num_fewshot"):
            effective_fewshot = task_obj.num_fewshot
            print(
                f"  Effective num_fewshot used (from task obj): {effective_fewshot}",
                file=log_f,
                flush=True,
            )
        else:
            print(
                f"  No task-specific num_fewshot found — using requested {num_fewshot}",
                file=log_f,
                flush=True,
            )
    except Exception as e:
        print(
            f"  Could not load task details for few-shot check: {e}",
            file=log_f,
            flush=True,
        )
        print(f"  Using requested num_fewshot: {num_fewshot}", file=log_f, flush=True)

    # Generation / other params
    print("\nGeneration parameters:", file=log_f, flush=True)
    print(f"  max_new_tokens: {512} (from config)", file=log_f, flush=True)
    print(f"  temperature: {0.0}", file=log_f, flush=True)
    print(f"  top_p: {1.0}", file=log_f, flush=True)
    print(f"  do_sample: {False}", file=log_f, flush=True)

    if start_time:
        runtime_so_far = time.time() - start_time
        print(
            f"\nEvaluation runtime so far: {runtime_so_far / 60:.1f} minutes",
            file=log_f,
            flush=True,
        )

    config_summary = {
        "task": task,
        "effective_num_fewshot": effective_fewshot,
        "requested_num_fewshot": num_fewshot,
        "batch_size_requested": batch_size,
        "batch_size_effective": lm.batch_size,
        "model_name": lm.model.config.name_or_path,
        "peft_used": bool(lm.peft_config) if hasattr(lm, "peft_config") else False,
        "start_time": datetime.now().isoformat(),
    }

    return config_summary
