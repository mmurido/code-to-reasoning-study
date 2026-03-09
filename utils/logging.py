import json
import time
from datetime import datetime

import torch
import wandb
from lm_eval.tasks import TaskManager
from omegaconf import OmegaConf


def _print_section(log_handle, title: str) -> None:
    """Print a section title to the log."""
    print(f"\n{title}", file=log_handle, flush=True)


def _count_model_parameters(model) -> tuple[int, int]:
    """Return the number of trainable and total model parameters."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


def _peft_to_dict(peft_cfg) -> dict | None:
    """Turn the PEFT config into a plain dictionary."""
    if peft_cfg is None:
        return None

    if hasattr(peft_cfg, "to_dict"):
        return peft_cfg.to_dict()
    if OmegaConf.is_config(peft_cfg):
        return OmegaConf.to_container(peft_cfg, resolve=True)

    return dict(peft_cfg)


def _get_effective_num_fewshot(
    task: str, requested_num_fewshot: int, log_handle
) -> int:
    """Try to read the effective num_fewshot for the task."""
    effective_num_fewshot = requested_num_fewshot

    try:
        task_manager = TaskManager()
        task_group = task_manager.load_task_or_group(task)

        if task in task_group:
            task_obj = task_group[task]
        elif hasattr(task_group, "tasks") and task in task_group.tasks:
            task_obj = task_group.tasks[task]
        else:
            task_obj = None

        if task_obj and hasattr(task_obj, "num_fewshot"):
            effective_num_fewshot = task_obj.num_fewshot
            print(
                f"  Effective num_fewshot used (from task obj): {effective_num_fewshot}",
                file=log_handle,
                flush=True,
            )
        else:
            print(
                f"  No task-specific num_fewshot found — using requested {requested_num_fewshot}",
                file=log_handle,
                flush=True,
            )

    except Exception as exc:
        print(
            f"  Could not load task details for few-shot check: {exc}",
            file=log_handle,
            flush=True,
        )
        print(
            f"  Using requested num_fewshot: {requested_num_fewshot}",
            file=log_handle,
            flush=True,
        )

    return effective_num_fewshot


def log_timestamp(log_handle) -> None:
    """Write the training start time to the log."""
    print(
        f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        file=log_handle,
        flush=True,
    )


def log_hydra_info(log_handle, cfg) -> None:
    """Write the full Hydra config to the log."""
    print(f"Full Hydra config:\n{OmegaConf.to_yaml(cfg)}", file=log_handle, flush=True)


def log_hardware_info(log_handle) -> None:
    """Write GPU and CUDA information to the log."""
    print("Hardware info:", file=log_handle, flush=True)
    print(f"GPUs detected: {torch.cuda.device_count()}", file=log_handle, flush=True)

    for gpu_index in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(gpu_index)
        gpu_memory_gb = torch.cuda.get_device_properties(gpu_index).total_memory / 1e9
        print(
            f"  GPU {gpu_index}: {gpu_name} | Mem: {gpu_memory_gb:.1f} GB",
            file=log_handle,
            flush=True,
        )

    print(f"CUDA version: {torch.version.cuda}", file=log_handle, flush=True)


def log_trainer_info(log_handle, trainer) -> None:
    """Write the trainer arguments to the log."""
    _print_section(log_handle, "Full TrainingArguments / SFTConfig:")
    print(json.dumps(trainer.args.to_dict(), indent=2), file=log_handle, flush=True)


def log_peft_info(log_handle, peft_cfg) -> None:
    """Write the PEFT config to the log and wandb."""
    _print_section(log_handle, "PEFT config:")

    peft_dict = _peft_to_dict(peft_cfg)
    if peft_dict is None:
        return

    print(json.dumps(peft_dict, indent=2, default=str), file=log_handle, flush=True)

    if wandb.run is not None:
        wandb.config.update({"peft_config": peft_dict})


def log_model_info(log_handle, model) -> None:
    """Write model parameter counts to the log and wandb."""
    _print_section(log_handle, "Model parameters:")

    trainable_params, total_params = _count_model_parameters(model)
    trainable_pct = trainable_params / total_params * 100

    print(
        f"Trainable: {trainable_params:,} ({trainable_pct:.2f}%)",
        file=log_handle,
        flush=True,
    )
    print(f"Total:     {total_params:,}", file=log_handle, flush=True)

    if wandb.run is not None:
        wandb.config.update(
            {
                "trainable_parameters": trainable_params,
                "total_parameters": total_params,
                "trainable_pct": trainable_pct,
            }
        )


def save_metadata(log_dir, cfg, trainer, model, peft_cfg):
    """Save the main experiment metadata to a JSON file."""
    trainable_params, total_params = _count_model_parameters(model)
    peft_dict = _peft_to_dict(peft_cfg)

    metadata = {
        "timestamp_start": datetime.now().isoformat(),
        "hydra_config": OmegaConf.to_container(cfg, resolve=True),
        "training_args": trainer.args.to_dict(),
        "peft_config": peft_dict,
        "model": {
            "hf_id": cfg.model.hf_id,
            "name": cfg.model.name,
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": trainable_params / total_params * 100,
        },
    }

    metadata_file = log_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as metadata_handle:
        json.dump(metadata, metadata_handle, indent=2, default=str)

    print(f"\nFull experiment info saved to: {metadata_file}", flush=True)
    return metadata_file


def log_eval_config(
    log_handle,
    task: str,
    lm,
    num_fewshot: int,
    batch_size: int,
    start_time: float = None,
) -> dict:
    """Write the main evaluation settings to the log and return a summary."""
    print(f"\nEvaluation started for task: {task}", file=log_handle, flush=True)
    print(
        f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        file=log_handle,
        flush=True,
    )

    _print_section(log_handle, "Model:")
    print(f"  Pretrained: {lm.model.config.name_or_path}", file=log_handle, flush=True)

    if hasattr(lm, "peft_config") and lm.peft_config:
        print(
            f"  PEFT adapter: {lm.peft_config.peft_model_id}",
            file=log_handle,
            flush=True,
        )

    print(f"  Batch size (requested): {batch_size}", file=log_handle, flush=True)
    print(f"  Effective batch size: {lm.batch_size}", file=log_handle, flush=True)

    effective_num_fewshot = _get_effective_num_fewshot(task, num_fewshot, log_handle)

    _print_section(log_handle, "Generation parameters:")
    print(f"  max_new_tokens: {512} (from config)", file=log_handle, flush=True)
    print(f"  temperature: {0.0}", file=log_handle, flush=True)
    print(f"  top_p: {1.0}", file=log_handle, flush=True)
    print(f"  do_sample: {False}", file=log_handle, flush=True)

    if start_time:
        runtime_so_far = time.time() - start_time
        print(
            f"\nEvaluation runtime so far: {runtime_so_far / 60:.1f} minutes",
            file=log_handle,
            flush=True,
        )

    config_summary = {
        "task": task,
        "effective_num_fewshot": effective_num_fewshot,
        "requested_num_fewshot": num_fewshot,
        "batch_size_requested": batch_size,
        "batch_size_effective": lm.batch_size,
        "model_name": lm.model.config.name_or_path,
        "peft_used": bool(lm.peft_config) if hasattr(lm, "peft_config") else False,
        "start_time": datetime.now().isoformat(),
    }

    return config_summary
