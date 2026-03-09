import contextlib
import json
import os
import time
import traceback
import torch
from pathlib import Path
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from omegaconf import DictConfig
from utils.baseline import get_baseline_dir
from utils.json import to_json_safe
from utils.logging import log_eval_config


def _configure_cuda() -> None:
    """Apply CUDA settings used before and after evaluation."""

    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def _load_evaluation_model(
    model_name: str,
    batch_size: int,
    peft_path: Path | None,
):
    """Load the model used for evaluation."""

    if peft_path is None:
        return HFLM(
            pretrained=model_name,
            batch_size=batch_size,
            parallelize=True,
        )

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    adapted_model = PeftModel.from_pretrained(
        base_model,
        peft_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    adapted_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return HFLM(
        pretrained=adapted_model,
        tokenizer=tokenizer,
    )


def _resolve_generation_kwargs(
    generation_kwargs: dict | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> dict:
    """Prepare generation settings used for evaluation."""

    if generation_kwargs is not None:
        return generation_kwargs

    return {
        "num_beams": 1,
        "early_stopping": True,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
    }


def _evaluate_task(
    task_name: str,
    eval_model,
    output_dir: Path,
    num_fewshot: int,
    batch_size: int,
    generation_kwargs: dict,
) -> None:
    """Run evaluation for one task."""

    task_dir = output_dir / task_name
    results_file = task_dir / "results.json"
    config_file = task_dir / "config.json"

    if results_file.exists():
        print(f"Skipping existing eval for {task_name}")
        return

    task_dir.mkdir(parents=True, exist_ok=True)
    log_file = task_dir / "eval.log"
    task_start_time = time.time()

    with (
        open(log_file, "w", encoding="utf-8") as log_handle,
        contextlib.redirect_stdout(log_handle),
        contextlib.redirect_stderr(log_handle),
    ):
        config_summary = log_eval_config(
            log_f=log_handle,
            task=task_name,
            lm=eval_model,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            start_time=task_start_time,
        )

        try:
            print(f"Running evaluation for {task_name}...", flush=True)

            results = evaluator.simple_evaluate(
                model=eval_model,
                tasks=[task_name],
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                log_samples=True,
                gen_kwargs=generation_kwargs,
            )

            temp_results_file = results_file.with_suffix(".json.tmp")
            with open(temp_results_file, "w", encoding="utf-8") as results_handle:
                json.dump(to_json_safe(results), results_handle, indent=2)

            temp_results_file.replace(results_file)
            print(f"Results saved to: {results_file}", flush=True)

            runtime_seconds = time.time() - task_start_time
            print(
                f"Task {task_name} completed in {runtime_seconds / 60:.1f} minutes.",
                flush=True,
            )

            with open(config_file, "w", encoding="utf-8") as config_handle:
                json.dump(config_summary, config_handle, indent=2)

            print(f"Effective eval config saved: {config_file}", flush=True)

        except Exception:
            print(f"Task {task_name} failed:", flush=True)
            traceback.print_exc()
            return


def run_lm_eval(
    model_name: str,
    tasks: list[str],
    output_dir: Path,
    peft_path: Path | None = None,
    batch_size: int = 1,
    num_fewshot: int = 3,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    generation_kwargs: dict | None = None,
) -> None:
    """Evaluate the model on the selected tasks."""

    _configure_cuda()

    eval_model = _load_evaluation_model(
        model_name=model_name,
        batch_size=batch_size,
        peft_path=peft_path,
    )
    resolved_generation_kwargs = _resolve_generation_kwargs(
        generation_kwargs=generation_kwargs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )

    for task_name in tasks:
        _evaluate_task(
            task_name=task_name,
            eval_model=eval_model,
            output_dir=output_dir,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            generation_kwargs=resolved_generation_kwargs,
        )

    _configure_cuda()
    del eval_model


def run_post_finetune_eval(cfg: DictConfig, experiment_dir: Path) -> None:
    """Evaluate the fine-tuned model."""

    adapter_path = experiment_dir / "train" / "checkpoints" / "final"
    output_dir = experiment_dir / "eval" / "finetuned"

    run_lm_eval(
        model_name=cfg.model.hf_id,
        tasks=cfg.eval.tasks,
        output_dir=output_dir,
        peft_path=adapter_path,
        batch_size=cfg.eval.batch_size,
        num_fewshot=cfg.eval.num_fewshot,
        max_new_tokens=cfg.eval.max_new_tokens,
        temperature=cfg.eval.temperature,
        top_p=cfg.eval.top_p,
        do_sample=cfg.eval.do_sample,
        generation_kwargs=cfg.eval.generation_kwargs,
    )


def run_baseline_eval(cfg: DictConfig, experiment_dir: Path) -> None:
    """Evaluate the base model without fine-tuning."""

    output_dir = get_baseline_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_lm_eval(
        model_name=cfg.model.hf_id,
        tasks=cfg.eval.tasks,
        output_dir=output_dir,
        peft_path=None,
        batch_size=cfg.eval.batch_size,
        num_fewshot=cfg.eval.num_fewshot,
        max_new_tokens=cfg.eval.max_new_tokens,
        temperature=cfg.eval.temperature,
        top_p=cfg.eval.top_p,
        do_sample=cfg.eval.do_sample,
        generation_kwargs=cfg.eval.generation_kwargs,
    )
