import os
import json
import time
import torch
import traceback
import contextlib
from utils.baseline import baseline_dir
from utils.logging import log_eval_config
from utils.json import json_safe
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator
from pathlib import Path
from omegaconf import DictConfig


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
    peft_method: str | None = None,
):

    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    model_args = {
        "pretrained": model_name,
        "batch_size": batch_size,
        "parallelize": True,
    }

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    if peft_path is not None:
        model = PeftModel.from_pretrained(
            base_model,
            peft_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model.eval()
        lm = HFLM(
            pretrained=model,
            tokenizer=AutoTokenizer.from_pretrained(model_name),
        )
    else:
        lm = HFLM(**model_args)

    gen_kwargs = generation_kwargs or {
        "num_beams": 1,
        "early_stopping": True,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
    }

    for task in tasks:
        task_dir = output_dir / task
        results_file = task_dir / "results.json"
        config_file = task_dir / "config.json"

        if results_file.exists():
            print(f"Skipping existing eval for {task}")
            continue

        task_dir.mkdir(parents=True, exist_ok=True)
        log_file = task_dir / "eval.log"

        task_start = time.time()

        with (
            open(log_file, "w") as f,
            contextlib.redirect_stdout(f),
            contextlib.redirect_stderr(f),
        ):
            config_summary = log_eval_config(
                log_f=f,
                task=task,
                lm=lm,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                start_time=task_start,
            )

            try:
                print(f"Running evaluation for {task}...", flush=True)
                results = evaluator.simple_evaluate(
                    model=lm,
                    tasks=[task],
                    num_fewshot=num_fewshot,
                    batch_size=batch_size,
                    log_samples=True,
                    gen_kwargs=gen_kwargs,
                )

                tmp_file = results_file.with_suffix(".json.tmp")
                with open(tmp_file, "w", encoding="utf-8") as rf:
                    json.dump(json_safe(results), rf, indent=2)

                tmp_file.replace(results_file)
                print(f"Results saved to: {results_file}", flush=True)

                runtime = time.time() - task_start
                print(
                    f"Task {task} completed in {runtime / 60:.1f} minutes.", flush=True
                )

                with open(config_file, "w", encoding="utf-8") as cf:
                    json.dump(config_summary, cf, indent=2)

                print(f"Effective eval config saved: {config_file}", flush=True)

            except Exception:
                print(f"Task {task} failed:", flush=True)
                traceback.print_exc()
                continue

    torch.cuda.empty_cache()
    del lm


def run_post_finetune_eval(cfg: DictConfig, exp_dir: Path) -> None:
    adapter_dir = exp_dir / "train" / "checkpoints" / "final"
    out_dir = exp_dir / "eval" / "finetuned"

    run_lm_eval(
        model_name=cfg.model.hf_id,
        tasks=cfg.eval.tasks,
        output_dir=out_dir,
        peft_path=adapter_dir,
        batch_size=cfg.eval.batch_size,
        num_fewshot=cfg.eval.num_fewshot,
        max_new_tokens=cfg.eval.max_new_tokens,
        temperature=cfg.eval.temperature,
        top_p=cfg.eval.top_p,
        do_sample=cfg.eval.do_sample,
        generation_kwargs=cfg.eval.generation_kwargs,
        peft_method=cfg.peft.method,
    )


def run_baseline_eval(cfg: DictConfig, exp_dir: Path) -> None:
    out_dir = baseline_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_lm_eval(
        model_name=cfg.model.hf_id,
        tasks=cfg.eval.tasks,
        output_dir=out_dir,
        peft_path=None,
        batch_size=cfg.eval.batch_size,
        num_fewshot=cfg.eval.num_fewshot,
        max_new_tokens=cfg.eval.max_new_tokens,
        temperature=cfg.eval.temperature,
        top_p=cfg.eval.top_p,
        do_sample=cfg.eval.do_sample,
        generation_kwargs=cfg.eval.generation_kwargs,
        peft_method=None,
    )
