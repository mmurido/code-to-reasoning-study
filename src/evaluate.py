import json
import contextlib
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from pathlib import Path
from omegaconf import DictConfig
from utils.baseline import baseline_dir
from utils.scheduler import schedule_lm_eval


def run_lm_eval(
    model_name: str,
    task: str,
    output_dir: Path,
    peft_path: Path | None = None,
    batch_size: int = 1,
    num_fewshot: int = 3,
):
    model_args = {
        "pretrained": model_name,
        "batch_size": batch_size,
        "device": "cuda",
    }

    if peft_path is not None:
        model_args["peft"] = str(peft_path)

    lm = HFLM(**model_args)

    task_dir = output_dir / task
    results_file = task_dir / "results.json"

    if results_file.exists():
        print(f"Skipping existing eval for {task}")
        return

    task_dir.mkdir(parents=True, exist_ok=True)
    log_file = task_dir / "eval.log"

    with (
        open(log_file, "w") as f,
        contextlib.redirect_stdout(f),
        contextlib.redirect_stderr(f),
    ):
        print(f"Starting task {task}")

        results = evaluator.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            log_samples=True,
        )

    with open(task_dir / f"{task}.json", "w") as f:
        json.dump(results, f, indent=2)


def run_post_finetune_eval(cfg: DictConfig, exp_dir: Path) -> None:
    adapter_dir = exp_dir / "train" / "checkpoints" / "final"
    out_dir = exp_dir / "eval" / "finetuned"

    schedule_lm_eval(
        model_name=cfg.model.hf_id,
        tasks=cfg.eval.tasks,
        output_dir=out_dir,
        peft_path=adapter_dir,
        gpus=[0, 1],
        batch_size=cfg.eval.batch_size,
        num_fewshot=cfg.eval.num_fewshot,
    )


def run_baseline_eval(cfg: DictConfig, exp_dir: Path) -> None:
    out_dir = baseline_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    schedule_lm_eval(
        model_name=cfg.model.hf_id,
        tasks=cfg.eval.tasks,
        output_dir=out_dir,
        peft_path=None,
        gpus=[0, 1],
        batch_size=cfg.eval.batch_size,
        num_fewshot=cfg.eval.num_fewshot,
    )
