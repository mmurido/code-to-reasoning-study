import contextlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from src.setup.data import load_data
from src.setup.model import load_model_and_tokenizer
from src.setup.peft import build_peft
from src.setup.trainer import build_trainer
from utils.logging import (
    log_hardware_info,
    log_hydra_info,
    log_model_info,
    log_peft_info,
    log_timestamp,
    log_trainer_info,
    save_metadata,
)


def _create_training_metrics(
    cfg, trainer, start_time: float, runtime_seconds: float
) -> dict:
    """Collect training metrics."""

    training_logs = trainer.state.log_history

    return {
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "runtime_seconds": runtime_seconds,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "log_history": training_logs,
        "final_train_loss": (
            training_logs[-1].get("train_loss") if training_logs else None
        ),
    }


def _log_training_context(log_dir: Path, log_handle, cfg, trainer, peft_config) -> None:
    """Save and log details of the training setup."""

    log_timestamp(log_handle)
    log_hydra_info(log_handle, cfg)
    log_hardware_info(log_handle)
    log_model_info(log_handle, trainer.model)
    log_peft_info(log_handle, peft_config)
    log_trainer_info(log_handle, trainer)
    save_metadata(log_dir, cfg, trainer, trainer.model, peft_config)


def _save_outputs(tokenizer, trainer, experiment_dir: Path) -> Path:
    """Save training outputs."""

    final_checkpoint_dir = experiment_dir / "train" / "checkpoints" / "final"
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_checkpoint_dir))
    tokenizer.save_pretrained(str(final_checkpoint_dir))

    return final_checkpoint_dir


def run_training(cfg, experiment_dir: Path) -> None:
    """Train the model and save logs, checkpoints, and metrics."""

    start_time = time.time()

    log_dir = experiment_dir / "train" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "train.log"
    metrics_file = log_dir / "metrics.json"

    print(f"Redirecting training output to: {log_file}")

    with (
        open(log_file, "a", encoding="utf-8") as log_handle,
        contextlib.redirect_stdout(log_handle),
        contextlib.redirect_stderr(log_handle),
    ):
        try:
            model, tokenizer = load_model_and_tokenizer(cfg)
            dataset = load_data(cfg)
            peft_config = build_peft(cfg)
            trainer = build_trainer(cfg, model, dataset, peft_config, experiment_dir)

            _log_training_context(log_dir, log_handle, cfg, trainer, peft_config)

            print("\nStarting trainer.train() ...", flush=True)
            trainer.train()

            runtime_seconds = time.time() - start_time
            print(
                f"\nTraining complete after {runtime_seconds / 3600:.2f} hours.",
                flush=True,
            )

            final_checkpoint_dir = _save_outputs(
                tokenizer=tokenizer,
                trainer=trainer,
                experiment_dir=experiment_dir,
            )
            print(f"Final model/adapter saved to: {final_checkpoint_dir}", flush=True)

            metrics = _create_training_metrics(
                cfg=cfg,
                trainer=trainer,
                start_time=start_time,
                runtime_seconds=runtime_seconds,
            )
            with open(metrics_file, "w", encoding="utf-8") as metrics_handle:
                json.dump(metrics, metrics_handle, indent=2, default=str)

            print(f"Metrics saved to: {metrics_file}", flush=True)

        except Exception:
            print("Training failed!", flush=True)
            import traceback

            traceback.print_exc(file=sys.stderr)
            raise

    print(f"Training process finished. Log: {log_file} | Metrics: {metrics_file}")
