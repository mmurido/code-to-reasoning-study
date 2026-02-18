import sys
import json
import time
import contextlib
from datetime import datetime
from omegaconf import OmegaConf
from pathlib import Path
from src.setup.data import load_data
from src.setup.model import load_model_and_tokenizer
from src.setup.peft import build_peft
from src.setup.trainer import build_trainer
from utils.peft import save_bitfit_only
from utils.logging import (
    log_timestamp,
    log_hydra_info,
    log_hardware_info,
    log_model_info,
    log_peft_info,
    log_trainer_info,
    save_metadata,
)


def run_training(cfg, exp_dir: Path):
    start_time = time.time()
    log_dir = exp_dir / "train" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "train.log"
    metrics_file = log_dir / "metrics.json"

    print(f"Redirecting training output to: {log_file}")

    with (
        open(log_file, "a", encoding="utf-8") as log_f,
        contextlib.redirect_stdout(log_f),
        contextlib.redirect_stderr(log_f),
    ):
        log_timestamp(log_f)
        log_hydra_info(log_f, cfg)
        log_hardware_info(log_f)

        try:
            model, tokenizer = load_model_and_tokenizer(cfg)
            dataset = load_data(cfg)
            peft_cfg = build_peft(cfg, model)

            if cfg.peft.method in ["bitfit"]:
                trainer = build_trainer(cfg, model, dataset, None, exp_dir)
            else:
                trainer = build_trainer(cfg, model, dataset, peft_cfg, exp_dir)

            log_model_info(log_f, trainer.model)
            log_peft_info(log_f, peft_cfg)
            log_trainer_info(log_f, trainer)
            save_metadata(log_dir, cfg, trainer, trainer.model, peft_cfg)

            print("\nStarting trainer.train() ...", flush=True)
            trainer.train()

            runtime = time.time() - start_time
            print(f"\nTraining complete after {runtime / 3600:.2f} hours.", flush=True)

            adapter_dir = exp_dir / "train/checkpoints/final"
            adapter_dir.mkdir(parents=True, exist_ok=True)

            if cfg.peft.method == "bitfit":
                save_bitfit_only(trainer.model, str(adapter_dir))
            else:
                trainer.save_model(str(adapter_dir))

            tokenizer.save_pretrained(str(adapter_dir))

            print(f"Final model/adapter saved to: {adapter_dir}", flush=True)

            log_history = trainer.state.log_history
            metrics = {
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "runtime_seconds": runtime,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "log_history": log_history,
                "final_train_loss": log_history[-1].get("train_loss", None)
                if log_history
                else None,
            }

            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, default=str)

            print(f"Metrics saved to: {metrics_file}", flush=True)

        except Exception:
            print("Training failed!", flush=True)
            import traceback

            traceback.print_exc(file=sys.stderr)
            raise

    print(f"Training process finished. Log: {log_file} | Metrics: {metrics_file}")
