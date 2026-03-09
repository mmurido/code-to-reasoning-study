#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig
from utils.experiment_setup import set_run_id, create_experiment_dir
from src.train import run_training
from src.evaluate import run_baseline_eval, run_post_finetune_eval
from utils.baseline import link_existing_baseline
from utils.huggingface import login_huggingface
from utils.wandb import init_wandb
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Run training and evaluation for one experiment."""

    login_huggingface()
    experiment_dir = None

    if cfg.do_train:
        set_run_id(cfg)
        experiment_dir = create_experiment_dir(cfg)

        wandb_dir = str(experiment_dir / "train" / "logs")
        init_wandb(cfg.run_id, wandb_dir)

        run_training(cfg, experiment_dir)

    if cfg.do_eval:
        existing_experiment = cfg.get("existing_exp")

        if not cfg.do_train and not existing_experiment:
            raise ValueError("do_eval without do_train requires existing_exp")

        experiment_dir = (
            Path(existing_experiment) if existing_experiment else experiment_dir
        )

        run_baseline_eval(cfg, experiment_dir)
        link_existing_baseline(cfg, experiment_dir)
        run_post_finetune_eval(cfg, experiment_dir)


if __name__ == "__main__":
    main()
