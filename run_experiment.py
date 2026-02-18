#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig
from utils.setup import create_run_id, create_experiment_dir
from src.train import run_training
from src.evaluate import run_baseline_eval, run_post_finetune_eval
from utils.baseline import link_existing_baseline
from utils.hf import login_hf
from utils.wandb import init_wandb
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    login_hf()
    exp_dir = None

    if cfg.do_train:
        create_run_id(cfg)
        exp_dir = create_experiment_dir(cfg)

        wandb_dir = str(exp_dir / "train/logs/")
        init_wandb(cfg.run_id, wandb_dir)

        run_training(cfg, exp_dir)

    if cfg.do_eval:
        if not cfg.do_train and not cfg.get("existing_exp"):
            raise ValueError("do_eval without do_train requires existing_exp")

        exp_dir = Path(cfg.existing_exp) if cfg.get("existing_exp") else exp_dir

        run_baseline_eval(cfg, exp_dir)
        link_existing_baseline(cfg, exp_dir)
        run_post_finetune_eval(cfg, exp_dir)


if __name__ == "__main__":
    main()
