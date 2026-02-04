#!/usr/bin/env python3
from args import parse_args
from config import resolve_config
from utils.setup import create_experiment_dir, update_experiment_config_with_overrides
from src.train import run_training
from src.evaluate import run_baseline_eval, run_post_finetune_eval
from utils.baseline import link_existing_baseline
from utils.hf import login_hf
from utils.wandb import init_wandb
from utils.logger import setup_phase_logger
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def main():
    args = parse_args()
    cfg = resolve_config(args)
    login_hf()

    exp_dir = None

    if args.do_train:
        exp_dir = create_experiment_dir(cfg, args)
        print("Experiment directory created")

        wandb_dir = str(exp_dir / "train/logs/")
        init_wandb(cfg.run.id, wandb_dir)
        print("Wandb initialized")

        train_logger = setup_phase_logger(exp_dir / "train/logs/train.log")
        train_logger.info("Starting training")
        run_training(cfg, exp_dir)
        train_logger.info("Training finished")

    if args.do_eval:
        if not args.do_train:
            if not args.existing_exp:
                raise ValueError("--do_eval without --do_train requires --existing_exp")
            exp_dir = Path(args.existing_exp)

        cfg = update_experiment_config_with_overrides(exp_dir, cfg)
        run_baseline_eval(cfg, exp_dir)
        link_existing_baseline(cfg, exp_dir)
        run_post_finetune_eval(cfg, exp_dir)

    # if args.do_analyze:
    #     analyze_results(cfg, exp_dir)


if __name__ == "__main__":
    main()
