#!/usr/bin/env python3
from args import parse_args
from config import resolve_config
from utils.setup import create_experiment_dir
from src.train import run_training
from src.evaluate import run_baseline_eval, run_post_finetune_eval
from utils.baseline import link_existing_baseline
from utils.hf import login_hf
from utils.wandb import init_wandb
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

        wandb_dir = str(exp_dir / "train/logs/")
        init_wandb(cfg.run.id, wandb_dir)
        run_training(cfg, exp_dir)

    if args.do_eval:
        if not args.do_train:
            if not args.existing_exp:
                raise ValueError("--do_eval without --do_train requires --existing_exp")
            exp_dir = Path(args.existing_exp)

        run_baseline_eval(cfg, exp_dir)
        link_existing_baseline(cfg, exp_dir)
        run_post_finetune_eval(cfg, exp_dir)


if __name__ == "__main__":
    main()
