from pathlib import Path
from omegaconf import DictConfig
from argparse import Namespace


def create_experiment_dir(cfg: DictConfig, args: Namespace) -> Path:
    exp_dir = Path(args.output_dir) / f"{cfg.run.id}__{cfg.run.timestamp}"
    print(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "train" / "checkpoints" / "final").mkdir(parents=True, exist_ok=True)
    (exp_dir / "train" / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "eval").mkdir(parents=True, exist_ok=True)

    print(f"Experiment directory created: {exp_dir}")
    return exp_dir
