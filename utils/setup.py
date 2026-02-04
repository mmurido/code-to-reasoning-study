import sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from argparse import Namespace
from datetime import datetime


def save_config(cfg: DictConfig, exp_dir: Path) -> None:
    config_path = exp_dir / "config.yaml"
    config_path.write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )


def save_command(exp_dir: Path) -> None:
    cmd = " ".join(sys.argv)
    (exp_dir / "cmd.txt").write_text(cmd + "\n", encoding="utf-8")


def create_experiment_dir(cfg: DictConfig, args: Namespace) -> Path:
    exp_dir = Path(args.output_dir) / f"{cfg.run.id}__{cfg.run.timestamp}"
    print(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "train" / "checkpoints" / "final").mkdir(parents=True, exist_ok=True)
    (exp_dir / "train" / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "eval").mkdir(parents=True, exist_ok=True)
    (exp_dir / "analysis").mkdir(exist_ok=True)

    save_config(cfg, exp_dir)
    save_command(exp_dir)

    return exp_dir


def update_experiment_config_with_overrides(
    exp_dir: Path,
    current_cfg: DictConfig,
) -> DictConfig:
    config_path = exp_dir / "config.yaml"

    if not config_path.is_file():
        save_config(current_cfg, exp_dir)
        return current_cfg

    saved_config = OmegaConf.load(config_path)
    merged = OmegaConf.merge(saved_config, current_cfg)

    OmegaConf.update(merged, "meta.config_updated_at", datetime.now().isoformat())
    OmegaConf.update(merged, "meta.source", "CLI overrides during eval")

    save_config(merged, exp_dir)
    return merged
