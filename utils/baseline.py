import os
from pathlib import Path
from omegaconf import DictConfig


def get_baseline_dir(cfg: DictConfig) -> Path:
    """Return the folder used to store baseline results."""
    return Path("baselines") / cfg.model.name


def link_existing_baseline(cfg: DictConfig, exp_dir: Path) -> None:
    """Create a link to an existing baseline inside the experiment folder."""

    src = get_baseline_dir(cfg)
    dst = exp_dir / "eval" / "baseline"

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        print(f"Symlink/dir already exists: {dst} (skipping creation)")
        return

    try:
        os.symlink(src, dst, target_is_directory=True)
        print(f"Symlink created: {dst} → {src}")
    except FileExistsError:
        print("Race condition: symlink already created")
