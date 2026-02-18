from pathlib import Path
from omegaconf import DictConfig, ListConfig, open_dict
from datetime import datetime


def create_run_id(cfg: DictConfig) -> str:
    model = cfg.model.name
    peft = cfg.peft.method
    dataset = cfg.dataset.name
    subsets = cfg.dataset.get("subsets", None)

    multiple_subsets = False
    if isinstance(subsets, (list, ListConfig)) and len(subsets) > 1:
        multiple_subsets = True
    elif isinstance(subsets, str) and "," in subsets:
        multiple_subsets = True

    if dataset == "starcoderdata" and multiple_subsets:
        subset_tag = "multi"
    else:
        subset_tag = subsets

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_id = f"{model}__{peft}__{dataset}-{subset_tag}__{timestamp}"

    with open_dict(cfg):
        cfg.run_id = run_id


def create_experiment_dir(cfg: DictConfig) -> Path:
    exp_dir = Path(cfg.output_dir) / cfg.run_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "train" / "checkpoints" / "final").mkdir(parents=True, exist_ok=True)
    (exp_dir / "train" / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "eval").mkdir(parents=True, exist_ok=True)

    print(f"Experiment directory created: {exp_dir}")
    return exp_dir
