import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from argparse import Namespace


def resolve_config(args: Namespace) -> DictConfig:
    with hydra.initialize(version_base=None, config_path="configs"):
        cfg = hydra.compose(config_name=f"experiment/{args.config}")

    OmegaConf.set_struct(cfg, False)
    cfg.run.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return cfg
