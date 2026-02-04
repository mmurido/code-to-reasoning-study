from typing import Any
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig


def build_lora(cfg: DictConfig) -> LoraConfig:
    peft_cfg = cfg.peft

    target_modules = peft_cfg.target_modules
    if hasattr(target_modules, "__dict__"):
        target_modules = OmegaConf.to_container(target_modules, resolve=True)

    return LoraConfig(
        r=peft_cfg.r,
        lora_alpha=peft_cfg.alpha,
        lora_dropout=peft_cfg.dropout,
        bias=peft_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


PEFT_BUILDERS = {
    "lora": build_lora,
}


def build_peft(cfg: DictConfig) -> Any:
    method = cfg.peft.method
    if method not in PEFT_BUILDERS:
        raise ValueError(f"Unknown PEFT method: {method}")
    builder = PEFT_BUILDERS[method]
    return builder(cfg)
