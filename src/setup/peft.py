from typing import Any
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PrefixTuningConfig


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


def build_prefix_tuning(cfg: DictConfig) -> PrefixTuningConfig:
    peft_cfg = cfg.peft

    num_virtual_tokens = peft_cfg.get("num_virtual_tokens", 20)
    token_dim = peft_cfg.get("token_dim", None)
    encoder_hidden_size = peft_cfg.get("encoder_hidden_size", None)
    prefix_projection = peft_cfg.get("prefix_projection", True)
    inference_mode = peft_cfg.get("inference_mode", False)

    return PrefixTuningConfig(
        task_type="CAUSAL_LM",
        inference_mode=inference_mode,
        num_virtual_tokens=num_virtual_tokens,
        token_dim=token_dim,
        encoder_hidden_size=encoder_hidden_size,
        prefix_projection=prefix_projection,
    )


PEFT_BUILDERS = {
    "lora": build_lora,
    "prefix_tuning": build_prefix_tuning,
}


def build_peft(cfg: DictConfig) -> Any:
    method = cfg.peft.method
    if method not in PEFT_BUILDERS:
        raise ValueError(f"Unknown PEFT method: {method}")
    builder = PEFT_BUILDERS[method]
    return builder(cfg)
