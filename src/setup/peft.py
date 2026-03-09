from typing import Any
from omegaconf import DictConfig, OmegaConf
from peft import IA3Config, LoraConfig, PrefixTuningConfig, VeraConfig


def _to_python(value):
    """Convert OmegaConf containers to plain Python objects."""
    if value is None:
        return None
    return OmegaConf.to_container(value, resolve=True)


def build_ia3(cfg: DictConfig) -> IA3Config:
    """Build an IA3 configuration from the experiment config."""

    peft_cfg = cfg.peft

    return IA3Config(
        task_type="CAUSAL_LM",
        target_modules=_to_python(peft_cfg.get("target_modules", None)),
        feedforward_modules=_to_python(peft_cfg.get("feedforward_modules", None)),
        fan_in_fan_out=peft_cfg.get("fan_in_fan_out", False),
    )


def build_vera(cfg: DictConfig) -> VeraConfig:
    """Build a VeRA configuration from the experiment config."""

    peft_cfg = cfg.peft

    return VeraConfig(
        task_type="CAUSAL_LM",
        r=peft_cfg.get("r", 256),
        vera_dropout=peft_cfg.get("vera_dropout", 0.0),
        d_initial=peft_cfg.get("d_initial", 0.1),
        target_modules=_to_python(peft_cfg.get("target_modules", None)),
        layers_to_transform=_to_python(peft_cfg.get("layers_to_transform", None)),
    )


def build_prefix_tuning(cfg: DictConfig) -> PrefixTuningConfig:
    """Build a prefix-tuning configuration from the experiment config."""

    peft_cfg = cfg.peft

    return PrefixTuningConfig(
        task_type="CAUSAL_LM",
        inference_mode=peft_cfg.get("inference_mode", False),
        num_virtual_tokens=peft_cfg.get("num_virtual_tokens", 20),
        token_dim=peft_cfg.get("token_dim", None),
        encoder_hidden_size=peft_cfg.get("encoder_hidden_size", None),
        prefix_projection=peft_cfg.get("prefix_projection", True),
    )


def build_lora(cfg: DictConfig) -> LoraConfig:
    """Build a LoRA configuration from the experiment config."""

    peft_cfg = cfg.peft

    return LoraConfig(
        r=peft_cfg.r,
        lora_alpha=peft_cfg.alpha,
        lora_dropout=peft_cfg.dropout,
        bias=peft_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=_to_python(peft_cfg.get("target_modules", None)),
    )


PEFT_BUILDERS = {
    "ia3": build_ia3,
    "vera": build_vera,
    "prefix_tuning": build_prefix_tuning,
    "lora": build_lora,
}


def build_peft(cfg: DictConfig) -> Any:
    """Build the PEFT configuration selected in the experiment config."""

    method = cfg.peft.method
    if method not in PEFT_BUILDERS:
        raise ValueError(f"Unknown PEFT method: {method}")

    return PEFT_BUILDERS[method](cfg)
