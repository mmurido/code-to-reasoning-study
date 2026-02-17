from typing import Any
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PrefixTuningConfig, IA3Config


def build_lora(cfg: DictConfig) -> LoraConfig:
    peft_cfg = cfg.peft

    target_modules = peft_cfg.get("target_modules", None)
    if target_modules is not None:
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


def build_bitfit(cfg: DictConfig) -> dict:
    peft_cfg = cfg.peft
    return {
        "method": "bitfit",
        "bias": peft_cfg.get("bias", "all"),
        "task_type": "CAUSAL_LM",
    }


def build_ia3(cfg: DictConfig) -> IA3Config:
    peft_cfg = cfg.peft

    target_modules = peft_cfg.get("target_modules", None)
    if target_modules is not None:
        target_modules = OmegaConf.to_container(target_modules, resolve=True)

    ff_modules = peft_cfg.get("feedforward_modules", None)
    if ff_modules is not None:
        ff_modules = OmegaConf.to_container(ff_modules, resolve=True)

    return IA3Config(
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        feedforward_modules=ff_modules,
        fan_in_fan_out=peft_cfg.get("fan_in_fan_out", False),
    )


PEFT_BUILDERS = {
    "lora": build_lora,
    "prefix_tuning": build_prefix_tuning,
    "bitfit": build_bitfit,
    "ia3": build_ia3,
}


def build_peft(cfg: DictConfig) -> Any:
    method = cfg.peft.method
    if method not in PEFT_BUILDERS:
        raise ValueError(f"Unknown PEFT method: {method}")
    builder = PEFT_BUILDERS[method]
    return builder(cfg)
