from peft import (
    LoraConfig,
    PrefixTuningConfig,
)


def build_lora(cfg):
    l = cfg["lora"]
    return LoraConfig(
        r=l["r"],
        lora_alpha=l["alpha"],
        lora_dropout=l["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=l["target_modules"],
    )


def build_prefix(cfg, model):
    p = cfg["prefix"]
    return PrefixTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=p["num_virtual_tokens"],
        encoder_hidden_size=model.config.hidden_size,
        prefix_projection=p["prefix_projection"],
    )


PEFT_BUILDERS = {
    "lora": build_lora,
    "prefix": build_prefix,
}


def build_peft(cfg, model):
    method = cfg["method"]

    if method not in PEFT_BUILDERS:
        raise ValueError(f"Unknown PEFT method: {method}")

    builder = PEFT_BUILDERS[method]

    # Some need model, some don't
    try:
        return builder(cfg, model)
    except TypeError:
        return builder(cfg)
