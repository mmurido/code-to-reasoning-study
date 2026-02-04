import torch
from typing import Tuple
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(
    cfg: DictConfig,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    model_cfg = cfg.model

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.config.use_cache = False
    return model, tokenizer
