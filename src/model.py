import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], torch_dtype=torch.float16, device_map="auto"
    )

    model.config.use_cache = False
    return model, tokenizer
