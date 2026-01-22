import os
import huggingface_hub


def login_hf():
    token = os.environ.get("HF_TOKEN")
    huggingface_hub.login(token=token)
