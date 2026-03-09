import os
from huggingface_hub import login


def login_huggingface():
    token = os.environ.get("HF_TOKEN")
    login(token=token)
