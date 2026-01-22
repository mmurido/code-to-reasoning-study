import os
import wandb


def init_logger(run_name):
    os.environ["WANDB_MODE"] = "offline"

    key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=key)

    wandb.init(project="thesis-peft", name=run_name)
