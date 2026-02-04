import os
import wandb


def init_wandb(run_name, output_dir):
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = output_dir
    key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=key)
    wandb.init(project="thesis-peft", name=run_name)
