from dotenv import load_dotenv

load_dotenv()

import argparse
import yaml

from src.data import load_data
from src.model import load_model
from src.trainer import build_trainer
from src.auth import login_hf
from src.peft import build_peft
from src.app_logger import init_app_logger
from src.experiment_logger import init_experiment_logger


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(common, method):
    merged = common.copy()
    merged.update(method)
    return merged


def main():
    # can run like this: python train.py --method lora
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["lora", "prefix", "bitfit", "ia3", "adapter"],
        help="PEFT method to use",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory where adapters will be saved"
    )
    args = parser.parse_args()
    output_dir = args.output_dir

    common_cfg = load_yaml("configs/common.yaml")
    method_cfg = load_yaml(f"configs/{args.method}.yaml")

    cfg = merge_configs(common_cfg, method_cfg)

    run_name = f"{cfg['model_name'].split('/')[-1]}-{cfg['language']}-{args.method}"
    logger = init_app_logger("train")

    logger.info("Selected PEFT method: %s", args.method)
    logger.info("Run name: %s", run_name)
    init_experiment_logger(run_name)

    logger.info("Logging into Hugging Face")
    login_hf()

    logger.info(
        "Loading dataset: %s (%s)", cfg["dataset"]["name"], cfg["dataset"]["language"]
    )
    dataset = load_data(cfg)

    logger.info("Loading model: %s", cfg["model"]["name"])
    model, tokenizer = load_model(cfg)

    logger.info("Building PEFT configuration")
    peft_cfg = build_peft(cfg, model)

    logger.info("Initializing Trainer")
    trainer = build_trainer(model, dataset, peft_cfg, cfg)

    logger.info("Starting training")
    trainer.train()

    logger.info("Training completed")

    logger.info("Saving adapters to %s", output_dir)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
