from pathlib import Path
from src.setup.data import load_data
from src.setup.model import load_model_and_tokenizer
from src.setup.peft import build_peft
from src.setup.trainer import build_trainer


def run_training(cfg, exp_dir: Path):
    model, tokenizer = load_model_and_tokenizer(cfg)
    print("Model and tokenizer were loaded")
    dataset = load_data(cfg)
    print("Dataset was loaded")
    peft_cfg = build_peft(cfg)
    print("Peft was built")

    trainer = build_trainer(cfg, model, dataset, peft_cfg, exp_dir)
    print("Trainer was built")
    trainer.train()

    adapter_dir = exp_dir / "train/checkpoints/final"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"Training complete. Model saved to {adapter_dir}")
