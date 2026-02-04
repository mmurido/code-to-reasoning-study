from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_peft_adapter(
    base_model_id: str,
    adapter_path: Path,
    save_dir: Path,
    device_map: str = "auto",
    torch_dtype: str = "auto",
) -> None:
    """Merge LoRA adapter into base model and save."""
    print(f"Merging PEFT adapter {adapter_path} into {base_model_id} → {save_dir}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map=device_map,
    )
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(
        save_dir,
        safe_serialization=True,
        max_shard_size="10GB",
    )
    tokenizer.save_pretrained(save_dir)
    print("Merge complete.")
