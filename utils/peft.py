import torch
from pathlib import Path
from transformers import AutoModelForCausalLM


def prepare_model_for_bitfit(model, bias="all"):
    # Freeze everything
    model.requires_grad_(False)

    # Unfreeze biases
    if bias == "all":
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad_(True)

    return model


def save_bitfit_only(model, save_directory):
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad and "bias" in name:
            state_dict[name] = param.data.detach().cpu()

    torch.save(state_dict, save_directory / "bitfit_biases.pt")
    print(
        f"Saved BitFit biases only: {save_directory / 'bitfit_biases.pt'} "
        f"({len(state_dict)} bias tensors)"
    )


def load_bitfit_only(
    model: AutoModelForCausalLM, bitfit_dir: Path
) -> AutoModelForCausalLM:
    biases_path = bitfit_dir / "bitfit_biases.pt"
    if not biases_path.exists():
        raise FileNotFoundError(f"BitFit biases not found at {biases_path}")

    state_dict = torch.load(biases_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Warning: {len(missing)} missing keys when loading BitFit biases")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys when loading BitFit biases")

    print(f"Successfully loaded {len(state_dict)} BitFit bias parameters")
    print("Sample bias loaded:", list(state_dict.keys())[:3])
    return model
