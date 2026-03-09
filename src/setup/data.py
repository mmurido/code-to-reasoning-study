import math
import time
from datasets import IterableDataset, interleave_datasets, load_dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")
# Pythia models share the same tokenizer, so this tokenizer is compatible across Pythia sizes.


def _parse_subsets(subsets) -> list[str]:
    """Read the subset names from the config."""

    if isinstance(subsets, str):
        subsets = [s.strip() for s in subsets.split(",") if s.strip()]
    return subsets


def _build_load_kwargs(ds_cfg: DictConfig, subset_name: str) -> dict:
    """Prepare settings for dataset streaming."""

    load_kwargs = {
        "path": ds_cfg.hf_id,
        "split": ds_cfg.get("split", "train"),
        "streaming": True,
        "trust_remote_code": True,
    }

    if ds_cfg.hf_id == "bigcode/starcoderdata":
        load_kwargs["data_dir"] = subset_name
    elif "fineweb" in ds_cfg.hf_id.lower():
        load_kwargs["name"] = subset_name
    else:
        raise ValueError(f"Don't know how to load subsets for {ds_cfg.hf_id}")

    return load_kwargs


def _build_subset_dataset(
    ds_raw,
    subset_name: str,
    text_field: str,
    target_tokens: int,
) -> IterableDataset:
    """Stream subset until the token target is reached."""

    def accumulate_until_target():
        current_tokens = 0
        start_time = time.time()

        for example in ds_raw:
            text = example.get(text_field, "")
            if not text:
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_count = len(tokens)

            if current_tokens + token_count > target_tokens:
                break

            current_tokens += token_count
            yield {"text": text}

            if current_tokens % 500_000 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[{subset_name}] {current_tokens:,}/{target_tokens:,} tokens "
                    f"({elapsed:.1f}s elapsed)"
                )

        print(
            f"[{subset_name}] Reached {current_tokens:,} tokens "
            f"(target was {target_tokens:,})"
        )

    return IterableDataset.from_generator(accumulate_until_target)


def load_data(cfg: DictConfig):
    """Build a streaming dataset by sampling each subset up to a token target."""

    ds_cfg = cfg.dataset

    subsets = _parse_subsets(ds_cfg.subsets)
    if not subsets:
        raise ValueError("dataset.subsets must contain at least one language")

    total_tokens_target = ds_cfg.get("total_tokens_target", 50_000_000)
    target_per_subset = math.ceil(total_tokens_target / len(subsets))
    text_field = ds_cfg.get("text_field", "content")

    print(
        f"Balancing to ~{total_tokens_target:,} total tokens "
        f"→ ~{target_per_subset:,} tokens per subset (across {len(subsets)} sources)"
    )

    subset_datasets = []

    for subset_name in subsets:
        print(f"[{subset_name}] Loading subset | target ~{target_per_subset:,} tokens")

        load_kwargs = _build_load_kwargs(ds_cfg, subset_name)
        ds_raw = load_dataset(**load_kwargs)

        ds_subset = _build_subset_dataset(
            ds_raw=ds_raw,
            subset_name=subset_name,
            text_field=text_field,
            target_tokens=target_per_subset,
        )
        subset_datasets.append(ds_subset)

    if len(subset_datasets) == 1:
        ds = subset_datasets[0]
    else:
        ds = interleave_datasets(
            subset_datasets,
            probabilities=[1.0 / len(subset_datasets)] * len(subset_datasets),
            stopping_strategy="all_exhausted",
        )

    return ds
