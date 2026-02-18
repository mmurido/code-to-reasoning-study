from datasets import load_dataset, interleave_datasets, IterableDataset
from transformers import AutoTokenizer
from omegaconf import DictConfig
import math
import time

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")


def load_data(cfg: DictConfig):
    ds_cfg = cfg.dataset

    subsets = ds_cfg.subsets
    if isinstance(subsets, str):
        subsets = [s.strip() for s in subsets.split(",") if s.strip()]

    if not subsets:
        raise ValueError("dataset.subsets must contain at least one language")

    total_tokens_target = ds_cfg.get("total_tokens_target", 50_000_000)
    target_per_subset = math.ceil(total_tokens_target / len(subsets))

    print(
        f"Balancing to ~{total_tokens_target:,} total tokens "
        f"→ ~{target_per_subset:,} tokens per subset (across {len(subsets)} sources)"
    )

    sub_datasets = []

    for subset_name in subsets:
        data_dir = subset_name

        print(
            f"[{subset_name}] Loading from data_dir='{data_dir}' | target ~{target_per_subset:,} tokens"
        )

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

        ds_raw = load_dataset(**load_kwargs)
        text_field = ds_cfg.get("text_field", "content")

        def accumulate_until_target():
            current_tokens = 0
            start_time = time.time()
            for example in ds_raw:
                text = example.get(text_field, "")
                if not text:
                    continue

                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_count = len(tokens)

                if current_tokens + token_count > target_per_subset:
                    break

                current_tokens += token_count
                yield example

                if current_tokens % 500_000 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[{subset_name}] {current_tokens:,}/{target_per_subset:,} tokens "
                        f"({elapsed:.1f}s elapsed)"
                    )

            print(
                f"[{subset_name}] Reached {current_tokens:,} tokens (target was {target_per_subset:,})"
            )

        ds_lang = IterableDataset.from_generator(accumulate_until_target)
        sub_datasets.append(ds_lang)

    if len(sub_datasets) == 1:
        ds = sub_datasets[0]
    else:
        ds = interleave_datasets(
            sub_datasets,
            probabilities=[1.0 / len(sub_datasets)] * len(sub_datasets),
            stopping_strategy="all_exhausted",
        )

    text_field = ds_cfg.get("text_field", "content")
    ds = ds.map(lambda x: {"text": x[text_field]}, batched=False)

    return ds
