from datasets import load_dataset, interleave_datasets, IterableDataset
from transformers import AutoTokenizer
from omegaconf import DictConfig
import math
import time

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")


def load_data(cfg: DictConfig):
    ds_cfg = cfg.dataset

    langs = ds_cfg.languages
    if isinstance(langs, str):
        langs = [l.strip() for l in langs.split(",") if l.strip()]

    if not langs:
        raise ValueError("dataset.languages must contain at least one language")

    total_tokens_target = ds_cfg.get("total_tokens_target", 50_000_000)
    target_per_lang = math.ceil(total_tokens_target / len(langs))

    print(
        f"Balancing to ~{total_tokens_target:,} total tokens "
        f"→ ~{target_per_lang:,} tokens per language (across {len(langs)} langs)"
    )

    sub_datasets = []
    lang_dir_map = {"c++": "cpp"}

    for lang in langs:
        lang_key = lang.lower()
        lang_dir = lang_dir_map.get(lang_key, lang_key.replace(" ", "-"))

        print(
            f"[{lang}] Loading from data_dir='{lang_dir}' | target ~{target_per_lang:,} tokens"
        )

        ds_lang_raw = load_dataset(
            ds_cfg.hf_id,
            data_dir=lang_dir,
            split=ds_cfg.get("split", "train"),
            streaming=True,
        )

        def accumulate_until_target():
            current_tokens = 0
            start_time = time.time()
            for example in ds_lang_raw:
                text = example.get(ds_cfg.get("text_field", "content"), "")
                if not text:
                    continue
                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_count = len(tokens)
                if current_tokens + token_count > target_per_lang:
                    break
                current_tokens += token_count
                yield example

                if current_tokens % 500_000 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[{lang}] {current_tokens:,}/{target_per_lang:,} tokens "
                        f"({elapsed:.1f}s elapsed)"
                    )

            print(
                f"[{lang}] Reached {current_tokens:,} tokens (target was {target_per_lang:,})"
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
