from datasets import load_dataset, interleave_datasets
from omegaconf import DictConfig


def load_data(cfg: DictConfig):
    ds_cfg = cfg.dataset

    langs = ds_cfg.languages
    if isinstance(langs, str):
        langs = [l.strip() for l in langs.split(",") if l.strip()]

    if not langs:
        raise ValueError("dataset.languages must contain at least one language")

    sub_datasets = []

    for lang in langs:
        ds_lang = load_dataset(
            ds_cfg.hf_id,
            name="default",
            split=ds_cfg.get("split", "train"),
            streaming=True,
        )

    sub_datasets.append(ds_lang)

    ds = (
        sub_datasets[0]
        if len(sub_datasets) == 1
        else interleave_datasets(
            sub_datasets,
            probabilities=[1 / len(sub_datasets)] * len(sub_datasets),
            stopping_strategy="all_exhausted",
        )
    )

    if ds_cfg.get("size") is not None:
        ds = ds.take(ds_cfg.size)

    text_field = ds_cfg.get("text_field", "content")
    ds = ds.map(lambda x: {"text": x[text_field]})

    return ds
