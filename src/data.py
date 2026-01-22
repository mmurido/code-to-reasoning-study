from datasets import load_dataset


def load_data(cfg):
    ds = load_dataset(
        cfg["dataset_name"], data_dir=cfg["language"], split="train", streaming=True
    )

    ds = ds.take(cfg["dataset_size"])
    ds = ds.map(lambda x: {"text": x["content"]})
    return ds
