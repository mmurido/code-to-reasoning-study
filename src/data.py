from datasets import load_dataset


def load_data(cfg):
    ds = load_dataset(
        cfg["dataset"]["name"],
        data_dir=cfg["dataset"]["language"],
        split="train",
        streaming=True,
    )

    ds = ds.take(cfg["dataset"]["size"])
    ds = ds.map(lambda x: {"text": x["content"]})
    return ds
