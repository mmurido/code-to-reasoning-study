import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="experiments")

    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--existing_exp", type=str, default=None)

    parser.add_argument("--do_train", action="store_true")

    parser.add_argument("--do_eval", action="store_true")

    parser.add_argument("--do_analyze", action="store_true")

    return parser.parse_args()
