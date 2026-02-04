import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--peft_path", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_fewshot", type=int, default=3)
    args = parser.parse_args()

    from src.evaluate import run_lm_eval

    run_lm_eval(
        model_name=args.model_name,
        task=args.task,
        output_dir=Path(args.output_dir),
        peft_path=Path(args.peft_path) if args.peft_path else None,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
    )


if __name__ == "__main__":
    main()
