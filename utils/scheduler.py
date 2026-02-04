import subprocess
import itertools
from pathlib import Path
import os


def schedule_lm_eval(
    model_name: str,
    tasks: list[str],
    output_dir: Path,
    peft_path: Path | None = None,
    gpus: list[int] = [0],
    batch_size: int = 1,
    num_fewshot: int = 3,
):
    gpu_cycle = itertools.cycle(gpus)
    processes = []

    for task in tasks:
        while len(processes) >= len(gpus):
            finished = []
            for p in processes:
                if p.poll() is not None:
                    finished.append(p)
            for p in finished:
                processes.remove(p)

        gpu = next(gpu_cycle)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        cmd = [
            "python3",
            "-m",
            "src.evaluate_worker",
            "--model_name",
            model_name,
            "--task",
            task,
            "--output_dir",
            str(output_dir),
            "--batch_size",
            str(batch_size),
            "--num_fewshot",
            str(num_fewshot),
        ]

        if peft_path is not None:
            cmd += ["--peft_path", str(peft_path)]

        print(f"Launching {task} on GPU {gpu}: {' '.join(cmd)}")
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    for p in processes:
        p.wait()
