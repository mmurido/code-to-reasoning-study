# Code to Reasoning Study

This repository contains the experimental pipeline developed for my bachelor's thesis at Università degli Studi di Napoli Federico II, titled *Impact of Unsupervised Code Fine-Tuning on LLMs' Reasoning*.

The project investigates whether unsupervised fine-tuning on source code can improve reasoning performance on tasks outside programming. In the thesis, this effect is referred to as the **Code-Induced Transfer Effect (CITE)**. The experiments compare code fine-tuning against a matched natural-language baseline across different Pythia model sizes, PEFT methods, and reasoning benchmarks.

## What this repository contains

This repository includes the full experimental pipeline:

- configs used for the experiments
- setup of model, tokenizer, dataset, and PEFT
- fine-tuning
- evaluation
- logging

The full thesis is included in the repository for methodological details and results, while full experiment outputs are not included.

## Running an experiment

Experiments are launched with Hydra overrides from the command line.

Run a full experiment:

```
python3 run_experiment.py \
  model=pythia-1b \
  peft=lora-pythia \
  dataset=starcoderdata-python
```

Run evaluation only, for example to resume it:

```
python3 run_experiment.py \
  model=pythia-1b \
  peft=lora-pythia \
  dataset=starcoderdata-python \
  do_train=false \
  existing_exp=/path/to/existing/experiment
```

Override specific settings for exploratory runs:

```
python3 run_experiment.py \
  model=pythia-1b \
  peft=lora-pythia \
  dataset=starcoderdata-python \
  training.max_steps=500
```
