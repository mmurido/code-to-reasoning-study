#!/bin/bash

# How to use:
# ./run.sh peft_method output_dir

METHOD=$1
OUTPUT_DIR=$2

if [ -z "$METHOD" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <peft_method> <output_dir>"
  exit 1
fi

echo "Pulling latest changes from GitHub..."
git pull

if [ ! -d "finetune_env" ]; then
  echo "Creating virtual environment..."
  python3 -m venv finetune_env
fi

source finetune_env/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Starting training..."
python train.py --method "$METHOD" --adapter_dir "$OUTPUT_DIR"

echo "Training finished!"
