#!/bin/bash

# How to use:
# ./run.sh peft_method output_dir

METHOD=$1
OUTPUT_DIR=$2
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"

if [ -z "$METHOD" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <peft_method> <output_dir>"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Pulling latest changes from GitHub..."
git pull

echo "Installing dependencies..."
python3 -m pip install --upgrade --user pip
python3 -m pip install --user -r requirements.txt

echo "Starting training... Logs will be saved to $LOG_FILE"

nohup python train.py --method "$METHOD" --adapter_dir "$OUTPUT_DIR" > "$LOG_FILE" 2>&1 &

echo "Training started in background."
echo "Use 'tail -f $LOG_FILE' to monitor progress."