#!/bin/bash

set -euo pipefail

if [ ! -f checkpoints/0.checkpoint ]; then
    echo "0.checkpoint file not found, please run scripts/0_setup.sh"
    exit 1
fi

trap 'echo "Error on line $LINENO"; exit 1' ERR

# check if the environment is set up correctly
if ! source .venv/bin/activate; then
    echo "Environment is not set up correctly, please run scripts/0_setup.sh"
    exit 1
fi

# run the ground truth script
echo "Running ground truth script, will take around 1 hour to complete"
python3 ground-state/ground_plot.py

touch checkpoints/1.checkpoint