#!/bin/bash
# run minimal example of TreeVQA, 
set -euo pipefail

if [ ! -f checkpoints/2.checkpoint ]; then
    echo "2.checkpoint file not found, please run scripts/2_minimal_example.sh"
    exit 1
fi

echo "Running analysis plot..."

source .venv/bin/activate
# run the python analysis_plot.py script and get the return value
cd plot_util
python3 cobyla_analysis.py
python3 spsa_analysis.py

touch checkpoints/3.checkpoint
echo "Analysis plot complete"