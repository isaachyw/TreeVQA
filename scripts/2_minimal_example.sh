#!/bin/bash
# run minimal example of TreeVQA, 
set -euo pipefail

if [ ! -f 1.checkpoint ]; then
    echo "1.checkpoint file not found, please run scripts/1_ground_truth.sh"
    exit 1
fi

echo "Running minimal example of TreeVQA..."

source .venv/bin/activate
# run the python batch_ne.py script and get the return value
python3 batch_ne.py config/simple-h2-test.json

./concurrent_test-h2_1tasks_cpu.sh


touch checkpoints/2.checkpoint
echo 

