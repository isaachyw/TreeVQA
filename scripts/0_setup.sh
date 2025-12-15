#!/bin/bash

set -euo pipefail
trap 'echo "Error on line $LINENO"; exit 1' ERR

# check if the uv is installed if not install it
if ! command -v uv &> /dev/null; then
    echo "uv could not be found, installing it..."
    curl -sSL https://astral.sh/uv/install.sh | sh
fi

# # install the dependencies
uv sync

source .venv/bin/activate

echo "Running main.py to install the juliacall package and julia dependencies... (this may take a while)"

if ! python3 main.py --help &> /dev/null; then
    echo "main.py failed to run, please check if the environment is set up correctly"
    exit 1
fi

# check if tex is installed
if ! command -v tex &> /dev/null; then
    echo "tex could not be found, make sure you have texlive installed"
    exit 1
fi

echo "Environment setup complete"
mkdir -p checkpoints
touch checkpoints/0.checkpoint