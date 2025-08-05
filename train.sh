#!/bin/bash

# Exit on any error
set -e

# === Run training ===
echo "Launching training with the following parameters:"

python main.py 

echo "Training complete."
