#!/bin/bash
#SBATCH --job-name=python_venv_job
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Load any required modules (like Python base version)
module load python/3.12.9

# Activate your virtual environment (adjust path as needed)
source /cluster/home/t134762uhn/myenv_tf/bin/activate

# Navigate to your project directory
cd /cluster/home/t134762uhn/

# Run your Python script
python main.py

# (Optional) deactivate venv (not necessary in batch script)
deactivate