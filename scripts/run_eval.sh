#!/bin/bash
#SBATCH --job-name=rmgpt_eval
#SBATCH --output=logs/rmgpt_eval_%j.out
#SBATCH --error=logs/rmgpt_eval_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

# Load Environment
module purge
source ~/.bashrc
conda activate moment_env_py311

# Project Root
PROJECT_ROOT="/gpfs/workdir/fernandeda/projects/RmGPT"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "--- Generating Confusion Matrix ---"
python scripts/evaluate.py