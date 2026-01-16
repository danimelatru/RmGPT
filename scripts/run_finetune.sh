#!/bin/bash
#SBATCH --job-name=rmgpt_finetune
#SBATCH --output=logs/rmgpt_finetune_%j.out
#SBATCH --error=logs/rmgpt_finetune_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19

# 1. Load Environment
module purge
source ~/.bashrc

# Activate your moment environment
conda activate moment_env_py311

# --- CRITICAL FIX: Force Offline Mode ---
# This prevents the "MaxRetryError" / connection timeout
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# ----------------------------------------

# 2. Configure Paths (Absolute paths for stability)
PROJECT_ROOT="/gpfs/workdir/fernandeda/projects/RmGPT"

# Explicitly go to project root
cd "$PROJECT_ROOT"

# Define path to moment library
MOMENT_ROOT="/gpfs/workdir/fernandeda/projects/moment"

# Add paths to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$MOMENT_ROOT:$PYTHONPATH"

# 3. Debugging Info
echo "--- SLURM JOB INFO ---"
echo "Job Name: RMGPT FINETUNE (Supervised)"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo "Project Root: $PROJECT_ROOT"
echo "Moment Root: $MOMENT_ROOT"
echo "----------------------"

# 4. Run Training
echo "--- Starting Fine-Tuning ---"
if [ -f "src/training/finetune.py" ]; then
    # -u unbuffers output so you see logs in real-time with 'tail -f'
    python -u src/training/finetune.py
else
    echo "[ERROR] Could not find src/training/finetune.py"
fi