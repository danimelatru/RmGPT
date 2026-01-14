#!/bin/bash
#SBATCH --job-name=rmgpt_pretrain
#SBATCH --output=logs/rmgpt_pretrain_%j.out
#SBATCH --error=logs/rmgpt_pretrain_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19

# 1. Load Environment
module purge
source ~/.bashrc

# Activate your moment environment
conda activate moment_env_py311

# 2. Configure Paths (Absolute paths for stability)
PROJECT_ROOT="/gpfs/workdir/fernandeda/projects/RmGPT"

# Explicitly go to project root
cd "$PROJECT_ROOT"

# Define paths to sister libraries
MOMENT_ROOT="/gpfs/workdir/fernandeda/projects/moment"
PHMD_ROOT="/gpfs/workdir/fernandeda/projects/phmd"

export PHMD_DATA="/gpfs/workdir/fernandeda/projects/dataset_storage/.phmd"

# Add ALL paths to PYTHONPATH so Python finds 'src', 'moment' AND 'phmd'
export PYTHONPATH="$PROJECT_ROOT:$MOMENT_ROOT:$PHMD_ROOT:$PYTHONPATH"

# 3. Debugging Info
echo "--- SLURM JOB INFO ---"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Python Interpreter: $(which python)"
echo "Project Root: $PROJECT_ROOT"
echo "Moment Root: $MOMENT_ROOT"
echo "Phmd Root: $PHMD_ROOT"
echo "Phmd Data: $PHMD_DATA" 
echo "PYTHONPATH: $PYTHONPATH"
echo "----------------------"

# 4. Run Training
echo "--- Starting Training ---"
if [ -f "src/training/pretrain.py" ]; then
    python -u src/training/pretrain.py
else
    echo "[ERROR] Could not find src/training/pretrain.py at $(pwd)/src/training/pretrain.py"
fi