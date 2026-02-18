#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --mem=30G
#SBATCH --job-name=contrastive_real_only
#SBATCH --output=./job_logs/contrastive_real_%j.out
#SBATCH --error=./job_logs/contrastive_real_%j.err

# Run Contrastive Learning on Real ADNI Data Only

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create job_logs directory if it doesn't exist
mkdir -p job_logs

# Generate a timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="job_logs/contrastive_real_${TIMESTAMP}.log"

echo "Starting Contrastive Learning (Real Data Only)..."
echo "Logging to $LOG_FILE"
python3 src/train_contrastive_real.py --epochs 100 > "$LOG_FILE" 2>&1

echo "Done. Check $LOG_FILE for details."
