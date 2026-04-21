#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --mem=30G
#SBATCH --job-name=baseline3c
#SBATCH --output=./job_logs/baseline_3class_%j.out
#SBATCH --error=./job_logs/baseline_3class_%j.err

set -euo pipefail

cd /home/msai/prithvi005/brain_research/experimentation
source .venv/bin/activate

SEED=${1:-100}
EPOCHS=${2:-80}
BATCH=${3:-32}
LR=${4:-0.001}
WD=${5:-0.0005}
TEST_SIZE=${6:-0.2}

mkdir -p job_logs results_baseline_3class

echo "==================================================="
echo "RUNNING 3-CLASS BASELINE (REAL DATA ONLY)"
echo "seed=$SEED epochs=$EPOCHS batch_size=$BATCH lr=$LR weight_decay=$WD test_size=$TEST_SIZE"
echo "synthetic_data_used=0 contrastive_pretrain_used=0 manifold_learning_used=0"
echo "==================================================="

python -m src.baseline_3class \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --lr "$LR" \
  --weight_decay "$WD" \
  --test_size "$TEST_SIZE" \
  --output_root ./results_baseline_3class

echo "==================================================="
echo "BASELINE 3-CLASS RUN COMPLETE"
echo "Artifacts in: ./results_baseline_3class"
echo "==================================================="
