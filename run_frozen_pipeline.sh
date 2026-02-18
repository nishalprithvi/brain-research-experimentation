#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --job-name=frozen_pipe
#SBATCH --output=./job_logs/frozen_%j.out
#SBATCH --error=./job_logs/frozen_%j.err

# Load Environment
source .venv/bin/activate

# Run Frozen Pipeline
# Runs Phase 3 (Pre-train) and Phase 4 (Fine-tune) using a specific (frozen) synthetic dataset.

SYN_PATH=$1

if [ -z "$SYN_PATH" ]; then
    echo "Usage: $0 <path_to_synthetic_data.bin> [SEED]"
    echo "Example: $0 ./results_guidance/filtered_synthetic_seed_100.bin 42"
    exit 1
fi

SEED=${2:-42}

echo "==================================================="
echo "STARTING FROZEN PIPELINE"
echo "Synthetic Data: $SYN_PATH"
echo "Seed: $SEED"
echo "==================================================="

# Phase 3: Pre-training (Contrastive)
echo "--- Phase 3: Pre-training ---"
python main.py --seed $SEED pretrain --epochs 100 --syn_path "$SYN_PATH"

# Phase 4: Fine-tuning
echo "--- Phase 4: Fine-tuning ---"
python main.py --seed $SEED finetune --epochs 50 --unfreeze --syn_path "$SYN_PATH"

echo "==================================================="
echo "FROZEN PIPELINE COMPLETE"
echo "==================================================="
