#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --job-name=frozen_3class
#SBATCH --output=./job_logs/frozen_3class_%j.out
#SBATCH --error=./job_logs/frozen_3class_%j.err

# Load Environment
source .venv/bin/activate

# Run Frozen Pipeline for 3-class
# Uses previously generated synthetic datasets for a specific seed.

SEED=$1

if [ -z "$SEED" ]; then
    echo "Usage: $0 <SEED>"
    echo "Example: $0 100"
    exit 1
fi

echo "==================================================="
echo "STARTING FROZEN PIPELINE (3-CLASS)"
echo "Seed: $SEED"
echo "==================================================="

# 1. Restore the Synthetic Data
# We copy the seed-specific data to the default filenames read by the pipeline
cp ./results_guidance_3class/filtered_synthetic_ad_seed_${SEED}.bin ./results_guidance_3class/filtered_synthetic_ad.bin
cp ./results_guidance_3class/filtered_synthetic_mci_seed_${SEED}.bin ./results_guidance_3class/filtered_synthetic_mci.bin

# Phase 4: Pre-training (Contrastive)
echo "--- Phase 4: Contrastive Pre-training ---"
python main_3class.py --seed $SEED pretrain --epochs 100

# Phase 5: Fine-tuning
echo "--- Phase 5: Fine-tuning ---"
python main_3class.py --seed $SEED finetune --epochs 50 --unfreeze

echo "==================================================="
echo "FROZEN PIPELINE (3-CLASS) COMPLETE"
echo "==================================================="
