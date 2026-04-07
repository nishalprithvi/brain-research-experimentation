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

set -euo pipefail
cd /home/msai/prithvi005/brain_research/experimentation
source .venv/bin/activate

SEED=${1:-}
if [ -z "$SEED" ]; then
    echo "Usage: $0 <SEED>"
    echo "Example: $0 100"
    exit 1
fi

# Tunables (defaults preserve old behavior where practical)
P4_EPOCHS=${P4_EPOCHS:-100}
P4_BATCH_SIZE=${P4_BATCH_SIZE:-32}
P4_USE_SYN=${P4_USE_SYN:-1}
P4_SYN_AD_CAP=${P4_SYN_AD_CAP:--1}
P4_SYN_MCI_CAP=${P4_SYN_MCI_CAP:--1}
P4_DROP_EDGE=${P4_DROP_EDGE:-0.2}
P4_TEMP=${P4_TEMP:-0.5}

P5_EPOCHS=${P5_EPOCHS:-50}
P5_UNFREEZE=${P5_UNFREEZE:-1}
P5_MAX_SYN_AD=${P5_MAX_SYN_AD:-100}
P5_MAX_SYN_MCI=${P5_MAX_SYN_MCI:-100}
P5_WEIGHT_MODE=${P5_WEIGHT_MODE:-none}
P5_LABEL_SMOOTH=${P5_LABEL_SMOOTH:-0.0}

echo "==================================================="
echo "STARTING FROZEN PIPELINE (3-CLASS)"
echo "Seed: $SEED"
echo "P4: epochs=$P4_EPOCHS batch=$P4_BATCH_SIZE use_syn=$P4_USE_SYN ad_cap=$P4_SYN_AD_CAP mci_cap=$P4_SYN_MCI_CAP drop_edge=$P4_DROP_EDGE temp=$P4_TEMP"
echo "P5: epochs=$P5_EPOCHS unfreeze=$P5_UNFREEZE max_syn_ad=$P5_MAX_SYN_AD max_syn_mci=$P5_MAX_SYN_MCI weight_mode=$P5_WEIGHT_MODE label_smooth=$P5_LABEL_SMOOTH"
echo "==================================================="

cp ./results_guidance_3class/filtered_synthetic_ad_seed_${SEED}.bin ./results_guidance_3class/filtered_synthetic_ad.bin
cp ./results_guidance_3class/filtered_synthetic_mci_seed_${SEED}.bin ./results_guidance_3class/filtered_synthetic_mci.bin

echo "--- Phase 4: Contrastive Pre-training ---"
PRETRAIN_CMD=(python main_3class.py --seed "$SEED" pretrain --epochs "$P4_EPOCHS" --batch_size "$P4_BATCH_SIZE" --pretrain_syn_ad_cap "$P4_SYN_AD_CAP" --pretrain_syn_mci_cap "$P4_SYN_MCI_CAP" --pretrain_drop_edge_prob "$P4_DROP_EDGE" --pretrain_temperature "$P4_TEMP")
if [[ "$P4_USE_SYN" == "0" ]]; then
  PRETRAIN_CMD+=(--no_pretrain_synthetic)
fi
"${PRETRAIN_CMD[@]}"

echo "--- Phase 5: Fine-tuning ---"
FINETUNE_CMD=(python main_3class.py --seed "$SEED" finetune --epochs "$P5_EPOCHS" --max_syn_ad "$P5_MAX_SYN_AD" --max_syn_mci "$P5_MAX_SYN_MCI" --loss_class_weight_mode "$P5_WEIGHT_MODE" --label_smoothing "$P5_LABEL_SMOOTH")
if [[ "$P5_UNFREEZE" == "1" ]]; then
  FINETUNE_CMD+=(--unfreeze)
fi
"${FINETUNE_CMD[@]}"

echo "==================================================="
echo "FROZEN PIPELINE (3-CLASS) COMPLETE"
echo "==================================================="
