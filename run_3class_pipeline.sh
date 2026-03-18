#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=6:00:00 
#SBATCH --mem=30G
#SBATCH --job-name=run_3class
#SBATCH --output=./job_logs/3class_%j.out
#SBATCH --error=./job_logs/3class_%j.err

# =============================================================================
# 3-CLASS CLASSIFICATION PIPELINE - END-TO-END JOB
# =============================================================================

PROJECT_ROOT=$(pwd)

# Exit immediately if a command exits with a non-zero status
set -e
set -o pipefail

LOG_DIR="${PROJECT_ROOT}/job_logs"
mkdir -p "$LOG_DIR"

RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG="${LOG_DIR}/pipeline_3class_${RUN_TIMESTAMP}.log"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RUN_LOG"
}

SEED=${1:-100}

log_message "========================================="
log_message "STARTING 3-CLASS FULL PIPELINE"
log_message "========================================="
log_message "Project Root: $PROJECT_ROOT"
log_message "Log File: $RUN_LOG"
log_message "Slurm Job ID: ${SLURM_JOB_ID:-Local}"
log_message "Seed: $SEED"

# =============================================================================
# ENV SETUP
# =============================================================================
log_message ""
log_message "--- Environment Setup ---"
source .venv/bin/activate
log_message "Activated virtual environment: .venv"

# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

# Phase 1: Generative Training
log_message ""
log_message "--- Phase 1: Generative Training ---"
log_message "Training VAE, Diffusion, and Latent Dense GCN..."
CMD_TRAIN="python main_3class.py --seed $SEED train --epochs_vae 50 --epochs_diff 100 --epochs_gcn 100"
log_message "Executing: $CMD_TRAIN"
$CMD_TRAIN 2>&1 | tee -a "$RUN_LOG"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    log_message "❌ Phase 1 (Training) Failed! code: $EXIT_CODE"
    exit $EXIT_CODE
fi
log_message "✅ Phase 1 (Training) Complete."


# Phase 2: Guided Synthetic Generation
log_message ""
log_message "--- Phase 2: Guided Synthetic Generation ---"
log_message "Generating synthetic AD and MCI graphs..."
CMD_GUIDE="python main_3class.py --seed $SEED guide --scale 2.0"
log_message "Executing: $CMD_GUIDE"
$CMD_GUIDE 2>&1 | tee -a "$RUN_LOG"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    log_message "❌ Phase 2 (Generation) Failed! code: $EXIT_CODE"
    exit $EXIT_CODE
fi
log_message "✅ Phase 2 (Generation) Complete."


# Phase 3: Synthetic Filtering
log_message ""
log_message "--- Phase 3: Synthetic Filtering ---"
log_message "Filtering synthetic data for realism and uniqueness..."
CMD_FILTER="python main_3class.py --seed $SEED filter --threshold_min 0.5 --threshold_max 0.98"
log_message "Executing: $CMD_FILTER"
$CMD_FILTER 2>&1 | tee -a "$RUN_LOG"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    log_message "❌ Phase 3 (Filtering) Failed! code: $EXIT_CODE"
    exit $EXIT_CODE
fi
log_message "✅ Phase 3 (Filtering) Complete."


# Phase 4: Pre-training (Contrastive)
log_message ""
log_message "--- Phase 4: Pre-training (Contrastive) ---"
log_message "Pre-training Contrastive Encoder..."
CMD_PRETRAIN="python main_3class.py --seed $SEED pretrain --epochs 100"
log_message "Executing: $CMD_PRETRAIN"
$CMD_PRETRAIN 2>&1 | tee -a "$RUN_LOG"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    log_message "❌ Phase 4 (Pre-training) Failed! code: $EXIT_CODE"
    exit $EXIT_CODE
fi
log_message "✅ Phase 4 (Pre-training) Complete."


# Phase 5: Fine-tuning
log_message ""
log_message "--- Phase 5: Fine-tuning ---"
log_message "Fine-tuning final Multi-Class Classifier..."
CMD_FINETUNE="python main_3class.py --seed $SEED finetune --epochs 50 --unfreeze"
log_message "Executing: $CMD_FINETUNE"
$CMD_FINETUNE 2>&1 | tee -a "$RUN_LOG"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    log_message "❌ Phase 5 (Fine-tuning) Failed! code: $EXIT_CODE"
    exit $EXIT_CODE
fi
log_message "✅ Phase 5 (Fine-tuning) Complete."

# =============================================================================
# SUMMARY
# =============================================================================
log_message ""
log_message "========================================="
log_message "🎉 3-CLASS FULL PIPELINE COMPLETED SUCCESSFULLY"
log_message "========================================="
log_message "Pipeline Log: $RUN_LOG"

exit 0
