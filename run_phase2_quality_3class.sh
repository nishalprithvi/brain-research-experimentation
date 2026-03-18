#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=03:00:00
#SBATCH --mem=30G
#SBATCH --job-name=phase2_qc_3c
#SBATCH --output=./job_logs/phase2_qc_3class_%j.out
#SBATCH --error=./job_logs/phase2_qc_3class_%j.err

set -e
set -o pipefail

source .venv/bin/activate

SEED=${1:-100}
SCALE=${2:-2.0}
N_AD=${3:-120}
N_MCI=${4:-120}
TEACHER_MODEL=${5:-latent_mlp}
SCALE_AD=${6:-3.0}
SCALE_MCI=${7:-2.0}
ENFORCE_GATES=${8:-0}
MIN_KEEP=${9:-50}
FALLBACK_KEEP_ALL=${10:-1}

LOG_DIR=./job_logs
mkdir -p "$LOG_DIR"
RUN_TS=$(date '+%Y%m%d_%H%M%S')
RUN_LOG="$LOG_DIR/phase2_quality_${RUN_TS}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START PHASE-2 QUALITY RUN" | tee -a "$RUN_LOG"
echo "seed=$SEED scale=$SCALE scale_ad=$SCALE_AD scale_mci=$SCALE_MCI n_ad=$N_AD n_mci=$N_MCI teacher_model=$TEACHER_MODEL enforce_gates=$ENFORCE_GATES min_keep=$MIN_KEEP fallback_keep_all=$FALLBACK_KEEP_ALL" | tee -a "$RUN_LOG"

CMD="python main_3class.py --seed $SEED guide \
  --scale $SCALE \
  --scale_ad $SCALE_AD \
  --scale_mci $SCALE_MCI \
  --teacher_model_type $TEACHER_MODEL \
  --n_ad_override $N_AD \
  --n_mci_override $N_MCI \
  --quality_track_samples 16 \
  --quality_track_stride 50 \
  --quality_conf_threshold_ad 0.85 \
  --quality_conf_threshold_mci 0.75 \
  --quality_spectral_topk 10 \
  --quality_spectral_tau -1.0 \
  --quality_dup_real_th 0.98 \
  --quality_dup_intra_th 0.995 \
  --quality_max_samples 256 \
  --quality_edge_threshold 0.2 \
  --quality_min_keep $MIN_KEEP"

if [ "$ENFORCE_GATES" = "1" ]; then
  CMD="$CMD --quality_enforce_gates"
fi
if [ "$FALLBACK_KEEP_ALL" = "1" ]; then
  CMD="$CMD --quality_fallback_keep_all"
fi

echo "Executing: $CMD" | tee -a "$RUN_LOG"
eval "$CMD" 2>&1 | tee -a "$RUN_LOG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] PHASE-2 QUALITY RUN COMPLETE" | tee -a "$RUN_LOG"
