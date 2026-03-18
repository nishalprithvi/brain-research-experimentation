#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --job-name=phase1_qc_3c
#SBATCH --output=./job_logs/phase1_qc_3class_%j.out
#SBATCH --error=./job_logs/phase1_qc_3class_%j.err

set -e
set -o pipefail

source .venv/bin/activate

SEEDS=${1:-100}
EPOCHS_VAE=${2:-50}
EPOCHS_DIFF=${3:-100}
EPOCHS_GCN=${4:-100}
BATCH_SIZE=${5:-16}
TEACHER_MODEL=${6:-latent_densegcn}

LOG_DIR="./job_logs"
mkdir -p "$LOG_DIR"
RUN_TS=$(date '+%Y%m%d_%H%M%S')
RUN_LOG="$LOG_DIR/phase1_quality_${RUN_TS}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START PHASE-1 QUALITY RUN" | tee -a "$RUN_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Seeds: $SEEDS" | tee -a "$RUN_LOG"

for seed in ${SEEDS//,/ }; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running seed=$seed" | tee -a "$RUN_LOG"

  python main_3class.py --seed "$seed" train \
    --epochs_vae "$EPOCHS_VAE" \
    --epochs_diff "$EPOCHS_DIFF" \
    --epochs_gcn "$EPOCHS_GCN" \
    --batch_size "$BATCH_SIZE" \
    --teacher_model_type "$TEACHER_MODEL" \
    --vae_aux_cls_weight 0.20 \
    --quality_eval_every 5 \
    --phase1_quality_dir ./results_phase1_quality \
    --diffusion_num_buckets 10 \
    --teacher_class_weight_mode sqrt_inverse \
    --teacher_loss_mode class_balanced_ce \
    --teacher_max_class_weight 2.0 \
    --teacher_collapse_reg 0.20 \
    --teacher_early_stop_patience 20 2>&1 | tee -a "$RUN_LOG"

  LAST_QUALITY_DIR=$(ls -1dt ./results_phase1_quality/phase1_*_seed_${seed} 2>/dev/null | head -n 1)
  if [ -n "$LAST_QUALITY_DIR" ] && [ -f "$LAST_QUALITY_DIR/phase1_quality_summary.json" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Summary for seed=$seed" | tee -a "$RUN_LOG"
    python - <<'PY' "$LAST_QUALITY_DIR/phase1_quality_summary.json" | tee -a "$RUN_LOG"
import json, sys
p = sys.argv[1]
with open(p, 'r') as f:
    d = json.load(f)
b = d.get('teacher_best', {})
print(f"  best_epoch={b.get('best_epoch')}")
print(f"  teacher_macro_f1={b.get('best_macro_f1')}")
print(f"  recall_cn={b.get('best_recall_class_0')}, recall_ad={b.get('best_recall_class_1')}, recall_mci={b.get('best_recall_class_2')}")
print(f"  ece={b.get('best_ece')}, brier={b.get('best_brier')}, entropy={b.get('best_entropy')}")
print(f"  temp={b.get('temperature')}, calibrated_ece={b.get('best_calibrated_ece')}, calibrated_brier={b.get('best_calibrated_brier')}")
PY
  fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] PHASE-1 QUALITY RUN COMPLETE" | tee -a "$RUN_LOG"
