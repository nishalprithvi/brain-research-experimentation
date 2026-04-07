#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --mem=30G
#SBATCH --job-name=phase5_sw_3c
#SBATCH --output=./job_logs/phase5_sweep_3class_%j.out
#SBATCH --error=./job_logs/phase5_sweep_3class_%j.err

set -euo pipefail

cd /home/msai/prithvi005/brain_research/experimentation
source .venv/bin/activate

SEED=${1:-100}
EPOCHS=${2:-30}
AD_LIST=${3:-100,120}
MCI_LIST=${4:-161,180}
WEIGHT_MODES=${5:-inverse,sqrt_inverse}
LABEL_SMOOTH_LIST=${6:-0.0,0.05}
UNFREEZE=${7:-1}
PRETRAIN_CKPT=${8:-}

LOG_DIR=./job_logs
mkdir -p "$LOG_DIR"
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$LOG_DIR/phase5_sweep_${RUN_TS}"
mkdir -p "$RUN_DIR"
RUN_LOG="$RUN_DIR/sweep.log"
SUMMARY_TSV="$RUN_DIR/summary.tsv"

echo -e "tag\tauc\tmacro_f1\tad_recall\tmci_recall\tad_cap\tmci_cap\tweight_mode\tlabel_smoothing\tunfreeze" > "$SUMMARY_TSV"

echo "[$(date +"%Y-%m-%d %H:%M:%S")] START PHASE-5 SWEEP" | tee -a "$RUN_LOG"
echo "seed=$SEED epochs=$EPOCHS ad_list=$AD_LIST mci_list=$MCI_LIST weight_modes=$WEIGHT_MODES label_smoothing=$LABEL_SMOOTH_LIST unfreeze=$UNFREEZE" | tee -a "$RUN_LOG"

if [[ -n "$PRETRAIN_CKPT" ]]; then
  cp -f "$PRETRAIN_CKPT" gcn_pretrained_3class.pth
  echo "Using pretrain checkpoint: $PRETRAIN_CKPT" | tee -a "$RUN_LOG"
fi

IFS="," read -ra ADS <<< "$AD_LIST"
IFS="," read -ra MCIS <<< "$MCI_LIST"
IFS="," read -ra WMODES <<< "$WEIGHT_MODES"
IFS="," read -ra LSMOOTH <<< "$LABEL_SMOOTH_LIST"

for ad_cap in "${ADS[@]}"; do
  for mci_cap in "${MCIS[@]}"; do
    for wmode in "${WMODES[@]}"; do
      for ls in "${LSMOOTH[@]}"; do
        tag="ad${ad_cap}_mci${mci_cap}_w${wmode}_ls${ls}_u${UNFREEZE}"
        tag_clean=$(echo "$tag" | tr "." "p")
        run_log="$RUN_DIR/${tag_clean}.log"

        echo "[$(date +"%Y-%m-%d %H:%M:%S")] RUN $tag" | tee -a "$RUN_LOG"

        cmd=(python main_3class.py --seed "$SEED" finetune --epochs "$EPOCHS" --max_syn_ad "$ad_cap" --max_syn_mci "$mci_cap" --loss_class_weight_mode "$wmode" --label_smoothing "$ls")
        if [[ "$UNFREEZE" == "1" ]]; then
          cmd+=(--unfreeze)
        fi

        "${cmd[@]}" 2>&1 | tee "$run_log"

        metrics=$(python - <<PY
import re
text=open("$run_log").read()
def g(p):
    m=re.search(p,text)
    return m.group(1) if m else "NA"
auc = g(r"Final Multi-Class AUC-ROC \(OVR\): ([0-9.]+)")
mf1 = g(r"Final Macro-F1: ([0-9.]+)")
ad  = g(r"\n\s*AD\s+[0-9.]+\s+([0-9.]+)\s+[0-9.]+\s+\d+")
mci = g(r"\n\s*MCI\s+[0-9.]+\s+([0-9.]+)\s+[0-9.]+\s+\d+")
print("\t".join([auc,mf1,ad,mci]))
PY
)
        auc=$(echo "$metrics" | cut -f1)
        mf1=$(echo "$metrics" | cut -f2)
        ad_recall=$(echo "$metrics" | cut -f3)
        mci_recall=$(echo "$metrics" | cut -f4)

        echo -e "${tag}\t${auc}\t${mf1}\t${ad_recall}\t${mci_recall}\t${ad_cap}\t${mci_cap}\t${wmode}\t${ls}\t${UNFREEZE}" >> "$SUMMARY_TSV"
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] END $tag -> AUC=${auc}, MF1=${mf1}, AD_R=${ad_recall}, MCI_R=${mci_recall}" | tee -a "$RUN_LOG"
        echo "--------------------------------------------------" | tee -a "$RUN_LOG"
      done
    done
  done
done

echo "[$(date +"%Y-%m-%d %H:%M:%S")] PHASE-5 SWEEP COMPLETE" | tee -a "$RUN_LOG"
echo "Summary: $SUMMARY_TSV" | tee -a "$RUN_LOG"
