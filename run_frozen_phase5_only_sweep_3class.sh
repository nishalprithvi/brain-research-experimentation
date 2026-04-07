#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:30:00
#SBATCH --mem=30G
#SBATCH --job-name=frozen_p5only
#SBATCH --output=./job_logs/frozen_p5only_%j.out
#SBATCH --error=./job_logs/frozen_p5only_%j.err

set -euo pipefail
cd /home/msai/prithvi005/brain_research/experimentation
source .venv/bin/activate

SEED=${1:-100}
PRETRAIN_CKPT=${2:-job_logs/phase4_p4_cap_match_v2_20260323_160723/gcn_pretrained_3class_p4_cap_match_v2.pth}
AD_LIST=${3:-110,120,130}
MCI_LIST=${4:-161,180,200}
WEIGHT_MODES=${5:-inverse}
LS_LIST=${6:-0.0,0.03}
P5_EPOCHS=${P5_EPOCHS:-30}
P5_UNFREEZE=${P5_UNFREEZE:-1}

RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=job_logs/frozen_p5only_${RUN_TS}
mkdir -p "$RUN_DIR"
SUMMARY="$RUN_DIR/summary.tsv"
LOG="$RUN_DIR/sweep.log"

echo -e "tag\tauc\tmacro_f1\tad_recall\tmci_recall\tp5_ad\tp5_mci\twmode\tls" > "$SUMMARY"
cp -f "$PRETRAIN_CKPT" gcn_pretrained_3class.pth
cp -f "results_guidance_3class/filtered_synthetic_ad_seed_${SEED}.bin" "results_guidance_3class/filtered_synthetic_ad.bin"
cp -f "results_guidance_3class/filtered_synthetic_mci_seed_${SEED}.bin" "results_guidance_3class/filtered_synthetic_mci.bin"

echo "[$(date +"%F %T")] START frozen phase5-only sweep" | tee -a "$LOG"
echo "seed=$SEED pretrain_ckpt=$PRETRAIN_CKPT ad_list=$AD_LIST mci_list=$MCI_LIST weight_modes=$WEIGHT_MODES ls=$LS_LIST" | tee -a "$LOG"

IFS="," read -ra ADS <<< "$AD_LIST"
IFS="," read -ra MCIS <<< "$MCI_LIST"
IFS="," read -ra WM <<< "$WEIGHT_MODES"
IFS="," read -ra LSS <<< "$LS_LIST"

for ad in "${ADS[@]}"; do
  for mci in "${MCIS[@]}"; do
    for w in "${WM[@]}"; do
      for ls in "${LSS[@]}"; do
        tag="ad${ad}_mci${mci}_w${w}_ls${ls}"
        tclean=$(echo "$tag" | tr . p)
        rlog="$RUN_DIR/${tclean}.log"
        echo "[$(date +"%F %T")] RUN $tag" | tee -a "$LOG"

        cmd=(python main_3class.py --seed "$SEED" finetune --epochs "$P5_EPOCHS" --max_syn_ad "$ad" --max_syn_mci "$mci" --loss_class_weight_mode "$w" --label_smoothing "$ls")
        if [[ "$P5_UNFREEZE" == "1" ]]; then cmd+=(--unfreeze); fi
        "${cmd[@]}" 2>&1 | tee "$rlog"

        metrics=$(python - <<PY
import re
text=open("$rlog").read()
def g(p):
 m=re.search(p,text)
 return m.group(1) if m else "NA"
auc=g(r"Final Multi-Class AUC-ROC \(OVR\): ([0-9.]+)")
mf1=g(r"Final Macro-F1: ([0-9.]+)")
ad=g(r"\n\s*AD\s+[0-9.]+\s+([0-9.]+)\s+[0-9.]+\s+\d+")
mci=g(r"\n\s*MCI\s+[0-9.]+\s+([0-9.]+)\s+[0-9.]+\s+\d+")
print("\t".join([auc,mf1,ad,mci]))
PY
)
        auc=$(echo "$metrics" | cut -f1)
        mf1=$(echo "$metrics" | cut -f2)
        adr=$(echo "$metrics" | cut -f3)
        mcir=$(echo "$metrics" | cut -f4)
        echo -e "$tag\t$auc\t$mf1\t$adr\t$mcir\t$ad\t$mci\t$w\t$ls" >> "$SUMMARY"
        echo "[$(date +"%F %T")] END $tag -> AUC=$auc MF1=$mf1 AD_R=$adr MCI_R=$mcir" | tee -a "$LOG"
        echo "--------------------------------------------------" | tee -a "$LOG"
      done
    done
  done
done

echo "[$(date +"%F %T")] COMPLETE frozen phase5-only sweep" | tee -a "$LOG"
echo "Summary: $SUMMARY" | tee -a "$LOG"
