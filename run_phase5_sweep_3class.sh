#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=03:00:00
#SBATCH --mem=30G
#SBATCH --job-name=phase5_sw_3c
#SBATCH --output=./job_logs/phase5_sweep_3class_%j.out
#SBATCH --error=./job_logs/phase5_sweep_3class_%j.err

set -e
set -o pipefail

source .venv/bin/activate

SEED=${1:-100}
EPOCHS=${2:-50}
AD_LIST=${3:-100,150,200}
MCI_LIST=${4:-100,150}

LOG_DIR=./job_logs
mkdir -p "$LOG_DIR"
RUN_TS=$(date '+%Y%m%d_%H%M%S')
RUN_LOG="$LOG_DIR/phase5_sweep_${RUN_TS}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START PHASE-5 SWEEP" | tee -a "$RUN_LOG"
echo "seed=$SEED epochs=$EPOCHS ad_list=$AD_LIST mci_list=$MCI_LIST" | tee -a "$RUN_LOG"

IFS=',' read -ra ADS <<< "$AD_LIST"
IFS=',' read -ra MCIS <<< "$MCI_LIST"

for ad_cap in "${ADS[@]}"; do
  for mci_cap in "${MCIS[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] RUN ad_cap=$ad_cap mci_cap=$mci_cap" | tee -a "$RUN_LOG"

    python main_3class.py --seed "$SEED" finetune --epochs "$EPOCHS" --unfreeze \
      --max_syn_ad "$ad_cap" --max_syn_mci "$mci_cap" 2>&1 | tee -a "$RUN_LOG"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] END ad_cap=$ad_cap mci_cap=$mci_cap" | tee -a "$RUN_LOG"
    echo "--------------------------------------------------" | tee -a "$RUN_LOG"
  done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] PHASE-5 SWEEP COMPLETE" | tee -a "$RUN_LOG"
