#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --job-name=seed3class
#SBATCH --output=./job_logs/search_3class_%j.out
#SBATCH --error=./job_logs/search_3class_%j.err

# Load Environment
source .venv/bin/activate

# Range of seeds to test (Same as 2-class setup)
SEEDS=(42 100 2024 12345 999)

LOG_FILE="seed_search_results_3class.txt"
echo "3-Class Seed Search Results" > $LOG_FILE
echo "Timestamp: $(date)" >> $LOG_FILE
echo "---------------------------------------------------" >> $LOG_FILE

for seed in "${SEEDS[@]}"; do
    echo "==================================================="
    echo "Running 3-Class Pipeline with SEED: $seed"
    echo "==================================================="
    
    # Phase 1: Generative Training
    python main_3class.py --seed $seed train --epochs_vae 50 --epochs_diff 100 --epochs_gcn 100
    
    # Phase 2: Guided Synthetic Generation
    python main_3class.py --seed $seed guide --scale 2.0
    
    # Phase 3: Synthetic Filtering
    python main_3class.py --seed $seed filter --threshold_min 0.5 --threshold_max 0.98
    
    # Phase 4: Contrastive Pre-training
    python main_3class.py --seed $seed pretrain --epochs 100
    
    # Phase 5: Fine-tuning
    # Capture output to extract AUC, F1, and Class-wise metrics
    OUTPUT=$(python main_3class.py --seed $seed finetune --epochs 50 --unfreeze)
    
    # Ensure raw output is still logged in the slurm out file for debugging
    echo "$OUTPUT"
    
    # Extract Final Metrics
    AUC=$(echo "$OUTPUT" | grep "Final Multi-Class AUC-ROC (OVR):" | awk '{print $5}')
    F1=$(echo "$OUTPUT" | grep "Final Macro-F1:" | awk '{print $3}')
    
    CN_F1=$(echo "$OUTPUT" | grep " CN " | awk '{print $4}')
    AD_F1=$(echo "$OUTPUT" | grep " AD " | awk '{print $4}')
    MCI_F1=$(echo "$OUTPUT" | grep " MCI " | awk '{print $4}')
    
    echo "SEED $seed COMPLETED. OVR_AUC: $AUC | MACRO_F1: $F1 | CN: $CN_F1 | AD: $AD_F1 | MCI: $MCI_F1"
    echo "Seed: $seed | OVR_AUC: $AUC | MACRO_F1: $F1 | F1(CN, AD, MCI): $CN_F1, $AD_F1, $MCI_F1" >> $LOG_FILE
    
    # Rename synthetic data and model weights to preserve them
    mv ./results_guidance_3class/filtered_synthetic_ad.bin ./results_guidance_3class/filtered_synthetic_ad_seed_${seed}.bin 2>/dev/null || true
    mv ./results_guidance_3class/filtered_synthetic_mci.bin ./results_guidance_3class/filtered_synthetic_mci_seed_${seed}.bin 2>/dev/null || true
    mv gcn_finetuned_3class.pth gcn_finetuned_3class_seed_${seed}.pth 2>/dev/null || true
    
done

echo "==================================================="
echo "Search Complete. Results stored in $LOG_FILE"
cat $LOG_FILE
