#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --job-name=seed_search
#SBATCH --output=./job_logs/search_%j.out
#SBATCH --error=./job_logs/search_%j.err

# Load Environment (Ensure we are in the correct env)
source .venv/bin/activate

# Runs the pipeline with different seeds to find the one that yields the best Fine-tuning AUC.

# Range of seeds to test
SEEDS=(42 100 2024 12345 999)

LOG_FILE="seed_search_results.txt"
echo "Seed Search Results" > $LOG_FILE
echo "Timestamp: $(date)" >> $LOG_FILE
echo "---------------------------------------------------" >> $LOG_FILE

for seed in "${SEEDS[@]}"; do
    echo "==================================================="
    echo "Running Pipeline with SEED: $seed"
    echo "==================================================="
    
    # 1. Generate (Guide)
    echo "Using Seed $seed for Generation..."
    # Note: We pass --seed to main.py. 
    # Current main.py structure requires --seed BEFORE command? No, argparse handles it if defined in parent.
    # Check structure: parser.add_argument is in parent.
    # So `python main.py --seed 42 guide ...`
    
    # Phase 1: Generation
    python main.py --seed $seed guide --target 1 --scale 3.0 --n_samples 2500
    
    # Phase 2: Filtering
    python main.py --seed $seed filter
    
    # Phase 3: Pre-training
    python main.py --seed $seed pretrain --epochs 100
    
    # Phase 4: Fine-tuning
    # Capture output to extract AUC
    OUTPUT=$(python main.py --seed $seed finetune --epochs 50 --unfreeze)
    
    # Extract Final AUC (Last occurrence of "Final AUC-ROC:")
    AUC=$(echo "$OUTPUT" | grep "Final AUC-ROC:" | tail -n 1 | awk '{print $3}')
    F1=$(echo "$OUTPUT" | grep "Final Macro-F1:" | tail -n 1 | awk '{print $3}')
    
    echo "SEED $seed COMPLETED. AUC: $AUC | F1: $F1"
    echo "Seed: $seed | AUC: $AUC | F1: $F1" >> $LOG_FILE
    
    # Save the synthetic data if it's the best so far? 
    # For now, just rename the output bin so we don't lose it.
    mv ./results_guidance/filtered_synthetic.bin ./results_guidance/filtered_synthetic_seed_${seed}.bin
    mv gcn_finetuned.pth gcn_finetuned_seed_${seed}.pth
    
done

echo "==================================================="
echo "Search Complete. Results:"
cat $LOG_FILE
