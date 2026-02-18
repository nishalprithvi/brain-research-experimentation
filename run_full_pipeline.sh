#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --job-name=brain_contrastive_pipeline
#SBATCH --output=./job_logs/output_pipeline_%j.out
#SBATCH --error=./job_logs/error_pipeline_%j.err

# =============================================================================
# BRAIN NETWORK CONTRASTIVE LEARNING PIPELINE - END-TO-END JOB
# =============================================================================

# Assume we are running from the experimentation folder
PROJECT_ROOT=$(pwd)

# Exit immediately if a command exits with a non-zero status
# and ensure pipelined commands propagate failure
set -e
set -o pipefail

LOG_DIR="${PROJECT_ROOT}/job_logs"
mkdir -p "$LOG_DIR"

RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG="${LOG_DIR}/pipeline_${RUN_TIMESTAMP}.log"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RUN_LOG"
}

log_message "========================================="
log_message "STARTING FULL PIPELINE: brain_contrastive_pipeline"
log_message "========================================="
log_message "Project Root: $PROJECT_ROOT"
log_message "Log File: $RUN_LOG"
log_message "Slurm Job ID: ${SLURM_JOB_ID:-Local}"

# =============================================================================
# 1. ENVIRONMENT SETUP
# =============================================================================
log_message ""
log_message "--- 1. Setting up Environment ---"

# Load Modules (Adjust as per cluster config)
module load cuda/12.8.0 2>/dev/null || echo "Module load cuda failed or not needed."
module load anaconda 2>/dev/null || echo "Module load anaconda failed or not needed."

# Activate Venv
# Check and Setup Virtual Environment
if [ -d ".venv" ]; then
    # logical check: try running python --version. If it fails, the binary is likely from a different OS.
    if ! .venv/bin/python --version > /dev/null 2>&1; then
        log_message "⚠️  Detected broken .venv (likely from another OS). Deleting..."
        rm -rf .venv
    fi
fi

if [ ! -d ".venv" ]; then
    log_message "Creating new virtual environment using python3.9..."
    # Explicitly use the cluster's python3.9
    /usr/bin/python3.9 -m venv .venv
fi

# Activate
source .venv/bin/activate
log_message "Activated virtual environment: .venv"

# Upgrade pip and install dependencies
log_message "Checking and installing dependencies..."
python -m pip install --upgrade pip

# 1. Install PyTorch with CUDA 12.1 support
log_message "Installing PyTorch..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 torchdata==0.7.1 --index-url https://download.pytorch.org/whl/cu121 | tee -a "$RUN_LOG"

# 2. Install DGL with CUDA 12.1 support
log_message "Installing DGL..."
pip install dgl==2.1.0+cu121 -f https://data.dgl.ai/wheels/cu121/repo.html | tee -a "$RUN_LOG"

# 3. Install remaining requirements
if [ -f "requirements.txt" ]; then
    log_message "Installing remaining requirements..."
    pip install -r requirements.txt | tee -a "$RUN_LOG"
else
    log_message "⚠️ No requirements.txt found. Skipping dependency installation."
fi

# Fix for DGL not finding CUDA libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import os, torch; print(os.path.dirname(os.path.dirname(torch.__file__)) + '/nvidia/cusparse/lib')")
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import os, torch; print(os.path.dirname(os.path.dirname(torch.__file__)) + '/nvidia/cublas/lib')")

# Verify Dependencies
python -c "import torch; import dgl; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, DGL: {dgl.__version__}')" 2>&1 | tee -a "$RUN_LOG"

# =============================================================================
# 2. PHASE 1: GENERATION (SCALE-UP)
# =============================================================================
log_message ""
log_message "--- 2. PHASE 1: GENERATION (Scale-Up) ---"
log_message "Generating 2,500 synthetic hard negatives..."

CMD_GUIDE="python main.py guide --target 1 --scale 3.0 --n_samples 2500"
log_message "Executing: $CMD_GUIDE"

$CMD_GUIDE 2>&1 | tee -a "$RUN_LOG"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    log_message "❌ Phase 1 (Generation) Failed! code: $EXIT_CODE"
    exit $EXIT_CODE
fi
log_message "✅ Phase 1 (Generation) Complete."


# =============================================================================
# 3. PHASE 2: FILTERING
# =============================================================================
log_message ""
log_message "--- 3. PHASE 2: FILTERING ---"
log_message "Filtering synthetic data for quality/uniqueness..."

CMD_FILTER="python main.py filter"
log_message "Executing: $CMD_FILTER"

$CMD_FILTER 2>&1 | tee -a "$RUN_LOG"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    log_message "❌ Phase 2 (Filtering) Failed! code: $EXIT_CODE"
    exit $EXIT_CODE
fi
log_message "✅ Phase 2 (Filtering) Complete."


# =============================================================================
# 4. PHASE 3: PRE-TRAINING (CONTRASTIVE)
# =============================================================================
log_message ""
log_message "--- 4. PHASE 3: PRE-TRAINING (GraphCL) ---"
log_message "Training GCN encoder using Contrastive Loss (100 Epochs)..."

CMD_PRETRAIN="python main.py pretrain --epochs 100"
log_message "Executing: $CMD_PRETRAIN"

$CMD_PRETRAIN 2>&1 | tee -a "$RUN_LOG"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    log_message "❌ Phase 3 (Pre-training) Failed! code: $EXIT_CODE"
    exit $EXIT_CODE
fi
log_message "✅ Phase 3 (Pre-training) Complete."


# =============================================================================
# 5. PHASE 4: FINE-TUNING
# =============================================================================
log_message ""
log_message "--- 5. PHASE 4: FINE-TUNING ---"
log_message "Fine-tuning GCN classifier on Real ADNI Data (50 Epochs, Unfrozen)..."

CMD_FINETUNE="python main.py finetune --epochs 50 --unfreeze"
log_message "Executing: $CMD_FINETUNE"

$CMD_FINETUNE 2>&1 | tee -a "$RUN_LOG"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    log_message "❌ Phase 4 (Fine-tuning) Failed! code: $EXIT_CODE"
    exit $EXIT_CODE
fi
log_message "✅ Phase 4 (Fine-tuning) Complete."


# =============================================================================
# SUMMARY
# =============================================================================
log_message ""
log_message "========================================="
log_message "🎉 FULL PIPELINE COMPLETED SUCCESSFULLY"
log_message "========================================="
log_message "Results saved in:"
log_message "  - Generation: ./results_guidance/synthetic_hard_negatives.bin"
log_message "  - Filtered:   ./results_guidance/filtered_synthetic.bin"
log_message "  - Encoder:    ./gcn_pretrained_contrastive.pth"
log_message "  - Final Model: ./gcn_finetuned.pth"
log_message "  - Pipeline Log: $RUN_LOG"

exit 0
