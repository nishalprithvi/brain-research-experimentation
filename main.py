
import argparse
import sys
import os

# Ensure src is in path so internal imports work if running from experimentation/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import run_training
from src.validate import run_validation
from src.guided_sampling import guided_sampling
from src.retrain_standard_gcn import train_standard_gcn
from src.filter_synthetic import filter_synthetic_data
from src.train_contrastive import train_contrastive
from src.finetune import train_finetune

def main():
    parser = argparse.ArgumentParser(description="Brain Network Classification Experimentation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train Command
    train_parser = subparsers.add_parser("train", help="Train the VAE and Diffusion Model")
    train_parser.add_argument('--epochs_vae', type=int, default=50, help='Epochs to train VAE')
    train_parser.add_argument('--epochs_diff', type=int, default=100, help='Epochs to train Diffusion Model')
    train_parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    
    # Validate Command
    val_parser = subparsers.add_parser("validate", help="Generate synthetic samples and visualize")
    val_parser.add_argument('--n_samples', type=int, default=10, help='Number of synthetic samples to generate')
    
    # Guide Command
    guide_parser = subparsers.add_parser("guide", help="Generate Hard Negatives using Classifier Guidance")
    guide_parser.add_argument('--target', type=int, default=1, help='Target Class (1=AD)')
    guide_parser.add_argument('--scale', type=float, default=2.0, help='Guidance Scale')
    guide_parser.add_argument('--n_samples', type=int, default=10, help='Number of samples')
    
    # Filter Command
    filter_parser = subparsers.add_parser("filter", help="Filter synthetic data for quality/uniqueness")
    filter_parser.add_argument('--syn_path', type=str, default='./results_guidance/synthetic_hard_negatives.bin')
    filter_parser.add_argument('--threshold_min', type=float, default=0.5)
    filter_parser.add_argument('--threshold_max', type=float, default=0.98)
    
    # Retrain Command
    retrain_parser = subparsers.add_parser("retrain", help="Retrain Standard GCN on Real + Synthetic Data")
    retrain_parser.add_argument('--syn_path', type=str, default='./results_guidance/filtered_synthetic.bin', help='Path to synthetic data')
    retrain_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')

    # Pretrain Command (Contrastive)
    pretrain_parser = subparsers.add_parser("pretrain", help="Pre-train GCN using Graph Contrastive Learning")
    pretrain_parser.add_argument('--syn_path', type=str, default='./results_guidance/filtered_synthetic.bin', help='Path to synthetic data')
    pretrain_parser.add_argument('--epochs', type=int, default=100, help='Pre-training epochs')
    
    # Finetune Command
    ft_parser = subparsers.add_parser("finetune", help="Fine-tune GCN on Real Data")
    ft_parser.add_argument('--epochs', type=int, default=50, help='Fine-tuning epochs')
    ft_parser.add_argument('--unfreeze', action='store_true', help='Unfreeze encoder weights')
    
    # Experiment Command (Run All)
    exp_parser = subparsers.add_parser("run_all", help="Run full pipeline: Train -> Validate")
    exp_parser.add_argument('--epochs_vae', type=int, default=50)
    exp_parser.add_argument('--epochs_diff', type=int, default=100)
    exp_parser.add_argument('--batch_size', type=int, default=16)
    exp_parser.add_argument('--n_samples', type=int, default=10)
    
    args = parser.parse_args()
    
    if args.command == "train":
        print("--- Starting Training ---")
        run_training(args)
    elif args.command == "validate":
        print("--- Starting Validation ---")
        run_validation(n_samples=args.n_samples)
    elif args.command == "guide":
        print("--- Starting Guided Sampling ---")
        guided_sampling(
            vae_path='vae_adni.pth',
            unet_path='diffusion_adni.pth',
            gcn_path='gcn_adni.pth',
            guidance_scale=args.scale,
            target_class=args.target,
            n_samples=args.n_samples
        )
    elif args.command == "filter":
        print("--- Starting Filtering ---")
        filter_synthetic_data(args.syn_path, threshold_min=args.threshold_min, threshold_max=args.threshold_max)
    elif args.command == "retrain":
        print("--- Starting Retraining ---")
        train_standard_gcn(syn_data_path=args.syn_path, epochs=args.epochs)
    elif args.command == "pretrain":
        print("--- Starting Contrastive Pre-Training ---")
        train_contrastive(epochs=args.epochs, syn_path=args.syn_path)
    elif args.command == "finetune":
        print("--- Starting Fine-Tuning ---")
        train_finetune(epochs=args.epochs, frozen=not args.unfreeze)
    elif args.command == "run_all":
        print("--- Starting Full Experiment ---")
        run_training(args)
        print("\n--- Training Complete. Starting Validation ---")
        run_validation(n_samples=args.n_samples)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
