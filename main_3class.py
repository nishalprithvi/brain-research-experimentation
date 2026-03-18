import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train_3class import run_training
from src.guided_sampling_3class import main as run_guided_sampling
from src.filter_synthetic_3class import main as run_filtering
from src.train_contrastive_3class import train_contrastive
from src.finetune_3class import train_finetune
from src.utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="3-Class Brain Network Classification Experimentation CLI")
    parser.add_argument('--seed', type=int, default=100, help='Global random seed (Use -1 for completely random/unseeded)')
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train Command
    train_parser = subparsers.add_parser("train", help="Train Phase 1: VAE, Diffusion, GCN")
    train_parser.add_argument('--epochs_vae', type=int, default=50)
    train_parser.add_argument('--epochs_diff', type=int, default=100)
    train_parser.add_argument('--epochs_gcn', type=int, default=100)
    train_parser.add_argument('--batch_size', type=int, default=16)
    
    # Guide Command
    guide_parser = subparsers.add_parser("guide", help="Phase 2: Generate Hard Negatives")
    guide_parser.add_argument('--scale', type=float, default=2.0)
    
    # Filter Command
    filter_parser = subparsers.add_parser("filter", help="Phase 3: Filter synthetic data")
    filter_parser.add_argument('--threshold_min', type=float, default=0.5)
    filter_parser.add_argument('--threshold_max', type=float, default=0.98)
    
    # Pretrain Command
    pretrain_parser = subparsers.add_parser("pretrain", help="Phase 4: Contrastive Pretraining")
    pretrain_parser.add_argument('--epochs', type=int, default=100)
    
    # Finetune Command
    ft_parser = subparsers.add_parser("finetune", help="Phase 5: Fine-tune Multi-class GCN")
    ft_parser.add_argument('--epochs', type=int, default=50)
    ft_parser.add_argument('--unfreeze', action='store_true')
    
    args = parser.parse_args()
    
    if args.seed is not None and args.seed != -1:
        set_seed(args.seed)
    elif args.seed == -1:
        print("[CONFIG] Running COMPLETELY UNSEEDED (Stochastic Mode)")
    
    if args.command == "train":
        print("--- Starting Phase 1: Generative Training ---")
        run_training(args)
    elif args.command == "guide":
        print("--- Starting Phase 2: Guided Generation ---")
        # Since guided_sampling_3class handles argparse internally in `main`, we bypass it 
        # by calling the script. We'll simply construct sys.argv.
        sys.argv = ['guided_sampling_3class.py', '--scale', str(args.scale)]
        run_guided_sampling()
    elif args.command == "filter":
        print("--- Starting Phase 3: Filtering ---")
        sys.argv = ['filter_synthetic_3class.py', '--threshold_min', str(args.threshold_min), '--threshold_max', str(args.threshold_max)]
        run_filtering()
    elif args.command == "pretrain":
        print("--- Starting Phase 4: Contrastive Pre-Training ---")
        train_contrastive(epochs=args.epochs)
    elif args.command == "finetune":
        print("--- Starting Phase 5: Fine-Tuning ---")
        train_finetune(epochs=args.epochs, frozen=not args.unfreeze)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
