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
    train_parser.add_argument('--vae_aux_cls_weight', type=float, default=0.0, help='Weight for optional supervised latent classification loss during VAE training')
    train_parser.add_argument('--quality_eval_every', type=int, default=5, help='Evaluate phase-1 quality metrics every N epochs')
    train_parser.add_argument('--phase1_quality_dir', type=str, default='./results_phase1_quality', help='Directory to store phase-1 quality outputs')
    train_parser.add_argument('--diffusion_num_buckets', type=int, default=10, help='Number of timestep buckets for diffusion quality checks')
    train_parser.add_argument('--teacher_model_type', type=str, default='latent_densegcn', choices=['latent_densegcn', 'latent_mlp'], help='Teacher model used in phase-1')
    train_parser.add_argument('--teacher_class_weight_mode', type=str, default='sqrt_inverse', choices=['none', 'inverse', 'sqrt_inverse', 'effective'], help='Class weighting mode for LatentDenseGCN teacher')
    train_parser.add_argument('--teacher_use_balanced_sampler', action='store_true', help='Enable class-balanced WeightedRandomSampler for LatentDenseGCN teacher')
    train_parser.add_argument('--teacher_loss_mode', type=str, default='ce', choices=['ce', 'class_balanced_ce'], help='Loss mode for LatentDenseGCN teacher')
    train_parser.add_argument('--teacher_max_class_weight', type=float, default=0.0, help='If >0, cap teacher class weights to this maximum value')
    train_parser.add_argument('--teacher_collapse_reg', type=float, default=0.05, help='Strength of anti-collapse regularizer on mean class probabilities')
    train_parser.add_argument('--teacher_early_stop_patience', type=int, default=20, help='Early stopping patience (epochs) based on teacher macro-F1')
    
    # Guide Command
    guide_parser = subparsers.add_parser("guide", help="Phase 2: Generate Hard Negatives")
    guide_parser.add_argument('--scale', type=float, default=2.0)
    guide_parser.add_argument('--scale_ad', type=float, default=-1.0)
    guide_parser.add_argument('--scale_mci', type=float, default=-1.0)
    guide_parser.add_argument('--teacher_model_type', type=str, default='latent_densegcn', choices=['latent_densegcn', 'latent_mlp'], help='Teacher model for guidance in phase-2')
    guide_parser.add_argument('--n_ad_override', type=int, default=-1, help='Override number of synthetic AD samples')
    guide_parser.add_argument('--n_mci_override', type=int, default=-1, help='Override number of synthetic MCI samples')
    guide_parser.add_argument('--skip_quality_eval', action='store_true', help='Skip phase-2 synthetic quality checks')
    guide_parser.add_argument('--quality_track_samples', type=int, default=16)
    guide_parser.add_argument('--quality_track_stride', type=int, default=50)
    guide_parser.add_argument('--quality_conf_threshold_ad', type=float, default=0.85)
    guide_parser.add_argument('--quality_conf_threshold_mci', type=float, default=0.75)
    guide_parser.add_argument('--quality_spectral_topk', type=int, default=10)
    guide_parser.add_argument('--quality_spectral_tau', type=float, default=2.0)
    guide_parser.add_argument('--quality_dup_real_th', type=float, default=0.98)
    guide_parser.add_argument('--quality_dup_intra_th', type=float, default=0.995)
    guide_parser.add_argument('--quality_max_samples', type=int, default=256)
    guide_parser.add_argument('--quality_edge_threshold', type=float, default=0.2)
    
    # Filter Command
    filter_parser = subparsers.add_parser("filter", help="Phase 3: Filter synthetic data")
    filter_parser.add_argument('--threshold_min', type=float, default=0.5)
    filter_parser.add_argument('--threshold_max', type=float, default=0.98)
    
    # Pretrain Command
    pretrain_parser = subparsers.add_parser("pretrain", help="Phase 4: Contrastive Pretraining")
    pretrain_parser.add_argument('--epochs', type=int, default=100)
    pretrain_parser.add_argument('--batch_size', type=int, default=32)
    pretrain_parser.add_argument('--syn_dir', type=str, default='./results_guidance_3class')
    pretrain_parser.add_argument('--no_pretrain_synthetic', action='store_true')
    pretrain_parser.add_argument('--pretrain_syn_ad_cap', type=int, default=-1)
    pretrain_parser.add_argument('--pretrain_syn_mci_cap', type=int, default=-1)
    pretrain_parser.add_argument('--pretrain_drop_edge_prob', type=float, default=0.2)
    pretrain_parser.add_argument('--pretrain_temperature', type=float, default=0.5)
    pretrain_parser.add_argument('--pretrain_quality_log_every', type=int, default=10)
    pretrain_parser.add_argument('--phase4_quality_dir', type=str, default='./results_phase4_quality')
    
    # Finetune Command
    ft_parser = subparsers.add_parser("finetune", help="Phase 5: Fine-tune Multi-class GCN")
    ft_parser.add_argument('--epochs', type=int, default=50)
    ft_parser.add_argument('--unfreeze', action='store_true')
    ft_parser.add_argument('--max_syn_ad', type=int, default=100)
    ft_parser.add_argument('--max_syn_mci', type=int, default=100)
    ft_parser.add_argument('--loss_class_weight_mode', type=str, default='none', choices=['none', 'inverse', 'sqrt_inverse', 'effective'])
    ft_parser.add_argument('--label_smoothing', type=float, default=0.0)
    
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
        # guided_sampling_3class parses its own CLI; pass through guide-specific args.
        sys.argv = [
            'guided_sampling_3class.py',
            '--scale', str(args.scale),
            '--scale_ad', str(args.scale_ad),
            '--scale_mci', str(args.scale_mci),
            '--teacher_model_type', str(args.teacher_model_type),
            '--n_ad_override', str(args.n_ad_override),
            '--n_mci_override', str(args.n_mci_override),
            '--quality_track_samples', str(args.quality_track_samples),
            '--quality_track_stride', str(args.quality_track_stride),
            '--quality_conf_threshold_ad', str(args.quality_conf_threshold_ad),
            '--quality_conf_threshold_mci', str(args.quality_conf_threshold_mci),
            '--quality_spectral_topk', str(args.quality_spectral_topk),
            '--quality_spectral_tau', str(args.quality_spectral_tau),
            '--quality_dup_real_th', str(args.quality_dup_real_th),
            '--quality_dup_intra_th', str(args.quality_dup_intra_th),
            '--quality_max_samples', str(args.quality_max_samples),
            '--quality_edge_threshold', str(args.quality_edge_threshold),
        ]
        if args.skip_quality_eval:
            sys.argv.append('--skip_quality_eval')
        run_guided_sampling()
    elif args.command == "filter":
        print("--- Starting Phase 3: Filtering ---")
        sys.argv = ['filter_synthetic_3class.py', '--threshold_min', str(args.threshold_min), '--threshold_max', str(args.threshold_max)]
        run_filtering()
    elif args.command == "pretrain":
        print("--- Starting Phase 4: Contrastive Pre-Training ---")
        train_contrastive(
            epochs=args.epochs,
            batch_size=args.batch_size,
            syn_dir=args.syn_dir,
            use_synthetic=not args.no_pretrain_synthetic,
            syn_ad_cap=args.pretrain_syn_ad_cap,
            syn_mci_cap=args.pretrain_syn_mci_cap,
            drop_edge_prob=args.pretrain_drop_edge_prob,
            temperature=args.pretrain_temperature,
            quality_log_every=args.pretrain_quality_log_every,
            phase4_quality_dir=args.phase4_quality_dir,
        )
    elif args.command == "finetune":
        print("--- Starting Phase 5: Fine-Tuning ---")
        train_finetune(
            epochs=args.epochs,
            frozen=not args.unfreeze,
            max_syn_ad=args.max_syn_ad,
            max_syn_mci=args.max_syn_mci,
            loss_class_weight_mode=args.loss_class_weight_mode,
            label_smoothing=args.label_smoothing,
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
