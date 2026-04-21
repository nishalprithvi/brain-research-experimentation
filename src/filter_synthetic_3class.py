import argparse
import os

import numpy as np
from dgl.data.utils import load_graphs, save_graphs

from src.data_loader_3class import load_adni_dgl_with_labels


def _get_adj_flat(glist):
    flat_list = []
    for g in glist:
        if "feat" in g.edata:
            w = g.edata["feat"].squeeze().cpu().numpy()
        elif "E_features" in g.edata:
            w = g.edata["E_features"].squeeze().cpu().numpy()
        else:
            w = np.ones(g.num_edges(), dtype=np.float32)

        src, dst = g.edges()
        src = src.cpu().numpy()
        dst = dst.cpu().numpy()

        adj = np.zeros((100, 100), dtype=np.float32)
        adj[src, dst] = w
        flat_list.append(adj.flatten())

    return np.array(flat_list, dtype=np.float32)


def filter_synthetic_data(
    synthetic_path,
    target_class,
    real_data_dir="./data",
    output_path="./results_guidance_3class/filtered_synthetic.bin",
    threshold_min=0.5,
    threshold_max=0.98,
):
    print(f"Loading synthetic data from {synthetic_path}...")
    try:
        syn_graphs, _ = load_graphs(synthetic_path)
    except Exception as e:
        print(f"Error loading synthetic data: {e}")
        return

    print(f"Loading real data from {real_data_dir}...")
    real_graphs, real_labels = load_adni_dgl_with_labels(data_dir=real_data_dir)

    valid = real_labels != -1
    real_graphs = [real_graphs[i] for i in range(len(real_graphs)) if valid[i]]
    real_labels = real_labels[valid]

    # Class-conditional reference pool (AD->AD, MCI->MCI)
    real_graphs_cls = [g for g, y in zip(real_graphs, real_labels.tolist()) if int(y) == int(target_class)]
    if len(real_graphs_cls) == 0:
        print(
            f"Warning: no real class-{target_class} references available; "
            "falling back to all real valid graphs."
        )
        real_graphs_cls = real_graphs

    print(f"Reference real graphs for class {target_class}: {len(real_graphs_cls)}")

    print("Converting graphs to vectors for comparing...")
    real_flat = _get_adj_flat(real_graphs_cls)
    syn_flat = _get_adj_flat(syn_graphs)

    print(f"Real Vectors: {real_flat.shape}")
    print(f"Syn Vectors: {syn_flat.shape}")

    real_mean = real_flat.mean(axis=1, keepdims=True)
    real_std = real_flat.std(axis=1, keepdims=True) + 1e-8
    real_norm = (real_flat - real_mean) / real_std

    syn_mean = syn_flat.mean(axis=1, keepdims=True)
    syn_std = syn_flat.std(axis=1, keepdims=True) + 1e-8
    syn_norm = (syn_flat - syn_mean) / syn_std

    print("Computing Correlation Matrix...")
    corr_matrix = np.dot(syn_norm, real_norm.T) / real_flat.shape[1]

    valid_indices = []
    print(f"Filtering with thresholds: Min={threshold_min}, Max={threshold_max}")
    for i in range(len(syn_norm)):
        max_sim = np.max(corr_matrix[i])
        if max_sim < threshold_min:
            continue
        if max_sim > threshold_max:
            continue
        valid_indices.append(i)

    print(f"Filter Complete. Kept {len(valid_indices)} / {len(syn_graphs)}.")

    if len(valid_indices) == 0:
        print("No samples passed the filter! Try adjusting guidance/filter thresholds.")
        return

    filtered_graphs = [syn_graphs[i] for i in valid_indices]
    save_graphs(output_path, filtered_graphs)
    print(f"Saved filtered data to {output_path}")

    log_path = os.path.dirname(output_path)
    with open(os.path.join(log_path, "experiment_log.txt"), "a", encoding="utf-8") as f:
        f.write("\n--- Class-Conditional Uniqueness Filtering (3-class) ---\n")
        f.write(f"Target class: {target_class}\n")
        f.write(f"Input Samples: {len(syn_graphs)}\n")
        f.write(f"Valid Samples: {len(valid_indices)}\n")
        f.write(f"Reference real class pool size: {len(real_graphs_cls)}\n")
        f.write(f"Thresholds: {threshold_min} < r < {threshold_max}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold_min", type=float, default=0.5)
    parser.add_argument("--threshold_max", type=float, default=0.98)
    args = parser.parse_args()

    base_dir = "./results_guidance_3class"

    # AD (class 1)
    ad_path = os.path.join(base_dir, "synthetic_ad.bin")
    ad_out = os.path.join(base_dir, "filtered_synthetic_ad.bin")
    if os.path.exists(ad_path):
        print("------- Filtering Synthetic AD (class-conditional) -------")
        filter_synthetic_data(
            ad_path,
            target_class=1,
            output_path=ad_out,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
        )
    else:
        print(f"Skipping AD filtering: {ad_path} not found.")

    # MCI (class 2)
    mci_path = os.path.join(base_dir, "synthetic_mci.bin")
    mci_out = os.path.join(base_dir, "filtered_synthetic_mci.bin")
    if os.path.exists(mci_path):
        print("\n------- Filtering Synthetic MCI (class-conditional) -------")
        filter_synthetic_data(
            mci_path,
            target_class=2,
            output_path=mci_out,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
        )
    else:
        print(f"Skipping MCI filtering: {mci_path} not found.")


if __name__ == "__main__":
    main()
