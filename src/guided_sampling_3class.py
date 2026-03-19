import argparse
import datetime
import json
import os

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from dgl.data.utils import save_graphs

from src.data_loader_3class import load_adni_data
from src.diffusion_model import DiffusionUNet
from src.gcn_model import LatentDenseGCN, LatentMLPTeacher
from src.vae_model import VAE


def _sanitize_matrix(adj):
    a = np.array(adj, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D adjacency, got shape={a.shape}")
    a = (a + a.T) / 2.0
    a = np.clip(a, 0.0, None)
    np.fill_diagonal(a, 0.0)
    return a


def _normalize_rows(x):
    x = x - x.mean(axis=1, keepdims=True)
    x = x / (x.std(axis=1, keepdims=True) + 1e-8)
    return x


def _corr_to_real(synthetic_matrices, real_matrices):
    syn_flat = synthetic_matrices.reshape(synthetic_matrices.shape[0], -1)
    real_flat = real_matrices.reshape(real_matrices.shape[0], -1)
    syn_norm = _normalize_rows(syn_flat)
    real_norm = _normalize_rows(real_flat)
    corr = np.dot(syn_norm, real_norm.T) / syn_flat.shape[1]
    max_corrs = np.max(corr, axis=1)
    return {
        "avg_max_corr": float(np.mean(max_corrs)),
        "p95_max_corr": float(np.percentile(max_corrs, 95)),
        "max_corr": float(np.max(max_corrs)),
        "max_corr_per_sample": max_corrs.tolist(),
    }


def _intra_duplicate_stats(synthetic_matrices):
    syn_flat = synthetic_matrices.reshape(synthetic_matrices.shape[0], -1)
    syn_norm = _normalize_rows(syn_flat)
    corr = np.dot(syn_norm, syn_norm.T) / syn_flat.shape[1]
    np.fill_diagonal(corr, -1.0)
    max_corrs = np.max(corr, axis=1)
    return {
        "avg_max_intra_corr": float(np.mean(max_corrs)),
        "p95_max_intra_corr": float(np.percentile(max_corrs, 95)),
        "max_intra_corr": float(np.max(max_corrs)),
        "max_intra_corr_per_sample": max_corrs.tolist(),
    }


def _pairwise_distance_stats(synthetic_matrices, max_samples=256):
    n = synthetic_matrices.shape[0]
    if n <= 1:
        return {"pairwise_l2_mean": 0.0, "pairwise_l2_std": 0.0, "pairwise_l2_p05": 0.0}
    idx = np.arange(n)
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(idx, size=max_samples, replace=False)
    x = synthetic_matrices[idx].reshape(len(idx), -1)
    x2 = np.sum(x * x, axis=1, keepdims=True)
    dist2 = np.maximum(x2 + x2.T - 2.0 * np.dot(x, x.T), 0.0)
    iu = np.triu_indices(dist2.shape[0], k=1)
    d = np.sqrt(dist2[iu] + 1e-12)
    return {
        "pairwise_l2_mean": float(np.mean(d)),
        "pairwise_l2_std": float(np.std(d)),
        "pairwise_l2_p05": float(np.percentile(d, 5)),
    }


def _spectral_topk(mats, topk=10, max_samples=None):
    n = mats.shape[0]
    idx = np.arange(n)
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(123)
        idx = rng.choice(idx, size=max_samples, replace=False)
    vals = []
    for i in idx:
        a = _sanitize_matrix(mats[i])
        eig = np.linalg.eigvalsh(a)
        eig = np.sort(eig)[::-1]
        vals.append(eig[:topk])
    vals = np.array(vals)
    return vals


def _graph_stats(mats, max_samples=256):
    n = mats.shape[0]
    idx = np.arange(n)
    if n > max_samples:
        rng = np.random.default_rng(7)
        idx = rng.choice(idx, size=max_samples, replace=False)
    deg_means = []
    deg_stds = []
    sparsity = []
    for i in idx:
        a = _sanitize_matrix(mats[i])
        deg = a.sum(axis=1)
        deg_means.append(float(np.mean(deg)))
        deg_stds.append(float(np.std(deg)))
        sparsity.append(float(np.mean(a <= 1e-8)))
    return {
        "degree_mean_avg": float(np.mean(deg_means)),
        "degree_std_avg": float(np.mean(deg_stds)),
        "sparsity_avg": float(np.mean(sparsity)),
    }


def _edge_hist_distance(syn_mats, real_mats, bins=40):
    syn_vals = syn_mats.reshape(-1)
    real_vals = real_mats.reshape(-1)
    syn_vals = syn_vals[syn_vals > 1e-8]
    real_vals = real_vals[real_vals > 1e-8]
    if len(syn_vals) == 0 or len(real_vals) == 0:
        return {"edge_hist_l1": 1.0, "edge_mean_syn": 0.0, "edge_mean_real": 0.0}
    hist_range = (0.0, max(float(np.max(syn_vals)), float(np.max(real_vals)), 1e-6))
    h_syn, _ = np.histogram(syn_vals, bins=bins, range=hist_range, density=True)
    h_real, _ = np.histogram(real_vals, bins=bins, range=hist_range, density=True)
    h_syn = h_syn / (h_syn.sum() + 1e-12)
    h_real = h_real / (h_real.sum() + 1e-12)
    l1 = float(np.sum(np.abs(h_syn - h_real)))
    return {
        "edge_hist_l1": l1,
        "edge_mean_syn": float(np.mean(syn_vals)),
        "edge_mean_real": float(np.mean(real_vals)),
    }


def _graph_topology_metrics(mats, max_samples=128, edge_threshold=0.2):
    n = mats.shape[0]
    idx = np.arange(n)
    if n > max_samples:
        rng = np.random.default_rng(99)
        idx = rng.choice(idx, size=max_samples, replace=False)
    effs = []
    cluster = []
    cpl = []
    conn_ratio = []
    for i in idx:
        a = _sanitize_matrix(mats[i])
        a_bin = (a > edge_threshold).astype(np.int32)
        np.fill_diagonal(a_bin, 0)
        g = nx.from_numpy_array(a_bin)
        if g.number_of_edges() == 0:
            effs.append(0.0)
            cluster.append(0.0)
            cpl.append(float(a.shape[0]))
            conn_ratio.append(0.0)
            continue
        effs.append(float(nx.global_efficiency(g)))
        cluster.append(float(nx.average_clustering(g)))
        if nx.is_connected(g):
            cpl.append(float(nx.average_shortest_path_length(g)))
            conn_ratio.append(1.0)
        else:
            largest_cc = max(nx.connected_components(g), key=len)
            g_cc = g.subgraph(largest_cc).copy()
            if g_cc.number_of_nodes() > 1:
                cpl.append(float(nx.average_shortest_path_length(g_cc)))
            else:
                cpl.append(float(a.shape[0]))
            conn_ratio.append(float(g_cc.number_of_nodes()) / float(a.shape[0]))
    return {
        "global_efficiency_mean": float(np.mean(effs)),
        "global_efficiency_std": float(np.std(effs)),
        "avg_clustering_mean": float(np.mean(cluster)),
        "char_path_length_mean": float(np.mean(cpl)),
        "largest_cc_ratio_mean": float(np.mean(conn_ratio)),
        "edge_threshold": float(edge_threshold),
    }


def _trajectory_monotonicity(trajectories):
    if not trajectories:
        return {"monotonicity_mean": 0.0, "monotonicity_std": 0.0}
    scores = []
    for seq in trajectories:
        if len(seq) < 2:
            continue
        diffs = np.diff(np.array(seq, dtype=np.float32))
        scores.append(float(np.mean(diffs >= -1e-4)))
    if not scores:
        return {"monotonicity_mean": 0.0, "monotonicity_std": 0.0}
    return {
        "monotonicity_mean": float(np.mean(scores)),
        "monotonicity_std": float(np.std(scores)),
    }


def _save_trajectories_plot(steps, trajectories, out_path, target_class):
    if not trajectories or not steps:
        return
    plt.figure(figsize=(8, 5))
    for seq in trajectories:
        plt.plot(steps, seq, alpha=0.35, linewidth=1.0)
    mean_curve = np.mean(np.array(trajectories), axis=0)
    plt.plot(steps, mean_curve, color="black", linewidth=2.0, label="Mean")
    plt.xlabel("Reverse Diffusion Timestep (descending)")
    plt.ylabel(f"Target Class {target_class} Probability")
    plt.title("Guidance Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _build_teacher(teacher_model_type, device):
    if teacher_model_type == "latent_mlp":
        return LatentMLPTeacher(latent_dim=256, hidden_dim=128, n_classes=3, dropout=0.2).to(device)
    return LatentDenseGCN(num_nodes=16, in_features=16, hidden_dim=32, n_classes=3).to(device)


def save_as_dgl(matrices, save_path):
    graphs = []
    for adj in matrices:
        a = _sanitize_matrix(adj)
        src, dst = np.nonzero(a > 0.0)
        g = dgl.graph((src, dst), num_nodes=a.shape[0])
        w = torch.tensor(a[src, dst], dtype=torch.float32).unsqueeze(1)
        g.edata["feat"] = w
        g.edata["E_features"] = w
        graphs.append(g)
    save_graphs(save_path, graphs)
    print(f"Saved {len(graphs)} DGL graphs to {save_path}")


def _phase2_quality_report(
    synthetic_matrices,
    target_class,
    teacher_probs,
    trajectories,
    trajectory_steps,
    quality_dir,
    cfg,
):
    os.makedirs(quality_dir, exist_ok=True)
    real_mats, real_labels = load_adni_data(data_dir="./data")
    valid = real_labels != -1
    real_mats = real_mats[valid]
    real_labels = real_labels[valid]
    class_real = real_mats[real_labels == target_class]
    if len(class_real) == 0:
        class_real = real_mats

    corr_real = _corr_to_real(synthetic_matrices, class_real)
    corr_intra = _intra_duplicate_stats(synthetic_matrices)
    diversity = _pairwise_distance_stats(synthetic_matrices, max_samples=cfg["quality_max_samples"])

    syn_spec = _spectral_topk(
        synthetic_matrices,
        topk=cfg["quality_spectral_topk"],
        max_samples=None,
    )
    real_spec = _spectral_topk(
        class_real,
        topk=cfg["quality_spectral_topk"],
        max_samples=cfg["quality_max_samples"],
    )
    real_spec_mean = np.mean(real_spec, axis=0)
    d_spec = np.linalg.norm(syn_spec - real_spec_mean[None, :], axis=1)
    d_spec_real = np.linalg.norm(real_spec - real_spec_mean[None, :], axis=1)
    recommended_tau = float(np.percentile(d_spec_real, 95))
    spectral_tau = cfg["quality_spectral_tau"] if cfg["quality_spectral_tau"] > 0 else recommended_tau

    graph_syn = _graph_stats(synthetic_matrices, max_samples=cfg["quality_max_samples"])
    graph_real = _graph_stats(class_real, max_samples=cfg["quality_max_samples"])
    topo_syn = _graph_topology_metrics(
        synthetic_matrices,
        max_samples=min(cfg["quality_max_samples"], 128),
        edge_threshold=cfg["quality_edge_threshold"],
    )
    topo_real = _graph_topology_metrics(
        class_real,
        max_samples=min(cfg["quality_max_samples"], 128),
        edge_threshold=cfg["quality_edge_threshold"],
    )
    edge_dist = _edge_hist_distance(synthetic_matrices, class_real, bins=40)

    target_probs = teacher_probs[:, target_class]
    conf_th = cfg["quality_conf_threshold_ad"] if target_class == 1 else cfg["quality_conf_threshold_mci"]
    pass_conf = target_probs >= conf_th
    pass_dup = np.array(corr_real["max_corr_per_sample"]) <= cfg["quality_dup_real_th"]
    pass_intra = np.array(corr_intra["max_intra_corr_per_sample"]) <= cfg["quality_dup_intra_th"]
    pass_spec = d_spec <= spectral_tau
    pass_all = pass_conf & pass_dup & pass_intra & pass_spec

    traj_stats = _trajectory_monotonicity(trajectories)
    traj_plot = os.path.join(quality_dir, f"guidance_trajectory_class_{target_class}.png")
    _save_trajectories_plot(trajectory_steps, trajectories, traj_plot, target_class)

    report = {
        "target_class": int(target_class),
        "num_synthetic": int(synthetic_matrices.shape[0]),
        "teacher_confidence": {
            "mean_target_prob": float(np.mean(target_probs)),
            "std_target_prob": float(np.std(target_probs)),
            "p10_target_prob": float(np.percentile(target_probs, 10)),
            "p50_target_prob": float(np.percentile(target_probs, 50)),
            "p90_target_prob": float(np.percentile(target_probs, 90)),
            "threshold": float(conf_th),
            "pass_rate": float(np.mean(pass_conf)),
        },
        "duplicate_risk": {
            "real": corr_real,
            "intra": corr_intra,
            "threshold_real": float(cfg["quality_dup_real_th"]),
            "threshold_intra": float(cfg["quality_dup_intra_th"]),
            "pass_rate_real": float(np.mean(pass_dup)),
            "pass_rate_intra": float(np.mean(pass_intra)),
        },
        "diversity": diversity,
        "spectral": {
            "topk": int(cfg["quality_spectral_topk"]),
            "tau": float(spectral_tau),
            "tau_input": float(cfg["quality_spectral_tau"]),
            "tau_recommended_p95_real": recommended_tau,
            "real_d_spec_mean": float(np.mean(d_spec_real)),
            "real_d_spec_p95": float(np.percentile(d_spec_real, 95)),
            "d_spec_mean": float(np.mean(d_spec)),
            "d_spec_p90": float(np.percentile(d_spec, 90)),
            "pass_rate": float(np.mean(pass_spec)),
        },
        "graph_stats_syn": graph_syn,
        "graph_stats_real": graph_real,
        "topology_syn": topo_syn,
        "topology_real": topo_real,
        "edge_distribution_similarity": edge_dist,
        "guidance_trajectory": {
            "tracked_samples": int(len(trajectories)),
            "tracked_steps": int(len(trajectory_steps)),
            **traj_stats,
            "plot_file": traj_plot,
        },
        "retention_summary": {
            "pass_rate_all_gates": float(np.mean(pass_all)),
            "num_pass_all": int(np.sum(pass_all)),
            "num_fail_any": int(len(pass_all) - np.sum(pass_all)),
        },
    }

    out_json = os.path.join(quality_dir, f"phase2_quality_class_{target_class}.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Phase2 Quality] Saved report: {out_json}")
    return report


def guided_sampling(
    vae_path,
    unet_path,
    gcn_path,
    target_class=1,
    n_samples=10,
    guidance_scale=2.0,
    save_dir="./results_guidance_3class",
    save_name="synthetic.bin",
    teacher_model_type="latent_densegcn",
    quality_eval=True,
    quality_dir=None,
    quality_track_samples=16,
    quality_track_stride=50,
    quality_conf_threshold_ad=0.85,
    quality_conf_threshold_mci=0.75,
    quality_spectral_topk=10,
    quality_spectral_tau=2.0,
    quality_dup_real_th=0.98,
    quality_dup_intra_th=0.995,
    quality_max_samples=256,
    quality_edge_threshold=0.2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading models for Class {target_class} Generation...")

    vae = VAE(input_dim=(100, 100), latent_dim=256).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    vae.eval()

    unet = DiffusionUNet().to(device)
    unet.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
    unet.eval()

    gcn = _build_teacher(teacher_model_type, device)
    gcn.load_state_dict(torch.load(gcn_path, map_location=device, weights_only=True))
    gcn.eval()

    print(f"Generating {n_samples} samples with Guidance Scale {guidance_scale} for Class {target_class}...")
    batch_size = 500
    n_batches = (n_samples + batch_size - 1) // batch_size
    all_matrices = []
    all_teacher_probs = []
    tracked_trajectories = []
    tracked_steps = []

    timesteps = 1000
    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for b in range(n_batches):
        current_batch_size = min(batch_size, n_samples - b * batch_size)
        print(f"  Batch {b+1}/{n_batches} (Size: {current_batch_size})")

        x_t = torch.randn((current_batch_size, 1, 16, 16), device=device)
        target = torch.full((current_batch_size,), target_class, device=device).long()
        local_track_n = 0
        local_trajs = []
        if quality_eval and b == 0 and quality_track_samples > 0:
            local_track_n = min(quality_track_samples, current_batch_size)
            local_trajs = [[] for _ in range(local_track_n)]

        for i in reversed(range(timesteps)):
            t = torch.full((current_batch_size,), i, device=device, dtype=torch.long)
            x_in = x_t.detach().requires_grad_(True)
            noise_pred = unet(x_in, t)

            beta_t = betas[i]
            alpha_t = alphas[i]
            alpha_hat = alphas_cumprod[i]
            sqrt_alpha_hat = torch.sqrt(alpha_hat)
            sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)
            x_0_hat = (x_in - sqrt_one_minus_alpha_hat * noise_pred) / sqrt_alpha_hat
            logits = gcn(x_0_hat)

            if local_track_n > 0 and (i % quality_track_stride == 0 or i == 0 or i == timesteps - 1):
                with torch.no_grad():
                    probs_t = torch.softmax(logits[:local_track_n], dim=1)[:, target_class].detach().cpu().numpy()
                for j in range(local_track_n):
                    local_trajs[j].append(float(probs_t[j]))
                if len(tracked_steps) < len(local_trajs[0]):
                    tracked_steps.append(int(i))

            loss = F.cross_entropy(logits, target)
            grad = torch.autograd.grad(outputs=loss, inputs=x_in)[0]

            with torch.no_grad():
                gradient_scale = guidance_scale * sqrt_one_minus_alpha_hat
                noise_pred_guided = noise_pred + gradient_scale * grad
                noise = torch.randn_like(x_t) if i > 0 else torch.zeros_like(x_t)
                coef1 = 1 / torch.sqrt(alpha_t)
                coef2 = (1 - alpha_t) / sqrt_one_minus_alpha_hat
                x_t = coef1 * (x_t - coef2 * noise_pred_guided) + torch.sqrt(beta_t) * noise

        with torch.no_grad():
            decoded_batch = vae.decode(x_t).cpu().numpy()[:, 0]
            all_matrices.append(decoded_batch)
            final_probs = torch.softmax(gcn(x_t), dim=1).detach().cpu().numpy()
            all_teacher_probs.append(final_probs)
        if local_trajs:
            tracked_trajectories.extend(local_trajs)

    matrices = np.concatenate(all_matrices, axis=0)
    teacher_probs = np.concatenate(all_teacher_probs, axis=0)
    print(f"Generated {len(matrices)} matrices total.")

    os.makedirs(save_dir, exist_ok=True)
    save_as_dgl(matrices, os.path.join(save_dir, save_name))

    report = None
    if quality_eval:
        quality_cfg = {
            "quality_conf_threshold_ad": quality_conf_threshold_ad,
            "quality_conf_threshold_mci": quality_conf_threshold_mci,
            "quality_spectral_topk": quality_spectral_topk,
            "quality_spectral_tau": quality_spectral_tau,
            "quality_dup_real_th": quality_dup_real_th,
            "quality_dup_intra_th": quality_dup_intra_th,
            "quality_max_samples": quality_max_samples,
            "quality_edge_threshold": quality_edge_threshold,
        }
        report = _phase2_quality_report(
            synthetic_matrices=matrices,
            target_class=target_class,
            teacher_probs=teacher_probs,
            trajectories=tracked_trajectories,
            trajectory_steps=tracked_steps,
            quality_dir=quality_dir or save_dir,
            cfg=quality_cfg,
        )

    log_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "target_class": int(target_class),
        "guidance_scale": float(guidance_scale),
        "n_samples": int(n_samples),
        "teacher_model_type": teacher_model_type,
        "saved_file": save_name,
    }
    if report is not None:
        log_data["quality_pass_rate_all_gates"] = report["retention_summary"]["pass_rate_all_gates"]
        log_data["quality_mean_target_prob"] = report["teacher_confidence"]["mean_target_prob"]
    log_path = os.path.join(save_dir, "experiment_log.txt")
    with open(log_path, "a") as f:
        f.write("-" * 50 + "\n")
        f.write(f"Experiment Run: {log_data['timestamp']}\n")
        for k, v in log_data.items():
            if k != "timestamp":
                f.write(f"{k}: {v}\n")
    print(f"Logged results to {log_path}")
    print(f"Finished generating Class {target_class}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--scale_ad", type=float, default=-1.0)
    parser.add_argument("--scale_mci", type=float, default=-1.0)
    parser.add_argument(
        "--teacher_model_type",
        type=str,
        default="latent_densegcn",
        choices=["latent_densegcn", "latent_mlp"],
    )
    parser.add_argument("--n_ad_override", type=int, default=-1)
    parser.add_argument("--n_mci_override", type=int, default=-1)
    parser.add_argument("--skip_quality_eval", action="store_true")
    parser.add_argument("--quality_track_samples", type=int, default=16)
    parser.add_argument("--quality_track_stride", type=int, default=50)
    parser.add_argument("--quality_conf_threshold_ad", type=float, default=0.85)
    parser.add_argument("--quality_conf_threshold_mci", type=float, default=0.75)
    parser.add_argument("--quality_spectral_topk", type=int, default=10)
    parser.add_argument("--quality_spectral_tau", type=float, default=2.0)
    parser.add_argument("--quality_dup_real_th", type=float, default=0.98)
    parser.add_argument("--quality_dup_intra_th", type=float, default=0.995)
    parser.add_argument("--quality_max_samples", type=int, default=256)
    parser.add_argument("--quality_edge_threshold", type=float, default=0.2)
    args = parser.parse_args()

    print("Evaluating real dataset distribution to balance classes...")
    _, labels = load_adni_data(data_dir="./data")
    valid = labels != -1
    labels = labels[valid]
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Real Distribution: {dist}")

    n_cn = dist.get(0, 0)
    n_ad = dist.get(1, 0)
    n_mci = dist.get(2, 0)
    target_ad_samples = max(0, n_cn - n_ad) if args.n_ad_override < 0 else max(0, args.n_ad_override)
    target_mci_samples = max(0, n_cn - n_mci) if args.n_mci_override < 0 else max(0, args.n_mci_override)
    print(f"Targeting {target_ad_samples} Synthetic AD and {target_mci_samples} Synthetic MCI.")
    scale_ad = args.scale if args.scale_ad <= 0 else args.scale_ad
    scale_mci = args.scale if args.scale_mci <= 0 else args.scale_mci

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    quality_root = os.path.join("./results_guidance_3class", "phase2_quality", run_ts)
    os.makedirs(quality_root, exist_ok=True)

    common_kwargs = {
        "vae_path": "vae_3class.pth",
        "unet_path": "diffusion_3class.pth",
        "gcn_path": "gcn_3class.pth",
        "teacher_model_type": args.teacher_model_type,
        "quality_eval": not args.skip_quality_eval,
        "quality_track_samples": args.quality_track_samples,
        "quality_track_stride": args.quality_track_stride,
        "quality_conf_threshold_ad": args.quality_conf_threshold_ad,
        "quality_conf_threshold_mci": args.quality_conf_threshold_mci,
        "quality_spectral_topk": args.quality_spectral_topk,
        "quality_spectral_tau": args.quality_spectral_tau,
        "quality_dup_real_th": args.quality_dup_real_th,
        "quality_dup_intra_th": args.quality_dup_intra_th,
        "quality_max_samples": args.quality_max_samples,
        "quality_edge_threshold": args.quality_edge_threshold,
    }

    if target_ad_samples > 0:
        guided_sampling(
            target_class=1,
            n_samples=target_ad_samples,
            guidance_scale=scale_ad,
            save_name="synthetic_ad.bin",
            quality_dir=os.path.join(quality_root, "class_ad"),
            **common_kwargs,
        )

    if target_mci_samples > 0:
        guided_sampling(
            target_class=2,
            n_samples=target_mci_samples,
            guidance_scale=scale_mci,
            save_name="synthetic_mci.bin",
            quality_dir=os.path.join(quality_root, "class_mci"),
            **common_kwargs,
        )


if __name__ == "__main__":
    main()
