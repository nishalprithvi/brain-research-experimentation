import argparse
import json
import os
from datetime import datetime

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.data.utils import load_graphs

from src.data_loader_3class import load_adni_dgl_with_labels
from src.standard_gcn import StandardGCN


class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim=64):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, projection_dim),
        )

    def forward(self, g):
        h = self.encoder(g, features=None)
        return self.projector(h)


def augment_graph(g, drop_edge_prob=0.2, mask_feat_prob=0.2):
    g_aug = g.clone()

    if drop_edge_prob > 0:
        num_edges = g_aug.num_edges()
        mask = torch.empty(num_edges).uniform_(0, 1) > drop_edge_prob
        remove_indices = torch.nonzero(~mask).squeeze()
        if len(remove_indices.shape) > 0 and len(remove_indices) > 0:
            g_aug.remove_edges(remove_indices)

    if mask_feat_prob > 0 and "feat" in g_aug.ndata:
        feat = g_aug.ndata["feat"]
        mask = torch.empty((feat.shape[0], feat.shape[1])).uniform_(0, 1) > mask_feat_prob
        g_aug.ndata["feat"] = feat * mask.float()

    return g_aug


def info_nce_loss(z1, z2, temperature=0.5):
    batch_size = z1.shape[0]
    device = z1.device

    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)

    logits = torch.matmul(z1, z2.T) / max(temperature, 1e-6)
    labels = torch.arange(batch_size, device=device).long()
    return nn.CrossEntropyLoss()(logits, labels)


def collate_cl(graphs, drop_edge_prob=0.2):
    view1, view2 = [], []
    for g in graphs:
        view1.append(augment_graph(g, drop_edge_prob=drop_edge_prob))
        view2.append(augment_graph(g, drop_edge_prob=drop_edge_prob))
    return dgl.batch(view1), dgl.batch(view2)


def _sanitize_graphs(glist):
    for g in glist:
        if "E_features" in g.edata:
            weights = g.edata["E_features"]
        elif "feat" in g.edata:
            weights = g.edata["feat"]
        else:
            weights = torch.ones(g.num_edges())

        if weights.dim() == 1:
            weights = weights.view(-1, 1)

        for k in list(g.edata.keys()):
            del g.edata[k]
        g.edata["feat"] = weights

        for k in list(g.ndata.keys()):
            del g.ndata[k]

        k = int(g.num_edges() * 0.2)
        if k > 0:
            w_flat = weights.view(-1)
            topk_val = torch.topk(w_flat, k).values[-1]
            mask = w_flat < topk_val
            remove_indices = torch.nonzero(mask).squeeze()
            if len(remove_indices) > 0:
                g.remove_edges(remove_indices)

    return glist


def _cap_graphs(graphs, cap):
    if cap is None or cap < 0:
        return graphs
    return graphs[: min(cap, len(graphs))]


def train_contrastive(
    epochs=100,
    batch_size=32,
    syn_dir="./results_guidance_3class",
    use_synthetic=True,
    syn_ad_cap=-1,
    syn_mci_cap=-1,
    drop_edge_prob=0.2,
    temperature=0.5,
    quality_log_every=10,
    phase4_quality_dir="./results_phase4_quality",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Real Data...")
    glist_real, _ = load_adni_dgl_with_labels(data_dir="./data")
    glist_real = _sanitize_graphs(glist_real)

    glist_syn = []
    syn_stats = {"ad_loaded": 0, "ad_used": 0, "mci_loaded": 0, "mci_used": 0}

    if use_synthetic:
        cls_to_cap = {"ad": syn_ad_cap, "mci": syn_mci_cap}
        for cls_name in ["ad", "mci"]:
            syn_path = os.path.join(syn_dir, f"filtered_synthetic_{cls_name}.bin")
            if os.path.exists(syn_path):
                print(f"Loading Synthetic Data from {syn_path}...")
                glist_tmp, _ = load_graphs(syn_path)
                glist_tmp = _sanitize_graphs(glist_tmp)
                syn_stats[f"{cls_name}_loaded"] = len(glist_tmp)
                glist_tmp = _cap_graphs(glist_tmp, cls_to_cap[cls_name])
                syn_stats[f"{cls_name}_used"] = len(glist_tmp)
                glist_syn.extend(glist_tmp)
            else:
                print(f"Warning: Synthetic Data not found at {syn_path}")
    else:
        print("[CONFIG] Synthetic Data DISABLED for Pre-Training.")

    full_dataset = glist_real + glist_syn
    print(f"Pre-Training on {len(full_dataset)} graphs (Real + Syn).")
    print(
        f"[Phase4 Data] real={len(glist_real)} syn_total={len(glist_syn)} "
        f"syn_ad={syn_stats['ad_used']} syn_mci={syn_stats['mci_used']}"
    )

    loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda gs: collate_cl(gs, drop_edge_prob=drop_edge_prob),
    )

    encoder = StandardGCN(num_nodes=100, hidden_dim=64, n_classes=64)
    from dgl.nn import SumPooling

    encoder.pooling = SumPooling()
    print("Replaced Encoder Pooling with SumPooling.")

    model = SimCLR(encoder, projection_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n--- Starting Contrastive Pre-Training ---")
    if len(loader) > 0:
        sample_g = next(iter(loader))[0]
        one_g = dgl.unbatch(sample_g)[0]
        density = one_g.num_edges() / (one_g.num_nodes() ** 2)
        print(f"DEBUG: Sample Graph Density: {density:.4f} (Edges/Nodes^2)")

    os.makedirs(phase4_quality_dir, exist_ok=True)
    quality_points = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_pos_sim = 0.0
        total_neg_sim = 0.0
        total_batches = 0
        last_z1 = None

        for b1, b2 in loader:
            b1 = b1.to(device)
            b2 = b2.to(device)

            z1 = model(b1)
            z2 = model(b2)
            loss = info_nce_loss(z1, z2, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()

            if epoch == 0 and total_loss == 0:
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item()
                print(f"DEBUG: Initial Gradient Norm: {grad_norm:.6f}")

            optimizer.step()

            z1n = nn.functional.normalize(z1.detach(), dim=1)
            z2n = nn.functional.normalize(z2.detach(), dim=1)
            sim = torch.matmul(z1n, z2n.T)
            pos_sim = sim.diag().mean().item()
            neg_mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
            neg_sim = sim[neg_mask].mean().item() if neg_mask.any() else 0.0

            total_loss += loss.item()
            total_pos_sim += pos_sim
            total_neg_sim += neg_sim
            total_batches += 1
            last_z1 = z1.detach()

        avg_loss = total_loss / max(total_batches, 1)
        avg_pos = total_pos_sim / max(total_batches, 1)
        avg_neg = total_neg_sim / max(total_batches, 1)
        collapse_std = float(last_z1.std(dim=0).mean().item()) if last_z1 is not None else 0.0

        if (epoch + 1) % max(quality_log_every, 1) == 0 or (epoch + 1) == epochs:
            print(
                f"Epoch {epoch+1}/{epochs} | Contrastive Loss: {avg_loss:.4f} "
                f"| PosSim: {avg_pos:.4f} | NegSim: {avg_neg:.4f} | EmbStd: {collapse_std:.4f}"
            )
            quality_points.append(
                {
                    "epoch": int(epoch + 1),
                    "loss": float(avg_loss),
                    "pos_sim": float(avg_pos),
                    "neg_sim": float(avg_neg),
                    "emb_std": float(collapse_std),
                }
            )

    torch.save(model.encoder.state_dict(), "gcn_pretrained_3class.pth")
    print("Saved Pre-Trained Encoder to gcn_pretrained_3class.pth")

    q_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "temperature": float(temperature),
        "drop_edge_prob": float(drop_edge_prob),
        "use_synthetic": bool(use_synthetic),
        "syn_ad_cap": int(syn_ad_cap),
        "syn_mci_cap": int(syn_mci_cap),
        "real_count": int(len(glist_real)),
        "syn_count": int(len(glist_syn)),
        "syn_stats": syn_stats,
        "quality_points": quality_points,
    }
    q_file = os.path.join(phase4_quality_dir, "phase4_pretrain_metrics.json")
    with open(q_file, "w", encoding="utf-8") as f:
        json.dump(q_payload, f, indent=2)
    print(f"Saved Phase-4 quality metrics to {q_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--syn_dir", type=str, default="./results_guidance_3class")
    parser.add_argument("--no_synthetic", action="store_true")
    parser.add_argument("--syn_ad_cap", type=int, default=-1)
    parser.add_argument("--syn_mci_cap", type=int, default=-1)
    parser.add_argument("--drop_edge_prob", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--quality_log_every", type=int, default=10)
    parser.add_argument("--phase4_quality_dir", type=str, default="./results_phase4_quality")
    args = parser.parse_args()

    train_contrastive(
        epochs=args.epochs,
        batch_size=args.batch_size,
        syn_dir=args.syn_dir,
        use_synthetic=not args.no_synthetic,
        syn_ad_cap=args.syn_ad_cap,
        syn_mci_cap=args.syn_mci_cap,
        drop_edge_prob=args.drop_edge_prob,
        temperature=args.temperature,
        quality_log_every=args.quality_log_every,
        phase4_quality_dir=args.phase4_quality_dir,
    )
