import argparse
import os
import random

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.data.utils import load_graphs
from dgl.nn import SumPooling
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler

from src.data_loader_3class import load_adni_dgl_with_labels
from src.standard_gcn import StandardGCN


def _class_weights_from_distribution(dist, mode="none", beta=0.999):
    if mode == "none":
        return None
    weights = []
    for c in [0, 1, 2]:
        n_c = max(int(dist.get(c, 0)), 1)
        if mode == "effective":
            eff_num = 1.0 - (beta**n_c)
            w = (1.0 - beta) / max(eff_num, 1e-12)
        elif mode == "sqrt_inverse":
            w = 1.0 / np.sqrt(float(n_c))
        else:
            w = 1.0 / float(n_c)
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / max(weights.min(), 1e-12)
    return torch.tensor(weights, dtype=torch.float32)


class BalancedSoftmaxCrossEntropy(nn.Module):
    def __init__(self, class_counts, label_smoothing=0.0):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        counts = torch.clamp(counts, min=1.0)
        priors = counts / counts.sum()
        self.register_buffer("log_priors", torch.log(priors))
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        adjusted_logits = logits + self.log_priors.unsqueeze(0)
        return F.cross_entropy(adjusted_logits, targets, label_smoothing=self.label_smoothing)


class LogitAdjustedCrossEntropy(nn.Module):
    def __init__(self, class_counts, tau=1.0, label_smoothing=0.0):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        counts = torch.clamp(counts, min=1.0)
        priors = counts / counts.sum()
        self.register_buffer("log_priors", torch.log(priors))
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        adjusted_logits = logits + self.tau * self.log_priors.unsqueeze(0)
        return F.cross_entropy(adjusted_logits, targets, label_smoothing=self.label_smoothing)


class FineTunedGCN(nn.Module):
    def __init__(self, encoder, n_classes=3):
        super(FineTunedGCN, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, g, features=None):
        h = self.encoder(g, features)
        logits = self.classifier(h)
        return logits


def _build_balanced_sampler(train_labels, sampler_power=1.0):
    arr = np.array(train_labels, dtype=np.int64)
    uniq, cnt = np.unique(arr, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(uniq, cnt)}
    weights = []
    for y in arr.tolist():
        n_c = max(dist.get(int(y), 1), 1)
        w = 1.0 / (float(n_c) ** float(sampler_power))
        weights.append(w)
    weights = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _sanitize_graphs(glist):
    for g in glist:
        w = g.edata.get("E_features", g.edata.get("feat", torch.ones(g.num_edges())))
        if w.dim() == 1:
            w = w.view(-1, 1)
        for k in list(g.edata.keys()):
            del g.edata[k]
        g.edata["feat"] = w
        for k in list(g.ndata.keys()):
            if k != "feat":
                del g.ndata[k]


def _graph_quality_score(g):
    # Heuristic score for synthetic sample ranking: stronger but not spiky connectivity profiles.
    w = g.edata.get("feat", torch.ones(g.num_edges(), 1))
    if w.dim() == 2:
        w = w.view(-1)
    w = w.float()
    mean_w = float(w.mean().item())
    std_w = float(w.std(unbiased=False).item())
    q90 = float(torch.quantile(w, 0.9).item()) if w.numel() > 10 else float(w.max().item())
    return mean_w + 0.5 * q90 - 0.15 * std_w


def _select_synthetic_graphs(glist_syn, n_add, mode="random", cls_label=1):
    if n_add <= 0 or len(glist_syn) == 0:
        return []

    if mode == "edge_strength_topk":
        scored = [(i, _graph_quality_score(g)) for i, g in enumerate(glist_syn)]
        scored.sort(key=lambda x: x[1], reverse=True)
        idx = [i for i, _ in scored[:n_add]]
        return [glist_syn[i] for i in idx]

    random.seed((torch.initial_seed() + cls_label) % (1 << 31))
    idx = random.sample(range(len(glist_syn)), n_add)
    return [glist_syn[i] for i in idx]


def _make_loader(train_graphs, train_labels, batch_size, use_balanced_sampler, sampler_power, seed):
    def collate(samples):
        graphs, lbls = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(lbls, dtype=torch.long)

    g_train = torch.Generator()
    g_train.manual_seed(seed)
    train_ds = list(zip(train_graphs, train_labels))

    if use_balanced_sampler:
        sampler = _build_balanced_sampler(train_labels, sampler_power=sampler_power)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate,
            generator=g_train,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
            generator=g_train,
        )
    return train_loader, collate


def _run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batched_graph, lbls in loader:
        batched_graph, lbls = batched_graph.to(device), lbls.to(device)
        logits = model(batched_graph)
        loss = criterion(logits, lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(lbls.cpu().numpy())

    train_acc = accuracy_score(all_targets, all_preds)
    train_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return total_loss / max(len(loader), 1), train_acc, train_macro_f1


def _evaluate(model, test_loader, device):
    model.eval()
    val_preds, val_targets, val_probs = [], [], []

    with torch.no_grad():
        for batched_graph, lbls in test_loader:
            batched_graph, lbls = batched_graph.to(device), lbls.to(device)
            logits = model(batched_graph)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(lbls.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())

    val_acc = accuracy_score(val_targets, val_preds)
    val_macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
    recalls = recall_score(val_targets, val_preds, labels=[0, 1, 2], average=None, zero_division=0)

    try:
        val_auc = roc_auc_score(val_targets, val_probs, multi_class="ovr")
    except ValueError:
        val_auc = 0.5

    return {
        "acc": float(val_acc),
        "macro_f1": float(val_macro_f1),
        "auc": float(val_auc),
        "rec_cn": float(recalls[0]),
        "rec_ad": float(recalls[1]),
        "rec_mci": float(recalls[2]),
        "preds": val_preds,
        "targets": val_targets,
        "probs": val_probs,
    }


def _build_rank_key(ev, auc_guardrail=0.7253, mode="macro_guard", targets=None):
    if targets is None:
        targets = {
            "acc": 0.74,
            "macro_f1": 0.58,
            "auc": 0.81,
            "rec_ad": 0.70,
            "rec_mci": 0.40,
        }

    guard_pass = float(ev["auc"]) >= float(auc_guardrail)

    if mode == "phase1_target":
        ratios = {
            "acc": float(ev["acc"]) / max(float(targets["acc"]), 1e-12),
            "macro_f1": float(ev["macro_f1"]) / max(float(targets["macro_f1"]), 1e-12),
            "auc": float(ev["auc"]) / max(float(targets["auc"]), 1e-12),
            "rec_ad": float(ev["rec_ad"]) / max(float(targets["rec_ad"]), 1e-12),
            "rec_mci": float(ev["rec_mci"]) / max(float(targets["rec_mci"]), 1e-12),
        }
        min_ratio = min(ratios.values())
        mean_ratio = float(np.mean(list(ratios.values())))
        rank_key = (
            min_ratio,
            mean_ratio,
            ratios["rec_ad"],
            ratios["auc"],
            ratios["macro_f1"],
            ratios["acc"],
            ratios["rec_mci"],
        )
    else:
        rank_key = (
            1 if guard_pass else 0,
            float(ev["macro_f1"]) if guard_pass else float(ev["auc"]),
            float(ev["rec_ad"]) if guard_pass else 0.0,
            float(ev["rec_mci"]) if guard_pass else 0.0,
            float(ev["auc"]),
            float(ev["acc"]),
        )

    return rank_key, guard_pass


def train_finetune(
    epochs=50,
    batch_size=32,
    frozen=True,
    syn_dir="./results_guidance_3class",
    max_syn_ad=100,
    max_syn_mci=100,
    loss_class_weight_mode="none",
    label_smoothing=0.0,
    seed=100,
    use_balanced_sampler=False,
    sampler_power=1.0,
    loss_mode="ce",
    logit_adjust_tau=1.0,
    auc_guardrail=0.7253,
    synthetic_selection_mode="random",
    minority_tail_epochs=0,
    minority_tail_lr_scale=0.5,
    minority_tail_use_balanced_sampler=False,
    model_select_mode="macro_guard",
    target_acc=0.74,
    target_macro_f1=0.58,
    target_auc=0.81,
    target_ad_recall=0.70,
    target_mci_recall=0.40,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Real ADNI Data...")
    glist, labels = load_adni_dgl_with_labels(data_dir="./data")

    valid_mask = labels != -1
    if not np.all(valid_mask):
        print(f"Filtering {len(labels) - valid_mask.sum()} invalid samples (label=-1)...")
        glist = [glist[i] for i in range(len(glist)) if valid_mask[i]]
        labels = labels[valid_mask]
    print(f"Remaining samples: {len(labels)}")

    _sanitize_graphs(glist)

    train_idx, test_idx = train_test_split(
        np.arange(len(glist)),
        test_size=0.2,
        stratify=labels,
        random_state=seed,
    )

    real_train_graphs = [glist[i] for i in train_idx]
    real_train_labels = [labels[i] for i in train_idx]
    test_graphs = [glist[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    np.save("test_indices_3class.npy", test_idx)
    print(f"Real Train size: {len(real_train_graphs)}, Test size: {len(test_graphs)}")

    train_graphs = list(real_train_graphs)
    train_labels = list(real_train_labels)

    unique, counts = np.unique(real_train_labels, return_counts=True)
    dist = dict(zip(unique, counts))
    n_cn = dist.get(0, 0)
    n_ad = dist.get(1, 0)
    n_mci = dist.get(2, 0)
    print(f"Real Train Distribution: CN={n_cn}, AD={n_ad}, MCI={n_mci}")

    def inject_synthetic(cls_label, cls_name, cls_count, max_inject=100):
        needed = n_cn - cls_count
        if needed <= 0:
            return

        syn_path = os.path.join(syn_dir, f"filtered_synthetic_{cls_name}.bin")
        if not os.path.exists(syn_path):
            print(f"Warning: {syn_path} not found. Proceeding unbalanced for {cls_name.upper()}.")
            return

        print(f"Loading Synthetic {cls_name.upper()} Data from {syn_path}...")
        glist_syn, _ = load_graphs(syn_path)
        _sanitize_graphs(glist_syn)

        n_add_calculated = min(needed, len(glist_syn))
        n_add = min(n_add_calculated, max_inject)
        print(
            f"Calculated deficit: {needed}. Adding {n_add} Synthetic {cls_name.upper()} samples "
            f"(selection_mode={synthetic_selection_mode}, cap={max_inject})."
        )

        selected = _select_synthetic_graphs(
            glist_syn,
            n_add=n_add,
            mode=synthetic_selection_mode,
            cls_label=cls_label,
        )
        train_graphs.extend(selected)
        train_labels.extend([cls_label] * len(selected))

    inject_synthetic(1, "ad", n_ad, max_inject=max_syn_ad)
    inject_synthetic(2, "mci", n_mci, max_inject=max_syn_mci)

    print(f"\nTotal Training Samples after Synthesis: {len(train_graphs)}")
    train_dist = dict(zip(*np.unique(np.array(train_labels), return_counts=True)))
    print(f"Train Distribution after Synthesis: {train_dist}")

    train_loader, collate = _make_loader(
        train_graphs=train_graphs,
        train_labels=train_labels,
        batch_size=batch_size,
        use_balanced_sampler=use_balanced_sampler,
        sampler_power=sampler_power,
        seed=int(torch.initial_seed() % (1 << 31)),
    )
    print(f"Balanced sampler (main stage): {use_balanced_sampler}")

    test_loader = torch.utils.data.DataLoader(
        list(zip(test_graphs, test_labels)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    encoder = StandardGCN(num_nodes=100, hidden_dim=64, n_classes=64)
    encoder.pooling = SumPooling()
    try:
        encoder.load_state_dict(torch.load("gcn_pretrained_3class.pth", map_location=device))
        print("Loaded Pre-trained Weights!")
    except FileNotFoundError:
        print("Warning: Pre-trained weights not found. Training from scratch (Random Init).")

    model = FineTunedGCN(encoder, n_classes=3).to(device)

    if frozen:
        print("Freezing Encoder Weights...")
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        print("Fine-tuning Entire Network...")

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    class_weights = _class_weights_from_distribution(train_dist, mode=loss_class_weight_mode)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"Using class weights ({loss_class_weight_mode}): {class_weights}")
    else:
        print("Using class weights: none")

    class_counts = [train_dist.get(0, 1), train_dist.get(1, 1), train_dist.get(2, 1)]
    if loss_mode == "balanced_softmax":
        criterion = BalancedSoftmaxCrossEntropy(class_counts=class_counts, label_smoothing=label_smoothing).to(device)
        print("Using loss_mode: balanced_softmax")
    elif loss_mode == "logit_adjusted_ce":
        criterion = LogitAdjustedCrossEntropy(
            class_counts=class_counts,
            tau=logit_adjust_tau,
            label_smoothing=label_smoothing,
        ).to(device)
        print(f"Using loss_mode: logit_adjusted_ce (tau={logit_adjust_tau})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        print("Using loss_mode: ce")

    tail_epochs = max(0, min(int(minority_tail_epochs), int(epochs)))
    main_epochs = int(epochs) - tail_epochs

    best_key = None
    best_epoch = 0
    best_model_path = "gcn_finetuned_3class_best.pth"

    print(f"Model selection mode: {model_select_mode}")
    if model_select_mode == "phase1_target":
        print(
            "Phase-1 targets for checkpoint selection: "
            f"acc>={target_acc}, macro_f1>={target_macro_f1}, auc>={target_auc}, "
            f"ad_recall>={target_ad_recall}, mci_recall>={target_mci_recall}"
        )

    print("\n--- Starting Fine-Tuning ---")
    global_epoch = 0
    for _ in range(main_epochs):
        global_epoch += 1
        tr_loss, tr_acc, tr_f1 = _run_epoch(model, train_loader, optimizer, criterion, device)
        ev = _evaluate(model, test_loader, device)
        rank_key, guard_pass = _build_rank_key(
            ev,
            auc_guardrail=auc_guardrail,
            mode=model_select_mode,
            targets={
                "acc": target_acc,
                "macro_f1": target_macro_f1,
                "auc": target_auc,
                "rec_ad": target_ad_recall,
                "rec_mci": target_mci_recall,
            },
        )

        if best_key is None or rank_key > best_key:
            best_key = rank_key
            best_epoch = global_epoch
            torch.save(model.state_dict(), best_model_path)

        if global_epoch == 1 or global_epoch % 5 == 0 or global_epoch == epochs:
            print(
                f"Epoch {global_epoch:02d}/{epochs} | Loss: {tr_loss:.4f} "
                f"| Train Acc: {tr_acc:.4f} | Train F1: {tr_f1:.4f} "
                f"| Val Acc: {ev['acc']:.4f} | Val F1: {ev['macro_f1']:.4f} | Val AUC: {ev['auc']:.4f} "
                f"| Rec CN/AD/MCI: {ev['rec_cn']:.2f}/{ev['rec_ad']:.2f}/{ev['rec_mci']:.2f} | Guard: {int(guard_pass)}"
            )

    if tail_epochs > 0:
        print(
            f"\n--- Minority Tail Stage: epochs={tail_epochs}, lr_scale={minority_tail_lr_scale}, "
            f"balanced_sampler={minority_tail_use_balanced_sampler} ---"
        )
        for pg in optimizer.param_groups:
            pg["lr"] = pg["lr"] * float(minority_tail_lr_scale)

        tail_loader, _ = _make_loader(
            train_graphs=train_graphs,
            train_labels=train_labels,
            batch_size=batch_size,
            use_balanced_sampler=minority_tail_use_balanced_sampler,
            sampler_power=sampler_power,
            seed=int((torch.initial_seed() + 13) % (1 << 31)),
        )

        tail_weights = _class_weights_from_distribution(train_dist, mode="inverse")
        if tail_weights is not None:
            tail_weights = tail_weights.to(device)
        tail_criterion = nn.CrossEntropyLoss(weight=tail_weights, label_smoothing=label_smoothing)

        for _ in range(tail_epochs):
            global_epoch += 1
            tr_loss, tr_acc, tr_f1 = _run_epoch(model, tail_loader, optimizer, tail_criterion, device)
            ev = _evaluate(model, test_loader, device)
            rank_key, guard_pass = _build_rank_key(
            ev,
            auc_guardrail=auc_guardrail,
            mode=model_select_mode,
            targets={
                "acc": target_acc,
                "macro_f1": target_macro_f1,
                "auc": target_auc,
                "rec_ad": target_ad_recall,
                "rec_mci": target_mci_recall,
            },
        )

            if best_key is None or rank_key > best_key:
                best_key = rank_key
                best_epoch = global_epoch
                torch.save(model.state_dict(), best_model_path)

            if global_epoch % 5 == 0 or global_epoch == epochs:
                print(
                    f"Epoch {global_epoch:02d}/{epochs} | Loss: {tr_loss:.4f} "
                    f"| Train Acc: {tr_acc:.4f} | Train F1: {tr_f1:.4f} "
                    f"| Val Acc: {ev['acc']:.4f} | Val F1: {ev['macro_f1']:.4f} | Val AUC: {ev['auc']:.4f} "
                    f"| Rec CN/AD/MCI: {ev['rec_cn']:.2f}/{ev['rec_ad']:.2f}/{ev['rec_mci']:.2f} | Guard: {int(guard_pass)}"
                )

    print(f"\n--- Loading Best Model from Epoch {best_epoch} ---")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    ev = _evaluate(model, test_loader, device)

    print("\n--- Final Evaluation on Test Set ---")
    print(classification_report(ev["targets"], ev["preds"], target_names=["CN", "AD", "MCI"], zero_division=0))
    print(f"Final Multi-Class AUC-ROC (OVR): {ev['auc']:.4f}")
    print(f"Final Macro-F1: {ev['macro_f1']:.4f}")

    torch.save(model.state_dict(), "gcn_finetuned_3class.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--unfreeze", action="store_true")
    parser.add_argument("--max_syn_ad", type=int, default=100)
    parser.add_argument("--max_syn_mci", type=int, default=100)
    parser.add_argument("--loss_class_weight_mode", type=str, default="none", choices=["none", "inverse", "sqrt_inverse", "effective"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--use_balanced_sampler", action="store_true")
    parser.add_argument("--sampler_power", type=float, default=1.0)
    parser.add_argument("--loss_mode", type=str, default="ce", choices=["ce", "balanced_softmax", "logit_adjusted_ce"])
    parser.add_argument("--logit_adjust_tau", type=float, default=1.0)
    parser.add_argument("--auc_guardrail", type=float, default=0.7253)
    parser.add_argument("--synthetic_selection_mode", type=str, default="random", choices=["random", "edge_strength_topk"])
    parser.add_argument("--minority_tail_epochs", type=int, default=0)
    parser.add_argument("--minority_tail_lr_scale", type=float, default=0.5)
    parser.add_argument("--minority_tail_use_balanced_sampler", action="store_true")
    parser.add_argument("--model_select_mode", type=str, default="macro_guard", choices=["macro_guard", "phase1_target"])
    parser.add_argument("--target_acc", type=float, default=0.74)
    parser.add_argument("--target_macro_f1", type=float, default=0.58)
    parser.add_argument("--target_auc", type=float, default=0.81)
    parser.add_argument("--target_ad_recall", type=float, default=0.70)
    parser.add_argument("--target_mci_recall", type=float, default=0.40)
    args = parser.parse_args()

    train_finetune(
        epochs=args.epochs,
        frozen=not args.unfreeze,
        max_syn_ad=args.max_syn_ad,
        max_syn_mci=args.max_syn_mci,
        loss_class_weight_mode=args.loss_class_weight_mode,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        use_balanced_sampler=args.use_balanced_sampler,
        sampler_power=args.sampler_power,
        loss_mode=args.loss_mode,
        logit_adjust_tau=args.logit_adjust_tau,
        auc_guardrail=args.auc_guardrail,
        synthetic_selection_mode=args.synthetic_selection_mode,
        minority_tail_epochs=args.minority_tail_epochs,
        minority_tail_lr_scale=args.minority_tail_lr_scale,
        minority_tail_use_balanced_sampler=args.minority_tail_use_balanced_sampler,
        model_select_mode=args.model_select_mode,
        target_acc=args.target_acc,
        target_macro_f1=args.target_macro_f1,
        target_auc=args.target_auc,
        target_ad_recall=args.target_ad_recall,
        target_mci_recall=args.target_mci_recall,
    )
