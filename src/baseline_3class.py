import argparse
import json
import os
from datetime import datetime

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data_loader_3class import load_adni_dgl_with_labels
from src.standard_gcn import StandardGCN
from src.utils import set_seed


CLASS_NAMES = ["CN", "AD", "MCI"]


def _sanitize_graphs(graphs):
    for g in graphs:
        if "E_features" in g.edata:
            w = g.edata["E_features"]
        elif "feat" in g.edata:
            w = g.edata["feat"]
        elif "w" in g.edata:
            w = g.edata["w"]
        elif "weight" in g.edata:
            w = g.edata["weight"]
        else:
            w = torch.ones(g.num_edges())
        if w.dim() == 1:
            w = w.view(-1, 1)
        for k in list(g.edata.keys()):
            del g.edata[k]
        g.edata["feat"] = w


def _collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched = dgl.batch(graphs)
    return batched, torch.tensor(labels, dtype=torch.long)


def _distribution(y):
    uniq, cnt = np.unique(y, return_counts=True)
    out = {int(k): int(v) for k, v in zip(uniq, cnt)}
    return {"CN": out.get(0, 0), "AD": out.get(1, 0), "MCI": out.get(2, 0)}


def _evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for bg, yl in loader:
            bg = bg.to(device)
            yl = yl.to(device)
            logits = model(bg)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            y_true.extend(yl.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    auc_ovr = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    auc_ovo = float(roc_auc_score(y_true, y_prob, multi_class="ovo"))

    report = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "auc_ovr": auc_ovr,
        "auc_ovo": auc_ovo,
        "per_class_precision": {"CN": float(p[0]), "AD": float(p[1]), "MCI": float(p[2])},
        "per_class_recall": {"CN": float(r[0]), "AD": float(r[1]), "MCI": float(r[2])},
        "per_class_f1": {"CN": float(f[0]), "AD": float(f[1]), "MCI": float(f[2])},
        "classification_report": report,
        "confusion_matrix": cm,
    }


def run_baseline(
    seed=100,
    epochs=80,
    batch_size=32,
    lr=1e-3,
    weight_decay=5e-4,
    test_size=0.2,
    data_dir="./data",
    output_root="./results_baseline_3class",
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"baseline_{ts}_seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 70)
    print("3-CLASS BASELINE RUN (REAL DATA ONLY, STANDARD GCN)")
    print("=" * 70)
    print(f"[Config] seed={seed}, epochs={epochs}, batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}")
    print(f"[Config] test_size={test_size}, data_dir={data_dir}")
    print("[Config] synthetic_data_used=False")
    print("[Config] contrastive_pretraining_used=False")
    print("[Config] manifold_learning_used=False, num_manifolds=0")

    glist, labels = load_adni_dgl_with_labels(data_dir=data_dir)
    valid_mask = labels != -1
    dropped = int((~valid_mask).sum())
    glist = [glist[i] for i in range(len(glist)) if valid_mask[i]]
    labels = labels[valid_mask]
    _sanitize_graphs(glist)

    print(f"[Data] total_graphs_after_label_filter={len(labels)} (dropped_invalid={dropped})")
    print(f"[Data] full_distribution={_distribution(labels)}")

    idx = np.arange(len(glist))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=seed
    )

    train_graphs = [glist[i] for i in train_idx]
    train_labels = labels[train_idx]
    test_graphs = [glist[i] for i in test_idx]
    test_labels = labels[test_idx]

    np.save(os.path.join(run_dir, "train_indices.npy"), train_idx)
    np.save(os.path.join(run_dir, "test_indices.npy"), test_idx)

    print(f"[Split] train_size={len(train_idx)}, test_size={len(test_idx)}")
    print(f"[Split] train_distribution={_distribution(train_labels)}")
    print(f"[Split] test_distribution={_distribution(test_labels)}")

    train_gen = torch.Generator()
    train_gen.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        list(zip(train_graphs, train_labels.tolist())),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        generator=train_gen,
    )
    test_loader = torch.utils.data.DataLoader(
        list(zip(test_graphs, test_labels.tolist())),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    model = StandardGCN(num_nodes=100, hidden_dim=64, n_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best = {"epoch": 0, "macro_f1": -1.0, "state": None, "metrics": None}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        y_true_tr, y_pred_tr = [], []

        for bg, yl in train_loader:
            bg = bg.to(device)
            yl = yl.to(device)

            logits = model(bg)
            loss = criterion(logits, yl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            y_true_tr.extend(yl.cpu().numpy())
            y_pred_tr.extend(logits.argmax(dim=1).cpu().numpy())

        train_acc = accuracy_score(y_true_tr, y_pred_tr)
        train_macro_f1 = f1_score(y_true_tr, y_pred_tr, average="macro", zero_division=0)

        eval_metrics = _evaluate(model, test_loader, device)

        if eval_metrics['macro_f1'] > best["macro_f1"]:
            best["epoch"] = epoch
            best["macro_f1"] = eval_metrics['macro_f1']
            best["metrics"] = eval_metrics
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[Epoch {epoch:03d}/{epochs}] "
                f"loss={running_loss/max(len(train_loader),1):.4f} "
                f"train_acc={train_acc:.4f} train_macro_f1={train_macro_f1:.4f} "
                f"test_acc={eval_metrics['accuracy']:.4f} test_macro_f1={eval_metrics['macro_f1']:.4f} "
                f"test_auc_ovr={eval_metrics['auc_ovr']:.4f}"
            )

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    final_metrics = _evaluate(model, test_loader, device)

    model_path = os.path.join(run_dir, "standard_gcn_3class_baseline_best.pth")
    torch.save(model.state_dict(), model_path)

    summary = {
        "run_type": "baseline_3class_real_only_standard_gcn",
        "timestamp": ts,
        "device": str(device),
        "config": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "test_size": test_size,
            "synthetic_data_used": False,
            "contrastive_pretraining_used": False,
            "manifold_learning_used": False,
            "num_manifolds": 0,
        },
        "data": {
            "n_total": int(len(labels)),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "full_distribution": _distribution(labels),
            "train_distribution": _distribution(train_labels),
            "test_distribution": _distribution(test_labels),
            "train_indices_file": os.path.join(run_dir, "train_indices.npy"),
            "test_indices_file": os.path.join(run_dir, "test_indices.npy"),
        },
        "selection": {
            "best_checkpoint_criterion": "test_macro_f1",
            "best_epoch": int(best["epoch"]),
            "best_epoch_metrics": best["metrics"],
        },
        "final_metrics": final_metrics,
        "artifacts": {
            "model_path": model_path,
        },
    }

    with open(os.path.join(run_dir, "baseline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(run_dir, "baseline_summary.txt"), "w") as f:
        f.write("3-Class Baseline (Real-only, StandardGCN)\n")
        f.write(f"Run dir: {run_dir}\n")
        f.write(f"Best epoch (macro-F1): {best['epoch']}\n")
        f.write(f"Final Accuracy: {final_metrics['accuracy']:.4f}\n")
        f.write(f"Final Macro-F1: {final_metrics['macro_f1']:.4f}\n")
        f.write(f"Final Weighted-F1: {final_metrics['weighted_f1']:.4f}\n")
        f.write(f"Final AUC-OVR: {final_metrics['auc_ovr']:.4f}\n")
        f.write(f"Final AUC-OVO: {final_metrics['auc_ovo']:.4f}\n")
        f.write(f"Recall CN/AD/MCI: {final_metrics['per_class_recall']['CN']:.4f}/"
                f"{final_metrics['per_class_recall']['AD']:.4f}/"
                f"{final_metrics['per_class_recall']['MCI']:.4f}\n")

    print("=" * 70)
    print("BASELINE RUN COMPLETE")
    print(f"[Output] run_dir={run_dir}")
    print(f"[Final] Accuracy={final_metrics['accuracy']:.4f} Macro-F1={final_metrics['macro_f1']:.4f} AUC-OVR={final_metrics['auc_ovr']:.4f}")
    print(f"[Final] Recall CN/AD/MCI={final_metrics['per_class_recall']['CN']:.4f}/{final_metrics['per_class_recall']['AD']:.4f}/{final_metrics['per_class_recall']['MCI']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-class baseline: StandardGCN on real ADNI data only")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_root", type=str, default="./results_baseline_3class")
    args = parser.parse_args()

    run_baseline(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        test_size=args.test_size,
        data_dir=args.data_dir,
        output_root=args.output_root,
    )
