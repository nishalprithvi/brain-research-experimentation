import argparse
import csv
import datetime
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.data_loader_3class import load_adni_data
from src.diffusion_model import DiffusionUNet
from src.gcn_model import LatentDenseGCN, LatentMLPTeacher
from src.vae_model import VAE


def _matrix_ssim_like(x, y, c1=1e-4, c2=9e-4):
    # x: (B, H, W), y: (B, H, W)
    x_flat = x.view(x.shape[0], -1)
    y_flat = y.view(y.shape[0], -1)

    mu_x = x_flat.mean(dim=1)
    mu_y = y_flat.mean(dim=1)
    var_x = x_flat.var(dim=1, unbiased=False)
    var_y = y_flat.var(dim=1, unbiased=False)
    cov_xy = ((x_flat - mu_x.unsqueeze(1)) * (y_flat - mu_y.unsqueeze(1))).mean(dim=1)

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    den = (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
    return (num / den).clamp(min=-1.0, max=1.0)


def _pearson_corr_per_sample(a, b):
    # a, b: (B, D)
    a_center = a - a.mean(dim=1, keepdim=True)
    b_center = b - b.mean(dim=1, keepdim=True)
    num = (a_center * b_center).sum(dim=1)
    den = (a_center.pow(2).sum(dim=1).sqrt() * b_center.pow(2).sum(dim=1).sqrt()) + 1e-8
    return num / den


def _expected_calibration_error(probs, labels, n_bins=10):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.tensor(0.0, device=probs.device)
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin
    return float(ece.item())


def _brier_multiclass(probs, labels, n_classes):
    y_one_hot = F.one_hot(labels, num_classes=n_classes).float()
    return float(((probs - y_one_hot).pow(2).sum(dim=1).mean()).item())


def _mean_entropy(probs):
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
    return float(entropy.mean().item())


def _to_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def _eval_vae(model, dataloader, device, num_classes=3):
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    ssim_sum = 0.0
    count = 0

    per_class_mse_sum = [0.0] * num_classes
    per_class_count = [0] * num_classes

    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            y = y.to(device)
            recon, _, _, _ = model(x)
            recon_m = recon.squeeze(1)

            per_sample_mse = ((recon_m - x) ** 2).mean(dim=(1, 2))
            per_sample_mae = (recon_m - x).abs().mean(dim=(1, 2))
            per_sample_ssim = _matrix_ssim_like(x, recon_m)

            mse_sum += float(per_sample_mse.sum().item())
            mae_sum += float(per_sample_mae.sum().item())
            ssim_sum += float(per_sample_ssim.sum().item())
            count += x.shape[0]

            for cls in range(num_classes):
                cls_mask = y == cls
                if cls_mask.any():
                    cls_mse = per_sample_mse[cls_mask]
                    per_class_mse_sum[cls] += float(cls_mse.sum().item())
                    per_class_count[cls] += int(cls_mask.sum().item())

    out = {
        "val_mse": mse_sum / max(count, 1),
        "val_mae": mae_sum / max(count, 1),
        "val_ssim_like": ssim_sum / max(count, 1),
    }
    for cls in range(num_classes):
        denom = max(per_class_count[cls], 1)
        out[f"val_mse_class_{cls}"] = per_class_mse_sum[cls] / denom
    return out


def _latent_stats(vae, train_loader, val_loader, device):
    vae.eval()
    train_lat, train_lab, val_lat, val_lab = [], [], [], []

    with torch.no_grad():
        for x, y in train_loader:
            x = x.float().to(device)
            _, mu, _, _ = vae(x)
            train_lat.append(mu.view(mu.shape[0], -1).cpu().numpy())
            train_lab.append(y.numpy())
        for x, y in val_loader:
            x = x.float().to(device)
            _, mu, _, _ = vae(x)
            val_lat.append(mu.view(mu.shape[0], -1).cpu().numpy())
            val_lab.append(y.numpy())

    train_lat = np.concatenate(train_lat, axis=0)
    train_lab = np.concatenate(train_lab, axis=0)
    val_lat = np.concatenate(val_lat, axis=0)
    val_lab = np.concatenate(val_lab, axis=0)

    combined_lat = np.concatenate([train_lat, val_lat], axis=0)
    combined_lab = np.concatenate([train_lab, val_lab], axis=0)

    silhouette = None
    if len(np.unique(combined_lab)) > 1:
        try:
            silhouette = float(silhouette_score(combined_lat, combined_lab))
        except Exception:
            silhouette = None

    train_mean = train_lat.mean(axis=0)
    val_mean = val_lat.mean(axis=0)
    train_cov = np.cov(train_lat, rowvar=False)
    val_cov = np.cov(val_lat, rowvar=False)

    return {
        "latent_silhouette": silhouette,
        "latent_mean_drift_l2": float(np.linalg.norm(train_mean - val_mean)),
        "latent_cov_fro_drift": float(np.linalg.norm(train_cov - val_cov, ord="fro")),
        "latent_train_cov_trace": float(np.trace(train_cov)),
        "latent_val_cov_trace": float(np.trace(val_cov)),
        "_val_cov": val_cov,
    }


def train_vae(
    model,
    train_loader,
    val_loader,
    epochs=50,
    eval_every=5,
    aux_cls_weight=0.0,
    device="cpu",
):
    aux_head = nn.Linear(256, 3).to(device) if aux_cls_weight > 0 else None
    params = list(model.parameters()) + (list(aux_head.parameters()) if aux_head is not None else [])
    optimizer = optim.Adam(params, lr=1e-3)
    metrics = []
    prev_val_cov = None
    aux_criterion = None
    if aux_head is not None:
        train_labels = train_loader.dataset.tensors[1].cpu().numpy()
        dist = {int(c): int((train_labels == c).sum()) for c in np.unique(train_labels)}
        class_w = _class_weights_from_distribution(dist, mode="sqrt_inverse").to(device)
        class_w = torch.clamp(class_w, max=2.0)
        aux_criterion = nn.CrossEntropyLoss(weight=class_w)
        print(f"[VAE] Aux latent classifier enabled (weight={aux_cls_weight}, class_w={class_w})")

    print(f"\n[VAE] Starting Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.float().to(device)
            y = y.long().to(device)
            optimizer.zero_grad()
            recon, mu, logvar, _ = model(x)
            recon_loss = F.mse_loss(recon, x.unsqueeze(1), reduction="sum")
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kld_loss
            if aux_head is not None:
                aux_logits = aux_head(mu.view(mu.shape[0], -1))
                aux_loss = aux_criterion(aux_logits, y)
                loss = loss + aux_cls_weight * aux_loss * x.shape[0]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        row = {
            "epoch": epoch + 1,
            "train_loss_per_sample": total_loss / len(train_loader.dataset),
        }

        if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
            recon_metrics = _eval_vae(model, val_loader, device)
            latent_metrics = _latent_stats(model, train_loader, val_loader, device)
            val_cov = latent_metrics.pop("_val_cov")

            if prev_val_cov is None:
                latent_metrics["latent_cov_epoch_stability"] = 0.0
            else:
                latent_metrics["latent_cov_epoch_stability"] = float(
                    np.linalg.norm(val_cov - prev_val_cov, ord="fro")
                )
            prev_val_cov = val_cov

            row.update(recon_metrics)
            row.update(latent_metrics)

            print(
                "[VAE] Epoch "
                f"{epoch+1}/{epochs} | TrainLoss: {row['train_loss_per_sample']:.4f} "
                f"| ValMSE: {row['val_mse']:.6f} | ValMAE: {row['val_mae']:.6f} "
                f"| SSIM-like: {row['val_ssim_like']:.4f} | LatentSil: {row.get('latent_silhouette')}"
            )

        metrics.append(row)

    return model, metrics


def train_diffusion(
    unet,
    vae,
    train_loader,
    epochs=100,
    num_buckets=10,
    device="cpu",
):
    optimizer = optim.Adam(unet.parameters(), lr=1e-4)
    unet.train()
    vae.eval()
    criterion = nn.MSELoss()
    metrics = []

    print(f"\n[Diffusion] Starting Training for {epochs} epochs...")

    timesteps = 1000
    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for epoch in range(epochs):
        total_loss = 0.0
        bucket_mse_sum = np.zeros(num_buckets, dtype=np.float64)
        bucket_corr_sum = np.zeros(num_buckets, dtype=np.float64)
        bucket_count = np.zeros(num_buckets, dtype=np.int64)

        for x, _ in train_loader:
            x = x.float().to(device)
            with torch.no_grad():
                _, _, _, z = vae(x)

            t = torch.randint(0, timesteps, (z.shape[0],), device=device).long()
            noise = torch.randn_like(z)

            sqrt_ac = torch.sqrt(alphas_cumprod)[t].view(-1, 1, 1, 1)
            sqrt_one_minus_ac = torch.sqrt(1 - alphas_cumprod)[t].view(-1, 1, 1, 1)
            noisy_z = sqrt_ac * z + sqrt_one_minus_ac * noise

            optimizer.zero_grad()
            noise_pred = unet(noisy_z, t)
            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                per_sample_mse = ((noise_pred - noise) ** 2).view(z.shape[0], -1).mean(dim=1)
                pred_flat = noise_pred.view(z.shape[0], -1)
                noise_flat = noise.view(z.shape[0], -1)
                per_sample_corr = _pearson_corr_per_sample(pred_flat, noise_flat)
                buckets = torch.clamp((t * num_buckets // timesteps), max=num_buckets - 1)

                for idx in range(z.shape[0]):
                    b = int(buckets[idx].item())
                    bucket_mse_sum[b] += float(per_sample_mse[idx].item())
                    bucket_corr_sum[b] += float(per_sample_corr[idx].item())
                    bucket_count[b] += 1

        row = {
            "epoch": epoch + 1,
            "train_loss_per_sample": total_loss / len(train_loader.dataset),
        }

        for b in range(num_buckets):
            if bucket_count[b] > 0:
                row[f"bucket_{b}_mse"] = bucket_mse_sum[b] / bucket_count[b]
                row[f"bucket_{b}_noise_corr"] = bucket_corr_sum[b] / bucket_count[b]
            else:
                row[f"bucket_{b}_mse"] = None
                row[f"bucket_{b}_noise_corr"] = None

        metrics.append(row)
        if (epoch + 1) % 10 == 0:
            print(
                f"[Diffusion] Epoch {epoch+1}/{epochs} | TrainLoss: {row['train_loss_per_sample']:.6f} "
                f"| EarlyBucketMSE: {row['bucket_0_mse']:.6f} | LateBucketMSE: {row[f'bucket_{num_buckets-1}_mse']:.6f}"
            )

    return unet, metrics


def _class_weights_from_distribution(dist, mode, beta=0.999):
    weights = []
    for c in range(3):
        n_c = dist.get(c, 0)
        if n_c <= 0:
            weights.append(1.0)
            continue
        if mode == "effective":
            eff_num = 1.0 - (beta ** n_c)
            w = (1.0 - beta) / max(eff_num, 1e-12)
            weights.append(w)
        elif mode == "sqrt_inverse":
            weights.append(1.0 / np.sqrt(float(n_c)))
        else:
            weights.append(1.0 / float(n_c))
    weights = np.array(weights, dtype=np.float32)
    weights = weights / max(weights.min(), 1e-12)
    return torch.tensor(weights, dtype=torch.float32)


def _fit_temperature(logits, labels, device):
    temperature = torch.ones(1, device=device, requires_grad=True)
    optimizer = optim.LBFGS([temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature.clamp_min(1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp_min(1e-3).item())


def train_latent_gcn(
    gcn,
    vae,
    train_loader,
    val_loader,
    epochs=100,
    class_weight_mode="none",
    use_balanced_sampler=False,
    loss_mode="ce",
    max_class_weight=0.0,
    collapse_reg_strength=0.05,
    early_stop_patience=20,
    device="cpu",
):
    vae.eval()
    print("\n[Latent GCN] Extracting Latents for train/val...")

    def extract(loader):
        latents, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.float().to(device)
                # Use deterministic mean-latent (mu) for teacher training stability.
                # Stochastic z increases class overlap and can cause collapse.
                _, mu, _, _ = vae(x)
                latents.append(mu.view(-1, 1, 16, 16).cpu())
                labels.append(y.cpu())
        return torch.cat(latents, dim=0), torch.cat(labels, dim=0)

    X_train, y_train = extract(train_loader)
    X_val, y_val = extract(val_loader)

    unique, counts = torch.unique(y_train, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"[Latent GCN] Class Distribution (train): {dist}")

    train_ds = TensorDataset(X_train.to(device), y_train.to(device))
    val_ds = TensorDataset(X_val.to(device), y_val.to(device))

    if use_balanced_sampler:
        sample_weights = []
        for y_item in y_train.tolist():
            sample_weights.append(1.0 / max(dist.get(int(y_item), 1), 1))
        sample_weights = torch.tensor(sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader_lat = DataLoader(train_ds, batch_size=16, sampler=sampler)
        print("[Latent GCN] Balanced sampler: enabled")
    else:
        train_loader_lat = DataLoader(train_ds, batch_size=16, shuffle=True)
        print("[Latent GCN] Balanced sampler: disabled")
    val_loader_lat = DataLoader(val_ds, batch_size=16, shuffle=False)

    weights = None
    if class_weight_mode != "none":
        weights = _class_weights_from_distribution(dist, class_weight_mode).to(device)
        if max_class_weight > 0:
            weights = torch.clamp(weights, max=max_class_weight)
        print(f"[Latent GCN] Using class weights ({class_weight_mode}): {weights}")
    else:
        print("[Latent GCN] Using class weights: none")
    print(f"[Latent GCN] Loss mode: {loss_mode}")

    optimizer = optim.Adam(gcn.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05, reduction="none")

    best_macro_f1 = -1.0
    best_epoch = -1
    best_payload = {}
    wait = 0
    metrics = []

    print(f"[Latent GCN] Starting Training for {epochs} epochs...")
    for epoch in range(epochs):
        gcn.train()
        total_loss = 0.0

        for xb, yb in train_loader_lat:
            optimizer.zero_grad()
            outputs = gcn(xb)
            ce_per_sample = criterion(outputs, yb)
            if loss_mode == "class_balanced_ce":
                class_losses = []
                for c in [0, 1, 2]:
                    mask = (yb == c)
                    if torch.any(mask):
                        class_losses.append(ce_per_sample[mask].mean())
                ce_loss = torch.stack(class_losses).mean() if class_losses else ce_per_sample.mean()
            else:
                ce_loss = ce_per_sample.mean()
            probs = torch.softmax(outputs, dim=1)
            mean_probs = probs.mean(dim=0)
            # Anti-collapse regularizer: encourage non-degenerate class usage.
            # Keeps optimization honest without changing task labels.
            reg = torch.sum(mean_probs * torch.log(mean_probs * 3.0 + 1e-8))
            loss = ce_loss + collapse_reg_strength * reg
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        gcn.eval()
        val_logits, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader_lat:
                logits = gcn(xb)
                val_logits.append(logits)
                val_targets.append(yb)

        val_logits = torch.cat(val_logits, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_probs = torch.softmax(val_logits, dim=1)
        val_preds = val_probs.argmax(dim=1)

        val_targets_np = val_targets.cpu().numpy()
        val_preds_np = val_preds.cpu().numpy()

        macro_f1 = float(f1_score(val_targets_np, val_preds_np, average="macro", zero_division=0))
        per_class_recall = recall_score(
            val_targets_np,
            val_preds_np,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )

        ece = _expected_calibration_error(val_probs, val_targets)
        brier = _brier_multiclass(val_probs, val_targets, n_classes=3)
        entropy = _mean_entropy(val_probs)
        cm = confusion_matrix(val_targets_np, val_preds_np, labels=[0, 1, 2])

        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(len(train_loader_lat), 1),
            "val_macro_f1": macro_f1,
            "val_recall_class_0": float(per_class_recall[0]),
            "val_recall_class_1": float(per_class_recall[1]),
            "val_recall_class_2": float(per_class_recall[2]),
            "val_ece": ece,
            "val_brier": brier,
            "val_entropy": entropy,
        }
        metrics.append(row)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch + 1
            wait = 0
            torch.save(gcn.state_dict(), "gcn_3class.pth")

            temp = _fit_temperature(val_logits.detach(), val_targets.detach(), device)
            cal_probs = torch.softmax(val_logits / temp, dim=1)
            best_payload = {
                "best_epoch": best_epoch,
                "best_macro_f1": best_macro_f1,
                "best_recall_class_0": float(per_class_recall[0]),
                "best_recall_class_1": float(per_class_recall[1]),
                "best_recall_class_2": float(per_class_recall[2]),
                "best_ece": ece,
                "best_brier": brier,
                "best_entropy": entropy,
                "temperature": temp,
                "best_calibrated_ece": _expected_calibration_error(cal_probs, val_targets),
                "best_calibrated_brier": _brier_multiclass(cal_probs, val_targets, n_classes=3),
                "best_confusion_matrix": cm.tolist(),
            }
        else:
            wait += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"[Latent GCN] Epoch {epoch+1}/{epochs} | Loss: {row['train_loss']:.4f} | "
                f"Macro-F1: {macro_f1:.4f} | Rec(AD): {row['val_recall_class_1']:.4f} | Rec(MCI): {row['val_recall_class_2']:.4f}"
            )

        if wait >= early_stop_patience:
            print(
                f"[Latent GCN] Early stopping triggered at epoch {epoch+1}; "
                f"best epoch={best_epoch}, best macro-F1={best_macro_f1:.4f}."
            )
            break

    gcn.load_state_dict(torch.load("gcn_3class.pth", map_location=device, weights_only=True))
    return gcn, metrics, best_payload


def _phase1_stability_summary(vae_metrics, diffusion_metrics, teacher_metrics):
    out = {}

    teacher_f1 = [row["val_macro_f1"] for row in teacher_metrics if row.get("val_macro_f1") is not None]
    if teacher_f1:
        out["teacher_macro_f1_std"] = float(np.std(teacher_f1))
        last_k = teacher_f1[-5:] if len(teacher_f1) >= 5 else teacher_f1
        out["teacher_macro_f1_lastk_std"] = float(np.std(last_k))
        out["teacher_macro_f1_lastk_range"] = float(np.max(last_k) - np.min(last_k))

    vae_mse = [row["val_mse"] for row in vae_metrics if row.get("val_mse") is not None]
    if vae_mse:
        last_k = vae_mse[-5:] if len(vae_mse) >= 5 else vae_mse
        out["vae_val_mse_lastk_std"] = float(np.std(last_k))

    diff_loss = [row["train_loss_per_sample"] for row in diffusion_metrics]
    if diff_loss:
        last_k = diff_loss[-5:] if len(diff_loss) >= 5 else diff_loss
        out["diffusion_loss_lastk_std"] = float(np.std(last_k))

    return out


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    matrices, labels = load_adni_data(data_dir="./data")

    valid_mask = labels != -1
    num_invalid = int(len(labels) - valid_mask.sum())
    if num_invalid > 0:
        print(f"[DataLoader] Filtering out {num_invalid} samples with invalid label (-1).")
        matrices = matrices[valid_mask]
        labels = labels[valid_mask]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(matrices, labels))

    x_train = torch.from_numpy(matrices[train_idx])
    y_train = torch.from_numpy(labels[train_idx]).long()
    x_val = torch.from_numpy(matrices[val_idx])
    y_val = torch.from_numpy(labels[val_idx]).long()

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    quality_dir = os.path.join(args.phase1_quality_dir, f"phase1_{run_id}_seed_{args.seed}")
    os.makedirs(quality_dir, exist_ok=True)

    run_meta = {
        "seed": args.seed,
        "epochs_vae": args.epochs_vae,
        "epochs_diff": args.epochs_diff,
        "epochs_gcn": args.epochs_gcn,
        "batch_size": args.batch_size,
        "vae_aux_cls_weight": float(args.vae_aux_cls_weight),
        "quality_eval_every": args.quality_eval_every,
        "diffusion_num_buckets": args.diffusion_num_buckets,
        "teacher_model_type": args.teacher_model_type,
        "teacher_class_weight_mode": args.teacher_class_weight_mode,
        "teacher_use_balanced_sampler": bool(args.teacher_use_balanced_sampler),
        "teacher_loss_mode": args.teacher_loss_mode,
        "teacher_max_class_weight": float(args.teacher_max_class_weight),
        "teacher_collapse_reg": float(args.teacher_collapse_reg),
        "teacher_early_stop_patience": args.teacher_early_stop_patience,
        "num_samples": int(len(labels)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "quality_dir": quality_dir,
    }

    with open(os.path.join(quality_dir, "phase1_run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    # 1. Train VAE + quality
    vae = VAE(input_dim=(100, 100), latent_dim=256).to(device)
    vae, vae_metrics = train_vae(
        vae,
        train_loader,
        val_loader,
        epochs=args.epochs_vae,
        eval_every=args.quality_eval_every,
        aux_cls_weight=args.vae_aux_cls_weight,
        device=device,
    )
    torch.save(vae.state_dict(), "vae_3class.pth")

    # 2. Train Diffusion + quality
    unet = DiffusionUNet().to(device)
    unet, diffusion_metrics = train_diffusion(
        unet,
        vae,
        train_loader,
        epochs=args.epochs_diff,
        num_buckets=args.diffusion_num_buckets,
        device=device,
    )
    torch.save(unet.state_dict(), "diffusion_3class.pth")

    # 3. Train Latent Dense GCN teacher + quality
    if args.teacher_model_type == "latent_mlp":
        gcn = LatentMLPTeacher(latent_dim=256, hidden_dim=128, n_classes=3, dropout=0.2).to(device)
    else:
        gcn = LatentDenseGCN(num_nodes=16, in_features=16, hidden_dim=32, n_classes=3).to(device)
    gcn, teacher_metrics, teacher_best = train_latent_gcn(
        gcn,
        vae,
        train_loader,
        val_loader,
        epochs=args.epochs_gcn,
        class_weight_mode=args.teacher_class_weight_mode,
        use_balanced_sampler=args.teacher_use_balanced_sampler,
        loss_mode=args.teacher_loss_mode,
        max_class_weight=args.teacher_max_class_weight,
        collapse_reg_strength=args.teacher_collapse_reg,
        early_stop_patience=args.teacher_early_stop_patience,
        device=device,
    )

    stability = _phase1_stability_summary(vae_metrics, diffusion_metrics, teacher_metrics)

    with open(os.path.join(quality_dir, "vae_quality_metrics.json"), "w") as f:
        json.dump(vae_metrics, f, indent=2)
    with open(os.path.join(quality_dir, "diffusion_quality_metrics.json"), "w") as f:
        json.dump(diffusion_metrics, f, indent=2)
    with open(os.path.join(quality_dir, "teacher_quality_metrics.json"), "w") as f:
        json.dump(teacher_metrics, f, indent=2)

    _to_csv(os.path.join(quality_dir, "vae_quality_metrics.csv"), vae_metrics)
    _to_csv(os.path.join(quality_dir, "diffusion_quality_metrics.csv"), diffusion_metrics)
    _to_csv(os.path.join(quality_dir, "teacher_quality_metrics.csv"), teacher_metrics)

    summary = {
        "teacher_best": teacher_best,
        "stability": stability,
        "latest_vae": vae_metrics[-1] if vae_metrics else {},
        "latest_diffusion": diffusion_metrics[-1] if diffusion_metrics else {},
    }
    with open(os.path.join(quality_dir, "phase1_quality_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if teacher_best.get("best_confusion_matrix") is not None:
        np.save(
            os.path.join(quality_dir, "teacher_best_confusion_matrix.npy"),
            np.array(teacher_best["best_confusion_matrix"], dtype=np.int64),
        )

    print("[Phase 1 Quality] Saved quality artifacts to:", quality_dir)
    print("[Phase 1 Quality] Teacher best macro-F1:", teacher_best.get("best_macro_f1"))
    print("Pipeline Complete Phase 1.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_vae", type=int, default=50)
    parser.add_argument("--epochs_diff", type=int, default=100)
    parser.add_argument("--epochs_gcn", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--vae_aux_cls_weight", type=float, default=0.0)
    parser.add_argument("--quality_eval_every", type=int, default=5)
    parser.add_argument("--phase1_quality_dir", type=str, default="./results_phase1_quality")
    parser.add_argument("--diffusion_num_buckets", type=int, default=10)
    parser.add_argument(
        "--teacher_model_type",
        type=str,
        default="latent_densegcn",
        choices=["latent_densegcn", "latent_mlp"],
    )
    parser.add_argument(
        "--teacher_class_weight_mode",
        type=str,
        default="none",
        choices=["none", "inverse", "sqrt_inverse", "effective"],
    )
    parser.add_argument("--teacher_use_balanced_sampler", action="store_true")
    parser.add_argument(
        "--teacher_loss_mode",
        type=str,
        default="ce",
        choices=["ce", "class_balanced_ce"],
    )
    parser.add_argument("--teacher_max_class_weight", type=float, default=0.0)
    parser.add_argument("--teacher_collapse_reg", type=float, default=0.05)
    parser.add_argument("--teacher_early_stop_patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
