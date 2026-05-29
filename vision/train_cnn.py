"""
Train the obstacle-perception CNN on the dataset collected by collect_data.py.

Loss:
    vis_loss : BCE on visibility logit (all samples)
    pos_loss : MSE on (x, y), VISIBLE SAMPLES ONLY (masked)
    total    : vis_loss + pos_loss

Position labels are normalised by POS_SCALE before MSE so the two losses are
comparable in magnitude; the saved model produces metres at inference via
model.predict().

Outputs:
    vision/cnn_obstacle.pt        — state_dict (weights only)
    vision/training_curves.png    — loss + val metrics across epochs
    vision/training_log.json      — same data as JSON, for the report

Run from project root:
    python3 vision/train_cnn.py [--epochs 30] [--batch 64] [--lr 1e-3]
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch
torch.set_num_threads(6) 

from model import ObstacleCNN, POS_SCALE


class VisionDataset(Dataset):
    """Wrap the .npz dataset. Returns (image_uint8_CHW, label_float)."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        # (N, 84, 84, 3) -> (N, 3, 84, 84). Keep uint8; model normalises.
        self.images = np.transpose(d["images"], (0, 3, 1, 2)).copy()
        self.labels = d["labels"].astype(np.float32)   # (N, 3): x, y, vis
        n_vis = int(self.labels[:, 2].sum())
        print(f"Loaded {len(self.images)} samples "
              f"({n_vis} visible, {len(self.images) - n_vis} not visible)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i])           # uint8 CHW
        lbl = torch.from_numpy(self.labels[i])           # float32 (3,)
        return img, lbl


def compute_loss(vis_logit, pos_norm, labels):
    """Masked loss: BCE on all, MSE on visible-only."""
    vis_target = labels[:, 2]                     # (B,)
    pos_target = labels[:, :2] / POS_SCALE        # (B, 2), normalised

    vis_loss = F.binary_cross_entropy_with_logits(
        vis_logit.squeeze(-1), vis_target
    )

    mask = vis_target > 0.5
    if mask.any():
        pos_loss = F.mse_loss(pos_norm[mask], pos_target[mask])
    else:
        # No visible samples in batch — extremely unlikely with batch=64,
        # but handle gracefully so the optimiser doesn't get a NaN.
        pos_loss = torch.tensor(0.0, device=vis_logit.device)

    return vis_loss + pos_loss, vis_loss, pos_loss


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns dict of validation metrics."""
    model.eval()
    total_vis_loss = 0.0
    total_pos_loss = 0.0
    n_batches      = 0
    n_correct_vis  = 0
    n_total        = 0
    pos_errors_m   = []   # mean abs error in METRES, visible samples only

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        vis_logit, pos_norm = model(imgs)
        _, vis_loss, pos_loss = compute_loss(vis_logit, pos_norm, labels)
        total_vis_loss += vis_loss.item()
        total_pos_loss += pos_loss.item()
        n_batches      += 1

        vis_pred = (torch.sigmoid(vis_logit.squeeze(-1)) > 0.5).float()
        n_correct_vis += (vis_pred == labels[:, 2]).sum().item()
        n_total       += len(labels)

        mask = labels[:, 2] > 0.5
        if mask.any():
            pos_m_pred   = pos_norm[mask] * POS_SCALE
            pos_m_target = labels[mask, :2]
            err_m        = (pos_m_pred - pos_m_target).abs().mean(dim=1)
            pos_errors_m.extend(err_m.cpu().tolist())

    return {
        "val_vis_loss":   total_vis_loss / n_batches,
        "val_pos_loss":   total_pos_loss / n_batches,
        "val_vis_acc":    n_correct_vis / n_total,
        "val_pos_mae_m":  float(np.mean(pos_errors_m)) if pos_errors_m else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=os.path.join(ROOT, "data", "vision_dataset.npz"))
    parser.add_argument("--out",    default=os.path.join(HERE, "cnn_obstacle.pt"))
    parser.add_argument("--curves", default=os.path.join(HERE, "training_curves.png"))
    parser.add_argument("--log",    default=os.path.join(HERE, "training_log.json"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch",  type=int, default=64)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--seed",   type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Data ───────────────────────────────────────────────────────────
    full = VisionDataset(args.data)
    n_val = max(1, len(full) // 10)
    n_train = len(full) - n_val
    train_set, val_set = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  train={n_train}  val={n_val}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch, shuffle=False, num_workers=0)

    # ── Model / optimiser ──────────────────────────────────────────────
    model = ObstacleCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Training loop ──────────────────────────────────────────────────
    history = {
        "epoch": [], "train_loss": [], "train_vis_loss": [], "train_pos_loss": [],
        "val_vis_loss": [], "val_pos_loss": [], "val_vis_acc": [], "val_pos_mae_m": [],
    }
    best_mae = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = ep_vis = ep_pos = 0.0
        n_batches = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            vis_logit, pos_norm = model(imgs)
            loss, vis_loss, pos_loss = compute_loss(vis_logit, pos_norm, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            ep_vis  += vis_loss.item()
            ep_pos  += pos_loss.item()
            n_batches += 1

        train_loss = ep_loss / n_batches
        val_metrics = evaluate(model, val_loader, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_vis_loss"].append(ep_vis / n_batches)
        history["train_pos_loss"].append(ep_pos / n_batches)
        for k, v in val_metrics.items():
            history[k].append(v)

        print(f"[epoch {epoch:>3}/{args.epochs}]  "
              f"train_loss={train_loss:.4f}  "
              f"val_vis_acc={val_metrics['val_vis_acc']:.3f}  "
              f"val_pos_mae={val_metrics['val_pos_mae_m']:.3f} m  "
              f"({time.time() - t0:.0f}s elapsed)")

        # Save best model by position MAE on visible samples (the
        # headline metric we report)
        if val_metrics["val_pos_mae_m"] < best_mae:
            best_mae = val_metrics["val_pos_mae_m"]
            torch.save(model.state_dict(), args.out)

    print(f"\nBest val pos MAE: {best_mae:.3f} m")
    print(f"Saved weights to {args.out}")

    # ── Training curves ────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping training curves")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(history["epoch"], history["train_loss"],     label="train total")
        axes[0].plot(history["epoch"], history["train_vis_loss"], label="train vis (BCE)")
        axes[0].plot(history["epoch"], history["train_pos_loss"], label="train pos (MSE)")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[0].set_title("Training losses")

        axes[1].plot(history["epoch"], history["val_vis_acc"])
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
        axes[1].set_ylim(0, 1.01); axes[1].grid(alpha=0.3)
        axes[1].set_title("Val visibility accuracy")

        axes[2].plot(history["epoch"], history["val_pos_mae_m"])
        axes[2].set_xlabel("epoch"); axes[2].set_ylabel("MAE (m)")
        axes[2].grid(alpha=0.3)
        axes[2].set_title("Val position MAE (visible samples)")

        plt.tight_layout()
        plt.savefig(args.curves, dpi=120)
        print(f"Saved curves to {args.curves}")

    with open(args.log, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved log to {args.log}")


if __name__ == "__main__":
    main()