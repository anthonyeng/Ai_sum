"""
Train the Temporal Transformer on TVSum50 for video segment importance scoring.

Uses the same data pipeline, splits, and evaluation as train_lstm.py
for fair comparison between BiLSTM and Transformer architectures.

Usage:
    python src/training/train_temporal_transformer.py

Output:
    outputs/models/temporal_transformer.pt
"""

import os
import sys
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.temporal_transformer import TemporalTransformer

# ── Paths (same as train_lstm.py) ────────────────────────────────────────────
FEATURE_DIR = "data/processed/features"
ANNO_FILE = "data/raw/tvsum/ydata-tvsum50-anno.tsv"
MODEL_OUT = "outputs/models/temporal_transformer.pt"

# ── Hyperparameters ───────────────────────────────────────────────────────────
TEST_RATIO = 0.2
RANDOM_SEED = 42

D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
LR = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 80
BATCH_SIZE = 4
PATIENCE = 15
WARMUP_EPOCHS = 5

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ── Data loading (identical to train_lstm.py) ─────────────────────────────────

def load_annotations():
    df = pd.read_csv(
        ANNO_FILE, sep="\t", header=None,
        names=["video_id", "category", "scores"],
    )
    annotations = {}
    for _, row in df.iterrows():
        video_id = str(row["video_id"]).strip()
        scores = np.array(
            [float(x) for x in str(row["scores"]).split(",") if x],
            dtype=np.float32,
        )
        annotations[video_id] = scores
    return annotations


def get_available_videos(feature_dir, annotations):
    videos = []
    for fname in os.listdir(feature_dir):
        if not fname.endswith(".npy"):
            continue
        vid = fname.replace(".npy", "")
        if vid in annotations:
            videos.append(vid)
    videos.sort()
    return videos


def split_videos(video_ids, test_ratio=0.2, seed=42):
    rng = random.Random(seed)
    shuffled = video_ids[:]
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_ratio))
    return shuffled[n_test:], shuffled[:n_test]


# ── Dataset ───────────────────────────────────────────────────────────────────

class VideoDataset(Dataset):
    def __init__(self, video_ids, feature_dir, annotations):
        self.samples = []

        for vid in video_ids:
            feat_path = os.path.join(feature_dir, f"{vid}.npy")
            if not os.path.exists(feat_path):
                print(f"[SKIP] {vid}")
                continue

            features = np.load(feat_path)
            scores = annotations[vid]

            min_len = min(len(features), len(scores))
            features = features[:min_len]
            scores = scores[:min_len]

            if len(features) == 0:
                continue

            # Normalize scores to [0, 1] (original range is 1-5)
            scores = (scores - 1.0) / 4.0

            self.samples.append((
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(scores, dtype=torch.float32),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    features, scores = zip(*batch)
    lengths = torch.tensor([len(f) for f in features], dtype=torch.long)
    max_len = lengths.max().item()

    padded_features = torch.zeros(len(batch), max_len, features[0].shape[-1])
    padded_scores = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, (f, s) in enumerate(zip(features, scores)):
        padded_features[i, :len(f)] = f
        padded_scores[i, :len(s)] = s
        mask[i, :len(f)] = True

    return padded_features, padded_scores, lengths, mask


# ── Learning rate warmup scheduler ────────────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup + cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for features, scores, lengths, mask in loader:
        features = features.to(DEVICE)
        scores = scores.to(DEVICE)
        mask = mask.to(DEVICE)

        preds = model(features, mask)

        loss = criterion(preds[mask], scores[mask])

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        n_samples += mask.sum().item()

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []

    for features, scores, lengths, mask in loader:
        features = features.to(DEVICE)
        scores = scores.to(DEVICE)
        mask = mask.to(DEVICE)

        preds = model(features, mask)

        loss = criterion(preds[mask], scores[mask])
        total_loss += loss.item() * mask.sum().item()
        n_samples += mask.sum().item()

        # Rescale back to [1, 5] for metrics
        p = preds[mask].cpu().numpy() * 4.0 + 1.0
        t = scores[mask].cpu().numpy() * 4.0 + 1.0
        all_preds.append(p)
        all_targets.append(t)

    avg_loss = total_loss / max(n_samples, 1)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    spearman, _ = spearmanr(all_targets, all_preds)

    return avg_loss, mse, mae, r2, spearman


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # MLflow tracking
    from src.tracking.mlflow_utils import ExperimentTracker
    tracker = ExperimentTracker("video_summarization")
    tracker.start_run("temporal_transformer", tags={"model_type": "transformer", "dataset": "tvsum50"})
    tracker.log_params({
        "model": "TemporalTransformer",
        "d_model": D_MODEL, "nhead": NHEAD, "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD, "dropout": DROPOUT,
        "lr": LR, "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "patience": PATIENCE, "warmup_epochs": WARMUP_EPOCHS,
        "seed": RANDOM_SEED,
    })

    print("=" * 60)
    print("TEMPORAL TRANSFORMER — Video Importance Scoring")
    print("=" * 60)

    print("\nLoading annotations...")
    annotations = load_annotations()

    print("Finding available videos...")
    video_ids = get_available_videos(FEATURE_DIR, annotations)
    print(f"Total videos: {len(video_ids)}")

    train_vids, test_vids = split_videos(video_ids, TEST_RATIO, RANDOM_SEED)
    print(f"Train: {len(train_vids)} videos, Test: {len(test_vids)} videos")

    train_set = VideoDataset(train_vids, FEATURE_DIR, annotations)
    test_set = VideoDataset(test_vids, FEATURE_DIR, annotations)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
    )

    print(f"\nTrain samples: {len(train_set)}, Test samples: {len(test_set)}")
    print(f"Device: {DEVICE}")

    model = TemporalTransformer(
        input_dim=512,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print(f"\nHyperparameters:")
    print(f"  d_model={D_MODEL}, nhead={NHEAD}, layers={NUM_LAYERS}, ffn={DIM_FEEDFORWARD}")
    print(f"  lr={LR}, weight_decay={WEIGHT_DECAY}, dropout={DROPOUT}")
    print(f"  epochs={EPOCHS}, batch_size={BATCH_SIZE}, patience={PATIENCE}")
    print(f"  warmup_epochs={WARMUP_EPOCHS}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'MSE':>8} | {'MAE':>8} | {'R²':>8} | {'Spearman':>8} | {'LR':>10}")
    print("-" * 90)

    for epoch in range(1, EPOCHS + 1):
        scheduler.step(epoch - 1)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_mse, val_mae, val_r2, val_spearman = evaluate(
            model, test_loader, criterion
        )

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:6d} | {train_loss:10.4f} | {val_loss:10.4f} | "
            f"{val_mse:8.4f} | {val_mae:8.4f} | {val_r2:8.4f} | "
            f"{val_spearman:8.4f} | {lr:10.6f}"
        )

        # Log to MLflow
        tracker.log_metrics({
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "mse": round(val_mse, 4),
            "mae": round(val_mae, 4),
            "r2": round(val_r2, 4),
            "spearman": round(val_spearman, 4),
            "lr": lr,
        }, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": {
                    "input_dim": 512,
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_layers": NUM_LAYERS,
                    "dim_feedforward": DIM_FEEDFORWARD,
                    "dropout": DROPOUT,
                },
                "metrics": {
                    "mse": val_mse,
                    "mae": val_mae,
                    "r2": val_r2,
                    "spearman": val_spearman,
                },
            }, MODEL_OUT)
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Final evaluation with best model
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (best checkpoint)")
    print("=" * 60)

    ckpt = torch.load(MODEL_OUT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, mse, mae, r2, spearman = evaluate(model, test_loader, criterion)

    print(f"MSE:      {mse:.4f}")
    print(f"MAE:      {mae:.4f}")
    print(f"R²:       {r2:.4f}")
    print(f"Spearman: {spearman:.4f}")
    print(f"Epoch:    {ckpt['epoch']}")
    print(f"\nModel saved to: {MODEL_OUT}")

    # Log final metrics and model to MLflow
    tracker.log_metrics({
        "final_mse": round(mse, 4),
        "final_mae": round(mae, 4),
        "final_r2": round(r2, 4),
        "final_spearman": round(spearman, 4),
        "best_epoch": ckpt["epoch"],
        "total_params": total_params,
    })
    tracker.log_model(MODEL_OUT)
    tracker.end_run()


if __name__ == "__main__":
    main()
