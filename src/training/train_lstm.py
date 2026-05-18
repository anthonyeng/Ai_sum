import os
import sys
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.lstm_model import BiLSTMSummarizer

FEATURE_DIR = "data/processed/features"
ANNO_FILE = "data/raw/tvsum/ydata-tvsum50-anno.tsv"
MODEL_OUT = "outputs/models/bilstm_video_split.pt"

TEST_RATIO = 0.2
RANDOM_SEED = 42

HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 60
BATCH_SIZE = 4
PATIENCE = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── data loading ──────────────────────────────────────────────

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


# ── dataset ───────────────────────────────────────────────────

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

            # normalize scores to [0, 1]
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


# ── training ──────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for features, scores, lengths, mask in loader:
        features = features.to(DEVICE)
        scores = scores.to(DEVICE)
        mask = mask.to(DEVICE)

        preds = model(features, lengths)

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

        preds = model(features, lengths)

        loss = criterion(preds[mask], scores[mask])
        total_loss += loss.item() * mask.sum().item()
        n_samples += mask.sum().item()

        # rescale back to [1, 5] for metrics
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

    return avg_loss, mse, mae, r2


# ── main ──────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    from src.tracking.mlflow_utils import ExperimentTracker
    tracker = ExperimentTracker("video_summarization")
    tracker.start_run("bilstm", tags={"model_type": "bilstm", "dataset": "tvsum50"})
    tracker.log_params({
        "model": "BiLSTMSummarizer",
        "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
        "dropout": DROPOUT, "lr": LR,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "patience": PATIENCE, "seed": RANDOM_SEED,
    })

    print("Loading annotations...")
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

    model = BiLSTMSummarizer(
        input_dim=512,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_mse, val_mae, val_r2 = evaluate(model, test_loader, criterion)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"MSE: {val_mse:.4f} | MAE: {val_mae:.4f} | R2: {val_r2:.4f} | "
            f"LR: {lr:.6f}"
        )
        tracker.log_metrics({
            "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
            "mse": round(val_mse, 4), "mae": round(val_mae, 4), "r2": round(val_r2, 4),
        }, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # final evaluation with best model
    print("\n--- Final Evaluation (best checkpoint) ---")
    model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE, weights_only=True))
    _, mse, mae, r2 = evaluate(model, test_loader, criterion)

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2:  {r2:.4f}")
    print(f"\nModel saved to: {MODEL_OUT}")

    tracker.log_metrics({"final_mse": round(mse, 4), "final_mae": round(mae, 4), "final_r2": round(r2, 4)})
    tracker.log_model(MODEL_OUT)
    tracker.end_run()


if __name__ == "__main__":
    main()
