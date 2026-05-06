"""
Train the video captioning model (BiGRU encoder + Bahdanau attention + LSTM decoder).

Prerequisites:
    python src/data/build_caption_dataset.py   # builds dataset.pkl + caption_vocab.pkl

Run:
    python src/training/train_caption.py
"""

import math
import os
import pickle
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.caption_model import VideoCaptionModel
from src.data.vocabulary import Vocabulary  # noqa: F401  — needed for pickle to resolve Vocabulary

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH = "data/processed/msvtt/dataset.pkl"
VOCAB_PATH = "outputs/models/caption_vocab.pkl"
MODEL_OUT = "outputs/models/caption_model.pt"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
EMBED_DIM = 256
ENCODER_HIDDEN = 512
DECODER_HIDDEN = 512
ATTENTION_DIM = 256
INPUT_DIM = 512
ENCODER_LAYERS = 2
DROPOUT = 0.3

BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
TEACHER_FORCING = 0.5
MAX_SEQ_LEN = 60          # cap video length to 60 segments (~2 min)
VAL_SPLIT = 0.1
CLIP_GRAD = 1.0
RANDOM_SEED = 42

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# ── Dataset ───────────────────────────────────────────────────────────────────

class CaptionDataset(Dataset):
    def __init__(self, samples, max_seq_len=60):
        self.samples = samples
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        features = torch.tensor(s["features"], dtype=torch.float32)
        if len(features) > self.max_seq_len:
            features = features[: self.max_seq_len]
        caption = torch.tensor(s["caption"], dtype=torch.long)
        return features, caption, len(features)


def collate_fn(batch):
    features, captions, lengths = zip(*batch)
    max_len = max(lengths)
    feat_dim = features[0].shape[-1]
    padded = torch.zeros(len(features), max_len, feat_dim)
    for i, (f, l) in enumerate(zip(features, lengths)):
        padded[i, :l] = f
    return padded, torch.stack(captions), torch.tensor(lengths, dtype=torch.long)


# ── Training / eval loops ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, teacher_forcing):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for features, captions, _ in loader:
        features = features.to(DEVICE)
        captions = captions.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(features, captions, teacher_forcing_ratio=teacher_forcing)
        output_flat = outputs[:, 1:].reshape(-1, outputs.size(-1))
        target_flat = captions[:, 1:].reshape(-1)
        loss = criterion(output_flat, target_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()
        non_pad = (target_flat != 0).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
    return total_loss / max(total_tokens, 1)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for features, captions, _ in loader:
            features = features.to(DEVICE)
            captions = captions.to(DEVICE)
            outputs = model(features, captions, teacher_forcing_ratio=0.0)
            output_flat = outputs[:, 1:].reshape(-1, outputs.size(-1))
            target_flat = captions[:, 1:].reshape(-1)
            loss = criterion(output_flat, target_flat)
            non_pad = (target_flat != 0).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
    return total_loss / max(total_tokens, 1)


def compute_bleu(model, loader, vocab):
    try:
        from nltk.translate.bleu_score import corpus_bleu
    except ImportError:
        return 0.0

    model.eval()
    references, hypotheses = [], []
    with torch.no_grad():
        for features, captions, _ in loader:
            features = features.to(DEVICE)
            captions = captions.to(DEVICE)
            outputs = model(features, captions, teacher_forcing_ratio=0.0)
            preds = outputs.argmax(dim=-1)
            for i in range(len(captions)):
                ref = vocab.decode(captions[i].cpu().tolist()).split()
                hyp = vocab.decode(preds[i].cpu().tolist()).split()
                references.append([ref])
                hypotheses.append(hyp)
    return corpus_bleu(references, hypotheses)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("Loading dataset...")
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)
    print(f"  Samples: {len(dataset)}")

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    print(f"  Vocab size: {len(vocab)}")

    random.shuffle(dataset)
    split = int(len(dataset) * (1 - VAL_SPLIT))
    train_set = CaptionDataset(dataset[:split], MAX_SEQ_LEN)
    val_set = CaptionDataset(dataset[split:], MAX_SEQ_LEN)
    print(f"  Train: {len(train_set)}  |  Val: {len(val_set)}")

    pin_mem = DEVICE == "cuda"
    # num_workers=0 avoids multiprocessing semaphore leaks on macOS + MPS
    n_workers = 0 if DEVICE == "mps" else 2
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=n_workers, pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=n_workers, pin_memory=pin_mem,
    )

    model = VideoCaptionModel(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        encoder_hidden=ENCODER_HIDDEN,
        decoder_hidden=DECODER_HIDDEN,
        attention_dim=ATTENTION_DIM,
        input_dim=INPUT_DIM,
        encoder_layers=ENCODER_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {DEVICE}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, TEACHER_FORCING)
        val_loss = eval_epoch(model, val_loader, criterion)
        bleu = compute_bleu(model, val_loader, vocab)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"PPL {math.exp(val_loss):.1f} | "
            f"BLEU-4 {bleu:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "vocab_size": len(vocab),
                    "config": {
                        "embed_dim": EMBED_DIM,
                        "encoder_hidden": ENCODER_HIDDEN,
                        "decoder_hidden": DECODER_HIDDEN,
                        "attention_dim": ATTENTION_DIM,
                        "input_dim": INPUT_DIM,
                        "encoder_layers": ENCODER_LAYERS,
                        "dropout": DROPOUT,
                    },
                },
                MODEL_OUT,
            )
            print(f"           -> best model saved (val={val_loss:.4f})")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Model → {MODEL_OUT}")


if __name__ == "__main__":
    main()
