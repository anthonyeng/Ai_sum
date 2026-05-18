"""
Centralized configuration for the AI Video Summarizer project.
All paths and hyperparameters in one place.
"""

import os
import torch

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summaries")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp_segments")

# ── TVSum paths ───────────────────────────────────────────────────────────────
TVSUM_INFO_FILE = os.path.join(DATA_RAW_DIR, "tvsum", "ydata-tvsum50-info.tsv")
TVSUM_ANNO_FILE = os.path.join(DATA_RAW_DIR, "tvsum", "ydata-tvsum50-anno.tsv")
TVSUM_VIDEO_DIR = os.path.join(DATA_RAW_DIR, "tvsum", "videos")
TVSUM_FEATURE_DIR = os.path.join(DATA_PROCESSED_DIR, "features")
TVSUM_FRAMES_DIR = os.path.join(DATA_PROCESSED_DIR, "frames")

# ── MSR-VTT paths ────────────────────────────────────────────────────────────
MSVTT_VIDEO_DIR = os.path.join(DATA_RAW_DIR, "msvtt", "TrainValVideo")
MSVTT_ANNO_FILE = os.path.join(DATA_RAW_DIR, "msvtt", "train_val_videodatainfo.json")
MSVTT_FULL_ANNO_FILE = os.path.join(DATA_RAW_DIR, "msvtt", "MSRVTT_data.json")
MSVTT_DATASET_PATH = os.path.join(DATA_PROCESSED_DIR, "msvtt", "dataset.pkl")

# ── Model paths ──────────────────────────────────────────────────────────────
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_video_split_temporal.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_video_split.pkl")
BILSTM_MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_video_split.pt")
CAPTION_MODEL_PATH = os.path.join(MODEL_DIR, "caption_model.pt")
CAPTION_VOCAB_PATH = os.path.join(MODEL_DIR, "caption_vocab.pkl")
TRANSFORMER_MODEL_PATH = os.path.join(MODEL_DIR, "temporal_transformer.pt")

# ── Device ───────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ── Video processing ─────────────────────────────────────────────────────────
SEGMENT_SECONDS = 2
MAX_VIDEO_DURATION_SEC = 600  # 10 minutes max
MAX_UPLOAD_SIZE_MB = 500

# ── Video summarization (Pipeline 1) ─────────────────────────────────────────
TEMPORAL_RADIUS = 1            # [prev, current, next]
THRESHOLD_FACTOR = 0.5         # segments above mean + factor * std
MAX_SUMMARY_RATIO = 0.4        # summary max 40% of original
MIN_SUMMARY_SEGMENTS = 3
CONTEXT_PAD = 1                # 1 segment of context padding
SIMILARITY_THRESHOLD = 0.97    # cosine similarity for dedup
CROSSFADE_SECONDS = 0.5

# ── Text summarization (Pipeline 2) ──────────────────────────────────────────
NUM_KEYFRAMES = 8
SCENE_THRESHOLD = 30           # histogram diff for scene change
TEXT_SIMILARITY_THRESHOLD = 0.85  # Jaccard for text dedup
MAX_CAPTION_LEN = 30

# ── ResNet18 ─────────────────────────────────────────────────────────────────
FEATURE_DIM = 512
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]

# ── Caption model hyperparameters ────────────────────────────────────────────
EMBED_DIM = 256
ENCODER_HIDDEN = 512
DECODER_HIDDEN = 512
ATTENTION_DIM = 256
INPUT_DIM = 512
ENCODER_LAYERS = 2
CAPTION_DROPOUT = 0.3
CAPTION_BATCH_SIZE = 128
CAPTION_EPOCHS = 30
CAPTION_LR = 1e-3
CAPTION_TEACHER_FORCING = 0.5
CAPTION_MAX_SEQ_LEN = 60
CAPTION_VAL_SPLIT = 0.1
CAPTION_CLIP_GRAD = 1.0

# ── Video summarization model hyperparameters ────────────────────────────────
XGB_N_ESTIMATORS = 400
XGB_MAX_DEPTH = 6
XGB_LR = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8

RF_N_ESTIMATORS = 100

LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_LR = 1e-3
LSTM_EPOCHS = 60
LSTM_BATCH_SIZE = 4
LSTM_PATIENCE = 10

# ── Temporal Transformer hyperparameters ─────────────────────────────────────
TRANSFORMER_D_MODEL = 256
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_DIM_FEEDFORWARD = 512
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_LR = 5e-4
TRANSFORMER_WEIGHT_DECAY = 1e-4
TRANSFORMER_EPOCHS = 80
TRANSFORMER_BATCH_SIZE = 4
TRANSFORMER_PATIENCE = 15
TRANSFORMER_WARMUP_EPOCHS = 5

# ── CLIP ViT-B/32 (pretrained visual embeddings) ─────────────────────────────
CLIP_MODEL_NAME = "ViT-B/32"
CLIP_DIM = 512

# ── Sentence-BERT (pretrained text embeddings) ───────────────────────────────
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SBERT_DIM = 384

# ── Audio features ───────────────────────────────────────────────────────────
AUDIO_FEATURE_DIM = 4   # [rms_energy, spectral_centroid, zcr, energy_variance]
AUDIO_SAMPLE_RATE = 16000

# ── Multimodal fusion ────────────────────────────────────────────────────────
FUSION_WEIGHTS = {
    "audio_energy": 0.25,
    "scene_change": 0.20,
    "motion": 0.15,
    "text_density": 0.20,
    "visual_variance": 0.20,
}

# ── Summarization ────────────────────────────────────────────────────────────
MMR_LAMBDA = 0.7           # MMR trade-off: 1.0=relevance only, 0.0=diversity only
TEXTRANK_DAMPING = 0.85
TEXTRANK_SIM_THRESHOLD = 0.1

# ── Common ───────────────────────────────────────────────────────────────────
TEST_RATIO = 0.2
RANDOM_SEED = 42
MIN_WORD_FREQ = 2
