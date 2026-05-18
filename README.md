# AI Video Summarizer

A multimodal machine learning system that summarizes long-form videos into concise visual and textual summaries. Combines trained and pretrained ML models across video, audio, and text modalities — no paid APIs, everything runs locally.

## Architecture

### Pipeline 1: Video-to-Video Summarization
Uses pretrained feature extraction + trained importance scoring to select key segments.

```
Input video → FFmpeg frame extraction → ResNet18 visual features (pretrained)
→ Temporal context modeling → XGBoost / Temporal Transformer importance scoring
→ Adaptive thresholding → Scene deduplication → FFmpeg summary video
```

### Pipeline 2: Video-to-Text Summarization
Combines a trained captioning model with pretrained speech recognition and semantic summarization.

```
Input video → Visual track: ResNet18 features → Trained Seq2Seq captioning model
           → Audio track: Faster-Whisper transcription (pretrained)
           → Sentence-BERT + MMR semantic summarization (pretrained embeddings)
           → Structured timestamped summary
```

### Pipeline 3: PDF-to-Text Summarization
```
PDF → PyMuPDF text extraction → Sentence-BERT + MMR semantic ranking → Summary
```

### Pipeline 4: Query-Focused Summarization
```
User query → Sentence-BERT encoding → Semantic search over transcript + captions
→ Timestamped results ranked by relevance
```

### Multimodal Feature Fusion
```
Video → ResNet18 features (512-dim) + CLIP ViT-B/32 features (512-dim)
      + Audio features (energy, spectral, ZCR) (4-dim)
      + Sentence-BERT transcript embeddings (384-dim)
      + Scene-change scores (1-dim) + Motion scores (1-dim)
      → Concatenated multimodal vector → Segment importance scoring
```

## ML Models

| Model | Type | Dataset | Purpose |
|-------|------|---------|---------|
| **VideoCaptionModel** (Seq2Seq) | Trained by us | MSR-VTT (200K captions) | Video frame captioning |
| **XGBoost** | Trained by us | TVSum50 | Segment importance scoring (classical ML) |
| **Temporal Transformer** | Trained by us | TVSum50 | Segment importance scoring (deep learning) |
| **BiLSTM** | Trained by us | TVSum50 | Baseline importance scoring |
| **Random Forest** | Trained by us | TVSum50 | Baseline importance scoring |
| **ResNet18** | Pretrained (ImageNet) | — | Visual feature extraction (512-dim) |
| **CLIP ViT-B/32** | Pretrained (OpenAI) | — | Semantic visual embeddings (512-dim) |
| **Faster-Whisper** | Pretrained (OpenAI) | — | Speech-to-text transcription |
| **Sentence-BERT** (all-MiniLM-L6-v2) | Pretrained | — | Semantic text embeddings (384-dim) |

## Summarization Methods

Three extractive summarization approaches, compared via evaluation metrics:

| Method | Type | Speed | Description |
|--------|------|-------|-------------|
| **Sentence-BERT + MMR** | Semantic (primary) | ~10s | Dense embeddings + Maximal Marginal Relevance ranking |
| **TextRank** | Graph-based (baseline) | ~0.1s | PageRank over TF-IDF similarity graph |
| **TF-IDF** | Statistical (baseline) | ~0.001s | Term frequency–inverse document frequency scoring |

## Video Importance Scoring Models

| Model | Type | Parameters | Captures |
|-------|------|-----------|----------|
| **XGBoost** | Classical ML (default) | ~10K trees | Temporal context [prev, current, next] |
| **Temporal Transformer** | Deep learning | 2.27M | Global self-attention over all segments |
| **BiLSTM** | Deep learning (baseline) | ~400K | Local sequential context |
| **Random Forest** | Classical ML (baseline) | ~100 trees | Per-segment features only |

## Setup

```bash
# Install dependencies
pip install -r requirements-ml.txt

# Install CLIP (optional, for semantic visual features)
pip install git+https://github.com/openai/CLIP.git

# Start PostgreSQL (must be running)
createdb video_summarizer  # if not exists

# Run the server
python api.py
# Open http://127.0.0.1:5003
```

## Training

```bash
# Train video captioning model (Seq2Seq on MSR-VTT)
python src/training/train_caption.py

# Train video summarization models (on TVSum50)
python src/training/train_xgb.py
python src/training/train_temporal_transformer.py
python src/training/train_lstm.py
python src/training/train_rf.py
```

## Inference

```bash
# Video-to-Video summarization (default: XGBoost)
python src/inference/summarize.py path/to/video.mp4

# Video-to-Video with Temporal Transformer
python src/inference/summarize.py path/to/video.mp4 --model transformer

# Video-to-Text summarization
python src/inference/text_summarize.py path/to/video.mp4

# PDF summarization
python src/inference/pdf_summarize.py path/to/file.pdf
```

## Evaluation

```bash
# Compare summarization methods (TF-IDF vs TextRank vs SBERT+MMR)
python src/summarization/compare.py --demo

# Compare with reference summary (computes ROUGE + BERTScore)
python src/summarization/compare.py --transcript transcript.txt --reference reference.txt

# Save results to JSON
python src/summarization/compare.py --demo --output results.json
```

## Evaluation Metrics

| Pipeline | Metrics |
|----------|---------|
| **Video-to-Video** | F-score, compression ratio, redundancy, diversity, coverage |
| **Video-to-Text (captioning)** | BLEU-1/2/3/4, ROUGE-L, METEOR |
| **Text summarization** | ROUGE-1/2/L, BERTScore |
| **System** | Inference time, memory usage, processing speed |

## Trained Model Architectures

**VideoCaptionModel** (trained on MSR-VTT):
- Encoder: BiGRU (2 layers, 512 hidden, bidirectional)
- Attention: Bahdanau (256-dim)
- Decoder: GRU (512 hidden) with autoregressive generation
- Training: Cross-entropy loss, teacher forcing (50%), gradient clipping
- Inference: Nucleus sampling (temperature=0.8, top_p=0.9)

**Temporal Transformer** (trained on TVSum50):
- Input projection: Linear(512 → 256) + LayerNorm
- Positional encoding: Sinusoidal (Vaswani et al., 2017)
- Encoder: 4 × TransformerEncoderLayer (d_model=256, 8 heads, FFN=512, GELU)
- Scoring head: Linear(256 → 128) → ReLU → Linear(128 → 1) → Sigmoid
- Training: MSE loss, AdamW, warmup + cosine annealing, gradient clipping
- Parameters: 2.27M

## Project Structure

```
src/
├── config/config.py           # Centralized configuration
├── data/                      # Data loading, vocabulary, preprocessing
├── features/
│   ├── video_features.py      # ResNet18 feature extraction
│   ├── clip_embeddings.py     # CLIP ViT-B/32 semantic embeddings
│   ├── text_embeddings.py     # Sentence-BERT text embeddings
│   ├── audio_features.py      # Audio energy, spectral, ZCR
│   └── multimodal_fusion.py   # Combine all modalities
├── models/
│   ├── caption_model.py       # Seq2Seq VideoCaptionModel
│   ├── temporal_transformer.py # Transformer importance scorer
│   └── lstm_model.py          # BiLSTM importance scorer
├── training/
│   ├── train_caption.py       # Train Seq2Seq on MSR-VTT
│   ├── train_temporal_transformer.py  # Train Transformer on TVSum50
│   ├── train_xgb.py           # Train XGBoost on TVSum50
│   ├── train_lstm.py          # Train BiLSTM on TVSum50
│   └── train_rf.py            # Train Random Forest on TVSum50
├── inference/
│   ├── summarize.py           # Video-to-Video pipeline
│   ├── text_summarize.py      # Video-to-Text pipeline
│   └── pdf_summarize.py       # PDF-to-Text pipeline
├── summarization/
│   ├── sbert_mmr.py           # Sentence-BERT + MMR (primary)
│   ├── textrank.py            # TextRank baseline
│   ├── tfidf_baseline.py      # TF-IDF baseline
│   ├── query_focused.py       # Query-focused semantic search
│   └── compare.py             # Method comparison + evaluation
├── evaluation/
│   ├── video_metrics.py       # F-score, compression, diversity
│   ├── caption_metrics.py     # BLEU, ROUGE-L, METEOR
│   ├── text_metrics.py        # ROUGE-1/2/L, BERTScore
│   └── system_metrics.py      # Timer, memory, speed profiling
└── utils/                     # Helpers, paths, logging
```

## Tech Stack

- **ML**: PyTorch, scikit-learn, XGBoost, Sentence-Transformers, CLIP
- **Video**: OpenCV, FFmpeg
- **Audio**: Faster-Whisper, librosa
- **Backend**: Flask, PostgreSQL
- **Frontend**: Vanilla HTML/CSS/JS
- **Evaluation**: ROUGE, BERTScore, BLEU, METEOR
- **Datasets**: MSR-VTT, TVSum50
