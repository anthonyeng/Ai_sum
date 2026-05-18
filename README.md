# AI Video Summarizer

A multimodal machine learning system that summarizes long-form videos into concise visual and textual summaries. Combines trained and pretrained ML models across video, audio, and text modalities.

## Architecture

### Pipeline 1: Video-to-Video Summarization
Uses pretrained feature extraction + trained importance scoring to select key segments.

```
Input video → FFmpeg frame extraction → ResNet18 visual features (pretrained)
→ Temporal context modeling → XGBoost importance scoring (trained on TVSum50)
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

## ML Models

| Model | Type | Dataset | Purpose |
|-------|------|---------|---------|
| **VideoCaptionModel** (Seq2Seq) | Trained by us | MSR-VTT (200K captions) | Video frame captioning |
| **XGBoost** | Trained by us | TVSum50 | Segment importance scoring |
| **BiLSTM** | Trained by us | TVSum50 | Baseline importance scoring |
| **Random Forest** | Trained by us | TVSum50 | Baseline importance scoring |
| **ResNet18** | Pretrained (ImageNet) | — | Visual feature extraction |
| **Faster-Whisper** | Pretrained (OpenAI) | — | Speech-to-text |
| **Sentence-BERT** (all-MiniLM-L6-v2) | Pretrained | — | Semantic text embeddings |

## Summarization Methods

Three extractive summarization approaches, compared via evaluation metrics:

| Method | Type | Speed | Description |
|--------|------|-------|-------------|
| **Sentence-BERT + MMR** | Semantic (primary) | ~10s | Dense embeddings + Maximal Marginal Relevance ranking |
| **TextRank** | Graph-based (baseline) | ~0.1s | PageRank over TF-IDF similarity graph |
| **TF-IDF** | Statistical (baseline) | ~0.001s | Term frequency–inverse document frequency scoring |

## Setup

```bash
# Install dependencies
pip install -r requirements-ml.txt

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
python src/training/train_lstm.py
python src/training/train_rf.py
```

## Evaluation

```bash
# Compare summarization methods (TF-IDF vs TextRank vs SBERT+MMR)
python src/summarization/compare.py --demo

# Compare with a custom transcript
python src/summarization/compare.py --transcript path/to/transcript.txt

# Compare with reference summary (computes ROUGE + BERTScore)
python src/summarization/compare.py --transcript transcript.txt --reference reference.txt

# Save results to JSON
python src/summarization/compare.py --demo --output results.json
```

## Trained Model Architecture

**VideoCaptionModel** (trained on MSR-VTT):
- Encoder: BiGRU (2 layers, 512 hidden, bidirectional)
- Attention: Bahdanau (256-dim)
- Decoder: GRU (512 hidden) with autoregressive generation
- Training: Cross-entropy loss, teacher forcing (50%), gradient clipping
- Inference: Nucleus sampling (temperature=0.8, top_p=0.9)

## Tech Stack

- **ML**: PyTorch, scikit-learn, XGBoost, Sentence-Transformers
- **Video**: OpenCV, FFmpeg
- **Audio**: Faster-Whisper, librosa
- **Backend**: Flask, PostgreSQL
- **Frontend**: Vanilla HTML/CSS/JS
- **Evaluation**: ROUGE, BERTScore, BLEU
