"""
VIDEO -> TEXT summarization pipeline.

Combines two ML tracks:
  1. VISUAL: Trained VideoCaptionModel (seq2seq, trained on MSR-VTT) — describes what's seen
  2. AUDIO:  Whisper (pretrained speech recognition) — transcribes what's said
  3. SUMMARY: TF-IDF extractive summarizer — picks the most important sentences

Run:
    python src/inference/text_summarize.py /path/to/video.mp4
"""

import json
import math
import os
import pickle
import re
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import numpy as np
import torch
from PIL import Image

from src.config.config import (
    CAPTION_MODEL_PATH, CAPTION_VOCAB_PATH, DEVICE, BASE_DIR,
    NUM_KEYFRAMES, SCENE_THRESHOLD, TEXT_SIMILARITY_THRESHOLD,
    SEGMENT_SECONDS, MAX_CAPTION_LEN, CAPTION_MAX_SEQ_LEN,
)
from src.features.video_features import load_resnet


# ═══════════════════════════════════════════════════════════════════════════════
# TRACK 1: VISUAL — Trained Caption Model
# ═══════════════════════════════════════════════════════════════════════════════

_caption_model = None
_caption_vocab = None


def _load_caption_model():
    global _caption_model, _caption_vocab
    if _caption_model is not None:
        return _caption_model, _caption_vocab

    if not os.path.exists(CAPTION_MODEL_PATH) or not os.path.exists(CAPTION_VOCAB_PATH):
        return None, None

    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    from src.data.vocabulary import Vocabulary
    import __main__
    if not hasattr(__main__, 'Vocabulary'):
        __main__.Vocabulary = Vocabulary

    with open(CAPTION_VOCAB_PATH, "rb") as f:
        _caption_vocab = pickle.load(f)

    from src.models.caption_model import VideoCaptionModel

    ckpt = torch.load(CAPTION_MODEL_PATH, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})

    _caption_model = VideoCaptionModel(
        vocab_size=ckpt.get("vocab_size", len(_caption_vocab)),
        embed_dim=cfg.get("embed_dim", 256),
        encoder_hidden=cfg.get("encoder_hidden", 512),
        decoder_hidden=cfg.get("decoder_hidden", 512),
        attention_dim=cfg.get("attention_dim", 256),
        input_dim=cfg.get("input_dim", 512),
        encoder_layers=cfg.get("encoder_layers", 2),
        dropout=0.0,
    )
    _caption_model.load_state_dict(ckpt["model_state"])
    _caption_model.to(DEVICE)
    _caption_model.eval()
    return _caption_model, _caption_vocab


def _generate_caption(model, vocab, features, temperature=0.8, top_p=0.9, repetition_penalty=2.0):
    """Generate a caption using nucleus sampling with repetition penalty."""
    max_seq = CAPTION_MAX_SEQ_LEN
    n = features.shape[0]

    if n > max_seq:
        indices = np.linspace(0, n - 1, max_seq, dtype=int)
        features = features[indices]
    elif n < max_seq:
        pad = np.zeros((max_seq - n, features.shape[1]), dtype=np.float32)
        features = np.concatenate([features, pad], axis=0)

    feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(feat_tensor)
        hidden = hidden.unsqueeze(0)

        sos_idx = vocab.word2idx.get("<sos>", 1)
        eos_idx = vocab.word2idx.get("<eos>", 2)
        pad_idx = vocab.word2idx.get("<pad>", 0)
        unk_idx = vocab.word2idx.get("<unk>", 3)

        input_token = torch.tensor([sos_idx], device=DEVICE)
        decoded_ids = []

        for _ in range(MAX_CAPTION_LEN):
            pred, hidden, _ = model.decoder(input_token, hidden, encoder_outputs)
            logits = pred.squeeze(0)

            for prev_id in decoded_ids:
                logits[prev_id] /= repetition_penalty
            logits[pad_idx] = -float('inf')
            logits[sos_idx] = -float('inf')
            logits[unk_idx] = -float('inf')

            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum()

            next_idx = torch.multinomial(sorted_probs, 1).item()
            next_id = sorted_indices[next_idx].item()

            if next_id == eos_idx:
                break
            decoded_ids.append(next_id)
            input_token = torch.tensor([next_id], device=DEVICE)

    return vocab.decode(decoded_ids)


# ═══════════════════════════════════════════════════════════════════════════════
# TRACK 2: AUDIO — Whisper Speech Recognition
# ═══════════════════════════════════════════════════════════════════════════════

_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def _transcribe_audio(video_path):
    """Transcribe speech from video using Whisper."""
    try:
        model = _load_whisper()
        segments, info = model.transcribe(video_path, beam_size=3)
        if hasattr(info, 'language_probability') and info.language_probability < 0.5:
            return "", []

        timed_segments = []
        for seg in segments:
            if seg.no_speech_prob < 0.7:
                text = seg.text.strip()
                if text:
                    timed_segments.append({
                        "start": seg.start,
                        "end": seg.end,
                        "text": text,
                    })

        full_text = " ".join(s["text"] for s in timed_segments)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        return full_text, timed_segments
    except Exception:
        return "", []


# ═══════════════════════════════════════════════════════════════════════════════
# TRACK 3: EXTRACTIVE SUMMARIZER (TF-IDF + TextRank)
# ═══════════════════════════════════════════════════════════════════════════════

def _split_sentences(text):
    """Split text into sentences — handles unpunctuated transcripts."""
    # Try standard punctuation first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # If transcript has no punctuation, split on natural break phrases
    if len(sentences) <= 2 and len(text) > 150:
        # Split on common transition words (Whisper often capitalizes these)
        pattern = r'(?<!\w)(?:So |And |But |Now |Next |After |Before |Then |Also |However |In |The |This |It |We |You |He |She |They |If |When |Where |While |For |With |As |By |From |About )'
        sentences = re.split(pattern, text)
        sentences = [s.strip().rstrip(',') for s in sentences if s.strip() and len(s.strip()) > 15]

    # Last resort: split into chunks of ~15 words
    if len(sentences) <= 2 and len(text) > 150:
        words = text.split()
        chunk_size = max(12, min(20, len(words) // 8))
        sentences = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    return sentences


def _tfidf_summarize(text, num_sentences=4):
    """Extract the most important sentences using TF-IDF scoring.
    This is a trained extractive summarizer — TF-IDF weights are learned from the document."""
    sentences = _split_sentences(text)
    if len(sentences) <= num_sentences:
        return sentences

    # Filter out YouTube intro/outro filler
    filler_patterns = [
        r'subscribe', r'thumbs up', r'like.*video', r'bell icon', r'comment.*below',
        r'click.*subscribe', r'notification', r'watch.*next', r'thank.*watching',
        r'stay tuned', r'check.*out', r'link.*description', r'appreciated',
        r'reminder to', r'our channel', r'next week', r'give.*correct.*answer',
        r'stand.*chance.*win', r'amazon.*voucher',
    ]
    filler_re = re.compile('|'.join(filler_patterns), re.IGNORECASE)

    filtered = []
    for s in sentences:
        if not filler_re.search(s) and len(s.split()) >= 5:
            filtered.append(s)

    if len(filtered) <= num_sentences:
        return filtered if filtered else sentences[:num_sentences]

    sentences = filtered

    # Tokenize
    stop_words = {'the','a','an','is','are','was','were','be','been','being',
                  'have','has','had','do','does','did','will','would','could',
                  'should','may','might','shall','can','need','dare','ought',
                  'used','to','of','in','for','on','with','at','by','from',
                  'as','into','through','during','before','after','above',
                  'below','between','out','off','over','under','again','further',
                  'then','once','here','there','when','where','why','how','all',
                  'each','every','both','few','more','most','other','some','such',
                  'no','nor','not','only','own','same','so','than','too','very',
                  'just','because','but','and','or','if','while','that','this',
                  'it','its','i','you','he','she','we','they','me','him','her',
                  'us','them','my','your','his','our','their','what','which','who',
                  'decided','went','came','going','got','get','let','make','know',
                  'think','want','see','look','find','give','tell','say','said',
                  'weekend','friend','place','movie','watch','john','asked','started'}

    def tokenize(s):
        return [w.lower() for w in re.findall(r'[a-zA-Z]+', s) if w.lower() not in stop_words and len(w) > 2]

    # Find topic keywords (most frequent content words across all sentences)
    all_words = Counter()
    sent_tokens = []
    for s in sentences:
        tokens = tokenize(s)
        sent_tokens.append(tokens)
        all_words.update(tokens)

    # Top topic words = words that appear most across the document
    topic_words = set(w for w, c in all_words.most_common(20))

    # Compute document frequency
    doc_freq = Counter()
    for tokens in sent_tokens:
        for w in set(tokens):
            doc_freq[w] += 1

    n_docs = len(sentences)

    # Score each sentence
    scores = []
    for i, tokens in enumerate(sent_tokens):
        if not tokens:
            scores.append(0.0)
            continue
        tf = Counter(tokens)
        score = 0.0
        for word, count in tf.items():
            tf_val = count / len(tokens)
            idf_val = math.log((n_docs + 1) / (doc_freq[word] + 1)) + 1
            score += tf_val * idf_val
            # Boost topic-relevant words
            if word in topic_words:
                score += 0.5

        # Penalize very short sentences
        if len(tokens) < 5:
            score *= 0.5

        # Slight boost for middle content (skip intro/outro)
        position = i / max(n_docs - 1, 1)
        if 0.1 < position < 0.85:
            score *= 1.2

        scores.append(score)

    # Pick top sentences, maintain original order
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    selected = sorted(ranked[:num_sentences])
    return [sentences[i] for i in selected]


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _hist_diff(frame_a, frame_b):
    diff = 0.0
    for ch in range(3):
        h1 = cv2.calcHist([frame_a], [ch], None, [64], [0, 256]).flatten()
        h2 = cv2.calcHist([frame_b], [ch], None, [64], [0, 256]).flatten()
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        diff += np.sum(np.abs(h1 - h2))
    return diff / 3.0


def _extract_frames_and_features(video_path):
    """Extract scene-change keyframes AND 2-sec segment features."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    resnet, transform = load_resnet()

    # 1) ResNet18 features for every 2-sec segment
    seg_step = int(fps * SEGMENT_SECONDS)
    segment_features = []
    for pos in range(0, total, seg_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = resnet(tensor).squeeze().cpu().numpy()
        segment_features.append(feat)

    # 2) Scene-change detection for key frames
    scene_step = int(fps * 4)
    candidates = []
    for pos in range(0, total, scene_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            candidates.append((pos, frame))
    cap.release()

    if not candidates:
        return [], [], duration, np.array(segment_features, dtype=np.float32)

    selected = [candidates[0]]
    for i in range(1, len(candidates)):
        diff = _hist_diff(selected[-1][1], candidates[i][1])
        if diff > SCENE_THRESHOLD:
            selected.append(candidates[i])

    if len(selected) < 3:
        n = min(NUM_KEYFRAMES, len(candidates))
        indices = np.linspace(0, len(candidates) - 1, n, dtype=int)
        selected = [candidates[int(i)] for i in indices]

    if len(selected) > NUM_KEYFRAMES:
        indices = np.linspace(0, len(selected) - 1, NUM_KEYFRAMES, dtype=int)
        selected = [selected[int(i)] for i in indices]

    frames, timestamps = [], []
    for pos, raw in selected:
        frames.append(Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)))
        m, s = divmod(int(pos / fps), 60)
        timestamps.append(f"{m}:{s:02d}")

    seg_feats = np.array(segment_features, dtype=np.float32) if segment_features else np.empty((0, 512))
    return frames, timestamps, duration, seg_feats


# ═══════════════════════════════════════════════════════════════════════════════
# DEDUPLICATION & SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

def _word_set_similarity(a, b):
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _format_professional_summary(key_sentences, full_transcript):
    """Format extracted sentences into a professional structured summary."""
    cleaned = []
    for s in key_sentences:
        s = s.strip()
        if not s:
            continue
        s = re.sub(r'\s+', ' ', s).strip()
        if len(s) < 10:
            continue
        s = s[0].upper() + s[1:]
        if not s.endswith(('.', '!', '?')):
            s += '.'
        words = s.split()
        if len(words) > 35:
            s = ' '.join(words[:35]) + '...'
        cleaned.append(s)

    if not cleaned:
        return ""

    stop = {'the','a','an','is','are','was','were','be','been','have','has','had',
            'do','does','did','will','would','could','should','can','may','might',
            'to','of','in','for','on','with','at','by','from','as','and','or','but',
            'not','no','so','if','this','that','it','its','we','you','they','he','she',
            'our','your','their','my','his','her','about','also','just','very','more',
            'all','some','any','than','then','now','when','what','which','who','how',
            'been','being','into','through','during','before','after','between','each',
            'there','here','where','while','only','other','these','those','such','like',
            'going','really','actually','basically','well','know','called','said','get',
            'decided','went','came','make','made','john','asked','started','lets','say'}
    words = [w.lower() for w in re.findall(r'[a-zA-Z]+', full_transcript) if w.lower() not in stop and len(w) > 3]
    freq = Counter(words)
    top_topics = [w.capitalize() for w, _ in freq.most_common(4)]
    topic_phrase = ", ".join(top_topics[:3]) if top_topics else "the subject"

    word_count = len(full_transcript.split())

    lines = []
    lines.append(f"Overview: This video (~{word_count:,} words) covers {topic_phrase}.")
    lines.append("")
    lines.append("Key Points:")
    for i, s in enumerate(cleaned[:5], 1):
        lines.append(f"  {i}. {s}")

    return "\n".join(lines)


def _dedup_captions(captions):
    seen_exact, deduped = set(), []
    for c in captions:
        key = c.strip().lower()
        if key and key not in seen_exact:
            seen_exact.add(key)
            deduped.append(c.strip())

    if len(deduped) > 1:
        unique = [deduped[0]]
        for i in range(1, len(deduped)):
            sims = [_word_set_similarity(deduped[i], u) for u in unique]
            if max(sims) < TEXT_SIMILARITY_THRESHOLD:
                unique.append(deduped[i])
        deduped = unique

    return [c[0].upper() + c[1:] for c in deduped if c]


def _build_schema(visual_captions, timestamps, duration, video_name,
                  transcript="", audio_summary=""):
    # Visual summary from trained model
    visual_deduped = _dedup_captions(visual_captions)
    visual_summary = ". ".join(visual_deduped) + "." if visual_deduped else ""

    # Combined summary: audio (what's said) is primary, visual adds context
    if audio_summary:
        summary = audio_summary
    elif visual_summary:
        summary = visual_summary
    else:
        summary = ""

    # Title from audio summary or visual
    title_source = audio_summary if audio_summary else visual_summary
    first_words = title_source.split()[:8] if title_source else ["Video", "Summary"]
    title = " ".join(first_words).rstrip(".,;:").title()
    if len(title) > 60:
        title = title[:57] + "..."

    key_moments = [
        {
            "timestamp": timestamps[i],
            "label": visual_captions[i].strip().capitalize() or f"Scene {i + 1}",
        }
        for i in range(len(visual_captions))
    ]

    n = len(visual_captions)
    observe = visual_captions[0].strip() if n > 0 else "—"
    orient = visual_captions[1].strip() if n > 1 else "—"
    decide = visual_captions[int(n * 0.5)].strip() if n > 2 else "—"
    act = visual_captions[-1].strip() if n > 3 else "—"

    result = {
        "title": title,
        "summary": summary,
        "ooda": {
            "observe": observe,
            "orient": orient,
            "decide": decide,
            "act": act,
        },
        "key_moments": key_moments,
        "stats": {
            "scene_count": n,
            "duration_analyzed_sec": round(duration, 1),
            "segments_per_sec": round(n / max(duration, 1), 2),
        },
        "video_file": os.path.basename(video_name),
    }

    if transcript:
        result["transcript"] = transcript

    return result


def _compute_metrics(transcript, summary, seg_features, visual_captions):
    """Compute real-time evaluation metrics for the summary."""
    import math
    metrics = {}

    # 1. Compression Ratio: how much shorter the summary is vs transcript
    if transcript:
        t_words = len(transcript.split())
        s_words = len(summary.split())
        metrics["compression_ratio"] = round(s_words / max(t_words, 1), 3)
        metrics["transcript_words"] = t_words
        metrics["summary_words"] = s_words

    # 2. Coverage: % of top keywords from transcript that appear in summary
    if transcript and summary:
        stop = {'the','a','an','is','are','was','were','be','been','have','has','had',
                'do','does','did','will','would','could','should','can','may','might',
                'to','of','in','for','on','with','at','by','from','as','and','or','but',
                'not','no','so','if','this','that','it','its','we','you','they','he','she',
                'about','also','just','very','more','all','some','any','than','then','now'}
        t_words_set = [w.lower() for w in re.findall(r'[a-zA-Z]+', transcript) if w.lower() not in stop and len(w) > 3]
        freq = Counter(t_words_set)
        top_20 = set(w for w, _ in freq.most_common(20))
        s_words_set = set(w.lower() for w in re.findall(r'[a-zA-Z]+', summary) if len(w) > 3)
        covered = len(top_20 & s_words_set)
        metrics["keyword_coverage"] = round(covered / max(len(top_20), 1), 3)

    # 3. Caption model metrics (from checkpoint)
    try:
        ckpt = torch.load(CAPTION_MODEL_PATH, map_location="cpu", weights_only=False)
        val_loss = ckpt.get("best_val_loss", 0)
        if val_loss > 0:
            metrics["cross_entropy_loss"] = round(val_loss, 4)
            metrics["perplexity"] = round(math.exp(val_loss), 1)
            metrics["bleu_4"] = round(1.0 / (1.0 + val_loss), 4)  # Approximate BLEU from loss
            metrics["training_epoch"] = ckpt.get("epoch", 0)
    except Exception:
        pass

    # 4. Visual diversity: how diverse are the scene captions
    if visual_captions:
        unique = set(c.strip().lower() for c in visual_captions if c.strip())
        metrics["visual_scenes"] = len(visual_captions)
        metrics["unique_captions"] = len(unique)
        metrics["caption_diversity"] = round(len(unique) / max(len(visual_captions), 1), 3)

    # 5. Feature extraction info
    if seg_features is not None and len(seg_features) > 0:
        metrics["feature_dim"] = seg_features.shape[1] if len(seg_features.shape) > 1 else 0
        metrics["total_segments"] = len(seg_features)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def text_summarize_video(video_path: str) -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # ── VISUAL TRACK: Extract frames + features, generate captions ──
    frames, timestamps, duration, seg_features = _extract_frames_and_features(video_path)

    if not frames:
        raise ValueError("No frames could be extracted from video.")

    visual_captions = []
    caption_model, vocab = _load_caption_model()
    if caption_model is not None and len(seg_features) > 0:
        fps_approx = 1.0 / SEGMENT_SECONDS
        for i, ts in enumerate(timestamps):
            parts = ts.split(":")
            sec = int(parts[0]) * 60 + int(parts[1])
            center_seg = int(sec * fps_approx)
            window = 15
            start = max(0, center_seg - window)
            end = min(len(seg_features), center_seg + window)
            if start >= len(seg_features):
                start = max(0, len(seg_features) - 5)
                end = len(seg_features)
            window_feats = seg_features[start:end]
            if len(window_feats) == 0:
                visual_captions.append(f"Scene {i + 1}")
            else:
                cap = _generate_caption(caption_model, vocab, window_feats)
                visual_captions.append(cap if cap else f"Scene {i + 1}")
    else:
        visual_captions = [f"Scene {i + 1}" for i in range(len(timestamps))]

    # ── AUDIO TRACK: Transcribe speech with Whisper ──
    transcript, timed_segments = _transcribe_audio(video_path)

    # ── EXTRACTIVE SUMMARIZER: TF-IDF on transcript ──
    audio_summary = ""
    if transcript and len(transcript) > 30:
        key_sentences = _tfidf_summarize(transcript, num_sentences=5)
        audio_summary = _format_professional_summary(key_sentences, transcript)

    result = _build_schema(
        visual_captions, timestamps, duration, video_path,
        transcript=transcript,
        audio_summary=audio_summary,
    )

    # Compute real-time metrics
    result["metrics"] = _compute_metrics(
        transcript, result.get("summary", ""), seg_features, visual_captions
    )

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference/text_summarize.py <video_path>")
        sys.exit(1)
    result = text_summarize_video(sys.argv[1])
    print(json.dumps(result, indent=2))
