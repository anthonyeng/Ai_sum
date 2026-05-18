"""
PDF -> TEXT summarization pipeline.

Extracts text from PDF, then uses TF-IDF extractive summarizer
to produce a professional summary.

Run:
    python src/inference/pdf_summarize.py /path/to/file.pdf
"""

import json
import math
import os
import re
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import fitz  # PyMuPDF


def _extract_pdf_text(pdf_path):
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    pages = []
    full_text = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            pages.append(text)
            full_text.append(text)
    doc.close()
    return "\n".join(full_text), len(pages)


def _clean_pdf_text(text):
    """Clean raw PDF text of artifacts."""
    # Remove bullet characters
    text = re.sub(r'[•\u2022\u2023\u25E6\u2043\u2219]', '.', text)
    # Remove "Image from..." references
    text = re.sub(r'Image from[^.]*\.?', '', text, flags=re.IGNORECASE)
    # Remove slide/page markers
    text = re.sub(r'(Slide|Page)\s*\d+', '', text, flags=re.IGNORECASE)
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _split_sentences(text):
    """Split text into sentences."""
    text = _clean_pdf_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out junk: too short, just labels, numbered lists without content
    cleaned = []
    for s in sentences:
        s = s.strip()
        if len(s) < 25:
            continue
        # Skip if it's just a title/heading (all caps or very short with colon)
        if s.isupper() and len(s) < 60:
            continue
        if re.match(r'^\d+\.\s*$', s):
            continue
        cleaned.append(s)
    return cleaned


def _tfidf_summarize(text, num_sentences=5):
    """Extract the most important sentences using TF-IDF scoring."""
    sentences = _split_sentences(text)
    if len(sentences) <= num_sentences:
        return sentences

    stop_words = {'the','a','an','is','are','was','were','be','been','being',
                  'have','has','had','do','does','did','will','would','could',
                  'should','may','might','shall','can','need','to','of','in',
                  'for','on','with','at','by','from','as','into','through',
                  'during','before','after','above','below','between','out',
                  'off','over','under','again','further','then','once','here',
                  'there','when','where','why','how','all','each','every',
                  'both','few','more','most','other','some','such','no','nor',
                  'not','only','own','same','so','than','too','very','just',
                  'because','but','and','or','if','while','that','this','it',
                  'its','i','you','he','she','we','they','me','him','her',
                  'us','them','my','your','his','our','their','what','which',
                  'who','also','about','like','been','being','these','those'}

    def tokenize(s):
        return [w.lower() for w in re.findall(r'[a-zA-Z]+', s) if w.lower() not in stop_words and len(w) > 2]

    doc_freq = Counter()
    sent_tokens = []
    for s in sentences:
        tokens = tokenize(s)
        sent_tokens.append(tokens)
        for w in set(tokens):
            doc_freq[w] += 1

    n_docs = len(sentences)
    all_words = Counter()
    for tokens in sent_tokens:
        all_words.update(tokens)
    topic_words = set(w for w, _ in all_words.most_common(20))

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
            if word in topic_words:
                score += 0.5
        if len(tokens) < 5:
            score *= 0.5
        # Boost early content (abstract/intro) and middle
        position = i / max(n_docs - 1, 1)
        if position < 0.15:
            score *= 1.3
        elif 0.15 < position < 0.7:
            score *= 1.1
        scores.append(score)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    selected = sorted(ranked[:num_sentences])
    return [sentences[i] for i in selected]


def _format_summary(key_sentences, full_text, num_pages):
    """Format into a professional structured summary."""
    cleaned = []
    for s in key_sentences:
        s = s.strip()
        if not s:
            continue
        # Clean up bullet points, special chars, references
        s = re.sub(r'^[\s\-\*\u2022\d\.]+\s*', '', s)
        s = re.sub(r'[•\u2022]', ',', s)
        s = re.sub(r'Image from[^.]*\.?', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s+', ' ', s).strip()
        if len(s) < 15:
            continue
        s = s[0].upper() + s[1:]
        if not s.endswith(('.', '!', '?')):
            s += '.'
        # Take only the first clear sentence if it's a run-on
        first_period = s.find('. ')
        if first_period > 20 and first_period < len(s) - 5:
            s = s[:first_period + 1]
        # Truncate overly long sentences
        words = s.split()
        if len(words) > 30:
            s = ' '.join(words[:30]) + '.'
        cleaned.append(s)

    if not cleaned:
        return ""

    # Detect main topics
    stop = {'the','a','an','is','are','was','were','be','been','have','has','had',
            'do','does','did','will','would','could','should','can','may','might',
            'to','of','in','for','on','with','at','by','from','as','and','or','but',
            'not','no','so','if','this','that','it','its','we','you','they','he','she',
            'our','your','their','my','his','her','about','also','just','very','more',
            'all','some','any','than','then','now','when','what','which','who','how',
            'been','being','into','through','during','before','after','between','each',
            'there','here','where','while','only','other','these','those','such','like',
            'used','using','based','paper','study','results','however','thus','therefore',
            'chapter','example','given','make','made','image','objects','operations'}
    words = [w.lower() for w in re.findall(r'[a-zA-Z]+', full_text) if w.lower() not in stop and len(w) > 3]
    freq = Counter(words)
    top_topics = [w.capitalize() for w, _ in freq.most_common(4)]
    topic_phrase = ", ".join(top_topics[:3]) if top_topics else "the subject"

    word_count = len(full_text.split())

    # Build structured summary with clear sections
    lines = []
    lines.append(f"Overview: This document ({num_pages} pages, ~{word_count:,} words) covers {topic_phrase}.")
    lines.append("")
    lines.append("Key Points:")
    for i, s in enumerate(cleaned[:5], 1):
        lines.append(f"  {i}. {s}")

    return "\n".join(lines)


def pdf_summarize(pdf_path: str) -> dict:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    full_text, num_pages = _extract_pdf_text(pdf_path)

    if not full_text or len(full_text) < 50:
        raise ValueError("Could not extract text from PDF.")

    key_sentences = _tfidf_summarize(full_text, num_sentences=5)
    summary = _format_summary(key_sentences, full_text, num_pages)

    # Title from first meaningful line
    lines = [l.strip() for l in full_text.split('\n') if l.strip() and len(l.strip()) > 5]
    title = lines[0][:60] if lines else "Document Summary"

    word_count = len(full_text.split())

    return {
        "title": title,
        "summary": summary,
        "full_text": full_text[:3000],  # First 3000 chars as preview
        "stats": {
            "pages": num_pages,
            "word_count": word_count,
            "sentences": len(_split_sentences(full_text)),
        },
        "file": os.path.basename(pdf_path),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference/pdf_summarize.py <pdf_path>")
        sys.exit(1)
    result = pdf_summarize(sys.argv[1])
    print(json.dumps(result, indent=2))
