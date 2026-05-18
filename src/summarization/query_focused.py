"""
Query-focused summarization module.

Allows users to ask natural language questions about a video and get
timestamped answers. Uses semantic search over transcript chunks,
visual captions, and (optionally) CLIP visual features.

Examples:
  - "summarize only the thermodynamics part"
  - "find where the speaker explains derivatives"
  - "show the equations"

Usage:
    from src.summarization.query_focused import query_summarize
    results = query_summarize(
        query="explain thermodynamics",
        transcript_chunks=[{"text": ..., "start": ..., "end": ...}],
        captions=[{"text": ..., "timestamp": ...}],
    )
"""

import numpy as np
from typing import Optional


def query_summarize(
    query: str,
    transcript_chunks: list[dict],
    captions: Optional[list[dict]] = None,
    clip_features: Optional[np.ndarray] = None,
    top_k: int = 5,
    text_weight: float = 0.7,
    visual_weight: float = 0.3,
) -> dict:
    """
    Find the most relevant video segments for a user query.

    Args:
        query: natural language question or topic
        transcript_chunks: list of {"text": str, "start": float, "end": float}
        captions: optional list of {"text": str, "timestamp": str}
        clip_features: optional (N, 512) CLIP visual features for visual search
        top_k: number of results to return
        text_weight: weight for text-based similarity (0-1)
        visual_weight: weight for visual similarity (0-1, only if clip_features given)

    Returns:
        dict with:
          - query: original query
          - results: list of matched segments with scores and timestamps
          - num_searched: total chunks searched
    """
    import os
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")

    from src.features.text_embeddings import encode_texts, encode_query, text_similarity

    if not transcript_chunks:
        return {"query": query, "results": [], "num_searched": 0}

    # Combine transcript and caption texts for richer search
    search_items = []
    for chunk in transcript_chunks:
        search_items.append({
            "text": chunk["text"],
            "start": chunk.get("start", 0.0),
            "end": chunk.get("end", 0.0),
            "source": "transcript",
        })

    if captions:
        for cap in captions:
            ts = cap.get("timestamp", "0:00")
            parts = ts.split(":")
            try:
                sec = int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else float(parts[0])
            except (ValueError, IndexError):
                sec = 0.0
            search_items.append({
                "text": cap["text"],
                "start": sec,
                "end": sec + 2.0,
                "source": "caption",
            })

    texts = [item["text"] for item in search_items]
    query_emb = encode_query(query)

    # Text similarity
    text_embs = encode_texts(texts)
    text_scores = text_similarity(text_embs, query_emb)

    # Visual similarity (if CLIP features available)
    visual_scores = np.zeros(len(search_items), dtype=np.float32)
    if clip_features is not None and len(clip_features) > 0:
        try:
            from src.features.clip_embeddings import encode_text_query, clip_similarity
            clip_query = encode_text_query(query)

            # Map each search item to nearest video segment
            for i, item in enumerate(search_items):
                seg_idx = int(item["start"] / 2.0)  # approximate segment index
                seg_idx = min(seg_idx, len(clip_features) - 1)
                seg_idx = max(0, seg_idx)
                visual_scores[i] = float(clip_features[seg_idx] @ clip_query)
        except (ImportError, Exception):
            visual_weight = 0.0
            text_weight = 1.0

    # Combined score
    if clip_features is not None and visual_weight > 0:
        combined = text_weight * text_scores + visual_weight * visual_scores
    else:
        combined = text_scores

    # Rank and select top-k
    ranked_indices = np.argsort(combined)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices):
        item = search_items[idx]
        start_sec = item["start"]
        m, s = divmod(int(start_sec), 60)

        results.append({
            "rank": rank + 1,
            "text": item["text"],
            "timestamp": f"{m}:{s:02d}",
            "start_sec": round(item["start"], 1),
            "end_sec": round(item["end"], 1),
            "score": round(float(combined[idx]), 4),
            "text_score": round(float(text_scores[idx]), 4),
            "visual_score": round(float(visual_scores[idx]), 4) if visual_weight > 0 else None,
            "source": item["source"],
        })

    return {
        "query": query,
        "results": results,
        "num_searched": len(search_items),
        "weights": {"text": text_weight, "visual": visual_weight},
    }


def query_summarize_with_context(
    query: str,
    transcript_chunks: list[dict],
    captions: Optional[list[dict]] = None,
    clip_features: Optional[np.ndarray] = None,
    top_k: int = 5,
    context_window: int = 1,
) -> dict:
    """
    Query-focused search with surrounding context.

    Same as query_summarize but also includes neighboring chunks
    for each match, providing more context.

    Args:
        context_window: number of chunks before/after each match to include
    """
    base_result = query_summarize(
        query, transcript_chunks, captions, clip_features, top_k
    )

    if not base_result["results"] or not transcript_chunks:
        return base_result

    # Build timestamp-to-index mapping for transcript
    chunk_starts = [c.get("start", 0.0) for c in transcript_chunks]

    enriched = []
    for match in base_result["results"]:
        if match["source"] != "transcript":
            enriched.append(match)
            continue

        # Find the matching chunk index
        best_idx = 0
        best_dist = float("inf")
        for i, s in enumerate(chunk_starts):
            dist = abs(s - match["start_sec"])
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        # Gather context window
        start_idx = max(0, best_idx - context_window)
        end_idx = min(len(transcript_chunks), best_idx + context_window + 1)

        context_texts = [transcript_chunks[i]["text"] for i in range(start_idx, end_idx)]
        match["context"] = " ".join(context_texts)
        match["context_range"] = {
            "start_sec": round(transcript_chunks[start_idx].get("start", 0.0), 1),
            "end_sec": round(transcript_chunks[end_idx - 1].get("end", 0.0), 1),
        }
        enriched.append(match)

    base_result["results"] = enriched
    return base_result
