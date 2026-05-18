"""
FastAPI backend for AI Video Summarizer.

Replaces Flask with:
  - Async endpoints
  - Background task processing for long videos
  - Auto-generated API docs at /docs
  - Static file serving for frontend

Run:
    uvicorn api_fastapi:app --host 127.0.0.1 --port 5003 --reload
"""

import json
import os
import re
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Database ──────────────────────────────────────────────────────────────────

DB_URL = os.environ.get("DATABASE_URL", "dbname=video_summarizer")


def get_db():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    return conn


# ── Paths ─────────────────────────────────────────────────────────────────────

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "summaries")
SUMMARIZER_SCRIPT = os.path.join(BASE_DIR, "src", "inference", "summarize.py")
TEXT_SCRIPT = os.path.join(BASE_DIR, "src", "inference", "text_summarize.py")
PDF_SCRIPT = os.path.join(BASE_DIR, "src", "inference", "pdf_summarize.py")
DATASET_SCRIPT = os.path.join(BASE_DIR, "src", "data", "build_caption_dataset.py")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "src", "training", "train_caption.py")

MAX_UPLOAD_MB = 500

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Video Summarizer API",
    description="Multimodal ML video summarization system",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in (".mp4", ".pdf")


def _unique_name(filename: str) -> str:
    base, ext = os.path.splitext(filename)
    safe = re.sub(r'[^\w\-.]', '_', base)
    return f"{safe}_{uuid.uuid4().hex[:8]}{ext}"


def _save_message(chat_id: int, role: str, content, msg_type: str = "text"):
    conn = get_db()
    cur = conn.cursor()
    content_json = json.dumps(content) if not isinstance(content, str) else json.dumps({"text": content})
    cur.execute(
        """INSERT INTO messages (chat_id, role, content, message_type)
           VALUES (%s, %s, %s::jsonb, %s)""",
        (chat_id, role, content_json, msg_type),
    )
    cur.execute("UPDATE chats SET updated_at = NOW() WHERE id = %s", (chat_id,))

    # Auto-title from first AI message
    if role == "assistant":
        cur.execute("SELECT title FROM chats WHERE id = %s", (chat_id,))
        row = cur.fetchone()
        if row and row[0] == "New Chat":
            text = content.get("text", content.get("summary", content.get("title", ""))) if isinstance(content, dict) else str(content)
            title = text[:50].strip() or "Summary"
            if len(title) > 50:
                title = title[:47] + "..."
            cur.execute("UPDATE chats SET title = %s WHERE id = %s", (title, chat_id))
    conn.close()


# ── Static files & Frontend ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(BASE_DIR, "index.html")
    with open(html_path) as f:
        return f.read()


@app.get("/styles.css")
async def serve_css():
    return FileResponse(os.path.join(BASE_DIR, "styles.css"), media_type="text/css")


@app.get("/download/{filename}")
async def download_file(filename: str):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="video/mp4")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"ok": True, "engine": "fastapi"}


# ── Chat CRUD ─────────────────────────────────────────────────────────────────

@app.get("/chats")
async def list_chats():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT id, title, created_at FROM chats ORDER BY updated_at DESC")
    chats = cur.fetchall()
    conn.close()
    return [{**c, "created_at": c["created_at"].isoformat()} for c in chats]


@app.post("/chats")
async def create_chat():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("INSERT INTO chats (title) VALUES ('New Chat') RETURNING id, title, created_at")
    chat = cur.fetchone()
    conn.close()
    return {**chat, "created_at": chat["created_at"].isoformat()}


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: int):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT id, title, created_at FROM chats WHERE id = %s", (chat_id,))
    chat = cur.fetchone()
    if not chat:
        conn.close()
        raise HTTPException(404, "Chat not found")

    cur.execute(
        "SELECT id, role, content, message_type, created_at FROM messages WHERE chat_id = %s ORDER BY created_at",
        (chat_id,),
    )
    messages = cur.fetchall()
    conn.close()

    return {
        **chat,
        "created_at": chat["created_at"].isoformat(),
        "messages": [{**m, "created_at": m["created_at"].isoformat()} for m in messages],
    }


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE chat_id = %s", (chat_id,))
    cur.execute("DELETE FROM chats WHERE id = %s", (chat_id,))
    conn.close()
    return {"ok": True}


# ── Summarization ─────────────────────────────────────────────────────────────

@app.post("/summarize")
async def summarize(
    file: UploadFile = File(...),
    chat_id: str = Form(None),
):
    """Video-to-Video summarization using XGBoost importance scoring."""
    if not file.filename or not _allowed(file.filename):
        raise HTTPException(400, "Only .mp4 files are supported")

    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_UPLOAD_MB}MB)")

    vid_name = _unique_name(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, vid_name)
    with open(input_path, "wb") as f:
        f.write(content)

    if chat_id:
        _save_message(int(chat_id), "user", {"type": "video_upload", "filename": file.filename}, "upload")

    try:
        subprocess.run(
            [sys.executable, SUMMARIZER_SCRIPT, input_path],
            capture_output=True, text=True, check=True, cwd=BASE_DIR,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Summarization failed: {e.stderr[-500:]}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    output_name = os.path.splitext(vid_name)[0] + "_summary.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    if not os.path.exists(output_path):
        raise HTTPException(500, "Output file not found")

    metrics_path = output_path.replace(".mp4", "_metrics.json")
    video_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as mf:
            video_metrics = json.load(mf)

    if chat_id:
        _save_message(int(chat_id), "assistant", {"type": "video_summary", "metrics": video_metrics}, "video_result")

    return {"video_url": f"/download/{output_name}", "metrics": video_metrics}


@app.post("/text-summarize")
async def text_summarize(
    file: UploadFile = File(...),
    chat_id: str = Form(None),
):
    """Video-to-Text summarization using Seq2Seq + Whisper + SBERT+MMR."""
    if not file.filename or not _allowed(file.filename):
        raise HTTPException(400, "Only .mp4 files are supported")

    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_UPLOAD_MB}MB)")

    vid_name = _unique_name(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, vid_name)
    with open(input_path, "wb") as f:
        f.write(content)

    if chat_id:
        _save_message(int(chat_id), "user", {"type": "video_upload", "filename": file.filename}, "upload")

    try:
        result = subprocess.run(
            [sys.executable, TEXT_SCRIPT, input_path],
            capture_output=True, text=True, check=True, cwd=BASE_DIR,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Text summarization failed: {e.stderr[-500:]}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise HTTPException(500, "Invalid JSON output from summarizer")

    if chat_id:
        _save_message(int(chat_id), "assistant", data, "text_result")

    return data


@app.post("/pdf-summarize")
async def pdf_summarize(
    file: UploadFile = File(...),
    chat_id: str = Form(None),
):
    """PDF-to-Text summarization using SBERT+MMR."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported")

    content = await file.read()
    pdf_name = _unique_name(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, pdf_name)
    with open(input_path, "wb") as f:
        f.write(content)

    if chat_id:
        _save_message(int(chat_id), "user", {"type": "pdf_upload", "filename": file.filename}, "upload")

    try:
        result = subprocess.run(
            [sys.executable, PDF_SCRIPT, input_path],
            capture_output=True, text=True, check=True, cwd=BASE_DIR,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"PDF summarization failed: {e.stderr[-500:]}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise HTTPException(500, "Invalid JSON output from summarizer")

    if chat_id:
        _save_message(int(chat_id), "assistant", data, "text_result")

    return data


# ── URL Summarization ─────────────────────────────────────────────────────────

@app.post("/url-summarize")
async def url_summarize(
    url: str = Form(...),
    mode: str = Form("text"),
    chat_id: str = Form(None),
):
    """Summarize a YouTube video by URL."""
    if not url.strip():
        raise HTTPException(400, "URL is required")

    vid_name = f"yt_{uuid.uuid4().hex[:8]}.mp4"
    dl_path = os.path.join(UPLOAD_FOLDER, vid_name)

    try:
        subprocess.run(
            ["yt-dlp", "-f", "mp4", "-o", dl_path, "--no-playlist", url],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(400, f"Download failed: {e.stderr[-300:]}")

    if not os.path.exists(dl_path):
        raise HTTPException(500, "Downloaded file not found")

    if chat_id:
        _save_message(int(chat_id), "user", {"type": "url_upload", "url": url, "mode": mode}, "upload")

    script = SUMMARIZER_SCRIPT if mode == "video" else TEXT_SCRIPT

    try:
        result = subprocess.run(
            [sys.executable, script, dl_path],
            capture_output=True, text=True, check=True, cwd=BASE_DIR,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Summarization failed: {e.stderr[-500:]}")
    finally:
        if os.path.exists(dl_path):
            os.remove(dl_path)

    if mode == "video":
        output_name = os.path.splitext(vid_name)[0] + "_summary.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        if not os.path.exists(output_path):
            raise HTTPException(500, "Output file not found")

        metrics_path = output_path.replace(".mp4", "_metrics.json")
        video_metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path) as mf:
                video_metrics = json.load(mf)

        if chat_id:
            _save_message(int(chat_id), "assistant", {"type": "video_summary", "metrics": video_metrics}, "video_result")
        return {"video_url": f"/download/{output_name}", "metrics": video_metrics}
    else:
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise HTTPException(500, "Invalid JSON output")
        if chat_id:
            _save_message(int(chat_id), "assistant", data, "text_result")
        return data


# ── Query-Focused Summarization ───────────────────────────────────────────────

@app.post("/query-summarize")
async def query_summarize_endpoint(body: dict):
    """Semantic search over a previously summarized video's transcript."""
    query = body.get("query", "").strip()
    chat_id = body.get("chat_id")

    if not query or not chat_id:
        raise HTTPException(400, "Provide chat_id and query")

    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """SELECT content FROM messages
           WHERE chat_id = %s AND message_type = 'text_result'
           ORDER BY created_at DESC LIMIT 1""",
        (chat_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, "No text summary found in this chat")

    content = row["content"] if isinstance(row["content"], dict) else json.loads(row["content"])
    transcript = content.get("transcript", "")

    if not transcript or len(transcript) < 20:
        raise HTTPException(404, "No transcript available for search")

    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    if len(sentences) <= 2 and len(transcript) > 100:
        words = transcript.split()
        chunk_size = max(12, min(20, len(words) // 8))
        sentences = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    chunks = []
    offset = 0.0
    for s in sentences:
        s = s.strip()
        if len(s) < 10:
            continue
        dur = max(2.0, len(s.split()) * 0.4)
        chunks.append({"text": s, "start": round(offset, 1), "end": round(offset + dur, 1)})
        offset += dur

    if not chunks:
        raise HTTPException(400, "Could not parse transcript")

    captions = [
        {"text": km.get("label", ""), "timestamp": km.get("timestamp", "0:00")}
        for km in content.get("key_moments", [])
    ]

    from src.summarization.query_focused import query_summarize
    result = query_summarize(
        query=query,
        transcript_chunks=chunks,
        captions=captions if captions else None,
        top_k=5,
    )
    return result


# ── Training Management ───────────────────────────────────────────────────────

_training_job = {"process": None, "phase": None, "output": []}


@app.post("/train/start/{phase}")
async def start_training(phase: str):
    """Start training: phase = 'dataset' or 'model'."""
    if _training_job["process"] and _training_job["process"].poll() is None:
        raise HTTPException(409, "Training already in progress")

    script = DATASET_SCRIPT if phase == "dataset" else TRAIN_SCRIPT

    _training_job["output"] = []
    _training_job["phase"] = phase
    _training_job["process"] = subprocess.Popen(
        [sys.executable, "-u", script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=BASE_DIR,
    )
    return {"status": "started", "phase": phase}


@app.get("/train/status")
async def training_status():
    proc = _training_job["process"]
    if not proc:
        return {"status": "idle"}

    # Read available output
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        _training_job["output"].append(line.strip())

    running = proc.poll() is None
    return {
        "status": "running" if running else "finished",
        "phase": _training_job["phase"],
        "output": _training_job["output"][-20:],
        "exit_code": proc.returncode if not running else None,
    }


@app.post("/train/stop")
async def stop_training():
    proc = _training_job["process"]
    if proc and proc.poll() is None:
        proc.terminate()
        return {"status": "stopped"}
    return {"status": "not_running"}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5003)
