import json
import os
import re
import subprocess
import sys
import threading
import uuid

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "summaries")
SUMMARIZER_SCRIPT = os.path.join(BASE_DIR, "src", "inference", "summarize.py")
TEXT_SCRIPT = os.path.join(BASE_DIR, "src", "inference", "text_summarize.py")
DATASET_SCRIPT = os.path.join(BASE_DIR, "src", "data", "build_caption_dataset.py")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "src", "training", "train_caption.py")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def _allowed(filename):
    return os.path.splitext(filename)[1].lower() == ".mp4"


def _unique_name(filename):
    safe = secure_filename(filename)
    base, ext = os.path.splitext(safe)
    return f"{base}_{uuid.uuid4().hex[:8]}{ext}"


# ── Training state ─────────────────────────────────────────────────────────────

_job = {
    "phase": None,       # "dataset" | "model"
    "running": False,
    "done": False,
    "error": None,
    # dataset extraction
    "progress": 0,
    "total": 0,
    "speed": "",
    "eta": "",
    # model training
    "epoch": 0,
    "epochs": 30,
    "train_loss": None,
    "val_loss": None,
    "bleu": None,
    "history": [],       # [{epoch, train_loss, val_loss, bleu}]
    "log": [],           # last 60 lines
}
_proc = None
_job_lock = threading.Lock()


def _push_log(line):
    with _job_lock:
        _job["log"] = (_job["log"] + [line])[-60:]


def _parse_line(line):
    """Parse a stdout line from either training script and update _job."""
    # ── tqdm progress (dataset build) ──────────────────────────────────────
    # "Processing videos:  45%|...| 3154/7010 [06:12<07:35,  8.47it/s]"
    m = re.search(r'(\d+)/(\d+)\s+\[[\d:]+<([\d:]+),\s*([\d.]+)it/s\]', line)
    if m:
        with _job_lock:
            _job["progress"] = int(m.group(1))
            _job["total"] = int(m.group(2))
            _job["eta"] = m.group(3)
            _job["speed"] = f"{float(m.group(4)):.1f} vid/s"
        return

    # ── vocab / dataset saved ──────────────────────────────────────────────
    if "Vocabulary size:" in line:
        m = re.search(r'Vocabulary size:\s*(\d+)', line)
        if m:
            _push_log(f"Vocabulary built — {m.group(1)} words")
        return

    if "Total samples:" in line:
        _push_log(line.strip())
        return

    if "Dataset saved" in line:
        _push_log("Dataset saved successfully")
        return

    # ── training epoch ─────────────────────────────────────────────────────
    # "Epoch   5/ 30 | Train 3.2100 | Val 3.4500 | PPL 31.5 | BLEU-4 0.0234"
    m = re.search(
        r'Epoch\s+(\d+)/\s*(\d+)\s*\|.*?Train\s+([\d.]+).*?Val\s+([\d.]+).*?BLEU-4\s+([\d.]+)',
        line
    )
    if m:
        epoch      = int(m.group(1))
        epochs     = int(m.group(2))
        train_loss = float(m.group(3))
        val_loss   = float(m.group(4))
        bleu       = float(m.group(5))
        with _job_lock:
            _job["epoch"]      = epoch
            _job["epochs"]     = epochs
            _job["train_loss"] = train_loss
            _job["val_loss"]   = val_loss
            _job["bleu"]       = bleu
            _job["progress"]   = epoch
            _job["total"]      = epochs
            _job["history"].append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "bleu": bleu,
            })
        _push_log(
            f"Epoch {epoch}/{epochs} — train {train_loss:.4f} | "
            f"val {val_loss:.4f} | BLEU {bleu:.4f}"
        )
        return

    # ── best model saved ───────────────────────────────────────────────────
    if "best model saved" in line.lower() or "saved best" in line.lower():
        _push_log(line.strip())
        return

    # ── model parameters / device ──────────────────────────────────────────
    if "Parameters:" in line or "Device:" in line or "Samples:" in line or "Vocab" in line:
        _push_log(line.strip())


def _run_process(script, phase):
    global _proc
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with _job_lock:
        _job.update({
            "phase": phase, "running": True, "done": False, "error": None,
            "progress": 0, "total": 0, "speed": "", "eta": "",
            "epoch": 0, "train_loss": None, "val_loss": None, "bleu": None,
            "history": [], "log": [],
        })

    _proc = subprocess.Popen(
        [sys.executable, "-u", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=BASE_DIR,
        env=env,
    )

    for raw_line in _proc.stdout:
        # tqdm uses \r — split on it and take the last non-empty chunk
        for line in re.split(r'\r', raw_line):
            line = line.strip()
            if line:
                _parse_line(line)

    _proc.wait()
    with _job_lock:
        _job["running"] = False
        if _proc.returncode == 0:
            _job["done"] = True
            _push_log("Finished successfully.")
        else:
            _job["error"] = f"Process exited with code {_proc.returncode}"
            _push_log(f"Error — exit code {_proc.returncode}")


# ── Training endpoints ────────────────────────────────────────────────────────

@app.route("/train/start/<phase>", methods=["POST"])
def train_start(phase):
    with _job_lock:
        if _job["running"]:
            return jsonify({"error": "A job is already running"}), 400

    scripts = {
        "dataset": DATASET_SCRIPT,
        "model": TRAIN_SCRIPT,
    }
    if phase not in scripts:
        return jsonify({"error": f"Unknown phase '{phase}'"}), 400

    t = threading.Thread(target=_run_process, args=(scripts[phase], phase), daemon=True)
    t.start()
    return jsonify({"ok": True, "phase": phase})


@app.route("/train/stop", methods=["POST"])
def train_stop():
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
        with _job_lock:
            _job["running"] = False
            _job["error"] = "Stopped by user"
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "No running process"})


@app.route("/train/reset", methods=["POST"])
def train_reset():
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
    with _job_lock:
        _job.update({
            "phase": None, "running": False, "done": False, "error": None,
            "progress": 0, "total": 0, "speed": "", "eta": "",
            "epoch": 0, "train_loss": None, "val_loss": None, "bleu": None,
            "history": [], "log": [],
        })
    return jsonify({"ok": True})


@app.route("/train/status", methods=["GET"])
def train_status():
    with _job_lock:
        return jsonify(dict(_job))


# ── UI serving ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "index.html"))

@app.route("/styles.css")
def styles():
    return send_file(os.path.join(BASE_DIR, "styles.css"))


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


# ── File existence check ──────────────────────────────────────────────────────

@app.route("/train/files", methods=["GET"])
def train_files():
    return jsonify({
        "dataset": os.path.exists(os.path.join(BASE_DIR, "data/processed/msvtt/dataset.pkl")),
        "vocab":   os.path.exists(os.path.join(BASE_DIR, "outputs/models/caption_vocab.pkl")),
        "model":   os.path.exists(os.path.join(BASE_DIR, "outputs/models/caption_model.pt")),
    })


# ── Video summary ─────────────────────────────────────────────────────────────

@app.route("/summarize", methods=["POST"])
def summarize():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if not file or not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not _allowed(file.filename):
        return jsonify({"error": "Only .mp4 files are allowed"}), 400

    saved_name = _unique_name(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, saved_name)
    file.save(input_path)

    try:
        result = subprocess.run(
            [sys.executable, SUMMARIZER_SCRIPT, input_path],
            capture_output=True, text=True, check=True, cwd=BASE_DIR,
        )
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Summarization failed", "stderr": e.stderr}), 500

    output_name = os.path.splitext(saved_name)[0] + "_summary.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    if not os.path.exists(output_path):
        return jsonify({"error": "Output file not found"}), 500

    return send_file(
        output_path,
        as_attachment=True,
        download_name=os.path.splitext(file.filename)[0] + "_summary.mp4",
        mimetype="video/mp4",
    )


# ── Text summary ──────────────────────────────────────────────────────────────

@app.route("/text-summarize", methods=["POST"])
def text_summarize():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if not file or not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not _allowed(file.filename):
        return jsonify({"error": "Only .mp4 files are allowed"}), 400

    saved_name = _unique_name(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, saved_name)
    file.save(input_path)

    try:
        result = subprocess.run(
            [sys.executable, TEXT_SCRIPT, input_path],
            capture_output=True, text=True, check=True, cwd=BASE_DIR,
        )
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Text summarization failed", "stderr": e.stderr}), 500
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return jsonify({"error": "Model output was not valid JSON", "raw": result.stdout}), 500

    return jsonify(data)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=False, threaded=True)
