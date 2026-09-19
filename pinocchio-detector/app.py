#!/usr/bin/env python3
"""Pinocchio — Deepfake Detection System.
Video upload → frame extraction → face/pose/brightness/lip-sync/motion/temporal analysis
→ Goldberg-style weighted score → verdict + timeline report.
"""

import os
import uuid
import time
import threading
import json
from datetime import datetime, timezone
from math import log as _log

from flask import (
    Flask,
    request,
    jsonify,
    send_file,
    render_template,
)
from werkzeug.utils import secure_filename
from io import BytesIO

import numpy as np
import cv2
import mediapipe as mp

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(HERE, "uploads")
FRAMES_FOLDER = os.path.join(HERE, "temp_frames")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    static_folder=os.path.join(HERE, "static"),
    static_url_path="/static",
    template_folder=os.path.join(HERE, "templates"),
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

ALLOWED = {"mp4", "mov", "avi", "mkv", "webm"}

# ---------------------------------------------------------------------------
# Thread-safe job store
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
MAX_JOB_AGE = 3600  # seconds


def _job(job_id: str, **kwargs) -> dict:
    base = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued for analysis...",
        "filename": kwargs.get("filename", ""),
        "file_size": kwargs.get("file_size", 0),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
    }
    base.update(kwargs)
    return base


def _purge_old_jobs() -> None:
    cutoff = time.time() - MAX_JOB_AGE
    now = time.time()
    with _jobs_lock:
        still = {}
        for jid, j in _jobs.items():
            try:
                ts = datetime.fromisoformat(j.get("created_at", "")).timestamp()
            except Exception:
                ts = 0.0
            if ts > cutoff:
                still[jid] = j
        _jobs.clear()
        _jobs.update(still)


# Spawn a background purge thread.
_thread = threading.Thread(target=_purge_old_jobs, daemon=True)
_thread.start()

# ---------------------------------------------------------------------------
# MediaPipe globals (initialized once)
# ---------------------------------------------------------------------------

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_detector = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_face_mesh_area=0.5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
pose_detector = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LIP_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 324, 297, 344, 288, 317, 8, 95]
_EYE_LEFT_IDX = [33, 133, 159, 145, 153, 154, 155, 246, 160, 159, 158, 173]
_EYE_RIGHT_IDX = [386, 263, 387, 368, 358, 347, 348, 349, 260, 378, 385, 377, 378]

# Goldberg-style weights (must sum to 100 %)
WEIGHTS = {
    "face": 30, "pose": 20, "brightness": 15, "lipsync": 25, "motion": 10,
}
THRESHOLDS = {
    "face": 0.55, "pose": 0.45, "brightness": 0.35, "lipsync": 0.40, "motion": 0.30,
}


def _safe_arange(start, stop, step):
    return [start + i * step for i in range(int((stop - start) / step))]


def _brightness_score(frames_brightness):
    if not frames_brightness:
        return 1.0
    arr = np.asarray(frames_brightness, dtype=float)
    mean_b = arr.mean()
    if mean_b < 1:
        return 1.0
    std_b = arr.std() / (mean_b + 1e-6)
    return max(0.0, 1.0 - min(1.0, std_b / 0.5))


def _motion_consistency(motion_vals):
    if not motion_vals:
        return 1.0
    arr = np.asarray(motion_vals, dtype=float)
    if arr.max() < 1e-6:
        return 1.0
    norm = arr / (arr.max() + 1e-6)
    entropy = -(norm * np.log2(norm + 1e-9)).sum() / len(norm)
    return float(np.clip(entropy / np.log2(len(norm) + 1e-9), 0, 1))


def _goldberg(raw: dict) -> tuple[float, str]:
    total = 0.0
    for k, w in WEIGHTS.items():
        total += raw.get(k, 0.0) * w / 100.0
    verdict = "GENUINE" if total >= 0.5 else "DEEPFAKE"
    return round(total, 3), verdict


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------


def _aspect_ratio(points, indices):
    """Aspect ratio heuristic from landmark points."""
    if len(indices) < 4:
        return 0.0
    pts = [points[i] for i in indices if i < len(p)]
    if len(pts) < 4:
        return 0.0
    # Mouth: vertical (top→bottom) / horizontal (left→right)
    top_y = min(p[1] for p in pts)
    bot_y = max(p[1] for p in pts)
    left_x = min(p[0] for p in pts)
    right_x = max(p[0] for p in pts)
    h = right_x - left_x + 1e-6
    v = bot_y - top_y
    return min(v / h, 1.0)


def analyze(job_id, filepath):
    """Run full analysis pipeline on a video file."""
    job = _jobs.get(job_id)
    if not job:
        return

    with _jobs_lock:
        job["status"] = "processing"
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        job["message"] = "Starting analysis…"

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        with _jobs_lock:
            job["status"] = "error"
            job["error"] = "Cannot open video file"
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    sample_step = max(1, n_frames_total // 50)
    frame_indices = []
    idx = 0
    while idx < n_frames_total:
        frame_indices.append(int(idx))
        idx += sample_step
    frame_indices.append(n_frames_total - 1)

    face_confidences = []
    presence_flags = []
    motion_vals = []
    brightness_vals = []
    lip_sync_vals = []
    pose_consistencies = []
    timeline = []

    prev_gray = None
    frame_idx = 0
    n_processed = 0

    for fidx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Face detection ---
        det_results = face_detector.process(rgb)
        present = bool(det_results.detections)
        conf = 0.0
        if present:
            conf = max(det_results.detections[0].score[0], conf)
        presence_flags.append(present)
        face_confidences.append(conf)

        # --- Brightness ---
        brightness_vals.append(float(gray.mean()))

        # --- Motion (optical flow) ---
        if prev_gray is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 5, 2
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                motion_vals.append(float(mag.mean()))
            except Exception:
                motion_vals.append(0.0)
        prev_gray = gray

        # --- Face mesh + lip sync proxy ---
        mesh_results = face_mesh.process(rgb)
        if mesh_results.multi_face_landmarks:
            lm = mesh_results.multi_face_landmarks[0]
            pts = [(l.x * frame.shape[1], l.y * frame.shape[0])
                   for l in lm.landmark]
            ratio = _aspect_ratio(pts, _LIP_IDX)
            lip_sync_vals.append(min(ratio / 0.3, 1.0))
        else:
            lip_sync_vals.append(0.0)

        # --- Pose consistency ---
        pose_results = pose_detector.process(rgb)
        if pose_results.pose_landmarks:
            n_land = len(pose_results.pose_landmarks.landmark)
            pose_consistencies.append(min(n_land / 33.0, 1.0))
        else:
            pose_consistencies.append(0.0)

        n_processed += 1
        pct = round(n_processed / len(frame_indices) * 100, 1)

        with _jobs_lock:
            job["progress"] = pct
            job["message"] = f"Analysing frame {frame_idx}/{n_frames_total}"

        if frame_idx % max(1, len(frame_indices) // 20) == 0:
            ts = fidx / (fps or 1.0)
            timeline.append({
                "frame": int(fidx),
                "time": round(ts, 2),
                "face_conf": round(conf, 3),
                "has_face": present,
                "brightness": round(brightness_vals[-1], 1),
            })

    cap.release()

    # --- Aggregate scores ---
    face_score = float(np.mean(face_confidences)) if face_confidences else 0.0
    face_score *= float(np.mean(presence_flags)) if presence_flags else 1.0
    pose_score = float(np.mean(pose_consistencies)) if pose_consistencies else 0.0
    brightness_score = _brightness_score(brightness_vals)
    lip_score = float(np.mean(lip_sync_vals)) if lip_sync_vals else 0.0
    motion_score = _motion_consistency(motion_vals)

    raw = {
        "face": face_score,
        "pose": pose_score,
        "brightness": brightness_score,
        "lipsync": lip_score,
        "motion": motion_score,
    }
    verdict_score, verdict = _goldberg(raw)

    report = {
        "job_id": job_id,
        "filename": job.get("filename", ""),
        "file_size": job.get("file_size", 0),
        "frame_count": n_frames_total,
        "processed_frames": n_processed,
        "scores": {k: round(v, 3) for k, v in raw.items()},
        "verdict": verdict,
        "score": verdict_score,
        "timeline": timeline,
        "thresholds": THRESHOLDS,
        "weights": WEIGHTS,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    with _jobs_lock:
        job["status"] = "completed"
        job["progress"] = 100
        job["message"] = f"Done — {verdict}"
        job["completed_at"] = report["completed_at"]
        job["result"] = report

        # Cleanup uploaded file after processing
    try:
        os.remove(filepath)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    f = request.files["video"]
    if not f or f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    ext = f.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported extension: {ext}"}), 400

    job_id = str(uuid.uuid4())
    filename = secure_filename(f.filename)
    file_size = request.content_length or 0
    filepath = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
    f.save(filepath)

    with _jobs_lock:
        _jobs[job_id] = _job(job_id, filename=filename, file_size=file_size, filepath=filepath)

    t = threading.Thread(target=analyze, args=(job_id, filepath), daemon=True)
    t.start()
    return jsonify({"job_id": job_id}), 202


@app.route("/status/<job_id>")
def status(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "message": job.get("message"),
        "verdict": job.get("result", {}).get("verdict") if job.get("result") else None,
    })


@app.route("/result/<job_id>")
def result(job_id):
    job = _jobs.get(job_id)
    if not job or not job.get("result"):
        return jsonify({"error": "No result yet"}), 404
    return jsonify(job["result"])


@app.route("/report/<job_id>")
def report(job_id):
    """Return downloadable JSON report."""
    job = _jobs.get(job_id)
    if not job or not job.get("result"):
        return jsonify({"error": "No result yet"}), 404
    report_data = job["result"]
    buf = BytesIO()
    buf.write(json.dumps(report_data, indent=2).encode())
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/json",
        as_attachment=True,
        download_name=f"pinocchio-report-{job_id}.json",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
