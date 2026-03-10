"""
DeepFake Detector — FastAPI Backend
====================================
Serves:
  POST /api/join           — bot joins a Zoom meeting
  POST /api/leave          — bot leaves
  GET  /api/status         — current session status
  GET  /api/participants   — live detection results
  POST /api/analyze-video  — upload a video file for deepfake analysis
  WS   /ws/live            — real-time WebSocket stream to dashboard
"""

import asyncio
import base64
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
from zoom_bot import ZoomBot
from detector import DeepfakeDetector, detect_and_crop_face
from websocket_manager import ConnectionManager

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("main")

import concurrent.futures

# ── Global singletons ─────────────────────────────────────────────────────────
ws_manager  = ConnectionManager()
detector    = DeepfakeDetector(settings.MODEL_PATH)
zoom_bot: Optional[ZoomBot] = None

# Shared thread-pool for CPU-bound inference calls.
_inference_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 DeepFake Detector backend starting...")
    logger.info(f"   Model  : {settings.MODEL_PATH}")
    logger.info(f"   Device : {detector.device}")
    yield
    logger.info("🛑 Shutting down...")
    if zoom_bot:
        await zoom_bot.leave()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepFake Zoom Detector",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend at http://localhost:8000 ───────────────────────────────────
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/app", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
    logger.info(f"Frontend served at http://localhost:{settings.PORT}/app")


# ── Request / Response schemas ────────────────────────────────────────────────
class JoinRequest(BaseModel):
    meeting_id: str
    password: str = ""
    display_name: str = "DeepFake Guardian"


class JoinResponse(BaseModel):
    success: bool
    message: str
    session_id: str = ""


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/join", response_model=JoinResponse)
async def join_meeting(req: JoinRequest):
    global zoom_bot
    try:
        zoom_bot = ZoomBot(
            meeting_id   = req.meeting_id,
            password     = req.password,
            display_name = req.display_name,
            detector     = detector,
            ws_manager   = ws_manager,
        )
        session_id = await zoom_bot.join()
        logger.info(f"Bot joined meeting {req.meeting_id} | session={session_id}")
        return JoinResponse(success=True, message="Bot joined meeting", session_id=session_id)
    except Exception as e:
        logger.error(f"Failed to join: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/leave")
async def leave_meeting():
    global zoom_bot
    if zoom_bot is None:
        raise HTTPException(status_code=400, detail="No active meeting session")
    await zoom_bot.leave()
    zoom_bot = None
    await ws_manager.broadcast({"type": "session_ended"})
    return {"success": True, "message": "Bot left meeting"}


@app.get("/api/status")
async def get_status():
    if zoom_bot is None:
        return {"active": False, "meeting_id": None, "participants": 0}
    return {
        "active":          zoom_bot.is_active,
        "meeting_id":      zoom_bot.meeting_id,
        "session_id":      zoom_bot.session_id,
        "participants":    len(zoom_bot.participants),
        "start_time":      zoom_bot.start_time,
        "frames_analyzed": zoom_bot.frames_analyzed,
    }


@app.get("/api/participants")
async def get_participants():
    if zoom_bot is None:
        return {"participants": []}
    return {"participants": list(zoom_bot.participants.values())}


@app.get("/api/alerts")
async def get_alerts():
    if zoom_bot is None:
        return {"alerts": []}
    return {"alerts": zoom_bot.alert_log[-50:]}


# ── Live Frame Analysis (Webcam) ──────────────────────────────────────────────

class FrameRequest(BaseModel):
    frames: List[str]   # list of base64 JPEG data URLs


@app.post("/api/analyze-frame")
async def analyze_frame(req: FrameRequest):
    """
    Accepts a list of base64-encoded JPEG frames from the browser webcam.
    Returns fake probability + face bounding boxes for canvas overlay.
    """
    if not req.frames:
        raise HTTPException(status_code=400, detail="No frames provided")

    # Decode base64 frames → numpy BGR arrays
    decoded = []
    for data_url in req.frames:
        try:
            # Strip data:image/jpeg;base64, prefix
            if "," in data_url:
                data_url = data_url.split(",", 1)[1]
            jpg_bytes = base64.b64decode(data_url)
            arr       = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                decoded.append(frame)
        except Exception:
            continue

    if not decoded:
        raise HTTPException(status_code=400, detail="Could not decode any frames")

    # Detect faces in the last frame for bounding box coordinates
    last_frame = decoded[-1]
    face_boxes = _detect_face_boxes(last_frame)

    # Prepare face crops for model
    face_crops = []
    if face_boxes:
        # Use detected face regions
        for (x, y, w, h) in face_boxes:
            crop = last_frame[max(0,y):y+h, max(0,x):x+w]
            if crop.size > 0:
                face_crops.append(cv2.resize(crop, (224, 224)))
    
    if not face_crops:
        # Fallback: resize full frames
        face_crops = [cv2.resize(f, (224, 224)) for f in decoded]

    # Run detection in thread pool (non-blocking)
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_inference_executor, detector.detect, face_crops)

    # Build face box list with labels for frontend canvas
    isFake = result["is_fake"]
    prob   = result["fake_probability"]
    label  = "FAKE" if isFake else "REAL"

    # Scale box coords to original frame size
    fh, fw = last_frame.shape[:2]
    faces_out = []
    for (x, y, w, h) in face_boxes:
        faces_out.append({
            "x":     int(x),
            "y":     int(y),
            "w":     int(w),
            "h":     int(h),
            "label": label,
            "prob":  round(prob, 3),
        })

    return {
        "fake_probability": result["fake_probability"],
        "real_probability": result["real_probability"],
        "is_fake":          result["is_fake"],
        "confidence":       result["confidence"],
        "n_frames_used":    result["n_frames_used"],
        "faces":            faces_out,
    }


@app.get("/api/mock-preview.jpg")
async def mock_preview():
    """
    Returns the latest frame seen by the backend mock source as JPEG.
    Frontend uses this when browser camera access is unavailable because the
    backend already owns the webcam device.
    """
    if zoom_bot is None or settings.INTEGRATION_MODE != "mock":
        raise HTTPException(status_code=404, detail="Mock preview unavailable")

    jpg = zoom_bot.get_mock_preview_jpeg()
    if not jpg:
        raise HTTPException(status_code=503, detail="Preview not ready yet")

    return Response(
        content=jpg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


# FIX: instantiate once at module load — CascadeClassifier construction is
# expensive (~10–30 ms); calling it on every /api/analyze-frame request was
# a significant hidden latency.
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _detect_face_boxes(frame: np.ndarray) -> list:
    """Return list of (x, y, w, h) for all detected faces."""
    # FIX: downscale to 480p before detection — Haar cascade is O(n_pixels)
    # so a 1080p frame takes ~5× longer than a 480p one.
    h, w = frame.shape[:2]
    scale = min(1.0, 480 / h)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    dets = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(dets) == 0:
        return []
    # Scale detections back to original resolution
    inv = 1.0 / scale
    return [(int(x * inv), int(y * inv), int(w_ * inv), int(h_ * inv)) for (x, y, w_, h_) in dets]


# ── Video Upload & Analysis ───────────────────────────────────────────────────

def analyze_video_file(video_path: str, sample_every: int = 10) -> dict:
    """
    Analyze a video file for deepfakes.
    Samples frames, detects faces, runs model, returns per-segment and overall result.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    duration_sec = total_frames / fps

    logger.info(f"Analyzing video: {total_frames} frames, {fps:.1f} fps, {duration_sec:.1f}s")

    segment_results = []
    frame_buffer    = []
    frame_idx       = 0
    segment_idx     = 0
    SEGMENT_FRAMES  = 16   # frames per detection segment

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every Nth frame
        if frame_idx % sample_every == 0:
            # Try face detection first; fall back to full resized frame
            face = detect_and_crop_face(frame)
            if face is None:
                face = cv2.resize(frame, (224, 224))
            frame_buffer.append(face)

            # Once we have enough frames, run a detection segment
            if len(frame_buffer) >= SEGMENT_FRAMES:
                result = detector.detect(frame_buffer[-SEGMENT_FRAMES:])
                timestamp = (frame_idx / fps)
                segment_results.append({
                    "segment":          segment_idx,
                    "timestamp_sec":    round(timestamp, 2),
                    "timestamp_label":  _fmt_time(timestamp),
                    "fake_probability": result["fake_probability"],
                    "real_probability": result["real_probability"],
                    "is_fake":          result["is_fake"],
                    "confidence":       result["confidence"],
                })
                segment_idx += 1

        frame_idx += 1

    cap.release()

    if not segment_results:
        return {
            "error": "No faces or frames could be analyzed in this video",
            "total_frames": total_frames,
        }

    # ── Overall verdict ───────────────────────────────────────────────────────
    all_probs    = [s["fake_probability"] for s in segment_results]
    avg_prob     = float(np.mean(all_probs))
    max_prob     = float(np.max(all_probs))
    fake_segments = sum(1 for s in segment_results if s["is_fake"])
    fake_ratio   = fake_segments / len(segment_results)

    # Verdict: fake if average OR majority of segments say fake
    is_fake_overall = avg_prob > 0.55 or fake_ratio > 0.4

    if avg_prob > 0.75:
        overall_confidence = "HIGH"
    elif avg_prob > 0.50:
        overall_confidence = "MEDIUM"
    else:
        overall_confidence = "LOW"

    return {
        "verdict":              "FAKE" if is_fake_overall else "REAL",
        "is_fake":              is_fake_overall,
        "avg_fake_probability": round(avg_prob, 4),
        "max_fake_probability": round(max_prob, 4),
        "overall_confidence":   overall_confidence,
        "fake_segments":        fake_segments,
        "total_segments":       len(segment_results),
        "fake_ratio":           round(fake_ratio, 3),
        "duration_sec":         round(duration_sec, 1),
        "total_frames":         total_frames,
        "frames_analyzed":      frame_idx,
        "segments":             segment_results,
    }


def _fmt_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Upload a video file (mp4, avi, mov, mkv) and get deepfake detection results.
    """
    # Validate file type
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}"
        )

    # Save to temp file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        logger.info(f"Received video: {file.filename} ({file_size_mb:.1f} MB)")

        # Limit file size to 200MB
        if file_size_mb > 200:
            raise HTTPException(status_code=400, detail="File too large. Max 200MB.")

        # Run analysis in thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, analyze_video_file, tmp_path)

        result["filename"] = file.filename
        result["file_size_mb"] = round(file_size_mb, 2)

        logger.info(
            f"Analysis done: {file.filename} → {result.get('verdict')} "
            f"(avg={result.get('avg_fake_probability', 0):.2%})"
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    logger.info(f"WebSocket client connected | total={len(ws_manager.active)}")
    try:
        if zoom_bot:
            await websocket.send_json({
                "type": "init",
                "participants": list(zoom_bot.participants.values()),
                "status": {
                    "active":     zoom_bot.is_active,
                    "meeting_id": zoom_bot.meeting_id,
                }
            })
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info(f"WebSocket client disconnected | total={len(ws_manager.active)}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info",
    )
