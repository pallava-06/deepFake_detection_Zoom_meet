<<<<<<< HEAD
# deepFake_detection_Zoom_meet
=======
# 🛡️ DeepFake Guardian — Real-Time Zoom Deepfake Detection

Real-time deepfake detection for Zoom video calls using **EfficientNet-B4 + BiLSTM**.
The system joins Zoom as a bot, streams each participant's video, detects faces, runs
the trained model, and surfaces live alerts on a beautiful web dashboard.

---

## 📁 Project Structure

```
deepfake-zoom-detector/
├── backend/
│   ├── main.py             ← FastAPI server + WebSocket hub
│   ├── config.py           ← All settings (reads from .env)
│   ├── zoom_bot.py         ← Zoom integration (SDK / Recall.ai / Mock)
│   ├── detector.py         ← Model inference wrapper
│   ├── websocket_manager.py← WebSocket broadcast manager
│   └── models/             ← 📌 Place your best_model.pth here
├── frontend/
│   └── index.html          ← Live dashboard (open in browser)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚡ Quick Start (Mock Mode — no Zoom needed)

Mock mode uses your local webcam to simulate a meeting — perfect for testing
that your model and pipeline work before connecting to real Zoom calls.

### Step 1 — Copy your trained model

```bash
cp /path/to/checkpoints/best_model.pth backend/models/best_model.pth
```

> If you haven't trained the model yet, run `python train_deepfake_model.py` first.
> Without the model file, the system runs in **demo mode** with random scores.

### Step 2 — Install dependencies

```bash
cd deepfake-zoom-detector
pip install -r requirements.txt
```

### Step 3 — Configure environment

```bash
cp .env.example .env
# Edit .env — for mock mode, the defaults are already correct
```

### Step 4 — Start the backend

```bash
cd backend
python main.py
```

You should see:
```
🚀 DeepFake Detector backend starting...
   Model  : models/best_model.pth
   Device : cuda   (or cpu)
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 5 — Open the dashboard

```
open frontend/index.html
```
or just double-click `frontend/index.html` in your file manager.

### Step 6 — Start detection

1. Select **Mock** mode in the dashboard
2. Click **Join & Start Detection**
3. Watch real-time detection results appear per participant

---

## 🔌 Integration Modes

### Mode A — Mock (Local Webcam)
Best for: **Testing and development**

```env
INTEGRATION_MODE=mock
MOCK_VIDEO_SOURCE=0          # 0 = default webcam
# MOCK_VIDEO_SOURCE=/path/to/video.mp4   # or a video file
```

No Zoom credentials needed.

---

### Mode B — Zoom Meeting SDK (Recommended for production)
Best for: **Direct, low-latency production use**

```env
INTEGRATION_MODE=sdk
ZOOM_SDK_KEY=your_key
ZOOM_SDK_SECRET=your_secret
```

**SDK Setup (Linux only):**

1. Download the Zoom Meeting SDK for Linux from:
   https://marketplace.zoom.us → Build App → Meeting SDK

2. Follow the official install guide. The Python bindings (`zoom_meeting_sdk`)
   must be importable from your Python environment.

3. Your Zoom Marketplace App needs:
   - SDK App type (not OAuth)
   - Meeting SDK credentials (key + secret)

4. The bot joins as a participant with `display_name` — all participants
   will see it in the meeting. Inform them beforehand (consent!).

---

### Mode C — Recall.ai (Easiest cloud option)
Best for: **Multi-meeting scaling without SDK setup**

```env
INTEGRATION_MODE=recall
RECALL_API_KEY=your_recall_key
RECALL_REGION=us-west-2
```

1. Sign up at https://recall.ai
2. Get your API key from the dashboard
3. Recall handles bot creation, joining, and video streaming
4. Charges per bot-minute — check their pricing

---

## 🐳 Docker Deployment

```bash
# Copy your model first
cp checkpoints/best_model.pth backend/models/best_model.pth

# Copy and edit .env
cp .env.example .env

# Build and run
cd docker
docker-compose up --build
```

The API will be at http://localhost:8000

---

## 🌐 API Reference

| Method | Endpoint            | Description                        |
|--------|---------------------|------------------------------------|
| POST   | `/api/join`         | Bot joins a Zoom meeting           |
| POST   | `/api/leave`        | Bot leaves the meeting             |
| GET    | `/api/status`       | Session status + live stats        |
| GET    | `/api/participants` | All participants + detection scores|
| GET    | `/api/alerts`       | Last 50 deepfake alerts            |
| WS     | `/ws/live`          | Real-time detection WebSocket      |

### Join request body:
```json
{
  "meeting_id":   "123 456 7890",
  "password":     "abc123",
  "display_name": "DeepFake Guardian"
}
```

### WebSocket message types:
```
session_started    → meeting joined
session_ended      → meeting left
participant_joined → new participant detected
detection_update   → updated fake/real scores for a participant
deepfake_alert     → threshold crossed — deepfake detected!
```

---

## 🎛️ Tuning Detection

Edit `.env` to adjust sensitivity:

```env
ANALYZE_EVERY_N_FRAMES=15   # Lower = more frequent checks (uses more GPU)
FAKE_THRESHOLD=0.65         # Lower = more sensitive (more false positives)
ALERT_PERSIST_FRAMES=5      # Higher = fewer false alerts (needs more frames)
```

---

## 🔒 Privacy & Ethics

- **Always obtain explicit consent** from all meeting participants before running detection
- **No video is stored** — frames are processed in memory only
- **Transparency** — the bot appears as a visible participant in the meeting
- **Confidence scores** are shown, not binary verdicts, to avoid false accusations
- Comply with GDPR, HIPAA, and Zoom's Developer Terms of Service
- For high-stakes decisions, require human review of all AI flags

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not found` | Copy `best_model.pth` to `backend/models/` |
| `Cannot open video source` | Check webcam index; try `MOCK_VIDEO_SOURCE=1` |
| `CUDA out of memory` | Reduce `N_FRAMES=8` in `.env` |
| `WebSocket disconnects` | Ensure backend is running on port 8000 |
| `Zoom SDK ImportError` | Install SDK per Zoom's Linux guide; use Recall mode instead |
| Very slow inference | GPU is not being used; install CUDA-enabled PyTorch |

---

## 📊 Performance Benchmarks

| Hardware            | FPS (inference) | Latency   |
|---------------------|-----------------|-----------|
| NVIDIA RTX 3080     | ~25–30 FPS      | ~35ms     |
| NVIDIA RTX 4090     | ~45 FPS         | ~22ms     |
| CPU (Intel i9)      | ~4–6 FPS        | ~180ms    |

With `ANALYZE_EVERY_N_FRAMES=15` at 30fps, the model analyzes ~2 frames/sec,
which is sufficient for real-time deepfake detection while leaving GPU headroom.

---

## 📦 Tech Stack

| Layer        | Technology                         |
|--------------|-------------------------------------|
| ML Model     | PyTorch · EfficientNet-B4 · BiLSTM |
| Backend      | FastAPI · Uvicorn · asyncio        |
| Face detect  | OpenCV Haar Cascade                |
| Zoom capture | Zoom Meeting SDK / Recall.ai       |
| Frontend     | Vanilla HTML/CSS/JS (no framework) |
| Deployment   | Docker · docker-compose            |
>>>>>>> a799040 (80%)
