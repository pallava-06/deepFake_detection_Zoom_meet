"""
ZoomBot — joins a Zoom meeting and streams video frames to the deepfake detector.

Supports three integration modes (set INTEGRATION_MODE in .env):
  "sdk"    — Zoom Meeting SDK for Linux (requires SDK install, see README)
  "recall" — Recall.ai managed bots (easiest, cloud-hosted)
  "mock"   — local webcam / video file (for testing without Zoom)

PERFORMANCE FIXES:
  1. cv2.VideoCapture.read() runs in a thread pool executor — no longer blocks
     the asyncio event loop (was the main cause of sluggishness).
  2. Frame buffer uses collections.deque(maxlen=N) — O(1) append/pop vs O(n)
     list.pop(0).
  3. _cache_mock_preview is rate-limited to every PREVIEW_EVERY frames — JPEG
     encoding is skipped on frames the frontend never sees.
  4. Frames are downscaled to 480p before entering the pipeline to reduce the
     cost of face detection and the subsequent resize to 224×224.
  5. Slightly relaxed sleep in the detection loop (0.005 s instead of 0.001 s)
     to give the event loop more breathing room between iterations.
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from typing import Optional

import cv2
import numpy as np

from config import settings
from detector import DeepfakeDetector, detect_and_crop_face
from websocket_manager import ConnectionManager

logger = logging.getLogger(__name__)

# How often (in mock frames) to re-encode the JPEG preview sent to the browser.
# 6 ≈ 5 fps preview at 30 fps capture — cheap to encode, smooth enough to watch.
_PREVIEW_EVERY = 6

# Downscale large frames to this height before processing (preserves aspect ratio).
# Face detection + resize to 224 is much faster on smaller frames.
_PIPELINE_MAX_HEIGHT = 480


def _downscale(frame: np.ndarray) -> np.ndarray:
    """Resize frame so its height ≤ _PIPELINE_MAX_HEIGHT (no-op if already small)."""
    h, w = frame.shape[:2]
    if h <= _PIPELINE_MAX_HEIGHT:
        return frame
    scale = _PIPELINE_MAX_HEIGHT / h
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────────────────────
# Participant state
# ─────────────────────────────────────────────────────────────────────────────

class Participant:
    def __init__(self, pid: str, name: str):
        self.id               = pid
        self.name             = name
        self.fake_probability = 0.0
        self.is_fake          = False
        self.confidence       = "LOW"
        self.frames_analyzed  = 0
        self.consecutive_fake = 0
        self.alert_triggered  = False
        self.last_updated     = time.time()

        # FIX: deque with a fixed max length — O(1) append+discard vs O(n) pop(0)
        self._frame_buffer: deque = deque(maxlen=settings.N_FRAMES)

    def push_frame(self, face_frame: np.ndarray):
        self._frame_buffer.append(face_frame)  # automatically drops oldest when full

    def get_frames(self) -> list:
        return list(self._frame_buffer)

    def update_result(self, result: dict):
        self.fake_probability  = result["fake_probability"]
        self.is_fake           = result["is_fake"]
        self.confidence        = result["confidence"]
        self.frames_analyzed  += result["n_frames_used"]
        self.last_updated      = time.time()

        if self.is_fake:
            self.consecutive_fake += 1
        else:
            self.consecutive_fake = 0

        if self.consecutive_fake >= settings.ALERT_PERSIST_FRAMES:
            self.alert_triggered = True

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "name":             self.name,
            "fake_probability": round(self.fake_probability, 3),
            "real_probability": round(1.0 - self.fake_probability, 3),
            "is_fake":          self.is_fake,
            "confidence":       self.confidence,
            "alert_triggered":  self.alert_triggered,
            "frames_analyzed":  self.frames_analyzed,
            "consecutive_fake": self.consecutive_fake,
            "last_updated":     self.last_updated,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ZoomBot
# ─────────────────────────────────────────────────────────────────────────────

class ZoomBot:
    """
    Joins a Zoom meeting (via SDK / Recall.ai / mock),
    grabs frames, runs detection, broadcasts results.
    """

    def __init__(
        self,
        meeting_id:   str,
        password:     str,
        display_name: str,
        detector:     DeepfakeDetector,
        ws_manager:   ConnectionManager,
    ):
        self.meeting_id   = meeting_id
        self.password     = password
        self.display_name = display_name
        self.detector     = detector
        self.ws_manager   = ws_manager

        self.session_id      = str(uuid.uuid4())[:8]
        self.is_active       = False
        self.start_time      = None
        self.frames_analyzed = 0
        self.participants: dict[str, Participant] = {}
        self.alert_log: list = []

        self._task: Optional[asyncio.Task] = None
        self._frame_counter    = 0
        self._preview_counter  = 0
        self._latest_mock_jpeg: Optional[bytes] = None

        # Shared thread-pool executor reference (set after join)
        self._executor = None

    # ── Join ──────────────────────────────────────────────────────────────────

    async def join(self) -> str:
        import concurrent.futures
        # Reuse a small dedicated thread pool for blocking I/O (cap.read, etc.)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        mode = settings.INTEGRATION_MODE
        logger.info(f"Joining meeting {self.meeting_id} via mode='{mode}'")

        if mode == "sdk":
            await self._join_via_sdk()
        elif mode == "recall":
            await self._join_via_recall()
        elif mode == "mock":
            await self._join_mock()
        else:
            raise ValueError(f"Unknown INTEGRATION_MODE: {mode}")

        self.is_active  = True
        self.start_time = time.time()

        self._task = asyncio.create_task(self._detection_loop())

        await self.ws_manager.broadcast({
            "type":       "session_started",
            "meeting_id": self.meeting_id,
            "session_id": self.session_id,
            "mode":       mode,
        })

        return self.session_id

    # ── Leave ─────────────────────────────────────────────────────────────────

    async def leave(self):
        self.is_active = False
        if self._task:
            self._task.cancel()
        await self._cleanup()
        if self._executor:
            self._executor.shutdown(wait=False)
        logger.info(f"Bot left meeting {self.meeting_id}")

    # ── Detection loop ────────────────────────────────────────────────────────

    async def _detection_loop(self):
        """
        Continuously reads frames from the active video source,
        detects faces, runs model, broadcasts results.
        """
        logger.info("Detection loop started")
        try:
            while self.is_active:
                frame = await self._read_frame()
                if frame is None:
                    await asyncio.sleep(0.03)
                    continue

                self._frame_counter += 1

                analyze_every = 5 if settings.INTEGRATION_MODE == "mock" else settings.ANALYZE_EVERY_N_FRAMES
                if self._frame_counter % analyze_every != 0:
                    # FIX: slightly larger yield — gives event loop more time for
                    # WebSocket sends and other coroutines between frame reads.
                    await asyncio.sleep(0.005)
                    continue

                participant_id   = getattr(frame, "participant_id", "user_0")
                participant_name = getattr(frame, "participant_name", "Test User")
                raw_frame = frame if isinstance(frame, np.ndarray) else frame.data

                await self._process_frame(participant_id, participant_name, raw_frame)
                await asyncio.sleep(0.005)

        except asyncio.CancelledError:
            logger.info("Detection loop cancelled")
        except Exception as e:
            logger.error(f"Detection loop error: {e}", exc_info=True)

    async def _process_frame(self, pid: str, name: str, raw_frame: np.ndarray):
        if pid not in self.participants:
            self.participants[pid] = Participant(pid, name)
            await self.ws_manager.broadcast({
                "type":           "participant_joined",
                "participant_id": pid,
                "name":           name,
            })
            logger.info(f"New participant: {name} ({pid})")

        p = self.participants[pid]

        # FIX: downscale before face detection — significantly reduces Haar
        # cascade cost on high-res frames (e.g. 1080p → 480p is ~5× faster).
        small = _downscale(raw_frame)

        if settings.INTEGRATION_MODE == "mock":
            face = cv2.resize(small, (224, 224))
        else:
            face = detect_and_crop_face(small)
            if face is None:
                return

        p.push_frame(face)

        if len(p.get_frames()) < 4:
            return

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor, self.detector.detect, p.get_frames()
        )

        p.update_result(result)
        self.frames_analyzed += 1

        await self.ws_manager.broadcast({
            "type":        "detection_update",
            "participant": p.to_dict(),
        })

        if p.alert_triggered and p.consecutive_fake == settings.ALERT_PERSIST_FRAMES:
            alert = {
                "type":           "deepfake_alert",
                "participant_id": pid,
                "name":           name,
                "probability":    p.fake_probability,
                "confidence":     p.confidence,
                "timestamp":      time.time(),
            }
            self.alert_log.append(alert)
            await self.ws_manager.broadcast(alert)
            logger.warning(
                f"🚨 DEEPFAKE ALERT: {name} | prob={p.fake_probability:.2%} | "
                f"confidence={p.confidence}"
            )


    # ══════════════════════════════════════════════════════════════════════════
    # Integration Mode Implementations
    # ══════════════════════════════════════════════════════════════════════════

    # ── MODE 1: Zoom Meeting SDK (Linux) ──────────────────────────────────────

    async def _join_via_sdk(self):
        try:
            import zoom_meeting_sdk as zoom_sdk   # noqa

            self._sdk = zoom_sdk.ZoomSDK()
            init_params = zoom_sdk.InitAuthSDKParams(
                sdk_key    = settings.ZOOM_SDK_KEY,
                sdk_secret = settings.ZOOM_SDK_SECRET,
            )
            self._sdk.initialize(init_params)

            join_params = zoom_sdk.JoinMeetingParams(
                meeting_number = self.meeting_id,
                password       = self.password,
                display_name   = self.display_name,
            )
            self._sdk.join_meeting(join_params)
            self._sdk.enable_raw_video(callback=self._sdk_video_callback)
            self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=30)
            logger.info("Zoom SDK bot joined and raw video enabled")

        except ImportError:
            raise RuntimeError(
                "Zoom Meeting SDK not installed. "
                "See README.md → 'Zoom SDK Setup' section, "
                "or set INTEGRATION_MODE=recall in .env to use Recall.ai instead."
            )

    def _sdk_video_callback(self, participant_id: str, yuv_frame, width: int, height: int):
        try:
            yuv = np.frombuffer(yuv_frame, dtype=np.uint8).reshape((height * 3 // 2, width))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

            class FrameObj: pass
            f = FrameObj()
            f.data             = bgr
            f.participant_id   = str(participant_id)
            f.participant_name = self._sdk.get_participant_name(participant_id)

            try:
                self._frame_queue.put_nowait(f)
            except asyncio.QueueFull:
                pass
        except Exception as e:
            logger.debug(f"SDK frame callback error: {e}")

    async def _read_frame_sdk(self):
        try:
            return await asyncio.wait_for(self._frame_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None


    # ── MODE 2: Recall.ai Managed Bots ───────────────────────────────────────

    async def _join_via_recall(self):
        try:
            import httpx

            if not settings.RECALL_API_KEY:
                raise ValueError("RECALL_API_KEY not set in .env")

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"https://{settings.RECALL_REGION}.recall.ai/api/v1/bot",
                    headers={"Authorization": f"Token {settings.RECALL_API_KEY}"},
                    json={
                        "meeting_url": f"https://zoom.us/j/{self.meeting_id}?pwd={self.password}",
                        "bot_name": self.display_name,
                        "recording_config": {"transcript": False},
                        "real_time_media": {
                            "rtmp_destination_url": None,
                            "websocket_video_destination_url": "__WILL_BE_SET__",
                        }
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                bot_data = resp.json()
                self._recall_bot_id = bot_data["id"]
                logger.info(f"Recall.ai bot created: {self._recall_bot_id}")

            self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=30)
            asyncio.create_task(self._recall_video_stream())

        except ImportError:
            raise RuntimeError("httpx not installed. Run: pip install httpx")

    async def _recall_video_stream(self):
        import websockets, base64, json as _json

        ws_url = f"wss://{settings.RECALL_REGION}.recall.ai/api/v1/bot/{self._recall_bot_id}/media-stream"
        async with websockets.connect(
            ws_url,
            extra_headers={"Authorization": f"Token {settings.RECALL_API_KEY}"}
        ) as ws:
            async for message in ws:
                data = _json.loads(message)
                if data.get("type") == "video_frame":
                    jpg_bytes = base64.b64decode(data["data"])
                    arr   = np.frombuffer(jpg_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    class FrameObj: pass
                    f = FrameObj()
                    f.data             = frame
                    f.participant_id   = data.get("participant_id", "unknown")
                    f.participant_name = data.get("participant_name", "Participant")
                    try:
                        self._frame_queue.put_nowait(f)
                    except asyncio.QueueFull:
                        pass

    async def _read_frame_recall(self):
        try:
            return await asyncio.wait_for(self._frame_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None


    # ── MODE 3: Mock (local webcam / video file) ──────────────────────────────

    async def _join_mock(self):
        source = getattr(settings, "MOCK_VIDEO_SOURCE", 0)
        try:
            src = int(source)
        except (ValueError, TypeError):
            src = str(source)

        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            logger.warning("Cannot open video source — using synthetic test frames")
            self._cap = None

        self.participants["mock_user"] = Participant("mock_user", "Mock Participant")
        logger.info(f"Mock mode active | source={src}")

    def _cap_read_blocking(self):
        """Blocking webcam read — must be called from a thread, not the event loop."""
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video
            return None
        return frame

    async def _read_frame_mock(self):
        if self._cap is None:
            # Synthetic frame path — stays in event loop (cheap numpy op)
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (220, 100), (420, 380), (200, 160, 120), -1)
        else:
            # FIX: run the blocking cap.read() in a thread pool so it does NOT
            # stall the event loop.  Previously this froze all WebSocket sends,
            # HTTP responses, and other async tasks until the read completed.
            loop  = asyncio.get_event_loop()
            frame = await loop.run_in_executor(self._executor, self._cap_read_blocking)
            if frame is None:
                return None

        # FIX: only JPEG-encode for the preview every _PREVIEW_EVERY frames —
        # imencode is surprisingly expensive (~1–2 ms) at 30 fps.
        self._preview_counter += 1
        if self._preview_counter % _PREVIEW_EVERY == 0:
            self._cache_mock_preview(frame)

        class FrameObj: pass
        f = FrameObj()
        f.data             = frame
        f.participant_id   = "mock_user"
        f.participant_name = "Mock Participant"

        # FIX: removed the unconditional asyncio.sleep(0.033) here.
        # The natural latency of cap.read() (which now runs in a thread) already
        # paces the loop at roughly the camera's native FPS without burning
        # the event loop on a 33 ms sleep that blocks everything else.
        return f


    # ── Unified frame reader ──────────────────────────────────────────────────

    async def _read_frame(self):
        mode = settings.INTEGRATION_MODE
        if mode == "sdk":
            return await self._read_frame_sdk()
        elif mode == "recall":
            return await self._read_frame_recall()
        else:
            return await self._read_frame_mock()


    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def _cleanup(self):
        mode = settings.INTEGRATION_MODE
        try:
            if mode == "sdk" and hasattr(self, "_sdk"):
                self._sdk.leave_meeting()
            elif mode == "recall" and hasattr(self, "_recall_bot_id"):
                import httpx
                async with httpx.AsyncClient() as client:
                    await client.delete(
                        f"https://{settings.RECALL_REGION}.recall.ai/api/v1/bot/{self._recall_bot_id}",
                        headers={"Authorization": f"Token {settings.RECALL_API_KEY}"},
                        timeout=10,
                    )
            elif mode == "mock" and hasattr(self, "_cap") and self._cap:
                self._cap.release()
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")
        finally:
            self._latest_mock_jpeg = None

    def _cache_mock_preview(self, frame: np.ndarray):
        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if ok:
            self._latest_mock_jpeg = enc.tobytes()

    def get_mock_preview_jpeg(self) -> Optional[bytes]:
        return self._latest_mock_jpeg
