"""
Configuration — reads from environment / .env file
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Server ────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Zoom SDK credentials ──────────────────────────
    # Get from https://marketplace.zoom.us → SDK App
    ZOOM_SDK_KEY:    str = ""
    ZOOM_SDK_SECRET: str = ""
    ZOOM_JWT_TOKEN:  str = ""   # optional — for bot auth

    # ── Recall.ai (alternative to raw SDK) ───────────
    RECALL_API_KEY:  str = ""
    RECALL_REGION:   str = "us-west-2"

    # ── Model ─────────────────────────────────────────
    # FIX: was "model\xception..." — \x is a Python hex escape → broken path
    # Use forward slashes (works on Windows + Linux) or raw string
    MODEL_PATH: str = "model/xception_deepfake_image_5o.h5"

    # ── Detection settings ────────────────────────────
    ANALYZE_EVERY_N_FRAMES: int   = 15     # sample 1 frame every N frames (~2/sec at 30fps)
    FAKE_THRESHOLD:         float = 0.65   # probability above this = deepfake alert
    ALERT_PERSIST_FRAMES:   int   = 5      # consecutive detections before alert fires
    N_FRAMES:               int   = 16     # frames buffered per participant

    # ── Mock video source ─────────────────────────────
    # 0 = default webcam, or set a path to a video file
    MOCK_VIDEO_SOURCE: str = "0"

    # ── Integration mode ─────────────────────────────
    # "sdk"    → use Zoom Meeting SDK directly (requires Linux SDK install)
    # "recall" → use Recall.ai API (easier, cloud-managed)
    # "mock"   → local webcam simulation (for testing without Zoom)
    INTEGRATION_MODE: str = "mock"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
