"""
DeepfakeDetector — supports both Keras (.keras / .h5) and PyTorch (.pth) models.
Auto-detects which framework to use based on the file extension.

KEY FIXES:
  1. Xception preprocessing: uses preprocess_input() → scales to [-1, 1]
     NOT /255.0 (that causes wrong scores)
  2. BATCH inference: all frames predicted in ONE model.predict() call
     instead of a per-frame loop (which caused the 85% stall / timeout)
  3. TensorFlow graph warm-up on load so first real call is fast
"""

import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMG_SIZE = 224


# ─────────────────────────────────────────────────────────────────────────────
# Face detector helper
# ─────────────────────────────────────────────────────────────────────────────

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_and_crop_face(frame: np.ndarray) -> Optional[np.ndarray]:
    """Detect the largest face and return a 224x224 BGR crop, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(dets) == 0:
        return None
    x, y, w, h = max(dets, key=lambda d: d[2] * d[3])
    crop = frame[max(0, y):y + h, max(0, x):x + w]
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE)) if crop.size > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────────────────────────────────────

class DeepfakeDetector:
    """
    Loads your trained model and exposes:
        detect(frames: list[np.ndarray]) -> dict

    Supports:
        Keras  -> .keras or .h5   (Xception / EfficientNet image classifiers)
        PyTorch -> .pth            (EfficientNet-B4 + BiLSTM sequence model)
    """

    def __init__(self, model_path: str, n_frames: int = 16):
        self.n_frames    = n_frames
        self.model_path  = model_path
        self.backend     = None
        self.model       = None
        self.device      = "cpu"
        self._keras_mode = "image"

        self._load_model(model_path)
        logger.info(
            f"DeepfakeDetector ready | backend={self.backend} "
            f"| keras_mode={self._keras_mode} | device={self.device}"
        )

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            logger.warning(
                f"Model not found at '{model_path}'. "
                "Running in DEMO mode — random scores.\n"
                "Place your model file at that path and restart."
            )
            return

        ext = os.path.splitext(model_path)[1].lower()
        logger.info(f"Loading model from '{model_path}'  (ext={ext})")

        if ext in (".keras", ".h5"):
            self._load_keras(model_path)
        elif ext in (".pth", ".pt"):
            self._load_pytorch(model_path)
        else:
            logger.warning(f"Unknown extension '{ext}'. Trying Keras first...")
            try:
                self._load_keras(model_path)
            except Exception:
                self._load_pytorch(model_path)

    def _load_keras(self, model_path: str):
        try:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow as tf

            self.model   = tf.keras.models.load_model(model_path)
            self.backend = "keras"

            try:
                inp_shape = self.model.input_shape
                logger.info(f"  Keras model input shape: {inp_shape}")
                if len(inp_shape) == 5:
                    self._keras_mode = "sequence"
                    logger.info("  Detected: SEQUENCE model")
                else:
                    self._keras_mode = "image"
                    logger.info("  Detected: IMAGE model (Xception/EfficientNet style)")
            except Exception as e:
                logger.warning(f"  Could not read input shape: {e} — defaulting to image mode")
                self._keras_mode = "image"

            # Warm up the TF graph NOW at startup so first real call is fast
            logger.info("  Warming up TensorFlow graph...")
            try:
                dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
                self.model.predict(dummy, verbose=0)
                logger.info("  Warm-up complete")
            except Exception as e:
                logger.warning(f"  Warm-up failed (non-fatal): {e}")

        except ImportError:
            raise RuntimeError("TensorFlow not installed. Run: pip install tensorflow")
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model: {e}")

    def _load_pytorch(self, model_path: str):
        try:
            import torch
            from torchvision import transforms
            from torchvision.models import efficientnet_b4
            import torch.nn as nn

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt        = torch.load(model_path, map_location=self.device)
            saved_args  = ckpt.get("args", {})

            class _Model(nn.Module):
                CNN_FEAT_DIM = 1792
                def __init__(self, hidden_dim=256, lstm_layers=2):
                    super().__init__()
                    bb = efficientnet_b4(weights=None)
                    self.cnn  = nn.Sequential(bb.features, bb.avgpool, nn.Flatten())
                    hd = hidden_dim // 2
                    self.lstm = nn.LSTM(
                        self.CNN_FEAT_DIM, hd, lstm_layers,
                        batch_first=True, bidirectional=True
                    )
                    self.head = nn.Sequential(
                        nn.Dropout(0.0), nn.Linear(hidden_dim, 128),
                        nn.ReLU(), nn.Linear(128, 1)
                    )
                def forward(self, x):
                    B, T, C, H, W = x.shape
                    f = self.cnn(x.view(B * T, C, H, W)).view(B, T, -1)
                    out, _ = self.lstm(f)
                    return self.head(out[:, -1, :]).squeeze(-1)

            model = _Model(
                hidden_dim  = saved_args.get("hidden_dim", 256),
                lstm_layers = saved_args.get("lstm_layers", 2),
            ).to(self.device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            self.model      = model
            self.backend    = "torch"
            self._torch     = torch
            self._transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            logger.info(f"  PyTorch model loaded on {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {e}")

    def detect(self, frames: list) -> dict:
        if self.model is None:
            import random
            return self._format_result(random.uniform(0.0, 1.0), 0)

        if not frames:
            return self._format_result(0.0, 0)

        processed = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
            processed.append(rgb)

        while len(processed) < self.n_frames:
            processed.append(processed[-1])
        processed = processed[:self.n_frames]

        try:
            if self.backend == "keras":
                prob = self._infer_keras(processed)
            else:
                prob = self._infer_torch(processed)
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return self._format_result(0.0, len(frames))

        return self._format_result(prob, len(frames))

    def _infer_keras(self, frames: list) -> float:
        if self._keras_mode == "image":
            return self._infer_keras_image_batch(frames)
        else:
            return self._infer_keras_sequence(frames)

    def _infer_keras_image_batch(self, frames: list) -> float:
        """
        FIX: ONE batched predict() call instead of 16 individual calls.

        Old broken approach — 16 slow calls:
            for frame in frames:
                model.predict(frame)   # slow, repeats TF graph overhead 16x

        New correct approach — 1 fast call:
            batch = stack(all 16 frames)       # (16, 224, 224, 3)
            preds = model.predict(batch)       # single call, fast
            prob  = mean(preds)
        """
        try:
            from tensorflow.keras.applications.xception import preprocess_input
        except ImportError:
            def preprocess_input(x):
                return (x.astype(np.float32) / 127.5) - 1.0

        # Stack all frames → (N, 224, 224, 3) and preprocess in one shot
        batch = np.stack([f.astype(np.float32) for f in frames], axis=0)
        batch = preprocess_input(batch)

        # SINGLE predict call for the whole batch
        preds = self.model.predict(batch, verbose=0)
        preds = np.array(preds)

        # Handle different output shapes
        if preds.ndim == 1:
            probs = preds                  # (N,)  sigmoid per sample
        elif preds.shape[1] == 1:
            probs = preds[:, 0]            # (N,1) sigmoid
        elif preds.shape[1] == 2:
            probs = preds[:, 1]            # (N,2) softmax → fake class = index 1
        else:
            probs = np.max(preds, axis=1)  # fallback

        return float(np.clip(np.mean(probs), 0.0, 1.0))

    def _infer_keras_sequence(self, frames: list) -> float:
        arr  = np.stack(frames).astype(np.float32) / 255.0
        arr  = arr[np.newaxis, ...]
        pred = self.model.predict(arr, verbose=0)
        pred = np.array(pred).flatten()
        prob = float(pred[1]) if len(pred) == 2 else float(np.mean(pred))
        return float(np.clip(prob, 0.0, 1.0))

    def _infer_torch(self, frames: list) -> float:
        import torch
        with torch.no_grad():
            tensor = torch.stack([self._transform(f) for f in frames])
            tensor = tensor.unsqueeze(0).to(self.device)
            logit  = self.model(tensor)
            return float(torch.sigmoid(logit).item())

    @staticmethod
    def _format_result(prob: float, n_frames: int) -> dict:
        confidence = "HIGH" if prob > 0.80 else "MEDIUM" if prob > 0.55 else "LOW"
        return {
            "fake_probability": round(prob, 4),
            "real_probability": round(1.0 - prob, 4),
            "is_fake":          prob > 0.65,
            "confidence":       confidence,
            "n_frames_used":    n_frames,
        }
