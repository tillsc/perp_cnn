from __future__ import annotations

import numpy as np

HF_REPO_ID = "tillsc/perp-cnn"
HF_FILENAME = "best.pt"
_DEFAULT_CONF = 0.25

_model = None


def load_model(repo_id: str = HF_REPO_ID):
    """Download model from HuggingFace Hub on first call, then return cached instance."""
    global _model
    if _model is None:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        model_path = hf_hub_download(repo_id=repo_id, filename=HF_FILENAME)
        _model = YOLO(model_path)
    return _model


def predict(image: np.ndarray, conf: float = _DEFAULT_CONF):
    """Run bowtip detection on a single image array (H×W×BGR or H×W×RGB).

    Returns the ultralytics Results list; access .boxes for detections.
    """
    model = load_model()
    return model(image, conf=conf)
