"""Downloads the speaker-diarization models on first use (about 33 MB, no account needed).

They are pyannote's speaker-diarization-community-1 models (CC-BY-4.0) and segmentation-3.0
(MIT), converted to ONNX and published with this plugin's releases.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from typing import Callable, Optional

from onnx_diarization import MODEL_DIR

BASE_URL = "https://github.com/benoitboidin/obsidian-meeting-transcriber/releases/download/models-v1"
FILES = {
    "segmentation-3.0.onnx": "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    "wespeaker-resnet34-masked.onnx": "ca1cdd0fb3af48bdb38a0fccdbb2df7ff71ea2a26eed42c4d8cbf6595f272655",
    "plda.npz": "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255",
    "xvec_transform.npz": "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f",
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_diarization_models(progress: Optional[Callable[[float], None]] = None) -> str:
    """Model folder, downloading and checking any missing file."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    missing = [n for n in FILES if not os.path.exists(os.path.join(MODEL_DIR, n))]
    for i, name in enumerate(missing):
        if progress:
            progress(i / len(missing))
        dest = os.path.join(MODEL_DIR, name)
        tmp = dest + ".part"
        try:
            req = urllib.request.Request(f"{BASE_URL}/{name}", headers={"User-Agent": "meeting-transcriber"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise RuntimeError(f"could not download the speaker models ({name}): {e}") from e
        if _sha256(tmp) != FILES[name]:
            os.remove(tmp)
            raise RuntimeError(f"downloaded {name} is corrupted (checksum mismatch); try again.")
        os.replace(tmp, dest)
    return MODEL_DIR
