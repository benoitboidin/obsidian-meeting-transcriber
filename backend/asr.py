"""Speech-to-text: faster-whisper (NVIDIA GPU or CPU) or mlx-whisper (Apple GPU).

One model is kept loaded and shared by the live transcript and the final pass;
a lock makes them take turns on the GPU.
"""
from __future__ import annotations

import ctypes
import gc
import os
import shutil
import subprocess
import sys
import threading
import traceback
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace as NS
from typing import Callable, Optional

import numpy as np

from audio import SR, Wav

Progress = Callable[[str, float], None]


@dataclass
class Word:
    start: float
    end: float
    text: str
    speaker: str = ""


# ----------------------------------------------------------------------- Device

def _preload_cublas() -> None:
    """faster-whisper (CTranslate2) needs cuBLAS 12: load it from the nvidia-cublas-cu12
    wheel if installed, so no system CUDA toolkit is needed."""
    try:
        import nvidia.cublas  # type: ignore
    except ImportError:
        return
    for d in nvidia.cublas.__path__:
        for name in ("libcublasLt.so.12", "libcublas.so.12"):
            p = os.path.join(d, "lib", name)
            if os.path.exists(p):
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def cuda_available() -> bool:
    if os.environ.get("MT_ASR") in ("cpu", "mlx"):
        return False
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def gpu_name() -> Optional[str]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return "NVIDIA GPU"
    try:
        out = subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out.splitlines()[0] if out else "NVIDIA GPU"
    except Exception:
        return "NVIDIA GPU"


def _mlx_available() -> bool:
    if sys.platform != "darwin" or os.environ.get("MT_ASR") == "cpu":
        return False
    try:
        import mlx_whisper  # noqa: F401  (only installed on Apple Silicon)
        return True
    except Exception:
        return False


def pick_device() -> str:
    """NVIDIA → faster-whisper on CUDA; Apple Silicon → mlx-whisper on the Apple GPU; else CPU."""
    if cuda_available():
        return "cuda"
    if os.environ.get("MT_ASR") == "mlx" or _mlx_available():
        return "mlx"
    return "cpu"


# ----------------------------------------------------------------------- Models

# Our model names → MLX-converted weights on HuggingFace.
MLX_REPOS = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
}


class MlxWhisper:
    """mlx-whisper behind faster-whisper's transcribe() interface (the subset we use).

    mlx-whisper has no VAD, so we run faster-whisper's Silero VAD first, transcribe only
    the speech, and map timestamps back — the same thing faster-whisper does internally.
    Skipping silences avoids Whisper's hallucinations on them and saves time.
    """

    def __init__(self, model_name: str):
        import mlx_whisper

        self._mlx = mlx_whisper
        self.repo = MLX_REPOS.get(model_name) or (
            model_name if "/" in model_name else f"mlx-community/whisper-{model_name}-mlx")

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None, **_):
        from faster_whisper.vad import SpeechTimestampsMap, VadOptions, get_speech_timestamps

        speech = get_speech_timestamps(audio, VadOptions())
        if not speech:
            return [], NS(language=language)
        joined = np.concatenate([audio[c["start"]:c["end"]] for c in speech])
        ts = SpeechTimestampsMap(speech, SR)
        r = self._mlx.transcribe(
            joined, path_or_hf_repo=self.repo, language=language or None,
            word_timestamps=True, condition_on_previous_text=False,
            hallucination_silence_threshold=2.0, verbose=None)
        segments = []
        for seg in r.get("segments", []):
            words = []
            for w in seg.get("words") or []:
                idx = ts.get_chunk_index((w["start"] + w["end"]) / 2)
                words.append(NS(start=ts.get_original_time(w["start"], idx),
                                end=ts.get_original_time(w["end"], idx), word=w["word"]))
            if words:
                segments.append(NS(end=words[-1].end, words=words))
        return segments, NS(language=r.get("language") or language)


def _load(model_name: str, device: str):
    if device == "mlx":
        return MlxWhisper(model_name)
    if device == "cuda":
        _preload_cublas()
    from faster_whisper import WhisperModel

    if device == "cuda":
        return WhisperModel(model_name, device="cuda", compute_type="float16")
    try:
        return WhisperModel(model_name, device="cpu", compute_type="int8")
    except Exception:
        # int8 is not available on every CPU (some ARM builds): fall back to fp32.
        return WhisperModel(model_name, device="cpu", compute_type="float32")


LOCK = threading.RLock()  # one transcription at a time, and guards the cache
_cache: dict[tuple[str, str], object] = {}


def _model(model_name: str, device: str):
    key = (model_name, device)
    if key not in _cache:
        release()  # only one model in memory
        _cache[key] = _load(model_name, device)
    return _cache[key]


def release() -> None:
    """Unload the model (frees GPU memory, e.g. for Ollama)."""
    with LOCK:
        _cache.clear()
        gc.collect()


# ------------------------------------------------------------------- Transcribe

class Transcriber:
    """Transcribes audio arrays with the best available engine, falling back from
    the Apple GPU to the CPU if MLX fails."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = pick_device()
        self.warnings: list[str] = []

    def _run(self, audio: np.ndarray, offset: float, language: Optional[str],
             on_time: Optional[Callable[[float], None]]):
        model = _model(self.model_name, self.device)
        segments, info = model.transcribe(
            audio, language=language or None, vad_filter=True, word_timestamps=True,
            beam_size=5, condition_on_previous_text=False)
        words: list[Word] = []
        for seg in segments:  # faster-whisper decodes lazily while we iterate
            for w in seg.words or []:
                words.append(Word(offset + w.start, offset + w.end, w.word))
            if on_time:
                on_time(offset + seg.end)
        return words, info.language

    def words(self, audio: np.ndarray, offset: float = 0.0, language: Optional[str] = None,
              on_time: Optional[Callable[[float], None]] = None) -> tuple[list[Word], Optional[str]]:
        """Words with absolute timestamps (offset added) and the detected language.
        `on_time` is called with the time reached, as decoding progresses."""
        if len(audio) < SR // 10:
            return [], language
        with LOCK:
            try:
                return self._run(audio, offset, language, on_time)
            except Exception as e:
                if self.device != "mlx":
                    raise
                traceback.print_exc()
                self.warnings.append(f"Apple GPU transcription failed, used the CPU instead: {e}")
                self.device = "cpu"
                return self._run(audio, offset, language, on_time)


def transcribe_file(tr: Transcriber, wav: Wav, chunks: list[tuple[float, float]],
                    language: Optional[str], label: str,
                    progress: Progress) -> tuple[list[Word], Optional[str]]:
    words: list[Word] = []
    langs: Counter[str] = Counter()
    total = wav.duration or 1.0
    for cs, ce in chunks:
        progress(f"transcribing {label}", min(cs / total, 1.0))
        w, lang = tr.words(wav.read(cs, ce), cs, language,
                           lambda t: progress(f"transcribing {label}", min(t / total, 1.0)))
        words += w
        if w and lang:
            langs[lang] += len(w)
    progress(f"transcribing {label}", 1.0)
    return words, language or (langs.most_common(1)[0][0] if langs else None)


def speech_regions(audio: np.ndarray) -> list[tuple[float, float]]:
    """Speech (start, end) in seconds, from faster-whisper's Silero VAD (ONNX, CPU)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    return [(c["start"] / SR, c["end"] / SR) for c in get_speech_timestamps(audio, VadOptions())]
