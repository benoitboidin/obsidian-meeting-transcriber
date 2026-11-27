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

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None,
                   initial_prompt: Optional[str] = None, **_):
        from faster_whisper.vad import SpeechTimestampsMap, VadOptions, get_speech_timestamps

        speech = get_speech_timestamps(audio, VadOptions())
        if not speech:
            return [], NS(language=language)
        joined = np.concatenate([audio[c["start"]:c["end"]] for c in speech])
        ts = SpeechTimestampsMap(speech, SR)
        r = self._mlx.transcribe(
            joined, path_or_hf_repo=self.repo, language=language or None,
            initial_prompt=initial_prompt, word_timestamps=True, condition_on_previous_text=False,
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

# Whisper sometimes drops punctuation and capitals for whole windows of a meeting. A short
# punctuated prompt (the end of the previous piece, or a phrase in the meeting's language)
# keeps the style; each piece is decoded on its own, so a bad piece cannot derail the next.
STYLE_PROMPTS = {
    "en": "Hello, everyone. Okay, let's get started.",
    "fr": "Bonjour à tous. Bon, on commence ?",
    "de": "Hallo zusammen. Also, fangen wir an.",
    "es": "Hola a todos. Bueno, empezamos.",
    "it": "Ciao a tutti. Allora, iniziamo.",
    "nl": "Hallo allemaal. Oké, laten we beginnen.",
    "pt": "Olá a todos. Bom, vamos começar.",
}
PIECE_SEC = 28.0      # one Whisper window per piece
LANG_LOCK_WORDS = 30  # once this many words are in, keep the detected language


def prompt_for(language: Optional[str], previous: str) -> Optional[str]:
    tail = previous.strip()[-200:]
    if tail and any(c in tail for c in ".?!。"):
        return tail[tail.find(" ") + 1:] if len(previous.strip()) > 200 else tail
    return STYLE_PROMPTS.get(language or "")


class Context:
    """Language detection shared by the pieces of one recording."""

    def __init__(self, language: Optional[str]):
        self.user = language or None
        self.langs: Counter[str] = Counter()

    def guess(self) -> Optional[str]:
        return self.user or (self.langs.most_common(1)[0][0] if self.langs else None)

    def locked(self) -> Optional[str]:
        """Language to force: the user's, or the detected one once confident."""
        if self.user:
            return self.user
        return self.guess() if sum(self.langs.values()) >= LANG_LOCK_WORDS else None


def plan_pieces(audio: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous pieces (samples) of at most ~PIECE_SEC, cut in pauses between speech."""
    from audio import quietest_point

    groups: list[list[float]] = []
    for s, e in speech_regions(audio):
        while e - s > PIECE_SEC:  # nobody pauses: cut at the quietest point near the limit
            ws = s + PIECE_SEC - 5.0
            cut = quietest_point(audio[int(ws * SR):int((s + PIECE_SEC) * SR)], ws)
            groups.append([s, cut])
            s = cut
        if groups and e - groups[-1][0] <= PIECE_SEC:
            groups[-1][1] = e
        else:
            groups.append([s, e])
    if not groups:
        return []
    n = len(audio)
    bounds = [0] + [int((a[1] + b[0]) / 2 * SR) for a, b in zip(groups, groups[1:])] + [n]
    # don't send long silences: trim each piece to its speech, with a little margin
    out = []
    for (s, e), a, b in zip(groups, bounds, bounds[1:]):
        out.append((max(a, int((s - 0.5) * SR)), min(b, int((e + 0.5) * SR))))
    return out


class Transcriber:
    """Transcribes audio arrays with the best available engine, falling back from
    the Apple GPU to the CPU if MLX fails."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = pick_device()
        self.warnings: list[str] = []

    def _run(self, audio: np.ndarray, offset: float, language: Optional[str],
             prompt: Optional[str], on_time: Optional[Callable[[float], None]]):
        model = _model(self.model_name, self.device)
        segments, info = model.transcribe(
            audio, language=language or None, vad_filter=True, word_timestamps=True,
            beam_size=5, condition_on_previous_text=False, initial_prompt=prompt)
        words: list[Word] = []
        for seg in segments:  # faster-whisper decodes lazily while we iterate
            for w in seg.words or []:
                words.append(Word(offset + w.start, offset + w.end, w.word))
            if on_time:
                on_time(offset + seg.end)
        return words, info.language

    def words(self, audio: np.ndarray, offset: float = 0.0, language: Optional[str] = None,
              on_time: Optional[Callable[[float], None]] = None,
              prompt: Optional[str] = None) -> tuple[list[Word], Optional[str]]:
        """Words of one piece (≤ 30 s) with absolute timestamps, and the detected language."""
        if len(audio) < SR // 10:
            return [], language
        with LOCK:
            try:
                return self._run(audio, offset, language, prompt, on_time)
            except Exception as e:
                if self.device != "mlx":
                    raise
                traceback.print_exc()
                self.warnings.append(f"Apple GPU transcription failed, used the CPU instead: {e}")
                self.device = "cpu"
                return self._run(audio, offset, language, prompt, on_time)

    def piece(self, audio: np.ndarray, offset: float, ctx: Context, previous: str,
              on_time: Optional[Callable[[float], None]] = None) -> list[Word]:
        """One piece, prompted with the previous text; updates the language detection."""
        words, lang = self.words(audio, offset, ctx.locked(), on_time, prompt_for(ctx.guess(), previous))
        if words and lang:
            ctx.langs[lang] += len(words)
        return words

    def long(self, audio: np.ndarray, offset: float, ctx: Context,
             on_time: Optional[Callable[[float], None]] = None) -> list[Word]:
        """Any length: split into pieces at pauses, each transcribed on its own."""
        words: list[Word] = []
        previous = ""
        for a, b in plan_pieces(audio):
            w = self.piece(audio[a:b], offset + a / SR, ctx, previous, on_time)
            words += w
            previous = "".join(x.text for x in w) or previous
            if on_time:
                on_time(offset + b / SR)
        return words


def transcribe_file(tr: Transcriber, wav: Wav, chunks: list[tuple[float, float]],
                    language: Optional[str], label: str,
                    progress: Progress) -> tuple[list[Word], Optional[str]]:
    ctx = Context(language)
    words: list[Word] = []
    total = wav.duration or 1.0
    for cs, ce in chunks:
        progress(f"transcribing {label}", min(cs / total, 1.0))
        words += tr.long(wav.read(cs, ce), cs, ctx,
                         lambda t: progress(f"transcribing {label}", min(t / total, 1.0)))
    progress(f"transcribing {label}", 1.0)
    return words, ctx.guess()


def speech_regions(audio: np.ndarray) -> list[tuple[float, float]]:
    """Speech (start, end) in seconds, from faster-whisper's Silero VAD (ONNX, CPU)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    return [(c["start"] / SR, c["end"] / SR) for c in get_speech_timestamps(audio, VadOptions())]
