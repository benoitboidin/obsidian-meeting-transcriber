"""Audio access: conversion to 16 kHz mono WAV, random access reads, chunk planning,
and reading a WAV file that ffmpeg is still writing (live transcript)."""
from __future__ import annotations

import os
import subprocess
import wave
from typing import Optional

import numpy as np

SR = 16000
# The plugin passes the ffmpeg path it found (GUI apps may not have it on PATH).
FFMPEG = os.environ.get("MT_FFMPEG") or "ffmpeg"


def _is_wav16k_mono(path: str) -> bool:
    try:
        with wave.open(path, "rb") as f:
            return f.getframerate() == SR and f.getnchannels() == 1 and f.getsampwidth() == 2
    except Exception:
        return False


def to_wav16k(path: str, tmpdir: str) -> str:
    """16 kHz mono s16 WAV (converted with ffmpeg unless it already is one)."""
    if _is_wav16k_mono(path):
        return path
    out = os.path.join(tmpdir, os.path.basename(path) + ".16k.wav")
    try:
        subprocess.run(
            [FFMPEG, "-nostdin", "-y", "-loglevel", "error", "-i", path,
             "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", out],
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"ffmpeg was not found (tried {FFMPEG!r}). Install it (macOS: brew install ffmpeg) "
            "and set its path in the plugin settings.")
    return out


class Wav:
    """Random access to a 16 kHz mono s16 WAV without loading it whole."""

    def __init__(self, path: str):
        self.f = wave.open(path, "rb")
        self.frames = self.f.getnframes()
        self.duration = self.frames / SR

    def read(self, start: float, end: float) -> np.ndarray:
        a = max(0, int(start * SR))
        b = min(self.frames, int(end * SR))
        self.f.setpos(a)
        data = self.f.readframes(max(0, b - a))
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    def close(self) -> None:
        self.f.close()


class GrowingWav:
    """Reads a 16 kHz mono s16 WAV while ffmpeg is still writing it.

    During recording ffmpeg leaves the data size as a placeholder, so the readable
    length comes from the file size; once ffmpeg finishes, the header is exact.
    """

    def __init__(self, path: str):
        self.path = path
        self.data_offset: Optional[int] = None

    def _find_data(self) -> Optional[tuple[int, int]]:
        """(offset of the samples, declared data size) once the header is on disk."""
        try:
            with open(self.path, "rb") as f:
                head = f.read(4096)
        except FileNotFoundError:
            return None
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return None
        pos = 12
        while pos + 8 <= len(head):
            cid, size = head[pos:pos + 4], int.from_bytes(head[pos + 4:pos + 8], "little")
            if cid == b"data":
                return pos + 8, size
            pos += 8 + size + (size & 1)
        return None

    def samples(self) -> int:
        """Number of complete samples currently readable."""
        found = self._find_data()
        if not found:
            return 0
        self.data_offset, declared = found
        size = os.path.getsize(self.path) - self.data_offset
        if 0 < declared < 0xFFFFFFF0:  # finalised header
            size = min(size, declared)
        return max(0, size) // 2

    def read(self, a: int, b: int) -> np.ndarray:
        """Samples [a, b) as float32."""
        if self.data_offset is None or b <= a:
            return np.zeros(0, dtype=np.float32)
        with open(self.path, "rb") as f:
            f.seek(self.data_offset + 2 * a)
            data = f.read(2 * (b - a))
        data = data[: len(data) // 2 * 2]
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def quietest_point(audio: np.ndarray, offset: float = 0.0) -> float:
    """Time (s, + offset) of the centre of the quietest 200 ms window in `audio`."""
    hop = SR // 5
    n = len(audio) // hop
    if not n:
        return offset + len(audio) / SR
    energy = (audio[: n * hop].reshape(n, hop) ** 2).mean(axis=1)
    return offset + (int(energy.argmin()) + 0.5) * 0.2


def plan_chunks(wav: Wav, chunk_sec: float, search: float = 60.0) -> list[tuple[float, float]]:
    """Split into ~chunk_sec pieces, cutting at the quietest 200 ms in the minute before each boundary."""
    dur = wav.duration
    cuts = [0.0]
    t = chunk_sec
    while t < dur - chunk_sec * 0.25:
        ws = max(cuts[-1] + 1.0, t - search)
        cut = quietest_point(wav.read(ws, t), ws)
        cuts.append(cut)
        t = cut + chunk_sec
    cuts.append(dur)
    return list(zip(cuts[:-1], cuts[1:]))
