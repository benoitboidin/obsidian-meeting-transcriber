"""Speaker diarization: pyannote's speaker-diarization-community-1, run with ONNX (no PyTorch).

Long recordings are diarized in chunks (default 30 min); speakers found in different chunks
are linked with their voice embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

import onnx_diarization
from audio import Wav
from models import ensure_diarization_models

Progress = Callable[[str, float], None]
Turn = tuple[float, float, str]
# speaker label -> (voice embedding, seconds of speech)
Voices = dict[str, tuple[np.ndarray, float]]

# Cosine distance under which two speakers from different chunks are considered the same person.
LINK_THRESHOLD = 0.5


@dataclass
class LocalSpeaker:
    chunk: int
    label: str
    emb: Optional[np.ndarray]
    dur: float


def diarize(wav: Wav, chunks: list[tuple[float, float]], min_speakers: Optional[int],
            max_speakers: Optional[int], progress: Progress,
            link_threshold: float = LINK_THRESHOLD) -> tuple[list[Turn], Voices]:
    """Diarize chunk by chunk, then link speakers across chunks.
    Returns global (start, end, speaker) turns and each speaker's voice embedding."""
    turns: list[tuple[float, float, int, str]] = []  # start, end, chunk, local label
    locals_: list[LocalSpeaker] = []
    for i, (cs, ce) in enumerate(chunks):
        def step(stage: str, frac: float, i: int = i) -> None:
            # segmentation is ~1/4 of the work, embeddings ~3/4
            within = 0.25 * frac if stage == "segmentation" else 0.25 + 0.75 * frac
            progress("identifying speakers", (i + within) / len(chunks))

        full, excl, centroids = onnx_diarization.diarize(
            wav.read(cs, ce), min_speakers=min_speakers if len(chunks) == 1 else None,
            max_speakers=max_speakers, progress=step)
        durations: dict[str, float] = {}
        for s, e, label in full:
            durations[label] = durations.get(label, 0.0) + (e - s)
        for label, dur in durations.items():
            e = centroids.get(label)
            ok = e is not None and np.all(np.isfinite(e)) and np.any(e)
            locals_.append(LocalSpeaker(i, label, np.asarray(e, dtype=np.float64) if ok else None, dur))
        for s, e, label in excl:
            turns.append((cs + s, cs + e, i, label))
    progress("identifying speakers", 1.0)

    mapping, voices = link_speakers(locals_, link_threshold, max_speakers)
    return [(s, e, mapping[(c, l)]) for s, e, c, l in turns], voices


def link_speakers(speakers: list[LocalSpeaker], threshold: float,
                  max_speakers: Optional[int]) -> tuple[dict[tuple[int, str], str], Voices]:
    """Agglomerative clustering of per-chunk speakers (duration-weighted centroids).

    Two speakers from the same chunk are never merged (pyannote already separated them).
    """
    clusters: list[dict] = []
    for sp in speakers:
        clusters.append({
            "members": [sp],
            "chunks": {sp.chunk},
            "sum": sp.emb * max(sp.dur, 1e-3) if sp.emb is not None else None,
        })

    def dist(a: dict, b: dict) -> float:
        if a["sum"] is None or b["sum"] is None or a["chunks"] & b["chunks"]:
            return float("inf")
        x, y = a["sum"], b["sum"]
        return 1.0 - float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))

    while len(clusters) > 1:
        best, bi, bj = float("inf"), -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = dist(clusters[i], clusters[j])
                if d < best:
                    best, bi, bj = d, i, j
        too_many = bool(max_speakers) and len(clusters) > max_speakers
        if bi < 0 or (best > threshold and not too_many):
            break
        a, b = clusters[bi], clusters.pop(bj)
        a["members"] += b["members"]
        a["chunks"] |= b["chunks"]
        a["sum"] = a["sum"] + b["sum"]

    # Number speakers by total speaking time, most talkative first
    clusters.sort(key=lambda c: -sum(m.dur for m in c["members"]))
    mapping: dict[tuple[int, str], str] = {}
    voices: Voices = {}
    for gi, c in enumerate(clusters):
        label = f"G{gi:03d}"
        for m in c["members"]:
            mapping[(m.chunk, m.label)] = label
        if c["sum"] is not None:
            voices[label] = (c["sum"] / (np.linalg.norm(c["sum"]) + 1e-12),
                             sum(m.dur for m in c["members"]))
    return mapping, voices


def run_diarization(wav: Wav, chunks: list[tuple[float, float]], min_speakers: Optional[int],
                    max_speakers: Optional[int], progress: Progress) -> tuple[list[Turn], Voices]:
    """Global (start, end, speaker) turns and voice embeddings; raises if unavailable."""
    ensure_diarization_models(lambda f: progress("downloading speaker models (33 MB, once)", f))
    return diarize(wav, chunks, min_speakers, max_speakers, progress)
