"""Speaker diarization with pyannote (speaker-diarization-community-1).

Long recordings are diarized in chunks (default 30 min); speakers found in different
chunks are linked with their pyannote embeddings.
"""
from __future__ import annotations

import gc
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from audio import SR, Wav

Progress = Callable[[str, float], None]
Turn = tuple[float, float, str]

# Cosine distance under which two speakers from different chunks are considered the same person.
LINK_THRESHOLD = 0.5


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pick_torch_device() -> torch.device:
    """Best torch device for diarization: CUDA, then Apple Metal (MPS), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_diarizer(hf_token: Optional[str], device: Optional[torch.device] = None):
    from pyannote.audio import Pipeline

    if not hf_token:
        raise RuntimeError("no HuggingFace token set in the plugin settings.")
    target = device or torch.device("cpu")
    first_err: Exception | None = None
    for name in ("pyannote/speaker-diarization-community-1",
                 "pyannote/speaker-diarization-3.1"):
        try:
            pipe = Pipeline.from_pretrained(name, token=hf_token)
            if pipe is None:
                raise RuntimeError(f"could not load {name}")
            try:
                pipe.to(target)
            except Exception:  # some backends lack ops used by the pipeline
                pipe.to(torch.device("cpu"))
            return pipe
        except Exception as e:  # try the fallback model
            first_err = first_err or e
    msg = str(first_err)
    if "403" in msg or "restricted" in msg or "Gated" in type(first_err).__name__:
        raise RuntimeError(
            "your HuggingFace account has not been granted access to the model. Accept the "
            "conditions at https://hf.co/pyannote/speaker-diarization-community-1 (same account as the token).")
    if "401" in msg:
        raise RuntimeError("the HuggingFace token is invalid or expired.")
    raise RuntimeError(f"could not load the pyannote model: {msg.splitlines()[0] if msg else first_err!r}")


@dataclass
class LocalSpeaker:
    chunk: int
    label: str
    emb: Optional[np.ndarray]
    dur: float


def diarize(pipe, wav: Wav, chunks: list[tuple[float, float]], min_speakers: Optional[int],
            max_speakers: Optional[int], progress: Progress,
            link_threshold: float = LINK_THRESHOLD) -> list[tuple[float, float, str]]:
    """Diarize chunk by chunk, then link speakers across chunks. Returns global (start, end, speaker) turns."""
    turns: list[tuple[float, float, int, str]] = []  # start, end, chunk, local label
    locals_: list[LocalSpeaker] = []
    for i, (cs, ce) in enumerate(chunks):
        progress("identifying speakers", i / len(chunks))
        audio = torch.from_numpy(wav.read(cs, ce)).unsqueeze(0)
        kwargs = {}
        if max_speakers:
            kwargs["max_speakers"] = max_speakers
        if min_speakers and len(chunks) == 1:  # a single chunk may contain fewer people
            kwargs["min_speakers"] = min_speakers
        out = pipe({"waveform": audio, "sample_rate": SR}, **kwargs)
        full = getattr(out, "speaker_diarization", out)
        excl = getattr(out, "exclusive_speaker_diarization", None) or full
        embs = getattr(out, "speaker_embeddings", None)
        for k, label in enumerate(full.labels()):
            e = None
            if embs is not None and k < len(embs) and np.all(np.isfinite(embs[k])):
                e = np.asarray(embs[k], dtype=np.float64)
            locals_.append(LocalSpeaker(i, label, e, full.label_duration(label)))
        for turn, _, spk in excl.itertracks(yield_label=True):
            turns.append((cs + turn.start, cs + turn.end, i, spk))
        del audio, out
        free_gpu()
    progress("identifying speakers", 1.0)

    mapping = link_speakers(locals_, link_threshold, max_speakers)
    return [(s, e, mapping[(c, l)]) for s, e, c, l in turns]


def link_speakers(speakers: list[LocalSpeaker], threshold: float,
                  max_speakers: Optional[int]) -> dict[tuple[int, str], str]:
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
    for gi, c in enumerate(clusters):
        for m in c["members"]:
            mapping[(m.chunk, m.label)] = f"G{gi:03d}"
    return mapping


def run_diarization(hf_token: Optional[str], wav: Wav, chunks: list[tuple[float, float]],
                    min_speakers: Optional[int], max_speakers: Optional[int],
                    progress: Progress) -> list[Turn]:
    """Global (start, end, speaker) turns; raises if diarization is unavailable."""
    device = pick_torch_device()
    pipe = load_diarizer(hf_token, device)
    try:
        return diarize(pipe, wav, chunks, min_speakers, max_speakers, progress)
    except Exception:
        if device.type != "mps":
            raise
        # MPS can lack ops used by pyannote: retry this pipeline on CPU.
        traceback.print_exc()
        pipe = None
        free_gpu()
        pipe = load_diarizer(hf_token, torch.device("cpu"))
        return diarize(pipe, wav, chunks, min_speakers, max_speakers, progress)
    finally:
        pipe = None
        free_gpu()
