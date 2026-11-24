"""Transcription + diarization pipeline.

Usage as CLI (for testing):
    uv run python pipeline.py --mic mic.wav [--system system.wav] [--me "Alice"] [--language fr]

Online meeting  : mic track = the user (named), system track = remote people (diarized).
In-person/file  : only one track, fully diarized.

Long recordings are processed in chunks (default 30 min, cut at the quietest
point near each boundary) so memory use stays bounded whatever the duration.
Speakers found in different chunks are linked with their pyannote embeddings.
"""
from __future__ import annotations

import argparse
import difflib
import gc
import json
import os
import subprocess
import sys
import tempfile
import wave
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np

# torch must be imported before faster_whisper so CUDA/cuDNN libs shipped with
# the torch wheels are loaded into the process for ctranslate2 as well.
import torch  # noqa: E402

Progress = Callable[[str, float], None]

SR = 16000
CHUNK_SEC = 30 * 60
# The plugin passes the ffmpeg path it found (GUI apps may not have it on PATH).
FFMPEG = os.environ.get("MT_FFMPEG") or "ffmpeg"
# Cosine distance under which two speakers from different chunks are considered the same person.
LINK_THRESHOLD = 0.5


@dataclass
class Word:
    start: float
    end: float
    text: str
    speaker: str = ""


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str


def _noop(_stage: str, _frac: float) -> None:
    pass


# ------------------------------------------------------------------------ Audio

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


def plan_chunks(wav: Wav, chunk_sec: float, search: float = 60.0) -> list[tuple[float, float]]:
    """Split into ~chunk_sec pieces, cutting at the quietest 200 ms in the minute before each boundary."""
    dur = wav.duration
    cuts = [0.0]
    t = chunk_sec
    while t < dur - chunk_sec * 0.25:
        ws = max(cuts[-1] + 1.0, t - search)
        win = wav.read(ws, t)
        hop = SR // 5
        n = len(win) // hop
        if n:
            energy = (win[: n * hop].reshape(n, hop) ** 2).mean(axis=1)
            cut = ws + (int(energy.argmin()) + 0.5) * 0.2
        else:
            cut = t
        cuts.append(cut)
        t = cut + chunk_sec
    cuts.append(dur)
    return list(zip(cuts[:-1], cuts[1:]))


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


# --------------------------------------------------------------------------- ASR

def pick_asr_device() -> str:
    """CTranslate2 speaks CUDA and (experimentally) Apple Metal; otherwise CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_whisper(model_name: str, device: str):
    from faster_whisper import WhisperModel

    if device == "cuda":
        return WhisperModel(model_name, device="cuda", compute_type="float16")
    if device == "mps":
        try:
            return WhisperModel(model_name, device="mps", compute_type="float16")
        except Exception:
            device = "cpu"  # experimental backend unavailable → fall through
    try:
        return WhisperModel(model_name, device="cpu", compute_type="int8")
    except Exception:
        # int8 is not available on every CPU (some ARM builds): fall back to fp32.
        return WhisperModel(model_name, device="cpu", compute_type="float32")


def transcribe(model, wav: Wav, chunks: list[tuple[float, float]], language: Optional[str],
               label: str, progress: Progress) -> tuple[list[Word], Optional[str]]:
    words: list[Word] = []
    langs: Counter[str] = Counter()
    total = wav.duration or 1.0
    for cs, ce in chunks:
        segments, info = model.transcribe(
            wav.read(cs, ce),
            language=language or None,
            vad_filter=True,
            word_timestamps=True,
            beam_size=5,
            condition_on_previous_text=False,
        )
        n = 0
        for seg in segments:
            for w in seg.words or []:
                words.append(Word(cs + w.start, cs + w.end, w.word))
                n += 1
            progress(f"transcribing {label}", min((cs + seg.end) / total, 1.0))
        if n:
            langs[info.language] += n
        progress(f"transcribing {label}", min(ce / total, 1.0))
    lang = language or (langs.most_common(1)[0][0] if langs else None)
    return words, lang


# --------------------------------------------------------------------- Diarize

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


def assign_speakers(words: list[Word], turns: list[tuple[float, float, str]]) -> None:
    """Give each word the speaker whose turn overlaps it most (or the nearest one)."""
    if not turns:
        for w in words:
            w.speaker = "SPEAKER_00"
        return
    turns = sorted(turns)
    starts = np.array([t[0] for t in turns])
    max_len = max(e - s for s, e, _ in turns)
    for w in words:
        # only turns that can overlap this word
        lo = int(np.searchsorted(starts, w.start - max_len, side="left"))
        hi = int(np.searchsorted(starts, w.end, side="right"))
        best, best_ov = None, 0.0
        for s, e, spk in turns[lo:hi]:
            ov = min(e, w.end) - max(s, w.start)
            if ov > best_ov:
                best, best_ov = spk, ov
        if best is None:
            mid = (w.start + w.end) / 2
            k = int(np.searchsorted(starts, mid))
            cands = turns[max(0, k - 2): k + 2]
            best = min(cands, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)))[2]
        w.speaker = best


def rename_by_appearance(words: list[Word]) -> None:
    """Renumber speakers SPEAKER_xx in order of first appearance."""
    mapping: dict[str, str] = {}
    for w in words:
        if w.speaker not in mapping:
            mapping[w.speaker] = f"SPEAKER_{len(mapping):02d}"
        w.speaker = mapping[w.speaker]


# ----------------------------------------------------------------------- Merge

def group(words: list[Word], max_gap: float = 1.5) -> list[Segment]:
    segs: list[Segment] = []
    for w in words:
        if segs and segs[-1].speaker == w.speaker and w.start - segs[-1].end <= max_gap:
            segs[-1].end = w.end
            segs[-1].text += w.text
        else:
            segs.append(Segment(w.start, w.end, w.speaker, w.text))
    for s in segs:
        s.text = s.text.strip()
    return [s for s in segs if s.text]


def _norm(t: str) -> str:
    return "".join(c for c in t.lower() if c.isalnum() or c.isspace()).strip()


def drop_bleed(mic: list[Segment], system: list[Segment], threshold: float = 0.6) -> list[Segment]:
    """Remove mic segments that are just the remote audio leaking from the speakers."""
    kept = []
    sys_starts = np.array([s.start for s in system]) if system else np.array([])
    max_len = max((s.end - s.start for s in system), default=0.0)
    for m in mic:
        lo = int(np.searchsorted(sys_starts, m.start - 1.0 - max_len))
        hi = int(np.searchsorted(sys_starts, m.end + 1.0, side="right"))
        overlapping = " ".join(
            s.text for s in system[lo:hi] if s.start < m.end + 1.0 and s.end > m.start - 1.0
        )
        if overlapping:
            a, b = _norm(m.text), _norm(overlapping)
            sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
            matched = sum(bl.size for bl in sm.get_matching_blocks())
            if a and matched / len(a) >= threshold:
                continue
        kept.append(m)
    return kept


# ------------------------------------------------------------------------ Main

def run(mic: str, system: Optional[str] = None, me: str = "Me",
        language: Optional[str] = None, hf_token: Optional[str] = None,
        whisper_model: str = "large-v3", min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None, progress: Progress = _noop,
        chunk_sec: float = CHUNK_SEC) -> dict:
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    warnings: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        progress("converting audio", 0.0)
        mic_wav = Wav(to_wav16k(mic, tmp))
        sys_wav = Wav(to_wav16k(system, tmp)) if system else None
        mic_chunks = plan_chunks(mic_wav, chunk_sec)
        sys_chunks = plan_chunks(sys_wav, chunk_sec) if sys_wav else []
        duration = max(mic_wav.duration, sys_wav.duration if sys_wav else 0.0)

        def run_asr(device: str) -> tuple[list[Word], list[Word], Optional[str]]:
            progress("loading whisper", 0.0)
            model = load_whisper(whisper_model, device)
            try:
                mic_words, lang = transcribe(model, mic_wav, mic_chunks, language,
                                             "your mic" if sys_wav else "audio", progress)
                remote: list[Word] = []
                if sys_wav:
                    remote, sys_lang = transcribe(model, sys_wav, sys_chunks, language or lang,
                                                  "remote audio", progress)
                    if sys_lang and len(remote) >= len(mic_words):
                        lang = sys_lang
                return mic_words, remote, lang
            finally:
                del model
                free_gpu()

        asr_device = pick_asr_device()
        try:
            mic_words, sys_words, lang = run_asr(asr_device)
        except Exception as e:
            if asr_device != "mps":
                raise
            warnings.append(f"Apple Metal transcription failed, retried on CPU: {e}")
            mic_words, sys_words, lang = run_asr("cpu")

        # Track that needs diarization: system (online) or mic (in-person / file)
        words, wav, chunks = (sys_words, sys_wav, sys_chunks) if sys_wav else (mic_words, mic_wav, mic_chunks)
        if words:
            progress("identifying speakers", 0.0)
            turns: list[tuple[float, float, str]] = []
            diar_device = pick_torch_device()
            try:
                pipe = load_diarizer(hf_token, diar_device)
                try:
                    turns = diarize(pipe, wav, chunks, min_speakers, max_speakers, progress)
                except Exception:
                    if diar_device.type != "mps":
                        raise
                    # MPS can lack ops used by pyannote: retry this pipeline on CPU.
                    del pipe
                    free_gpu()
                    pipe = load_diarizer(hf_token, torch.device("cpu"))
                    turns = diarize(pipe, wav, chunks, min_speakers, max_speakers, progress)
                del pipe
            except Exception as e:  # keep the transcript even if diarization fails
                warnings.append(f"Diarization skipped: {e}")
                turns = []
            free_gpu()
            assign_speakers(words, turns)
            rename_by_appearance(words)

        mic_wav.close()
        if sys_wav:
            sys_wav.close()

    if sys_wav:
        for w in mic_words:
            w.speaker = me
        sys_segs = group(sys_words)
        mic_segs = drop_bleed(group(mic_words), sys_segs)
        segments = sorted(mic_segs + sys_segs, key=lambda s: s.start)
    else:
        segments = group(mic_words)

    progress("done", 1.0)
    return {
        "language": lang or "unknown",
        "duration": duration,
        "segments": [asdict(s) for s in segments],
        "warnings": warnings,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", required=True, help="mic track, or the single audio/video file")
    ap.add_argument("--system", help="system-audio track (online meetings)")
    ap.add_argument("--me", default="Me")
    ap.add_argument("--language")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--min-speakers", type=int)
    ap.add_argument("--max-speakers", type=int)
    ap.add_argument("--chunk-minutes", type=float, default=CHUNK_SEC / 60)
    a = ap.parse_args()
    last = [""]

    def progress(s: str, f: float) -> None:
        msg = f"[{s}] {f:.0%}"
        if msg != last[0]:
            print(msg, file=sys.stderr, flush=True)
            last[0] = msg

    res = run(a.mic, a.system, a.me, a.language, None, a.model, a.min_speakers,
              a.max_speakers, progress=progress, chunk_sec=a.chunk_minutes * 60)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
