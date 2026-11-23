"""Transcription + diarization pipeline.

Usage as CLI (for testing):
    uv run python pipeline.py --mic mic.wav [--system system.wav] [--me "Benoît"] [--language fr]

Online meeting  : mic track = the user (named), system track = remote people (diarized).
In-person/file  : only one track, fully diarized.
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
from dataclasses import dataclass, asdict
from typing import Callable, Optional

# torch must be imported before faster_whisper so CUDA/cuDNN libs shipped with
# the torch wheels are loaded into the process for ctranslate2 as well.
import torch  # noqa: E402

Progress = Callable[[str, float], None]


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


def to_wav16k(path: str, tmpdir: str) -> str:
    """Convert any audio/video file to 16 kHz mono WAV with ffmpeg."""
    out = os.path.join(tmpdir, os.path.basename(path) + ".16k.wav")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", path,
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out],
        check=True,
    )
    return out


def load_wav(path: str):
    """Read a 16 kHz mono s16 WAV (as produced by to_wav16k) into float32 numpy."""
    import wave
    import numpy as np

    with wave.open(path, "rb") as f:
        data = f.readframes(f.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- ASR

def transcribe(model, path: str, language: Optional[str], label: str,
               progress: Progress) -> tuple[list[Word], str, float]:
    segments, info = model.transcribe(
        load_wav(path),
        language=language or None,
        vad_filter=True,
        word_timestamps=True,
        beam_size=5,
        condition_on_previous_text=False,
    )
    words: list[Word] = []
    duration = info.duration or 1.0
    for seg in segments:
        for w in seg.words or []:
            words.append(Word(w.start, w.end, w.word))
        progress(f"transcribing {label}", min(seg.end / duration, 1.0))
    return words, info.language, info.duration


# --------------------------------------------------------------------- Diarize

def load_diarizer(hf_token: Optional[str]):
    from pyannote.audio import Pipeline

    last_err: Exception | None = None
    for name in ("pyannote/speaker-diarization-community-1",
                 "pyannote/speaker-diarization-3.1"):
        try:
            try:
                pipe = Pipeline.from_pretrained(name, token=hf_token)
            except TypeError:  # pyannote 3.x
                pipe = Pipeline.from_pretrained(name, use_auth_token=hf_token)
            if pipe is None:
                raise RuntimeError(f"could not load {name} (token / gated access?)")
            if torch.cuda.is_available():
                pipe.to(torch.device("cuda"))
            return pipe
        except Exception as e:  # try the next one
            last_err = e
    raise RuntimeError(
        "Could not load a pyannote diarization pipeline. Check your HuggingFace "
        "token and that you accepted the model terms on huggingface.co. "
        f"Last error: {last_err}"
    )


def diarize(pipe, path: str, min_speakers: Optional[int],
            max_speakers: Optional[int]) -> list[tuple[float, float, str]]:
    waveform, sr = torch.from_numpy(load_wav(path)).unsqueeze(0), 16000
    kwargs = {}
    if min_speakers:
        kwargs["min_speakers"] = min_speakers
    if max_speakers:
        kwargs["max_speakers"] = max_speakers
    out = pipe({"waveform": waveform, "sample_rate": sr}, **kwargs)
    ann = getattr(out, "exclusive_speaker_diarization", None) or \
        getattr(out, "speaker_diarization", None) or out
    return [(turn.start, turn.end, spk) for turn, _, spk in ann.itertracks(yield_label=True)]


def assign_speakers(words: list[Word], turns: list[tuple[float, float, str]]) -> None:
    """Give each word the speaker whose turn overlaps it most (or the nearest one)."""
    if not turns:
        for w in words:
            w.speaker = "SPEAKER_00"
        return
    turns = sorted(turns)
    for w in words:
        best, best_ov = None, 0.0
        for s, e, spk in turns:
            if s > w.end:
                break
            ov = min(e, w.end) - max(s, w.start)
            if ov > best_ov:
                best, best_ov = spk, ov
        if best is None:
            mid = (w.start + w.end) / 2
            best = min(turns, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)))[2]
        w.speaker = best


def rename_by_appearance(words: list[Word]) -> None:
    """pyannote labels are arbitrary; renumber SPEAKER_xx in order of first appearance."""
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
    for m in mic:
        overlapping = " ".join(
            s.text for s in system if s.start < m.end + 1.0 and s.end > m.start - 1.0
        )
        if overlapping:
            a, b = _norm(m.text), _norm(overlapping)
            # ratio of the mic text that is found in the overlapping system text
            sm = difflib.SequenceMatcher(None, a, b)
            matched = sum(bl.size for bl in sm.get_matching_blocks())
            if a and matched / len(a) >= threshold:
                continue
        kept.append(m)
    return kept


# ------------------------------------------------------------------------ Main

def run(mic: str, system: Optional[str] = None, me: str = "Me",
        language: Optional[str] = None, hf_token: Optional[str] = None,
        whisper_model: str = "large-v3", min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None, progress: Progress = _noop) -> dict:
    from faster_whisper import WhisperModel

    hf_token = hf_token or os.environ.get("HF_TOKEN")
    cuda = torch.cuda.is_available()
    warnings: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        progress("converting audio", 0.0)
        mic_wav = to_wav16k(mic, tmp)
        sys_wav = to_wav16k(system, tmp) if system else None

        progress("loading whisper", 0.0)
        model = WhisperModel(whisper_model, device="cuda" if cuda else "cpu",
                             compute_type="float16" if cuda else "int8")
        mic_words, lang, duration = transcribe(
            model, mic_wav, language, "your mic" if sys_wav else "audio", progress)
        sys_words: list[Word] = []
        if sys_wav:
            sys_words, sys_lang, sys_dur = transcribe(
                model, sys_wav, language or lang, "remote audio", progress)
            duration = max(duration, sys_dur)
            if not mic_words:
                lang = sys_lang
        del model
        free_gpu()

        # Track that needs diarization: system (online) or mic (in-person / file)
        to_diarize = (sys_words, sys_wav) if sys_wav else (mic_words, mic_wav)
        if to_diarize[0]:
            progress("diarizing", 0.0)
            try:
                pipe = load_diarizer(hf_token)
                turns = diarize(pipe, to_diarize[1], min_speakers, max_speakers)
                del pipe
            except Exception as e:  # keep the transcript even if diarization fails
                warnings.append(f"Diarization skipped: {e}")
                turns = []
            free_gpu()
            assign_speakers(to_diarize[0], turns)
            rename_by_appearance(to_diarize[0])
            progress("diarizing", 1.0)

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
        "language": lang,
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
    a = ap.parse_args()
    res = run(a.mic, a.system, a.me, a.language, None, a.model, a.min_speakers,
              a.max_speakers,
              progress=lambda s, f: print(f"[{s}] {f:.0%}", file=sys.stderr))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
