"""Transcription + diarization pipeline.

Usage as CLI (for testing):
    uv run python pipeline.py --mic mic.wav [--system system.wav] [--me "Alice"] [--language fr]

Online meeting  : mic track = the user (named), system track = remote people (diarized).
In-person/file  : only one track, fully diarized.

Long recordings are processed in chunks (default 30 min, cut at the quietest point
near each boundary) so memory use stays bounded whatever the duration.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from asr import Transcriber, Word, transcribe_file
from audio import Wav, plan_chunks, to_wav16k

if TYPE_CHECKING:
    from live import LiveSession

Progress = Callable[[str, float], None]

CHUNK_SEC = 30 * 60


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str


def _noop(_stage: str, _frac: float) -> None:
    pass


# ---------------------------------------------------------------------- Merge

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


def rename_by_appearance(words: list[Word]) -> dict[str, str]:
    """Renumber speakers SPEAKER_xx in order of first appearance; returns old → new labels."""
    mapping: dict[str, str] = {}
    for w in words:
        if w.speaker not in mapping:
            mapping[w.speaker] = f"SPEAKER_{len(mapping):02d}"
        w.speaker = mapping[w.speaker]
    return mapping


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


# ----------------------------------------------------------------------- Main

def run(mic: str, system: Optional[str] = None, me: str = "Me",
        language: Optional[str] = None, hf_token: Optional[str] = None,
        whisper_model: str = "large-v3", min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None, progress: Progress = _noop,
        chunk_sec: float = CHUNK_SEC, live: Optional["LiveSession"] = None,
        after_asr: Callable[[], None] = lambda: None) -> dict:
    """`live`: a live session that already transcribed this recording while it was made.
    `after_asr`: called once transcription is done (e.g. to unload Whisper before diarization)."""
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    warnings: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        progress("converting audio", 0.0)
        mic_wav = Wav(to_wav16k(mic, tmp))
        sys_wav = Wav(to_wav16k(system, tmp)) if system else None
        mic_chunks = plan_chunks(mic_wav, chunk_sec)
        sys_chunks = plan_chunks(sys_wav, chunk_sec) if sys_wav else []
        duration = max(mic_wav.duration, sys_wav.duration if sys_wav else 0.0)

        if live is not None:
            progress("finishing live transcript", 0.0)
            if not live.finish():
                warnings.append(f"Live transcript failed ({live.error}); transcribed again after the meeting.")
                live = None
        if live is not None:
            mic_words, sys_words = live.words("mic"), live.words("system")
            lang = live.detected_language()
            warnings += live.tr.warnings
        else:
            tr = Transcriber(whisper_model)
            progress("loading whisper", 0.0)
            mic_words, lang = transcribe_file(tr, mic_wav, mic_chunks, language,
                                              "your mic" if sys_wav else "audio", progress)
            sys_words: list[Word] = []
            if sys_wav:
                sys_words, sys_lang = transcribe_file(tr, sys_wav, sys_chunks, language or lang,
                                                      "remote audio", progress)
                if sys_lang and len(sys_words) >= len(mic_words):
                    lang = sys_lang
            warnings += tr.warnings

        after_asr()

        # Track that needs diarization: system (online) or mic (in-person / file)
        words, wav, chunks = (sys_words, sys_wav, sys_chunks) if sys_wav else (mic_words, mic_wav, mic_chunks)
        speakers: dict[str, dict] = {}
        if words:
            progress("identifying speakers", 0.0)
            try:
                from diarization import run_diarization
                turns, voices = run_diarization(hf_token, wav, chunks, min_speakers, max_speakers, progress)
            except Exception as e:  # keep the transcript even if diarization fails
                warnings.append(f"Diarization skipped: {e}")
                turns, voices = [], {}
            assign_speakers(words, turns)
            for old, new in rename_by_appearance(words).items():
                if old in voices:
                    emb, secs = voices[old]
                    speakers[new] = {"embedding": [round(float(x), 5) for x in emb], "seconds": round(secs, 1)}

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
        # voice embedding per diarized speaker, for recognising them in later meetings
        "speakers": speakers,
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
