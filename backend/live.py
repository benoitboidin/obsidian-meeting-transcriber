"""Live transcript: transcribes the recording while ffmpeg is still writing it.

Each track is cut at natural pauses (every ~5-30 s) and transcribed with the same
model as the final pass, so the words are kept for the final note: when the
recording stops, only the last few seconds are left to transcribe.
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Optional

import numpy as np

from asr import Context, Transcriber, Word, speech_regions
from audio import SR, GrowingWav, quietest_point

MIN_SEC = 5.0      # wait for at least this much new audio before transcribing
MAX_SEC = 28.0     # never transcribe more than this at once (one Whisper window)
MIN_PAUSE = 0.4    # a pause this long is a safe place to cut


class Track:
    def __init__(self, name: str, path: str):
        self.name = name
        self.wav = GrowingWav(path)
        self.pos = 0  # samples already transcribed (or skipped as silence)
        self.words: list[Word] = []
        self.previous = ""  # last transcribed text, prompts the next piece


class LiveSession:
    def __init__(self, tracks: dict[str, str], model_name: str, language: Optional[str]):
        self.tracks = [Track(n, p) for n, p in tracks.items()]
        self.tr = Transcriber(model_name)
        self.ctx = Context(language)
        self.segments: list[dict] = []  # append-only, polled by the plugin
        self.finishing = False
        self.cancelled = False
        self.error: Optional[str] = None
        self.done = threading.Event()
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------------------------------------------ public

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if self.done.is_set():
            return "done"
        return "finishing" if self.finishing else "running"

    def lag(self) -> float:
        """Seconds of recorded audio not transcribed yet (the most behind track)."""
        return max((max(0, t.wav.samples() - t.pos) / SR for t in self.tracks), default=0.0)

    def since(self, k: int) -> list[dict]:
        with self._lock:
            return self.segments[k:]

    def finish(self, timeout: Optional[float] = None) -> bool:
        """Transcribe what is left (the recording has stopped) and wait. False on error."""
        self.finishing = True
        self.done.wait(timeout)
        return self.done.is_set() and not self.error

    def cancel(self) -> None:
        self.cancelled = True

    def words(self, track: str) -> list[Word]:
        for t in self.tracks:
            if t.name == track:
                return t.words
        return []

    def detected_language(self) -> Optional[str]:
        return self.ctx.guess()

    # ---------------------------------------------------------------- internal

    def _loop(self) -> None:
        try:
            while not self.cancelled:
                final = self.finishing
                worked = False
                for t in self.tracks:
                    worked = self._step(t, final) or worked
                if final and not worked:
                    break
                if not worked:
                    time.sleep(0.5)
        except Exception as e:
            traceback.print_exc()
            self.error = str(e) or repr(e)
        finally:
            self.done.set()

    def _cut(self, audio: np.ndarray, last: bool, final: bool) -> Optional[int]:
        """Where to cut `audio` (samples), or None to wait for more.
        `last`: the recording has stopped and `audio` runs to its end."""
        n = len(audio)
        if last:
            return n
        regions = speech_regions(audio)
        if not regions:  # silence: skip it, keeping the last second as context
            return n if final else max(0, n - SR) or None
        # pauses: between speech regions, and after the last one
        pauses = [(a[1], b[0]) for a, b in zip(regions, regions[1:])]
        pauses.append((regions[-1][1], n / SR))
        for start, end in reversed(pauses):
            mid = (start + end) / 2
            if end - start >= MIN_PAUSE and mid >= 3.0 and end <= n / SR:
                # trailing pause: only usable once it is long enough to be a real one
                if end == n / SR and end - start < 2 * MIN_PAUSE:
                    continue
                return int(mid * SR)
        if n >= MAX_SEC * SR:  # nobody pauses: cut at the quietest point of the last 5 s
            return int(quietest_point(audio[-5 * SR:], (n - 5 * SR) / SR) * SR)
        return None

    def _step(self, t: Track, final: bool) -> bool:
        pending = t.wav.samples() - t.pos
        if pending <= 0 or (not final and pending < MIN_SEC * SR):
            return False
        take = min(pending, int(MAX_SEC * SR))
        audio = t.wav.read(t.pos, t.pos + take)
        cut = self._cut(audio, final and take == pending, final)
        if not cut:
            return False
        words = self.tr.piece(audio[:cut], t.pos / SR, self.ctx, t.previous)
        t.previous = "".join(w.text for w in words) or t.previous
        t.words += words
        t.pos += cut
        self._publish(t.name, words)
        return True

    def _publish(self, track: str, words: list[Word], max_gap: float = 1.5) -> None:
        segs: list[dict] = []
        for w in words:
            if segs and w.start - segs[-1]["end"] <= max_gap:
                segs[-1]["end"] = w.end
                segs[-1]["text"] += w.text
            else:
                segs.append({"start": w.start, "end": w.end, "track": track, "text": w.text})
        with self._lock:
            for s in segs:
                s["text"] = s["text"].strip()
                if s["text"]:
                    self.segments.append(s)
