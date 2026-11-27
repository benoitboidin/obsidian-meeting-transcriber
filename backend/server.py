"""Small local HTTP server wrapping the pipeline as background jobs, plus live sessions.

    uv run python server.py --port 8765

Only listens on 127.0.0.1. One job runs at a time (GPU), others queue.
"""
from __future__ import annotations

import argparse
import sys
import threading
import traceback
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import asr
import pipeline
from live import LiveSession

app = FastAPI()
jobs: dict[str, dict] = {}
lives: dict[str, LiveSession] = {}
job_lock = threading.Lock()


def _release_if_idle(busy_job: bool = False) -> None:
    """Free the Whisper model (GPU memory for diarization and Ollama) unless another
    recording is being transcribed live or another job is about to need it."""
    if any(s.status in ("running", "finishing") for s in lives.values()):
        return
    if job_lock.locked() and not busy_job:
        return
    asr.release()


class JobRequest(BaseModel):
    mic: str
    system: Optional[str] = None
    me: str = "Me"
    language: Optional[str] = None
    hf_token: Optional[str] = None
    whisper_model: str = "large-v3"
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    live_id: Optional[str] = None


class LiveRequest(BaseModel):
    mic: str
    system: Optional[str] = None
    language: Optional[str] = None
    whisper_model: str = "large-v3"


def _worker(job_id: str, req: JobRequest) -> None:
    job = jobs[job_id]

    def progress(stage: str, frac: float) -> None:
        job["stage"], job["progress"] = stage, frac

    live = lives.get(req.live_id) if req.live_id else None
    with job_lock:
        job["status"] = "running"
        try:
            job["result"] = pipeline.run(
                req.mic, req.system, req.me, req.language, req.hf_token,
                req.whisper_model, req.min_speakers, req.max_speakers, progress, live=live,
                after_asr=lambda: _release_if_idle(busy_job=True))
            job["status"] = "done"
        except Exception as e:
            traceback.print_exc()
            job["status"], job["error"] = "error", str(e)
        finally:
            if req.live_id:
                lives.pop(req.live_id, None)
    _release_if_idle()


@app.get("/health")
def health():
    device = asr.pick_device()
    try:  # the diarizer still uses PyTorch for now
        import torch
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        mps = False
    cuda = device == "cuda"
    return {"ok": True, "cuda": cuda, "mps": mps, "asr": device,
            "device": "cuda" if cuda else "mps" if mps else "cpu",
            "gpu": asr.gpu_name() if cuda else None,
            "platform": sys.platform}


@app.post("/jobs")
def create_job(req: JobRequest):
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "stage": "queued", "progress": 0.0}
    threading.Thread(target=_worker, args=(job_id, req), daemon=True).start()
    return {"id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "unknown job")
    job = jobs[job_id]
    if job["status"] in ("done", "error"):
        jobs.pop(job_id)  # deliver result once
    return job


@app.post("/live")
def start_live(req: LiveRequest):
    tracks = {"mic": req.mic}
    if req.system:
        tracks["system"] = req.system
    live_id = uuid.uuid4().hex
    lives[live_id] = LiveSession(tracks, req.whisper_model, req.language)
    return {"id": live_id}


@app.get("/live/{live_id}")
def get_live(live_id: str, since: int = 0):
    s = lives.get(live_id)
    if s is None:
        raise HTTPException(404, "unknown live session")
    return {"status": s.status, "error": s.error, "lag": round(s.lag(), 1),
            "language": s.detected_language(), "segments": s.since(since)}


@app.delete("/live/{live_id}")
def cancel_live(live_id: str):
    s = lives.pop(live_id, None)
    if s:
        s.cancel()
    _release_if_idle()
    return {"ok": True}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
