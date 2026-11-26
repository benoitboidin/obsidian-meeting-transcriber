"""Small local HTTP server wrapping pipeline.run() as background jobs.

    uv run python server.py --port 8765

Only listens on 127.0.0.1. One job runs at a time (GPU), others queue.
"""
from __future__ import annotations

import argparse
import threading
import traceback
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pipeline

app = FastAPI()
jobs: dict[str, dict] = {}
gpu_lock = threading.Lock()


class JobRequest(BaseModel):
    mic: str
    system: Optional[str] = None
    me: str = "Me"
    language: Optional[str] = None
    hf_token: Optional[str] = None
    whisper_model: str = "large-v3"
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


def _worker(job_id: str, req: JobRequest) -> None:
    job = jobs[job_id]

    def progress(stage: str, frac: float) -> None:
        job["stage"], job["progress"] = stage, frac

    with gpu_lock:
        job["status"] = "running"
        try:
            job["result"] = pipeline.run(
                req.mic, req.system, req.me, req.language, req.hf_token,
                req.whisper_model, req.min_speakers, req.max_speakers, progress)
            job["status"] = "done"
        except Exception as e:
            traceback.print_exc()
            job["status"], job["error"] = "error", str(e)
        finally:
            pipeline.free_gpu()


@app.get("/health")
def health():
    torch = pipeline.torch
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return {"ok": True, "cuda": cuda, "mps": mps,
            "device": "cuda" if cuda else "mps" if mps else "cpu",
            "asr": pipeline.pick_asr_device(),
            "gpu": torch.cuda.get_device_name(0) if cuda else None}


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
