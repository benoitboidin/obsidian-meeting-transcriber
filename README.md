# Meeting Transcriber (Obsidian, local models)

Records online (mic + system audio) or in-person (mic only) meetings, then writes a diarized transcript and an optional summary into a note. Everything runs locally:

- **faster-whisper** (large-v3, GPU) for transcription
- **pyannote** (speaker-diarization-community-1) for speaker labels
- **Ollama** (default `qwen3.5:9b`) for the summary

## Install

```bash
cd ~/Documents/obsidian-meeting-transcriber
npm install && npm run build
(cd backend && uv sync)
ln -s ~/Documents/obsidian-meeting-transcriber <vault>/.obsidian/plugins/meeting-transcriber
```

Enable **Meeting Transcriber** in Obsidian → Settings → Community plugins.

One-time diarization setup: accept the terms of
[pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1) on HuggingFace,
create a read token, and paste it in the plugin settings. Without a token, transcripts still work but everyone is `SPEAKER_00`.

## Use

| Command | What it does |
|---|---|
| Start/stop recording — online meeting | Mic (= you, labelled with your name) + system audio (= remote people, diarized) |
| Start/stop recording — in-person meeting | Mic only, all voices diarized |
| Transcribe an audio/video file | Vault media file, or type an absolute path |
| Rename speakers in current note | `SPEAKER_00` → `Edwin` across transcript, summary and frontmatter |
| Re-generate summary | Rebuilds `## Summary` from the transcript in the note |

Ribbon mic icon = start/stop online recording. The status bar shows the timer; click it to stop.

Use headphones for online meetings when possible; otherwise remote voices leaking into your mic are detected and removed (text-similarity dedup).

## Backend CLI (debugging)

```bash
cd backend
uv run python pipeline.py --mic audio.m4a                         # single track, diarized
uv run python pipeline.py --mic mic.wav --system system.wav --me Benoît
```
