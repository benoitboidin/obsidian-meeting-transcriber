# Meeting Transcriber

Record online and in-person meetings and get a **speaker-labelled transcript and summary** written straight into your notes — with **local models only**. Your audio never leaves your computer.

- **Transcription:** Whisper large-v3 — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on NVIDIA GPUs/CPU, [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) on the Apple GPU
- **Speaker diarization:** [pyannote](https://github.com/pyannote/pyannote-audio) (`speaker-diarization-community-1`)
- **Summary:** any [Ollama](https://ollama.com) model (default `qwen3.5:9b`)

## Features

- **Online meetings** (Meet, Teams, Zoom…): your microphone and the system audio are recorded on separate tracks. Your voice is labelled with your name; the remote participants are separated by diarization. Remote audio leaking from your speakers into your mic is detected and removed.
- **In-person meetings:** microphone only, every voice is diarized.
- **Existing recordings:** transcribe any audio/video file (vault file or absolute path).
- **Summary** with decisions and action items (`- [ ]` tasks), in the meeting's language.
- **Rename speakers** (`SPEAKER_00` → `Alice`) across transcript, summary and frontmatter in one go.
- Output to a new note (with frontmatter: date, duration, language, participants) or at the cursor.

Example output:

```markdown
## Summary
### Summary
The team reviewed the Q3 release …
### Decisions
- Ship on 12 October.
### Action items
- [ ] Alice — update the release notes (Friday)

## Transcript
> [!quote]- Transcript (00:42:10)
> **[00:00:12] Me:** Shall we start?
> **[00:00:15] SPEAKER_00:** Yes, first point is …
```

## Requirements

This plugin drives local programs; it is **desktop-only**.

| Requirement | Why |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Installs and runs the Python transcription server |
| [ffmpeg](https://ffmpeg.org) | Recording and audio conversion |
| **Linux with PipeWire or PulseAudio** | Recording on Linux (mic + system audio). |
| **macOS 14+ (Apple Silicon)** | Recording on macOS (see below). *Transcribe an audio/video file* also works on Intel Macs except for the PyTorch diarization step. |
| NVIDIA GPU or Apple Silicon (recommended) | A few minutes per hour of audio; CPU works but is much slower |
| [Ollama](https://ollama.com) (optional) | Summaries |
| HuggingFace account (free) | One-time download of the pyannote diarization model |

## Setup

1. Install the plugin and enable it.
2. Install `uv` and `ffmpeg`; optionally Ollama and `ollama pull qwen3.5:9b`.
3. On huggingface.co, accept the terms of [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1), create a *read* token, and paste it in the plugin settings. Without it you still get transcripts, but all speakers are `SPEAKER_00`.
4. Set **Your name** in settings, then click **Check backend**. The first run installs the Python environment (PyTorch etc., a few GB) and can take several minutes.

### macOS

macOS cannot capture system audio without a loopback device, so online meetings need [BlackHole](https://existential.audio/blackhole/):

1. `brew install --cask blackhole-2ch`
2. In **Audio MIDI Setup**, create a **Multi-Output Device** with your speakers/headphones **and** BlackHole 2ch, then select it as the system output (so you still hear the meeting).
3. Grant Obsidian **Microphone** and **Screen & System Audio Recording** access in **System Settings → Privacy & Security**.
4. Reload the plugin settings and click **Re-detect** next to *Detected binaries* and under *Recording*: the mic and the BlackHole loopback should appear.

In-person meetings only need the microphone. On Apple Silicon, everything runs on the Apple GPU: transcription with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) and speaker detection with PyTorch (MPS), with an automatic fall back to the CPU if either fails. `large-v3-turbo` is a good default on Macs: much faster than `large-v3` and nearly as accurate. Intel Macs transcribe on the CPU.

## Usage

| Command | What it does |
|---|---|
| Start/stop recording — online meeting | Mic (you) + system audio (others, diarized) |
| Start/stop recording — in-person meeting | Mic only, all voices diarized |
| Transcribe an audio/video file | Pick a vault media file, or type an absolute path |
| Rename speakers in current note | Replace speaker labels everywhere in the note |
| Re-generate summary from transcript in current note | Rebuilds the `## Summary` section |

The ribbon mic icon starts/stops an online recording; while recording, the status bar shows a timer — click it to stop.

Tips: use headphones in online meetings for the cleanest separation; setting the exact number of speakers improves diarization; `large-v3-turbo` is ~3× faster than `large-v3`.

## Disclosures

- **External programs:** the plugin runs `ffmpeg` (recording/conversion), `pw-dump` (listing audio devices, Linux) and `uv`, which starts a Python server bundled in the plugin. The server's source is written to the backend folder (default `~/.local/share/meeting-transcriber/backend`) and listens on `127.0.0.1` only.
- **Network use:** on first run, `uv` downloads Python packages from PyPI, and the server downloads model weights from HuggingFace (using your token for pyannote). Summaries call your Ollama server (default `http://127.0.0.1:11434`). **No audio, transcript or note content is sent to any remote service** by this plugin, and there is no telemetry.
- **Files outside the vault:** recordings are stored in `~/.local/share/meeting-transcriber/recordings` (deleted after transcription unless *Keep audio* is on); the Python environment lives in the backend folder. Both paths are configurable.
- **Environment and system information:** the plugin reads your home folder path and the `PATH`, `HOME`, `XDG_DATA_HOME` and `APPDATA` environment variables, only to find `uv`/`ffmpeg` and its data folders, and passes the environment on to the programs it starts. Nothing from it is stored or sent anywhere.
- **Vault file list:** *Transcribe an audio/video file* lists the audio and video files in your vault so you can pick one. No other files are read.
- **Consent:** recording people may require their consent depending on where you live. You are responsible for informing participants.

## Development

```bash
npm install
npm run dev        # watch build
npm run build      # production build
npm run lint       # the same lint rules as Obsidian's review bot
cd backend && uv run python pipeline.py --mic audio.m4a [--system system.wav --me Alice]
```

Releasing: `npm version patch` (updates `manifest.json` and `versions.json` and creates the tag), then `git push origin main --follow-tags`. The GitHub Action builds, attests and publishes the release with `main.js` and `manifest.json`.

## License

MIT
