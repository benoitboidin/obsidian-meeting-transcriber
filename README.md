# Meeting Transcriber

Record online and in-person meetings and get a **speaker-labelled transcript and summary** written straight into your notes — with **local models only**. Your audio never leaves your computer.

- **Transcription:** Whisper large-v3 — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on NVIDIA GPUs/CPU, [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) on the Apple GPU
- **Speaker diarization:** [pyannote](https://github.com/pyannote/pyannote-audio)'s `speaker-diarization-community-1`, run with ONNX (same results, no PyTorch, no account needed)
- **Summary:** any [Ollama](https://ollama.com) model (default `qwen3.5:9b`)

## Features

- **Online meetings** (Meet, Teams, Zoom…): your microphone and the system audio are recorded on separate tracks. Your voice is labelled with your name; the remote participants are separated by diarization. Remote audio leaking from your speakers into your mic is detected and removed.
- **In-person meetings:** microphone only, every voice is diarized.
- **Live transcript** in the sidebar while you record. The final note is ready a few seconds after you stop: the meeting has already been transcribed.
- **Voices are remembered:** rename a speaker once (`SPEAKER_01` → `Alice`) and Alice is recognized by her voice in later meetings.
- **Existing recordings:** transcribe any audio/video file (vault file or absolute path).
- **Summary** with decisions and action items (`- [ ]` tasks), in the meeting's language.
- **Rename speakers** (`SPEAKER_00` → `Alice`) across transcript, summary and frontmatter in one go; this also teaches the plugin Alice's voice.
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
| **macOS 14+ (Apple Silicon)** | Recording on macOS (see below). *Transcribe an audio/video file* also works on Intel Macs. |
| NVIDIA GPU or Apple Silicon (recommended) | Fast transcription; a CPU works but is much slower |
| [Ollama](https://ollama.com) (optional) | Summaries |

## Setup

1. Install the plugin and enable it.
2. Install `uv` and `ffmpeg`; optionally Ollama and `ollama pull qwen3.5:9b`.
3. Set **Your name** in settings, then click **Check backend**. The first run installs the Python environment (about 1.5 GB on Linux, most of it NVIDIA's cuBLAS; about 0.6 GB on macOS) and takes a minute or two. Whisper (about 3 GB for `large-v3`) and the speaker models (33 MB) download on first use.

### macOS

macOS cannot capture system audio without a loopback device, so online meetings need [BlackHole](https://existential.audio/blackhole/):

1. `brew install --cask blackhole-2ch`
2. In **Audio MIDI Setup**, create a **Multi-Output Device** with your speakers/headphones **and** BlackHole 2ch, then select it as the system output (so you still hear the meeting).
3. Grant Obsidian **Microphone** and **Screen & System Audio Recording** access in **System Settings → Privacy & Security**.
4. Reload the plugin settings and click **Re-detect** next to *Detected binaries* and under *Recording*: the mic and the BlackHole loopback should appear.

In-person meetings only need the microphone. On Apple Silicon, transcription runs on the Apple GPU with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (falling back to the CPU automatically if it fails); speaker detection runs on the CPU. `large-v3-turbo` is a good default on Macs: much faster than `large-v3` and nearly as accurate. Intel Macs transcribe on the CPU.

## Usage

| Command | What it does |
|---|---|
| Start/stop recording — online meeting | Mic (you) + system audio (others, diarized) |
| Start/stop recording — in-person meeting | Mic only, all voices diarized |
| Transcribe an audio/video file | Pick a vault media file, or type an absolute path |
| Rename speakers in current note | Replace speaker labels everywhere in the note, and remember their voices |
| Show live transcript | Open the live transcript sidebar |
| Re-generate summary from transcript in current note | Rebuilds the `## Summary` section |

The ribbon mic icon starts/stops an online recording; while recording, the status bar shows a timer — click it to stop.

Tips: use headphones in online meetings for the cleanest separation; setting the exact number of speakers improves diarization; `large-v3-turbo` is ~3× faster than `large-v3`.

## Disclosures

- **External programs:** the plugin runs `ffmpeg` (recording/conversion), `pw-dump` (listing audio devices, Linux) and `uv`, which starts a Python server bundled in the plugin. The server's source is written to the backend folder (default `~/.local/share/meeting-transcriber/backend`) and listens on `127.0.0.1` only.
- **Network use:** on first run, `uv` downloads Python packages from PyPI, the server downloads Whisper from HuggingFace and the speaker models from this repository's [`models-v1` release](https://github.com/benoitboidin/obsidian-meeting-transcriber/releases/tag/models-v1) (checksums verified). Summaries call your Ollama server (default `http://127.0.0.1:11434`). **No audio, transcript or note content is sent to any remote service** by this plugin, and there is no telemetry.
- **Voice prints:** to recognize people, the plugin stores a voice embedding (256 numbers) for each person you name, and for each speaker of each transcribed meeting, in `voices/` inside the data folder (default `~/.local/share/meeting-transcriber/voices`). They never go into the vault and never leave your computer. Forget a person in settings → *Voices*, or turn recognition off. Voice prints are biometric data: name people only with their consent.
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

Releasing: `npm version patch` (updates `manifest.json` and `versions.json` and creates the tag), then `git push origin main --follow-tags`. The GitHub Action builds, attests and publishes the release with `main.js`, `manifest.json` and `styles.css`.

## License

MIT for the plugin. The speaker models are pyannote's [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) (CC-BY-4.0, converted to ONNX) and [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (MIT, ONNX export by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)), by pyannote.audio / pyannoteAI and WeSpeaker. Whisper is MIT (OpenAI).
