import { App, Notice, PluginSettingTab, Setting } from "obsidian";
import * as os from "os";
import * as path from "path";
import * as fs from "fs";
import type MeetingTranscriberPlugin from "./main";
import { listSources, type AudioSource } from "./recorder";

export interface MTSettings {
	myName: string;
	language: string; // "" = auto
	whisperModel: string;
	hfToken: string;
	minSpeakers: number; // 0 = auto
	maxSpeakers: number; // 0 = auto
	uvPath: string;
	ffmpegPath: string;
	backendPort: number;
	backendDir: string;
	backendIdleMinutes: number; // 0 = never stop
	micSource: string; // "" = default
	monitorSource: string; // "" = default sink monitor
	audioDir: string;
	keepAudio: boolean;
	outputMode: "new-note" | "cursor";
	notesFolder: string;
	collapseTranscript: boolean;
	summarize: boolean;
	ollamaUrl: string;
	ollamaModel: string;
}

/** Per-user data folder, outside any vault (keeps the Python env and audio out of sync/git). */
export function dataDir(): string {
	const home = os.homedir();
	if (process.platform === "darwin") return path.join(home, "Library", "Application Support", "meeting-transcriber");
	if (process.platform === "win32") return path.join(process.env.APPDATA ?? path.join(home, "AppData", "Roaming"), "meeting-transcriber");
	return path.join(process.env.XDG_DATA_HOME ?? path.join(home, ".local", "share"), "meeting-transcriber");
}

/** Directories where Homebrew / user-installed tools live but Obsidian's GUI PATH usually misses. */
function binDirs(): string[] {
	const home = os.homedir();
	return ["/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin", path.join(home, ".local", "bin"), "/usr/bin", "/bin"];
}

/** Absolute path of `name` if one of the well-known locations contains it, else `name` (PATH lookup). */
export function findBin(name: string): string {
	for (const d of binDirs()) {
		const c = path.join(d, name);
		if (fs.existsSync(c)) return c;
	}
	return name;
}

/** Keeps a configured binary if it still exists, otherwise re-detects it. */
export function resolveBinary(configured: string, name: string): string {
	const isPath = configured.includes("/") || configured.includes("\\");
	if (isPath && fs.existsSync(configured)) return configured;
	return findBin(name);
}

export const DEFAULT_SETTINGS: MTSettings = {
	myName: "Me",
	language: "",
	whisperModel: "large-v3",
	hfToken: "",
	minSpeakers: 0,
	maxSpeakers: 0,
	uvPath: findBin("uv"),
	ffmpegPath: findBin("ffmpeg"),
	backendPort: 8765,
	backendDir: path.join(dataDir(), "backend"),
	backendIdleMinutes: 10,
	micSource: "",
	monitorSource: "",
	audioDir: path.join(dataDir(), "recordings"),
	keepAudio: false,
	outputMode: "new-note",
	notesFolder: "Meetings",
	collapseTranscript: true,
	summarize: true,
	ollamaUrl: "http://127.0.0.1:11434",
	ollamaModel: "qwen3.5:9b",
};

export class MTSettingTab extends PluginSettingTab {
	constructor(app: App, private plugin: MeetingTranscriberPlugin) {
		super(app, plugin);
	}

	async display(): Promise<void> {
		const { containerEl } = this;
		const s = this.plugin.settings;
		const save = () => this.plugin.saveSettings();
		containerEl.empty();

		new Setting(containerEl).setName("Transcription").setHeading();
		new Setting(containerEl)
			.setName("Your name")
			.setDesc("Used to label your microphone track in online meetings.")
			.addText((t) => t.setValue(s.myName).onChange(async (v) => { s.myName = v.trim() || "Me"; await save(); }));
		new Setting(containerEl)
			.setName("Language")
			.setDesc("Leave on auto-detect for mixed FR/EN use.")
			.addDropdown((d) => d
				.addOptions({ "": "Auto-detect", fr: "Français", en: "English", de: "Deutsch", es: "Español", it: "Italiano" })
				.setValue(s.language)
				.onChange(async (v) => { s.language = v; await save(); }));
		new Setting(containerEl)
			.setName("Whisper model")
			.setDesc("large-v3 = best accuracy; large-v3-turbo = ~3× faster, slightly less accurate.")
			.addDropdown((d) => d
				.addOptions({ "large-v3": "large-v3", "large-v3-turbo": "large-v3-turbo", medium: "medium", small: "small" })
				.setValue(s.whisperModel)
				.onChange(async (v) => { s.whisperModel = v; await save(); }));
		new Setting(containerEl)
			.setName("HuggingFace token")
			.setDesc("Needed once to download pyannote diarization. Accept the terms of pyannote/speaker-diarization-community-1 on huggingface.co first.")
			.addText((t) => {
				t.inputEl.type = "password";
				t.setPlaceholder("hf_…").setValue(s.hfToken).onChange(async (v) => { s.hfToken = v.trim(); await save(); });
			});
		new Setting(containerEl)
			.setName("Speakers (min / max)")
			.setDesc("0 = let the model decide. Setting the exact number improves diarization.")
			.addText((t) => t.setValue(String(s.minSpeakers)).onChange(async (v) => { s.minSpeakers = parseInt(v) || 0; await save(); }))
			.addText((t) => t.setValue(String(s.maxSpeakers)).onChange(async (v) => { s.maxSpeakers = parseInt(v) || 0; await save(); }));

		new Setting(containerEl).setName("Recording").setHeading();
		const isMac = process.platform === "darwin";
		if (isMac) {
			new Setting(containerEl)
				.setName("macOS system audio")
				.setDesc("System audio needs a loopback device: install BlackHole 2ch (`brew install --cask blackhole-2ch`), then create a Multi-Output Device with your speakers + BlackHole in Audio MIDI Setup and use it as output. Grant Obsidian microphone and Screen & System Audio Recording permissions. macOS records the mic only unless a loopback device is present.")
				.addButton((b) => b.setButtonText("Re-detect").onClick(() => this.display()));
		}
		let sources: AudioSource[] = [];
		try { sources = await listSources(s.ffmpegPath); } catch { /* ffmpeg missing */ }
		const srcOptions = (kind: AudioSource["kind"], label: string) => {
			const o: Record<string, string> = { "": label };
			for (const x of sources) if (x.kind === kind) o[x.name] = x.name;
			return o;
		};
		const loopbackMissing = isMac && !sources.some((x) => x.kind === "monitor");
		new Setting(containerEl)
			.setName("Microphone")
			.setDesc(isMac ? "Input device used for your own voice." : "")
			.addDropdown((d) => d.addOptions(srcOptions("mic", "System default"))
				.setValue(s.micSource).onChange(async (v) => { s.micSource = v; await save(); }));
		new Setting(containerEl)
			.setName("System audio (remote participants)")
			.setDesc(isMac
				? (loopbackMissing
					? "⚠️ No loopback input found — install BlackHole 2ch to record online meetings."
					: "Your BlackHole/loopback input device.")
				: "A '.monitor' source of your output device.")
			.addDropdown((d) => d.addOptions(srcOptions("monitor", isMac ? "Default loopback" : "Monitor of default output"))
				.setValue(s.monitorSource).onChange(async (v) => { s.monitorSource = v; await save(); }));
		new Setting(containerEl)
			.setName("Audio folder")
			.setDesc("Where recordings are written (kept outside the vault so obsidian-git ignores them).")
			.addText((t) => t.setValue(s.audioDir).onChange(async (v) => { s.audioDir = v.trim(); await save(); }));
		new Setting(containerEl)
			.setName("Keep audio after transcription")
			.addToggle((t) => t.setValue(s.keepAudio).onChange(async (v) => { s.keepAudio = v; await save(); }));

		new Setting(containerEl).setName("Output").setHeading();
		new Setting(containerEl)
			.setName("Where to write")
			.addDropdown((d) => d
				.addOptions({ "new-note": "New note in meetings folder", cursor: "At cursor in active note" })
				.setValue(s.outputMode)
				.onChange(async (v) => { s.outputMode = v as MTSettings["outputMode"]; await save(); }));
		new Setting(containerEl)
			.setName("Meetings folder")
			.addText((t) => t.setValue(s.notesFolder).onChange(async (v) => { s.notesFolder = v.trim(); await save(); }));
		new Setting(containerEl)
			.setName("Collapse transcript")
			.setDesc("Wrap the transcript in a folded callout.")
			.addToggle((t) => t.setValue(s.collapseTranscript).onChange(async (v) => { s.collapseTranscript = v; await save(); }));

		new Setting(containerEl).setName("Summary (Ollama)").setHeading();
		new Setting(containerEl)
			.setName("Generate summary")
			.addToggle((t) => t.setValue(s.summarize).onChange(async (v) => { s.summarize = v; await save(); }));
		new Setting(containerEl)
			.setName("Ollama URL")
			.addText((t) => t.setValue(s.ollamaUrl).onChange(async (v) => { s.ollamaUrl = v.trim(); await save(); }));
		new Setting(containerEl)
			.setName("Ollama model")
			.addText((t) => t.setValue(s.ollamaModel).onChange(async (v) => { s.ollamaModel = v.trim(); await save(); }));

		new Setting(containerEl).setName("Backend").setHeading();
		new Setting(containerEl)
			.setName("uv path")
			.addText((t) => t.setValue(s.uvPath).onChange(async (v) => { s.uvPath = v.trim(); await save(); }));
		new Setting(containerEl)
			.setName("ffmpeg path")
			.addText((t) => t.setValue(s.ffmpegPath).onChange(async (v) => { s.ffmpegPath = v.trim(); await save(); }));
		new Setting(containerEl)
			.setName("Detected binaries")
			.setDesc(`uv: ${s.uvPath}\nffmpeg: ${s.ffmpegPath}`)
			.addButton((b) => b.setButtonText("Re-detect").onClick(async () => {
				s.uvPath = findBin("uv");
				s.ffmpegPath = findBin("ffmpeg");
				await save();
				this.display();
			}));
		new Setting(containerEl)
			.setName("Backend folder")
			.setDesc("Where the Python server and its environment (several GB: PyTorch, models' code) are installed.")
			.addText((t) => t.setValue(s.backendDir).onChange(async (v) => { s.backendDir = v.trim(); await save(); }));
		new Setting(containerEl)
			.setName("Port")
			.addText((t) => t.setValue(String(s.backendPort)).onChange(async (v) => { s.backendPort = parseInt(v) || 8765; await save(); }));
		new Setting(containerEl)
			.setName("Stop backend after idle (minutes)")
			.setDesc("Frees GPU memory. 0 = keep running while Obsidian is open.")
			.addText((t) => t.setValue(String(s.backendIdleMinutes)).onChange(async (v) => { s.backendIdleMinutes = parseInt(v) || 0; await save(); }));
		new Setting(containerEl)
			.setName("Check backend")
			.setDesc("Installs Python deps if needed (first time can take several minutes), starts the server and reports GPU.")
			.addButton((b) => b.setButtonText("Check").onClick(async () => {
				b.setDisabled(true).setButtonText("Starting…");
				try {
					const h = await this.plugin.backend.ensureRunning();
					const perf = h.cuda ? `GPU: ${h.gpu}` : h.mps ? "Apple GPU (MPS, experimental)" : "CPU only (slow)";
					new Notice(`Backend OK — ${perf}`);
				} catch (e) {
					new Notice(`Backend error: ${(e as Error).message}`, 10000);
				} finally {
					b.setDisabled(false).setButtonText("Check");
				}
			}));
	}
}
