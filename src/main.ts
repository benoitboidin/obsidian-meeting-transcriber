import { FileSystemAdapter, Notice, Plugin, setIcon, TFile } from "obsidian";
import * as fs from "fs";
import * as path from "path";
import { Backend, TranscriptResult } from "./backend";
import { askTitle, MediaPickerModal, RenameSpeakersModal } from "./modals";
import { hms, plainTranscript, transcriptFromNote, writeResult } from "./noteWriter";
import { Recorder, RecordingMode } from "./recorder";
import { DEFAULT_SETTINGS, MTSettings, MTSettingTab, resolveBinary } from "./settings";
import { summarize } from "./summarizer";

export default class MeetingTranscriberPlugin extends Plugin {
	settings!: MTSettings;
	backend!: Backend;
	private recorder!: Recorder;
	private statusEl!: HTMLElement;
	private ribbonEl!: HTMLElement;
	private timer: number | null = null;
	private recordTarget: TFile | null = null;

	async onload(): Promise<void> {
		await this.loadSettings();
		const vaultBase = this.vaultBase();
		this.backend = new Backend(() => this.settings);
		this.recorder = new Recorder(() => this.settings.ffmpegPath);

		this.statusEl = this.addStatusBarItem();
		this.statusEl.addClass("mod-clickable");
		this.statusEl.onClickEvent(() => { if (this.recorder.isRecording) void this.stopRecording(); });
		this.ribbonEl = this.addRibbonIcon("mic", "Start/stop meeting recording (online)", () => void this.toggle("online"));

		this.addCommand({ id: "toggle-online", name: "Start/stop recording — online meeting (mic + system audio)", callback: () => void this.toggle("online") });
		this.addCommand({ id: "toggle-in-person", name: "Start/stop recording — in-person meeting (mic only)", callback: () => void this.toggle("in-person") });
		this.addCommand({
			id: "transcribe-file", name: "Transcribe an audio/video file",
			callback: () => new MediaPickerModal(this.app, vaultBase, (p) => void this.transcribeFile(p)).open(),
		});
		this.addCommand({
			id: "rename-speakers", name: "Rename speakers in current note",
			checkCallback: (checking) => {
				const file = this.app.workspace.getActiveFile();
				if (!file || file.extension !== "md") return false;
				if (!checking) void this.renameSpeakers(file);
				return true;
			},
		});
		this.addCommand({
			id: "resummarize", name: "Re-generate summary from transcript in current note",
			checkCallback: (checking) => {
				const file = this.app.workspace.getActiveFile();
				if (!file || file.extension !== "md") return false;
				if (!checking) void this.resummarize(file);
				return true;
			},
		});

		this.addSettingTab(new MTSettingTab(this.app, this));
		this.updateUi();
	}

	async onunload(): Promise<void> {
		if (this.recorder.isRecording) {
			try {
				const rec = await this.recorder.stop();
				new Notice(`Meeting recording stopped because the plugin was unloaded. Audio kept in ${rec.dir}`);
			} catch { /* ignore */ }
		}
		this.stopTimer();
		this.backend.stop();
	}

	private vaultBase(): string {
		const a = this.app.vault.adapter;
		if (!(a instanceof FileSystemAdapter)) throw new Error("Meeting Transcriber needs a desktop vault");
		return a.getBasePath();
	}

	async loadSettings(): Promise<void> {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
		// Saved settings may point at a binary that no longer exists (or a bare name that the GUI PATH cannot find).
		const uv = resolveBinary(this.settings.uvPath, "uv");
		const ffmpeg = resolveBinary(this.settings.ffmpegPath, "ffmpeg");
		if (uv !== this.settings.uvPath || ffmpeg !== this.settings.ffmpegPath) {
			this.settings.uvPath = uv;
			this.settings.ffmpegPath = ffmpeg;
			await this.saveSettings();
		}
	}

	async saveSettings(): Promise<void> {
		await this.saveData(this.settings);
	}

	// ------------------------------------------------------------------ recording

	private async toggle(mode: RecordingMode): Promise<void> {
		if (this.recorder.isRecording) return this.stopRecording();
		if (process.platform !== "linux" && process.platform !== "darwin") {
			new Notice("Recording is supported on Linux and macOS. You can still use “Transcribe an audio/video file”.", 8000);
			return;
		}
		try {
			await this.recorder.start(mode, this.settings.audioDir, this.settings.micSource, this.settings.monitorSource);
		} catch (e) {
			new Notice(`Could not start recording: ${(e as Error).message}`, 15000);
			return;
		}
		this.recordTarget = this.settings.outputMode === "cursor" ? this.app.workspace.getActiveFile() : null;
		new Notice(mode === "online" ? "Recording mic + system audio…" : "Recording microphone…");
		this.startTimer();
		this.updateUi();
		// Warm up the backend while the meeting runs.
		this.backend.ensureRunning().catch(() => { /* reported later */ });
	}

	private async stopRecording(): Promise<void> {
		let rec;
		try {
			rec = await this.recorder.stop();
		} catch (e) {
			new Notice(`Stop failed: ${(e as Error).message}`);
			return;
		} finally {
			this.stopTimer();
			this.updateUi();
		}
		const title = this.settings.outputMode === "new-note" ? await askTitle(this.app) : "";
		await this.process(
			{ mic: rec.mic, system: rec.system, me: this.settings.myName },
			rec.startedAt, title, rec.dir, this.recordTarget,
		);
	}

	private async transcribeFile(absPath: string): Promise<void> {
		const target = this.settings.outputMode === "cursor" ? this.app.workspace.getActiveFile() : null;
		const title = this.settings.outputMode === "new-note" ? await askTitle(this.app) : "";
		const mtime = fs.statSync(absPath).mtime;
		await this.process({ mic: absPath, me: this.settings.myName }, mtime, title || path.parse(absPath).name, null, target);
	}

	/** Shared pipeline: transcribe → summarize → write → cleanup. */
	private async process(
		input: { mic: string; system?: string; me: string }, date: Date, title: string,
		audioDir: string | null, target: TFile | null,
	): Promise<void> {
		const notice = new Notice("Meeting Transcriber: starting backend…", 0);
		let res: TranscriptResult;
		try {
			res = await this.backend.transcribe(input, (stage, frac) =>
				notice.setMessage(`Meeting Transcriber: ${stage}${frac > 0 && frac < 1 ? ` ${Math.round(frac * 100)}%` : "…"}`));
		} catch (e) {
			notice.hide();
			new Notice(`Transcription failed: ${(e as Error).message}${audioDir ? `\nAudio kept in ${audioDir}` : ""}`, 0);
			return;
		}
		if (!res.segments.length) {
			notice.hide();
			new Notice(`No speech detected.${audioDir ? ` Audio kept in ${audioDir}` : ""}`, 10000);
			return;
		}

		let summary: string | null = null;
		if (this.settings.summarize) {
			try {
				summary = await summarize(this.settings, plainTranscript(res.segments), res.language,
					(m) => notice.setMessage(`Meeting Transcriber: ${m}`));
			} catch (e) {
				res.warnings = [...(res.warnings ?? []), `Summary failed (is Ollama running?): ${(e as Error).message}`];
			}
		}

		const file = await writeResult(this.app, this.settings, res, summary, date, title,
			target ? { file: target, mode: "cursor" } : undefined);
		notice.hide();
		new Notice(`Transcript written to ${file.basename} (${hms(res.duration)})`);
		if (!target) await this.app.workspace.getLeaf(false).openFile(file);

		if (audioDir && !this.settings.keepAudio) fs.rmSync(audioDir, { recursive: true, force: true });
	}

	// ------------------------------------------------------------------ note tools

	private async renameSpeakers(file: TFile): Promise<void> {
		const speakers = RenameSpeakersModal.speakersIn(await this.app.vault.read(file));
		if (!speakers.length) { new Notice("No transcript found in this note"); return; }
		new RenameSpeakersModal(this.app, file, speakers).open();
	}

	private async resummarize(file: TFile): Promise<void> {
		const content = await this.app.vault.read(file);
		const transcript = transcriptFromNote(content);
		if (!transcript) { new Notice("No transcript found in this note"); return; }
		const lang = this.app.metadataCache.getFileCache(file)?.frontmatter?.language as string | undefined;
		const notice = new Notice("Summarizing…", 0);
		try {
			const summary = await summarize(this.settings, transcript, lang, (m) => notice.setMessage(m));
			await this.app.vault.process(file, (c) => {
				const block = `## Summary\n\n${summary.trim()}\n\n`;
				const existing = /## Summary\n[\s\S]*?(?=\n## |$)/;
				if (existing.test(c)) return c.replace(existing, block.trimEnd() + "\n");
				const idx = c.indexOf("## Transcript");
				return idx >= 0 ? c.slice(0, idx) + block + c.slice(idx) : `${c.trimEnd()}\n\n${block}`;
			});
			new Notice("Summary updated");
		} catch (e) {
			new Notice(`Summary failed: ${(e as Error).message}`, 10000);
		} finally {
			notice.hide();
		}
	}

	// ------------------------------------------------------------------ UI

	private startTimer(): void {
		this.stopTimer();
		this.timer = window.setInterval(() => this.updateUi(), 1000);
		this.registerInterval(this.timer);
	}

	private stopTimer(): void {
		if (this.timer !== null) window.clearInterval(this.timer);
		this.timer = null;
	}

	private updateUi(): void {
		const start = this.recorder.startedAt;
		if (this.recorder.isRecording && start) {
			const label = this.recorder.mode === "online" ? "REC online" : "REC";
			this.statusEl.setText(`🔴 ${label} ${hms((Date.now() - start.getTime()) / 1000)}`);
			this.statusEl.setAttr("aria-label", "Click to stop recording");
			this.statusEl.show();
			setIcon(this.ribbonEl, "square");
			this.ribbonEl.setAttr("aria-label", "Stop meeting recording");
		} else {
			this.statusEl.hide();
			setIcon(this.ribbonEl, "mic");
			this.ribbonEl.setAttr("aria-label", "Start/stop meeting recording (online)");
		}
	}
}

