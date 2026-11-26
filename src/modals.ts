import { App, FuzzySuggestModal, Modal, Notice, Setting, TFile } from "obsidian";
import * as fs from "fs";
import * as path from "path";
import { TRANSCRIPT_LINE } from "./noteWriter";

/** Asks for an optional meeting title. Resolves "" if skipped. */
export class TitleModal extends Modal {
	private value = "";
	private done = false;

	constructor(app: App, private resolve: (title: string) => void) {
		super(app);
	}

	onOpen(): void {
		this.titleEl.setText("Meeting title");
		new Setting(this.contentEl).setName("Title (optional)").addText((t) => {
			t.setPlaceholder("e.g. Weekly sync").onChange((v) => (this.value = v));
			t.inputEl.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); this.submit(); } });
			window.setTimeout(() => t.inputEl.focus(), 50);
		});
		new Setting(this.contentEl).addButton((b) => b.setButtonText("Transcribe").setCta().onClick(() => this.submit()));
	}

	private submit(): void {
		this.done = true;
		this.resolve(this.value.trim());
		this.close();
	}

	onClose(): void {
		if (!this.done) this.resolve("");
		this.contentEl.empty();
	}
}

export function askTitle(app: App): Promise<string> {
	return new Promise((resolve) => new TitleModal(app, resolve).open());
}

const MEDIA_EXT = ["wav", "mp3", "m4a", "ogg", "opus", "flac", "webm", "mp4", "mov", "mkv", "aac"];

type MediaItem = { label: string; absPath: string };

/** Pick an audio/video file from the vault, or type an absolute path. */
export class MediaPickerModal extends FuzzySuggestModal<MediaItem> {
	constructor(app: App, private vaultBase: string, private onPick: (absPath: string) => void) {
		super(app);
		this.setPlaceholder("Pick an audio/video file in the vault, or type an absolute path (/home/…)");
	}

	getItems(): MediaItem[] {
		const items = this.app.vault.getFiles()
			.filter((f: TFile) => MEDIA_EXT.includes(f.extension.toLowerCase()))
			.map((f) => ({ label: f.path, absPath: path.join(this.vaultBase, f.path) }));
		const q = this.inputEl.value.trim().replace(/^~(?=\/)/, process.env.HOME ?? "~");
		if (q.startsWith("/") && fs.existsSync(q) && fs.statSync(q).isFile()) {
			items.unshift({ label: `📂 ${q}`, absPath: q });
		}
		return items;
	}

	getItemText(item: MediaItem): string {
		return item.label;
	}

	onChooseItem(item: MediaItem): void {
		this.onPick(item.absPath);
	}
}

/** Rename speaker labels in a note's transcript (and SPEAKER_xx mentions elsewhere). */
export class RenameSpeakersModal extends Modal {
	private names = new Map<string, string>();

	constructor(app: App, private file: TFile, private speakers: string[]) {
		super(app);
	}

	static speakersIn(content: string): string[] {
		const set = new Set<string>();
		for (const line of content.split("\n")) {
			const m = line.replace(/^>\s?/, "").match(TRANSCRIPT_LINE);
			if (m) set.add(m[2]);
		}
		return [...set];
	}

	onOpen(): void {
		this.titleEl.setText("Rename speakers");
		for (const sp of this.speakers) {
			new Setting(this.contentEl).setName(sp).addText((t) =>
				t.setPlaceholder(sp).onChange((v) => this.names.set(sp, v.trim())));
		}
		new Setting(this.contentEl).addButton((b) => b.setButtonText("Apply").setCta().onClick(() => this.apply()));
	}

	private async apply(): Promise<void> {
		const map = [...this.names.entries()].filter(([from, to]) => to && to !== from);
		if (!map.length) { this.close(); return; }
		const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
		await this.app.vault.process(this.file, (c) => {
			for (const [from, to] of map) {
				// transcript labels
				c = c.replace(new RegExp(`(\\*\\*\\[\\d\\d:\\d\\d:\\d\\d\\] )${esc(from)}(:\\*\\*)`, "g"), `$1${to}$2`);
				// generic SPEAKER_xx labels anywhere (summary, frontmatter)
				if (/^SPEAKER_\d+$/.test(from)) c = c.replace(new RegExp(`\\b${esc(from)}\\b`, "g"), to);
			}
			return c;
		});
		// frontmatter participants for non-generic names
		await this.app.fileManager.processFrontMatter(this.file, (fm) => {
			const f = fm as { participants?: unknown };
			if (Array.isArray(f.participants)) {
				f.participants = f.participants.map((p: unknown) => map.find(([from]) => from === p)?.[1] ?? p);
			}
		});
		new Notice(`Renamed ${map.length} speaker(s)`);
		this.close();
	}

	onClose(): void {
		this.contentEl.empty();
	}
}
