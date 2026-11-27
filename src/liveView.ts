import { ItemView, TFile, WorkspaceLeaf } from "obsidian";
import type { Backend, LiveSegment } from "./backend";
import { hms } from "./noteWriter";

export const LIVE_VIEW_TYPE = "meeting-transcriber-live";

type ViewState =
	| { kind: "idle" }
	| { kind: "starting" }
	| { kind: "live"; lag: number }
	| { kind: "processing"; message: string }
	| { kind: "saved"; file: TFile }
	| { kind: "error"; message: string };

/** Right-sidebar view showing the transcript while the meeting is recorded. */
export class LiveTranscriptView extends ItemView {
	private statusEl!: HTMLElement;
	private listEl!: HTMLElement;
	private labels: Record<LiveSegment["track"], string> = { mic: "", system: "" };

	constructor(leaf: WorkspaceLeaf) {
		super(leaf);
	}

	getViewType(): string {
		return LIVE_VIEW_TYPE;
	}

	getDisplayText(): string {
		return "Live transcript";
	}

	getIcon(): string {
		return "audio-lines";
	}

	async onOpen(): Promise<void> {
		this.contentEl.empty();
		this.contentEl.addClass("mt-live");
		this.statusEl = this.contentEl.createDiv({ cls: "mt-live-status" });
		this.listEl = this.contentEl.createDiv({ cls: "mt-live-list" });
		this.show({ kind: "idle" });
		await Promise.resolve();
	}

	/** Clears the view for a new recording. `labels`: speaker shown for each track ("" = none). */
	reset(labels: Record<LiveSegment["track"], string>): void {
		this.labels = labels;
		this.listEl.empty();
	}

	show(state: ViewState): void {
		const el = this.statusEl;
		el.empty();
		el.removeClass("is-live", "is-error");
		switch (state.kind) {
			case "idle":
				el.setText("Start a recording to see the transcript here as people speak.");
				break;
			case "starting":
				el.setText("Starting the transcription server…");
				break;
			case "live":
				el.addClass("is-live");
				el.setText(state.lag > 20 ? `Live · catching up (${Math.round(state.lag)} s behind)` : "Live");
				break;
			case "processing":
				el.setText(state.message);
				break;
			case "saved": {
				el.appendText("Saved to ");
				const a = el.createEl("a", { text: state.file.basename, href: "#" });
				a.addEventListener("click", (ev) => {
					ev.preventDefault();
					void this.app.workspace.getLeaf(false).openFile(state.file);
				});
				break;
			}
			case "error":
				el.addClass("is-error");
				el.setText(state.message);
				break;
		}
	}

	append(segments: LiveSegment[]): void {
		if (!segments.length) return;
		const list = this.listEl;
		const atBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 40;
		for (const seg of segments) {
			const line = list.createDiv({ cls: `mt-live-line mt-track-${seg.track}` });
			line.createSpan({ cls: "mt-live-time", text: hms(seg.start) });
			const who = this.labels[seg.track];
			if (who) line.createSpan({ cls: "mt-live-speaker", text: who });
			line.createSpan({ cls: "mt-live-text", text: seg.text });
		}
		if (atBottom) list.scrollTop = list.scrollHeight;
	}
}

/** Drives one live session: starts it on the backend, polls new text into the view. */
export class LiveSession {
	private id: Promise<string | null>;
	private next = 0;
	private timer: number | null = null;
	private failed = false;

	constructor(
		private backend: Backend,
		files: { mic: string; system?: string },
		private view: () => LiveTranscriptView | null,
	) {
		this.view()?.show({ kind: "starting" });
		this.id = backend.startLive(files).then(
			(id) => {
				this.timer = window.setInterval(() => void this.poll(id), 2000);
				return id;
			},
			(e: Error) => {
				this.fail(`Live transcript unavailable: ${e.message}. The transcript will be made after the meeting.`);
				return null;
			},
		);
	}

	private fail(message: string): void {
		this.failed = true;
		this.view()?.show({ kind: "error", message });
	}

	private async poll(id: string): Promise<void> {
		if (this.failed) return;
		try {
			const st = await this.backend.pollLive(id, this.next);
			this.next += st.segments.length;
			const view = this.view();
			view?.append(st.segments);
			if (st.status === "error") {
				this.stopPolling();
				this.fail(`Live transcript stopped: ${st.error ?? "unknown error"}. The transcript will be made after the meeting.`);
			} else {
				view?.show({ kind: "live", lag: st.lag });
			}
		} catch {
			// backend busy or restarting: try again on the next tick
		}
	}

	private stopPolling(): void {
		if (this.timer !== null) window.clearInterval(this.timer);
		this.timer = null;
	}

	/** Recording stopped: returns the session id to hand to the final job (null if live failed). */
	async stop(): Promise<string | null> {
		const id = await this.id;
		this.stopPolling();
		if (!id) return null;
		if (!this.failed) await this.poll(id); // last lines
		if (!this.failed) return id;
		await this.backend.cancelLive(id);
		return null;
	}

	/** Recording discarded. */
	async cancel(): Promise<void> {
		const id = await this.id;
		this.stopPolling();
		if (id) await this.backend.cancelLive(id);
	}
}
