import { App, MarkdownView, normalizePath, TFile } from "obsidian";
import type { Segment, TranscriptResult } from "./backend";
import type { MTSettings } from "./settings";

export function hms(sec: number): string {
	const t = Math.max(0, Math.floor(sec));
	const p = (n: number) => String(n).padStart(2, "0");
	return `${p(Math.floor(t / 3600))}:${p(Math.floor((t % 3600) / 60))}:${p(t % 60)}`;
}

export function speakersOf(segments: Segment[]): string[] {
	return [...new Set(segments.map((s) => s.speaker))];
}

/** Plain text used as LLM input. */
export function plainTranscript(segments: Segment[]): string {
	return segments.map((s) => `[${hms(s.start)}] ${s.speaker}: ${s.text}`).join("\n");
}

export const TRANSCRIPT_LINE = /\*\*\[(\d\d:\d\d:\d\d)\] ([^*\n]+?):\*\* (.*)/;

/** Extract "[hh:mm:ss] Speaker: text" lines back from a note (for re-summarize / rename). */
export function transcriptFromNote(content: string): string {
	const out: string[] = [];
	for (const line of content.split("\n")) {
		const m = line.replace(/^>\s?/, "").match(TRANSCRIPT_LINE);
		if (m) out.push(`[${m[1]}] ${m[2]}: ${m[3]}`);
	}
	return out.join("\n");
}

export function renderBody(res: TranscriptResult, summary: string | null, s: MTSettings): string {
	const parts: string[] = [];
	if (res.warnings?.length) {
		parts.push(`> [!warning] Transcription warnings\n${res.warnings.map((w) => `> ${w.split("\n")[0]}`).join("\n")}\n`);
	}
	if (summary !== null) parts.push(`## Summary\n\n${summary.trim()}\n`);
	const lines = res.segments.map((seg) => `**[${hms(seg.start)}] ${seg.speaker}:** ${seg.text}`);
	if (s.collapseTranscript) {
		parts.push(`## Transcript\n\n> [!quote]- Transcript (${hms(res.duration)})\n${lines.map((l) => `> ${l}\n>`).join("\n")}\n`);
	} else {
		parts.push(`## Transcript\n\n${lines.join("\n\n")}\n`);
	}
	return parts.join("\n");
}

function frontmatter(res: TranscriptResult, date: Date): string {
	const p = (n: number) => String(n).padStart(2, "0");
	const day = `${date.getFullYear()}-${p(date.getMonth() + 1)}-${p(date.getDate())}`;
	const people = speakersOf(res.segments).map((x) => JSON.stringify(x)).join(", ");
	return `---\ndate: ${day}\ntime: "${p(date.getHours())}:${p(date.getMinutes())}"\nduration: "${hms(res.duration)}"\nlanguage: ${res.language}\nparticipants: [${people}]\ntype: meeting\n---\n`;
}

function safeName(t: string): string {
	return t.replace(/[\\/:*?"<>|#^[\]]/g, "-").trim();
}

/** Writes the result according to settings; returns the file written to. */
export async function writeResult(
	app: App, s: MTSettings, res: TranscriptResult, summary: string | null, date: Date, title: string,
	target?: { file: TFile; mode: "cursor" | "append" },
): Promise<TFile> {
	const body = renderBody(res, summary, s);

	if (target || s.outputMode === "cursor") {
		const view = app.workspace.getActiveViewOfType(MarkdownView);
		const file = target?.file ?? view?.file;
		if (file) {
			if (view && view.file === file && (target?.mode ?? "cursor") === "cursor") {
				view.editor.replaceSelection(`\n${body}\n`);
			} else {
				await app.vault.process(file, (c) => `${c.trimEnd()}\n\n${body}\n`);
			}
			return file;
		}
		// no active note → fall through to new note
	}

	const p = (n: number) => String(n).padStart(2, "0");
	const base = `${date.getFullYear()}-${p(date.getMonth() + 1)}-${p(date.getDate())} ${p(date.getHours())}h${p(date.getMinutes())}`;
	const name = safeName(title ? `${base} ${title}` : `${base} Meeting`);
	const folder = normalizePath(s.notesFolder || "/");
	if (folder !== "/" && !app.vault.getAbstractFileByPath(folder)) await app.vault.createFolder(folder);
	let filePath = normalizePath(`${folder}/${name}.md`);
	for (let i = 2; app.vault.getAbstractFileByPath(filePath); i++) filePath = normalizePath(`${folder}/${name} ${i}.md`);
	const heading = title ? `# ${title}\n\n` : "";
	return app.vault.create(filePath, `${frontmatter(res, date)}\n${heading}${body}`);
}
