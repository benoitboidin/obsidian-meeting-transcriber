import { requestUrl } from "obsidian";
import type { MTSettings } from "./settings";

const LANG_NAMES: Record<string, string> = {
	fr: "French", en: "English", de: "German", es: "Spanish", it: "Italian", nl: "Dutch", pt: "Portuguese",
};

// ~12k tokens per chunk keeps us well inside a 32k context with room for the answer.
const CHUNK_CHARS = 45000;

function languageInstruction(lang?: string): string {
	const name = lang ? LANG_NAMES[lang] : undefined;
	return name ? `Write your answer in ${name}.` : "Write your answer in the same language as the transcript.";
}

const SECTIONS = `Use exactly this Markdown structure (translate the headings into the answer language):
### Summary
A short paragraph, then the key points as bullets, grouped by topic.
### Decisions
Bullets. Write "None" if there are none.
### Action items
- [ ] Owner — task (deadline if mentioned)
Write "None" if there are none.

Only use information present in the transcript. Speaker labels like SPEAKER_00 are unknown people; keep them as-is. Do not add any preamble.`;

async function chat(s: MTSettings, system: string, user: string): Promise<string> {
	const r = await requestUrl({
		url: `${s.ollamaUrl.replace(/\/$/, "")}/api/chat`,
		method: "POST",
		contentType: "application/json",
		body: JSON.stringify({
			model: s.ollamaModel,
			stream: false,
			think: false,
			keep_alive: "2m",
			options: { num_ctx: 32768, temperature: 0.2 },
			messages: [
				{ role: "system", content: system },
				{ role: "user", content: user },
			],
		}),
	});
	const content = (r.json as { message?: { content?: string } } | null)?.message?.content ?? "";
	return content.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}

function chunk(text: string): string[] {
	if (text.length <= CHUNK_CHARS) return [text];
	const lines = text.split("\n");
	const chunks: string[] = [];
	let cur = "";
	for (const l of lines) {
		if (cur.length + l.length > CHUNK_CHARS && cur) { chunks.push(cur); cur = ""; }
		cur += l + "\n";
	}
	if (cur) chunks.push(cur);
	return chunks;
}

/** `transcript` is plain "[hh:mm:ss] Speaker: text" lines. */
export async function summarize(
	s: MTSettings, transcript: string, lang: string | undefined, onProgress?: (msg: string) => void,
): Promise<string> {
	const system = `You are an assistant that writes concise, factual meeting minutes. ${languageInstruction(lang)}`;
	const parts = chunk(transcript);
	if (parts.length === 1) {
		onProgress?.("Summarizing…");
		return chat(s, system, `${SECTIONS}\n\nTranscript:\n${transcript}`);
	}
	const partials: string[] = [];
	for (let i = 0; i < parts.length; i++) {
		onProgress?.(`Summarizing part ${i + 1}/${parts.length}…`);
		partials.push(await chat(s, system,
			`This is part ${i + 1} of ${parts.length} of a meeting transcript. Write detailed notes of this part: key points, decisions, action items with owners.\n\nTranscript part:\n${parts[i]}`));
	}
	// Very long meetings: condense groups of notes until they fit in one final call.
	let notes = partials.map((p, i) => `--- Part ${i + 1} ---\n${p}`);
	for (let round = 1; notes.join("\n\n").length > CHUNK_CHARS && notes.length > 1; round++) {
		const groups: string[][] = [[]];
		for (const n of notes) {
			const g = groups[groups.length - 1];
			if (g.length && (g.join("\n\n").length + n.length > CHUNK_CHARS)) groups.push([n]); else g.push(n);
		}
		const next: string[] = [];
		for (let i = 0; i < groups.length; i++) {
			onProgress?.(`Condensing notes (round ${round}, ${i + 1}/${groups.length})…`);
			next.push(`--- Section ${i + 1} ---\n` + await chat(s, system,
				`Below are notes on consecutive parts of one meeting. Merge them into shorter notes, keeping every decision and action item with its owner.\n\n${groups[i].join("\n\n")}`));
		}
		notes = next;
	}
	onProgress?.("Merging summaries…");
	return chat(s, system,
		`Below are notes on consecutive parts of one meeting. Merge them into final minutes.\n${SECTIONS}\n\n${notes.join("\n\n")}`);
}
