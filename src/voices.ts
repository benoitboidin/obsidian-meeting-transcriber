import * as fs from "fs";
import * as path from "path";
import type { SpeakerVoice } from "./backend";

/** The embedding model the stored voices come from; voices from another model are not comparable. */
export const VOICE_MODEL = "wespeaker-resnet34-lm";

/** Cosine similarity above which a meeting speaker is taken to be a known person. */
export const DEFAULT_MATCH_THRESHOLD = 0.5;

/** A speaker needs this much speech for a reliable voice match. */
const MIN_SECONDS = 4;

/** Cap on the speech a stored voice is weighted with, so it keeps adapting to new meetings. */
const MAX_WEIGHT_SECONDS = 600;

export interface Voice {
	name: string;
	embedding: number[];
	seconds: number;
	meetings: number;
	updated: string;
}

interface VoiceFile {
	version: 1;
	model: string;
	voices: Voice[];
}

/** Voice embeddings of one meeting, keyed by the label written in the note. */
interface MeetingFile {
	version: 1;
	model: string;
	speakers: Record<string, SpeakerVoice>;
}

function normalize(v: number[]): number[] {
	const n = Math.hypot(...v) || 1;
	return v.map((x) => x / n);
}

function cosine(a: number[], b: number[]): number {
	let dot = 0, na = 0, nb = 0;
	for (let i = 0; i < a.length; i++) {
		dot += a[i] * b[i];
		na += a[i] * a[i];
		nb += b[i] * b[i];
	}
	return dot / (Math.sqrt(na * nb) || 1);
}

function readJson<T>(p: string): T | null {
	try {
		return JSON.parse(fs.readFileSync(p, "utf8")) as T;
	} catch {
		return null;
	}
}

function writeJson(p: string, data: unknown): void {
	fs.mkdirSync(path.dirname(p), { recursive: true });
	const tmp = `${p}.tmp`;
	fs.writeFileSync(tmp, JSON.stringify(data));
	fs.renameSync(tmp, p);
}

/**
 * Known voices, and the voices of each transcribed meeting (so renaming a speaker
 * later can teach their voice). Stored outside the vault: voice prints are biometric data.
 */
export class VoiceStore {
	constructor(private dir: () => string) {}

	private get voicesPath(): string {
		return path.join(this.dir(), "voices.json");
	}

	private meetingPath(id: string): string {
		return path.join(this.dir(), "meetings", `${id.replace(/[^\w-]/g, "")}.json`);
	}

	list(): Voice[] {
		const f = readJson<VoiceFile>(this.voicesPath);
		return f && f.model === VOICE_MODEL ? f.voices : [];
	}

	private save(voices: Voice[]): void {
		writeJson(this.voicesPath, { version: 1, model: VOICE_MODEL, voices } satisfies VoiceFile);
	}

	forget(name: string): void {
		this.save(this.list().filter((v) => v.name !== name));
	}

	/** Adds a voice sample for `name` (averaged with what is known, weighted by speech duration). */
	learn(name: string, sample: SpeakerVoice): void {
		const voices = this.list();
		const v = voices.find((x) => x.name === name);
		const now = new Date().toISOString();
		if (!v) {
			voices.push({ name, embedding: normalize(sample.embedding), seconds: sample.seconds, meetings: 1, updated: now });
		} else {
			const wOld = Math.min(v.seconds, MAX_WEIGHT_SECONDS);
			const wNew = Math.max(sample.seconds, 1);
			const e = normalize(sample.embedding);
			v.embedding = normalize(v.embedding.map((x, i) => x * wOld + e[i] * wNew));
			v.seconds += sample.seconds;
			v.meetings += 1;
			v.updated = now;
		}
		this.save(voices);
	}

	/**
	 * Names the meeting's speakers that sound like a known person: each person at most once,
	 * best matches first. Returns label → name.
	 */
	match(speakers: Record<string, SpeakerVoice>, threshold: number): Record<string, string> {
		const known = this.list();
		const pairs: { label: string; name: string; sim: number }[] = [];
		for (const [label, sp] of Object.entries(speakers)) {
			if (sp.seconds < MIN_SECONDS) continue;
			for (const v of known) {
				const sim = cosine(sp.embedding, v.embedding);
				if (sim >= threshold) pairs.push({ label, name: v.name, sim });
			}
		}
		pairs.sort((a, b) => b.sim - a.sim);
		const out: Record<string, string> = {};
		const used = new Set<string>();
		for (const p of pairs) {
			if (p.label in out || used.has(p.name)) continue;
			out[p.label] = p.name;
			used.add(p.name);
		}
		return out;
	}

	saveMeeting(id: string, speakers: Record<string, SpeakerVoice>): void {
		writeJson(this.meetingPath(id), { version: 1, model: VOICE_MODEL, speakers } satisfies MeetingFile);
	}

	/**
	 * Speakers of a meeting were renamed in its note: learn their voices under the new
	 * names and keep the meeting's labels in sync. Returns the names whose voice was learned.
	 */
	renamed(id: string, map: [string, string][]): string[] {
		const p = this.meetingPath(id);
		const m = readJson<MeetingFile>(p);
		if (!m || m.model !== VOICE_MODEL) return [];
		const learned: string[] = [];
		for (const [from, to] of map) {
			const sp = m.speakers[from];
			if (!sp) continue;
			this.learn(to, sp);
			learned.push(to);
			delete m.speakers[from];
			m.speakers[to] = sp;
		}
		writeJson(p, m);
		return learned;
	}
}
