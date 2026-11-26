import { ChildProcess, execFile, spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";

export type RecordingMode = "online" | "in-person";

export interface Recording {
	dir: string;
	mic: string;
	system?: string;
	mode: RecordingMode;
	startedAt: Date;
	durationSec: number;
}

export interface AudioSource {
	name: string;
	kind: "mic" | "monitor";
}

/** Names of virtual loopback devices used to capture system audio on macOS. */
const LOOPBACK = /blackhole|loopback|soundflower/i;

/** Mac capture goes through AVFoundation; Linux through PipeWire/PulseAudio aliases. */
const isMac = process.platform === "darwin";

interface AvfDevice {
	index: number;
	name: string;
}

/** Lists selectable audio sources (mics and, on Linux/macOS, loopback devices). */
export async function listSources(ffmpegPath = "ffmpeg"): Promise<AudioSource[]> {
	if (isMac) {
		const devices = await listAvfDevices(ffmpegPath);
		return devices.map((d) => ({ name: d.name, kind: LOOPBACK.test(d.name) ? "monitor" : "mic" }));
	}
	return listPulseSources();
}

/** The part of a `pw-dump` entry we read. */
interface PwObject {
	type?: string;
	info?: { props?: { "node.name"?: string; "media.class"?: string } };
}

/** Lists PulseAudio-compatible source names (mics + "<sink>.monitor") from PipeWire. */
function listPulseSources(): Promise<AudioSource[]> {
	return new Promise((resolve, reject) =>
		execFile("pw-dump", { maxBuffer: 64 * 1024 * 1024 }, (err, stdout) => {
			if (err) return reject(new Error(`pw-dump failed: ${err.message}`));
			const out: AudioSource[] = [];
			for (const o of JSON.parse(stdout) as PwObject[]) {
				const props = o?.info?.props;
				const name = props?.["node.name"];
				if (o?.type !== "PipeWire:Interface:Node" || !name) continue;
				if (props["media.class"] === "Audio/Source") out.push({ name, kind: "mic" });
				if (props["media.class"] === "Audio/Sink") out.push({ name: `${name}.monitor`, kind: "monitor" });
			}
			resolve(out);
		}));
}

/** Parses `ffmpeg -f avfoundation -list_devices true -i ""` (ffmpeg exits non-zero but prints the list on stderr). */
function listAvfDevices(ffmpegPath: string): Promise<AvfDevice[]> {
	return new Promise((resolve, reject) =>
		execFile(ffmpegPath, ["-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
			{ maxBuffer: 8 * 1024 * 1024 }, (err, stdout, stderr) => {
				const text = `${stdout}\n${stderr}`;
				const devices: AvfDevice[] = [];
				let inAudio = false;
				for (const line of text.split(/\r?\n/)) {
					if (/AVFoundation audio devices:/.test(line)) { inAudio = true; continue; }
					if (/AVFoundation video devices:/.test(line)) { inAudio = false; continue; }
					const m = inAudio && line.match(/\[(\d+)\]\s+(.+?)\s*$/);
					if (m) devices.push({ index: Number(m[1]), name: m[2] });
				}
				if (devices.length) resolve(devices);
				else reject(new Error(err?.message ?? "ffmpeg could not list AVFoundation audio devices"));
			}));
}

function pickMic(devices: AvfDevice[], wanted: string): AvfDevice | undefined {
	if (wanted) {
		const hit = devices.find((d) => d.name === wanted) ?? devices.find((d) => d.name.includes(wanted));
		if (hit) return hit;
	}
	return devices.find((d) => /microphone|\bmic\b/i.test(d.name) && !LOOPBACK.test(d.name))
		?? devices.find((d) => !LOOPBACK.test(d.name))
		?? devices[0];
}

function pickLoopback(devices: AvfDevice[], wanted: string): AvfDevice | undefined {
	if (wanted) {
		const hit = devices.find((d) => d.name === wanted) ?? devices.find((d) => d.name.includes(wanted));
		if (hit) return hit;
	}
	return devices.find((d) => LOOPBACK.test(d.name));
}

function permissionHint(stderr: string): string {
	if (/not authorized|operation not permitted|permission|denied|input\/output error/i.test(stderr)) {
		return "\nIf macOS blocked audio access, allow Obsidian in System Settings → Privacy & Security → Microphone (and Screen & System Audio Recording for system audio).";
	}
	return "";
}

function stamp(d: Date): string {
	const p = (n: number) => String(n).padStart(2, "0");
	return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(d.getHours())}-${p(d.getMinutes())}-${p(d.getSeconds())}`;
}

interface Capture {
	file: string;
	args: string[];
	label: string;
}

export class Recorder {
	private procs: ChildProcess[] = [];
	private current: Omit<Recording, "durationSec"> | null = null;
	private stderrTail = "";

	constructor(private ffmpegPath: () => string) {}

	get isRecording(): boolean {
		return this.procs.length > 0;
	}

	get mode(): RecordingMode | null {
		return this.current?.mode ?? null;
	}

	get startedAt(): Date | null {
		return this.current?.startedAt ?? null;
	}

	async start(mode: RecordingMode, audioDir: string, micSource: string, monitorSource: string): Promise<void> {
		if (this.isRecording) throw new Error("Already recording");

		const startedAt = new Date();
		const dir = path.join(audioDir, stamp(startedAt));
		fs.mkdirSync(dir, { recursive: true });
		const micFile = path.join(dir, "mic.wav");
		const sysFile = path.join(dir, "system.wav");
		const out = ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"];

		const captures: Capture[] = [];
		if (isMac) {
			const devices = await listAvfDevices(this.ffmpegPath());
			const mic = pickMic(devices, micSource);
			if (!mic) throw new Error("No microphone found. Grant Obsidian microphone access in System Settings → Privacy & Security.");
			captures.push({
				label: "microphone",
				file: micFile,
				args: ["-hide_banner", "-loglevel", "error", "-y", "-f", "avfoundation", "-i", `:${mic.index}`, ...out, micFile],
			});
			if (mode === "online") {
				const loop = pickLoopback(devices, monitorSource);
				if (!loop) {
					throw new Error("No system-audio loopback device found. Install BlackHole 2ch (`brew install --cask blackhole-2ch`), create a Multi-Output Device with your speakers + BlackHole, then select it in settings. You can still record in-person meetings (microphone only).");
				}
				captures.push({
					label: "system audio",
					file: sysFile,
					args: ["-hide_banner", "-loglevel", "error", "-y", "-f", "avfoundation", "-i", `:${loop.index}`, ...out, sysFile],
				});
			}
		} else {
			// PipeWire/PulseAudio aliases follow the current default devices.
			const mic = micSource || "default";
			const monitor = monitorSource || "@DEFAULT_MONITOR@";
			captures.push({
				label: "microphone",
				file: micFile,
				args: ["-hide_banner", "-loglevel", "error", "-y", "-f", "pulse", "-i", mic, ...out, micFile],
			});
			if (mode === "online") {
				captures.push({
					label: "system audio",
					file: sysFile,
					args: ["-hide_banner", "-loglevel", "error", "-y", "-f", "pulse", "-i", monitor, ...out, sysFile],
				});
			}
		}

		this.stderrTail = "";
		const procs = captures.map((c) => {
			const proc = spawn(this.ffmpegPath(), c.args, { stdio: ["pipe", "ignore", "pipe"] });
			proc.stderr?.on("data", (d: Buffer) => {
				this.stderrTail = (this.stderrTail + `[${c.label}] ${d.toString()}`).slice(-2000);
			});
			return proc;
		});

		// Fail fast if ffmpeg dies immediately (bad device, missing binary, denied permission…)
		try {
			await this.settle(procs, isMac ? 1500 : 800);
		} catch (e) {
			for (const p of procs) { try { p.kill("SIGKILL"); } catch { /* gone */ } }
			throw new Error(`${(e as Error).message}${permissionHint(this.stderrTail)}`);
		}

		this.procs = procs;
		this.current = { dir, mic: micFile, system: mode === "online" ? sysFile : undefined, mode, startedAt };
	}

	private settle(procs: ChildProcess[], ms: number): Promise<void> {
		return new Promise((resolve, reject) => {
			let settled = false;
			const done = (err?: Error) => {
				if (settled) return;
				settled = true;
				window.clearTimeout(timer);
				if (err) reject(err); else resolve();
			};
			const timer = window.setTimeout(() => done(), ms);
			for (const p of procs) {
				p.once("error", (e) => done(e));
				p.once("exit", (code) => done(new Error(`ffmpeg exited (${code}): ${this.stderrTail}`)));
			}
		});
	}

	/** Stops ffmpeg cleanly (writes 'q') so the WAV headers are finalised. */
	async stop(): Promise<Recording> {
		const procs = this.procs;
		const cur = this.current;
		if (!procs.length || !cur) throw new Error("Not recording");
		await Promise.all(procs.map((p) => new Promise<void>((resolve) => {
			if (p.exitCode !== null || p.signalCode !== null) return resolve();
			const kill = window.setTimeout(() => { try { p.kill("SIGINT"); } catch { /* gone */ } }, 5000);
			const force = window.setTimeout(() => { try { p.kill("SIGKILL"); } catch { /* gone */ } resolve(); }, 8000);
			p.once("exit", () => { window.clearTimeout(kill); window.clearTimeout(force); resolve(); });
			try { p.stdin?.write("q"); p.stdin?.end(); } catch { try { p.kill("SIGINT"); } catch { /* gone */ } }
		})));
		this.procs = [];
		this.current = null;
		return { ...cur, durationSec: (Date.now() - cur.startedAt.getTime()) / 1000 };
	}
}
