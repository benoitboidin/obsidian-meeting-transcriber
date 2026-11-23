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

/** Lists PulseAudio-compatible source names (mics + "<sink>.monitor") from PipeWire. */
export function listSources(): Promise<string[]> {
	return new Promise((resolve, reject) =>
		execFile("pw-dump", { maxBuffer: 64 * 1024 * 1024 }, (err, stdout) => {
			if (err) return reject(err);
			const out: string[] = [];
			for (const o of JSON.parse(stdout)) {
				const props = o?.info?.props;
				if (o?.type !== "PipeWire:Interface:Node" || !props?.["node.name"]) continue;
				if (props["media.class"] === "Audio/Source") out.push(props["node.name"]);
				if (props["media.class"] === "Audio/Sink") out.push(`${props["node.name"]}.monitor`);
			}
			resolve(out);
		}));
}

function stamp(d: Date): string {
	const p = (n: number) => String(n).padStart(2, "0");
	return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(d.getHours())}-${p(d.getMinutes())}-${p(d.getSeconds())}`;
}

export class Recorder {
	private proc: ChildProcess | null = null;
	private current: Omit<Recording, "durationSec"> | null = null;
	private stderrTail = "";

	constructor(private ffmpegPath: () => string) {}

	get isRecording(): boolean {
		return this.proc !== null;
	}

	get mode(): RecordingMode | null {
		return this.current?.mode ?? null;
	}

	get startedAt(): Date | null {
		return this.current?.startedAt ?? null;
	}

	async start(mode: RecordingMode, audioDir: string, micSource: string, monitorSource: string): Promise<void> {
		if (this.proc) throw new Error("Already recording");
		// PipeWire/PulseAudio aliases follow the current default devices.
		const mic = micSource || "default";
		const monitor = monitorSource || "@DEFAULT_MONITOR@";

		const startedAt = new Date();
		const dir = path.join(audioDir, stamp(startedAt));
		fs.mkdirSync(dir, { recursive: true });
		const micFile = path.join(dir, "mic.wav");
		const sysFile = path.join(dir, "system.wav");

		const out = ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"];
		const args = ["-hide_banner", "-loglevel", "error", "-y", "-f", "pulse", "-i", mic];
		if (mode === "online") args.push("-f", "pulse", "-i", monitor);
		args.push("-map", "0:a", ...out, micFile);
		if (mode === "online") args.push("-map", "1:a", ...out, sysFile);

		const proc = spawn(this.ffmpegPath(), args, { stdio: ["pipe", "ignore", "pipe"] });
		this.stderrTail = "";
		proc.stderr?.on("data", (d) => { this.stderrTail = (this.stderrTail + d.toString()).slice(-2000); });

		// Fail fast if ffmpeg dies immediately (bad device, missing binary…)
		await new Promise<void>((resolve, reject) => {
			const t = setTimeout(() => { proc.removeAllListeners("exit"); resolve(); }, 800);
			proc.once("error", (e) => { clearTimeout(t); reject(e); });
			proc.once("exit", (code) => { clearTimeout(t); reject(new Error(`ffmpeg exited (${code}): ${this.stderrTail}`)); });
		});

		this.proc = proc;
		this.current = { dir, mic: micFile, system: mode === "online" ? sysFile : undefined, mode, startedAt };
	}

	/** Stops ffmpeg cleanly (writes 'q') so the WAV headers are finalised. */
	async stop(): Promise<Recording> {
		const proc = this.proc;
		const cur = this.current;
		if (!proc || !cur) throw new Error("Not recording");
		await new Promise<void>((resolve) => {
			const kill = setTimeout(() => proc.kill("SIGINT"), 5000);
			proc.once("exit", () => { clearTimeout(kill); resolve(); });
			try { proc.stdin?.write("q"); proc.stdin?.end(); } catch { proc.kill("SIGINT"); }
		});
		this.proc = null;
		this.current = null;
		return { ...cur, durationSec: (Date.now() - cur.startedAt.getTime()) / 1000 };
	}
}
