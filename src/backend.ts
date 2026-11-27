import { requestUrl } from "obsidian";
import { ChildProcess, spawn } from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import type { MTSettings } from "./settings";
// Bundled as text by esbuild so the community-store install (main.js only) is self-contained.
import ASR_PY from "../backend/asr.py";
import AUDIO_PY from "../backend/audio.py";
import DIARIZATION_PY from "../backend/diarization.py";
import LIVE_PY from "../backend/live.py";
import PIPELINE_PY from "../backend/pipeline.py";
import SERVER_PY from "../backend/server.py";
import PYPROJECT from "../backend/pyproject.toml";

const BACKEND_FILES: Record<string, string> = {
	"asr.py": ASR_PY,
	"audio.py": AUDIO_PY,
	"diarization.py": DIARIZATION_PY,
	"live.py": LIVE_PY,
	"pipeline.py": PIPELINE_PY,
	"server.py": SERVER_PY,
	"pyproject.toml": PYPROJECT,
};

/** Writes the bundled backend sources into `dir` when missing or outdated. */
function installBackendFiles(dir: string): void {
	fs.mkdirSync(dir, { recursive: true });
	for (const [name, content] of Object.entries(BACKEND_FILES)) {
		const p = path.join(dir, name);
		if (!fs.existsSync(p) || fs.readFileSync(p, "utf8") !== content) fs.writeFileSync(p, content);
	}
}

export interface Segment {
	start: number;
	end: number;
	speaker: string;
	text: string;
}

export interface SpeakerVoice {
	/** Unit-length voice embedding (WeSpeaker ResNet34). */
	embedding: number[];
	/** Seconds of speech the embedding was computed from. */
	seconds: number;
}

export interface TranscriptResult {
	language: string;
	duration: number;
	segments: Segment[];
	/** Voice of each diarized speaker, by label. */
	speakers?: Record<string, SpeakerVoice>;
	warnings: string[];
}

export interface Health {
	ok: boolean;
	cuda: boolean;
	/** Whisper engine: "cuda"/"cpu" = faster-whisper, "mlx" = mlx-whisper on the Apple GPU. */
	asr?: "cuda" | "mlx" | "cpu";
	gpu: string | null;
}

function binDirs(bin: string): string[] {
	return bin.includes("/") || bin.includes("\\") ? [path.dirname(bin)] : [];
}

/** PATH for the backend: GUI apps on macOS start with a minimal PATH that misses Homebrew and ~/.local/bin. */
function backendPath(s: MTSettings): string {
	const extra = [
		...binDirs(s.uvPath),
		...binDirs(s.ffmpegPath),
		path.join(os.homedir(), ".local", "bin"),
		"/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin",
	];
	const current = (process.env.PATH ?? "").split(path.delimiter).filter(Boolean);
	return [...new Set([...extra, ...current])].join(path.delimiter);
}

/** Absolute path for a configured binary (or its first match on `envPath`). */
function resolveExecutable(bin: string, envPath: string): string | null {
	if (bin.includes("/") || bin.includes("\\")) return fs.existsSync(bin) ? bin : null;
	for (const dir of envPath.split(path.delimiter)) {
		if (!dir) continue;
		const p = path.join(dir, bin);
		if (fs.existsSync(p)) return p;
	}
	return null;
}

export interface JobInput {
	mic: string;
	system?: string;
	me: string;
	/** Live session that already transcribed this recording while it was made. */
	liveId?: string;
}

export interface LiveSegment {
	start: number;
	end: number;
	track: "mic" | "system";
	text: string;
}

export interface LiveStatus {
	status: "running" | "finishing" | "done" | "error";
	error: string | null;
	/** Seconds of recorded audio not transcribed yet. */
	lag: number;
	language: string | null;
	segments: LiveSegment[];
}

const sleep = (ms: number) => new Promise((r) => window.setTimeout(r, ms));

interface JobStatus {
	status: "queued" | "running" | "done" | "error";
	stage: string;
	progress: number;
	result?: TranscriptResult;
	error?: string;
}

export class Backend {
	private proc: ChildProcess | null = null;
	private log = "";
	private idleTimer: number | null = null;
	private starting: Promise<Health> | null = null;
	/** Live sessions in progress: the backend must not be stopped for idleness meanwhile. */
	private liveSessions = 0;

	constructor(private settings: () => MTSettings) {}

	private get backendDir(): string {
		return this.settings().backendDir;
	}

	private get base(): string {
		return `http://127.0.0.1:${this.settings().backendPort}`;
	}

	private async health(): Promise<Health | null> {
		try {
			const r = await requestUrl({ url: `${this.base}/health`, throw: false });
			return r.status === 200 ? (r.json as Health) : null;
		} catch {
			return null;
		}
	}

	ensureRunning(): Promise<Health> {
		if (!this.starting) this.starting = this.start().finally(() => { this.starting = null; });
		return this.starting;
	}

	private async start(useSystemdScope = process.platform === "linux"): Promise<Health> {
		this.clearIdle();
		const h = await this.health();
		if (h) return h; // already running (ours or started manually)

		const s = this.settings();
		installBackendFiles(this.backendDir);
		this.log = "";
		const env = { ...process.env } as Record<string, string>;
		env.PATH = backendPath(s);
		const uv = resolveExecutable(s.uvPath, env.PATH);
		if (!uv) {
			throw new Error(`uv was not found at "${s.uvPath}". Install it (macOS: brew install uv) and set its path in settings.`);
		}
		const ffmpeg = resolveExecutable(s.ffmpegPath, env.PATH);
		if (!ffmpeg) {
			throw new Error(`ffmpeg was not found at "${s.ffmpegPath}". Install it (macOS: brew install ffmpeg) and set its path in settings.`);
		}
		env.MT_FFMPEG = ffmpeg;
		const cmd = [uv, "run", "--project", this.backendDir, "python", "server.py", "--port", String(s.backendPort)];
		// On systemd Linux, run the backend in its own scope: if it ever runs out of memory,
		// the OOM killer stops the backend only, not Obsidian.
		const scoped = useSystemdScope && fs.existsSync("/usr/bin/systemd-run");
		const argv = scoped ? ["/usr/bin/systemd-run", "--user", "--scope", "--quiet", "--collect", ...cmd] : cmd;
		const proc = spawn(argv[0], argv.slice(1),
			{ cwd: this.backendDir, env, detached: true, stdio: ["ignore", "pipe", "pipe"] });
		const onData = (d: Buffer) => { this.log = (this.log + d.toString()).slice(-4000); };
		proc.stdout?.on("data", onData);
		proc.stderr?.on("data", onData);
		let exited = null as string | null;
		proc.once("error", (e) => { exited = e.message; });
		proc.once("exit", (code) => { exited = `exit code ${code}`; this.proc = null; });
		this.proc = proc;

		// First launch may install dependencies (uv sync) → allow a long time.
		const deadline = Date.now() + 15 * 60 * 1000;
		while (Date.now() < deadline) {
			if (exited && scoped && this.log.includes("systemd-run")) {
				// No user systemd session available → retry without the scope.
				return this.start(false);
			}
			if (exited) {
				const hint = exited.includes("ENOENT") ? ` — is uv installed? (https://docs.astral.sh/uv/) Set its path in settings.` : "";
				throw new Error(`Backend failed to start (${exited})${hint}\n${this.log.slice(-1500)}`);
			}
			const h2 = await this.health();
			if (h2) return h2;
			await sleep(1000);
		}
		this.stop();
		throw new Error("Backend did not start within 15 minutes");
	}

	async transcribe(input: JobInput, onProgress: (stage: string, frac: number) => void): Promise<TranscriptResult> {
		const s = this.settings();
		await this.ensureRunning();
		this.clearIdle();
		try {
			const r = await requestUrl({
				url: `${this.base}/jobs`,
				method: "POST",
				contentType: "application/json",
				body: JSON.stringify({
					mic: input.mic,
					system: input.system ?? null,
					me: input.me,
					language: s.language || null,
					whisper_model: s.whisperModel,
					min_speakers: s.minSpeakers || null,
					max_speakers: s.maxSpeakers || null,
					live_id: input.liveId ?? null,
				}),
			});
			const id = (r.json as { id: string }).id;
			if (input.liveId) this.endLive();
			for (;;) {
				await sleep(1500);
				const j = (await requestUrl({ url: `${this.base}/jobs/${id}` })).json as JobStatus;
				if (j.status === "done") return j.result as TranscriptResult;
				if (j.status === "error") throw new Error(j.error ?? "transcription failed");
				onProgress(j.stage, j.progress);
			}
		} finally {
			this.scheduleIdleStop();
		}
	}

	/** Starts transcribing a recording while ffmpeg writes it; returns the session id. */
	async startLive(input: { mic: string; system?: string }): Promise<string> {
		const s = this.settings();
		await this.ensureRunning();
		this.clearIdle();
		const r = await requestUrl({
			url: `${this.base}/live`,
			method: "POST",
			contentType: "application/json",
			body: JSON.stringify({
				mic: input.mic,
				system: input.system ?? null,
				language: s.language || null,
				whisper_model: s.whisperModel,
			}),
		});
		this.liveSessions++;
		return (r.json as { id: string }).id;
	}

	/** New live segments since index `since`. */
	async pollLive(id: string, since: number): Promise<LiveStatus> {
		return (await requestUrl({ url: `${this.base}/live/${id}?since=${since}` })).json as LiveStatus;
	}

	async cancelLive(id: string): Promise<void> {
		this.endLive();
		await requestUrl({ url: `${this.base}/live/${id}`, method: "DELETE", throw: false }).catch(() => null);
	}

	private endLive(): void {
		this.liveSessions = Math.max(0, this.liveSessions - 1);
	}

	private clearIdle(): void {
		if (this.idleTimer !== null) window.clearTimeout(this.idleTimer);
		this.idleTimer = null;
	}

	private scheduleIdleStop(): void {
		this.clearIdle();
		const min = this.settings().backendIdleMinutes;
		if (min > 0 && this.proc && this.liveSessions === 0) this.idleTimer = window.setTimeout(() => this.stop(), min * 60 * 1000);
	}

	stop(): void {
		this.clearIdle();
		const p = this.proc;
		this.proc = null;
		if (p?.pid) {
			try { process.kill(-p.pid, "SIGTERM"); } catch { /* already gone */ }
		}
	}
}
