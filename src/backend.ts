import { requestUrl } from "obsidian";
import { ChildProcess, spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";
import type { MTSettings } from "./settings";
// Bundled as text by esbuild so the community-store install (main.js only) is self-contained.
import PIPELINE_PY from "../backend/pipeline.py";
import SERVER_PY from "../backend/server.py";
import PYPROJECT from "../backend/pyproject.toml";

const BACKEND_FILES: Record<string, string> = {
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

export interface TranscriptResult {
	language: string;
	duration: number;
	segments: Segment[];
	warnings: string[];
}

export interface Health {
	ok: boolean;
	cuda: boolean;
	gpu: string | null;
}

export interface JobInput {
	mic: string;
	system?: string;
	me: string;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export class Backend {
	private proc: ChildProcess | null = null;
	private log = "";
	private idleTimer: number | null = null;
	private starting: Promise<Health> | null = null;

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
		if (s.hfToken) env.HF_TOKEN = s.hfToken;
		const cmd = [s.uvPath, "run", "--project", this.backendDir, "python", "server.py", "--port", String(s.backendPort)];
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
					hf_token: s.hfToken || null,
					whisper_model: s.whisperModel,
					min_speakers: s.minSpeakers || null,
					max_speakers: s.maxSpeakers || null,
				}),
			});
			const id = (r.json as { id: string }).id;
			for (;;) {
				await sleep(1500);
				const j = (await requestUrl({ url: `${this.base}/jobs/${id}` })).json;
				if (j.status === "done") return j.result as TranscriptResult;
				if (j.status === "error") throw new Error(j.error);
				onProgress(j.stage, j.progress);
			}
		} finally {
			this.scheduleIdleStop();
		}
	}

	private clearIdle(): void {
		if (this.idleTimer !== null) window.clearTimeout(this.idleTimer);
		this.idleTimer = null;
	}

	private scheduleIdleStop(): void {
		this.clearIdle();
		const min = this.settings().backendIdleMinutes;
		if (min > 0 && this.proc) this.idleTimer = window.setTimeout(() => this.stop(), min * 60 * 1000);
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
