import { type ChildProcess, execFile, spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

/**
 * Embedded Python script that loads LuxTTS once and generates all cues.
 * The LuxTTS Python API auto-transcribes reference audio via Whisper,
 * avoiding the CLI's requirement for a manual --prompt-text argument.
 */
const BATCH_TTS_SCRIPT = `
import json, sys, traceback, torch
import soundfile as sf

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    from zipvoice.luxvoice import LuxTTS
    tts = LuxTTS(device=device)
    enc = tts.encode_prompt(cfg["prompt_wav"], duration=cfg.get("prompt_duration", 8))
    for i, cue in enumerate(cfg["cues"]):
        wav = tts.generate_speech(
            text=cue["text"],
            encode_dict=enc,
            num_steps=8,
            speed=cfg.get("speed", 1.0),
        )
        wav = wav.squeeze(0).detach().cpu()
        peak = float(torch.max(torch.abs(wav)).item()) if wav.numel() > 0 else 0.0
        if peak > 1e-6:
            gain = min(0.85 / peak, 1000.0)
            wav = torch.clamp(wav * gain, -0.98, 0.98)
        sf.write(cue["output_path"], wav.numpy(), 48000)
        duration_ms = int(round(float(wav.shape[-1]) / 48000 * 1000))
        print(json.dumps({
            "type": "progress",
            "current": i + 1,
            "total": len(cfg["cues"]),
            "duration_ms": duration_ms,
            "output_path": cue["output_path"],
        }), flush=True)
    print(json.dumps({"type": "complete"}), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"type": "error", "message": traceback.format_exc()}), flush=True)
        sys.exit(1)
`.trim();

interface BatchTtsCue {
	text: string;
	outputPath: string;
}

interface GeneratedCueAudio {
	outputPath: string;
	durationMs: number;
}

class LuxTtsService {
	private pythonPath = "python3";
	private luxTtsPath: string | null = null;
	private cachedScriptPath: string | null = null;

	setLuxTtsPath(repoPath: string): void {
		this.luxTtsPath = repoPath;
	}

	setPythonPath(pythonPath: string): void {
		this.pythonPath = pythonPath;
	}

	private getEnv(): NodeJS.ProcessEnv {
		return this.luxTtsPath
			? {
					...process.env,
					PYTHONPATH: `${this.luxTtsPath}${process.env.PYTHONPATH ? `:${process.env.PYTHONPATH}` : ""}`,
				}
			: { ...process.env };
	}

	/** Write the batch TTS Python script to disk (cached). */
	private async ensureScript(): Promise<string> {
		if (this.cachedScriptPath) {
			try {
				await fs.access(this.cachedScriptPath);
				return this.cachedScriptPath;
			} catch {
				// File was cleaned up, recreate
			}
		}
		const scriptPath = path.join(os.tmpdir(), "unbound_lux_tts_batch.py");
		await fs.writeFile(scriptPath, BATCH_TTS_SCRIPT, "utf-8");
		this.cachedScriptPath = scriptPath;
		return scriptPath;
	}

	/**
	 * Check if LuxTTS is available by trying to import zipvoice.
	 */
	async checkAvailability(): Promise<{
		available: boolean;
		error?: string;
	}> {
		try {
			await execFileAsync(this.pythonPath, ["-c", "import zipvoice; print('ok')"], {
				timeout: 15000,
				env: this.getEnv(),
			});
			return { available: true };
		} catch (error) {
			return {
				available: false,
				error: error instanceof Error ? error.message : String(error),
			};
		}
	}

	/**
	 * Generate speech for multiple cues using a reference voice WAV.
	 * Loads the LuxTTS model once and processes all cues in a single
	 * Python subprocess. Reference audio is auto-transcribed via Whisper.
	 */
	async batchTextToSpeech(options: {
		promptWavPath: string;
		cues: BatchTtsCue[];
		speed?: number;
		promptDurationSeconds?: number;
		onProgress?: (current: number, total: number) => void;
	}): Promise<GeneratedCueAudio[]> {
		const { promptWavPath, cues, speed, promptDurationSeconds, onProgress } = options;
		if (cues.length === 0) return [];

		const scriptPath = await this.ensureScript();
		const durationsByOutput = new Map<string, number>();

		const configPath = path.join(
			os.tmpdir(),
			`unbound_tts_cfg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}.json`,
		);

		await fs.writeFile(
			configPath,
			JSON.stringify({
				prompt_wav: promptWavPath,
				cues: cues.map((c) => ({ text: c.text, output_path: c.outputPath })),
				speed: speed ?? 1.0,
				prompt_duration: promptDurationSeconds ?? 8,
			}),
		);

		try {
			await new Promise<void>((resolve, reject) => {
				const child: ChildProcess = spawn(this.pythonPath, [scriptPath, configPath], {
					env: this.getEnv(),
					stdio: ["ignore", "pipe", "pipe"],
				});

				let stderr = "";
				let settled = false;

				const settle = (fn: () => void) => {
					if (!settled) {
						settled = true;
						fn();
					}
				};

				child.stdout?.on("data", (data: Buffer) => {
					for (const line of data.toString().split("\n")) {
						if (!line.trim()) continue;
						try {
							const msg = JSON.parse(line);
							if (msg.type === "progress" && onProgress) {
								onProgress(msg.current, msg.total);
								if (
									typeof msg.output_path === "string" &&
									Number.isFinite(msg.duration_ms)
								) {
									durationsByOutput.set(
										msg.output_path,
										Math.max(1, Math.round(Number(msg.duration_ms))),
									);
								}
							} else if (msg.type === "progress") {
								if (
									typeof msg.output_path === "string" &&
									Number.isFinite(msg.duration_ms)
								) {
									durationsByOutput.set(
										msg.output_path,
										Math.max(1, Math.round(Number(msg.duration_ms))),
									);
								}
							} else if (msg.type === "error") {
								settle(() => reject(new Error(msg.message)));
								child.kill();
							}
						} catch {
							// Non-JSON output (model loading logs), ignore
						}
					}
				});

				child.stderr?.on("data", (data: Buffer) => {
					stderr += data.toString();
				});

				child.on("close", (code) => {
					if (code === 0) {
						settle(() => resolve());
					} else {
						settle(() =>
							reject(
								new Error(
									`LuxTTS generation failed (exit ${code}): ${stderr.slice(-1000)}`,
								),
							),
						);
					}
				});

				child.on("error", (err) => {
					settle(() => reject(new Error(`Failed to spawn LuxTTS: ${err.message}`)));
				});

				// Safety timeout: 10 minutes
				const timer = setTimeout(() => {
					child.kill();
					settle(() => reject(new Error("LuxTTS generation timed out (10 min)")));
				}, 600_000);

				child.on("close", () => clearTimeout(timer));
			});
		} finally {
			await fs.unlink(configPath).catch(() => {
				// Ignore cleanup errors
			});
		}

		return cues.map((cue) => ({
			outputPath: cue.outputPath,
			durationMs: durationsByOutput.get(cue.outputPath) ?? 0,
		}));
	}
}

// Singleton instance
export const luxTts = new LuxTtsService();
