import { execFile } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { app, type IpcMainInvokeEvent, ipcMain, safeStorage } from "electron";
import { ethers } from "ethers";
import { type ZGChatMessage, zgCompute } from "../../services/0g-compute";
import { lipSync } from "../../services/lip-sync";
import { luxTts } from "../../services/lux-tts";
import { extractCaptionAudioSource } from "../captions/generate";
import { AI_SETTINGS_FILE, S3FD_MODEL_PATH, WAV2LIP_GAN_MODEL_PATH } from "../constants";
import { getFfmpegBinaryPath } from "../ffmpeg/binary";
import { buildAtempoFilters } from "../ffmpeg/filters";
import {
	deleteLipSyncModels,
	downloadLipSyncModels,
	getLipSyncModelStatus,
} from "../lipsync/models";
import { normalizeVideoSourcePath } from "../utils";

const execFileAsync = promisify(execFile);

interface AiSettingsOnDisk {
	encryptedPrivateKey?: string; // base64-encoded safeStorage ciphertext
	privateKey?: string; // legacy plaintext — migrated on next save
	network: "testnet" | "mainnet";
	selectedChatProvider?: string;
	selectedSttProvider?: string;
}

interface AiSettings {
	privateKey: string;
	network: "testnet" | "mainnet";
	selectedChatProvider?: string;
	selectedSttProvider?: string;
}

function detectDubbedAudioExtension(buffer: Uint8Array): ".mp3" | ".wav" | ".m4a" {
	if (
		buffer.length >= 12 &&
		buffer[0] === 0x52 &&
		buffer[1] === 0x49 &&
		buffer[2] === 0x46 &&
		buffer[3] === 0x46 &&
		buffer[8] === 0x57 &&
		buffer[9] === 0x41 &&
		buffer[10] === 0x56 &&
		buffer[11] === 0x45
	) {
		return ".wav";
	}

	if (
		buffer.length >= 8 &&
		buffer[4] === 0x66 &&
		buffer[5] === 0x74 &&
		buffer[6] === 0x79 &&
		buffer[7] === 0x70
	) {
		return ".m4a";
	}

	return ".mp3";
}

function decryptKey(stored: AiSettingsOnDisk): string {
	if (stored.encryptedPrivateKey) {
		if (!safeStorage.isEncryptionAvailable()) {
			// Encrypted data but no decryption capability — unreadable
			return "";
		}
		const buf = Buffer.from(stored.encryptedPrivateKey, "base64");
		return safeStorage.decryptString(buf);
	}
	// Plaintext field (legacy or environments without encryption)
	return stored.privateKey ?? "";
}

async function loadAiSettings(): Promise<AiSettings | null> {
	try {
		const data = await fs.readFile(AI_SETTINGS_FILE, "utf-8");
		const onDisk = JSON.parse(data) as AiSettingsOnDisk;
		const privateKey = decryptKey(onDisk);
		return {
			privateKey,
			network: onDisk.network,
			selectedChatProvider: onDisk.selectedChatProvider,
			selectedSttProvider: onDisk.selectedSttProvider,
		};
	} catch {
		return null;
	}
}

async function saveAiSettingsToDisk(settings: AiSettings): Promise<void> {
	const dir = path.dirname(AI_SETTINGS_FILE);
	await fs.mkdir(dir, { recursive: true });

	const canEncrypt = safeStorage.isEncryptionAvailable();
	const onDisk: AiSettingsOnDisk = {
		// Use OS encryption when available; fall back to plaintext field otherwise
		encryptedPrivateKey:
			settings.privateKey && canEncrypt
				? safeStorage.encryptString(settings.privateKey).toString("base64")
				: undefined,
		privateKey: settings.privateKey && !canEncrypt ? settings.privateKey : undefined,
		network: settings.network,
		selectedChatProvider: settings.selectedChatProvider,
		selectedSttProvider: settings.selectedSttProvider,
	};
	await fs.writeFile(AI_SETTINGS_FILE, JSON.stringify(onDisk, null, 2), "utf-8");
}

async function clearAiSettingsFromDisk(network: "testnet" | "mainnet" = "mainnet"): Promise<void> {
	const dir = path.dirname(AI_SETTINGS_FILE);
	await fs.mkdir(dir, { recursive: true });
	const onDisk: AiSettingsOnDisk = {
		network,
		selectedChatProvider: undefined,
		selectedSttProvider: undefined,
	};
	await fs.writeFile(AI_SETTINGS_FILE, JSON.stringify(onDisk, null, 2), "utf-8");
}

function getMaterializedDubbedAudioDir() {
	return path.join(app.getPath("temp"), "unbound-dubbed-audio");
}

function getLipSyncOutputDir() {
	return path.join(app.getPath("temp"), "unbound-lipsync-outputs");
}

// Auto-initialization promise so get-ai-settings can await it
let autoInitPromise: Promise<void> | null = null;

export function registerAiHandlers() {
	// Attempt to initialize broker on startup from saved settings
	autoInitPromise = loadAiSettings().then(async (settings) => {
		if (settings?.privateKey) {
			try {
				await zgCompute.initialize(settings.privateKey, settings.network || "mainnet");
			} catch (error) {
				console.error("Failed to initialize 0G broker on startup:", error);
			}
		}
	});

	// Point LuxTTS at the local repo (sibling directory)
	const luxTtsPath = path.resolve(app.getAppPath(), "..", "LuxTTS");
	luxTts.setLuxTtsPath(luxTtsPath);

	ipcMain.handle("get-ai-settings", async () => {
		try {
			// Wait for auto-init to finish so isInitialized reflects reality
			if (autoInitPromise) await autoInitPromise;
			const settings = await loadAiSettings();
			return {
				success: true,
				network: settings?.network || "mainnet",
				selectedChatProvider: settings?.selectedChatProvider || "",
				selectedSttProvider: settings?.selectedSttProvider || "",
				walletAddress: zgCompute.getWalletAddress() || "",
				isInitialized: zgCompute.isInitialized(),
				hasPrivateKey: Boolean(settings?.privateKey),
			};
		} catch (error) {
			return {
				success: false,
				network: "mainnet",
				selectedChatProvider: "",
				selectedSttProvider: "",
				walletAddress: "",
				isInitialized: false,
				hasPrivateKey: false,
				error: String(error),
			};
		}
	});

	ipcMain.handle(
		"save-ai-settings",
		async (
			_,
			settings: {
				network?: "testnet" | "mainnet";
				selectedChatProvider?: string;
				selectedSttProvider?: string;
			},
		) => {
			try {
				// Private key is never accepted via save-ai-settings.
				// It is only persisted through initialize-ai-wallet after validation.
				const existing = (await loadAiSettings()) || {
					privateKey: "",
					network: "mainnet" as const,
				};
				const updated: AiSettings = {
					privateKey: existing.privateKey,
					network: settings.network ?? existing.network,
					selectedChatProvider:
						settings.selectedChatProvider ?? existing.selectedChatProvider,
					selectedSttProvider:
						settings.selectedSttProvider ?? existing.selectedSttProvider,
				};
				await saveAiSettingsToDisk(updated);
				return { success: true };
			} catch (error) {
				return { success: false, error: String(error) };
			}
		},
	);

	ipcMain.handle("generate-ai-wallet", async () => {
		try {
			const wallet = ethers.Wallet.createRandom();
			return { success: true, address: wallet.address, privateKey: wallet.privateKey };
		} catch (error) {
			return { success: false, error: String(error) };
		}
	});

	ipcMain.handle(
		"initialize-ai-wallet",
		async (_, options?: { privateKey?: string; network?: "testnet" | "mainnet" }) => {
			try {
				let privateKey = options?.privateKey;
				const network = options?.network ?? "mainnet";

				if (!privateKey) {
					const settings = await loadAiSettings();
					if (!settings?.privateKey) {
						return { success: false, error: "No private key configured." };
					}
					privateKey = settings.privateKey;
				}

				// Validate key format before doing anything — this throws on invalid hex
				try {
					new ethers.Wallet(privateKey);
				} catch {
					return {
						success: false,
						error: "Invalid private key format.",
					};
				}

				// Initialize broker (connects to network)
				await zgCompute.initialize(privateKey, network);

				// Persist encrypted key + network AFTER success.
				// Clear provider selections when network changes to avoid stale cross-network refs.
				const existing = (await loadAiSettings()) || {
					privateKey: "",
					network: "mainnet" as const,
				};
				const networkChanged = existing.network !== network;
				await saveAiSettingsToDisk({
					privateKey,
					network,
					selectedChatProvider: networkChanged
						? undefined
						: existing.selectedChatProvider,
					selectedSttProvider: networkChanged ? undefined : existing.selectedSttProvider,
				});

				return {
					success: true,
					walletAddress: zgCompute.getWalletAddress(),
					networkChanged,
				};
			} catch (error) {
				return { success: false, error: String(error) };
			}
		},
	);

	ipcMain.handle("logout-ai-wallet", async () => {
		try {
			const existing = await loadAiSettings();
			await clearAiSettingsFromDisk(existing?.network ?? "mainnet");
			zgCompute.reset();
			return {
				success: true,
				network: existing?.network ?? "mainnet",
			};
		} catch (error) {
			return {
				success: false,
				error: String(error),
			};
		}
	});

	ipcMain.handle("get-ai-balance", async () => {
		try {
			const balance = await zgCompute.getBalance();
			return { success: true, ...balance };
		} catch (error) {
			return { success: false, error: String(error) };
		}
	});

	ipcMain.handle("deposit-ai-funds", async (_, options: { amount: number }) => {
		try {
			await zgCompute.depositFunds(options.amount);
			return { success: true };
		} catch (error) {
			return { success: false, error: String(error) };
		}
	});

	ipcMain.handle(
		"transfer-ai-funds",
		async (_, options: { provider: string; amount: number }) => {
			try {
				await zgCompute.transferToProvider(options.provider, options.amount);
				return { success: true };
			} catch (error) {
				return { success: false, error: String(error) };
			}
		},
	);

	ipcMain.handle("list-ai-providers", async () => {
		try {
			const providers = await zgCompute.listProviders();
			console.log(
				`[list-ai-providers] chatbot: ${providers.chatbot.length}, stt: ${providers.stt.length}`,
			);
			return { success: true, ...providers };
		} catch (error) {
			console.error("[list-ai-providers] Error:", error);
			return { success: false, error: String(error), chatbot: [], stt: [] };
		}
	});

	ipcMain.handle("test-ai-connection", async () => {
		try {
			const result = await zgCompute.testConnection();
			return result;
		} catch (error) {
			return { success: false, error: String(error) };
		}
	});

	ipcMain.handle(
		"generate-ai-captions",
		async (
			_,
			options: { videoPath: string; language?: string; provider?: string },
		) => {
			try {
				if (!zgCompute.isInitialized()) {
					return { success: false, error: "Wallet not initialized." };
				}

				const settings = await loadAiSettings();
				const sttProvider = options.provider || settings?.selectedSttProvider;
				if (!sttProvider) {
					return { success: false, error: "No speech-to-text provider selected." };
				}

				const ffmpegPath = getFfmpegBinaryPath();
				const normalizedVideoPath = normalizeVideoSourcePath(options.videoPath);
				if (!normalizedVideoPath) {
					return { success: false, error: "Missing source video path." };
				}

				const tempBase = path.join(
					app.getPath("temp"),
					`unbound-ai-captions-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
				);
				const wavPath = `${tempBase}.wav`;

				try {
					await extractCaptionAudioSource({
						videoPath: normalizedVideoPath,
						ffmpegPath,
						wavPath,
					});

					const result = await zgCompute.speechToText(
						wavPath,
						sttProvider,
						options.language,
					);

					// Convert segments to caption cues
					const cues = (result.segments || []).map((seg, i) => ({
						id: `ai-cue-${i + 1}`,
						startMs: Math.round(seg.start * 1000),
						endMs: Math.round(seg.end * 1000),
						text: seg.text.trim(),
					}));

					// If no segments but we have text, create a single cue
					if (cues.length === 0 && result.text) {
						cues.push({
							id: "ai-cue-1",
							startMs: 0,
							endMs: 30000,
							text: result.text.trim(),
						});
					}

					return {
						success: true,
						cues,
						message: `Generated ${cues.length} caption cue${cues.length === 1 ? "" : "s"} via 0G AI.`,
					};
				} finally {
					// Cleanup temp wav file
					await fs.unlink(wavPath).catch(() => {
						// Ignore cleanup errors
					});
				}
			} catch (error) {
				console.error("Failed to generate AI captions:", error);
				return {
					success: false,
					error: error instanceof Error ? error.message : String(error),
				};
			}
		},
	);

	ipcMain.handle(
		"translate-ai-captions",
		async (
			_,
			options: {
				cues: Array<{ id: string; startMs: number; endMs: number; text: string }>;
				sourceLanguage: string;
				targetLanguage: string;
				provider?: string;
			},
		) => {
			try {
				if (!zgCompute.isInitialized()) {
					return { success: false, error: "Wallet not initialized." };
				}

				const settings = await loadAiSettings();
				const chatProvider = options.provider || settings?.selectedChatProvider;
				if (!chatProvider) {
					return { success: false, error: "No chatbot provider selected." };
				}

				const { cues, sourceLanguage, targetLanguage } = options;
				if (!cues.length) {
					return { success: false, error: "No captions to translate." };
				}

				const BATCH_SIZE = 50;
				const translatedCues: Array<{
					id: string;
					startMs: number;
					endMs: number;
					text: string;
				}> = [];

				for (let i = 0; i < cues.length; i += BATCH_SIZE) {
					const batch = cues.slice(i, i + BATCH_SIZE);
					const indexedLines = batch
						.map((cue, idx) => `[${i + idx}] ${cue.text}`)
						.join("\n");

					const systemPrompt = `You are a professional subtitle translator. Translate from ${sourceLanguage} to ${targetLanguage}. Preserve the [index] prefix on each line. Output ONLY translated lines, one per line, same count as input. Do not add explanations.`;

					const result = await zgCompute.chatCompletion(
						[
							{ role: "system", content: systemPrompt },
							{ role: "user", content: indexedLines },
						],
						chatProvider,
					);

					const responseText = result.choices[0]?.message?.content || "";
					const responseLines = responseText
						.split("\n")
						.map((line) => line.replace(/^\[\d+\]\s*/, "").trim())
						.filter((line) => line.length > 0);

					for (let j = 0; j < batch.length; j++) {
						const sourceCue = batch[j];
						translatedCues.push({
							id: `translated-${targetLanguage}-${i + j + 1}`,
							startMs: sourceCue.startMs,
							endMs: sourceCue.endMs,
							text: responseLines[j] || sourceCue.text,
						});
					}
				}

				return {
					success: true,
					cues: translatedCues,
					message: `Translated ${translatedCues.length} caption${translatedCues.length === 1 ? "" : "s"} to ${targetLanguage}.`,
				};
			} catch (error) {
				console.error("Failed to translate AI captions:", error);
				return {
					success: false,
					error: error instanceof Error ? error.message : String(error),
				};
			}
		},
	);

	ipcMain.handle(
		"generate-ai-metadata",
		async (
			_,
			options: {
				transcript: string;
				type: "title" | "description" | "chapters";
				provider?: string;
			},
		) => {
			try {
				if (!zgCompute.isInitialized()) {
					return { success: false, error: "Wallet not initialized." };
				}

				const settings = await loadAiSettings();
				const chatProvider = options.provider || settings?.selectedChatProvider;
				if (!chatProvider) {
					return { success: false, error: "No chatbot provider selected." };
				}

				const systemPrompt =
					"You are a helpful assistant for video creators. Generate concise, engaging metadata for videos based on their transcript.";

				let userPrompt: string;
				switch (options.type) {
					case "title":
						userPrompt = `Based on this video transcript, generate a catchy, concise title (max 80 characters). Return ONLY the title, no quotes or extra text.\n\nTranscript:\n${options.transcript.slice(0, 3000)}`;
						break;
					case "description":
						userPrompt = `Based on this video transcript, generate a compelling YouTube-style description (2-4 sentences). Return ONLY the description text.\n\nTranscript:\n${options.transcript.slice(0, 3000)}`;
						break;
					case "chapters":
						userPrompt = `Based on this video transcript, generate chapter markers in the format "MM:SS Title". Generate 3-8 chapters. Return ONLY the chapter list, one per line.\n\nTranscript:\n${options.transcript.slice(0, 5000)}`;
						break;
				}

				const result = await zgCompute.chatCompletion(
					[
						{ role: "system", content: systemPrompt },
						{ role: "user", content: userPrompt },
					],
					chatProvider,
				);

				const content = result.choices[0]?.message?.content || "";
				return { success: true, content, type: options.type };
			} catch (error) {
				console.error("Failed to generate AI metadata:", error);
				return {
					success: false,
					error: error instanceof Error ? error.message : String(error),
				};
			}
		},
	);

	// ── LuxTTS (Voice Dubbing — Local) ──────────────────────────────────

	ipcMain.handle("check-lux-tts-available", async () => {
		return luxTts.checkAvailability();
	});

	ipcMain.handle(
		"clone-voice-from-video",
		async (
			_,
			{
				videoPath,
				sampleWindow,
			}: {
				videoPath: string;
				sampleWindow?: { startMs: number; endMs: number };
			},
		) => {
			try {
				const ffmpegPath = getFfmpegBinaryPath();
				const normalizedPath = normalizeVideoSourcePath(videoPath);
				if (!normalizedPath) {
					return { success: false, error: "Missing source video path." };
				}

				const requestedStartMs = Number(sampleWindow?.startMs);
				const requestedEndMs = Number(sampleWindow?.endMs);
				const hasRequestedWindow =
					Number.isFinite(requestedStartMs) &&
					Number.isFinite(requestedEndMs) &&
					requestedEndMs > requestedStartMs;
				const clipStartMs = hasRequestedWindow ? Math.max(0, requestedStartMs) : 0;
				const clipDurationMs = hasRequestedWindow
					? Math.min(10_000, Math.max(1_500, requestedEndMs - requestedStartMs))
					: 10_000;

				// Extract a spoken reference clip when captions provide a window;
				// otherwise fall back to the first 10 seconds of the recording.
				const refDir = path.join(app.getPath("userData"), "voice-references");
				await fs.mkdir(refDir, { recursive: true });
				const refWavPath = path.join(
					refDir,
					`ref-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.wav`,
				);

				await execFileAsync(ffmpegPath, [
					"-y",
					"-ss",
					(clipStartMs / 1000).toFixed(3),
					"-i",
					normalizedPath,
					"-vn",
					"-ac",
					"1",
					"-ar",
					"24000",
					"-t",
					(clipDurationMs / 1000).toFixed(3),
					"-c:a",
					"pcm_s16le",
					refWavPath,
				]);

				return { success: true, refWavPath };
			} catch (error) {
				console.error("Failed to extract voice reference:", error);
				return {
					success: false,
					error: error instanceof Error ? error.message : String(error),
				};
			}
		},
	);

	ipcMain.handle(
		"generate-dubbed-audio",
		async (
			event: IpcMainInvokeEvent,
			{
				cues,
				refWavPath,
				totalDurationMs,
			}: {
				cues: Array<{ startMs: number; endMs: number; text: string }>;
				refWavPath: string;
				totalDurationMs: number;
			},
		) => {
			try {
				if (!cues.length) {
					return { success: false, error: "No cues to generate audio for." };
				}

				// Verify reference audio exists
				try {
					await fs.access(refWavPath);
				} catch {
					return {
						success: false,
						error: "Voice reference file not found. Clone voice first.",
					};
				}

				const ffmpegPath = getFfmpegBinaryPath();
				const tempDir = path.join(
					app.getPath("temp"),
					`unbound-dub-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
				);
				await fs.mkdir(tempDir, { recursive: true });

				const tempFiles = cues.map((_, i) => path.join(tempDir, `cue_${i}.wav`));

				try {
					// Generate all TTS in a single batch (model loaded once,
					// reference audio auto-transcribed via Whisper)
					const generatedCueAudio = await luxTts.batchTextToSpeech({
						promptWavPath: refWavPath,
						cues: cues.map((cue, i) => ({
							text: cue.text,
							outputPath: tempFiles[i],
						})),
						promptDurationSeconds: 8,
						onProgress: (current, total) => {
							event.sender.send("dub-progress", { current, total });
						},
					});
					const generatedDurationByPath = new Map(
						generatedCueAudio.map((cueAudio) => [
							cueAudio.outputPath,
							cueAudio.durationMs,
						]),
					);

					// Fit each TTS segment back into its caption window, then stitch.
					const outputFile = path.join(tempDir, "dubbed_output.mp3");
					const inputs: string[] = [];
					const filterParts: string[] = [];
					const mixInputs: string[] = [];

					for (let i = 0; i < cues.length; i++) {
						inputs.push("-i", tempFiles[i]);
						const delayMs = cues[i].startMs;
						const generatedDurationMs =
							generatedDurationByPath.get(tempFiles[i]) ??
							Math.max(1, Math.round(cues[i].endMs - cues[i].startMs));
						const targetDurationMs = Math.max(
							1,
							Math.round(cues[i].endMs - cues[i].startMs),
						);
						const cueFilters = [
							...buildAtempoFilters(generatedDurationMs / targetDurationMs),
							`apad=whole_dur=${(targetDurationMs / 1000).toFixed(3)}`,
							`atrim=duration=${(targetDurationMs / 1000).toFixed(3)}`,
							"asetpts=PTS-STARTPTS",
							`adelay=${delayMs}|${delayMs}`,
						];
						filterParts.push(`[${i}:a]${cueFilters.join(",")}[a${i}]`);
						mixInputs.push(`[a${i}]`);
					}

					const totalDurationSec = totalDurationMs / 1000;
					const filterComplex = [
						...filterParts,
						`${mixInputs.join("")}amix=inputs=${cues.length}:duration=longest:normalize=0,alimiter=limit=0.95[out]`,
					].join(";");

					await execFileAsync(
						ffmpegPath,
						[
							"-y",
							...inputs,
							"-filter_complex",
							filterComplex,
							"-map",
							"[out]",
							"-t",
							String(totalDurationSec),
							"-c:a",
							"libmp3lame",
							"-b:a",
							"192k",
							outputFile,
						],
						{ maxBuffer: 50 * 1024 * 1024 },
					);

					const audioData = await fs.readFile(outputFile);
					return {
						success: true,
						audioData: audioData.buffer.slice(
							audioData.byteOffset,
							audioData.byteOffset + audioData.byteLength,
						),
					};
				} finally {
					// Cleanup temp directory (not ref audio — that persists)
					await fs.rm(tempDir, { recursive: true, force: true }).catch(() => {
						// Ignore cleanup errors
					});
				}
			} catch (error) {
				console.error("Failed to generate dubbed audio:", error);
				return {
					success: false,
					error: error instanceof Error ? error.message : String(error),
				};
			}
		},
	);

	ipcMain.handle(
		"materialize-dubbed-audio",
		async (_, { audioData }: { audioData: ArrayBuffer }) => {
			try {
				const outputDir = getMaterializedDubbedAudioDir();
				await fs.mkdir(outputDir, { recursive: true });
				const uid = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
				const audioBuffer = Buffer.from(audioData);
				const inputExtension = detectDubbedAudioExtension(audioBuffer);
				const inputPath = path.join(outputDir, `dub-src-${uid}${inputExtension}`);
				const outputPath = path.join(outputDir, `dub-${uid}.wav`);
				await fs.writeFile(inputPath, audioBuffer);
				const ffmpegPath = getFfmpegBinaryPath();
				await execFileAsync(ffmpegPath, [
					"-y",
					"-i",
					inputPath,
					"-ar",
					"48000",
					"-ac",
					"1",
					"-c:a",
					"pcm_s16le",
					outputPath,
				]);
				await fs.rm(inputPath, { force: true });
				return { success: true, path: outputPath };
			} catch (error) {
				console.error("Failed to materialize dubbed audio:", error);
				return {
					success: false,
					error: error instanceof Error ? error.message : String(error),
				};
			}
		},
	);

	ipcMain.handle(
		"cleanup-materialized-dubbed-audio",
		async (_, { filePath }: { filePath: string }) => {
			try {
				const outputDir = path.resolve(getMaterializedDubbedAudioDir());
				const resolvedPath = path.resolve(filePath);
				if (!resolvedPath.startsWith(`${outputDir}${path.sep}`)) {
					return {
						success: false,
						error: "Refusing to delete dubbed audio outside the temp directory.",
					};
				}

				await fs.rm(resolvedPath, { force: true });
				return { success: true };
			} catch (error) {
				console.error("Failed to clean up materialized dubbed audio:", error);
				return {
					success: false,
					error: error instanceof Error ? error.message : String(error),
				};
			}
		},
	);

	// ── Lip Sync (Wav2Lip) ─────────────────────────────────────────────

	ipcMain.handle("check-lip-sync-available", async () => {
		return lipSync.checkAvailability();
	});

	ipcMain.handle("get-lip-sync-model-status", async () => {
		return getLipSyncModelStatus();
	});

	ipcMain.handle("download-lip-sync-models", async (event) => {
		try {
			await downloadLipSyncModels(event.sender);
			return { success: true };
		} catch (error) {
			return {
				success: false,
				error: error instanceof Error ? error.message : String(error),
			};
		}
	});

	ipcMain.handle("delete-lip-sync-models", async () => {
		try {
			await deleteLipSyncModels();
			return { success: true };
		} catch (error) {
			return {
				success: false,
				error: error instanceof Error ? error.message : String(error),
			};
		}
	});

	ipcMain.handle(
		"generate-lip-sync-video",
		async (
			event,
			{
				videoPath: rawVideoPath,
				dubbedAudioData,
			}: {
				videoPath: string;
				dubbedAudioData: ArrayBuffer;
			},
		) => {
			try {
				const videoPath = normalizeVideoSourcePath(rawVideoPath);
				if (!videoPath) {
					return {
						success: false,
						error: "Video path is required.",
					};
				}

				// Materialize dubbed audio ArrayBuffer → temp WAV via FFmpeg
				const outputDir = getLipSyncOutputDir();
				await fs.mkdir(outputDir, { recursive: true });
				const uid = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
				const tempAudioPath = path.join(outputDir, `lipsync-audio-${uid}.wav`);
				const tempRawAudioPath = path.join(outputDir, `lipsync-raw-${uid}.bin`);
				const rawOutputPath = path.join(outputDir, `lipsync-raw-${uid}.avi`);
				const finalOutputPath = path.join(outputDir, `lipsync-${uid}.mp4`);

				// Write raw audio data to disk
				await fs.writeFile(tempRawAudioPath, Buffer.from(dubbedAudioData));

				// Convert to WAV (16kHz mono) via FFmpeg
				const ffmpeg = getFfmpegBinaryPath();
				await execFileAsync(ffmpeg, [
					"-y",
					"-i",
					tempRawAudioPath,
					"-ar",
					"16000",
					"-ac",
					"1",
					"-f",
					"wav",
					tempAudioPath,
				]);
				await fs.rm(tempRawAudioPath, { force: true });

				// Run Wav2Lip
				await lipSync.runLipSync({
					videoPath,
					audioPath: tempAudioPath,
					outputPath: rawOutputPath,
					checkpointPath: WAV2LIP_GAN_MODEL_PATH,
					faceDetPath: S3FD_MODEL_PATH,
					onProgress: (progress) => {
						event.sender.send("lip-sync-progress", progress);
					},
				});

				// Re-encode via FFmpeg (libx264, crf 18) + mux with dubbed audio
				await execFileAsync(ffmpeg, [
					"-y",
					"-i",
					rawOutputPath,
					"-i",
					tempAudioPath,
					"-c:v",
					"libx264",
					"-crf",
					"18",
					"-preset",
					"medium",
					"-c:a",
					"aac",
					"-b:a",
					"192k",
					"-shortest",
					finalOutputPath,
				]);

				// Cleanup intermediates
				await fs.rm(tempAudioPath, { force: true }).catch(() => undefined);
				await fs.rm(rawOutputPath, { force: true }).catch(() => undefined);

				return { success: true, outputPath: finalOutputPath };
			} catch (error) {
				console.error("Failed to generate lip-sync video:", error);
				return {
					success: false,
					error: error instanceof Error ? error.message : String(error),
				};
			}
		},
	);

	ipcMain.handle("cleanup-lip-sync-video", async (_, { filePath }: { filePath: string }) => {
		try {
			const outputDir = path.resolve(getLipSyncOutputDir());
			const resolvedPath = path.resolve(filePath);
			if (!resolvedPath.startsWith(`${outputDir}${path.sep}`)) {
				return {
					success: false,
					error: "Refusing to delete file outside the lip-sync output directory.",
				};
			}
			await fs.rm(resolvedPath, { force: true });
			return { success: true };
		} catch (error) {
			console.error("Failed to clean up lip-sync video:", error);
			return {
				success: false,
				error: error instanceof Error ? error.message : String(error),
			};
		}
	});

	ipcMain.handle(
		"ai-chat",
		async (event: IpcMainInvokeEvent, options: { messages: ZGChatMessage[] }) => {
			try {
				if (!zgCompute.isInitialized()) {
					return { success: false, error: "Wallet not initialized." };
				}

				const settings = await loadAiSettings();
				const chatProvider = settings?.selectedChatProvider;
				if (!chatProvider) {
					return { success: false, error: "No chatbot provider selected." };
				}

				let fullContent = "";
				for await (const chunk of zgCompute.chatCompletionStream(
					options.messages,
					chatProvider,
				)) {
					const delta = chunk.choices[0]?.delta?.content || "";
					if (delta) {
						fullContent += delta;
						event.sender.send("ai-chat-stream-chunk", { content: delta });
					}
				}
				event.sender.send("ai-chat-stream-done", { content: fullContent });
				return { success: true, content: fullContent };
			} catch (error) {
				console.error("AI chat error:", error);
				const errorMsg = error instanceof Error ? error.message : String(error);
				event.sender.send("ai-chat-stream-done", { content: "", error: errorMsg });
				return { success: false, error: errorMsg };
			}
		},
	);
}
