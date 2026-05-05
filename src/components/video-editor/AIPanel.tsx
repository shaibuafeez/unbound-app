import {
	ArrowLineDown,
	Check,
	CircleNotch,
	Copy,
	GearSix,
	Microphone,
	Sparkle,
	SpeakerHigh,
	Translate,
	Trash,
} from "@phosphor-icons/react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import {
	Accordion,
	AccordionContent,
	AccordionItem,
	AccordionTrigger,
} from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import {
	CAPTION_LANGUAGE_LABELS,
	type CaptionCue,
	type CaptionTrack,
	type DubbedAudio,
	type LipSyncResult,
} from "./types";

interface AIPanelProps {
	videoPath: string | null;
	transcript: string;
	captionTracks: CaptionTrack[];
	activeCaptionTrackId: string | null;
	totalDurationMs: number;
	refWavPath: string | null;
	dubbedAudio: DubbedAudio | null;
	onCaptionsGenerated?: (
		cues: Array<{ id: string; startMs: number; endMs: number; text: string }>,
		language?: string,
	) => void;
	onTranslationGenerated?: (track: CaptionTrack) => void;
	onActiveTrackChange?: (trackId: string) => void;
	onVoiceCloned?: (refWavPath: string) => void;
	onDubbedAudioGenerated?: (dubbed: DubbedAudio) => void;
	lipSyncVideoPath: string | null;
	onLipSyncGenerated?: (result: LipSyncResult) => void;
}

const MAX_VOICE_REFERENCE_MS = 8_000;
const MAX_VOICE_REFERENCE_GAP_MS = 1_000;
const VOICE_REFERENCE_PADDING_MS = 250;

function selectVoiceReferenceWindow(cues: CaptionCue[]): { startMs: number; endMs: number } | null {
	const candidates = cues
		.filter(
			(cue) =>
				cue.text.trim().length > 0 &&
				Number.isFinite(cue.startMs) &&
				Number.isFinite(cue.endMs) &&
				cue.endMs > cue.startMs,
		)
		.sort((left, right) => left.startMs - right.startMs);

	if (candidates.length === 0) {
		return null;
	}

	let best: {
		startMs: number;
		endMs: number;
		score: number;
	} | null = null;

	for (let startIndex = 0; startIndex < candidates.length; startIndex++) {
		let windowStartMs = candidates[startIndex].startMs;
		let windowEndMs = candidates[startIndex].endMs;
		let speechDurationMs = candidates[startIndex].endMs - candidates[startIndex].startMs;
		let charCount = candidates[startIndex].text.trim().length;
		let previousEndMs = windowEndMs;

		for (let endIndex = startIndex + 1; endIndex < candidates.length; endIndex++) {
			const nextCue = candidates[endIndex];
			if (nextCue.startMs - previousEndMs > MAX_VOICE_REFERENCE_GAP_MS) {
				break;
			}
			if (nextCue.endMs - windowStartMs > MAX_VOICE_REFERENCE_MS) {
				break;
			}

			windowEndMs = nextCue.endMs;
			previousEndMs = nextCue.endMs;
			speechDurationMs += nextCue.endMs - nextCue.startMs;
			charCount += nextCue.text.trim().length;
		}

		const score = speechDurationMs + charCount * 20;
		if (!best || score > best.score) {
			best = {
				startMs: windowStartMs,
				endMs: windowEndMs,
				score,
			};
		}
	}

	if (!best) {
		return null;
	}

	const paddedStartMs = Math.max(0, best.startMs - VOICE_REFERENCE_PADDING_MS);
	const paddedEndMs = Math.min(
		best.endMs + VOICE_REFERENCE_PADDING_MS,
		paddedStartMs + MAX_VOICE_REFERENCE_MS,
	);

	return paddedEndMs > paddedStartMs ? { startMs: paddedStartMs, endMs: paddedEndMs } : null;
}

function buildDubGenerationCues(
	cues: CaptionCue[],
): Array<{ startMs: number; endMs: number; text: string }> {
	return cues
		.filter(
			(cue) =>
				Number.isFinite(cue.startMs) &&
				Number.isFinite(cue.endMs) &&
				cue.endMs > cue.startMs &&
				cue.text.trim().length > 0,
		)
		.map((cue) => ({
			startMs: Math.max(0, Math.round(cue.startMs)),
			endMs: Math.max(Math.round(cue.startMs) + 1, Math.round(cue.endMs)),
			text: cue.text.replace(/\s+/g, " ").trim(),
		}))
		.sort((left, right) => left.startMs - right.startMs);
}

function handleCheckoutUrl(checkoutUrl?: string) {
	if (checkoutUrl) {
		toast.error("Credits exhausted.", {
			action: {
				label: "Buy credits",
				onClick: () => window.electronAPI.openExternalUrl(checkoutUrl),
			},
		});
	}
}

export function AIPanel({
	videoPath,
	transcript,
	captionTracks,
	activeCaptionTrackId,
	totalDurationMs,
	refWavPath,
	dubbedAudio,
	onCaptionsGenerated,
	onTranslationGenerated,
	onActiveTrackChange,
	onVoiceCloned,
	onDubbedAudioGenerated,
	lipSyncVideoPath,
	onLipSyncGenerated,
}: AIPanelProps) {
	// Huru settings state
	const [apiKeyInput, setApiKeyInput] = useState("");
	const [emailInput, setEmailInput] = useState("");
	const [isConfigured, setIsConfigured] = useState(false);
	const [consumerEmail, setConsumerEmail] = useState("");
	const [setupLoading, setSetupLoading] = useState(false);
	const [setupError, setSetupError] = useState("");
	const [logoutLoading, setLogoutLoading] = useState(false);

	// Feature state
	const [captionLoading, setCaptionLoading] = useState(false);
	const [captionError, setCaptionError] = useState("");
	const [metadataLoading, setMetadataLoading] = useState<string | null>(null);
	const [metadataError, setMetadataError] = useState("");
	const [generatedTitle, setGeneratedTitle] = useState("");
	const [generatedDescription, setGeneratedDescription] = useState("");
	const [generatedChapters, setGeneratedChapters] = useState("");

	// Caption language / translation state
	const [captionLanguage, setCaptionLanguage] = useState("auto");
	const [targetLanguage, setTargetLanguage] = useState("es");
	const [translationLoading, setTranslationLoading] = useState(false);
	const [translationError, setTranslationError] = useState("");

	// LuxTTS / Dubbing state
	const [luxTtsAvailable, setLuxTtsAvailable] = useState<boolean | null>(null);
	const [voiceCloningLoading, setVoiceCloningLoading] = useState(false);
	const [dubbingLoading, setDubbingLoading] = useState(false);
	const [dubProgress, setDubProgress] = useState<{ current: number; total: number } | null>(null);

	// Lip Sync state
	const [lipSyncAvailable, setLipSyncAvailable] = useState<boolean | null>(null);
	const [lipSyncModelStatus, setLipSyncModelStatus] = useState<{
		wav2lipExists: boolean;
		s3fdExists: boolean;
	} | null>(null);
	const [lipSyncLoading, setLipSyncLoading] = useState(false);
	const [lipSyncProgress, setLipSyncProgress] = useState<{
		phase: string;
		current: number;
		total: number;
	} | null>(null);
	const [modelDownloading, setModelDownloading] = useState(false);

	// Load settings on mount
	useEffect(() => {
		window.electronAPI.getHuruSettings().then((result) => {
			if (result.success) {
				setIsConfigured(result.isConfigured);

				setConsumerEmail(result.consumerEmail || "");
			}
		});
		window.electronAPI.checkLuxTtsAvailable().then((result) => {
			setLuxTtsAvailable(result.available);
		});
	}, []);

	// Listen for dub progress events
	useEffect(() => {
		const cleanup = window.electronAPI.onDubProgress((payload) => {
			setDubProgress(payload);
		});
		return cleanup;
	}, []);

	// Lip sync: check availability + model status
	useEffect(() => {
		window.electronAPI.checkLipSyncAvailable().then((result) => {
			setLipSyncAvailable(result.available);
		});
		window.electronAPI.getLipSyncModelStatus().then(setLipSyncModelStatus);
	}, []);

	// Listen for lip sync progress events
	useEffect(() => {
		const cleanup = window.electronAPI.onLipSyncProgress((payload) => {
			setLipSyncProgress(payload);
		});
		return cleanup;
	}, []);

	// Listen for lip sync model download progress
	useEffect(() => {
		const cleanup = window.electronAPI.onLipSyncModelDownloadProgress((payload) => {
			if (payload.status === "downloaded") {
				setModelDownloading(false);
				window.electronAPI.getLipSyncModelStatus().then(setLipSyncModelStatus);
			} else if (payload.status === "error") {
				setModelDownloading(false);
			}
		});
		return cleanup;
	}, []);

	const saveSettings = useCallback(async () => {
		if (!apiKeyInput || !emailInput) {
			setSetupError("API key and email are required.");
			return;
		}
		setSetupLoading(true);
		setSetupError("");
		try {
			const result = await window.electronAPI.saveHuruSettings({
				apiKey: apiKeyInput,
				consumerEmail: emailInput,
			});
			if (result.success) {
				setIsConfigured(true);

				setConsumerEmail(emailInput);
				setApiKeyInput("");
			} else {
				setSetupError(result.error || "Failed to save settings.");
			}
		} finally {
			setSetupLoading(false);
		}
	}, [apiKeyInput, emailInput]);

	const logout = useCallback(async () => {
		setLogoutLoading(true);
		try {
			const result = await window.electronAPI.logoutHuru();
			if (result.success) {
				setIsConfigured(false);

				setConsumerEmail("");
				setApiKeyInput("");
				setEmailInput("");
			}
		} finally {
			setLogoutLoading(false);
		}
	}, []);

	const generateCaptions = useCallback(async () => {
		if (!videoPath) {
			const message = "Record or import a video first.";
			setCaptionError(message);
			toast.error(message);
			return;
		}
		setCaptionLoading(true);
		setCaptionError("");
		try {
			const lang = captionLanguage !== "auto" ? captionLanguage : undefined;
			const result = await window.electronAPI.generateAiCaptions({
				videoPath,
				language: lang,
			});
			if (result.success && result.cues) {
				onCaptionsGenerated?.(result.cues, captionLanguage);
				toast.success(result.message || `Generated ${result.cues.length} caption cue(s).`);
			} else if (result.checkoutUrl) {
				setCaptionError("Credits exhausted.");
				toast.error("Credits exhausted.", {
					action: {
						label: "Buy credits",
						onClick: () => window.electronAPI.openExternalUrl(result.checkoutUrl!),
					},
				});
			} else {
				const message = result.error || "Caption generation failed.";
				setCaptionError(message);
				toast.error(message);
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : "Caption generation failed.";
			setCaptionError(message);
			toast.error(message);
		} finally {
			setCaptionLoading(false);
		}
	}, [videoPath, captionLanguage, onCaptionsGenerated]);

	const translateCaptions = useCallback(async () => {
		const sourceTrack = captionTracks.find((t) => t.isSource);
		if (!sourceTrack) {
			const message = "Generate captions first.";
			setTranslationError(message);
			toast.error(message);
			return;
		}
		if (!targetLanguage) {
			const message = "Select a target language first.";
			setTranslationError(message);
			toast.error(message);
			return;
		}
		setTranslationLoading(true);
		setTranslationError("");
		try {
			const sourceLangLabel =
				CAPTION_LANGUAGE_LABELS[sourceTrack.language] || sourceTrack.language;
			const targetLangLabel = CAPTION_LANGUAGE_LABELS[targetLanguage] || targetLanguage;
			const result = await window.electronAPI.translateAiCaptions({
				cues: sourceTrack.cues,
				sourceLanguage: sourceLangLabel,
				targetLanguage: targetLangLabel,
			});
			if (result.success && result.cues) {
				const newTrack: CaptionTrack = {
					id: `translation-${targetLanguage}-${Date.now()}`,
					language: targetLanguage,
					label: targetLangLabel,
					cues: result.cues,
					isSource: false,
				};
				onTranslationGenerated?.(newTrack);
				toast.success(result.message || `Translated ${result.cues.length} caption cue(s).`);
			} else {
				handleCheckoutUrl(result.checkoutUrl);
				if (!result.checkoutUrl) {
					const message = result.error || "AI caption translation failed.";
					setTranslationError(message);
					toast.error(message);
				}
			}
		} catch (error) {
			const message =
				error instanceof Error ? error.message : "AI caption translation failed.";
			setTranslationError(message);
			toast.error(message);
		} finally {
			setTranslationLoading(false);
		}
	}, [captionTracks, targetLanguage, onTranslationGenerated]);

	const cloneVoice = useCallback(async () => {
		if (!videoPath) return;
		const activeTrack = captionTracks.find((t) => t.id === activeCaptionTrackId);
		const sampleWindow = activeTrack ? selectVoiceReferenceWindow(activeTrack.cues) : null;
		setVoiceCloningLoading(true);
		try {
			const result = await window.electronAPI.cloneVoiceFromVideo({
				videoPath,
				sampleWindow: sampleWindow ?? undefined,
			});
			if (result.success && result.refWavPath) {
				onVoiceCloned?.(result.refWavPath);
			} else {
				console.error("Voice cloning failed:", result.error);
			}
		} finally {
			setVoiceCloningLoading(false);
		}
	}, [activeCaptionTrackId, captionTracks, videoPath, onVoiceCloned]);

	const generateDub = useCallback(async () => {
		const activeTrack = captionTracks.find((t) => t.id === activeCaptionTrackId);
		if (!activeTrack || !refWavPath || !totalDurationMs) return;
		const dubCues = buildDubGenerationCues(activeTrack.cues);
		if (dubCues.length === 0) {
			toast.error("No caption timing available for dubbing.");
			return;
		}
		setDubbingLoading(true);
		setDubProgress(null);
		try {
			const result = await window.electronAPI.generateDubbedAudio({
				cues: dubCues,
				refWavPath,
				totalDurationMs,
			});
			if (result.success && result.audioData) {
				const dubbed: DubbedAudio = {
					audioData: result.audioData,
					refWavPath,
					language: activeTrack.language,
					label: `${activeTrack.label} Dub`,
					createdAt: Date.now(),
				};
				onDubbedAudioGenerated?.(dubbed);
			} else {
				console.error("Dub generation failed:", result.error);
			}
		} finally {
			setDubbingLoading(false);
			setDubProgress(null);
		}
	}, [captionTracks, activeCaptionTrackId, refWavPath, totalDurationMs, onDubbedAudioGenerated]);

	const downloadModels = useCallback(async () => {
		setModelDownloading(true);
		try {
			const result = await window.electronAPI.downloadLipSyncModels();
			if (!result.success) {
				console.error("Model download failed:", result.error);
				setModelDownloading(false);
			}
		} catch {
			setModelDownloading(false);
		}
	}, []);

	const deleteModels = useCallback(async () => {
		await window.electronAPI.deleteLipSyncModels();
		setLipSyncModelStatus({ wav2lipExists: false, s3fdExists: false });
	}, []);

	const generateLipSync = useCallback(async () => {
		if (!videoPath || !dubbedAudio) return;
		setLipSyncLoading(true);
		setLipSyncProgress(null);
		try {
			const result = await window.electronAPI.generateLipSyncVideo({
				videoPath,
				dubbedAudioData: dubbedAudio.audioData,
			});
			if (result.success && result.outputPath) {
				onLipSyncGenerated?.({
					videoPath: result.outputPath,
					sourceVideoPath: videoPath,
					dubbedAudioLabel: dubbedAudio.label,
					createdAt: Date.now(),
				});
			} else {
				console.error("Lip sync generation failed:", result.error);
			}
		} finally {
			setLipSyncLoading(false);
			setLipSyncProgress(null);
		}
	}, [videoPath, dubbedAudio, onLipSyncGenerated]);

	const generateAllMetadata = useCallback(async () => {
		if (!transcript) {
			const message = "Generate captions first.";
			setMetadataError(message);
			toast.error(message);
			return;
		}
		setMetadataLoading("all");
		setMetadataError("");
		try {
			const types = ["title", "description", "chapters"] as const;
			for (const type of types) {
				setMetadataLoading(type);
				const result = await window.electronAPI.generateAiMetadata({
					transcript,
					type,
				});
				if (result.success && result.content) {
					switch (type) {
						case "title":
							setGeneratedTitle(result.content);
							break;
						case "description":
							setGeneratedDescription(result.content);
							break;
						case "chapters":
							setGeneratedChapters(result.content);
							break;
					}
				} else {
					handleCheckoutUrl(result.checkoutUrl);
					if (!result.checkoutUrl) {
						const message = result.error || `Failed to generate ${type}.`;
						setMetadataError(message);
						toast.error(message);
					}
					return;
				}
			}
			toast.success("Generated title, description, and chapters.");
		} finally {
			setMetadataLoading(null);
		}
	}, [transcript]);

	const copyToClipboard = useCallback((text: string) => {
		navigator.clipboard.writeText(text);
	}, []);

	// ─── Not Configured ───────────────────────────────────────────
	if (!isConfigured) {
		return (
			<section className="flex flex-col gap-3">
				<p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
					AI Assistant
				</p>

				<div className="flex flex-col gap-3 rounded-lg bg-foreground/[0.03] p-3">
					<div className="flex flex-col gap-1">
						<span className="text-[11px] font-medium text-foreground">
							Set up AI features
						</span>
						<span className="text-[10px] text-muted-foreground">
							Enter your Huru API key and email to enable AI-powered translation and
							metadata generation.
						</span>
					</div>

					<div className="flex flex-col gap-2">
						<Input
							type="password"
							value={apiKeyInput}
							onChange={(e) => setApiKeyInput(e.target.value)}
							placeholder="API key..."
							className="h-8 text-xs font-mono"
						/>
						<Input
							type="email"
							value={emailInput}
							onChange={(e) => setEmailInput(e.target.value)}
							placeholder="Email address..."
							className="h-8 text-xs"
						/>
						<Button
							className="h-9 justify-start gap-2 bg-[#D4D0C8] text-[#0A0A0A] hover:bg-[#D4D0C8]/90"
							onClick={saveSettings}
							disabled={!apiKeyInput || !emailInput || setupLoading}
						>
							{setupLoading ? (
								<>
									<CircleNotch className="h-4 w-4 animate-spin" />
									Connecting...
								</>
							) : (
								<>
									<Sparkle className="h-4 w-4" weight="fill" />
									Connect
								</>
							)}
						</Button>
					</div>

					<button
						type="button"
						className="text-[10px] text-muted-foreground hover:text-foreground transition-colors text-left"
						onClick={() => window.electronAPI.openExternalUrl("https://huruai.xyz")}
					>
						Get an API key at huruai.xyz
					</button>

					{setupError && <span className="text-[10px] text-red-500">{setupError}</span>}
				</div>

				<div className="flex items-center gap-1.5 px-1">
					<span className="text-[10px] text-muted-foreground">Powered by Huru</span>
				</div>
			</section>
		);
	}

	// ─── Connected ───────────────────────────────────────────────
	return (
		<section className="flex flex-col gap-3">
			<p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
				AI Assistant
			</p>

			{/* Status Bar */}
			<div className="flex items-center justify-between rounded-lg bg-foreground/[0.03] px-3 py-2">
				<div className="flex items-center gap-2">
					<div className="h-2 w-2 rounded-full bg-green-500" />
					<span className="text-[11px] font-medium text-foreground">Connected</span>
				</div>
				<span className="text-[11px] text-muted-foreground">{consumerEmail}</span>
			</div>

			{/* Actions */}
			<div className="flex flex-col gap-2 rounded-lg bg-foreground/[0.03] p-3">
				<span className="text-[9px] font-semibold uppercase tracking-widest text-muted-foreground">
					Actions
				</span>

				{/* Source language selector */}
				<div className="flex flex-col gap-1">
					<span className="text-[10px] text-muted-foreground">Language</span>
					<Select value={captionLanguage} onValueChange={setCaptionLanguage}>
						<SelectTrigger className="h-8 text-xs">
							<SelectValue />
						</SelectTrigger>
						<SelectContent>
							{Object.entries(CAPTION_LANGUAGE_LABELS).map(([code, label]) => (
								<SelectItem key={code} value={code} className="text-xs">
									{label}
								</SelectItem>
							))}
						</SelectContent>
					</Select>
				</div>

				<Button
					className="h-9 justify-start gap-2 bg-[#D4D0C8] text-[#0A0A0A] hover:bg-[#D4D0C8]/90"
					onClick={generateCaptions}
					disabled={!videoPath || captionLoading}
				>
					{captionLoading ? (
						<CircleNotch className="h-4 w-4 animate-spin" />
					) : (
						<Sparkle className="h-4 w-4" weight="fill" />
					)}
					Generate Captions
				</Button>
				<span className="text-[10px] text-muted-foreground -mt-1 ml-1">
					{!videoPath ? "Record or import a video first" : "Uses local Whisper"}
				</span>
				{captionError ? (
					<span className="text-[10px] text-red-500 -mt-1 ml-1">{captionError}</span>
				) : null}

				<Button
					className="h-9 justify-start gap-2 bg-[#D4D0C8] text-[#0A0A0A] hover:bg-[#D4D0C8]/90"
					onClick={generateAllMetadata}
					disabled={!transcript || metadataLoading !== null}
				>
					{metadataLoading ? (
						<CircleNotch className="h-4 w-4 animate-spin" />
					) : (
						<Sparkle className="h-4 w-4" weight="fill" />
					)}
					Generate Metadata
				</Button>
				<span className="text-[10px] text-muted-foreground -mt-1 ml-1">
					{!transcript ? "Generate captions first" : "Title, description & chapters"}
				</span>
				{metadataError ? (
					<span className="text-[10px] text-red-500 -mt-1 ml-1">{metadataError}</span>
				) : null}
			</div>

			{/* Caption Tracks */}
			{captionTracks.length > 0 && (
				<div className="flex flex-col gap-2 rounded-lg bg-foreground/[0.03] p-3">
					<span className="text-[9px] font-semibold uppercase tracking-widest text-muted-foreground">
						Caption Tracks
					</span>
					<div className="flex flex-col gap-1">
						{captionTracks.map((track) => (
							<button
								key={track.id}
								type="button"
								className={`flex items-center gap-2 rounded px-2 py-1.5 text-left text-[11px] transition-colors ${
									track.id === activeCaptionTrackId
										? "bg-[#D4D0C8]/10 text-[#D4D0C8] font-medium"
										: "text-foreground hover:bg-foreground/[0.05]"
								}`}
								onClick={() => onActiveTrackChange?.(track.id)}
							>
								<span
									className={`h-1.5 w-1.5 rounded-full ${
										track.id === activeCaptionTrackId
											? "bg-[#D4D0C8]"
											: "bg-muted-foreground/30"
									}`}
								/>
								<span className="flex-1">{track.label}</span>
								{track.isSource && (
									<span className="text-[9px] text-muted-foreground">source</span>
								)}
								<span className="text-[9px] text-muted-foreground">
									{track.cues.length}
								</span>
							</button>
						))}
					</div>

					{/* Translate row */}
					<div className="flex flex-col gap-1.5 pt-1 border-t border-foreground/[0.06]">
						<span className="text-[10px] text-muted-foreground">Translate to</span>
						<div className="flex gap-2">
							<Select value={targetLanguage} onValueChange={setTargetLanguage}>
								<SelectTrigger className="h-8 text-xs flex-1">
									<SelectValue />
								</SelectTrigger>
								<SelectContent>
									{Object.entries(CAPTION_LANGUAGE_LABELS)
										.filter(([code]) => code !== "auto")
										.map(([code, label]) => (
											<SelectItem key={code} value={code} className="text-xs">
												{label}
											</SelectItem>
										))}
								</SelectContent>
							</Select>
							<Button
								variant="outline"
								className="h-8 text-xs gap-1.5"
								onClick={translateCaptions}
								disabled={
									translationLoading || !captionTracks.some((t) => t.isSource)
								}
							>
								{translationLoading ? (
									<CircleNotch className="h-3.5 w-3.5 animate-spin" />
								) : (
									<Translate className="h-3.5 w-3.5" />
								)}
								Translate
							</Button>
						</div>
						{translationError ? (
							<span className="text-[10px] text-red-500">{translationError}</span>
						) : null}
					</div>
				</div>
			)}

			{/* Voice Dubbing */}
			<div className="flex flex-col gap-2 rounded-lg bg-foreground/[0.03] p-3">
				<span className="text-[9px] font-semibold uppercase tracking-widest text-muted-foreground">
					Voice Dubbing
				</span>

				{/* LuxTTS availability */}
				{luxTtsAvailable === null ? (
					<div className="flex items-center gap-1.5">
						<CircleNotch className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
						<span className="text-[10px] text-muted-foreground">
							Checking LuxTTS...
						</span>
					</div>
				) : luxTtsAvailable ? (
					<div className="flex items-center gap-1.5">
						<Check className="h-3.5 w-3.5 text-green-500" weight="bold" />
						<span className="text-[10px] text-muted-foreground">
							LuxTTS available (local)
						</span>
					</div>
				) : (
					<div className="flex flex-col gap-1">
						<span className="text-[10px] text-red-500">LuxTTS not found</span>
						<span className="text-[10px] text-muted-foreground">
							Install LuxTTS and its dependencies to enable voice dubbing.
						</span>
					</div>
				)}

				{/* Clone Voice */}
				<Button
					variant="outline"
					className="h-9 justify-start gap-2 text-xs"
					onClick={cloneVoice}
					disabled={!luxTtsAvailable || !videoPath || voiceCloningLoading}
				>
					{voiceCloningLoading ? (
						<CircleNotch className="h-4 w-4 animate-spin" />
					) : (
						<Microphone className="h-4 w-4" />
					)}
					{voiceCloningLoading ? "Extracting voice..." : "Clone Voice from Video"}
				</Button>
				{refWavPath && (
					<div className="flex items-center gap-1.5">
						<Check className="h-3.5 w-3.5 text-green-500" weight="bold" />
						<span className="text-[10px] text-muted-foreground">Voice cloned</span>
					</div>
				)}

				{/* Generate Dub */}
				{(() => {
					const activeTrack = captionTracks.find((t) => t.id === activeCaptionTrackId);
					const trackLabel = activeTrack?.label || "Dub";
					return (
						<Button
							className="h-9 justify-start gap-2 bg-[#D4D0C8] text-[#0A0A0A] hover:bg-[#D4D0C8]/90"
							onClick={generateDub}
							disabled={
								!luxTtsAvailable || !refWavPath || !activeTrack || dubbingLoading
							}
						>
							{dubbingLoading ? (
								<CircleNotch className="h-4 w-4 animate-spin" />
							) : (
								<SpeakerHigh className="h-4 w-4" weight="fill" />
							)}
							{dubbingLoading && dubProgress
								? `Generating... (${dubProgress.current}/${dubProgress.total})`
								: `Generate ${trackLabel} Dub`}
						</Button>
					);
				})()}
				{!refWavPath && (
					<span className="text-[10px] text-muted-foreground -mt-1 ml-1">
						Clone your voice first
					</span>
				)}
				{dubbedAudio && (
					<div className="flex items-center gap-1.5">
						<Check className="h-3.5 w-3.5 text-green-500" weight="bold" />
						<span className="text-[10px] text-muted-foreground">
							{dubbedAudio.label} ready
						</span>
					</div>
				)}
			</div>

			{/* Lip Sync */}
			{dubbedAudio && (
				<div className="flex flex-col gap-2 rounded-lg bg-foreground/[0.03] p-3">
					<span className="text-[9px] font-semibold uppercase tracking-widest text-muted-foreground">
						Lip Sync
					</span>

					{lipSyncAvailable === false && (
						<span className="text-[10px] text-muted-foreground">
							Requires Python with torch and cv2 installed
						</span>
					)}

					{lipSyncAvailable &&
						lipSyncModelStatus &&
						!lipSyncModelStatus.wav2lipExists && (
							<Button
								className="h-9 justify-start gap-2"
								variant="outline"
								onClick={downloadModels}
								disabled={modelDownloading}
							>
								{modelDownloading ? (
									<CircleNotch className="h-4 w-4 animate-spin" />
								) : (
									<ArrowLineDown className="h-4 w-4" />
								)}
								{modelDownloading ? "Downloading..." : "Download Models (~240MB)"}
							</Button>
						)}

					{lipSyncAvailable &&
						lipSyncModelStatus?.wav2lipExists &&
						lipSyncModelStatus?.s3fdExists && (
							<>
								<Button
									className="h-9 justify-start gap-2 bg-[#D4D0C8] text-[#0A0A0A] hover:bg-[#D4D0C8]/90"
									onClick={generateLipSync}
									disabled={lipSyncLoading}
								>
									{lipSyncLoading ? (
										<CircleNotch className="h-4 w-4 animate-spin" />
									) : (
										<Sparkle className="h-4 w-4" weight="fill" />
									)}
									{lipSyncLoading && lipSyncProgress
										? `${lipSyncProgress.phase.replace(/_/g, " ")} (${lipSyncProgress.current}/${lipSyncProgress.total})`
										: "Generate Lip Sync"}
								</Button>
								{lipSyncVideoPath && (
									<div className="flex items-center gap-1.5">
										<Check
											className="h-3.5 w-3.5 text-green-500"
											weight="bold"
										/>
										<span className="text-[10px] text-muted-foreground">
											Lip-synced video ready
										</span>
									</div>
								)}
								<Button
									className="h-7 justify-start gap-2 text-[10px]"
									variant="ghost"
									onClick={deleteModels}
									disabled={lipSyncLoading || modelDownloading}
								>
									<Trash className="h-3.5 w-3.5" />
									Delete Models
								</Button>
							</>
						)}
				</div>
			)}

			{/* Results */}
			{(generatedTitle || generatedDescription || generatedChapters) && (
				<div className="flex flex-col gap-2 rounded-lg bg-foreground/[0.03] p-3">
					<span className="text-[9px] font-semibold uppercase tracking-widest text-muted-foreground">
						Results
					</span>

					{generatedTitle && (
						<div className="flex flex-col gap-1">
							<div className="flex items-center justify-between">
								<span className="text-[10px] font-medium text-muted-foreground">
									Title
								</span>
								<Button
									variant="ghost"
									size="sm"
									className="h-6 w-6 p-0"
									onClick={() => copyToClipboard(generatedTitle)}
								>
									<Copy className="h-3.5 w-3.5" />
								</Button>
							</div>
							<p className="text-[11px] leading-relaxed text-foreground">
								{generatedTitle}
							</p>
						</div>
					)}

					{generatedDescription && (
						<div className="flex flex-col gap-1">
							<div className="flex items-center justify-between">
								<span className="text-[10px] font-medium text-muted-foreground">
									Description
								</span>
								<Button
									variant="ghost"
									size="sm"
									className="h-6 w-6 p-0"
									onClick={() => copyToClipboard(generatedDescription)}
								>
									<Copy className="h-3.5 w-3.5" />
								</Button>
							</div>
							<p className="whitespace-pre-wrap text-[11px] leading-relaxed text-foreground">
								{generatedDescription}
							</p>
						</div>
					)}

					{generatedChapters && (
						<div className="flex flex-col gap-1">
							<div className="flex items-center justify-between">
								<span className="text-[10px] font-medium text-muted-foreground">
									Chapters
								</span>
								<Button
									variant="ghost"
									size="sm"
									className="h-6 w-6 p-0"
									onClick={() => copyToClipboard(generatedChapters)}
								>
									<Copy className="h-3.5 w-3.5" />
								</Button>
							</div>
							<p className="whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-foreground">
								{generatedChapters}
							</p>
						</div>
					)}
				</div>
			)}

			{/* Settings Accordion */}
			<Accordion type="single" collapsible className="border-none">
				<AccordionItem value="settings" className="border-none">
					<AccordionTrigger className="py-2 text-[11px]">
						<div className="flex items-center gap-1.5">
							<GearSix className="h-3.5 w-3.5" />
							Settings
						</div>
					</AccordionTrigger>
					<AccordionContent className="pb-2 pt-0">
						<div className="flex flex-col gap-3">
							<div className="flex items-center justify-between">
								<span className="text-[10px] text-muted-foreground">Email</span>
								<span className="text-[10px] text-foreground">{consumerEmail}</span>
							</div>

							<div className="flex flex-col gap-1">
								<span className="text-[10px] text-muted-foreground">
									Update API key
								</span>
								<div className="flex gap-2">
									<Input
										type="password"
										value={apiKeyInput}
										onChange={(e) => setApiKeyInput(e.target.value)}
										placeholder="New API key..."
										className="h-8 text-xs font-mono flex-1"
									/>
									<Button
										variant="outline"
										className="h-8 text-xs"
										onClick={async () => {
											if (!apiKeyInput) return;
											const result =
												await window.electronAPI.saveHuruSettings({
													apiKey: apiKeyInput,
												});
											if (result.success) {
												setApiKeyInput("");
												toast.success("API key updated.");
											}
										}}
										disabled={!apiKeyInput}
									>
										Save
									</Button>
								</div>
							</div>

							<Button
								variant="outline"
								className="h-8 gap-1.5 text-xs text-red-500 hover:text-red-500"
								onClick={logout}
								disabled={logoutLoading}
							>
								{logoutLoading ? (
									<CircleNotch className="h-3.5 w-3.5 animate-spin" />
								) : (
									<Trash className="h-3.5 w-3.5" />
								)}
								Log out
							</Button>
						</div>
					</AccordionContent>
				</AccordionItem>
			</Accordion>

			{/* Powered by badge */}
			<div className="flex items-center gap-1.5 px-1">
				<span className="text-[10px] text-muted-foreground">Powered by Huru</span>
			</div>
		</section>
	);
}
