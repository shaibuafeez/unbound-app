import {
	ArrowLineDown,
	Check,
	CircleNotch,
	Copy,
	GearSix,
	Microphone,
	ShieldCheck,
	Sparkle,
	SpeakerHigh,
	Translate,
	Trash,
	Wallet,
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

interface ProviderInfo {
	provider: string;
	model: string;
	serviceType: string;
	inputPrice?: string;
	outputPrice?: string;
	verifiability?: string;
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
	// Keep the original caption timing boundaries for cloned dubbing.
	// Merging multiple cues into a larger phrase sounds smoother, but it also
	// stretches spoken words across silent gaps and causes local drift against
	// the source recording.
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
	// Wallet state
	const [privateKeyInput, setPrivateKeyInput] = useState("");
	const [network, setNetwork] = useState<"mainnet" | "testnet">("mainnet");
	const [walletAddress, setWalletAddress] = useState("");
	const [isInitialized, setIsInitialized] = useState(false);
	const [hasPrivateKey, setHasPrivateKey] = useState(false);
	const [initLoading, setInitLoading] = useState(false);
	const [initError, setInitError] = useState("");
	const [logoutLoading, setLogoutLoading] = useState(false);

	// Generated wallet display
	const [generatedWallet, setGeneratedWallet] = useState<{
		address: string;
		privateKey: string;
	} | null>(null);

	// Bumped on each successful initialization to trigger refetches
	const [initEpoch, setInitEpoch] = useState(0);

	// Balance state
	const [totalBalance, setTotalBalance] = useState("");
	const [availableBalance, setAvailableBalance] = useState("");

	// Account actions
	const [depositAmount, setDepositAmount] = useState("");
	const [transferAmount, setTransferAmount] = useState("");
	const [transferTarget, setTransferTarget] = useState("");
	const [depositLoading, setDepositLoading] = useState(false);
	const [transferLoading, setTransferLoading] = useState(false);

	// Providers
	const [chatProviders, setChatProviders] = useState<ProviderInfo[]>([]);
	const [sttProviders, setSttProviders] = useState<ProviderInfo[]>([]);
	const [selectedChatProvider, setSelectedChatProvider] = useState("");
	const [selectedSttProvider, setSelectedSttProvider] = useState("");
	const [providersLoading, setProvidersLoading] = useState(false);
	const [providersError, setProvidersError] = useState("");

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

	// UX state
	const [showImportSection, setShowImportSection] = useState(false);
	const [backupDismissed, setBackupDismissed] = useState(false);

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
		window.electronAPI.getAiSettings().then((result) => {
			if (result.success) {
				setNetwork((result.network as "mainnet" | "testnet") || "mainnet");
				setSelectedChatProvider(result.selectedChatProvider || "");
				setSelectedSttProvider(result.selectedSttProvider || "");
				setWalletAddress(result.walletAddress || "");
				setIsInitialized(result.isInitialized);
				setHasPrivateKey(result.hasPrivateKey);
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

	// Fetch balance when initialized (or re-initialized with new network)
	// biome-ignore lint/correctness/useExhaustiveDependencies: initEpoch is an intentional refetch trigger
	useEffect(() => {
		if (!isInitialized) return;
		window.electronAPI.getAiBalance().then((result) => {
			if (result.success) {
				setTotalBalance(result.totalBalance || "0");
				setAvailableBalance(result.availableBalance || "0");
			}
		});
	}, [isInitialized, initEpoch]);

	// Fetch providers when initialized (or re-initialized with new network)
	// biome-ignore lint/correctness/useExhaustiveDependencies: initEpoch is an intentional refetch trigger
	useEffect(() => {
		if (!isInitialized) {
			setChatProviders([]);
			setSttProviders([]);
			setProvidersLoading(false);
			setProvidersError("");
			return;
		}

		let cancelled = false;

		const loadProviders = async () => {
			setProvidersLoading(true);
			setProvidersError("");

			try {
				const result = await Promise.race([
					window.electronAPI.listAiProviders(),
					new Promise<never>((_, reject) => {
						window.setTimeout(() => {
							reject(new Error("Provider lookup timed out. Try reconnecting."));
						}, 12000);
					}),
				]);

				if (cancelled) return;

				if (!result.success) {
					throw new Error(result.error || "Failed to load providers.");
				}

				const newChat = result.chatbot || [];
				const newStt = result.stt || [];
				setChatProviders(newChat);
				setSttProviders(newStt);

				// Clear selections that no longer exist in the refreshed lists.
				const persistUpdates: {
					selectedChatProvider?: string;
					selectedSttProvider?: string;
				} = {};

				setSelectedChatProvider((prev) => {
					if (prev && !newChat.some((p) => p.provider === prev)) {
						persistUpdates.selectedChatProvider = "";
						return "";
					}
					if (!prev && newChat.length > 0) {
						persistUpdates.selectedChatProvider = newChat[0].provider;
						return newChat[0].provider;
					}
					return prev;
				});

				setSelectedSttProvider((prev) => {
					if (prev && !newStt.some((p) => p.provider === prev)) {
						persistUpdates.selectedSttProvider = "";
						return "";
					}
					if (!prev && newStt.length > 0) {
						persistUpdates.selectedSttProvider = newStt[0].provider;
						return newStt[0].provider;
					}
					return prev;
				});

				if (
					persistUpdates.selectedChatProvider !== undefined ||
					persistUpdates.selectedSttProvider !== undefined
				) {
					void window.electronAPI.saveAiSettings(persistUpdates);
				}

				if (newChat.length === 0 && newStt.length === 0) {
					setProvidersError("No providers are currently available for this network.");
				}
			} catch (error) {
				if (cancelled) return;
				setChatProviders([]);
				setSttProviders([]);
				setProvidersError(
					error instanceof Error ? error.message : "Failed to load providers.",
				);
			} finally {
				if (!cancelled) {
					setProvidersLoading(false);
				}
			}
		};

		void loadProviders();

		return () => {
			cancelled = true;
		};
	}, [isInitialized, initEpoch]);

	// Reset transferTarget when it no longer matches either selected provider
	useEffect(() => {
		if (!transferTarget) return;
		if (transferTarget !== selectedChatProvider && transferTarget !== selectedSttProvider) {
			setTransferTarget("");
		}
	}, [selectedChatProvider, selectedSttProvider, transferTarget]);

	const generateWallet = useCallback(async () => {
		const result = await window.electronAPI.generateAiWallet();
		if (result.success && result.address && result.privateKey) {
			setGeneratedWallet({ address: result.address, privateKey: result.privateKey });
			setPrivateKeyInput(result.privateKey);
		}
	}, []);

	const initializeWallet = useCallback(async () => {
		setInitLoading(true);
		setInitError("");
		try {
			// Key is validated + persisted only after successful broker init
			const result = await window.electronAPI.initializeAiWallet({
				privateKey: privateKeyInput || undefined,
				network,
			});

			if (result.success) {
				setWalletAddress(result.walletAddress || "");
				setIsInitialized(true);
				setHasPrivateKey(true);
				setPrivateKeyInput("");
				setGeneratedWallet(null);
				setInitEpoch((e) => e + 1);
				// Clear stale provider selections when network changes
				if (result.networkChanged) {
					setSelectedChatProvider("");
					setSelectedSttProvider("");
					setTransferTarget("");
				}
			} else {
				setInitError(result.error || "Failed to initialize wallet");
			}
		} finally {
			setInitLoading(false);
		}
	}, [privateKeyInput, network]);

	const logoutWallet = useCallback(async () => {
		setLogoutLoading(true);
		setInitError("");
		try {
			const result = await window.electronAPI.logoutAiWallet();
			if (result.success) {
				setWalletAddress("");
				setIsInitialized(false);
				setHasPrivateKey(false);
				setPrivateKeyInput("");
				setGeneratedWallet(null);
				setSelectedChatProvider("");
				setSelectedSttProvider("");
				setChatProviders([]);
				setSttProviders([]);
				setProvidersError("");
				setTotalBalance("");
				setAvailableBalance("");
				setDepositAmount("");
				setTransferAmount("");
				setTransferTarget("");
				setShowImportSection(true);
				if (result.network) {
					setNetwork(result.network);
				}
			} else {
				setInitError(result.error || "Failed to log out wallet");
			}
		} finally {
			setLogoutLoading(false);
		}
	}, []);

	const refreshBalance = useCallback(async () => {
		const result = await window.electronAPI.getAiBalance();
		if (result.success) {
			setTotalBalance(result.totalBalance || "0");
			setAvailableBalance(result.availableBalance || "0");
		}
	}, []);

	const handleDeposit = useCallback(async () => {
		const amount = Number.parseFloat(depositAmount);
		if (!amount || amount <= 0) return;
		setDepositLoading(true);
		try {
			const result = await window.electronAPI.depositAiFunds({ amount });
			if (result.success) {
				setDepositAmount("");
				await refreshBalance();
			}
		} finally {
			setDepositLoading(false);
		}
	}, [depositAmount, refreshBalance]);

	const handleTransfer = useCallback(async () => {
		const amount = Number.parseFloat(transferAmount);
		if (!amount || amount <= 0 || !transferTarget) return;
		setTransferLoading(true);
		try {
			const result = await window.electronAPI.transferAiFunds({
				provider: transferTarget,
				amount,
			});
			if (result.success) {
				setTransferAmount("");
				await refreshBalance();
			}
		} finally {
			setTransferLoading(false);
		}
	}, [transferAmount, transferTarget, refreshBalance]);

	const handleChatProviderChange = useCallback(async (value: string) => {
		setSelectedChatProvider(value);
		await window.electronAPI.saveAiSettings({ selectedChatProvider: value });
	}, []);

	const handleSttProviderChange = useCallback(async (value: string) => {
		setSelectedSttProvider(value);
		await window.electronAPI.saveAiSettings({ selectedSttProvider: value });
	}, []);

	const generateCaptions = useCallback(async () => {
		if (!videoPath) {
			const message = "Record or import a video first.";
			setCaptionError(message);
			toast.error(message);
			return;
		}
		if (!selectedSttProvider) {
			const message = "Select a speech-to-text provider first.";
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
				provider: selectedSttProvider,
			});
			if (result.success && result.cues) {
				onCaptionsGenerated?.(result.cues, captionLanguage);
				toast.success(result.message || `Generated ${result.cues.length} caption cue(s).`);
			} else {
				const message = result.error || "AI caption generation failed.";
				setCaptionError(message);
				toast.error(message);
			}
		} catch (error) {
			const message =
				error instanceof Error ? error.message : "AI caption generation failed.";
			setCaptionError(message);
			toast.error(message);
		} finally {
			setCaptionLoading(false);
		}
	}, [videoPath, selectedSttProvider, captionLanguage, onCaptionsGenerated]);

	const translateCaptions = useCallback(async () => {
		const sourceTrack = captionTracks.find((t) => t.isSource);
		if (!sourceTrack) {
			const message = "Generate captions first.";
			setTranslationError(message);
			toast.error(message);
			return;
		}
		if (!selectedChatProvider) {
			const message = "Select a chatbot provider first.";
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
				provider: selectedChatProvider,
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
				const message = result.error || "AI caption translation failed.";
				setTranslationError(message);
				toast.error(message);
			}
		} catch (error) {
			const message =
				error instanceof Error ? error.message : "AI caption translation failed.";
			setTranslationError(message);
			toast.error(message);
		} finally {
			setTranslationLoading(false);
		}
	}, [captionTracks, selectedChatProvider, targetLanguage, onTranslationGenerated]);

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

	// Available for individual metadata generation if needed
	// @ts-expect-error kept for future per-field generation UI
	const _generateMetadata = useCallback(
		async (type: "title" | "description" | "chapters") => {
			if (!transcript) {
				const message = "Generate captions first.";
				setMetadataError(message);
				toast.error(message);
				return;
			}
			if (!selectedChatProvider) {
				const message = "Select a chatbot provider first.";
				setMetadataError(message);
				toast.error(message);
				return;
			}
			setMetadataLoading(type);
			setMetadataError("");
			try {
				const result = await window.electronAPI.generateAiMetadata({
					transcript,
					type,
					provider: selectedChatProvider,
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
					toast.success(`Generated ${type}.`);
				} else {
					const message = result.error || `Failed to generate ${type}.`;
					setMetadataError(message);
					toast.error(message);
				}
			} finally {
				setMetadataLoading(null);
			}
		},
		[selectedChatProvider, transcript],
	);

	const generateAllMetadata = useCallback(async () => {
		if (!transcript) {
			const message = "Generate captions first.";
			setMetadataError(message);
			toast.error(message);
			return;
		}
		if (!selectedChatProvider) {
			const message = "Select a chatbot provider first.";
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
					provider: selectedChatProvider,
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
					const message = result.error || `Failed to generate ${type}.`;
					setMetadataError(message);
					toast.error(message);
					return;
				}
			}
			toast.success("Generated title, description, and chapters.");
		} finally {
			setMetadataLoading(null);
		}
	}, [selectedChatProvider, transcript]);

	const copyToClipboard = useCallback((text: string) => {
		navigator.clipboard.writeText(text);
	}, []);

	const truncateAddress = (addr: string) =>
		addr ? `${addr.slice(0, 6)}...${addr.slice(-4)}` : "";

	const formatBalance = (val: string) => {
		const n = Number.parseFloat(val);
		return Number.isNaN(n) ? "0.00" : n.toFixed(2);
	};

	const hasTransferTargets = Boolean(selectedChatProvider || selectedSttProvider);

	const getModelForProvider = (address: string, list: ProviderInfo[]) => {
		const found = list.find((p) => p.provider === address);
		return found?.model || truncateAddress(address);
	};

	// ─── Not Connected ───────────────────────────────────────────
	if (!isInitialized || !walletAddress) {
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
							Create an AI account to unlock captions and metadata generation.
						</span>
					</div>

					{/* Recovery key backup warning */}
					{generatedWallet && !backupDismissed && (
						<div className="flex flex-col gap-1.5 rounded-md bg-amber-500/10 p-2.5">
							<div className="flex items-center justify-between">
								<span className="text-[10px] font-semibold text-amber-600 dark:text-amber-400">
									Save your recovery key
								</span>
								<Button
									variant="ghost"
									size="sm"
									className="h-5 px-1.5 text-[9px] text-muted-foreground"
									onClick={() => setBackupDismissed(true)}
								>
									Dismiss
								</Button>
							</div>
							<span className="text-[10px] text-amber-600/80 dark:text-amber-400/80">
								This key cannot be recovered. Copy it somewhere safe.
							</span>
							<div className="flex items-center gap-1.5 rounded bg-foreground/[0.05] px-2 py-1.5">
								<span className="flex-1 break-all font-mono text-[10px] text-foreground">
									{generatedWallet.privateKey}
								</span>
								<Button
									variant="ghost"
									size="sm"
									className="h-6 w-6 shrink-0 p-0"
									onClick={() => copyToClipboard(generatedWallet.privateKey)}
								>
									<Copy className="h-3.5 w-3.5" />
								</Button>
							</div>
						</div>
					)}

					{/* Create New Account / Continue */}
					{generatedWallet ? (
						<Button
							className="h-9 justify-start gap-2 bg-[#D4D0C8] text-[#0A0A0A] hover:bg-[#D4D0C8]/90"
							onClick={initializeWallet}
							disabled={initLoading}
						>
							{initLoading ? (
								<>
									<CircleNotch className="h-4 w-4 animate-spin" />
									Connecting...
								</>
							) : (
								<>
									<Sparkle className="h-4 w-4" weight="fill" />
									Continue
								</>
							)}
						</Button>
					) : (
						<Button
							className="h-9 justify-start gap-2 bg-[#D4D0C8] text-[#0A0A0A] hover:bg-[#D4D0C8]/90"
							onClick={generateWallet}
						>
							<Sparkle className="h-4 w-4" weight="fill" />
							Create New Account
						</Button>
					)}

					{/* Import existing key */}
					<div className="flex flex-col gap-2">
						<button
							type="button"
							className="flex items-center gap-1.5 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
							onClick={() => setShowImportSection(!showImportSection)}
						>
							<span
								className="inline-block transition-transform duration-200"
								style={{
									transform: showImportSection ? "rotate(90deg)" : "rotate(0deg)",
								}}
							>
								▸
							</span>
							I already have an account
						</button>

						{showImportSection && (
							<div className="flex flex-col gap-2 pl-3">
								<Input
									type="password"
									value={privateKeyInput}
									onChange={(e) => setPrivateKeyInput(e.target.value)}
									placeholder="Enter recovery key..."
									className="h-8 text-xs font-mono"
								/>
								<div className="flex items-center gap-2">
									<span className="text-[10px] text-muted-foreground">
										Network:
									</span>
									<Select
										value={network}
										onValueChange={(v) =>
											setNetwork(v as "mainnet" | "testnet")
										}
									>
										<SelectTrigger className="h-8 w-28 text-xs">
											<SelectValue />
										</SelectTrigger>
										<SelectContent>
											<SelectItem value="mainnet">Mainnet</SelectItem>
											<SelectItem value="testnet">Testnet</SelectItem>
										</SelectContent>
									</Select>
								</div>
								<Button
									variant="outline"
									className="h-8 text-xs"
									onClick={initializeWallet}
									disabled={(!privateKeyInput && !hasPrivateKey) || initLoading}
								>
									{initLoading ? (
										<>
											<CircleNotch className="mr-2 h-3.5 w-3.5 animate-spin" />
											Connecting...
										</>
									) : (
										"Connect"
									)}
								</Button>
							</div>
						)}
					</div>

					{initError && <span className="text-[10px] text-red-500">{initError}</span>}
				</div>

				{/* Privacy badge */}
				<div className="flex items-center gap-1.5 px-1">
					<ShieldCheck className="h-3.5 w-3.5 text-green-500" weight="fill" />
					<span className="text-[10px] text-muted-foreground">
						Private, verified via TEE
					</span>
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
				<div className="flex items-center gap-1.5">
					<Wallet className="h-3.5 w-3.5 text-muted-foreground" />
					<span className="text-[11px] font-mono text-foreground">
						{formatBalance(totalBalance)} cr
					</span>
				</div>
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
					disabled={!videoPath || !selectedSttProvider || captionLoading}
				>
					{captionLoading ? (
						<CircleNotch className="h-4 w-4 animate-spin" />
					) : (
						<Sparkle className="h-4 w-4" weight="fill" />
					)}
					Generate Captions
				</Button>
				<span className="text-[10px] text-muted-foreground -mt-1 ml-1">
					{!videoPath ? "Record or import a video first" : "Uses speech-to-text AI"}
				</span>
				{captionError ? (
					<span className="text-[10px] text-red-500 -mt-1 ml-1">{captionError}</span>
				) : null}

				<Button
					className="h-9 justify-start gap-2 bg-[#D4D0C8] text-[#0A0A0A] hover:bg-[#D4D0C8]/90"
					onClick={generateAllMetadata}
					disabled={!transcript || !selectedChatProvider || metadataLoading !== null}
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
									translationLoading ||
									!selectedChatProvider ||
									!captionTracks.some((t) => t.isSource)
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
							{/* AI Engine */}
							<div className="flex flex-col gap-2">
								<span className="text-[10px] font-medium text-muted-foreground">
									AI Engine
								</span>

								{/* Chat provider */}
								<div className="flex flex-col gap-1">
									<span className="text-[10px] text-muted-foreground">Chat</span>
									<Select
										value={selectedChatProvider}
										onValueChange={handleChatProviderChange}
									>
										<SelectTrigger className="h-8 text-xs">
											<SelectValue
												placeholder={
													providersLoading
														? "Loading..."
														: chatProviders.length > 0
															? "Select chat engine"
															: "No chat engines available"
												}
											/>
										</SelectTrigger>
										<SelectContent>
											{providersLoading ? (
												<SelectItem
													value="__chat_loading__"
													className="text-xs"
													disabled
												>
													Loading...
												</SelectItem>
											) : chatProviders.length > 0 ? (
												chatProviders.map((p) => (
													<SelectItem
														key={p.provider}
														value={p.provider}
														className="text-xs"
														textValue={p.model}
													>
														<div className="flex flex-col">
															<span>{p.model}</span>
															<span className="text-[9px] text-muted-foreground">
																{truncateAddress(p.provider)}
															</span>
														</div>
													</SelectItem>
												))
											) : (
												<SelectItem
													value="__chat_empty__"
													className="text-xs"
													disabled
												>
													No chat engines available
												</SelectItem>
											)}
										</SelectContent>
									</Select>
								</div>

								{/* STT provider */}
								<div className="flex flex-col gap-1">
									<span className="text-[10px] text-muted-foreground">
										Speech
									</span>
									<Select
										value={selectedSttProvider}
										onValueChange={handleSttProviderChange}
									>
										<SelectTrigger className="h-8 text-xs">
											<SelectValue
												placeholder={
													providersLoading
														? "Loading..."
														: sttProviders.length > 0
															? "Select speech engine"
															: "No speech engines available"
												}
											/>
										</SelectTrigger>
										<SelectContent>
											{providersLoading ? (
												<SelectItem
													value="__stt_loading__"
													className="text-xs"
													disabled
												>
													Loading...
												</SelectItem>
											) : sttProviders.length > 0 ? (
												sttProviders.map((p) => (
													<SelectItem
														key={p.provider}
														value={p.provider}
														className="text-xs"
														textValue={p.model}
													>
														<div className="flex flex-col">
															<span>{p.model}</span>
															<span className="text-[9px] text-muted-foreground">
																{truncateAddress(p.provider)}
															</span>
														</div>
													</SelectItem>
												))
											) : (
												<SelectItem
													value="__stt_empty__"
													className="text-xs"
													disabled
												>
													No speech engines available
												</SelectItem>
											)}
										</SelectContent>
									</Select>
								</div>

								{providersError && (
									<span className="text-[10px] text-red-500">
										{providersError}
									</span>
								)}
							</div>

							{/* Network */}
							<div className="flex flex-col gap-2">
								<span className="text-[10px] font-medium text-muted-foreground">
									Network
								</span>
								<div className="flex items-center gap-2">
									<Select
										value={network}
										onValueChange={(v) =>
											setNetwork(v as "mainnet" | "testnet")
										}
									>
										<SelectTrigger className="h-8 w-28 text-xs">
											<SelectValue />
										</SelectTrigger>
										<SelectContent>
											<SelectItem value="mainnet">Mainnet</SelectItem>
											<SelectItem value="testnet">Testnet</SelectItem>
										</SelectContent>
									</Select>
									<Button
										variant="outline"
										className="h-8 text-xs"
										onClick={initializeWallet}
										disabled={initLoading || logoutLoading}
									>
										{initLoading ? (
											<CircleNotch className="h-3.5 w-3.5 animate-spin" />
										) : (
											"Reconnect"
										)}
									</Button>
									<Button
										variant="outline"
										className="h-8 gap-1.5 text-xs text-red-500 hover:text-red-500"
										onClick={logoutWallet}
										disabled={initLoading || logoutLoading}
									>
										{logoutLoading ? (
											<CircleNotch className="h-3.5 w-3.5 animate-spin" />
										) : (
											<Trash className="h-3.5 w-3.5" />
										)}
										Log out
									</Button>
								</div>
								{initError && (
									<span className="text-[10px] text-red-500">{initError}</span>
								)}

								<div className="flex items-center justify-between">
									<span className="text-[10px] text-muted-foreground">
										Balance
									</span>
									<span className="text-[11px] font-mono">
										{formatBalance(totalBalance)} credits
									</span>
								</div>
								<div className="flex items-center justify-between">
									<span className="text-[10px] text-muted-foreground">
										Available
									</span>
									<span className="text-[11px] font-mono">
										{formatBalance(availableBalance)} credits
									</span>
								</div>
							</div>

							{/* Fund Management */}
							<div className="flex flex-col gap-2">
								<span className="text-[10px] font-medium text-muted-foreground">
									Fund Management
								</span>

								{/* Deposit */}
								<div className="flex gap-2">
									<Input
										type="number"
										value={depositAmount}
										onChange={(e) => setDepositAmount(e.target.value)}
										placeholder="Amount"
										className="h-8 text-xs flex-1"
										min="0"
										step="0.1"
									/>
									<Button
										variant="outline"
										className="h-8 text-xs"
										onClick={handleDeposit}
										disabled={depositLoading || !depositAmount}
									>
										{depositLoading ? (
											<CircleNotch className="h-3.5 w-3.5 animate-spin" />
										) : (
											"Deposit"
										)}
									</Button>
								</div>

								{/* Transfer to provider */}
								<div className="flex flex-col gap-1.5">
									<div className="flex gap-2">
										<Select
											value={transferTarget}
											onValueChange={setTransferTarget}
										>
											<SelectTrigger className="h-8 text-xs flex-1">
												<SelectValue
													placeholder={
														providersLoading
															? "Loading..."
															: hasTransferTargets
																? "Select engine"
																: "Select engines first"
													}
												/>
											</SelectTrigger>
											<SelectContent>
												{providersLoading && (
													<SelectItem
														value="__providers_loading__"
														className="text-xs"
														disabled
													>
														Loading...
													</SelectItem>
												)}
												{!providersLoading && !hasTransferTargets && (
													<SelectItem
														value="__providers_missing__"
														className="text-xs"
														disabled
													>
														Select a chat or speech engine first
													</SelectItem>
												)}
												{!providersLoading && selectedChatProvider && (
													<SelectItem
														value={selectedChatProvider}
														className="text-xs"
													>
														Chat:{" "}
														{getModelForProvider(
															selectedChatProvider,
															chatProviders,
														)}
													</SelectItem>
												)}
												{!providersLoading &&
													selectedSttProvider &&
													selectedSttProvider !==
														selectedChatProvider && (
														<SelectItem
															value={selectedSttProvider}
															className="text-xs"
														>
															Speech:{" "}
															{getModelForProvider(
																selectedSttProvider,
																sttProviders,
															)}
														</SelectItem>
													)}
											</SelectContent>
										</Select>
									</div>
									<div className="flex gap-2">
										<Input
											type="number"
											value={transferAmount}
											onChange={(e) => setTransferAmount(e.target.value)}
											placeholder="Amount"
											className="h-8 text-xs flex-1"
											min="0"
											step="0.1"
										/>
										<Button
											variant="outline"
											className="h-8 text-xs"
											onClick={handleTransfer}
											disabled={
												transferLoading ||
												!transferAmount ||
												!transferTarget
											}
										>
											{transferLoading ? (
												<CircleNotch className="h-3.5 w-3.5 animate-spin" />
											) : (
												"Transfer"
											)}
										</Button>
									</div>
								</div>
							</div>

							{/* Account */}
							<div className="flex items-center justify-between">
								<span className="text-[10px] text-muted-foreground">Account</span>
								<span className="font-mono text-[10px] text-foreground">
									{truncateAddress(walletAddress)}
								</span>
							</div>
						</div>
					</AccordionContent>
				</AccordionItem>
			</Accordion>

			{/* Privacy badge */}
			<div className="flex items-center gap-1.5 px-1">
				<ShieldCheck className="h-3.5 w-3.5 text-green-500" weight="fill" />
				<span className="text-[10px] text-muted-foreground">Private, verified via TEE</span>
			</div>
		</section>
	);
}
