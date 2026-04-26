import fs from "node:fs/promises";
import path from "node:path";
import { createZGComputeNetworkBroker } from "@0glabs/0g-serving-broker";
import { ethers } from "ethers";
import { ZG_RPC_MAINNET, ZG_RPC_TESTNET } from "../ipc/constants";

export interface ZGChatMessage {
	role: "system" | "user" | "assistant";
	content: string;
}

export interface ZGChatCompletionResponse {
	id: string;
	choices: Array<{
		index: number;
		message: { role: string; content: string };
		finish_reason: string;
	}>;
	usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

export interface ZGStreamChunk {
	id: string;
	choices: Array<{
		index: number;
		delta: { role?: string; content?: string };
		finish_reason: string | null;
	}>;
	usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null;
}

export interface ZGSpeechToTextResponse {
	text: string;
	segments?: Array<{
		id: number;
		start: number;
		end: number;
		text: string;
	}>;
	usage?: Record<string, unknown>;
}

export interface ZGProvider {
	provider: string;
	model: string;
	serviceType: string;
	url?: string;
	inputPrice?: string;
	outputPrice?: string;
	verifiability?: string;
}

// biome-ignore lint/suspicious/noExplicitAny: broker type from external SDK
type Broker = any;

class ZGComputeService {
	private broker: Broker | null = null;
	private wallet: ethers.Wallet | null = null;

	async initialize(
		privateKey: string,
		network: "mainnet" | "testnet" = "mainnet",
	): Promise<void> {
		const rpcUrl = network === "mainnet" ? ZG_RPC_MAINNET : ZG_RPC_TESTNET;
		const provider = new ethers.JsonRpcProvider(rpcUrl);
		this.wallet = new ethers.Wallet(privateKey, provider);
		this.broker = await createZGComputeNetworkBroker(this.wallet);
	}

	isInitialized(): boolean {
		return this.broker !== null && this.wallet !== null;
	}

	reset(): void {
		this.broker = null;
		this.wallet = null;
	}

	getWalletAddress(): string | null {
		return this.wallet?.address ?? null;
	}

	private ensureInitialized(): void {
		if (!this.broker || !this.wallet) {
			throw new Error("Wallet not initialized. Please set up your wallet in AI settings.");
		}
	}

	private async ensureProviderReady(providerAddress: string): Promise<void> {
		this.ensureInitialized();
		try {
			await this.broker.inference.acknowledgeProviderSigner(providerAddress);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			if (
				message.includes("already") ||
				message.includes("acknowledged") ||
				message.includes("duplicate")
			) {
				return;
			}
			throw error;
		}
	}

	static generateWallet(): { address: string; privateKey: string } {
		const wallet = ethers.Wallet.createRandom();
		return { address: wallet.address, privateKey: wallet.privateKey };
	}

	async getBalance(): Promise<{
		totalBalance: string;
		availableBalance: string;
		lockedBalance: string;
	}> {
		this.ensureInitialized();
		try {
			const account = await this.broker.ledger.getLedger();
			return {
				totalBalance: ethers.formatEther(account.totalBalance ?? account.balance ?? 0n),
				availableBalance: ethers.formatEther(
					account.availableBalance ?? account.balance ?? 0n,
				),
				lockedBalance: ethers.formatEther(account.lockedBalance ?? 0n),
			};
		} catch (error) {
			// If ledger doesn't exist yet, return zeros
			if (String(error).includes("does not exist") || String(error).includes("not found")) {
				return { totalBalance: "0", availableBalance: "0", lockedBalance: "0" };
			}
			throw error;
		}
	}

	async getSubAccountBalance(
		providerAddress: string,
	): Promise<{ balance: string; pendingRefund: string }> {
		this.ensureInitialized();
		try {
			const [subAccount] = await this.broker.inference.getAccountWithDetail(providerAddress);
			return {
				balance: ethers.formatEther(subAccount.balance ?? 0n),
				pendingRefund: ethers.formatEther(subAccount.pendingRefund ?? 0n),
			};
		} catch {
			return { balance: "0", pendingRefund: "0" };
		}
	}

	async depositFunds(amount: number): Promise<void> {
		this.ensureInitialized();
		await this.broker.ledger.depositFund(amount);
	}

	async transferToProvider(providerAddress: string, amount: number): Promise<void> {
		this.ensureInitialized();
		const amountWei = ethers.parseEther(String(amount));
		await this.broker.ledger.transferFund(providerAddress, "inference", amountWei);
	}

	async listProviders(): Promise<{ chatbot: ZGProvider[]; stt: ZGProvider[] }> {
		this.ensureInitialized();
		const services = await this.broker.inference.listService();
		// Map SDK objects to plain serializable objects for Electron IPC
		// (SDK may include BigInt or class instances that can't be structured-cloned)
		const toPlain = (s: Record<string, unknown>): ZGProvider => ({
			provider: String(s.provider ?? ""),
			model: String(s.model ?? ""),
			serviceType: String(s.serviceType ?? ""),
			url: s.url ? String(s.url) : undefined,
			inputPrice: s.inputPrice != null ? String(s.inputPrice) : undefined,
			outputPrice: s.outputPrice != null ? String(s.outputPrice) : undefined,
			verifiability: s.verifiability ? String(s.verifiability) : undefined,
		});
		const chatbot = services
			.filter((s: Record<string, unknown>) => s.serviceType === "chatbot")
			.map(toPlain);
		const stt = services
			.filter((s: Record<string, unknown>) => s.serviceType === "speech-to-text")
			.map(toPlain);
		return { chatbot, stt };
	}

	async testConnection(): Promise<{ success: boolean; error?: string }> {
		try {
			this.ensureInitialized();
			const services = await this.broker.inference.listService();
			const chatbotProvider = services.find((s: ZGProvider) => s.serviceType === "chatbot");
			if (!chatbotProvider) {
				return { success: true }; // Broker works, just no chatbot providers
			}
			return { success: true };
		} catch (error) {
			return {
				success: false,
				error: error instanceof Error ? error.message : String(error),
			};
		}
	}

	async chatCompletion(
		messages: ZGChatMessage[],
		providerAddress: string,
		options?: { temperature?: number; max_tokens?: number },
	): Promise<ZGChatCompletionResponse> {
		this.ensureInitialized();
		await this.ensureProviderReady(providerAddress);

		const { endpoint, model } = await this.broker.inference.getServiceMetadata(providerAddress);
		const headers = await this.broker.inference.getRequestHeaders(providerAddress);

		const response = await fetch(`${endpoint}/chat/completions`, {
			method: "POST",
			headers: { "Content-Type": "application/json", ...headers },
			body: JSON.stringify({
				model,
				messages,
				temperature: options?.temperature ?? 0.7,
				max_tokens: options?.max_tokens ?? 4096,
			}),
		});

		if (!response.ok) {
			const errorText = await response.text().catch(() => "Unknown error");
			throw new Error(`0G API error (${response.status}): ${errorText}`);
		}

		const data = (await response.json()) as ZGChatCompletionResponse;

		// Process response for fee settlement
		let chatID = response.headers.get("ZG-Res-Key") || response.headers.get("zg-res-key");
		if (!chatID) chatID = data.id;

		await this.broker.inference.processResponse(
			providerAddress,
			chatID,
			JSON.stringify(data.usage ?? {}),
		);

		return data;
	}

	async *chatCompletionStream(
		messages: ZGChatMessage[],
		providerAddress: string,
		options?: { temperature?: number; max_tokens?: number },
	): AsyncGenerator<ZGStreamChunk, void, undefined> {
		this.ensureInitialized();
		await this.ensureProviderReady(providerAddress);

		const { endpoint, model } = await this.broker.inference.getServiceMetadata(providerAddress);
		const headers = await this.broker.inference.getRequestHeaders(providerAddress);

		const response = await fetch(`${endpoint}/chat/completions`, {
			method: "POST",
			headers: { "Content-Type": "application/json", ...headers },
			body: JSON.stringify({
				model,
				messages,
				stream: true,
				temperature: options?.temperature ?? 0.7,
				max_tokens: options?.max_tokens ?? 4096,
			}),
		});

		if (!response.ok) {
			const errorText = await response.text().catch(() => "Unknown error");
			throw new Error(`0G API error (${response.status}): ${errorText}`);
		}

		if (!response.body) {
			throw new Error("No response body for streaming request");
		}

		let chatID = response.headers.get("ZG-Res-Key") || response.headers.get("zg-res-key");
		let streamChatID: string | null = null;
		let lastUsage: Record<string, unknown> | null = null;

		const reader = response.body.getReader();
		const decoder = new TextDecoder();
		let buffer = "";

		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			buffer += decoder.decode(value, { stream: true });
			const lines = buffer.split("\n");
			buffer = lines.pop() || "";

			for (const line of lines) {
				const trimmed = line.trim();
				if (!trimmed || trimmed === "data: [DONE]") continue;
				if (!trimmed.startsWith("data:")) continue;

				try {
					const json = JSON.parse(trimmed.slice(5).trim()) as ZGStreamChunk;
					if (!streamChatID && json.id) streamChatID = json.id;
					if (json.usage) lastUsage = json.usage as Record<string, unknown>;
					yield json;
				} catch {
					// Skip malformed chunks
				}
			}
		}

		// Process remaining buffer
		if (
			buffer.trim() &&
			buffer.trim() !== "data: [DONE]" &&
			buffer.trim().startsWith("data:")
		) {
			try {
				const json = JSON.parse(buffer.trim().slice(5).trim()) as ZGStreamChunk;
				if (!streamChatID && json.id) streamChatID = json.id;
				if (json.usage) lastUsage = json.usage as Record<string, unknown>;
				yield json;
			} catch {
				// Skip malformed final chunk
			}
		}

		// Process response for fee settlement
		const finalChatID = chatID || streamChatID;
		if (finalChatID) {
			await this.broker.inference.processResponse(
				providerAddress,
				finalChatID,
				JSON.stringify(lastUsage ?? {}),
			);
		}
	}

	async speechToText(
		audioFilePath: string,
		providerAddress: string,
		language?: string,
	): Promise<ZGSpeechToTextResponse> {
		this.ensureInitialized();
		await this.ensureProviderReady(providerAddress);

		const { endpoint, model } = await this.broker.inference.getServiceMetadata(providerAddress);
		const headers = await this.broker.inference.getRequestHeaders(providerAddress);

		const audioBuffer = await fs.readFile(audioFilePath);
		const fileName = path.basename(audioFilePath);
		const blob = new Blob([audioBuffer], { type: "audio/wav" });

		const formData = new FormData();
		formData.append("file", blob, fileName);
		formData.append("model", model);
		formData.append("response_format", "verbose_json");
		if (language && language !== "auto") {
			formData.append("language", language);
		}

		const response = await fetch(`${endpoint}/audio/transcriptions`, {
			method: "POST",
			headers: { ...headers },
			body: formData,
		});

		if (!response.ok) {
			const errorText = await response.text().catch(() => "Unknown error");
			throw new Error(`0G STT API error (${response.status}): ${errorText}`);
		}

		const data = (await response.json()) as ZGSpeechToTextResponse;

		// Process response for fee settlement
		const chatID = response.headers.get("ZG-Res-Key") || response.headers.get("zg-res-key");
		await this.broker.inference.processResponse(
			providerAddress,
			chatID,
			JSON.stringify(data.usage ?? {}),
		);

		return data;
	}
}

// Singleton instance
export const zgCompute = new ZGComputeService();
