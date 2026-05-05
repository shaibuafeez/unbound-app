export interface HuruClientConfig {
	apiKey: string;
	consumerEmail: string;
	baseUrl?: string;
}

export interface HuruChatMessage {
	role: string;
	content: string;
}

export interface HuruChatResponse {
	id: string;
	choices: Array<{
		index: number;
		message: { role: string; content: string };
		finish_reason: string;
	}>;
	usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

export interface HuruStreamChunk {
	id: string;
	choices: Array<{
		index: number;
		delta: { role?: string; content?: string };
		finish_reason: string | null;
	}>;
}

export interface HuruErrorBody {
	error?: string;
	message?: string;
	checkout_url?: string;
}

export interface HuruSpeechToTextResponse {
	text: string;
	segments?: Array<{
		id: number;
		start: number;
		end: number;
		text: string;
	}>;
}

const DEFAULT_BASE_URL = "https://huruai.xyz";
const DEFAULT_MODEL = "huru/chat-1";
const STT_MODEL = "huru/stt-1";

class HuruClient {
	private apiKey: string | null = null;
	private consumerEmail: string | null = null;
	private baseUrl: string = DEFAULT_BASE_URL;

	configure(config: HuruClientConfig): void {
		this.apiKey = config.apiKey;
		this.consumerEmail = config.consumerEmail;
		this.baseUrl = config.baseUrl || DEFAULT_BASE_URL;
	}

	isConfigured(): boolean {
		return this.apiKey !== null && this.consumerEmail !== null;
	}

	reset(): void {
		this.apiKey = null;
		this.consumerEmail = null;
		this.baseUrl = DEFAULT_BASE_URL;
	}

	getConfig(): { consumerEmail: string; baseUrl: string } | null {
		if (!this.consumerEmail) return null;
		return { consumerEmail: this.consumerEmail, baseUrl: this.baseUrl };
	}

	private ensureConfigured(): void {
		if (!this.apiKey || !this.consumerEmail) {
			throw new Error("Huru API not configured. Please set up your API key and email.");
		}
	}

	private getHeaders(): Record<string, string> {
		return {
			Authorization: `Bearer ${this.apiKey}`,
			"Content-Type": "application/json",
			"X-Consumer-Email": this.consumerEmail!,
		};
	}

	private async handleErrorResponse(response: Response): Promise<never> {
		let body: HuruErrorBody = {};
		try {
			body = (await response.json()) as HuruErrorBody;
		} catch {
			// body stays empty
		}

		const message = body.error || body.message || `HTTP ${response.status}`;

		switch (response.status) {
			case 401:
				throw new HuruApiError("Invalid API key.", 401);
			case 402:
				throw new HuruPaymentError(
					body.message || "Credits exhausted. Please purchase more.",
					body.checkout_url,
				);
			case 429:
				throw new HuruApiError("Rate limit exceeded. Please try again later.", 429);
			case 503:
				throw new HuruApiError("Provider unavailable. Please try again later.", 503);
			default:
				throw new HuruApiError(
					`Huru API error (${response.status}): ${message}`,
					response.status,
				);
		}
	}

	async chatCompletion(
		messages: HuruChatMessage[],
		options?: { model?: string },
	): Promise<HuruChatResponse> {
		this.ensureConfigured();

		const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
			method: "POST",
			headers: this.getHeaders(),
			body: JSON.stringify({
				model: options?.model || DEFAULT_MODEL,
				messages,
			}),
		});

		if (!response.ok) {
			await this.handleErrorResponse(response);
		}

		return (await response.json()) as HuruChatResponse;
	}

	async *chatCompletionStream(
		messages: HuruChatMessage[],
		options?: { model?: string },
	): AsyncGenerator<HuruStreamChunk, void, undefined> {
		this.ensureConfigured();

		const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
			method: "POST",
			headers: this.getHeaders(),
			body: JSON.stringify({
				model: options?.model || DEFAULT_MODEL,
				messages,
				stream: true,
			}),
		});

		if (!response.ok) {
			await this.handleErrorResponse(response);
		}

		if (!response.body) {
			throw new Error("No response body for streaming request");
		}

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
					const json = JSON.parse(trimmed.slice(5).trim()) as HuruStreamChunk;
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
				const json = JSON.parse(buffer.trim().slice(5).trim()) as HuruStreamChunk;
				yield json;
			} catch {
				// Skip malformed final chunk
			}
		}
	}

	async speechToText(
		audioData: Blob,
		fileName: string,
		language?: string,
	): Promise<HuruSpeechToTextResponse> {
		this.ensureConfigured();

		const formData = new FormData();
		formData.append("file", audioData, fileName);
		formData.append("model", STT_MODEL);
		formData.append("response_format", "verbose_json");
		if (language && language !== "auto") {
			formData.append("language", language);
		}

		const response = await fetch(`${this.baseUrl}/v1/audio/transcriptions`, {
			method: "POST",
			headers: {
				Authorization: `Bearer ${this.apiKey}`,
				"X-Consumer-Email": this.consumerEmail!,
			},
			body: formData,
		});

		if (!response.ok) {
			await this.handleErrorResponse(response);
		}

		return (await response.json()) as HuruSpeechToTextResponse;
	}
}

export class HuruApiError extends Error {
	constructor(
		message: string,
		public readonly statusCode: number,
	) {
		super(message);
		this.name = "HuruApiError";
	}
}

export class HuruPaymentError extends Error {
	constructor(
		message: string,
		public readonly checkoutUrl?: string,
	) {
		super(message);
		this.name = "HuruPaymentError";
	}
}

// Singleton instance
export const huruClient = new HuruClient();
