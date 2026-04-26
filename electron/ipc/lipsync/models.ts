import { constants as fsConstants } from "node:fs";
import fs from "node:fs/promises";
import type Electron from "electron";
import { downloadFileWithProgress } from "../captions/whisper";
import {
	LIP_SYNC_MODEL_DIR,
	S3FD_DOWNLOAD_URL,
	S3FD_MODEL_PATH,
	WAV2LIP_GAN_DOWNLOAD_URL,
	WAV2LIP_GAN_MODEL_PATH,
} from "../constants";

export function sendLipSyncModelDownloadProgress(
	webContents: Electron.WebContents,
	payload: {
		status: "idle" | "downloading" | "downloaded" | "error";
		progress: number;
		file?: string;
		error?: string;
	},
) {
	webContents.send("lip-sync-model-download-progress", payload);
}

export async function getLipSyncModelStatus(): Promise<{
	wav2lipExists: boolean;
	s3fdExists: boolean;
}> {
	const [wav2lipExists, s3fdExists] = await Promise.all([
		fs
			.access(WAV2LIP_GAN_MODEL_PATH, fsConstants.R_OK)
			.then(() => true)
			.catch(() => false),
		fs
			.access(S3FD_MODEL_PATH, fsConstants.R_OK)
			.then(() => true)
			.catch(() => false),
	]);
	return { wav2lipExists, s3fdExists };
}

export async function downloadLipSyncModels(webContents: Electron.WebContents): Promise<void> {
	await fs.mkdir(LIP_SYNC_MODEL_DIR, { recursive: true });

	const downloadModel = async (url: string, destPath: string, label: string) => {
		const tempPath = `${destPath}.download`;
		sendLipSyncModelDownloadProgress(webContents, {
			status: "downloading",
			progress: 0,
			file: label,
		});

		try {
			await fs.rm(tempPath, { force: true });
			await downloadFileWithProgress(url, tempPath, (progress) => {
				sendLipSyncModelDownloadProgress(webContents, {
					status: "downloading",
					progress,
					file: label,
				});
			});
			await fs.rename(tempPath, destPath);
		} catch (error) {
			await fs.rm(tempPath, { force: true }).catch(() => undefined);
			throw error;
		}
	};

	try {
		await downloadModel(WAV2LIP_GAN_DOWNLOAD_URL, WAV2LIP_GAN_MODEL_PATH, "wav2lip_gan.pth");
		await downloadModel(S3FD_DOWNLOAD_URL, S3FD_MODEL_PATH, "s3fd.pth");
		sendLipSyncModelDownloadProgress(webContents, {
			status: "downloaded",
			progress: 100,
		});
	} catch (error) {
		sendLipSyncModelDownloadProgress(webContents, {
			status: "error",
			progress: 0,
			error: String(error),
		});
		throw error;
	}
}

export async function deleteLipSyncModels(): Promise<void> {
	await Promise.all([
		fs.rm(WAV2LIP_GAN_MODEL_PATH, { force: true }),
		fs.rm(S3FD_MODEL_PATH, { force: true }),
	]);
}
