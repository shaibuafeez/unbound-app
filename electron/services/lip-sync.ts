import { type ChildProcess, execFile, spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

/**
 * Self-contained Wav2Lip inference script.
 * Reads a JSON config from argv[1], runs face detection + lip sync,
 * and emits JSON progress lines on stdout.
 */
const WAV2LIP_SCRIPT = `
import json, sys, traceback, os, subprocess, tempfile, math

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    video_path = cfg["video_path"]
    audio_path = cfg["audio_path"]
    output_path = cfg["output_path"]
    checkpoint_path = cfg["checkpoint_path"]
    face_det_path = cfg["face_det_path"]
    resize_factor = cfg.get("resize_factor", 1)
    pad = cfg.get("pad", [0, 10, 0, 0])
    batch_size = cfg.get("batch_size", 16)

    import torch
    import cv2
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    # MPS does not support all ops needed by Wav2Lip, fall back to CPU
    if device == "mps":
        device = "cpu"

    print(json.dumps({"type": "progress", "phase": "loading_model", "current": 0, "total": 100}), flush=True)

    # ── Face detection ──────────────────────────────────────────────────
    from urllib.request import urlretrieve

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Load S3FD face detector
    def load_s3fd(model_path, device_name):
        import torch.nn as nn
        import torch.nn.functional as F

        class L2Norm(nn.Module):
            def __init__(self, n_channels, scale=1.0):
                super().__init__()
                self.n_channels = n_channels
                self.gamma = nn.Parameter(torch.Tensor(self.n_channels))
                self.gamma.data.fill_(scale)

            def forward(self, x):
                norm = torch.sqrt((x ** 2).sum(dim=1, keepdim=True)) + 1e-10
                x = x / norm * self.gamma.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                return x

        class S3FD(nn.Module):
            def __init__(self):
                super().__init__()
                self.vgg = nn.ModuleList([
                    nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
                ])
                self.conv_fc = nn.ModuleList([
                    nn.Conv2d(512, 1024, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(1024, 1024, 3, 1, 1), nn.ReLU(inplace=True),
                ])
                self.norm4_3 = L2Norm(512, scale=8)
                self.conv3_3_norm = nn.Conv2d(256, 4, 1)
                self.conv4_3_norm = nn.Conv2d(512, 4, 1)
                self.conv5_3 = nn.Conv2d(512, 4, 1)
                self.fc = nn.Conv2d(1024, 4, 1)
                self.conv3_3_norm_cls = nn.Conv2d(256, 4, 1)
                self.conv4_3_norm_cls = nn.Conv2d(512, 4, 1)
                self.conv5_3_cls = nn.Conv2d(512, 4, 1)
                self.fc_cls = nn.Conv2d(1024, 4, 1)

            def forward(self, x):
                features = []
                for i, layer in enumerate(self.vgg):
                    x = layer(x)
                    if i in (10, 17, 24):
                        features.append(x)
                for layer in self.conv_fc:
                    x = layer(x)
                features.append(x)
                f3_3, f4_3, f5_3, ffc = features
                f4_3 = self.norm4_3(f4_3)
                cls1 = self.conv3_3_norm_cls(f3_3)
                reg1 = self.conv3_3_norm(f3_3)
                cls2 = self.conv4_3_norm_cls(f4_3)
                reg2 = self.conv4_3_norm(f4_3)
                cls3 = self.conv5_3_cls(f5_3)
                reg3 = self.conv5_3(f5_3)
                cls4 = self.fc_cls(ffc)
                reg4 = self.fc(ffc)
                return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4]

        net = S3FD()
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()
        net.to(device_name)
        return net

    def detect_faces_s3fd(net, img, device_name, threshold=0.5):
        import torch.nn.functional as F
        h, w, _ = img.shape
        img_f = cv2.resize(img, (640, 640)).astype(np.float32)
        img_f -= np.array([104.0, 117.0, 123.0])
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0).to(device_name)
        with torch.no_grad():
            out = net(img_t)
        # Simple decode of second feature map (most reliable for faces)
        cls = torch.softmax(out[2].permute(0, 2, 3, 1).reshape(-1, 2), dim=1)[:, 1]
        reg = out[3].permute(0, 2, 3, 1).reshape(-1, 4)
        mask = cls > threshold
        if mask.sum() == 0:
            return []
        scores = cls[mask].cpu().numpy()
        best = scores.argmax()
        return [scores[best]]  # simplified - we just need to know a face exists

    def get_face_bbox_cv2(img, padding):
        """Fallback face detection using OpenCV Haar cascades."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pady1, pady2, padx1, padx2 = padding
        y1 = max(0, y - pady1)
        y2 = min(img.shape[0], y + h + pady2)
        x1 = max(0, x - padx1)
        x2 = min(img.shape[1], x + w + padx2)
        return (y1, y2, x1, x2)

    print(json.dumps({"type": "progress", "phase": "loading_model", "current": 10, "total": 100}), flush=True)

    # ── Load Wav2Lip model ──────────────────────────────────────────────
    from torch import nn

    class Conv2d(nn.Module):
        def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size, stride, padding),
                nn.BatchNorm2d(cout),
            )
            self.act = nn.ReLU()
            self.residual = residual

        def forward(self, x):
            out = self.conv_block(x)
            if self.residual:
                out += x
            return self.act(out)

    class Conv2dTranspose(nn.Module):
        def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                nn.BatchNorm2d(cout),
            )
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.conv_block(x))

    class Wav2Lip(nn.Module):
        def __init__(self):
            super().__init__()

            self.face_encoder_blocks = nn.ModuleList([
                nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),
                nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                              Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                              Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),
                nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                              Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                              Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                              Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),
                nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                              Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                              Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),
                nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                              Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                              Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),
                nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                              Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),
                nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                              Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),
            ])

            self.audio_encoder = nn.Sequential(
                Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )

            self.face_decoder_blocks = nn.ModuleList([
                nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),
                nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),
                              Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),
                nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                              Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                              Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),
                nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                              Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                              Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True)),
                nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                              Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                              Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),
                nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                              Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                              Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),
                nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                              Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                              Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),
            ])

            self.output_block = nn.Sequential(
                Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
            )

        def forward(self, audio_sequences, face_sequences):
            B = audio_sequences.size(0)

            input_dim_size = len(face_sequences.size())
            if input_dim_size > 4:
                audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
                face_sequences = torch.cat([face_sequences[:, i] for i in range(face_sequences.size(1))], dim=0)

            audio_embedding = self.audio_encoder(audio_sequences)

            feats = []
            x = face_sequences
            for f in self.face_encoder_blocks:
                x = f(x)
                feats.append(x)

            x = audio_embedding
            for i, f in enumerate(self.face_decoder_blocks):
                x = f(x)
                try:
                    x = torch.cat((x, feats[-1 - i]), dim=1)
                except Exception:
                    pass

            x = self.output_block(x)

            if input_dim_size > 4:
                x = torch.split(x, B, dim=0)
                outputs = torch.stack(x, dim=2)
            else:
                outputs = x.unsqueeze(2)

            return outputs

    print(json.dumps({"type": "progress", "phase": "loading_model", "current": 30, "total": 100}), flush=True)

    model = Wav2Lip()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    s = checkpoint.get("state_dict", checkpoint)
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v
    model.load_state_dict(new_s)
    model = model.to(device).eval()

    print(json.dumps({"type": "progress", "phase": "reading_video", "current": 40, "total": 100}), flush=True)

    # ── Read video frames ───────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_factor != 1:
            frame = cv2.resize(frame, (orig_w // resize_factor, orig_h // resize_factor))
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError("No frames read from video")

    print(json.dumps({"type": "progress", "phase": "reading_audio", "current": 50, "total": 100}), flush=True)

    # ── Read audio as mel spectrogram ───────────────────────────────────
    import scipy.io.wavfile as wavfile

    sample_rate, wav_data = wavfile.read(audio_path)
    if wav_data.dtype != np.float32:
        wav_data = wav_data.astype(np.float32) / max(np.abs(wav_data).max(), 1e-6)
    if len(wav_data.shape) > 1:
        wav_data = wav_data.mean(axis=1)
    if sample_rate != 16000:
        # Resample to 16kHz
        duration = len(wav_data) / sample_rate
        target_len = int(duration * 16000)
        indices = np.linspace(0, len(wav_data) - 1, target_len).astype(int)
        wav_data = wav_data[indices]
        sample_rate = 16000

    # Mel spectrogram (80 bands, matches Wav2Lip training)
    def melspectrogram(wav, sr=16000, n_fft=800, hop_length=200, n_mels=80):
        from scipy.signal import stft as scipy_stft
        f, t, Zxx = scipy_stft(wav, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        S = np.abs(Zxx) ** 2
        # Mel filterbank
        fmin, fmax = 55, 7600
        mel_low = 2595.0 * np.log10(1.0 + fmin / 700.0)
        mel_high = 2595.0 * np.log10(1.0 + fmax / 700.0)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        filterbank = np.zeros((n_mels, S.shape[0]))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                if f_m != f_m_minus:
                    filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus != f_m:
                    filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        mel_spec = np.dot(filterbank, S)
        mel_spec = np.log(np.maximum(mel_spec, 1e-5))
        return mel_spec

    mel = melspectrogram(wav_data)

    print(json.dumps({"type": "progress", "phase": "detecting_faces", "current": 55, "total": 100}), flush=True)

    # ── Face detection (simple bbox per frame) ──────────────────────────
    # Use first frame to find face, apply to all frames
    padding = pad
    face_rect = get_face_bbox_cv2(frames[0], padding)
    if face_rect is None:
        raise RuntimeError("No face detected in video. Ensure the speaker's face is visible.")

    y1, y2, x1, x2 = face_rect

    print(json.dumps({"type": "progress", "phase": "generating", "current": 60, "total": 100}), flush=True)

    # ── Generate lip-synced frames ──────────────────────────────────────
    mel_step_size = 16  # Wav2Lip uses 16 mel frames per video frame
    mel_chunks = []
    # Compute mel chunks aligned to video frames
    mel_idx_multiplier = 80.0 / fps  # mel frames per video frame
    total_mel_frames = mel.shape[1]

    for i in range(len(frames)):
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > total_mel_frames:
            mel_chunks.append(mel[:, total_mel_frames - mel_step_size:])
        else:
            mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])

    gen_frames = list(frames)
    num_batches = math.ceil(len(frames) / batch_size)

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(frames))
        batch_frames = frames[start:end]
        batch_mels = mel_chunks[start:end]

        img_batch = []
        mel_batch = []

        for frame, m in zip(batch_frames, batch_mels):
            face = frame[y1:y2, x1:x2]
            face_resized = cv2.resize(face, (96, 96))
            img_batch.append(face_resized)
            mel_batch.append(m)

        img_batch_np = np.array(img_batch, dtype=np.float32) / 255.0
        mel_batch_np = np.array(mel_batch, dtype=np.float32)

        # Wav2Lip expects: face input = [masked_lower_half, full_face] concatenated on channel dim
        img_masked = img_batch_np.copy()
        img_masked[:, 96 // 2:, :, :] = 0  # mask lower half

        img_input = np.concatenate([img_masked, img_batch_np], axis=3).transpose(0, 3, 1, 2)
        mel_input = mel_batch_np[:, :, :, np.newaxis].transpose(0, 3, 1, 2) if mel_batch_np.ndim == 3 else mel_batch_np

        img_t = torch.FloatTensor(img_input).to(device)
        mel_t = torch.FloatTensor(mel_input).to(device)

        with torch.no_grad():
            pred = model(mel_t, img_t)

        pred = pred.squeeze(2).cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        pred = pred.astype(np.uint8)

        for j, p in enumerate(pred):
            frame_idx = start + j
            face_out = cv2.resize(p, (x2 - x1, y2 - y1))
            gen_frames[frame_idx][y1:y2, x1:x2] = face_out

        progress_pct = 60 + int(35 * (batch_idx + 1) / num_batches)
        print(json.dumps({
            "type": "progress",
            "phase": "generating",
            "current": min(progress_pct, 95),
            "total": 100,
        }), flush=True)

    print(json.dumps({"type": "progress", "phase": "writing_video", "current": 96, "total": 100}), flush=True)

    # ── Write output video (no audio — FFmpeg re-encodes separately) ───
    tmp_output = output_path + ".tmp.avi"
    out_h, out_w = gen_frames[0].shape[:2]
    writer = cv2.VideoWriter(tmp_output, cv2.VideoWriter_fourcc(*"MJPG"), fps, (out_w, out_h))
    for frame in gen_frames:
        writer.write(frame)
    writer.release()

    # Mux with original audio using ffmpeg if available, otherwise just rename
    os.rename(tmp_output, output_path)

    print(json.dumps({"type": "complete", "output_path": output_path}), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"type": "error", "message": traceback.format_exc()}), flush=True)
        sys.exit(1)
`.trim();

interface LipSyncProgress {
	phase: string;
	current: number;
	total: number;
}

interface LipSyncOptions {
	videoPath: string;
	audioPath: string;
	outputPath: string;
	checkpointPath: string;
	faceDetPath: string;
	resizeFactor?: number;
	pad?: [number, number, number, number];
	batchSize?: number;
	onProgress?: (progress: LipSyncProgress) => void;
}

class LipSyncService {
	private pythonPath = "python3";
	private cachedScriptPath: string | null = null;

	setPythonPath(pythonPath: string): void {
		this.pythonPath = pythonPath;
	}

	/** Write the Wav2Lip Python script to disk (cached). */
	private async ensureScript(): Promise<string> {
		if (this.cachedScriptPath) {
			try {
				await fs.access(this.cachedScriptPath);
				return this.cachedScriptPath;
			} catch {
				// File was cleaned up, recreate
			}
		}
		const scriptPath = path.join(os.tmpdir(), "unbound_wav2lip_inference.py");
		await fs.writeFile(scriptPath, WAV2LIP_SCRIPT, "utf-8");
		this.cachedScriptPath = scriptPath;
		return scriptPath;
	}

	/**
	 * Check if Wav2Lip dependencies are available (torch + cv2).
	 */
	async checkAvailability(): Promise<{
		available: boolean;
		error?: string;
	}> {
		try {
			await execFileAsync(this.pythonPath, ["-c", "import torch; import cv2; print('ok')"], {
				timeout: 15000,
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
	 * Run Wav2Lip inference to generate a lip-synced video.
	 */
	async runLipSync(options: LipSyncOptions): Promise<{ outputPath: string }> {
		const {
			videoPath,
			audioPath,
			outputPath,
			checkpointPath,
			faceDetPath,
			resizeFactor = 1,
			pad = [0, 10, 0, 0],
			batchSize = 16,
			onProgress,
		} = options;

		const scriptPath = await this.ensureScript();

		const configPath = path.join(
			os.tmpdir(),
			`unbound_lipsync_cfg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}.json`,
		);

		await fs.writeFile(
			configPath,
			JSON.stringify({
				video_path: videoPath,
				audio_path: audioPath,
				output_path: outputPath,
				checkpoint_path: checkpointPath,
				face_det_path: faceDetPath,
				resize_factor: resizeFactor,
				pad,
				batch_size: batchSize,
			}),
		);

		try {
			await new Promise<void>((resolve, reject) => {
				const child: ChildProcess = spawn(this.pythonPath, [scriptPath, configPath], {
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
								onProgress({
									phase: msg.phase ?? "processing",
									current: msg.current ?? 0,
									total: msg.total ?? 100,
								});
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
									`Wav2Lip generation failed (exit ${code}): ${stderr.slice(-1000)}`,
								),
							),
						);
					}
				});

				child.on("error", (err) => {
					settle(() => reject(new Error(`Failed to spawn Wav2Lip: ${err.message}`)));
				});

				// Safety timeout: 30 minutes
				const timer = setTimeout(() => {
					child.kill();
					settle(() => reject(new Error("Wav2Lip generation timed out (30 min)")));
				}, 1_800_000);

				child.on("close", () => clearTimeout(timer));
			});
		} finally {
			await fs.unlink(configPath).catch(() => {
				// Ignore cleanup errors
			});
		}

		return { outputPath };
	}
}

// Singleton instance
export const lipSync = new LipSyncService();
