import { describe, expect, it } from "vitest";
import { DEFAULT_CROP_REGION } from "@/components/video-editor/types";
import { ModernVideoExporter } from "./modernVideoExporter";
import { VideoExporter } from "./videoExporter";

function createBaseConfig() {
	return {
		width: 1280,
		height: 720,
		frameRate: 30,
		bitrate: 8_000_000,
		videoUrl: "file:///tmp/source.mp4",
		wallpaper: "",
		zoomRegions: [],
		showShadow: false,
		shadowIntensity: 0,
		backgroundBlur: 0,
		cropRegion: DEFAULT_CROP_REGION,
		audioRegions: [],
		sourceAudioFallbackPaths: ["/tmp/dub.mp3"],
	};
}

describe("native audio plan source precedence", () => {
	it("prefers dubbed fallback audio for legacy export when requested", () => {
		const exporter = new VideoExporter({
			...createBaseConfig(),
			preferSourceAudioFallback: true,
		}) as any;
		exporter.getNativeVideoSourcePath = () => "/tmp/source.mp4";

		const plan = exporter.buildNativeAudioPlan({
			hasAudio: true,
			duration: 10,
			streamDuration: 10,
		});

		expect(plan).toEqual({
			audioMode: "copy-source",
			audioSourcePath: "/tmp/dub.mp3",
		});
	});

	it("keeps embedded audio for legacy export when no override is requested", () => {
		const exporter = new VideoExporter({
			...createBaseConfig(),
			preferSourceAudioFallback: false,
		}) as any;
		exporter.getNativeVideoSourcePath = () => "/tmp/source.mp4";

		const plan = exporter.buildNativeAudioPlan({
			hasAudio: true,
			duration: 10,
			streamDuration: 10,
		});

		expect(plan).toEqual({
			audioMode: "copy-source",
			audioSourcePath: "/tmp/source.mp4",
		});
	});

	it("prefers dubbed fallback audio for modern export when requested", () => {
		const exporter = new ModernVideoExporter({
			...createBaseConfig(),
			preferSourceAudioFallback: true,
		}) as any;
		exporter.getNativeVideoSourcePath = () => "/tmp/source.mp4";

		const plan = exporter.buildNativeAudioPlan({
			hasAudio: true,
			duration: 10,
			streamDuration: 10,
		});

		expect(plan).toEqual({
			audioMode: "copy-source",
			audioSourcePath: "/tmp/dub.mp3",
		});
	});
});
