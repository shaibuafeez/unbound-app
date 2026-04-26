import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { EventEmitter } from "node:events";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("electron", () => ({
	app: {
		getPath: () => "/tmp",
		setPath: () => undefined,
		isReady: () => true,
	},
	BrowserWindow: {
		getAllWindows: () => [],
	},
}));

import { setNativeCaptureOutputBuffer, setWindowsCaptureOutputBuffer } from "../state";
import { waitForNativeCaptureStart } from "./mac";
import { waitForWindowsCaptureStart } from "./windows";

function createMockChildProcess() {
	const proc = new EventEmitter() as ChildProcessWithoutNullStreams;
	proc.stdout = new EventEmitter() as ChildProcessWithoutNullStreams["stdout"];
	proc.stderr = new EventEmitter() as ChildProcessWithoutNullStreams["stderr"];
	return proc;
}

afterEach(() => {
	setNativeCaptureOutputBuffer("");
	setWindowsCaptureOutputBuffer("");
});

describe("native capture start waiters", () => {
	it("resolves immediately when macOS startup output was buffered before waiting", async () => {
		setNativeCaptureOutputBuffer("Recording started\n");

		await expect(waitForNativeCaptureStart(createMockChildProcess())).resolves.toBeUndefined();
	});

	it("resolves immediately when Windows startup output was buffered before waiting", async () => {
		setWindowsCaptureOutputBuffer("Recording started\n");

		await expect(waitForWindowsCaptureStart(createMockChildProcess())).resolves.toBeUndefined();
	});
});
