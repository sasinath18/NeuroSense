import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import test from "node:test";

const htmlPath = fileURLToPath(new URL("../index.html", import.meta.url));
const html = readFileSync(htmlPath, "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/gi)];
assert.equal(scripts.length, 1, "Expected one inline application script.");

function sourceBetween(script, startMarker, endMarker) {
    const start = script.indexOf(startMarker);
    const end = script.indexOf(endMarker, start);
    assert.ok(start >= 0, `Missing source marker: ${startMarker}`);
    assert.ok(end > start, `Missing end marker: ${endMarker}`);
    return script.slice(start, end).trim();
}

const script = scripts[0][1];
const recordFunction = sourceBetween(
    script,
    "function recordFaceStressSample(",
    "function buildFaceStressFeatureWindows("
);
const buildFunction = sourceBetween(
    script,
    "function buildFaceStressFeatureWindows(",
    "function beginFaceOnlyScan("
);

function createContext(overrides = {}) {
    const context = {
        FACE_STRESS_SAMPLE_INTERVAL_MS: 200,
        FACE_STRESS_WINDOW_SECONDS: 20,
        FACE_STRESS_STEP_SECONDS: 5,
        FACE_STRESS_MIN_WINDOW_COVERAGE: 0.60,
        faceOnlyScanActive: true,
        faceOnlyScanStartedAt: 5_000,
        faceStressLastSampleAt: 0,
        faceStressSamples: [],
        ...overrides,
    };
    vm.createContext(context);
    vm.runInContext(`${recordFunction}\n${buildFunction}`, context);
    return context;
}

function sampleAt(timestamp, score) {
    return {
        timestamp,
        scores: { jawOpen: score },
    };
}

test("FaceScan blendshape sampler enforces a 5 Hz maximum sampling rate", () => {
    const context = createContext();
    const result = {
        faceBlendshapes: [{
            categories: [{ categoryName: "jawOpen", score: 0.25 }],
        }],
    };

    assert.equal(context.recordFaceStressSample(result, 5_000), true);
    assert.equal(context.recordFaceStressSample(result, 5_100), false);
    assert.equal(context.recordFaceStressSample(result, 5_199), false);
    assert.equal(context.recordFaceStressSample(result, 5_200), true);
    assert.equal(context.faceStressSamples.length, 2);
    assert.equal(context.faceStressSamples[0].timestamp, 0);
    assert.equal(context.faceStressSamples[1].timestamp, 0.2);
});

test("FaceScan sampler ignores missing blendshapes and non-finite scores", () => {
    const context = createContext();
    assert.equal(context.recordFaceStressSample({}, 5_000), false);
    assert.equal(context.recordFaceStressSample({
        faceBlendshapes: [{
            categories: [{ categoryName: "jawOpen", score: Number.NaN }],
        }],
    }, 5_000), false);
    assert.equal(context.faceStressSamples.length, 0);
});

test("120-second scan creates 21 aligned 20-second windows at a 5-second step", () => {
    const context = createContext();
    const samples = Array.from({ length: 600 }, (_, index) =>
        sampleAt(index * 0.2, (index % 10) / 10)
    );

    const windows = context.buildFaceStressFeatureWindows(samples, 120);

    assert.equal(windows.length, 21);
    assert.equal(windows[0].startSeconds, 0);
    assert.equal(windows[1].startSeconds, 5);
    assert.equal(windows[20].startSeconds, 100);
    assert.equal(windows[20].endSeconds, 120);
    assert.equal(windows[0].sampleCount, 100);
    assert.ok(Math.abs(
        windows[0].features.blendshape_jawOpen_mean - 0.45
    ) < 1e-12);
    assert.ok(Math.abs(
        windows[0].features.blendshape_jawOpen_std - Math.sqrt(0.0825)
    ) < 1e-12);
});

test("20-second window requires at least 60% of the expected 5 Hz samples", () => {
    const context = createContext();
    const fewerThanMinimum = Array.from({ length: 59 }, (_, index) =>
        sampleAt(index * 0.2, 0.5)
    );
    const minimumCoverage = Array.from({ length: 60 }, (_, index) =>
        sampleAt(index * 0.2, 0.5)
    );

    assert.equal(
        context.buildFaceStressFeatureWindows(fewerThanMinimum, 20).length,
        0
    );
    const windows = context.buildFaceStressFeatureWindows(minimumCoverage, 20);
    assert.equal(windows.length, 1);
    assert.equal(windows[0].sampleCount, 60);
});

test("feature windows are unavailable when scan duration is below 20 seconds", () => {
    const context = createContext();
    const samples = Array.from({ length: 100 }, (_, index) =>
        sampleAt(index * 0.2, 0.5)
    );

    assert.equal(context.buildFaceStressFeatureWindows(samples, 19.99).length, 0);
});
