"""Real-capture verification for GAMST's temporal capabilities (count, jump, rhythm, zone + memory).

Each mode runs a local light model on every frame of a real recording, feeds the structured result to the
matching GAMST provider, replays the provider's events through a GAMST session and answers the memory
questions over MCP, then compares everything with a human/instrument ground-truth sidecar. The same pipeline
also runs on 1 Hz samples of the same recording (what a 1 FPS VLM observer would see).

Models are never downloaded implicitly: run `gamst-capture fetch-models` once (pinned URLs, sha256-checked).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODELS = {
    "pose": ("pose_landmarker_lite.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
             "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"),
    "face": ("blaze_face_short_range.tflite",
             "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
             "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"),
    "person": ("efficientdet_lite0.tflite",
               "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite",
               "4b59100025bea1235a84c1038879a6cccc9f6c49f5e41144e91e74d99e780993"),
}
NOSE, L_HIP, R_HIP, L_HEEL, R_HEEL, L_TOE, R_TOE = 0, 23, 24, 29, 30, 31, 32
NOSE_HEIGHT_FRACTION = 0.93  # nose height / stature for an upright adult (anthropometric mean)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_models(directory: str) -> dict:
    """Explicit, pinned download of the three MediaPipe light models."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, (name, url, digest) in MODELS.items():
        path = root / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        actual = sha256(path)
        if actual != digest:
            path.unlink()
            raise RuntimeError(f"{name}: sha256 {actual} does not match pinned {digest}")
        out[key] = str(path)
    return out


def model_path(directory: str, key: str) -> str:
    path = Path(directory) / MODELS[key][0]
    if not path.exists() or sha256(path) != MODELS[key][2]:
        raise SystemExit(f"missing or unverified model {path}; run `gamst-capture fetch-models --models {directory}`")
    return str(path)


def frames(video: str):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        yield i, int(round(i * 1000 / fps)), fps, bgr
        i += 1


def _gamst():
    try:
        from gamst import Gamst, SensorMetadata
        from gamst.mcp import MCPServer
        from gamst.plugins import InputFrame
    except ImportError as exc:  # pragma: no cover - environment guidance
        raise SystemExit("GAMST is not importable; export PYTHONPATH=/path/to/catch-your-mind") from exc
    return Gamst, SensorMetadata, MCPServer, InputFrame


class Lane:
    """One provider fed at a fixed decimation (1 = native, round(fps) = 1 Hz control)."""

    def __init__(self, provider, step: int, modality: str, epoch: str):
        self.provider, self.step, self.modality, self.epoch = provider, step, modality, epoch
        self.seq, self.outputs = 0, []

    def _sensor(self):
        _, SensorMetadata, _, _ = _gamst()
        sensor = SensorMetadata("gamst-capture", "verify", self.seq, stream_epoch=self.epoch)
        self.seq += 1
        return sensor

    def feed(self, index: int, ts: int, data: dict):
        if index % self.step:
            return None
        _, _, _, InputFrame = _gamst()
        sensor = self._sensor()
        out = self.provider.process(InputFrame(data, self.modality, ts, sensor=sensor))
        self.outputs.append((ts, sensor, out))
        return out

    def finish(self):
        """End of stream for providers that confirm their last event late (repetition counting)."""
        if not hasattr(self.provider, "finish"):
            return None
        final = self.provider.finish()
        ts = self.outputs[-1][0] + 1 if self.outputs else 0
        self.outputs.append((ts, self._sensor(), final))
        return final

    def events(self):
        return [e for _, _, out in self.outputs for e in out["events"]]


async def memory_answers(lane: Lane, event_name: str) -> dict:
    """Replay the lane's evented observations into a GAMST session and answer from MCP memory only."""
    Gamst, _, MCPServer, _ = _gamst()
    base = int(time.time() * 1000) - (lane.outputs[-1][0] if lane.outputs else 0) - 5000
    async with Gamst(consent=True) as human:
        human.register_capability(lane.provider.spec)
        for ts, sensor, out in lane.outputs:
            if out["events"]:
                await human.push_observation(lane.provider.spec.id, out, sensor=sensor, timestamp_ms=base + ts)
        page = await MCPServer(human).call_tool("get_observation_page",
                                                {"capability_id": lane.provider.spec.id, "events_only": True, "limit": 100})
    events = []
    for record in page.get("items", []):
        value = (record.get("observation") or {}).get("value") or {}
        events += [e for e in value.get("events", []) if e.get("event") == event_name]
    last = max((e.get("landing_ms") or e.get("timestamp_ms") or 0 for e in events), default=None)
    return {"answered_from": "mcp:get_observation_page(events_only)", "events": len(events),
            "last_event_s": None if last is None else round(last / 1000, 3),
            "page_truncated": bool(page.get("truncated")), "page_bytes": len(json.dumps(page))}


def _landmarks(results) -> list[list[float]] | None:
    if not results.pose_landmarks:
        return None
    return [[p.x, p.y] for p in results.pose_landmarks[0]]


def pose_stream(video: str, models: str):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path(models, "pose")),
                                        running_mode=vision.RunningMode.VIDEO)
    with vision.PoseLandmarker.create_from_options(opts) as pose:
        last = None
        for i, ts, fps, bgr in frames(video):
            res = pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                 data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), ts)
            lm = _landmarks(res) or last
            last = lm
            yield i, ts, fps, bgr.shape, lm


def run_count(video: str, truth: dict, models: str) -> dict:
    from gamst.providers.repetition import RepetitionCounter
    lanes = {}
    for i, ts, fps, _, lm in pose_stream(video, models):
        if not lanes:
            step1 = max(1, int(round(fps)))
            lanes = {"native": Lane(RepetitionCounter(channels=66), 1, "video", "native"),
                     "1fps": Lane(RepetitionCounter(channels=66, max_gap_ms=int(1000 * step1 / fps) + 200),
                                  step1, "video", "1fps")}
        if lm is None:
            continue
        data = {"channels": [v for p in lm for v in p]}
        for lane in lanes.values():
            lane.feed(i, ts, data)
    gt = truth.get("count")
    report = {}
    for name, lane in lanes.items():
        final = lane.finish()
        report[name] = {"count": final["count"], "period_s": final.get("period_s")}
        if gt is not None:
            report[name]["error"] = final["count"] - gt
            report[name]["within_1"] = abs(final["count"] - gt) <= 1
    report["memory"] = asyncio.run(memory_answers(lanes["native"], "repetition"))
    return {"mode": "count", "truth": gt, **report}


def run_jump(video: str, truth: dict, models: str) -> dict:
    from gamst.providers.flight import FlightPhysics
    stature = truth.get("subject_height_m")
    if not stature:
        raise SystemExit("jump mode needs subject_height_m in the truth sidecar (metric scale for clearance)")
    scale, samples, lanes = None, [], {}
    for i, ts, fps, shape, lm in pose_stream(video, models):
        if lm is None:
            continue
        h = shape[0]
        foot_y = max(lm[k][1] for k in (L_HEEL, R_HEEL, L_TOE, R_TOE)) * h
        hip_y = (lm[L_HIP][1] + lm[R_HIP][1]) / 2 * h
        if scale is None:
            samples.append((foot_y - lm[NOSE][1] * h))
            if ts < 1000:
                continue
            scale = stature * NOSE_HEIGHT_FRACTION / statistics.median(samples)  # metres per pixel, standing still
            step1 = max(1, int(round(fps)))
            lanes = {"native": Lane(FlightPhysics(max_gap_ms=int(3000 / fps) + 50), 1, "video", "native"),
                     "1fps": Lane(FlightPhysics(max_gap_ms=int(1000 * step1 / fps) + 200, baseline_s=1.0), step1, "video", "1fps")}
        data = {"foot_height_m": (h - foot_y) * scale, "body_height_m": (h - hip_y) * scale}
        for lane in lanes.values():
            lane.feed(i, ts, data)
    gt = [j["height_m"] for j in truth.get("jumps", [])]
    report = {"metres_per_pixel": scale}
    for name, lane in lanes.items():
        jumps = [e for e in lane.events() if e["event"] == "jump"]
        rows = [{"height_flight_m": e["height_flight_m"], "rise_m": e.get("rise_m"), "flight_s": e["flight_s"],
                 "timing_method": e["timing_method"], "fitted_gravity_m_s2": e.get("fitted_gravity_m_s2")} for e in jumps]
        for k, row in enumerate(rows):
            if k < len(gt):
                row["truth_m"] = gt[k]
                row["error_flight_m"] = round(row["height_flight_m"] - gt[k], 4)
        report[name] = {"jumps_detected": len(rows), "jumps_truth": len(gt), "jumps": rows}
    report["memory"] = asyncio.run(memory_answers(lanes["native"], "jump")) if lanes else None
    return {"mode": "jump", **report}


def run_rhythm(video: str, truth: dict, models: str) -> dict:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from gamst.providers.rhythm import RhythmEstimator
    det = vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=model_path(models, "face"))))
    roi, lanes = None, {}
    for i, ts, fps, bgr in frames(video):
        if not lanes:
            step1 = max(1, int(round(fps)))
            lanes = {"native": Lane(RhythmEstimator(band="heart", input_kind="rgb"), 1, "video", "native"),
                     "1fps": Lane(RhythmEstimator(band="heart", input_kind="rgb", max_gap_ms=int(1000 * step1 / fps) + 200),
                                  step1, "video", "1fps")}
        if roi is None or i % 30 == 0:
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            if res.detections:
                bb = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height).bounding_box
                roi = (max(0, bb.origin_x + bb.width // 10), max(0, bb.origin_y + bb.height // 10),
                       int(bb.width * 0.8), int(bb.height * 0.8))
        if roi is None:
            continue
        x, y, w, h = roi
        patch = bgr[y:y + h, x:x + w]
        if patch.size == 0:
            continue
        b, g, r = patch.reshape(-1, 3).mean(axis=0)
        for lane in lanes.values():
            lane.feed(i, ts, {"rgb": [float(r) + 1e-3, float(g) + 1e-3, float(b) + 1e-3]})
    gt = truth.get("heart_bpm")
    report = {}
    for name, lane in lanes.items():
        rates = [out["per_minute"] for _, _, out in lane.outputs if out.get("status") == "measuring"]
        statuses = sorted({out["status"] for _, _, out in lane.outputs})
        bpm = statistics.median(rates) if rates else None
        report[name] = {"heart_bpm": None if bpm is None else round(bpm, 1), "statuses": statuses}
        if gt is not None and bpm is not None:
            report[name]["error_bpm"] = round(bpm - gt, 1)
            report[name]["within_5"] = abs(bpm - gt) <= 5
    return {"mode": "rhythm", "truth_bpm": gt, **report}


def _loudness_track(video: str, fps: float, n_frames: int) -> list[float]:
    """Per-frame loudness score in [0, 1] from the recording's own audio (-60 dBFS -> 0, -10 dBFS -> 1)."""
    if shutil.which("ffmpeg") is None:
        return [0.0] * n_frames
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True).stdout
    x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    hop = 16000 / fps
    out = []
    for k in range(n_frames):
        seg = x[int(k * hop):int((k + 1) * hop)]
        db = 20 * math.log10(float(np.sqrt(np.mean(seg ** 2))) + 1e-9) if seg.size else -100.0
        out.append(min(1.0, max(0.0, (db + 60) / 50)))
    return out


def run_zone(video: str, truth: dict, models: str, rule: dict) -> dict:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from gamst.providers.risk_field import RiskField
    det = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path(models, "person")), score_threshold=0.4,
        category_allowlist=["person"]))
    cap = cv2.VideoCapture(video)
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    loud = _loudness_track(video, fps, n)
    step1 = max(1, int(round(fps)))
    lanes = {"native": Lane(RiskField(rule), 1, "video", "native"),
             "1fps": Lane(RiskField(rule, max_gap_ms=int(1000 * step1 / fps) + 500), step1, "video", "1fps")}
    for i, ts, _, bgr in frames(video):
        h, w = bgr.shape[:2]
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        points = [[(d.bounding_box.origin_x + d.bounding_box.width / 2) / w,
                   (d.bounding_box.origin_y + d.bounding_box.height) / h] for d in res.detections][:64]
        data = {"points": points, "signals": {"audio.loudness": loud[i] if i < len(loud) else 0.0}}
        for lane in lanes.values():
            lane.feed(i, ts, data)
    truth_alarms = truth.get("alarms", [])
    report = {}
    for name, lane in lanes.items():
        intervals, start = [], None
        for e in lane.events():
            if e["event"] == "alarm_on":
                start = e["timestamp_ms"] / 1000
            elif e["event"] == "alarm_off" and start is not None:
                intervals.append([start, e["timestamp_ms"] / 1000])
                start = None
        if start is not None:
            intervals.append([start, None])
        hits = sum(any(s - 1.0 <= a[0] <= e for a in intervals) for s, e in truth_alarms)
        report[name] = {"alarms": intervals, "truth_alarms": len(truth_alarms), "detected_within_1s": hits,
                        "false_alarms": sum(not any(s - 1.0 <= a[0] <= e for s, e in truth_alarms) for a in intervals)}
    report["memory"] = asyncio.run(memory_answers(lanes["native"], "alarm_on"))
    return {"mode": "zone", "rule_id": rule.get("id"), **report}


def verify(mode: str, video: str, truth_path: str | None, models: str, rule_path: str | None) -> dict[str, Any]:
    truth = json.loads(Path(truth_path).read_text()) if truth_path else {}
    started = time.time()
    if mode == "count":
        result = run_count(video, truth, models)
    elif mode == "jump":
        result = run_jump(video, truth, models)
    elif mode == "rhythm":
        result = run_rhythm(video, truth, models)
    elif mode == "zone":
        if not rule_path:
            raise SystemExit("zone mode needs --rule RULE.json")
        result = run_zone(video, truth, models, json.loads(Path(rule_path).read_text()))
    else:
        raise SystemExit(f"unknown mode {mode}")
    result.update(video=str(video), video_sha256=sha256(Path(video)), wall_s=round(time.time() - started, 2))
    return result
