"""Real GAMST ingestion, temporal context, policy decisions and optional VLM."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import asdict
from typing import Any
from urllib import request
import uuid

import cv2
import numpy as np

from .perception import FrameMeasurement, SpinEvent

REPLAY_LAG_MS = 6 * 60 * 60 * 1000


class OpenAIVLM:
    """Minimal OpenAI-compatible chat-completions client; no SDK dependency."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/chat/completions"):
            self.base_url += "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls) -> "OpenAIVLM | None":
        base_url = os.getenv("GAMSTCAPTURE_VLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        model = os.getenv("GAMSTCAPTURE_VLM_MODEL") or os.getenv("OPENAI_MODEL")
        api_key = os.getenv("GAMSTCAPTURE_VLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not base_url and not model:
            return None
        if not base_url or not model:
            raise ValueError("set both GAMSTCAPTURE_VLM_BASE_URL and GAMSTCAPTURE_VLM_MODEL")
        return cls(base_url, model, api_key)

    def complete(self, prompt: str, jpeg_b64: str, context: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 300,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt + "\nBounded GAMST context:\n" + json.dumps(context)},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + jpeg_b64}},
                ],
            }],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = request.Request(self.base_url, data=json.dumps(payload).encode("utf-8"),
                              headers=headers, method="POST")
        with request.urlopen(req, timeout=self.timeout_s) as response:
            result = json.loads(response.read().decode("utf-8"))
        usage = result.get("usage") or {}
        return {
            "text": result["choices"][0]["message"].get("content", ""),
            "model": result.get("model", self.model),
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            },
        }


class GamstCoinSession:
    """One replay clock and one asyncio lifecycle for all GAMST operations."""

    def __init__(self, *, fps: float, vlm: Any | None = None, vlm_from_env: bool = True):
        self.fps = fps
        self.vlm = OpenAIVLM.from_env() if vlm is None and vlm_from_env else vlm
        self.origin_ms = int(time.time() * 1000) - REPLAY_LAG_MS
        self.stream_epoch = uuid.uuid4().hex
        self.human: Any = None
        self.policy: Any = None
        self.change_active = False
        self.last_dispatch_ms: int | None = None
        self.last_semantic_capture_ms: int | None = None
        self.gamst_signals = 0
        self.policy_evaluations = 0
        self.policy_decisions: list[dict[str, Any]] = []
        self.vlm_calls: list[dict[str, Any]] = []
        self.measurement_records: dict[int, str] = {}
        self.spin_start_records: dict[int, str] = {}
        self.temporal_queries = 0

    @property
    def vlm_enabled(self) -> bool:
        return self.vlm is not None

    async def __aenter__(self) -> "GamstCoinSession":
        try:
            from gamst import CapabilitySpec, Gamst, OutputField
            from gamst.call_policy import CallPolicy
        except ImportError as exc:
            raise RuntimeError(
                "GAMST is required. Set PYTHONPATH=/Users/jinminseong/Desktop/catch-your-mind "
                "or install that checkout with pip install -e.") from exc
        self.human = Gamst(consent=True, store=":memory:", retention_hours=12,
                           window_ms=30000, sensor_inference_interval_ms=5000,
                           max_records=20000)
        await self.human.__aenter__()
        common = dict(provider="gamst-capture.opencv", version="1",
                      streaming_mode="frame", freshness_ms=60000,
                      interpretation="measurement")
        self.human.register_capability(CapabilitySpec(
            id="gamst-capture.coin-rotation", output_name="coin.rotation",
            modality="video", fields=(
                OutputField("frame_index", "integer"),
                OutputField("media_time_ms", "integer", "ms"),
                OutputField("detected", "boolean"),
                OutputField("axis_ratio", "number"),
                OutputField("brightness", "number"),
                OutputField("motion", "number"),
                OutputField("spinning", "boolean"),
                OutputField("rotations", "integer", "rotations"),
                OutputField("rpm", "number", "rpm"),
                OutputField("evidence_ids", "array"),
            ), stateful=True, reset_on_gap=True, **common))
        self.human.register_capability(CapabilitySpec(
            id="gamst-capture.spin-event", output_name="coin.spin_event",
            modality="video", fields=(
                OutputField("spin_id", "integer"),
                OutputField("event", "string"),
                OutputField("event_media_time_ms", "integer", "ms"),
                OutputField("rotations", "integer", "rotations"),
                OutputField("duration_ms", "integer", "ms"),
                OutputField("mean_rpm", "number", "rpm"),
                OutputField("reason", "string"),
                OutputField("evidence_ids", "array"),
            ), **common))
        self.human.register_capability(CapabilitySpec(
            id="gamst-capture.vlm-result", output_name="coin.vlm_result",
            modality="text", provider="gamst-capture.vlm", version="1",
            fields=(OutputField("spin_id", "integer"), OutputField("event", "string"),
                    OutputField("description", "string"), OutputField("evidence_ids", "array")),
            streaming_mode="offline", freshness_ms=60000,
            interpretation="semantic_description"))
        self.policy = CallPolicy(mode="sentinel", semantic_interval_ms=3600000,
                                 cooldown_ms=0, max_age_ms=60000,
                                 change_enter=0.5, change_exit=0.25)
        self.last_semantic_capture_ms = self.origin_ms
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.human is not None:
            await self.human.__aexit__(exc_type, exc, tb)
            self.human = None

    def capture_ms(self, media_time_ms: int) -> int:
        return self.origin_ms + media_time_ms

    def _sensor(self, sequence: int):
        from gamst import SensorMetadata
        return SensorMetadata("gamst-capture", "coin-video", sequence,
                              stream_epoch=self.stream_epoch, capture_sequence=sequence,
                              clock_id="coin-media", clock_epoch=self.stream_epoch)

    async def _push(self, capability_id: str, name: str, value: dict[str, Any],
                    modality: str, capture_ms: int, sequence: int) -> dict[str, Any]:
        record = await self.human.push_signal(
            name, value, modality, timestamp_ms=capture_ms,
            model_id="gamst-capture.vlm" if capability_id.endswith("vlm-result") else "gamst-capture.opencv",
            model_version="1", quality="sufficient", capability_id=capability_id,
            sensor=self._sensor(sequence))
        self.gamst_signals += 1
        return record

    async def observe(self, measurement: FrameMeasurement, events: list[SpinEvent],
                      frame: np.ndarray) -> None:
        sequence = measurement.frame_index
        capture = self.capture_ms(measurement.timestamp_ms)
        evidence_id = f"frame-{measurement.frame_index:06d}"
        value = measurement.as_dict()
        value["media_time_ms"] = value.pop("timestamp_ms")
        value["evidence_ids"] = [evidence_id]
        record = await self._push("gamst-capture.coin-rotation", "coin.rotation", value,
                                  "video", capture, sequence)
        self.measurement_records[measurement.frame_index] = record["id"]
        event_by_kind = {event.kind: event for event in events}
        for event in events:
            evidence = [record["id"]]
            if event.kind == "start":
                self.spin_start_records[event.spin_id] = record["id"]
            elif event.spin_id in self.spin_start_records:
                evidence.insert(0, self.spin_start_records[event.spin_id])
            event_value = {
                "spin_id": event.spin_id, "event": event.kind,
                "event_media_time_ms": event.timestamp_ms, "rotations": event.rotations,
                "duration_ms": event.duration_ms, "mean_rpm": event.mean_rpm,
                "reason": event.reason, "evidence_ids": evidence,
            }
            await self._push("gamst-capture.spin-event", "coin.spin_event", event_value,
                             "video", capture, sequence)
        selected_event = event_by_kind.get("start") or event_by_kind.get("stop")
        jpeg_b64 = None
        visual_reference = None
        if selected_event is not None:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError("could not encode VLM evidence frame")
            jpeg_bytes = encoded.tobytes()
            jpeg_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            visual_reference = {
                "source_id": "gamst-capture", "sensor_id": "coin-video",
                "stream_epoch": self.stream_epoch, "capture_ms": capture,
                "sha256": hashlib.sha256(jpeg_bytes).hexdigest(),
            }
        decision = self.policy.recommend(
            record, now_ms=capture, change_score=1.0 if measurement.spinning else 0.0,
            change_active=self.change_active, last_dispatch_ms=self.last_dispatch_ms,
            last_semantic_capture_ms=self.last_semantic_capture_ms,
            query=selected_event is not None and selected_event.kind == "stop",
            needs_visual=True, visual_reference=visual_reference)
        self.change_active = decision["proposed_change_active"]
        self.policy_evaluations += 1
        if decision["desired_route"] != "none":
            item = {"frame_index": measurement.frame_index,
                    "media_time_ms": measurement.timestamp_ms,
                    "event": selected_event.kind if selected_event else None,
                    "decision": decision, "dispatch_status": "not_recommended"}
            if decision["recommended_route"] in {"small_vlm", "strong_vlm"}:
                self.last_dispatch_ms = capture
                self.last_semantic_capture_ms = capture
                if self.vlm is None:
                    item["dispatch_status"] = "vlm_disabled"
                else:
                    item["dispatch_status"] = "attempted"
                    await self._dispatch(selected_event, measurement, frame, jpeg_b64, record)
            self.policy_decisions.append(item)

    async def _compact_context(self, cutoff_ms: int) -> dict[str, Any]:
        packet = await self.human.build_temporal_context(
            query_cutoff_ms=cutoff_ms, window="1h", max_bytes=131072)
        self.temporal_queries += 1
        events = []
        for event in packet.get("events", [])[-12:]:
            events.append({key: event.get(key) for key in
                           ("event_id", "event_type", "capture_timestamp_ms", "value", "evidence_ids")})
        return {"query_cutoff_ms": cutoff_ms, "packet_checksum": packet.get("checksum"),
                "clock_epoch": self.stream_epoch, "recent_events": events}

    async def _dispatch(self, event: SpinEvent, measurement: FrameMeasurement,
                        frame: np.ndarray, jpeg_b64: str, record: dict[str, Any]) -> None:
        capture = self.capture_ms(measurement.timestamp_ms)
        context = await self._compact_context(capture)
        if event.kind == "start":
            prompt = ("A cheap OpenCV tracker reports coin spin onset. Confirm whether a coin and "
                      "plausible spin motion are visible. Be concise and do not invent a rotation count.")
        else:
            prompt = (f"A cheap tracker reports spin stop with {event.rotations} full rotations and "
                      f"mean {event.mean_rpm:.2f} RPM. Inspect the final image: respond with final face "
                      "heads/tails/uncertain and whether the numeric count is plausible. Use concise JSON; "
                      "the tracker, not the image model, owns the count.")
        call: dict[str, Any] = {
            "spin_id": event.spin_id, "event": event.kind,
            "frame_index": measurement.frame_index, "media_time_ms": measurement.timestamp_ms,
            "capture_timestamp_ms": capture, "evidence_ids": [record["id"]],
            "route": "small_vlm", "temporal_context": context,
        }
        self.vlm_calls.append(call)
        try:
            result = await __import__("asyncio").to_thread(self.vlm.complete, prompt, jpeg_b64, context)
            call["status"] = "ok"
            call["response"] = result.get("text", "")
            call["model"] = result.get("model")
            call["usage"] = result.get("usage", {})
            description = str(result.get("text", ""))[:8192]
            await self._push("gamst-capture.vlm-result", "coin.vlm_result", {
                "spin_id": event.spin_id, "event": event.kind, "description": description,
                "evidence_ids": [record["id"]],
            }, "text", capture, measurement.frame_index)
        except Exception as exc:
            call["status"] = "error"
            call["error"] = f"{type(exc).__name__}: {exc}"

    def counters(self) -> dict[str, int]:
        usage = [call.get("usage", {}) for call in self.vlm_calls]
        return {
            "gamst_signals": self.gamst_signals,
            "policy_evaluations": self.policy_evaluations,
            "vlm_calls": len(self.vlm_calls),
            "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in usage),
            "completion_tokens": sum(int(item.get("completion_tokens", 0)) for item in usage),
            "total_tokens": sum(int(item.get("total_tokens", 0)) for item in usage),
        }
