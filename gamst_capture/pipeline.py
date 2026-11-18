"""Frame-by-frame video/webcam pipeline, annotation and JSON evidence report."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from .gamst_session import GamstCoinSession
from .perception import CoinDetector, Detection, FrameMeasurement, RotationCounter, SpinEvent
from .synthetic import truth_path


def _annotate(frame: np.ndarray, detection: Detection, measurement: FrameMeasurement,
              counters: dict[str, int], vlm_enabled: bool) -> np.ndarray:
    output = frame.copy()
    if detection.detected:
        if detection.contour is not None:
            cv2.drawContours(output, [detection.contour], -1, (80, 255, 150), 2)
        if detection.center and detection.axes:
            cv2.circle(output, tuple(np.rint(detection.center).astype(int)), 3, (80, 255, 150), -1)
    height, width = output.shape[:2]
    panel = output.copy()
    cv2.rectangle(panel, (10, 10), (min(width - 10, 355), 151), (15, 20, 28), -1)
    cv2.addWeighted(panel, 0.78, output, 0.22, 0, output)
    state = "SPINNING" if measurement.spinning else "IDLE / STOPPED"
    vlm = f"VLM calls: {counters['vlm_calls']}" if vlm_enabled else "VLM disabled (0 calls)"
    lines = [
        "GAMST Coin Spin Halo",
        f"{state}  |  rotations: {measurement.rotations}",
        f"RPM: {measurement.rpm:.1f}  |  axis ratio: {measurement.axis_ratio:.3f}",
        f"Frames: {counters['frames_processed']}  |  GAMST signals: {counters['gamst_signals']}",
        vlm,
    ]
    for index, text in enumerate(lines):
        cv2.putText(output, text, (22, 34 + index * 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52 if index else 0.60, (235, 241, 248), 1, cv2.LINE_AA)
    cv2.putText(output, f"t = {measurement.timestamp_ms / 1000:.3f}s", (width - 145, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 228, 238), 1, cv2.LINE_AA)
    return output


async def process_video(source: str | int, *, report_path: str | Path,
                        annotated_path: str | Path, vlm: Any | None = None,
                        vlm_from_env: bool = True, display: bool = False,
                        max_frames: int | None = None) -> dict[str, Any]:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video/webcam source: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    report_output = Path(report_path)
    annotated_output = Path(annotated_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    annotated_output.parent.mkdir(parents=True, exist_ok=True)
    codec = "mp4v" if annotated_output.suffix.lower() == ".mp4" else "MJPG"
    writer = cv2.VideoWriter(str(annotated_output), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not create annotated video: {annotated_output}")
    detector = CoinDetector()
    counter = RotationCounter()
    measurements: list[FrameMeasurement] = []
    events: list[SpinEvent] = []
    wall_origin = time.monotonic()
    last_measurement: FrameMeasurement | None = None
    last_frame: np.ndarray | None = None
    interrupted = False
    try:
        async with GamstCoinSession(fps=fps, vlm=vlm, vlm_from_env=vlm_from_env) as session:
            try:
                while max_frames is None or len(measurements) < max_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame_index = len(measurements)
                    if isinstance(source, int):
                        media_ms = round((time.monotonic() - wall_origin) * 1000)
                    else:
                        media_ms = round(frame_index * 1000 / fps)
                    detection = detector.detect(frame)
                    measurement, new_events = counter.update(detection, frame_index, media_ms)
                    await session.observe(measurement, new_events, frame)
                    measurements.append(measurement)
                    events.extend(new_events)
                    last_measurement = measurement
                    last_frame = frame.copy()
                    counters = {"frames_processed": len(measurements), **session.counters()}
                    annotated = _annotate(frame, detection, measurement, counters, session.vlm_enabled)
                    writer.write(annotated)
                    if display:
                        cv2.imshow("GAMST Coin Spin Halo", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
            except KeyboardInterrupt:
                interrupted = True
            final_events = counter.finish()
            if final_events and last_measurement is not None and last_frame is not None:
                await session.observe(last_measurement, final_events, last_frame)
                events.extend(final_events)
            counters = {"frames_processed": len(measurements), **session.counters()}
            spin_dicts = [event.as_dict() for event in events]
            for event, data in zip(events, spin_dicts):
                data["capture_timestamp_ms"] = session.capture_ms(event.timestamp_ms)
                data["confirmation_capture_timestamp_ms"] = session.capture_ms(event.confirmation_timestamp_ms)
            stops = [event for event in events if event.kind == "stop"]
            total_spin_ms = sum(event.duration_ms for event in stops)
            mean_rpm = counter.rotations * 60000 / total_spin_ms if total_spin_ms else 0.0
            truth = None
            if not isinstance(source, int) and truth_path(source).exists():
                truth = json.loads(truth_path(source).read_text(encoding="utf-8"))
            report = {
                "schema": "gamst-capture.report.v1",
                "source": "webcam" if isinstance(source, int) else str(source),
                "fps": fps,
                "width": width,
                "height": height,
                "duration_s": round(len(measurements) / fps, 6),
                "interrupted": interrupted,
                "rotations": counter.rotations,
                "mean_rpm": round(mean_rpm, 4),
                "spins": spin_dicts,
                "frames": [measurement.as_dict() for measurement in measurements],
                "rpm_timeline": [{"frame_index": item.frame_index, "media_time_ms": item.timestamp_ms,
                                  "rpm": round(item.rpm, 4)} for item in measurements],
                "counters": counters,
                "vlm": {
                    "enabled": session.vlm_enabled,
                    "status": "enabled" if session.vlm_enabled else "VLM disabled",
                    "calls": session.vlm_calls,
                },
                "gamst": {
                    "capture_origin_ms": session.origin_ms,
                    "stream_epoch": session.stream_epoch,
                    "capabilities": ["gamst-capture.coin-rotation", "gamst-capture.spin-event",
                                     "gamst-capture.vlm-result"],
                    "temporal_queries": session.temporal_queries,
                    "policy_decisions": session.policy_decisions,
                },
                "ground_truth": truth,
            }
            if truth is not None:
                report["accuracy"] = {
                    "exact_rotation_count": counter.rotations == truth["rotations"],
                    "rotation_error": counter.rotations - truth["rotations"],
                }
    finally:
        capture.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
