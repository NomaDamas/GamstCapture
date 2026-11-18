"""CPU-only coin segmentation and hysteretic rotation counting."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any

import cv2
import numpy as np


@dataclass
class Detection:
    detected: bool
    axis_ratio: float = 0.0
    brightness: float = 0.0
    center: tuple[float, float] | None = None
    axes: tuple[float, float] | None = None
    angle: float = 0.0
    contour: np.ndarray | None = field(default=None, repr=False)


class CoinDetector:
    """Track the largest saturated, coin-like connected component.

    The foreground contract is deliberately simple: a colored coin on a darker,
    less saturated background. No model, network, or calibration is required.
    """

    def __init__(self, min_area: float = 24.0):
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> Detection:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((5, 65, 55), np.uint8),
                           np.array((45, 255, 255), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Detection(False)
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < self.min_area:
            return Detection(False)
        if len(contour) >= 5:
            (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
            major = max(float(axis_a), float(axis_b))
            minor = min(float(axis_a), float(axis_b))
        else:
            (cx, cy), (axis_a, axis_b), angle = cv2.minAreaRect(contour)
            major = max(float(axis_a), float(axis_b))
            minor = min(float(axis_a), float(axis_b))
        if major < 2.0:
            return Detection(False)
        ratio = float(np.clip(minor / major, 0.0, 1.0))
        component_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)
        brightness = float(cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), component_mask)[0])
        return Detection(True, ratio, brightness, (float(cx), float(cy)),
                         (minor, major), float(angle), contour)


@dataclass
class SpinEvent:
    kind: str
    spin_id: int
    frame_index: int
    timestamp_ms: int
    confirmation_frame_index: int
    confirmation_timestamp_ms: int
    rotations: int = 0
    duration_ms: int = 0
    mean_rpm: float = 0.0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "spin_id": self.spin_id,
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "confirmation_frame_index": self.confirmation_frame_index,
            "confirmation_timestamp_ms": self.confirmation_timestamp_ms,
            "rotations": self.rotations,
            "duration_ms": self.duration_ms,
            "mean_rpm": round(self.mean_rpm, 4),
            "reason": self.reason,
        }


@dataclass
class FrameMeasurement:
    frame_index: int
    timestamp_ms: int
    detected: bool
    axis_ratio: float
    brightness: float
    motion: float
    spinning: bool
    rotations: int
    rpm: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "detected": self.detected,
            "axis_ratio": round(self.axis_ratio, 6),
            "brightness": round(self.brightness, 4),
            "motion": round(self.motion, 6),
            "spinning": self.spinning,
            "rotations": self.rotations,
            "rpm": round(self.rpm, 4),
        }


class RotationCounter:
    """Count edge-on transitions; two half-turn transitions make one rotation.

    Independent high/low thresholds provide hysteresis, and a minimum interval
    rejects compression/noise-induced duplicate edges. Spin boundaries use a
    two-frame motion onset and a sustained quiet interval.
    """

    def __init__(self, *, edge_enter: float = 0.55, edge_exit: float = 0.72,
                 min_half_period_s: float = 0.08, motion_enter: float = 0.75,
                 motion_exit: float = 0.20, stop_quiet_s: float = 0.34):
        self.edge_enter = edge_enter
        self.edge_exit = edge_exit
        self.min_half_period_s = min_half_period_s
        self.motion_enter = motion_enter
        self.motion_exit = motion_exit
        self.stop_quiet_s = stop_quiet_s
        self.ratios: deque[float] = deque(maxlen=3)
        self.previous_ratio: float | None = None
        self.previous_time: float | None = None
        self.motion_run = 0
        self.motion_candidate_start: float | None = None
        self.spinning = False
        self.spin_id = 0
        self.spin_start = 0.0
        self.spin_start_frame = 0
        self.in_edge = False
        self.edge_count = 0
        self.last_edge_time: float | None = None
        self.last_motion_time = 0.0
        self.current_rpm = 0.0
        self.completed_rotations = 0
        self.spins: list[SpinEvent] = []
        self._last_frame_index = 0
        self._last_timestamp_ms = 0

    @property
    def rotations(self) -> int:
        return self.completed_rotations + (self.edge_count // 2 if self.spinning else 0)

    def update(self, detection: Detection, frame_index: int, timestamp_ms: int) -> tuple[FrameMeasurement, list[SpinEvent]]:
        now = timestamp_ms / 1000.0
        self._last_frame_index = frame_index
        self._last_timestamp_ms = timestamp_ms
        events: list[SpinEvent] = []
        motion = 0.0
        if detection.detected:
            self.ratios.append(detection.axis_ratio)
            ratio = float(np.median(self.ratios))
        else:
            self.ratios.clear()
            ratio = 0.0
        dt = 0.0 if self.previous_time is None else max(0.0, now - self.previous_time)
        if detection.detected and self.previous_ratio is not None and dt > 0:
            derivative = abs(ratio - self.previous_ratio) / dt
            motion = float(np.clip(derivative / 4.0, 0.0, 1.0))
            moving = derivative >= self.motion_enter and abs(ratio - self.previous_ratio) >= 0.015
            if moving:
                if self.motion_run == 0:
                    self.motion_candidate_start = self.previous_time
                self.motion_run += 1
            else:
                self.motion_run = 0
                self.motion_candidate_start = None
            if not self.spinning and self.motion_run >= 2:
                self.spinning = True
                self.spin_id += 1
                self.spin_start = self.motion_candidate_start if self.motion_candidate_start is not None else self.previous_time
                self.spin_start_frame = max(0, frame_index - self.motion_run)
                self.in_edge = ratio <= self.edge_enter
                self.edge_count = 0
                self.last_edge_time = None
                self.current_rpm = 0.0
                self.last_motion_time = now
                events.append(SpinEvent("start", self.spin_id, self.spin_start_frame,
                                        round(self.spin_start * 1000), frame_index,
                                        timestamp_ms, reason="sustained_axis_ratio_motion"))
            if self.spinning:
                if derivative >= self.motion_exit:
                    self.last_motion_time = now
                if not self.in_edge and ratio <= self.edge_enter:
                    self.in_edge = True
                    if self.last_edge_time is None or now - self.last_edge_time >= self.min_half_period_s:
                        if self.last_edge_time is not None:
                            half_period = now - self.last_edge_time
                            self.current_rpm = 30.0 / half_period
                        self.last_edge_time = now
                        self.edge_count += 1
                elif self.in_edge and ratio >= self.edge_exit:
                    self.in_edge = False
                if now - self.last_motion_time >= self.stop_quiet_s:
                    events.append(self._stop(frame_index, timestamp_ms,
                                             round(self.last_motion_time * 1000), "sustained_quiet"))
        elif self.spinning:
            events.append(self._stop(frame_index, timestamp_ms, self._last_timestamp_ms,
                                     "coin_tracking_lost"))
        if detection.detected:
            self.previous_ratio = ratio
            self.previous_time = now
        else:
            self.previous_ratio = None
            self.previous_time = None
            self.motion_run = 0
        measurement = FrameMeasurement(frame_index, timestamp_ms, detection.detected,
                                       ratio, detection.brightness, motion, self.spinning,
                                       self.rotations, self.current_rpm if self.spinning else 0.0)
        return measurement, events

    def _stop(self, confirmation_frame: int, confirmation_ms: int, stop_ms: int,
              reason: str) -> SpinEvent:
        count = self.edge_count // 2
        duration_ms = max(0, stop_ms - round(self.spin_start * 1000))
        mean_rpm = count * 60000.0 / duration_ms if duration_ms > 0 else 0.0
        event = SpinEvent("stop", self.spin_id, confirmation_frame, stop_ms,
                          confirmation_frame, confirmation_ms, count, duration_ms,
                          mean_rpm, reason)
        self.completed_rotations += count
        self.spinning = False
        self.current_rpm = 0.0
        self.edge_count = 0
        self.last_edge_time = None
        self.in_edge = False
        self.motion_run = 0
        self.spins.append(event)
        return event

    def finish(self) -> list[SpinEvent]:
        if not self.spinning:
            return []
        return [self._stop(self._last_frame_index, self._last_timestamp_ms,
                           self._last_timestamp_ms, "end_of_video")]
