"""Deterministic rendered coin video with machine-readable ground truth."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def truth_path(video_path: str | Path) -> Path:
    return Path(video_path).with_suffix(".truth.json")


def generate_synthetic(path: str | Path, *, fps: int = 30, duration_s: float = 8.0,
                       rotations: int = 8, spin_start_s: float = 1.0,
                       spin_end_s: float = 6.0, seed: int = 20261118,
                       size: int = 320) -> dict[str, Any]:
    """Render exact whole rotations with smooth, varying angular velocity.

    Compression, sensor noise and blur are applied after rendering. The truth
    sidecar is the only source of expected counts; perception never reads it.
    """
    if fps <= 0 or rotations <= 0 or not 0 <= spin_start_s < spin_end_s <= duration_s:
        raise ValueError("require 0 <= spin start < spin end <= duration and positive rotations/fps")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    total_frames = round(duration_s * fps)
    start_frame = round(spin_start_s * fps)
    end_frame = round(spin_end_s * fps)
    spin_intervals = end_frame - start_frame
    if spin_intervals < rotations * 4:
        raise ValueError("spin is too fast for the selected frame rate")
    positions = np.arange(spin_intervals, dtype=np.float64) / spin_intervals
    weights = (1.0 + 0.35 * np.sin(2 * np.pi * 3 * positions)
               + 0.20 * np.sin(2 * np.pi * 7 * positions + 0.4))
    increments = rotations * weights / weights.sum()
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    cumulative[-1] = float(rotations)
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if output.suffix.lower() == ".mp4" else "MJPG"))
    writer = cv2.VideoWriter(str(output), fourcc, fps, (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create {output}")
    rng = np.random.default_rng(seed)
    center = np.array((size // 2, size // 2), dtype=np.float64)
    radius = size * 0.235
    try:
        for frame_index in range(total_frames):
            if frame_index < start_frame:
                turn = 0.0
            elif frame_index >= end_frame:
                turn = float(rotations)
            else:
                turn = float(cumulative[frame_index - start_frame])
            phase = 2 * math.pi * turn
            cosine = math.cos(phase)
            frame = np.full((size, size, 3), (24, 29, 38), dtype=np.uint8)
            # A subtle static background makes foreground segmentation realistic.
            cv2.circle(frame, tuple(center.astype(int)), int(radius + 18), (31, 37, 48), 2)
            shift = np.array((4 * math.sin(phase * 0.23), 2 * math.sin(phase * 0.17)))
            coin_center = tuple(np.rint(center + shift).astype(int))
            width = max(1, int(round(radius * abs(cosine))))
            color = (32, 190, 238) if cosine >= 0 else (48, 142, 204)
            cv2.ellipse(frame, coin_center, (width, int(round(radius))), 0, 0, 360,
                        color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.ellipse(frame, coin_center, (max(1, int(width * 0.82)), int(radius * 0.82)),
                        0, 0, 360, (65, 105, 125), thickness=2, lineType=cv2.LINE_AA)
            # Opposite faces carry different marks, but the counter never uses them.
            if width > 22:
                mark = "H" if cosine >= 0 else "T"
                cv2.putText(frame, mark, (coin_center[0] - 15, coin_center[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.25, (35, 75, 95), 3, cv2.LINE_AA)
            frame = cv2.GaussianBlur(frame, (3, 3), 0.65)
            noise = rng.normal(0, 2.2, frame.shape).astype(np.float32)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    truth = {
        "fixture": "rendered-spinning-coin-v1",
        "seed": seed,
        "fps": fps,
        "frames": total_frames,
        "width": size,
        "height": size,
        "rotations": rotations,
        "spin_start_s": spin_start_s,
        "spin_end_s": spin_end_s,
        "spin_start_frame": start_frame,
        "spin_end_frame": end_frame,
        "angular_speed": "deterministic varying speed; normalized integrated phase",
        "noise_sigma": 2.2,
        "gaussian_blur": {"kernel": [3, 3], "sigma": 0.65},
    }
    truth_path(output).write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    return truth
