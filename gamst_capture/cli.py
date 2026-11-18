"""Command-line entry point for the GAMST coin-spin halo."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .pipeline import process_video
from .synthetic import generate_synthetic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gamst-capture", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic = subparsers.add_parser("generate-synthetic", help="render a deterministic test video")
    synthetic.add_argument("--output", default="outputs/synthetic_coin.mp4")
    synthetic.add_argument("--fps", type=int, default=30)
    synthetic.add_argument("--rotations", type=int, default=8)
    synthetic.add_argument("--seed", type=int, default=20261118)
    run = subparsers.add_parser("run", help="count rotations in a video or webcam")
    run.add_argument("source", nargs="?", help="input video path; omit when using --webcam")
    run.add_argument("--webcam", type=int, help="webcam device index (q or Ctrl-C to stop)")
    run.add_argument("--report", default="outputs/report.json")
    run.add_argument("--annotated-video", default="outputs/annotated.mp4")
    run.add_argument("--display", action="store_true", help="show annotated frames while processing")
    run.add_argument("--max-frames", type=int, help="optional capture/processing bound")
    run.add_argument("--no-vlm", action="store_true", help="ignore VLM environment configuration")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate-synthetic":
        truth = generate_synthetic(args.output, fps=args.fps, rotations=args.rotations, seed=args.seed)
        print(f"Synthetic video: {args.output}")
        print(f"Ground truth: {truth['rotations']} rotations, {truth['frames']} frames, "
              f"spin {truth['spin_start_s']:.2f}s–{truth['spin_end_s']:.2f}s")
        return 0
    if (args.source is None) == (args.webcam is None):
        raise SystemExit("provide exactly one video source or --webcam INDEX")
    source: str | int = args.webcam if args.webcam is not None else args.source
    report = asyncio.run(process_video(
        source, report_path=args.report, annotated_path=args.annotated_video,
        vlm_from_env=not args.no_vlm, display=args.display, max_frames=args.max_frames))
    counters = report["counters"]
    print(f"Rotations: {report['rotations']}")
    print(f"Mean RPM: {report['mean_rpm']:.2f}")
    print(f"Frames processed: {counters['frames_processed']}")
    print(f"GAMST signals: {counters['gamst_signals']}")
    suffix = "" if report["vlm"]["enabled"] else " (VLM disabled)"
    print(f"VLM calls: {counters['vlm_calls']}{suffix}")
    if counters["total_tokens"]:
        print(f"VLM tokens: {counters['total_tokens']}")
    if report.get("accuracy"):
        print(f"Exact synthetic count: {report['accuracy']['exact_rotation_count']}")
    print(f"Report: {Path(args.report)}")
    print(f"Annotated video: {Path(args.annotated_video)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
