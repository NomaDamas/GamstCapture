import asyncio
import json


from gamst_capture.pipeline import process_video


def test_exact_synthetic_count_and_spin_windows(synthetic_video, tmp_path):
    report = asyncio.run(process_video(
        synthetic_video, report_path=tmp_path / "report.json",
        annotated_path=tmp_path / "annotated.mp4", vlm_from_env=False))
    stops = [event for event in report["spins"] if event["kind"] == "stop"]
    starts = [event for event in report["spins"] if event["kind"] == "start"]
    assert report["rotations"] == 8
    assert report["accuracy"]["exact_rotation_count"] is True
    assert len(starts) == len(stops) == 1
    assert abs(starts[0]["timestamp_ms"] - 1000) <= 250
    assert abs(stops[0]["timestamp_ms"] - 6000) <= 350
    assert stops[0]["rotations"] == 8
    assert report["counters"]["frames_processed"] == 240
    assert report["counters"]["gamst_signals"] >= 240
    assert report["counters"]["vlm_calls"] == 0
    assert report["vlm"]["status"] == "VLM disabled"
    assert (tmp_path / "annotated.mp4").stat().st_size > 1000
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["rpm_timeline"] == report["rpm_timeline"]


class FakeVLM:
    def __init__(self):
        self.calls = []

    def complete(self, prompt, jpeg_b64, context):
        self.calls.append((prompt, jpeg_b64, context))
        return {"text": '{"face":"uncertain","count_plausible":true}',
                "model": "deterministic-test-vlm",
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}


def test_gamst_receives_signals_and_escalates_twice_per_spin(synthetic_video, tmp_path):
    vlm = FakeVLM()
    report = asyncio.run(process_video(
        synthetic_video, report_path=tmp_path / "vlm-report.json",
        annotated_path=tmp_path / "vlm-annotated.mp4", vlm=vlm, vlm_from_env=False))
    assert report["rotations"] == 8
    assert report["counters"]["gamst_signals"] == 244  # 240 frames + 2 events + 2 VLM observations
    assert report["counters"]["vlm_calls"] == 2
    assert len(vlm.calls) == 2
    assert [call["event"] for call in report["vlm"]["calls"]] == ["start", "stop"]
    assert all(call["status"] == "ok" for call in report["vlm"]["calls"])
    assert all(call["temporal_context"]["packet_checksum"] for call in report["vlm"]["calls"])
    assert all(context["recent_events"] for _, _, context in vlm.calls)
    assert report["counters"]["total_tokens"] == 220
    assert len(report["gamst"]["policy_decisions"]) == 2
