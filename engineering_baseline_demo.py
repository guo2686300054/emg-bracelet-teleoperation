"""Build a reproducible synthetic three-class engineering baseline artifact."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Sequence, cast

from benchmark_emg_inference import benchmark_bundle
from data_recorder import DataRecorder
from emg_protocol import (
    AcquisitionMetadata,
    DeviceKey,
    EmgFrame,
    HandSide,
    NotificationPacketProtocol,
    QualityFlags,
    RateDescriptor,
    SignalChain,
)
from recording_context import RecordingContext
from train_emg_baseline import train_baseline
from training_contract import CANONICAL_LABELS, EMG_CHANNEL_COUNT
from training_dataset import prepare_training_dataset
from training_samples import load_training_manifest


SYNTHETIC_PROVENANCE = "synthetic_test"
DEPLOYMENT_STATUS = "engineering_only"
DISCLAIMER = (
    "Synthetic engineering validation only. Accuracy and latency from this run do not "
    "represent real-user, real-device, or dexterous-hand control performance."
)
_FLAGS = (
    QualityFlags.VALID
    | QualityFlags.HOST_WALL_TIME_VALID
    | QualityFlags.HOST_MONOTONIC_VALID
    | QualityFlags.HOST_RECEIVE_INDEX_VALID
)


def _synthetic_value(label: str, sample_index: int, subject_number: int) -> float:
    phase = sample_index / 11.0 + subject_number * 0.07
    if label == "rest":
        return 0.12 * math.sin(phase) + 0.03 * math.sin(phase * 0.37)
    if label == "fist":
        return (4.8 if sample_index % 2 else -4.8) + 0.35 * math.sin(phase)
    return (sample_index % 25) * 0.22 + 0.18 * math.sin(phase * 0.61)


def _write_synthetic_session(
    root: Path, subject_number: int, label: str, *, samples: int = 1_000
) -> Path:
    subject_id = f"sub-{subject_number:032x}"
    session_id = f"synthetic-{subject_number}-{label}"
    device = DeviceKey("dev-0123456789abcdef0123456789abcdef")
    acquisition = AcquisitionMetadata(
        sample_rate=RateDescriptor(200.0, "protocol", "synthetic-demo-contract-v1", True),
        host_observed_rate=RateDescriptor(
            200.0, "host_observed", "synthetic-demo-clock-v1", True
        ),
        signal_chain=SignalChain(sample_format="float32"),
        notification_packet_protocol=NotificationPacketProtocol(),
    )
    recorder = DataRecorder(
        root,
        subject_id=subject_id,
        device_id=device,
        acquisition=acquisition,
        channels=EMG_CHANNEL_COUNT,
        side=HandSide.LEFT,
        session_id=session_id,
        flush_every=128,
        recording_context=RecordingContext(
            subject_id,
            label,
            "hold",
            "synthetic_engineering_demo_v1",
            HandSide.LEFT,
            SYNTHETIC_PROVENANCE,
        ),
    )
    recorder.configure_session_boundary(start_host_receive_index=0, start_drop_total=0)
    try:
        for index in range(samples):
            value = _synthetic_value(label, index, subject_number)
            channels = cast(
                tuple[Real, ...],
                tuple(
                    value * (1.0 + channel * 0.025)
                    + channel * 0.035
                    + 0.01 * math.sin(index / (7.0 + channel))
                    for channel in range(EMG_CHANNEL_COUNT)
                ),
            )
            recorder.record(
                EmgFrame(
                    channels,
                    1_700_000_000_000_000_000 + index * 5_000_000,
                    10_000_000_000 + index * 5_000_000,
                    index + 1,
                    index,
                    action_label=label,
                    action_phase="hold",
                    quality_flags=_FLAGS,
                )
            )
        recorder.update_session_boundary(
            end_host_receive_index=samples,
            received_count=samples,
            eligible_count=samples,
            written_count=samples,
            queue_drop_total=0,
            queue_drop_session=0,
            tail_pending_count=0,
            tail_loss_count=0,
            incomplete_reason=None,
        )
    finally:
        recorder.close()
    return recorder.session_dir


def run_demo(output_root: str | Path, *, seed: int = 20260920) -> Path:
    """Generate synthetic data, train, evaluate, benchmark, and write a warning report."""

    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(f"demo output already exists: {destination}")
    destination.mkdir(parents=True)
    session_root = destination / "synthetic_sessions"
    sessions = [
        _write_synthetic_session(session_root, subject, label)
        for subject in range(1, 4)
        for label in CANONICAL_LABELS
    ]
    manifest = destination / "training_manifest.json"
    prepare_training_dataset(sessions, manifest, window_ms=200, step_ms=50, seed=seed)
    result = train_baseline(
        load_training_manifest(manifest, session_root),
        destination / "bundle",
        seed=seed,
    )
    latency = benchmark_bundle(result.bundle, warmup=100, iterations=2_000, seed=seed)
    report = {
        "schema": "emg.engineering_baseline.report",
        "version": "1.0",
        "warning": DISCLAIMER,
        "provenance": result.bundle.provenance,
        "deployment_status": result.bundle.deployment_status,
        "real_accuracy_claim_allowed": False,
        "live_control_eligibility": "not_evaluated",
        "deployment_gate_passed": False,
        "labels": list(CANONICAL_LABELS),
        "evaluation": result.evaluation,
        "latency_benchmark": latency,
    }
    if report["provenance"] != SYNTHETIC_PROVENANCE:
        raise RuntimeError("synthetic demo produced an unexpected provenance")
    if report["deployment_status"] != DEPLOYMENT_STATUS:
        raise RuntimeError("synthetic demo produced an unsafe deployment status")
    (destination / "engineering_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    splits = _require_mapping(result.evaluation.get("splits"), "evaluation.splits")
    test_metrics = _require_mapping(splits.get("test"), "evaluation.splits.test")
    latency_ns = _require_mapping(latency.get("latency_ns"), "latency.latency_ns")
    latency_total = _require_mapping(latency_ns.get("total"), "latency.latency_ns.total")
    test_accuracy = _require_number(test_metrics.get("accuracy"), "test accuracy")
    latency_p99 = _require_number(latency_total.get("p99"), "inference total p99")
    (destination / "README.md").write_text(
        "\n".join(
            [
                "# 三分类工程基线（合成数据）",
                "",
                f"> **警告：{DISCLAIMER}**",
                "",
                f"- provenance: `{SYNTHETIC_PROVENANCE}`",
                f"- deployment_status: `{DEPLOYMENT_STATUS}`",
                "- live_control_eligibility: `not_evaluated`",
                "- deployment_gate_passed: `false`",
                f"- test accuracy: `{test_accuracy:.6f}`（仅证明工程链路）",
                f"- confusion matrix: `{test_metrics['confusion_matrix']}`",
                f"- error windows: `{test_metrics['error_windows']}`",
                f"- inference total p99: `{latency_p99:.0f} ns`",
                "",
                "此结果不得替代真实硬件、真实佩戴和受试者隔离测试。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return destination


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be an object")
    return value


def _require_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"{name} must be numeric")
    return float(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args(argv)
    try:
        result = run_demo(args.output, seed=args.seed)
    except (OSError, TypeError, ValueError) as exc:
        print(f"engineering demo rejected: {exc}")
        return 2
    print(f"wrote synthetic engineering baseline: {result.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
