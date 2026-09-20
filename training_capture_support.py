"""Pure validation and display helpers for controlled training capture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from data_recorder import RecorderQualitySnapshot, validate_session_id
from emg_protocol import DeviceKey, HandSide, RateDescriptor
from recording_context import RecordingContext, validate_experiment_id
from subject_identity import validate_subject_id
from training_contract import CANONICAL_LABELS


TRAINING_ACTIONS = CANONICAL_LABELS
TRAINING_ACTION_PHASE = "hold"
def _bounded_seconds(value: object, field: str, minimum: int, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{field} must be an integer from {minimum} through {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class TrainingCaptureSettings:
    """Strict settings for one canonical labelled training session.

    ``device_id`` is deliberately immutable and typed as the opaque identifier
    obtained from the active BLE connection; the operator must not type it.
    """

    subject_id: str
    session_id: str
    device_id: DeviceKey
    hand_side: HandSide
    action_label: str
    experiment_batch: str
    countdown_seconds: int = 3
    duration_seconds: int = 30

    def __post_init__(self) -> None:
        validate_subject_id(self.subject_id)
        object.__setattr__(self, "session_id", validate_session_id(self.session_id))
        if not isinstance(self.device_id, DeviceKey):
            raise ValueError("device_id must come from the connected DeviceKey")
        if self.hand_side not in {HandSide.LEFT, HandSide.RIGHT}:
            raise ValueError("hand_side must be HandSide.LEFT or HandSide.RIGHT")
        if self.action_label not in TRAINING_ACTIONS:
            raise ValueError("action_label must be rest, fist, or open_hand")
        validate_experiment_id(self.experiment_batch)
        _bounded_seconds(self.countdown_seconds, "countdown_seconds", 0, 60)
        _bounded_seconds(self.duration_seconds, "duration_seconds", 1, 3600)

    @property
    def action_phase(self) -> str:
        return TRAINING_ACTION_PHASE

    def to_recording_context(self) -> RecordingContext:
        return RecordingContext(
            subject_id=self.subject_id,
            action_label=self.action_label,
            action_phase=TRAINING_ACTION_PHASE,
            experiment_id=self.experiment_batch,
            hand_side=self.hand_side,
            training_provenance="canonical_session",
        )


def _nonnegative_number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{field} must be a finite non-negative number")
    return float(value)


def _nonnegative_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _snapshot_value(snapshot: object, field: str) -> Any:
    if isinstance(snapshot, RecorderQualitySnapshot):
        return getattr(snapshot, field)
    if isinstance(snapshot, Mapping):
        if field not in snapshot:
            raise ValueError(f"recorder snapshot is missing {field}")
        return snapshot[field]
    raise ValueError("recorder_snapshot must be RecorderQualitySnapshot or mapping")


def format_live_quality(
    *,
    pipeline_snapshot: Mapping[str, Any],
    recorder_snapshot: RecorderQualitySnapshot | Mapping[str, Any],
    sample_rate: RateDescriptor,
    host_observed_rate_hz: Optional[float] = None,
) -> str:
    """Format live quality evidence without upgrading host timing to ADC fact."""
    if not isinstance(pipeline_snapshot, Mapping):
        raise ValueError("pipeline_snapshot must be a mapping")
    required_pipeline = {"dropped_count", "queue_depth", "freshness_seconds"}
    if not required_pipeline.issubset(pipeline_snapshot):
        raise ValueError("pipeline_snapshot is missing required quality fields")
    dropped = _nonnegative_integer(pipeline_snapshot["dropped_count"], "dropped_count")
    queue_depth = _nonnegative_integer(pipeline_snapshot["queue_depth"], "queue_depth")
    freshness = pipeline_snapshot["freshness_seconds"]
    freshness_text = (
        "尚无有效样本"
        if freshness is None
        else f"{_nonnegative_number(freshness, 'freshness_seconds'):.3f} s"
    )

    available = _snapshot_value(recorder_snapshot, "sequence_detection_available")
    if not isinstance(available, bool):
        raise ValueError("sequence_detection_available must be bool")
    duplicate = _nonnegative_integer(
        _snapshot_value(recorder_snapshot, "duplicate_count"), "duplicate_count"
    )
    out_of_order = _nonnegative_integer(
        _snapshot_value(recorder_snapshot, "out_of_order_count"), "out_of_order_count"
    )
    gaps = _nonnegative_integer(
        _snapshot_value(recorder_snapshot, "gap_count"), "gap_count"
    )
    rows = _nonnegative_integer(
        _snapshot_value(recorder_snapshot, "recorded_rows"), "recorded_rows"
    )
    generation = _snapshot_value(recorder_snapshot, "connection_generation")
    if generation is not None:
        generation = _nonnegative_integer(generation, "connection_generation")

    if not isinstance(sample_rate, RateDescriptor):
        raise ValueError("sample_rate must be RateDescriptor")
    if host_observed_rate_hz is None:
        observed_text = "不可用"
    else:
        observed_text = f"{_nonnegative_number(host_observed_rate_hz, 'host_observed_rate_hz'):.2f} Hz"
    if sample_rate.confirmed:
        rate_text = (
            f"确认采样率={sample_rate.value_hz:g} Hz "
            f"({sample_rate.source_kind}, {sample_rate.evidence_ref})"
        )
    else:
        rate_text = "确认采样率=不可用"
    sequence_text = (
        f"设备序列重复={duplicate}, 设备序列乱序={out_of_order}, 设备序列缺口={gaps}"
        if available
        else "设备序列检测=不可检测（未配置可信序列协议）"
    )
    return "\n".join(
        (
            f"上位机队列丢弃={dropped}, 当前队列深度={queue_depth}, 新鲜度={freshness_text}",
            f"{sequence_text}, 连接代次={generation if generation is not None else '未知'}, 已录样本={rows}",
            f"{rate_text}; 主机观测速率={observed_text}（仅传输观测，不等于确认采样率）",
        )
    )


def format_stop_report(report: Mapping[str, Any]) -> str:
    """Render the training decision and every blocking/warning check message."""
    if not isinstance(report, Mapping):
        raise ValueError("report must be a mapping")
    checks = report.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("report checks must be a mapping")
    findings: list[tuple[str, str, str]] = []
    for name, check in checks.items():
        if not isinstance(name, str) or not isinstance(check, Mapping):
            raise ValueError("report checks are malformed")
        status = check.get("status")
        message = check.get("message")
        if status not in {"pass", "warning", "fail"} or not isinstance(message, str) or not message:
            raise ValueError("report check status/message is invalid")
        if status in {"warning", "fail"}:
            findings.append((status, name, message))
    training_usable = report.get("training_usable")
    if not isinstance(training_usable, bool):
        raise ValueError("training_usable must be bool")
    eligible = training_usable and not findings
    lines = [f"训练资格：{'可训练' if eligible else '不可训练'}"]
    if eligible:
        lines.append("所有质量检查均通过。")
    elif findings:
        lines.append("拒绝/警告原因：")
        lines.extend(
            f"- [{status.upper()}] {name}: {message}"
            for status, name, message in findings
        )
    else:
        lines.append("拒绝原因：质量报告未提供具体失败或警告信息。")
    return "\n".join(lines)
