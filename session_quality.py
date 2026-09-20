"""Bounded offline quality report for one canonical EMG session."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from acquisition_pipeline import (
    AcquisitionPipeline,
    RawAuditPolicy,
    RawNotification,
    RawPacketAudit,
)
from data_recorder import (
    METADATA_SCHEMA_ID,
    METADATA_SCHEMA_MAJOR,
    METADATA_SCHEMA_MINOR,
    _RecoveryStreamValidator,
    _decode_recovery_acquisition,
    _validated_metadata_version,
    _validated_session_boundary,
    validate_subject_key,
)
from emg_protocol import DeviceKey, HandSide, NotificationPacketProtocol, QualityFlags
from recording_context import RecordingContext
from training_contract import (
    validate_sample_rate_source_kind,
    validate_training_annotation,
    validate_training_provenance,
)


REPORT_SCHEMA = "emg.session.quality.v2"
SOURCE_SNAPSHOT_VERSION = 1
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_CSV_BYTES = 512 * 1024 * 1024
MAX_RAW_BYTES = 512 * 1024 * 1024
MAX_CSV_ROWS = 1_000_000
MAX_RAW_RECORDS = 1_000_000
MAX_RAW_LINE_BYTES = 1 * 1024 * 1024
MAX_ERROR_SAMPLES = 20
UINT64_MAX = (1 << 64) - 1
SATURATION_WARNING_RATE = 0.01
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_WARNING = 2


def _check(status: str, message: str, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status, "message": message}
    if details:
        result["details"] = details
    return result


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _reject_nonfinite_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_loads(text: str):
    return json.loads(text, parse_constant=_reject_nonfinite_json_constant)


def _new_report(path: Path) -> dict[str, Any]:
    return {
        "report_schema": REPORT_SCHEMA,
        "session_dir": str(path),
        "overall_status": "fail",
        "training_usable": False,
        "source_snapshot": {"version": SOURCE_SNAPSHOT_VERSION, "files": []},
        "checks": {},
        "metrics": {
            "csv_rows": 0,
            "timestamp_tie_count": 0,
            "channels": {},
            "quality_flag_counts": {},
            "raw_audit": {
                "state": "not_available",
                "files": [],
                "packet_count": 0,
                "parse_error_count": 0,
                "adjacent_identical_payload_count": 0,
            },
        },
        "terminology": {
            "observed_rate": "host_notification_rate_not_adc_sampling_rate",
            "note": "Host timing and CSV row rate are never reported as ADC sampling rate.",
        },
    }


def _record_snapshot_file(
    report: dict[str, Any], *, relative_name: str, role: str, size_bytes: int, sha256: str
) -> None:
    report["source_snapshot"]["files"].append(
        {
            "relative_name": relative_name,
            "role": role,
            "size_bytes": size_bytes,
            "sha256": sha256,
        }
    )


class _HashingRawReader(io.RawIOBase):
    def __init__(self, stream, maximum_bytes: int) -> None:
        super().__init__()
        self._stream = stream
        self._maximum_bytes = maximum_bytes
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        data = self._stream.read(len(buffer))
        if not data:
            return 0
        self.byte_count += len(data)
        if self.byte_count > self._maximum_bytes:
            raise ValueError("input exceeds hard byte limit while streaming")
        self.digest.update(data)
        buffer[: len(data)] = data
        return len(data)


def _finish(report: dict[str, Any]) -> dict[str, Any]:
    report["source_snapshot"]["files"].sort(
        key=lambda item: (item["role"], item["relative_name"])
    )
    statuses = [item["status"] for item in report["checks"].values()]
    if "fail" in statuses:
        overall = "fail"
    elif "warning" in statuses:
        overall = "warning"
    else:
        overall = "pass"
    report["overall_status"] = overall
    report["training_usable"] = overall == "pass"
    report["summary"] = dict(sorted(Counter(statuses).items()))
    return report


def _load_metadata(path: Path, report: dict[str, Any]) -> Optional[dict[str, Any]]:
    try:
        with path.open("rb") as stream:
            content = stream.read(MAX_METADATA_BYTES + 1)
        if len(content) > MAX_METADATA_BYTES:
            report["checks"]["input_limits"] = _check(
                "fail", "metadata.json exceeds hard byte limit", size_bytes=len(content)
            )
            return None
        _record_snapshot_file(
            report,
            relative_name="metadata.json",
            role="metadata",
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        payload = _strict_json_loads(content.decode("utf-8-sig"))
    except FileNotFoundError:
        report["checks"]["metadata_file"] = _check("fail", "metadata.json is missing")
        return None
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        report["checks"]["metadata_file"] = _check(
            "fail", f"metadata.json cannot be parsed: {type(exc).__name__}"
        )
        return None
    if not isinstance(payload, dict):
        report["checks"]["metadata_file"] = _check("fail", "metadata root must be an object")
        return None
    report["checks"]["metadata_file"] = _check("pass", "metadata.json parsed")
    return payload


def _training_provenance_contract(
    metadata: dict[str, Any], schema_minor: int
) -> tuple[Optional[str], Optional[str]]:
    if schema_minor < 8:
        return None, "historical schema does not carry required training provenance"
    try:
        return validate_training_provenance(metadata.get("training_provenance")), None
    except ValueError as exc:
        return None, f"invalid or missing training provenance: {exc}"


def _prepare_contract(metadata: dict[str, Any]):
    errors: list[str] = []
    if metadata.get("schema_id") != METADATA_SCHEMA_ID:
        errors.append("schema_id mismatch")
    if metadata.get("schema_major") != METADATA_SCHEMA_MAJOR:
        errors.append("schema_major mismatch")
    schema_minor = metadata.get("schema_minor")
    if (
        not isinstance(schema_minor, int)
        or isinstance(schema_minor, bool)
        or schema_minor < 0
        or schema_minor > METADATA_SCHEMA_MINOR
    ):
        errors.append("unsupported schema_minor")
        schema_minor = METADATA_SCHEMA_MINOR
    if metadata.get("status") != "complete":
        errors.append("session status is not complete")
    if metadata.get("recorder_state") != "CLOSED":
        errors.append("recorder_state must be CLOSED")
    if metadata.get("resource_state") != "CLOSED":
        errors.append("resource_state must be CLOSED")
    if metadata.get("csv_file") != "samples.csv":
        errors.append("csv_file must be exactly samples.csv")
    channels = metadata.get("channels")
    if not isinstance(channels, int) or isinstance(channels, bool) or not 1 <= channels <= 256:
        errors.append("invalid channels")
    try:
        validate_subject_key(metadata.get("subject_id"))
    except (TypeError, ValueError) as exc:
        errors.append(f"invalid subject identity: {exc}")
    try:
        device_id = str(DeviceKey(metadata.get("device_id")))
    except (TypeError, ValueError) as exc:
        device_id = ""
        errors.append(f"invalid device identity: {exc}")
    session_id = metadata.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        errors.append("invalid session identity")
    if schema_minor >= 4:
        try:
            hand_side = HandSide(metadata.get("hand_side"))
        except (TypeError, ValueError):
            hand_side = HandSide.UNKNOWN
            errors.append("invalid hand_side")
    else:
        hand_side = HandSide.UNKNOWN
    provenance_kind, provenance_error = _training_provenance_contract(
        metadata, schema_minor
    )
    if schema_minor >= 8 and provenance_error is not None:
        errors.append(provenance_error)
    expected_annotation = None
    if schema_minor >= 8:
        extra = metadata.get("extra")
        try:
            if not isinstance(extra, dict):
                raise ValueError("metadata.extra must be an object")
            required_context = {"action_label", "action_phase", "experiment_id"}
            optional_versions = {"firmware_version", "protocol_version"}
            if not required_context.issubset(extra) or set(extra) - required_context - optional_versions:
                raise ValueError("metadata.extra action context is incomplete or invalid")
            for field in optional_versions.intersection(extra):
                _validated_metadata_version(extra[field], field)
            context = RecordingContext(
                subject_id=metadata.get("subject_id"),
                action_label=extra["action_label"],
                action_phase=extra["action_phase"],
                experiment_id=extra["experiment_id"],
                hand_side=hand_side,
            )
            expected_annotation = (context.action_label, context.action_phase)
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"invalid metadata action context: {exc}")
    if schema_minor >= 4:
        if metadata.get("writer_generation_semantics") != "producer_instance_generation":
            errors.append("invalid writer generation semantics")
        if metadata.get("connection_generation_semantics") != "ble_connection_lifecycle_generation":
            errors.append("invalid connection generation semantics")
    try:
        acquisition, migrations = _decode_recovery_acquisition(
            metadata.get("acquisition"), schema_minor
        )
        if migrations:
            errors.append("quality validation requires migration of historical acquisition metadata")
    except (TypeError, ValueError, KeyError) as exc:
        acquisition = None
        errors.append(f"invalid acquisition metadata: {exc}")
    boundary = None
    try:
        boundary = _validated_session_boundary(
            metadata.get("session_boundary"), schema_minor
        )
        if boundary["incomplete_reason"] is not None:
            errors.append("session_boundary incomplete_reason must be null")
    except (TypeError, ValueError, KeyError) as exc:
        errors.append(f"invalid session_boundary: {exc}")
    if acquisition is not None:
        if acquisition.signal_chain.sample_format == "unknown":
            errors.append("sample_format is unknown")
        top_rate = metadata.get("sample_rate_hz")
        if (
            isinstance(top_rate, bool)
            or not isinstance(top_rate, (int, float))
            or not math.isfinite(float(top_rate))
            or top_rate <= 0
        ):
            errors.append("metadata sample_rate_hz must be a finite non-boolean positive number")
        elif top_rate != acquisition.sample_rate.value_hz:
            errors.append("metadata sample rate differs from acquisition descriptor")
        if metadata.get("sample_rate_semantics") != "sample_time_base_only":
            errors.append("invalid sample rate semantics")
    validator = None
    if not errors and acquisition is not None and isinstance(channels, int):
        try:
            validator = _RecoveryStreamValidator(
                channels=channels,
                schema_minor=schema_minor,
                acquisition=acquisition,
                device_id=device_id,
                session_id=session_id,
                expected_annotation=expected_annotation,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid canonical identity: {exc}")
    return (
        errors,
        acquisition,
        boundary,
        validator,
        schema_minor,
        provenance_kind,
        provenance_error,
    )


def _channel_state(channels: int) -> dict[str, dict[str, Any]]:
    return {
        f"channel_{index}": {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "first": None,
            "constant": False,
            "saturation_count": 0,
            "saturation_rate": 0.0,
        }
        for index in range(1, channels + 1)
    }


def _update_channel(state, name: str, value: float, limits):
    item = state[name]
    item["count"] += 1
    item["minimum"] = value if item["minimum"] is None else min(item["minimum"], value)
    item["maximum"] = value if item["maximum"] is None else max(item["maximum"], value)
    if item["first"] is None:
        item["first"] = value
    if limits is not None and value in limits:
        item["saturation_count"] += 1


def _validate_csv(session_dir: Path, metadata, acquisition, boundary, validator, report):
    path = session_dir / "samples.csv"
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        report["checks"]["canonical_csv"] = _check("fail", "samples.csv is missing")
        return {}, 0, 0, 0
    if size > MAX_CSV_BYTES:
        report["checks"]["input_limits"] = _check(
            "fail", "samples.csv exceeds hard byte limit", size_bytes=size
        )
        report["checks"]["canonical_csv"] = _check("fail", "CSV was not read because it exceeds the hard limit")
        return {}, 0, 0, 0
    channels = metadata["channels"]
    stats = _channel_state(channels)
    sample_format = acquisition.signal_chain.sample_format
    limits = {"uint8": (0, 255), "int16": (-32768, 32767)}.get(sample_format)
    csv_packets: dict[int, list[dict[str, Any]]] = {}
    empty_labels = 0
    unknown_labels = 0
    invalid_annotations = 0
    row_count = 0
    first_host_index = None
    last_host_index = None
    unique_host_count = 0
    flag_counts: Counter[str] = Counter()
    errors: list[str] = []
    csv_source = None
    try:
        with path.open("rb") as binary_stream:
            csv_source = _HashingRawReader(binary_stream, MAX_CSV_BYTES)
            with io.BufferedReader(csv_source) as buffered_stream:
                with io.TextIOWrapper(
                    buffered_stream, encoding="utf-8-sig", newline=""
                ) as stream:
                    reader = csv.DictReader(stream)
                    if reader.fieldnames != validator.expected_columns:
                        errors.append("CSV schema mismatch")
                    else:
                        for row_number, row in enumerate(reader, 1):
                            if row_number > MAX_CSV_ROWS:
                                errors.append("CSV row count exceeds hard limit")
                                break
                            try:
                                validator.validate(row)
                            except (TypeError, ValueError, KeyError) as exc:
                                errors.append(f"row {row_number}: {exc}")
                                break
                            host_index = int(row["host_receive_index"])
                            if first_host_index is None:
                                first_host_index = host_index
                            if last_host_index != host_index:
                                unique_host_count += 1
                            last_host_index = host_index
                            label = row["action_label"].strip()
                            if not label:
                                empty_labels += 1
                            elif label.casefold() == "unknown":
                                unknown_labels += 1
                            try:
                                validate_training_annotation(
                                    row["action_label"], row["action_phase"]
                                )
                            except ValueError:
                                invalid_annotations += 1
                            flags = QualityFlags(int(row["quality_flags"]))
                            for flag in QualityFlags:
                                if flag is not QualityFlags.NONE and flags & flag:
                                    flag_counts[flag.name] += 1
                            channel_values = []
                            for index in range(1, channels + 1):
                                raw = row[f"channel_{index}"]
                                value = (
                                    int(raw)
                                    if sample_format in {"uint8", "int16"}
                                    else float(raw)
                                )
                                channel_values.append(value)
                                _update_channel(
                                    stats, f"channel_{index}", value, limits
                                )
                            csv_packets.setdefault(host_index, []).append(
                                {
                                    "channel_values": tuple(channel_values),
                                    "sample_in_packet": int(row["sample_in_packet"]),
                                    "host_wall_timestamp_ns": int(
                                        row["host_wall_timestamp_ns"]
                                    ),
                                    "host_monotonic_ns": int(
                                        row["host_monotonic_ns"]
                                    ),
                                    "connection_generation": int(
                                        row["connection_generation"]
                                    ),
                                }
                            )
                            row_count += 1
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        errors.append(f"CSV read failed: {type(exc).__name__}")
    if csv_source is not None:
        _record_snapshot_file(
            report,
            relative_name="samples.csv",
            role="samples",
            size_bytes=csv_source.byte_count,
            sha256=csv_source.digest.hexdigest(),
        )
    tie_count = validator.host_monotonic_tie_count
    for item in stats.values():
        item["constant"] = item["count"] > 1 and item["minimum"] == item["maximum"]
        item["saturation_rate"] = (
            item["saturation_count"] / item["count"] if item["count"] else 0.0
        )
        item.pop("first", None)
    report["metrics"]["csv_rows"] = row_count
    report["metrics"]["timestamp_tie_count"] = tie_count
    report["metrics"]["channels"] = stats
    report["metrics"]["quality_flag_counts"] = dict(sorted(flag_counts.items()))
    report["checks"]["canonical_csv"] = _check(
        "fail" if errors else "pass",
        "; ".join(errors) if errors else "all rows satisfy DataRecorder canonical validation",
        validator="data_recorder._RecoveryStreamValidator",
    )
    protocol_shape_errors = [
        host_index
        for host_index, rows in csv_packets.items()
        if len(rows) != 1 or rows[0]["sample_in_packet"] != 0
    ]
    report["checks"]["protocol_sample_shape"] = _check(
        "fail" if protocol_shape_errors else "pass",
        (
            "CSV contains multiple samples for one BLE notification, but the production "
            "AcquisitionPipeline parser yields exactly one EmgFrame per notification"
            if protocol_shape_errors
            else "CSV packet/sample shape matches the one-frame-per-notification production parser"
        ),
        incompatible_host_receive_indexes=protocol_shape_errors[:20],
    )
    covered_status = "fail" if errors else "pass"
    covered_message = (
        "not certified because canonical CSV validation failed"
        if errors
        else "verified by data_recorder._RecoveryStreamValidator"
    )
    for check_name in (
        "session_sample_index",
        "host_receive_index",
        "host_monotonic_ns",
        "channel_value_range",
        "quality_flags",
    ):
        report["checks"][check_name] = _check(
            covered_status, covered_message
        )
    count_errors = []
    if metadata.get("row_count") != row_count:
        count_errors.append("metadata row_count differs from validated CSV rows")
    if metadata.get("last_persisted_row") != row_count:
        count_errors.append("last_persisted_row differs from validated CSV rows")
    if boundary is not None:
        for field in ("received_count", "eligible_count", "written_count"):
            if boundary[field] != row_count:
                count_errors.append(f"session_boundary {field} differs from CSV rows")
        if boundary.get("host_monotonic_tie_count", tie_count) != tie_count:
            count_errors.append("session_boundary tie count differs from CSV")
        for field in ("queue_drop_total", "queue_drop_session", "tail_pending_count", "tail_loss_count"):
            if boundary[field] != 0:
                count_errors.append(f"{field} is non-zero")
        start = boundary["start_host_receive_index"]
        end = boundary["end_host_receive_index"]
        if row_count:
            if start is None or first_host_index != start + 1:
                count_errors.append("session_boundary start does not precede the first CSV host index")
            if end is None or last_host_index != end:
                count_errors.append("session_boundary end differs from the last CSV host index")
            if start is not None and end is not None and end - start != unique_host_count:
                count_errors.append("session_boundary span differs from CSV packet count")
            if boundary["received_count"] != unique_host_count:
                count_errors.append("session_boundary received_count differs from CSV packet count")
        elif start != end:
            count_errors.append("empty CSV requires an empty host_receive_index boundary")
    report["checks"]["metadata_counts"] = _check(
        "fail" if count_errors else "pass",
        "; ".join(count_errors) if count_errors else "metadata counts match validated CSV",
    )
    signal_warnings = []
    for name, item in stats.items():
        if item["constant"]:
            signal_warnings.append(f"{name} is constant")
        if item["saturation_rate"] > SATURATION_WARNING_RATE:
            signal_warnings.append(f"{name} saturation exceeds 1 percent")
    report["checks"]["channel_quality"] = _check(
        "warning" if signal_warnings else "pass",
        "; ".join(signal_warnings) if signal_warnings else "channel ranges, constancy and saturation are acceptable",
    )
    gate_errors = []
    if row_count == 0:
        gate_errors.append("CSV contains no samples")
    if empty_labels:
        gate_errors.append(f"{empty_labels} rows have empty action_label")
    if unknown_labels:
        gate_errors.append(f"{unknown_labels} rows have unknown action_label")
    if invalid_annotations:
        gate_errors.append(
            f"{invalid_annotations} rows violate the canonical action_label/action_phase contract"
        )
    if not acquisition.sample_rate.confirmed or acquisition.sample_rate.value_hz is None:
        gate_errors.append("sample_time_rate is unconfirmed or unknown")
    try:
        validate_sample_rate_source_kind(acquisition.sample_rate.source_kind)
    except ValueError as exc:
        gate_errors.append(f"sample_time_rate {exc}")
    if errors:
        gate_errors.append("canonical CSV validation failed")
    if protocol_shape_errors:
        gate_errors.append("CSV packet/sample shape is incompatible with the production parser")
    report["checks"]["training_gate"] = _check(
        "fail" if gate_errors else "pass",
        "; ".join(gate_errors) if gate_errors else "labels and confirmed sample time base are training-ready",
    )
    return csv_packets, row_count, tie_count, empty_labels


def _ordered_raw_files(session_dir: Path) -> list[Path]:
    backups = []
    for candidate in session_dir.glob("raw_packets.jsonl.*"):
        try:
            backups.append((int(candidate.name.rsplit(".", 1)[1]), candidate))
        except (IndexError, ValueError):
            continue
    result = [path for _, path in sorted(backups, reverse=True)]
    main = session_dir / "raw_packets.jsonl"
    if main.is_file():
        result.append(main)
    return result


def _iter_raw_lines(path: Path, report: dict[str, Any]):
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as stream:
            line_number = 0

            def read_line() -> bytes:
                nonlocal byte_count
                data = stream.readline(MAX_RAW_LINE_BYTES + 1)
                byte_count += len(data)
                digest.update(data)
                return data

            while True:
                raw_line = read_line()
                if not raw_line:
                    return
                line_number += 1
                if len(raw_line) > MAX_RAW_LINE_BYTES:
                    while raw_line and not raw_line.endswith(b"\n"):
                        raw_line = read_line()
                    yield line_number, None, "line exceeds hard byte limit"
                    continue
                try:
                    yield line_number, raw_line.decode("utf-8"), None
                except UnicodeDecodeError as exc:
                    yield line_number, None, f"UnicodeDecodeError: {exc}"
    finally:
        _record_snapshot_file(
            report,
            relative_name=path.name,
            role="raw_audit",
            size_bytes=byte_count,
            sha256=digest.hexdigest(),
        )

def _json_uint(record: dict[str, Any], field: str, *, minimum: int = 0) -> int:
    value = record.get(field)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= UINT64_MAX
    ):
        raise ValueError(f"{field} must be a native JSON uint64 integer")
    return value


def _exact_fields(record: dict[str, Any], expected: set[str], kind: str) -> None:
    if set(record) != expected:
        raise ValueError(f"{kind} fields are invalid")


class _BoundedErrorSamples(list[str]):
    def __init__(self) -> None:
        super().__init__()
        self.total = 0

    def append(self, message: str) -> None:
        self.total += 1
        if len(self) < MAX_ERROR_SAMPLES:
            super().append(message)


def _raw_claimed_enabled(metadata: dict[str, Any]) -> bool:
    direct = _mapping(metadata.get("raw_audit"))
    extra = _mapping(metadata.get("extra"))
    nested = _mapping(extra.get("raw_audit"))
    return bool(
        direct.get("enabled") is True
        or extra.get("raw_audit_enabled") is True
        or nested.get("enabled") is True
    )


def _raw_not_available(report, *, claimed: bool):
    state = "partial" if claimed else "not_available"
    report["metrics"]["raw_audit"]["state"] = state
    status = "fail" if claimed else "not_available"
    message = "RawAudit was declared enabled but no raw files exist" if claimed else "RawAudit not available"
    report["checks"].update({
        "raw_audit": _check(status, message),
        "raw_completeness": _check(status, message),
        "raw_protocol_replay": _check("not_available", "RawAudit not available"),
        "raw_csv_mapping": _check("not_available", "RawAudit not available"),
        "adjacent_identical_payloads": _check("not_available", "RawAudit not available"),
    })


def _validate_raw(session_dir, metadata, protocol, boundary, csv_packets, report):
    files = _ordered_raw_files(session_dir)
    claimed = _raw_claimed_enabled(metadata)
    if not files:
        _raw_not_available(report, claimed=claimed)
        return
    try:
        file_sizes = {path: path.stat().st_size for path in files}
        total_bytes = sum(file_sizes.values())
    except OSError as exc:
        message = f"RawAudit file metadata cannot be read: {type(exc).__name__}: {exc}"
        report["metrics"]["raw_audit"].update(
            state="partial", files=[path.name for path in files]
        )
        report["checks"].update({
            "raw_audit": _check("fail", message),
            "raw_completeness": _check("fail", message),
            "raw_protocol_replay": _check("not_available", "RawAudit was not read"),
            "raw_csv_mapping": _check("not_available", "RawAudit was not read"),
            "adjacent_identical_payloads": _check("not_available", "RawAudit was not read"),
        })
        return
    if total_bytes > MAX_RAW_BYTES:
        report["checks"]["input_limits"] = _check(
            "fail", "RawAudit files exceed hard byte limit", size_bytes=total_bytes
        )
        report["metrics"]["raw_audit"].update(state="partial", files=[path.name for path in files])
        report["checks"]["raw_audit"] = _check("fail", "RawAudit was not read because it exceeds the hard limit")
        report["checks"]["raw_completeness"] = _check("fail", "RawAudit is partial because input limit was exceeded")
        report["checks"]["raw_protocol_replay"] = _check("not_available", "RawAudit was not read")
        report["checks"]["raw_csv_mapping"] = _check("not_available", "RawAudit was not read")
        report["checks"]["adjacent_identical_payloads"] = _check("not_available", "RawAudit was not read")
        return
    protocol_expected = asdict(protocol)
    policy_count = 0
    protocol_count = 0
    policy_complete = False
    validated_policy = None
    malformed = _BoundedErrorSamples()
    replay_errors = _BoundedErrorSamples()
    mapping_errors = _BoundedErrorSamples()
    completeness_errors = _BoundedErrorSamples()
    raw_indexes: set[int] = set()
    valid_indexes: set[int] = set()
    invalid_indexes: set[int] = set()
    packet_count = 0
    parse_error_count = 0
    identical_count = 0
    previous_payload = None
    previous_index = None
    first_index = None
    index_continuity_error = False
    first_effective_file = None
    physical_line_count = 0
    stop_reading = False

    def add_malformed(message: str) -> None:
        malformed.append(message)

    policy_fields = {
        "record_type", "max_bytes", "backup_count", "payload_prefix_bytes",
        "record_valid", "record_invalid", "retention_policy",
        "complete_payload_guarantee", "non_wire_length_policy",
    }
    protocol_fields = {
        "record_type", "mode", "wire_packet_size", "logical_packet_size",
        "padding_rule", "evidence_ref",
    }
    packet_fields = {
        "host_wall_timestamp_ns", "host_monotonic_ns", "host_receive_index",
        "connection_generation", "wire_protocol_mode", "original_length",
        "payload_hex", "parse_error",
    }
    truncated_fields = packet_fields | {"truncated", "sha256"}
    for path in files:
        file_position = 0
        try:
            for line_number, line, line_error in _iter_raw_lines(path, report):
                physical_line_count += 1
                file_position += 1
                if first_effective_file is None:
                    first_effective_file = path
                if physical_line_count > MAX_RAW_RECORDS:
                    completeness_errors.append("RawAudit physical line count exceeds hard limit")
                    stop_reading = True
                    break
                if line_error is not None:
                    add_malformed(f"{path.name}:{line_number} {line_error}")
                    continue
                try:
                    record = _strict_json_loads(line)
                except (ValueError, RecursionError) as exc:
                    add_malformed(
                        f"{path.name}:{line_number} invalid JSON: {type(exc).__name__}"
                    )
                    continue
                if not isinstance(record, dict):
                    add_malformed(f"{path.name}:{line_number} is not an object")
                    continue
                record_type = record.get("record_type")
                if record_type == "raw_audit_policy":
                    policy_count += 1
                    if path != first_effective_file or file_position != 1:
                        completeness_errors.append("RawAudit policy header is duplicated, late, or out of order")
                    try:
                        _exact_fields(record, policy_fields, "raw_audit_policy")
                        policy_max_bytes = _json_uint(record, "max_bytes", minimum=1)
                        policy_backup_count = _json_uint(record, "backup_count")
                        policy_prefix_bytes = _json_uint(record, "payload_prefix_bytes")
                        if type(record["record_valid"]) is not bool or type(record["record_invalid"]) is not bool:
                            raise ValueError("record flags must be booleans")
                        if record["retention_policy"] != "bounded_rotating_files":
                            raise ValueError("invalid retention_policy")
                        if record["complete_payload_guarantee"] != "exact_configured_wire_size":
                            raise ValueError("invalid complete_payload_guarantee")
                        if record["non_wire_length_policy"] != "full_if_limits_allow_else_prefix_length_sha256":
                            raise ValueError("invalid non_wire_length_policy")
                        RawAuditPolicy(
                            enabled=True,
                            max_bytes=policy_max_bytes,
                            backup_count=policy_backup_count,
                            payload_prefix_bytes=policy_prefix_bytes,
                            record_valid=record["record_valid"],
                            record_invalid=record["record_invalid"],
                        )
                        if policy_prefix_bytes < protocol.wire_packet_size:
                            raise ValueError("payload_prefix_bytes cannot retain a complete wire packet")
                        required_size = max(
                            RawPacketAudit.complete_record_size(protocol),
                            RawPacketAudit.complete_record_size(
                                protocol, include_parse_error=True
                            ),
                        )
                        if policy_max_bytes < required_size:
                            raise ValueError("max_bytes cannot retain a complete wire packet record")
                        policy_complete = record["record_valid"] and record["record_invalid"]
                        if (
                            policy_count == 1
                            and path == first_effective_file
                            and file_position == 1
                        ):
                            validated_policy = dict(record)
                    except (KeyError, TypeError, ValueError) as exc:
                        add_malformed(f"{path.name}:{line_number} invalid policy header: {exc}")
                    continue
                if record_type == "notification_packet_protocol":
                    protocol_count += 1
                    if path != first_effective_file or file_position != 2:
                        completeness_errors.append("RawAudit protocol header is duplicated, late, or out of order")
                    try:
                        _exact_fields(record, protocol_fields, "notification_packet_protocol")
                        _json_uint(record, "wire_packet_size", minimum=1)
                        _json_uint(record, "logical_packet_size", minimum=1)
                        actual = {key: record[key] for key in protocol_expected}
                        if actual != protocol_expected:
                            replay_errors.append("RawAudit protocol header differs from metadata")
                    except (KeyError, TypeError, ValueError) as exc:
                        add_malformed(f"{path.name}:{line_number} invalid protocol header: {exc}")
                    continue
                try:
                    truncated = "truncated" in record or "sha256" in record
                    _exact_fields(record, truncated_fields if truncated else packet_fields, "packet")
                    host_wall = _json_uint(record, "host_wall_timestamp_ns")
                    host_monotonic = _json_uint(record, "host_monotonic_ns")
                    host_index = _json_uint(record, "host_receive_index", minimum=1)
                    connection_generation = _json_uint(record, "connection_generation")
                    original_length = _json_uint(record, "original_length")
                    if not isinstance(record["wire_protocol_mode"], str):
                        raise ValueError("wire_protocol_mode must be a string")
                    if not isinstance(record["payload_hex"], str):
                        raise ValueError("payload_hex must be a string")
                    parse_error_value = record["parse_error"]
                    if parse_error_value is not None and not isinstance(parse_error_value, str):
                        raise ValueError("parse_error must be null or a string")
                    if isinstance(parse_error_value, str) and len(parse_error_value) > 256:
                        raise ValueError("parse_error exceeds production bound")
                    if isinstance(parse_error_value, str) and not parse_error_value.isascii():
                        raise ValueError("parse_error must use the production ASCII escaped form")
                    if truncated:
                        if record["truncated"] is not True:
                            raise ValueError("truncated must be true")
                        if not isinstance(record["sha256"], str) or len(record["sha256"]) != 64:
                            raise ValueError("sha256 must be 64 hex characters")
                        int(record["sha256"], 16)
                        if record["sha256"] != record["sha256"].lower():
                            raise ValueError("sha256 must use canonical lowercase hex")
                except (KeyError, TypeError, ValueError) as exc:
                    add_malformed(f"{path.name}:{line_number} invalid packet: {exc}")
                    continue
                packet_count += 1
                if previous_index is not None and host_index <= previous_index:
                    mapping_errors.append("RawAudit host_receive_index is not strictly increasing")
                    index_continuity_error = True
                elif previous_index is not None and host_index != previous_index + 1:
                    index_continuity_error = True
                if first_index is None:
                    first_index = host_index
                previous_index = host_index
                raw_indexes.add(host_index)
                if record.get("wire_protocol_mode") != protocol.mode:
                    replay_errors.append(f"raw index {host_index} protocol mode mismatch")
                parse_error = parse_error_value not in (None, "")
                if parse_error:
                    parse_error_count += 1
                    invalid_indexes.add(host_index)
                else:
                    valid_indexes.add(host_index)
                try:
                    payload = bytes.fromhex(record["payload_hex"])
                except ValueError:
                    replay_errors.append(f"raw index {host_index} payload_hex is invalid")
                    previous_payload = None
                    continue
                if record["payload_hex"] != payload.hex():
                    replay_errors.append(f"raw index {host_index} payload_hex is not canonical lowercase hex")
                if validated_policy is not None:
                    prefix_bytes = validated_policy["payload_prefix_bytes"]
                    max_bytes = validated_policy["max_bytes"]
                    complete_template = {
                        key: record[key] for key in packet_fields
                    }
                    complete_template["payload_hex"] = ""
                    complete_size = len(
                        (json.dumps(complete_template, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
                    ) + 2 * original_length
                    production_would_truncate = (
                        original_length > prefix_bytes or complete_size > max_bytes
                    )
                    if truncated != production_would_truncate:
                        replay_errors.append(
                            f"raw index {host_index} truncation disagrees with production RawPacketAudit policy"
                        )
                    if truncated:
                        truncated_template = dict(record)
                        truncated_template["payload_hex"] = ""
                        empty_size = len(
                            (json.dumps(truncated_template, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
                        )
                        expected_prefix_length = min(
                            original_length,
                            prefix_bytes,
                            max(0, (max_bytes - empty_size) // 2),
                        )
                        if len(payload) != expected_prefix_length:
                            replay_errors.append(
                                f"raw index {host_index} retained prefix length disagrees with production writer"
                            )
                        if original_length == protocol.wire_packet_size:
                            replay_errors.append(
                                f"raw index {host_index} configured wire packet cannot be truncated"
                            )
                if truncated:
                    completeness_errors.append(f"raw index {host_index} payload is truncated")
                    previous_payload = None
                    continue
                if original_length != len(payload):
                    replay_errors.append(f"raw index {host_index} original_length mismatch")
                try:
                    raw_item = RawNotification(
                        host_wall_timestamp_ns=host_wall,
                        host_monotonic_ns=host_monotonic,
                        host_receive_index=host_index,
                        payload=payload,
                        connection_generation=connection_generation,
                    )
                    frame = AcquisitionPipeline.parse_notification(
                        raw_item,
                        sample_index=0,
                        overflow=False,
                        generation=0,
                        packet_protocol=protocol,
                    )
                    replay_failed = False
                except (KeyError, TypeError, ValueError):
                    frame = None
                    replay_failed = True
                if replay_failed != parse_error:
                    replay_errors.append(f"raw index {host_index} parse_error disagrees with protocol replay")
                if not parse_error and frame is not None:
                    rows = csv_packets.get(host_index, [])
                    if len(rows) != 1:
                        mapping_errors.append(
                            f"raw index {host_index} does not map to exactly one parsed CSV sample"
                        )
                    else:
                        row = rows[0]
                        mismatches = []
                        if row["sample_in_packet"] != 0:
                            mismatches.append("sample_in_packet")
                        if tuple(frame.channel_values) != row["channel_values"]:
                            mismatches.append("channel_values")
                        for field in (
                            "host_wall_timestamp_ns",
                            "host_monotonic_ns",
                            "connection_generation",
                        ):
                            if getattr(frame, field) != row[field]:
                                mismatches.append(field)
                        if mismatches:
                            mapping_errors.append(
                                f"raw index {host_index} differs from CSV: {','.join(mismatches)}"
                            )
                if previous_payload is not None and payload == previous_payload:
                    identical_count += 1
                previous_payload = payload
        except OSError as exc:
            add_malformed(
                f"{path.name}: RawAudit read failed: {type(exc).__name__}: {exc}"
            )
        if stop_reading:
            break
    if policy_count != 1:
        completeness_errors.append("RawAudit policy header must occur exactly once")
    elif not policy_complete:
        completeness_errors.append("RawAudit policy does not record both valid and invalid packets")
    if protocol_count != 1:
        completeness_errors.append("RawAudit protocol header must occur exactly once")
    if validated_policy is not None:
        suffixes = sorted(
            int(path.name.rsplit(".", 1)[1])
            for path in files
            if path.name != "raw_packets.jsonl"
        )
        backup_count = validated_policy["backup_count"]
        if len(suffixes) > backup_count or any(index > backup_count for index in suffixes):
            completeness_errors.append("RawAudit rotated file suffix exceeds policy backup_count")
        if any(actual != expected for expected, actual in enumerate(suffixes, 1)):
            completeness_errors.append("RawAudit rotated file suffixes are not contiguous from .1")
        max_bytes = validated_policy["max_bytes"]
        for raw_path, size in file_sizes.items():
            if size > max_bytes:
                completeness_errors.append(
                    f"{raw_path.name} exceeds RawAudit policy max_bytes"
                )
    if boundary is not None and not any(
        boundary[field] for field in ("queue_drop_session", "tail_pending_count", "tail_loss_count")
    ):
        start = boundary["start_host_receive_index"]
        end = boundary["end_host_receive_index"]
        if start is not None and end is not None:
            expected_count = end - start
            if first_index != start + 1:
                completeness_errors.append("RawAudit rotation prefix is missing")
            if (
                previous_index != end
                or packet_count != expected_count
                or index_continuity_error
            ):
                completeness_errors.append("RawAudit does not cover the complete session boundary")
    csv_indexes = set(csv_packets)
    if valid_indexes != csv_indexes:
        mapping_errors.append("valid RawAudit indexes differ from CSV host_receive_index values")
    if invalid_indexes & csv_indexes:
        mapping_errors.append("parse-error RawAudit indexes appear in CSV")
    if valid_indexes & invalid_indexes:
        mapping_errors.append("one RawAudit host index is both valid and invalid")
    state = "partial" if any(
        bucket.total for bucket in (malformed, replay_errors, mapping_errors, completeness_errors)
    ) else "complete"
    report["metrics"]["raw_audit"] = {
        "state": state,
        "files": [path.name for path in files],
        "physical_line_count": physical_line_count,
        "packet_count": packet_count,
        "parse_error_count": parse_error_count,
        "malformed_error_count": malformed.total,
        "error_sample_limit": MAX_ERROR_SAMPLES,
        "adjacent_identical_payload_count": identical_count,
        "coverage_host_receive_index": [min(raw_indexes), max(raw_indexes)] if raw_indexes else None,
    }
    report["checks"]["raw_audit"] = _check(
        "fail" if malformed.total else "pass",
        (
            "; ".join(malformed)
            + (f"; {malformed.total - len(malformed)} additional errors omitted" if malformed.total > len(malformed) else "")
            if malformed.total
            else f"streamed {packet_count} RawAudit packet records"
        ),
        error_count=malformed.total,
    )
    report["checks"]["raw_completeness"] = _check(
        "fail" if completeness_errors else "pass",
        "; ".join(dict.fromkeys(completeness_errors)) if completeness_errors else "RawAudit state is complete",
        state=state,
    )
    report["checks"]["raw_protocol_replay"] = _check(
        "fail" if replay_errors else "pass",
        "; ".join(dict.fromkeys(replay_errors)) if replay_errors else "Raw packets agree with metadata protocol and parse_error",
    )
    report["checks"]["raw_csv_mapping"] = _check(
        "fail" if mapping_errors else "pass",
        "; ".join(dict.fromkeys(mapping_errors)) if mapping_errors else "Raw valid packets map exactly to CSV rows",
    )
    report["checks"]["raw_parse_errors"] = _check(
        "warning" if parse_error_count else "pass",
        f"{parse_error_count} packets were rejected by protocol parsing" if parse_error_count else "no RawAudit parse errors",
    )
    report["checks"]["adjacent_identical_payloads"] = _check(
        "warning" if identical_count else "pass",
        f"{identical_count} adjacent payload pairs are identical; 相同值不可断言重传" if identical_count else "no adjacent identical payloads",
        retransmission_asserted=False,
    )


def analyze_session(session_dir: Any) -> dict[str, Any]:
    path = Path(session_dir).expanduser().resolve()
    report = _new_report(path)
    if not path.is_dir():
        report["checks"]["session_directory"] = _check("fail", "session directory is missing")
        return _finish(report)
    report["checks"]["session_directory"] = _check("pass", "session directory exists")
    metadata = _load_metadata(path / "metadata.json", report)
    if metadata is None:
        report["checks"]["schema_status"] = _check(
            "fail", "metadata schema cannot be validated because metadata is unavailable"
        )
        report["checks"]["training_gate"] = _check(
            "fail", "metadata is not a complete canonical training session"
        )
        return _finish(report)
    schema_minor = metadata.get("schema_minor")
    if (
        isinstance(schema_minor, int)
        and not isinstance(schema_minor, bool)
        and 0 <= schema_minor < 6
        and metadata.get("schema_id") == METADATA_SCHEMA_ID
        and metadata.get("schema_major") == METADATA_SCHEMA_MAJOR
    ):
        reason = (
            f"historical schema v1.{schema_minor} is retained for read-only/recovery "
            "compatibility and is not eligible for v1.8 training"
        )
        report["checks"]["schema_status"] = _check("warning", reason)
        report["checks"]["canonical_csv"] = _check(
            "warning", "historical CSV was not reclassified as damaged; use recovery tooling for validation"
        )
        report["checks"]["training_provenance"] = _check(
            "warning",
            "historical schema does not carry required training provenance",
            provenance_kind=None,
        )
        report["checks"]["training_gate"] = _check("warning", reason)
        return _finish(report)
    (
        errors,
        acquisition,
        boundary,
        validator,
        schema_minor,
        provenance_kind,
        provenance_error,
    ) = _prepare_contract(metadata)
    historical = schema_minor <= 7
    report["checks"]["training_provenance"] = _check(
        "warning" if historical else "fail" if provenance_error else "pass",
        provenance_error or f"training provenance {provenance_kind} is valid",
        provenance_kind=provenance_kind,
    )
    report["checks"]["schema_status"] = _check(
        "fail" if errors else ("warning" if historical else "pass"),
        "; ".join(errors)
        if errors
        else (
            f"historical schema v1.{schema_minor} validated for read-only/recovery compatibility; "
            "training requires schema v1.8"
            if historical
            else "canonical metadata schema v1.8 and complete status verified"
        ),
    )
    if errors or acquisition is None or boundary is None or validator is None:
        report["checks"]["canonical_csv"] = _check("fail", "canonical CSV validation unavailable because metadata contract failed")
        gate_message = "metadata is not a complete canonical training session"
        if provenance_error is not None and not historical:
            gate_message += f"; {provenance_error}"
        report["checks"]["training_gate"] = _check("fail", gate_message)
        _validate_raw(path, metadata, acquisition.notification_packet_protocol if acquisition else NotificationPacketProtocol(), boundary, {}, report)
        return _finish(report)
    csv_packets, _, _, _ = _validate_csv(
        path, metadata, acquisition, boundary, validator, report
    )
    if historical and report["checks"]["training_gate"]["status"] == "pass":
        report["checks"]["training_gate"] = _check(
            "warning",
            f"historical schema v1.{schema_minor} is read-only compatible but training_usable=false; "
            "metadata-to-CSV action binding requires schema v1.8",
        )
    _validate_raw(
        path,
        metadata,
        acquisition.notification_packet_protocol,
        boundary,
        csv_packets,
        report,
    )
    return _finish(report)


def main(argv: Optional[Iterable[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    parser = argparse.ArgumentParser(description="Check one canonical EMG session offline")
    parser.add_argument("session_dir")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = analyze_session(args.session_dir)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    print(encoded)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    if report["overall_status"] == "pass":
        return EXIT_PASS
    if report["overall_status"] == "warning":
        return EXIT_WARNING
    return EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
