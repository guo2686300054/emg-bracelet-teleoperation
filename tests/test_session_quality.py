import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import session_quality
from data_recorder import DataRecorder
from emg_protocol import (
    AcquisitionMetadata,
    DeviceKey,
    EmgFrame,
    HandSide,
    NotificationPacketProtocol,
    PADDED28_PROTOCOL,
    QualityFlags,
    RateDescriptor,
    SignalChain,
)
from recording_context import RecordingContext


SUBJECT = "sub-0123456789abcdef0123456789abcdef"
DEVICE = DeviceKey("dev-0123456789abcdef0123456789abcdef")
BASE_FLAGS = (
    QualityFlags.VALID
    | QualityFlags.HOST_WALL_TIME_VALID
    | QualityFlags.HOST_MONOTONIC_VALID
    | QualityFlags.HOST_RECEIVE_INDEX_VALID
)
PROTOCOL = NotificationPacketProtocol(
    PADDED28_PROTOCOL, 28, 16, "zero_suffix", "test:sha256"
)


def _acquisition(*, confirmed_rate=True):
    rate = (
        RateDescriptor(200.0, "protocol", "spec-v1", True)
        if confirmed_rate
        else RateDescriptor()
    )
    return AcquisitionMetadata(
        sample_rate=rate,
        signal_chain=SignalChain(sample_format="uint8", unit="adc_code"),
        notification_packet_protocol=PROTOCOL,
    )


def _session(
    tmp_path,
    *,
    row_count=3,
    labels=None,
    confirmed_rate=True,
    monotonic=None,
    channel_rows=None,
    root_name="data",
    training_provenance="canonical_session",
):
    recorder = DataRecorder(
        tmp_path / root_name,
        subject_id=SUBJECT,
        device_id=DEVICE,
        acquisition=_acquisition(confirmed_rate=confirmed_rate),
        channels=8,
        session_id="session-a",
        flush_every=1,
        side=HandSide.LEFT,
        recording_context=RecordingContext(
            SUBJECT,
            "fist",
            "hold",
            "quality_test_v1",
            HandSide.LEFT,
            training_provenance=training_provenance,
        ),
    )
    recorder.configure_session_boundary(start_host_receive_index=0, start_drop_total=0)
    labels = labels if labels is not None else ["fist"] * row_count
    monotonic = monotonic if monotonic is not None else [2_000 + index for index in range(row_count)]
    channel_rows = channel_rows if channel_rows is not None else [
        tuple(10 * channel + index for channel in range(1, 9))
        for index in range(row_count)
    ]
    for index in range(row_count):
        recorder.record(
            EmgFrame(
                channel_rows[index],
                host_wall_timestamp_ns=1_000 + index,
                host_monotonic_ns=monotonic[index],
                host_receive_index=index + 1,
                sample_index=100 + index,
                generation=7,
                connection_generation=3,
                action_label=labels[index],
                action_phase="hold",
                quality_flags=BASE_FLAGS,
            ),
            session_sample_index=index,
        )
    tie_count = sum(monotonic[index] == monotonic[index - 1] for index in range(1, row_count))
    recorder.update_session_boundary(
        end_host_receive_index=row_count,
        received_count=row_count,
        eligible_count=row_count,
        written_count=row_count,
        queue_drop_total=0,
        queue_drop_session=0,
        tail_pending_count=0,
        tail_loss_count=0,
        incomplete_reason=None,
    )
    recorder.close(complete=True)
    metadata = json.loads(recorder.metadata_path.read_text(encoding="utf-8"))
    metadata["session_boundary"]["host_monotonic_tie_count"] = tie_count
    recorder.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return recorder.session_dir


def _read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mutate_csv(session, mutate):
    path = session / "samples.csv"
    fieldnames, rows = _read_csv(path)
    mutate(fieldnames, rows)
    _write_csv(path, fieldnames, rows)


def _raw_record(index, payload, parse_error=None):
    return {
        "host_wall_timestamp_ns": 1_000 + index - 1,
        "host_monotonic_ns": 2_000 + index - 1,
        "host_receive_index": index,
        "connection_generation": 3,
        "wire_protocol_mode": PADDED28_PROTOCOL,
        "original_length": len(payload),
        "payload_hex": payload.hex(),
        "parse_error": parse_error,
    }


def _payload_for_host(host_index):
    logical = bytearray(16)
    row_index = host_index - 1
    for channel in range(1, 9):
        logical[channel * 2 - 1] = 10 * channel + row_index
    return bytes(logical) + bytes(12)


def _write_raw(session, records, *, filename="raw_packets.jsonl", include_headers=True):
    path = session / filename
    with path.open("w", encoding="utf-8") as stream:
        if include_headers:
            stream.write(json.dumps({
                "record_type": "raw_audit_policy",
                "max_bytes": 1024 * 1024,
                "backup_count": 3,
                "payload_prefix_bytes": 256,
                "record_valid": True,
                "record_invalid": True,
                "retention_policy": "bounded_rotating_files",
                "complete_payload_guarantee": "exact_configured_wire_size",
                "non_wire_length_policy": "full_if_limits_allow_else_prefix_length_sha256",
            }) + "\n")
            stream.write(json.dumps({
                "record_type": "notification_packet_protocol",
                "mode": PADDED28_PROTOCOL,
                "wire_packet_size": 28,
                "logical_packet_size": 16,
                "padding_rule": "zero_suffix",
                "evidence_ref": "test:sha256",
            }) + "\n")
        for record in records:
            stream.write(json.dumps(record) + "\n")
    return path


def _read_raw_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _replace_raw_records(path, records):
    encoded = "".join(json.dumps(record) + "\n" for record in records)
    path.write_text(encoded, encoding="utf-8")


def test_real_v18_full_columns_pass_and_ties_are_counted(tmp_path):
    session = _session(tmp_path, monotonic=[2_000, 2_000, 2_001])
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "pass"
    assert report["training_usable"] is True
    assert report["metrics"]["csv_rows"] == 3
    assert report["metrics"]["timestamp_tie_count"] == 1
    assert report["checks"]["canonical_csv"]["status"] == "pass"
    assert report["checks"]["training_provenance"] == {
        "status": "pass",
        "message": "training provenance canonical_session is valid",
        "details": {"provenance_kind": "canonical_session"},
    }
    for name in ("session_sample_index", "host_receive_index", "host_monotonic_ns", "channel_value_range", "quality_flags"):
        assert report["checks"][name]["status"] == "pass"
    assert report["metrics"]["raw_audit"]["state"] == "not_available"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda metadata: metadata.pop("training_provenance"),
        lambda metadata: metadata["training_provenance"].pop("version"),
        lambda metadata: metadata["training_provenance"].update(version="2.0"),
        lambda metadata: metadata["training_provenance"].update(kind="ordinary_gui"),
        lambda metadata: metadata.update(training_provenance="canonical_session"),
    ],
)
def test_training_provenance_missing_or_tampered_fails_closed(tmp_path, mutation):
    session = _session(tmp_path)
    metadata_path = session / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutation(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = session_quality.analyze_session(session)

    assert report["training_usable"] is False
    assert report["checks"]["training_provenance"]["status"] == "fail"
    assert report["checks"]["training_provenance"]["details"]["provenance_kind"] is None
    assert "provenance" in report["checks"]["training_gate"]["message"]


def test_ordinary_gui_recording_without_provenance_is_not_training_usable(tmp_path):
    session = _session(tmp_path, training_provenance=None)
    report = session_quality.analyze_session(session)
    assert report["training_usable"] is False
    assert report["checks"]["training_provenance"]["status"] == "fail"
    assert "missing" in report["checks"]["training_provenance"]["message"]


@pytest.mark.parametrize(
    "kind", ["canonical_session", "synthetic_test", "external_benchmark"]
)
def test_all_training_dataset_provenance_kinds_remain_compatible(tmp_path, kind):
    session = _session(tmp_path, training_provenance=kind)
    report = session_quality.analyze_session(session)
    assert report["training_usable"] is True
    assert report["checks"]["training_provenance"]["status"] == "pass"
    assert report["checks"]["training_provenance"]["details"]["provenance_kind"] == kind


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda fields, rows: (fields.remove("generation"), [row.pop("generation") for row in rows]), "CSV schema mismatch"),
        (lambda fields, rows: rows[0].__setitem__("device_id", "dev-11111111111111111111111111111111"), "identity"),
        (lambda fields, rows: rows[0].__setitem__("sample_rate_hz", "201"), "sample rate"),
    ],
)
def test_canonical_validator_rejects_deleted_column_identity_and_rate_tampering(tmp_path, mutation, expected):
    session = _session(tmp_path)
    _mutate_csv(session, mutation)
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert report["checks"]["canonical_csv"]["status"] == "fail"
    assert expected.lower() in report["checks"]["canonical_csv"]["message"].lower()


@pytest.mark.parametrize("case", ["empty_label", "unknown_label", "unknown_rate", "empty_csv"])
def test_training_gate_fails_closed(tmp_path, case):
    if case == "empty_label":
        session = _session(tmp_path)
        _mutate_csv(session, lambda fields, rows: rows[1].update(action_label=""))
    elif case == "unknown_label":
        session = _session(tmp_path)
        _mutate_csv(session, lambda fields, rows: rows[1].update(action_label="unknown"))
    elif case == "unknown_rate":
        session = _session(tmp_path, confirmed_rate=False)
    elif case == "empty_csv":
        session = _session(tmp_path, row_count=0)
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert report["training_usable"] is False
    assert report["checks"]["training_gate"]["status"] == "fail"


@pytest.mark.parametrize(
    ("schema_minor", "csv_status"),
    [(5, "warning"), (6, "pass"), (7, "pass")],
)
def test_historical_schema_is_not_damaged_and_never_training_usable(
    tmp_path, schema_minor, csv_status
):
    session = _session(tmp_path)
    path = session / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["schema_minor"] = schema_minor
    metadata["extra"]["action_label"] = "rest"
    if schema_minor == 6:
        metadata["session_boundary"].pop("host_monotonic_tie_count")
    path.write_text(json.dumps(metadata), encoding="utf-8")

    report = session_quality.analyze_session(session)

    assert report["overall_status"] == "warning"
    assert report["training_usable"] is False
    assert report["checks"]["schema_status"]["status"] == "warning"
    assert "read-only/recovery compatibility" in report["checks"]["schema_status"]["message"]
    assert report["checks"]["canonical_csv"]["status"] == csv_status
    assert report["checks"]["training_gate"]["status"] == "warning"
    assert "v1.8" in report["checks"]["training_gate"]["message"]


@pytest.mark.parametrize("source_kind", ["host_observed", "host_csv_analysis", "made_up_source"])
def test_training_gate_requires_explicit_trusted_sample_rate_source(tmp_path, source_kind):
    session = _session(tmp_path)
    metadata_path = session / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["acquisition"]["sample_rate"]["source_kind"] = source_kind
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _mutate_csv(
        session,
        lambda fields, rows: [
            row.__setitem__("sample_rate_source_kind", source_kind) for row in rows
        ],
    )
    report = session_quality.analyze_session(session)
    assert report["checks"]["canonical_csv"]["status"] == "pass"
    assert report["checks"]["training_gate"]["status"] == "fail"
    assert "trusted" in report["checks"]["training_gate"]["message"]


@pytest.mark.parametrize(
    "value", [True, False, "200", float("nan"), float("inf"), -1.0, 0.0]
)
def test_top_level_sample_rate_requires_finite_nonboolean_positive_number(tmp_path, value):
    session = _session(tmp_path)
    metadata_path = session / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["sample_rate_hz"] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = session_quality.analyze_session(session)

    assert report["checks"]["schema_status"]["status"] == "fail"
    assert report["checks"]["training_gate"]["status"] == "fail"
    assert report["training_usable"] is False


@pytest.mark.parametrize("value", [200, 200.0])
def test_top_level_sample_rate_accepts_finite_positive_numeric_values(tmp_path, value):
    session = _session(tmp_path)
    metadata_path = session / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["sample_rate_hz"] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = session_quality.analyze_session(session)

    assert report["checks"]["schema_status"]["status"] == "pass"
    assert report["checks"]["training_gate"]["status"] == "pass"
    assert report["training_usable"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_label", "custom"),
        ("action_label", "open"),
        ("action_label", "pinch"),
        ("action_label", "wrist_flex"),
        ("action_phase", "transition"),
    ],
)
def test_training_gate_rejects_noncanonical_action_contract(tmp_path, field, value):
    session = _session(tmp_path)
    _mutate_csv(session, lambda fields, rows: rows[0].update({field: value}))
    report = session_quality.analyze_session(session)
    assert report["checks"]["canonical_csv"]["status"] == "fail"
    assert report["checks"]["training_gate"]["status"] == "fail"
    assert "CSV action annotation differs from metadata" in report["checks"]["canonical_csv"]["message"]


def test_metadata_is_the_action_anchor_and_valid_but_different_csv_label_is_rejected(tmp_path):
    session = _session(tmp_path)
    metadata_path = session / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["extra"]["action_label"] = "rest"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    report = session_quality.analyze_session(session)
    assert report["checks"]["canonical_csv"]["status"] == "fail"
    assert report["checks"]["training_gate"]["status"] == "fail"
    assert "CSV action annotation differs from metadata" in report["checks"]["canonical_csv"]["message"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda extra: extra.pop("action_label"),
        lambda extra: extra.update(action_phase="transition"),
        lambda extra: extra.update(experiment_id="../escape"),
        lambda extra: extra.update(patient_name="Alice"),
        lambda extra: extra.update(firmware_version={"nested": "value"}),
    ],
)
def test_metadata_action_context_is_strict_and_training_unusable(tmp_path, mutation):
    session = _session(tmp_path)
    metadata_path = session / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutation(metadata["extra"])
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    report = session_quality.analyze_session(session)
    assert report["checks"]["schema_status"]["status"] == "fail"
    assert report["checks"]["training_gate"]["status"] == "fail"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda metadata: metadata.update(recorder_state="OPEN"),
        lambda metadata: metadata.update(resource_state="OPEN"),
        lambda metadata: metadata["session_boundary"].update(incomplete_reason="tail loss"),
        lambda metadata: metadata["session_boundary"].update(received_count=4),
        lambda metadata: metadata["session_boundary"].update(eligible_count=2),
        lambda metadata: metadata["session_boundary"].update(queue_drop_total=1),
        lambda metadata: metadata["session_boundary"].update(queue_drop_session=1),
        lambda metadata: metadata["session_boundary"].update(tail_pending_count=1),
        lambda metadata: metadata["session_boundary"].update(tail_loss_count=1),
    ],
)
def test_complete_training_requires_closed_lossless_exact_boundary(tmp_path, mutation):
    session = _session(tmp_path)
    metadata_path = session / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutation(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert report["training_usable"] is False


def test_boundary_host_indexes_must_match_csv_without_raw(tmp_path):
    session = _session(tmp_path)
    metadata_path = session / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["session_boundary"].update(
        start_host_receive_index=100,
        end_host_receive_index=103,
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    report = session_quality.analyze_session(session)
    assert report["checks"]["canonical_csv"]["status"] == "pass"
    assert report["checks"]["metadata_counts"]["status"] == "fail"
    assert report["training_usable"] is False


def test_canonical_multi_sample_packet_allows_equal_host_index(tmp_path):
    session = _session(tmp_path)
    _mutate_csv(
        session,
        lambda fields, rows: rows[1].update(
            host_receive_index=rows[0]["host_receive_index"],
            host_wall_timestamp_ns=rows[0]["host_wall_timestamp_ns"],
            host_monotonic_ns=rows[0]["host_monotonic_ns"],
            sample_in_packet="1",
        ),
    )
    report = session_quality.analyze_session(session)
    assert report["checks"]["canonical_csv"]["status"] == "pass"
    assert report["checks"]["protocol_sample_shape"]["status"] == "fail"
    assert report["overall_status"] == "fail"


def test_multi_sample_packet_with_raw_is_rejected_by_explicit_production_shape(tmp_path):
    session = _session(tmp_path)
    _mutate_csv(
        session,
        lambda fields, rows: rows[1].update(
            host_receive_index=rows[0]["host_receive_index"],
            host_wall_timestamp_ns=rows[0]["host_wall_timestamp_ns"],
            host_monotonic_ns=rows[0]["host_monotonic_ns"],
            sample_in_packet="1",
        ),
    )
    _write_raw(
        session,
        [_raw_record(index, _payload_for_host(index)) for index in range(1, 4)],
    )
    report = session_quality.analyze_session(session)
    assert report["checks"]["canonical_csv"]["status"] == "pass"
    assert report["checks"]["protocol_sample_shape"]["status"] == "fail"
    assert "one EmgFrame" in report["checks"]["protocol_sample_shape"]["message"]
    assert report["overall_status"] == "fail"


def test_raw_complete_replays_protocol_and_maps_to_csv(tmp_path):
    session = _session(tmp_path)
    records = [_raw_record(index, _payload_for_host(index)) for index in range(1, 4)]
    _write_raw(session, records)
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "pass"
    assert report["metrics"]["raw_audit"]["state"] == "complete"
    assert report["checks"]["raw_protocol_replay"]["status"] == "pass"
    assert report["checks"]["raw_csv_mapping"]["status"] == "pass"


def test_source_snapshot_hashes_exact_parsed_files_without_absolute_names(tmp_path):
    session = _session(tmp_path)
    raw_path = _write_raw(
        session,
        [_raw_record(index, _payload_for_host(index)) for index in range(1, 4)],
    )
    report = session_quality.analyze_session(session)
    assert report["training_usable"] is True
    snapshot = report["source_snapshot"]
    assert snapshot["version"] == session_quality.SOURCE_SNAPSHOT_VERSION
    entries = {item["relative_name"]: item for item in snapshot["files"]}
    assert set(entries) == {"metadata.json", "samples.csv", "raw_packets.jsonl"}
    for name, role in (
        ("metadata.json", "metadata"),
        ("samples.csv", "samples"),
        ("raw_packets.jsonl", "raw_audit"),
    ):
        content = (session / name).read_bytes()
        assert entries[name] == {
            "relative_name": name,
            "role": role,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        assert Path(entries[name]["relative_name"]).name == entries[name]["relative_name"]
    assert raw_path.name == "raw_packets.jsonl"


@pytest.mark.parametrize("layout", ["headers_at_end", "reversed", "duplicate", "late_rotation"])
def test_raw_headers_are_unique_ordered_prefix_of_first_effective_file(tmp_path, layout):
    session = _session(tmp_path)
    raw_path = _write_raw(
        session,
        [_raw_record(index, _payload_for_host(index)) for index in range(1, 4)],
    )
    records = _read_raw_records(raw_path)
    policy, protocol, packets = records[0], records[1], records[2:]
    if layout == "headers_at_end":
        _replace_raw_records(raw_path, packets + [policy, protocol])
    elif layout == "reversed":
        _replace_raw_records(raw_path, [protocol, policy] + packets)
    elif layout == "duplicate":
        _replace_raw_records(raw_path, [policy, protocol, protocol] + packets)
    else:
        _replace_raw_records(raw_path, packets[1:])
        _replace_raw_records(
            session / "raw_packets.jsonl.1",
            [policy, protocol, packets[0]],
        )
        late = _read_raw_records(raw_path)
        _replace_raw_records(raw_path, [late[0], protocol] + late[1:])
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert report["checks"]["raw_completeness"]["status"] == "fail"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("host_receive_index", 1.9),
        ("host_monotonic_ns", "2000"),
        ("connection_generation", True),
        ("parse_error", False),
    ],
)
def test_raw_packet_rejects_noncanonical_json_scalar_types(tmp_path, field, value):
    session = _session(tmp_path)
    raw_path = _write_raw(session, [_raw_record(1, _payload_for_host(1))])
    records = _read_raw_records(raw_path)
    records[2][field] = value
    _replace_raw_records(raw_path, records)
    report = session_quality.analyze_session(session)
    assert report["checks"]["raw_audit"]["status"] == "fail"
    assert report["metrics"]["raw_audit"]["malformed_error_count"] >= 1


@pytest.mark.parametrize("mutation", ["policy_extra", "protocol_missing", "packet_extra", "packet_missing"])
def test_raw_records_require_exact_field_sets(tmp_path, mutation):
    session = _session(tmp_path)
    raw_path = _write_raw(session, [_raw_record(1, _payload_for_host(1))])
    records = _read_raw_records(raw_path)
    if mutation == "policy_extra":
        records[0]["extra"] = 1
    elif mutation == "protocol_missing":
        records[1].pop("evidence_ref")
    elif mutation == "packet_extra":
        records[2]["extra"] = 1
    else:
        records[2].pop("parse_error")
    _replace_raw_records(raw_path, records)
    report = session_quality.analyze_session(session)
    assert report["checks"]["raw_audit"]["status"] == "fail"


@pytest.mark.parametrize(
    ("header_index", "field", "value"),
    [(0, "max_bytes", "1024"), (0, "record_valid", 1), (1, "wire_packet_size", 28.0)],
)
def test_raw_headers_reject_noncanonical_scalar_types(tmp_path, header_index, field, value):
    session = _session(tmp_path)
    raw_path = _write_raw(session, [_raw_record(1, _payload_for_host(1))])
    records = _read_raw_records(raw_path)
    records[header_index][field] = value
    _replace_raw_records(raw_path, records)
    report = session_quality.analyze_session(session)
    assert report["checks"]["raw_audit"]["status"] == "fail"


@pytest.mark.parametrize(
    ("field", "value"),
    [("payload_prefix_bytes", 0), ("max_bytes", 32)],
)
def test_raw_policy_must_satisfy_production_complete_wire_capture(tmp_path, field, value):
    session = _session(tmp_path)
    raw_path = _write_raw(session, [_raw_record(1, _payload_for_host(1))])
    records = _read_raw_records(raw_path)
    records[0][field] = value
    _replace_raw_records(raw_path, records)
    report = session_quality.analyze_session(session)
    assert report["checks"]["raw_audit"]["status"] == "fail"
    assert field in report["checks"]["raw_audit"]["message"]


@pytest.mark.parametrize("case", ["wire_truncated", "nonwire_full", "bad_prefix", "incomplete_full"])
def test_raw_packet_shape_must_be_generatable_by_production_writer(tmp_path, case):
    session = _session(tmp_path)
    raw_path = _write_raw(session, [_raw_record(1, _payload_for_host(1))])
    records = _read_raw_records(raw_path)
    packet = records[2]
    if case == "wire_truncated":
        packet.update(truncated=True, sha256="0" * 64, payload_hex=packet["payload_hex"][:2])
    elif case == "nonwire_full":
        packet.update(original_length=300, payload_hex=(b"x" * 300).hex())
    elif case == "bad_prefix":
        packet.update(
            original_length=300,
            truncated=True,
            sha256="0" * 64,
            payload_hex="00",
        )
    else:
        packet.update(original_length=40, payload_hex=(b"x" * 10).hex())
    _replace_raw_records(raw_path, records)
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert report["checks"]["raw_protocol_replay"]["status"] == "fail"


def test_raw_rotation_suffix_is_rejected_when_backup_count_is_zero(tmp_path):
    session = _session(tmp_path)
    main = _write_raw(
        session,
        [_raw_record(index, _payload_for_host(index)) for index in range(1, 4)],
    )
    records = _read_raw_records(main)
    records[0]["backup_count"] = 0
    _replace_raw_records(session / "raw_packets.jsonl.1", records[:3])
    _replace_raw_records(main, records[3:])
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert "backup_count" in report["checks"]["raw_completeness"]["message"]


@pytest.mark.parametrize(("suffix", "backup_count"), [(2, 1), (2, 3)])
def test_raw_rotation_suffix_out_of_range_or_gap_is_rejected(
    tmp_path, suffix, backup_count
):
    session = _session(tmp_path)
    main = _write_raw(
        session,
        [_raw_record(index, _payload_for_host(index)) for index in range(1, 4)],
    )
    records = _read_raw_records(main)
    records[0]["backup_count"] = backup_count
    _replace_raw_records(session / f"raw_packets.jsonl.{suffix}", records)
    main.write_text("", encoding="utf-8")
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert report["checks"]["raw_completeness"]["status"] == "fail"


def test_raw_file_must_not_exceed_declared_policy_max_bytes(tmp_path):
    session = _session(tmp_path)
    raw_path = _write_raw(
        session,
        [_raw_record(index, _payload_for_host(index)) for index in range(1, 4)],
    )
    records = _read_raw_records(raw_path)
    records[0]["max_bytes"] = 1
    _replace_raw_records(raw_path, records)
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert "max_bytes" in report["checks"]["raw_audit"]["message"]


def test_raw_rotation_missing_prefix_is_partial_and_fails_closed(tmp_path):
    session = _session(tmp_path)
    metadata_path = session / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["raw_audit"] = {"enabled": True}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    records = [_raw_record(index, _payload_for_host(index)) for index in (2, 3)]
    _write_raw(session, records, filename="raw_packets.jsonl.1")
    _write_raw(session, [], filename="raw_packets.jsonl", include_headers=False)
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert report["metrics"]["raw_audit"]["state"] == "partial"
    assert report["checks"]["raw_completeness"]["status"] == "fail"
    assert "prefix" in report["checks"]["raw_completeness"]["message"]


def test_raw_claimed_enabled_but_missing_fails_closed(tmp_path):
    session = _session(tmp_path)
    metadata_path = session / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["raw_audit"] = {"enabled": True}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert report["metrics"]["raw_audit"]["state"] == "partial"
    assert report["checks"]["raw_completeness"]["status"] == "fail"


def test_raw_nonzero_padding_requires_parse_error_and_identical_is_not_retransmission(tmp_path):
    first_values = tuple(10 * channel for channel in range(1, 9))
    session = _session(
        tmp_path,
        channel_rows=[
            first_values,
            first_values,
            tuple(value + 2 for value in first_values),
        ],
    )
    repeated = _payload_for_host(1)
    nonzero_padding = _payload_for_host(3)[:-1] + b"\x01"
    _write_raw(session, [_raw_record(1, repeated), _raw_record(2, repeated), _raw_record(3, nonzero_padding)])
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert report["checks"]["raw_protocol_replay"]["status"] == "fail"
    assert report["metrics"]["raw_audit"]["adjacent_identical_payload_count"] == 1
    assert "相同值不可断言重传" in report["checks"]["adjacent_identical_payloads"]["message"]
    assert report["checks"]["adjacent_identical_payloads"]["details"]["retransmission_asserted"] is False


def test_warning_has_distinct_exit_code_and_cli_forces_utf8_on_chinese_path(tmp_path):
    first_values = tuple(10 * channel for channel in range(1, 9))
    session = _session(
        tmp_path,
        root_name="中文数据",
        channel_rows=[first_values, first_values, tuple(value + 2 for value in first_values)],
    )
    repeated = _payload_for_host(1)
    _write_raw(session, [_raw_record(1, repeated), _raw_record(2, repeated), _raw_record(3, _payload_for_host(3))])
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp936"
    completed = subprocess.run(
        [sys.executable, str(Path(session_quality.__file__)), str(session)],
        capture_output=True,
        env=env,
        check=False,
    )
    payload = json.loads(completed.stdout.decode("utf-8"))
    assert completed.returncode == session_quality.EXIT_WARNING
    assert payload["overall_status"] == "warning"
    assert payload["training_usable"] is False
    assert "中文数据" in payload["session_dir"]


@pytest.mark.parametrize("tamper", ["channel", "host_monotonic_ns", "connection_generation"])
def test_raw_replay_rejects_semantic_mismatch_with_csv(tmp_path, tamper):
    session = _session(tmp_path)
    records = [_raw_record(index, _payload_for_host(index)) for index in range(1, 4)]
    if tamper == "channel":
        _mutate_csv(
            session,
            lambda fields, rows: rows[0].update(channel_1=str(int(rows[0]["channel_1"]) + 1)),
        )
    else:
        records[1][tamper] += 1
    _write_raw(session, records)
    report = session_quality.analyze_session(session)
    assert report["checks"]["canonical_csv"]["status"] == "pass"
    assert report["checks"]["raw_csv_mapping"]["status"] == "fail"
    assert report["overall_status"] == "fail"


def test_oversize_csv_fails_closed_without_reading_rows(tmp_path, monkeypatch):
    session = _session(tmp_path)
    monkeypatch.setattr(session_quality, "MAX_CSV_BYTES", 1)
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert report["checks"]["input_limits"]["status"] == "fail"
    assert report["metrics"]["csv_rows"] == 0


def test_huge_uint64_boundary_is_checked_without_materializing_range(tmp_path):
    session = _session(tmp_path)
    metadata_path = session / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["session_boundary"]["end_host_receive_index"] = (1 << 64) - 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _write_raw(session, [_raw_record(index, _payload_for_host(index)) for index in range(1, 4)])
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert report["checks"]["raw_completeness"]["status"] == "fail"


def test_invalid_utf8_raw_returns_structured_failure_for_api_and_cli(tmp_path):
    session = _session(tmp_path, root_name="中文坏包")
    raw_path = _write_raw(
        session,
        [_raw_record(index, _payload_for_host(index)) for index in range(1, 4)],
    )
    with raw_path.open("ab") as stream:
        stream.write(b"\xff\xfe\n")
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert report["checks"]["raw_audit"]["status"] == "fail"
    assert "UnicodeDecodeError" in report["checks"]["raw_audit"]["message"]

    completed = subprocess.run(
        [sys.executable, str(Path(session_quality.__file__)), str(session)],
        capture_output=True,
        check=False,
    )
    payload = json.loads(completed.stdout.decode("utf-8"))
    assert completed.returncode == session_quality.EXIT_FAIL
    assert payload["checks"]["raw_audit"]["status"] == "fail"


def test_raw_physical_line_limit_counts_malformed_lines(tmp_path, monkeypatch):
    session = _session(tmp_path)
    raw_path = session / "raw_packets.jsonl"
    raw_path.write_text("not-json\nnot-json\nnot-json\n", encoding="utf-8")
    monkeypatch.setattr(session_quality, "MAX_RAW_RECORDS", 1)
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert report["metrics"]["raw_audit"]["physical_line_count"] == 2
    assert report["metrics"]["raw_audit"]["malformed_error_count"] == 1
    assert "physical line count" in report["checks"]["raw_completeness"]["message"]


def test_raw_single_line_byte_limit_fails_bounded(tmp_path, monkeypatch):
    session = _session(tmp_path)
    raw_path = session / "raw_packets.jsonl"
    raw_path.write_bytes(b"x" * 100 + b"\n")
    monkeypatch.setattr(session_quality, "MAX_RAW_LINE_BYTES", 16)
    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert report["metrics"]["raw_audit"]["physical_line_count"] == 1
    assert report["metrics"]["raw_audit"]["malformed_error_count"] == 1
    assert "line exceeds hard byte limit" in report["checks"]["raw_audit"]["message"]


@pytest.mark.parametrize("location", ["metadata", "raw"])
@pytest.mark.parametrize(
    ("malformed_json", "error_type"),
    [
        ('{"value":' + "9" * 5_000 + "}", "ValueError"),
        ("[" * 2_000 + "0" + "]" * 2_000, "RecursionError"),
    ],
)
def test_json_resource_errors_are_structured_for_api_and_cli(
    tmp_path, location, malformed_json, error_type
):
    session = _session(tmp_path, root_name=f"json-{location}-{error_type}")
    if location == "metadata":
        (session / "metadata.json").write_text(malformed_json, encoding="utf-8")
        expected_check = "metadata_file"
    else:
        raw_path = _write_raw(
            session,
            [_raw_record(index, _payload_for_host(index)) for index in range(1, 4)],
        )
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write(malformed_json + "\n")
        expected_check = "raw_audit"

    report = session_quality.analyze_session(session)
    assert report["overall_status"] == "fail"
    assert report["checks"][expected_check]["status"] == "fail"
    assert error_type in report["checks"][expected_check]["message"]
    if location == "raw":
        assert report["metrics"]["raw_audit"]["malformed_error_count"] == 1

    completed = subprocess.run(
        [sys.executable, str(Path(session_quality.__file__)), str(session)],
        capture_output=True,
        check=False,
    )
    payload = json.loads(completed.stdout.decode("utf-8"))
    assert completed.returncode == session_quality.EXIT_FAIL
    assert payload["checks"][expected_check]["status"] == "fail"
    assert error_type in payload["checks"][expected_check]["message"]


@pytest.mark.parametrize("location", ["metadata", "raw"])
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_constants_are_structured_for_api_and_cli(
    tmp_path, location, constant
):
    session = _session(tmp_path, root_name=f"nonfinite-{location}-{constant.strip('-')}")
    if location == "metadata":
        malformed = '{"value":' + constant + "}"
        (session / "metadata.json").write_text(malformed, encoding="utf-8")
        expected_check = "metadata_file"
    else:
        raw_path = _write_raw(
            session,
            [_raw_record(index, _payload_for_host(index)) for index in range(1, 4)],
        )
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write('{"value":' + constant + "}\n")
        expected_check = "raw_audit"
    report = session_quality.analyze_session(session)
    assert report["checks"][expected_check]["status"] == "fail"
    assert "ValueError" in report["checks"][expected_check]["message"]
    completed = subprocess.run(
        [sys.executable, str(Path(session_quality.__file__)), str(session)],
        capture_output=True,
        check=False,
    )
    payload = json.loads(completed.stdout.decode("utf-8"))
    assert completed.returncode == session_quality.EXIT_FAIL
    assert payload["checks"][expected_check]["status"] == "fail"
