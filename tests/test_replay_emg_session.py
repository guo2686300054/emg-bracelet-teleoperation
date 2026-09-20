from __future__ import annotations

import csv
import hashlib
import json
import threading

import pytest

from legacy_dataset import CHANNEL_COLUMNS, import_legacy_dataset
from replay_emg_session import ReplayInputError, replay_session
from shared_memory_v2 import (
    FLAG_CONNECTION_GENERATION_VALID,
    FLAG_DEVICE_PACKET_SEQUENCE_VALID,
    FLAG_DEVICE_TIME_VALID,
    FLAG_DISCONNECTED,
    FLAG_REPLAY,
    FLAG_SYNTHETIC_TIME,
    SharedMemoryProtocolError,
    SharedMemoryReader,
    SharedMemoryWriter,
)


def write_derived(tmp_path, *, rows=5):
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True)
    source = source_dir / "client_data.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(CHANNEL_COLUMNS)
        for index in range(rows):
            writer.writerow([(index + channel) % 256 for channel in range(1, 9)])
    annotation = {
        "schema": "emg_session_annotations",
        "version": "1.1",
        "source_file": source.name,
        "source_directory_name": source_dir.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_size_bytes": source.stat().st_size,
        "source_row_count": rows,
        "session_id": "legacy-session",
        "status": "completed",
        "training_usable_raw": False,
        "split_group_id": "subject-1",
        "full_session_group_id": "subject-1:legacy-session",
        "legacy_write_factor": 1,
        "segments": [{
            "start_row": 0, "end_row_exclusive": rows,
            "action_label": "rest", "action_phase": "rest",
            "confidence": "reported", "training_usable_raw": False,
            "eligible_for_future_training_review": True, "basis": "test",
        }],
    }
    annotation_path = source_dir / "annotations.json"
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    result = import_legacy_dataset(annotation_path, tmp_path / "session")
    return result.derived_csv


def test_replay_writes_readable_flagged_terminal_and_honest_report(tmp_path):
    source = write_derived(tmp_path, rows=4)
    shared = tmp_path / "replay.bin"
    report = replay_session(source, shared, assumed_sample_rate_hz=200.0)
    assert report.status == "completed"
    assert report.training_usable is False
    assert report.frames_written == 4
    assert report.final_sequence == 5
    assert report.end_marker_written
    assert report.source_csv_sha256 and report.manifest_sha256
    assert report.replay_code_version == "2.0"
    with SharedMemoryReader(shared) as reader:
        frame = reader.read()
        assert frame is not None
        assert frame.sequence == report.final_sequence
        assert frame.flags & FLAG_DISCONNECTED
        assert frame.flags & FLAG_REPLAY
        assert frame.flags & FLAG_SYNTHETIC_TIME
        assert frame.flags & FLAG_CONNECTION_GENERATION_VALID
        assert not frame.flags & FLAG_DEVICE_PACKET_SEQUENCE_VALID
        assert not frame.flags & FLAG_DEVICE_TIME_VALID
        assert not frame.is_live_control_eligible(1)
        with pytest.raises(SharedMemoryProtocolError, match="live control"):
            frame.require_live_control_eligible(1)


def test_cancel_wakes_and_writes_terminal_even_before_first_sample(tmp_path):
    source = write_derived(tmp_path, rows=5)
    cancel = threading.Event()
    cancel.set()
    report = replay_session(
        source, tmp_path / "cancel.bin", assumed_sample_rate_hz=200,
        cancel_event=cancel, realtime=True,
    )
    assert report.status == "cancelled"
    assert report.frames_written == 0
    assert report.final_sequence == 1
    assert report.end_marker_written
    with SharedMemoryReader(tmp_path / "cancel.bin") as reader:
        assert reader.read().flags & FLAG_DISCONNECTED


def test_data_and_cleanup_errors_preserved_while_terminal_is_written(tmp_path):
    source = write_derived(tmp_path, rows=1)

    class FailingWriter:
        def __init__(self, path):
            self.real = SharedMemoryWriter(path)
            self.generation = self.real.generation
            self.calls = 0

        def write_frame(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("primary write failed")
            return self.real.write_frame(*args, **kwargs)

        def close(self):
            self.real.close()
            raise RuntimeError("cleanup failed")

    report = replay_session(
        source, tmp_path / "failure.bin", assumed_sample_rate_hz=200,
        writer_factory=FailingWriter,
    )
    assert report.status == "failed"
    assert report.end_marker_written
    assert "primary write failed" in report.errors[0]
    assert "cleanup failed" in report.errors[1]
    with SharedMemoryReader(tmp_path / "failure.bin") as reader:
        assert reader.read().flags & FLAG_DISCONNECTED


@pytest.mark.parametrize("rate", [0, 9.9, 5000.1, float("nan"), True])
def test_invalid_or_unreasonable_rate_rejected(tmp_path, rate):
    source = write_derived(tmp_path)
    with pytest.raises(ReplayInputError, match="between 10 and 5000"):
        replay_session(source, tmp_path / "bad.bin", assumed_sample_rate_hz=rate)


def test_public_validator_rejects_tamper_and_missing_manifest(tmp_path):
    source = write_derived(tmp_path)
    source.write_text(source.read_text(encoding="utf-8") + "tamper", encoding="utf-8")
    with pytest.raises(ReplayInputError, match="validation failed"):
        replay_session(source, tmp_path / "bad.bin", assumed_sample_rate_hz=200)
    missing = tmp_path / "missing" / "derived_samples.csv"
    missing.parent.mkdir()
    missing.write_text("x\n", encoding="utf-8")
    with pytest.raises(ReplayInputError, match="validation failed"):
        replay_session(missing, tmp_path / "missing.bin", assumed_sample_rate_hz=200)


def test_classifier_rate_must_match_exactly(tmp_path):
    from test_realtime_emg_inference import make_classifier
    from realtime_emg_inference import RealtimeDecisionFilter

    source = write_derived(tmp_path, rows=40)
    decision = RealtimeDecisionFilter(make_classifier())
    with pytest.raises(ReplayInputError, match="exactly match"):
        replay_session(
            source, tmp_path / "rate.bin", assumed_sample_rate_hz=100,
            decision_filter=decision,
        )


def test_progress_and_generations(tmp_path):
    source = write_derived(tmp_path, rows=2)
    progress = []
    first = replay_session(
        source, tmp_path / "one.bin", assumed_sample_rate_hz=100,
        progress_callback=lambda done, total: progress.append((done, total)),
    )
    second = replay_session(source, tmp_path / "two.bin", assumed_sample_rate_hz=100)
    assert progress == [(1, 2), (2, 2)]
    assert first.generation != second.generation
    assert first.final_sequence == second.final_sequence == 3


@pytest.mark.parametrize("mutation", ["csv", "manifest", "delete"])
def test_validation_closure_change_during_replay_fails_closed(tmp_path, mutation):
    source = write_derived(tmp_path, rows=3)
    manifest = source.with_name("transform_manifest.json")
    annotation = tmp_path / "source" / "annotations.json"
    original_csv_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    original_manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    changed = False

    def mutate(done, total):
        nonlocal changed
        if changed or done != 1:
            return
        changed = True
        if mutation == "csv":
            source.write_bytes(source.read_bytes() + b"\n")
        elif mutation == "manifest":
            manifest.write_bytes(manifest.read_bytes() + b" ")
        else:
            annotation.unlink()

    report = replay_session(
        source, tmp_path / f"changed-{mutation}.bin",
        assumed_sample_rate_hz=200, progress_callback=mutate,
    )
    assert report.status == "failed"
    assert report.end_marker_written
    assert any("source_changed_during_replay" in item for item in report.errors)
    assert report.source_csv_sha256 == original_csv_hash
    assert report.manifest_sha256 == original_manifest_hash
