import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import training_dataset
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
from training_dataset import TrainingDatasetError, load_session, prepare_training_dataset


DEV = DeviceKey("dev-0123456789abcdef0123456789abcdef")
FLAGS = (
    QualityFlags.VALID
    | QualityFlags.HOST_WALL_TIME_VALID
    | QualityFlags.HOST_MONOTONIC_VALID
    | QualityFlags.HOST_RECEIVE_INDEX_VALID
)


class TrainingDatasetTests(unittest.TestCase):
    def make_session(
        self,
        root,
        subject_number,
        session_id,
        labels,
        *,
        rate=200.0,
        phases=None,
        channels=8,
        sample_format="float32",
        protocol=None,
        side=HandSide.LEFT,
        evidence_ref="device-spec-v1",
    ):
        subject_id = f"sub-{subject_number:032x}"
        acquisition = AcquisitionMetadata(
            sample_rate=RateDescriptor(rate, "protocol", evidence_ref, True),
            host_observed_rate=RateDescriptor(51.0, "host_observed", "capture-1", True),
            signal_chain=SignalChain(sample_format=sample_format),
            notification_packet_protocol=protocol or NotificationPacketProtocol(),
        )
        phases = phases or ["hold"] * len(labels)
        if not labels or len(phases) != len(labels):
            raise ValueError("test fixture requires equally sized, non-empty labels and phases")
        recording_context = RecordingContext(
            subject_id,
            labels[0],
            phases[0],
            "training_dataset_test_v1",
            side,
            "synthetic_test",
        )
        recorder = DataRecorder(
            root,
            subject_id=subject_id,
            device_id=DEV,
            acquisition=acquisition,
            channels=channels,
            side=side,
            session_id=session_id,
            flush_every=1,
            recording_context=recording_context,
        )
        recorder.configure_session_boundary(start_host_receive_index=0, start_drop_total=0)
        for index, (label, phase) in enumerate(zip(labels, phases)):
            values = tuple(10 + index + channel for channel in range(channels))
            if sample_format == "float32":
                values = tuple(float(value) for value in values)
            recorder.record(
                EmgFrame(
                    values,
                    1_000_000 + index,
                    2_000_000 + index,
                    index + 1,
                    index,
                    action_label=label,
                    action_phase=phase,
                    quality_flags=FLAGS,
                )
            )
        recorder.update_session_boundary(
            end_host_receive_index=len(labels),
            received_count=len(labels),
            eligible_count=len(labels),
            written_count=len(labels),
            queue_drop_total=0,
            queue_drop_session=0,
            tail_pending_count=0,
            tail_loss_count=0,
            incomplete_reason=None,
        )
        recorder.close()
        return recorder.session_dir

    @staticmethod
    def mutate_metadata(session_dir, mutation):
        path = Path(session_dir) / "metadata.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutation(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def mutate_csv(session_dir, row_index, field, value):
        path = Path(session_dir) / "samples.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fields = reader.fieldnames
        rows[row_index][field] = value
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @classmethod
    def mutate_annotation(cls, session_dir, *, label=None, phase=None):
        path = Path(session_dir) / "samples.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            row_count = sum(1 for _ in csv.DictReader(stream))
        for row_index in range(row_count):
            if label is not None:
                cls.mutate_csv(session_dir, row_index, "action_label", label)
            if phase is not None:
                cls.mutate_csv(session_dir, row_index, "action_phase", phase)

        def update_metadata(payload):
            if label is not None:
                payload["extra"]["action_label"] = label
            if phase is not None:
                payload["extra"]["action_phase"] = phase

        cls.mutate_metadata(session_dir, update_metadata)

    @staticmethod
    def write_raw(session_dir, row_count, *, tamper_first_channel=False):
        session_dir = Path(session_dir)
        metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
        protocol = metadata["acquisition"]["notification_packet_protocol"]
        records = [
            {
                "record_type": "raw_audit_policy",
                "max_bytes": 1024 * 1024,
                "backup_count": 3,
                "payload_prefix_bytes": 256,
                "record_valid": True,
                "record_invalid": True,
                "retention_policy": "bounded_rotating_files",
                "complete_payload_guarantee": "exact_configured_wire_size",
                "non_wire_length_policy": "full_if_limits_allow_else_prefix_length_sha256",
            },
            {"record_type": "notification_packet_protocol", **protocol},
        ]
        for row_index in range(row_count):
            payload = bytearray(16)
            for channel_index in range(8):
                payload[channel_index * 2 + 1] = 10 + row_index + channel_index
            if row_index == 0 and tamper_first_channel:
                payload[1] += 1
            records.append(
                {
                    "host_wall_timestamp_ns": 1_000_000 + row_index,
                    "host_monotonic_ns": 2_000_000 + row_index,
                    "host_receive_index": row_index + 1,
                    "connection_generation": 0,
                    "wire_protocol_mode": protocol["mode"],
                    "original_length": len(payload),
                    "payload_hex": payload.hex(),
                    "parse_error": None,
                }
            )
        raw_path = session_dir / "raw_packets.jsonl"
        raw_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        return raw_path

    def test_formal_session_admission_requires_exactly_eight_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for channels in (1, 7, 9):
                session = self.make_session(
                    root,
                    channels,
                    f"channels-{channels}",
                    ["rest"] * 40,
                    channels=channels,
                )
                with self.assertRaisesRegex(TrainingDatasetError, "exactly 8"):
                    load_session(session)
            accepted = self.make_session(
                root, 8, "channels-8", ["rest"] * 40, channels=8
            )
            self.assertEqual(load_session(accepted).channel_count, 8)

    def test_success_manifest_windows_hashes_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = [
                self.make_session(root, number, f"session-{number}", ["rest"] * 4)
                for number in range(1, 4)
            ]
            output = root / "training.json"
            manifest = prepare_training_dataset(
                sessions, output, window_ms=10, step_ms=5, seed=23
            )
            self.assertEqual(manifest["schema"], "emg.training.dataset_manifest")
            self.assertEqual(manifest["version"], "1.1")
            self.assertEqual(manifest["window"]["window_size_samples"], 2)
            self.assertEqual(manifest["window"]["step_size_samples"], 1)
            self.assertEqual(manifest["data_contract"]["sample_rate"]["value_hz"], 200.0)
            self.assertEqual(
                manifest["data_contract"]["sample_rate"]["evidence_ref"], "device-spec-v1"
            )
            self.assertEqual(sum(item["window_count"] for item in manifest["inputs"]), 9)
            for item in manifest["inputs"]:
                self.assertEqual(len(item["source_files"]["metadata.json"]["sha256"]), 64)
                self.assertEqual(len(item["source_files"]["samples.csv"]["sha256"]), 64)
            completed = subprocess.run(
                [
                    sys.executable,
                    "training_dataset.py",
                    *map(str, sessions),
                    "--output",
                    str(root / "cli.json"),
                    "--window-ms",
                    "10",
                    "--step-ms",
                    "5",
                ],
                cwd=Path(__file__).parents[1],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("wrote training dataset manifest", completed.stdout)

    def test_empty_action_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.make_session(Path(directory), 1, "empty-label", ["rest"] * 2)
            self.mutate_annotation(session, label="")
            with self.assertRaisesRegex(TrainingDatasetError, "action_label"):
                load_session(session)

    def test_unknown_or_host_derived_sample_time_rate_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unknown = self.make_session(root, 1, "unknown-rate", ["rest"] * 2)
            self.mutate_metadata(
                unknown,
                lambda payload: (
                    payload["acquisition"]["sample_rate"].update(
                        value_hz=None,
                        source_kind="unknown",
                        evidence_ref="unknown",
                        confirmed=False,
                    ),
                    payload.update(sample_rate_hz=None),
                ),
            )
            with self.assertRaisesRegex(TrainingDatasetError, "confirmed sample.*rate"):
                load_session(unknown)

            host = self.make_session(root, 2, "host-rate", ["rest"] * 2, rate=51.0)
            self.mutate_metadata(
                host,
                lambda payload: payload["acquisition"]["sample_rate"].update(
                    source_kind="host_observed", evidence_ref="capture-1"
                ),
            )
            with self.assertRaisesRegex(TrainingDatasetError, "trusted allowlist"):
                load_session(host)

    def test_incomplete_session_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.make_session(Path(directory), 1, "incomplete", ["rest"] * 2)
            self.mutate_metadata(session, lambda payload: payload.update(status="incomplete"))
            with self.assertRaisesRegex(TrainingDatasetError, "not complete"):
                load_session(session)

    def test_training_provenance_is_explicit_versioned_and_required_for_training(self):
        for mode in ("missing", "unknown", "wrong_version"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                session = self.make_session(
                    Path(directory), 1, f"provenance-{mode}", ["rest"] * 2
                )

                def mutate(payload):
                    if mode == "missing":
                        payload.pop("training_provenance")
                    elif mode == "unknown":
                        payload["training_provenance"]["kind"] = "unknown"
                    else:
                        payload["training_provenance"]["version"] = "2.0"

                self.mutate_metadata(session, mutate)
                with self.assertRaisesRegex(
                    TrainingDatasetError, "missing training provenance|training provenance"
                ):
                    load_session(session)

    def test_index_and_time_regressions_are_rejected(self):
        for field, value, message in (
            ("session_sample_index", "0", "session_sample_index"),
            ("sample_index", "0", "sample_index"),
            ("host_monotonic_ns", "1", "monotonic"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                session = self.make_session(Path(directory), 1, field, ["rest"] * 3)
                self.mutate_csv(session, 1, field, value)
                with self.assertRaisesRegex(TrainingDatasetError, message):
                    load_session(session)

    def test_windows_never_cross_single_action_session_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = [
                self.make_session(root, number, f"rest-{number}", ["rest"] * 2)
                for number in range(1, 4)
            ] + [
                self.make_session(root, number, f"fist-{number}", ["fist"] * 2)
                for number in range(1, 4)
            ]
            manifest = prepare_training_dataset(
                sessions, root / "manifest.json", window_ms=10, step_ms=5
            )
            windows = [window for split in manifest["splits"].values() for window in split["windows"]]
            allowed = {("rest", 0, 2), ("fist", 0, 2)}
            self.assertTrue(all((item["label"], item["start_row"], item["end_row_exclusive"]) in allowed for item in windows))

    def test_group_leakage_is_impossible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = []
            for subject in range(1, 5):
                sessions.append(self.make_session(root, subject, f"s{subject}-a", ["rest"] * 2))
                sessions.append(self.make_session(root, subject, f"s{subject}-b", ["fist"] * 2))
            manifest = prepare_training_dataset(
                sessions, root / "manifest.json", window_ms=10, step_ms=10, seed=7
            )
            group_sets = [set(split["subject_ids"]) for split in manifest["splits"].values()]
            self.assertFalse(group_sets[0] & group_sets[1])
            self.assertFalse(group_sets[0] & group_sets[2])
            self.assertFalse(group_sets[1] & group_sets[2])
            locations = {}
            for split_name, split in manifest["splits"].items():
                for window in split["windows"]:
                    locations.setdefault(window["subject_id"], set()).add(split_name)
            self.assertTrue(all(len(names) == 1 for names in locations.values()))

    def test_fixed_seed_is_reproducible_independent_of_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = [
                self.make_session(root, number, f"s-{number}", ["rest"] * 2)
                for number in range(1, 7)
            ]
            first = prepare_training_dataset(
                sessions, root / "one.json", window_ms=10, step_ms=10, seed=99
            )
            second = prepare_training_dataset(
                list(reversed(sessions)), root / "two.json", window_ms=10, step_ms=10, seed=99
            )
            self.assertEqual(first["splits"], second["splits"])
            self.assertEqual(first["label_statistics"], second["label_statistics"])

    def test_path_escape_and_source_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self.make_session(root, 1, "safe", ["rest"] * 2)
            self.mutate_metadata(
                session, lambda payload: payload.update(csv_file="../samples.csv")
            )
            with self.assertRaisesRegex(TrainingDatasetError, "samples.csv"):
                load_session(session)

            session = self.make_session(root, 2, "safe-two", ["rest"] * 2)
            with self.assertRaisesRegex(TrainingDatasetError, "source data"):
                prepare_training_dataset(
                    [session], session / "manifest.json", window_ms=10, step_ms=10
                )

    def test_any_invalid_explicit_input_rejects_entire_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_sessions = [
                self.make_session(root, number, f"valid-{number}", ["rest"] * 2)
                for number in range(1, 4)
            ]
            invalid = self.make_session(root, 2, "bad", ["rest"] * 2)
            self.mutate_annotation(invalid, label="")
            output = root / "manifest.json"
            with self.assertRaisesRegex(TrainingDatasetError, "explicit input.*action_label"):
                prepare_training_dataset(
                    [*valid_sessions, invalid], output, window_ms=10, step_ms=10
                )
            self.assertFalse(output.exists())

    def test_session_grouping_is_disabled_in_api_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = [
                self.make_session(root, number, f"group-{number}", ["rest"] * 2)
                for number in range(1, 4)
            ]
            with self.assertRaisesRegex(TrainingDatasetError, "only stable subject"):
                prepare_training_dataset(
                    sessions,
                    root / "bad.json",
                    window_ms=10,
                    step_ms=10,
                    group_by="session",
                )
            completed = subprocess.run(
                [sys.executable, "training_dataset.py", *map(str, sessions), "--output", str(root / "cli.json"), "--window-ms", "10", "--step-ms", "10", "--group-by", "session"],
                cwd=Path(__file__).parents[1],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("unrecognized arguments", completed.stderr)

    def test_action_label_and_phase_allowlists_reject_non_training_rows(self):
        for label in (
            "", "unknown", "unlabeled", "transition", "excluded", "grip",
            "open", "pinch", "wrist_flex",
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                session = self.make_session(Path(directory), 1, "bad-label", ["rest"])
                self.mutate_annotation(session, label=label)
                with self.assertRaisesRegex(TrainingDatasetError, "action_label"):
                    load_session(session)
        for phase in ("", "unknown", "transition", "excluded"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                session = self.make_session(Path(directory), 1, "bad-phase", ["rest"])
                self.mutate_annotation(session, phase=phase)
                with self.assertRaisesRegex(TrainingDatasetError, "action_phase"):
                    load_session(session)

    def test_session_boundary_must_be_lossless_and_match_csv(self):
        cases = (
            ("start_host_receive_index", 999),
            ("end_host_receive_index", 1000),
            ("received_count", 1),
            ("eligible_count", 1),
            ("written_count", 1),
            ("queue_drop_total", 1),
            ("queue_drop_session", 1),
            ("tail_pending_count", 1),
            ("tail_loss_count", 1),
            ("incomplete_reason", "tail not drained"),
        )
        for field, value in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                session = self.make_session(Path(directory), 1, f"boundary-{field}", ["rest"] * 2)
                self.mutate_metadata(
                    session,
                    lambda payload, field=field, value=value: payload["session_boundary"].update({field: value}),
                )
                with self.assertRaisesRegex(TrainingDatasetError, "session_boundary"):
                    load_session(session)

    def test_boundary_uses_open_start_and_inclusive_end_host_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = self.make_session(root, 1, "canonical-boundary", ["rest"] * 2)
            loaded = load_session(canonical)
            self.assertEqual(loaded.row_count, 2)

            legacy_style = self.make_session(root, 2, "legacy-boundary", ["rest"] * 2)
            self.mutate_csv(legacy_style, 0, "host_receive_index", "0")
            self.mutate_csv(legacy_style, 1, "host_receive_index", "1")
            with self.assertRaisesRegex(
                TrainingDatasetError, "host_receive_index.*strictly continuous"
            ):
                load_session(legacy_style)

    def test_canonical_stream_validator_is_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.make_session(Path(directory), 1, "bad-packet", ["rest"] * 2)
            self.mutate_csv(session, 0, "sample_in_packet", "1")
            with self.assertRaisesRegex(TrainingDatasetError, "canonical stream validation"):
                load_session(session)

    def test_one_notification_one_sample_and_continuous_host_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = self.make_session(root, 1, "duplicate-host", ["rest"] * 3)
            self.mutate_csv(duplicate, 1, "host_receive_index", "1")
            with self.assertRaisesRegex(
                TrainingDatasetError, "canonical stream validation|strictly continuous"
            ):
                load_session(duplicate)

            gap = self.make_session(root, 2, "host-gap", ["rest"] * 3)
            self.mutate_csv(gap, 1, "host_receive_index", "3")
            with self.assertRaisesRegex(TrainingDatasetError, "strictly continuous"):
                load_session(gap)

            multiple = self.make_session(root, 3, "multiple-samples", ["rest"] * 2)
            self.mutate_csv(multiple, 1, "host_receive_index", "1")
            self.mutate_csv(multiple, 1, "host_wall_timestamp_ns", "1000000")
            self.mutate_csv(multiple, 1, "host_monotonic_ns", "2000000")
            self.mutate_csv(multiple, 1, "sample_in_packet", "1")
            with self.assertRaisesRegex(TrainingDatasetError, "sample_in_packet"):
                load_session(multiple)

    def test_hashes_come_from_same_reads_used_for_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.make_session(Path(directory), 1, "same-read", ["rest"] * 2)
            metadata_path = Path(session) / "metadata.json"
            samples_path = Path(session) / "samples.csv"

            def obsolete_second_pass(_path):
                samples_path.write_bytes(b"unvalidated replacement")
                return "0" * 64

            with mock.patch.object(
                training_dataset, "_sha256", create=True, side_effect=obsolete_second_pass
            ) as obsolete:
                loaded = load_session(session)
            obsolete.assert_not_called()
            self.assertEqual(
                loaded.metadata_sha256,
                hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                loaded.samples_sha256,
                hashlib.sha256(samples_path.read_bytes()).hexdigest(),
            )

    def test_source_mutation_during_validation_is_never_certified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_session = self.make_session(root, 1, "metadata-race", ["rest"])
            metadata_path = Path(metadata_session) / "metadata.json"
            original_rate_validator = training_dataset._confirmed_sample_rate

            def mutate_metadata_after_decode(metadata, schema_minor):
                result = original_rate_validator(metadata, schema_minor)
                metadata_path.write_bytes(metadata_path.read_bytes() + b" ")
                return result

            with mock.patch.object(
                training_dataset,
                "_confirmed_sample_rate",
                side_effect=mutate_metadata_after_decode,
            ):
                with self.assertRaisesRegex(TrainingDatasetError, "metadata.json changed"):
                    load_session(metadata_session)

            csv_session = self.make_session(root, 2, "csv-race", ["rest"])
            samples_path = Path(csv_session) / "samples.csv"
            original_row_validator = training_dataset._RecoveryStreamValidator.validate
            validation_calls = 0

            def mutate_csv_after_row(validator, row):
                nonlocal validation_calls
                result = original_row_validator(validator, row)
                validation_calls += 1
                if validation_calls == 2:
                    content = samples_path.read_bytes()
                    samples_path.write_bytes(content.replace(b"rest", b"fist", 1))
                return result

            with mock.patch.object(
                training_dataset._RecoveryStreamValidator,
                "validate",
                new=mutate_csv_after_row,
            ):
                with self.assertRaisesRegex(TrainingDatasetError, "samples.csv changed"):
                    load_session(csv_session)

    def test_sample_rate_allowlist_is_shared_with_quality_gate(self):
        import session_quality
        from training_contract import validate_sample_rate_source_kind

        self.assertIs(
            training_dataset.validate_sample_rate_source_kind,
            validate_sample_rate_source_kind,
        )
        self.assertIs(
            session_quality.validate_sample_rate_source_kind,
            validate_sample_rate_source_kind,
        )

    def test_cross_session_data_contract_mismatch_is_rejected(self):
        protocol = NotificationPacketProtocol(
            mode=PADDED28_PROTOCOL,
            wire_packet_size=28,
            logical_packet_size=16,
            padding_rule="zero_suffix",
            evidence_ref="raw-audit-v3",
        )
        variants = (
            {"sample_format": "uint8"},
            {"protocol": protocol},
            {"side": HandSide.RIGHT},
            {"rate": 250.0},
            {"evidence_ref": "device-spec-v2"},
        )
        for index, variant in enumerate(variants, 1):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                baseline = self.make_session(root, 1, "baseline", ["rest"] * 2)
                changed = self.make_session(root, 2, f"changed-{index}", ["rest"] * 2, **variant)
                with self.assertRaisesRegex(TrainingDatasetError, "all sessions must share"):
                    prepare_training_dataset(
                        [baseline, changed], root / "manifest.json", window_ms=10, step_ms=10
                    )

    def test_full_signal_chain_semantics_are_part_of_the_contract(self):
        for field, value in (("unit", "mV"), ("scaling", "gain=2"), ("filter", "20-450Hz")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                baseline = self.make_session(root, 1, "baseline", ["rest"] * 2)
                changed = self.make_session(root, 2, "changed", ["rest"] * 2)
                self.mutate_metadata(
                    changed,
                    lambda payload, field=field, value=value: payload["acquisition"]["signal_chain"].update({field: value}),
                )
                with self.assertRaisesRegex(TrainingDatasetError, "all sessions must share"):
                    prepare_training_dataset(
                        [baseline, changed], root / "manifest.json", window_ms=10, step_ms=10
                    )

    def test_every_split_must_cover_every_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = [
                self.make_session(root, 1, "rest-a", ["rest"] * 2),
                self.make_session(root, 2, "rest-b", ["rest"] * 2),
                self.make_session(root, 3, "fist", ["fist"] * 2),
            ]
            with self.assertRaisesRegex(TrainingDatasetError, "label coverage"):
                prepare_training_dataset(
                    sessions, root / "manifest.json", window_ms=10, step_ms=10
                )

    def test_source_label_cannot_disappear_because_its_run_is_too_short(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = [
                self.make_session(root, number, f"short-{number}", ["fist"])
                for number in range(1, 4)
            ]
            with self.assertRaisesRegex(TrainingDatasetError, r"no complete label\+phase window"):
                prepare_training_dataset(
                    sessions, root / "manifest.json", window_ms=10, step_ms=10
                )

    def test_manifest_uses_relative_content_facts_and_no_raw_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = [
                self.make_session(root, number, f"portable-{number}", ["rest"] * 2)
                for number in range(1, 4)
            ]
            manifest = prepare_training_dataset(
                sessions, root / "manifest.json", window_ms=10, step_ms=10
            )
            rendered = json.dumps(manifest)
            self.assertNotIn(str(root.resolve()), rendered)
            for split in manifest["splits"].values():
                for window in split["windows"]:
                    self.assertEqual(
                        set(window),
                        {"window_id", "session_id", "subject_id", "label", "action_phase", "start_row", "end_row_exclusive", "samples_sha256"},
                    )
                    self.assertEqual(len(window["window_id"]), 64)
            loaded = load_session(sessions[0])
            self.assertEqual(loaded.action_runs, (("rest", "hold", 0, 2),))
            self.assertFalse(hasattr(loaded, "labels_and_phases"))

    def test_errors_do_not_disclose_absolute_input_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "secret" / "missing-session"
            try:
                load_session(missing)
            except TrainingDatasetError as error:
                self.assertNotIn(str(Path(directory).resolve()), str(error))
            else:
                self.fail("missing session should be rejected")

    def test_metadata_resource_limit_is_enforced_before_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.make_session(Path(directory), 1, "bounded", ["rest"])
            with mock.patch("training_dataset.MAX_METADATA_BYTES", 1):
                with self.assertRaisesRegex(TrainingDatasetError, "resource limit"):
                    load_session(session)

    def test_pathological_json_is_wrapped_for_api_and_cli(self):
        malicious_documents = {
            "oversized_integer": b'{"value":' + (b"9" * 5000) + b"}",
            "deep_nesting": (b"[" * 2000) + b"0" + (b"]" * 2000),
            "nonfinite_extra": b'{"extra":{"bad":NaN}}',
        }
        for name, document in malicious_documents.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                session = self.make_session(root, 1, name, ["rest"])
                (Path(session) / "metadata.json").write_bytes(document)
                self.assertLess(len(document), 1024 * 1024)
                with self.assertRaisesRegex(
                    TrainingDatasetError, "cannot read metadata.json"
                ):
                    load_session(session)

                completed = subprocess.run(
                    [
                        sys.executable,
                        "training_dataset.py",
                        str(session),
                        "--output",
                        str(root / "manifest.json"),
                        "--window-ms",
                        "10",
                        "--step-ms",
                        "10",
                    ],
                    cwd=Path(__file__).parents[1],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("training dataset error:", completed.stderr)
                self.assertIn("cannot read metadata.json", completed.stderr)
                self.assertNotIn("Traceback", completed.stderr)

    def test_row_session_window_and_dataset_limits_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self.make_session(root, 1, "rows", ["rest"] * 2)
            with mock.patch("training_dataset.MAX_CSV_ROWS_PER_SESSION", 1):
                with self.assertRaisesRegex(TrainingDatasetError, "row_count"):
                    load_session(session)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = [
                self.make_session(root, number, f"bounded-{number}", ["rest"])
                for number in range(1, 4)
            ]
            with mock.patch("training_dataset.MAX_SESSIONS", 2):
                with self.assertRaisesRegex(TrainingDatasetError, "session count"):
                    prepare_training_dataset(
                        sessions, root / "sessions.json", window_ms=5, step_ms=5
                    )
            with mock.patch("training_dataset.MAX_DATASET_ROWS", 2):
                with self.assertRaisesRegex(TrainingDatasetError, "total row"):
                    prepare_training_dataset(
                        sessions, root / "rows.json", window_ms=5, step_ms=5
                    )
            with mock.patch("training_dataset.MAX_MANIFEST_WINDOWS", 2):
                with self.assertRaisesRegex(TrainingDatasetError, "window resource"):
                    prepare_training_dataset(
                        sessions, root / "windows.json", window_ms=5, step_ms=5
                    )

    def test_cli_error_is_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self.make_session(root, 1, "bad-cli", ["rest"])
            self.mutate_annotation(session, label="")
            completed = subprocess.run(
                [
                    sys.executable,
                    "training_dataset.py",
                    str(session),
                    "--output",
                    str(root / "out.json"),
                    "--window-ms",
                    "10",
                    "--step-ms",
                    "10",
                ],
                cwd=Path(__file__).parents[1],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("training dataset error:", completed.stderr)
            self.assertIn("action_label", completed.stderr)

    def test_quality_gate_is_mandatory_for_raw_and_unavailable_raw_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unavailable = self.make_session(
                root, 1, "raw-unavailable", ["rest"] * 3, channels=8
            )
            unavailable_report = session_quality.analyze_session(unavailable)
            self.assertTrue(unavailable_report["training_usable"])
            self.assertEqual(
                unavailable_report["metrics"]["raw_audit"]["state"], "not_available"
            )
            self.assertEqual(load_session(unavailable).row_count, 3)

            complete = self.make_session(
                root, 2, "raw-complete", ["rest"] * 3, channels=8
            )
            self.write_raw(complete, 3)
            complete_report = session_quality.analyze_session(complete)
            self.assertTrue(complete_report["training_usable"])
            self.assertEqual(
                complete_report["checks"]["raw_csv_mapping"]["status"], "pass"
            )
            self.assertEqual(load_session(complete).row_count, 3)

            tampered = self.make_session(
                root, 3, "raw-tampered", ["rest"] * 3, channels=8
            )
            self.write_raw(tampered, 3, tamper_first_channel=True)
            tampered_report = session_quality.analyze_session(tampered)
            self.assertFalse(tampered_report["training_usable"])
            self.assertEqual(
                tampered_report["checks"]["raw_csv_mapping"]["status"], "fail"
            )
            with self.assertRaisesRegex(TrainingDatasetError, "quality gate rejected"):
                load_session(tampered)

    def test_quality_snapshot_rejects_csv_and_metadata_changed_after_pass(self):
        for source in ("csv", "metadata"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                session = self.make_session(
                    root, 1, f"snapshot-{source}", ["rest"] * 3, channels=8
                )
                self.write_raw(session, 3)
                original_analyze = session_quality.analyze_session

                def analyze_then_mutate(path, source=source):
                    report = original_analyze(path)
                    self.assertTrue(report["training_usable"])
                    if source == "csv":
                        self.mutate_csv(session, 0, "channel_1", "99.0")
                    else:
                        self.mutate_metadata(
                            session,
                            lambda payload: payload["extra"].update(
                                protocol_version="snapshot-change"
                            ),
                        )
                    return report

                with mock.patch.object(
                    training_dataset.session_quality,
                    "analyze_session",
                    side_effect=analyze_then_mutate,
                ):
                    with self.assertRaisesRegex(
                        TrainingDatasetError, "sources changed after quality analysis"
                    ):
                        load_session(session)

    def test_quality_snapshot_rejects_raw_add_delete_and_modify_after_pass(self):
        for mutation in ("add", "delete", "modify"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                session = self.make_session(
                    root, 1, f"raw-snapshot-{mutation}", ["rest"] * 3, channels=8
                )
                raw_path = self.write_raw(session, 3)
                original_analyze = session_quality.analyze_session

                def analyze_then_mutate(path, mutation=mutation):
                    report = original_analyze(path)
                    self.assertTrue(report["training_usable"])
                    if mutation == "add":
                        (Path(session) / "raw_packets.jsonl.1").write_bytes(
                            raw_path.read_bytes()
                        )
                    elif mutation == "delete":
                        raw_path.unlink()
                    else:
                        raw_path.write_bytes(raw_path.read_bytes() + b" ")
                    return report

                with mock.patch.object(
                    training_dataset.session_quality,
                    "analyze_session",
                    side_effect=analyze_then_mutate,
                ):
                    with self.assertRaisesRegex(
                        TrainingDatasetError, "sources changed after quality analysis"
                    ):
                        load_session(session)

    def test_precommit_snapshot_rejects_changes_during_manifest_construction(self):
        for mutation in ("metadata", "csv", "raw_add", "raw_delete", "raw_modify"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sessions = [
                    self.make_session(
                        root,
                        number,
                        f"precommit-{mutation}-{number}",
                        ["rest"] * 3,
                        channels=8,
                    )
                    for number in range(1, 4)
                ]
                for session in sessions:
                    self.write_raw(session, 3)
                target = sessions[0]
                raw_path = Path(target) / "raw_packets.jsonl"
                output = root / "manifest.json"
                original_windows = training_dataset._windows
                changed = False

                def windows_then_mutate(loaded, size, step):
                    nonlocal changed
                    windows = original_windows(loaded, size, step)
                    if not changed:
                        changed = True
                        if mutation == "metadata":
                            self.mutate_metadata(
                                target,
                                lambda payload: payload["extra"].update(
                                    protocol_version="precommit-change"
                                ),
                            )
                        elif mutation == "csv":
                            self.mutate_csv(target, 0, "channel_1", "99.0")
                        elif mutation == "raw_add":
                            (Path(target) / "raw_packets.jsonl.1").write_bytes(
                                raw_path.read_bytes()
                            )
                        elif mutation == "raw_delete":
                            raw_path.unlink()
                        else:
                            raw_path.write_bytes(raw_path.read_bytes() + b" ")
                    return windows

                with mock.patch.object(
                    training_dataset, "_windows", side_effect=windows_then_mutate
                ):
                    with self.assertRaisesRegex(
                        TrainingDatasetError,
                        "changed before training admission|sources changed after quality analysis",
                    ):
                        prepare_training_dataset(
                            sessions, output, window_ms=10, step_ms=10
                        )
                self.assertFalse(output.exists())

    def test_manifest_publish_success_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = [
                self.make_session(root, number, f"publish-{number}", ["rest"] * 2)
                for number in range(1, 4)
            ]
            output = root / "manifest.json"
            prepare_training_dataset(sessions, output, window_ms=10, step_ms=10)
            self.assertTrue(output.is_file())
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["schema"],
                "emg.training.dataset_manifest",
            )
            self.assertEqual(list(root.glob(f".{output.name}.*.tmp")), [])

    def test_interrupted_write_and_cleanup_failure_are_structured(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = [
                self.make_session(root, number, f"interrupt-{number}", ["rest"] * 2)
                for number in range(1, 4)
            ]
            output = root / "manifest.json"
            original_unlink = Path.unlink

            def interrupted_dump(_payload, stream, **_kwargs):
                stream.write("{\n")
                raise OSError("injected write interruption")

            def deny_temp_cleanup(path, *args, **kwargs):
                if path.name.startswith(f".{output.name}.") and path.suffix == ".tmp":
                    raise PermissionError("injected cleanup failure")
                return original_unlink(path, *args, **kwargs)

            with mock.patch.object(training_dataset.json, "dump", interrupted_dump), mock.patch.object(
                Path, "unlink", deny_temp_cleanup
            ):
                with self.assertRaisesRegex(
                    TrainingDatasetError,
                    "cannot publish output manifest.*OSError.*residual temporary manifest cleanup failed",
                ) as raised:
                    prepare_training_dataset(sessions, output, window_ms=10, step_ms=10)
            self.assertNotIn(str(root.resolve()), str(raised.exception))
            self.assertFalse(output.exists())
            leftovers = list(root.glob(f".{output.name}.*.tmp"))
            self.assertEqual(len(leftovers), 1)
            original_unlink(leftovers[0])

    def test_fsync_failure_removes_temporary_and_final_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = [
                self.make_session(root, number, f"fsync-{number}", ["rest"] * 2)
                for number in range(1, 4)
            ]
            output = root / "manifest.json"
            with mock.patch.object(training_dataset.os, "fsync", side_effect=OSError("injected")):
                with self.assertRaisesRegex(
                    TrainingDatasetError, "cannot publish output manifest.*OSError"
                ):
                    prepare_training_dataset(sessions, output, window_ms=10, step_ms=10)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(f".{output.name}.*.tmp")), [])

    def test_post_publish_temporary_unlink_failure_warns_and_keeps_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = [
                self.make_session(root, number, f"unlink-once-{number}", ["rest"] * 2)
                for number in range(1, 4)
            ]
            output = root / "manifest.json"
            original_unlink = Path.unlink
            failed_once = False

            def fail_temp_once(path, *args, **kwargs):
                nonlocal failed_once
                if (
                    not failed_once
                    and path.name.startswith(f".{output.name}.")
                    and path.suffix == ".tmp"
                ):
                    failed_once = True
                    raise PermissionError("injected one-time cleanup failure")
                return original_unlink(path, *args, **kwargs)

            with mock.patch.object(
                training_dataset, "_WINDOWS_NO_REPLACE_RENAME", False
            ), mock.patch.object(Path, "unlink", fail_temp_once):
                with warnings.catch_warnings(), self.assertLogs(
                    training_dataset._LOGGER, level="WARNING"
                ) as captured:
                    warnings.simplefilter("error", RuntimeWarning)
                    prepare_training_dataset(sessions, output, window_ms=10, step_ms=10)
            self.assertTrue(failed_once)
            self.assertIn("orphaned temporary file", "\n".join(captured.output))
            self.assertTrue(output.is_file())
            leftovers = list(root.glob(f".{output.name}.*.tmp"))
            self.assertEqual(len(leftovers), 1)
            original_unlink(leftovers[0])

    def test_post_publish_persistent_temp_unlink_failure_is_only_a_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = [
                self.make_session(root, number, f"unlink-always-{number}", ["rest"] * 2)
                for number in range(1, 4)
            ]
            output = root / "manifest.json"
            original_unlink = Path.unlink

            def always_fail_temp(path, *args, **kwargs):
                if path.name.startswith(f".{output.name}.") and path.suffix == ".tmp":
                    raise PermissionError("injected persistent cleanup failure")
                return original_unlink(path, *args, **kwargs)

            with mock.patch.object(
                training_dataset, "_WINDOWS_NO_REPLACE_RENAME", False
            ), mock.patch.object(Path, "unlink", always_fail_temp):
                with self.assertLogs(training_dataset._LOGGER, level="WARNING") as captured:
                    prepare_training_dataset(sessions, output, window_ms=10, step_ms=10)
            self.assertNotIn(str(root.resolve()), "\n".join(captured.output))
            self.assertTrue(output.is_file())
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["schema"],
                "emg.training.dataset_manifest",
            )
            leftovers = list(root.glob(f".{output.name}.*.tmp"))
            self.assertEqual(len(leftovers), 1)
            leftovers[0].write_text("tampered residual", encoding="utf-8")
            self.assertTrue(output.is_file())
            original_unlink(leftovers[0])

    def test_samefile_then_unlink_race_cannot_delete_concurrent_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = [
                self.make_session(root, number, f"replace-after-link-{number}", ["rest"] * 2)
                for number in range(1, 4)
            ]
            output = root / "manifest.json"
            sentinel = "concurrent replacement\n"
            original_unlink = Path.unlink
            injected = False

            def replace_target_on_first_temp_cleanup(path, *args, **kwargs):
                nonlocal injected
                if (
                    not injected
                    and path.name.startswith(f".{output.name}.")
                    and path.suffix == ".tmp"
                ):
                    injected = True
                    original_unlink(output)
                    output.write_text(sentinel, encoding="utf-8")
                    raise PermissionError("injected cleanup race")
                return original_unlink(path, *args, **kwargs)

            with mock.patch.object(
                training_dataset, "_WINDOWS_NO_REPLACE_RENAME", False
            ), mock.patch.object(
                Path, "unlink", replace_target_on_first_temp_cleanup
            ), mock.patch.object(
                training_dataset.os.path, "samefile", return_value=True
            ) as samefile:
                with self.assertLogs(training_dataset._LOGGER, level="WARNING"):
                    prepare_training_dataset(sessions, output, window_ms=10, step_ms=10)
            self.assertTrue(injected)
            samefile.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), sentinel)
            leftovers = list(root.glob(f".{output.name}.*.tmp"))
            self.assertEqual(len(leftovers), 1)
            original_unlink(leftovers[0])

    def test_post_commit_logging_failure_cannot_change_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = [
                self.make_session(root, number, f"log-failure-{number}", ["rest"] * 2)
                for number in range(1, 4)
            ]
            output = root / "manifest.json"
            original_unlink = Path.unlink

            def fail_temp_cleanup(path, *args, **kwargs):
                if path.name.startswith(f".{output.name}.") and path.suffix == ".tmp":
                    raise PermissionError("injected cleanup failure")
                return original_unlink(path, *args, **kwargs)

            with mock.patch.object(
                training_dataset, "_WINDOWS_NO_REPLACE_RENAME", False
            ), mock.patch.object(
                Path, "unlink", fail_temp_cleanup
            ), mock.patch.object(
                training_dataset._LOGGER,
                "warning",
                side_effect=RuntimeError("injected logging handler failure"),
            ) as warning_log:
                manifest = prepare_training_dataset(
                    sessions, output, window_ms=10, step_ms=10
                )

            warning_log.assert_called_once()
            self.assertEqual(manifest["schema"], "emg.training.dataset_manifest")
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["schema"],
                "emg.training.dataset_manifest",
            )
            leftovers = list(root.glob(f".{output.name}.*.tmp"))
            self.assertEqual(len(leftovers), 1)
            original_unlink(leftovers[0])

    def test_publish_never_overwrites_existing_or_concurrently_created_target(self):
        for timing in ("existing", "concurrent"):
            with self.subTest(timing=timing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sessions = [
                    self.make_session(root, number, f"collision-{timing}-{number}", ["rest"] * 2)
                    for number in range(1, 4)
                ]
                output = root / "manifest.json"
                sentinel = "owned by another writer\n"
                if timing == "existing":
                    output.write_text(sentinel, encoding="utf-8")
                    with self.assertRaisesRegex(TrainingDatasetError, "already exists"):
                        prepare_training_dataset(sessions, output, window_ms=10, step_ms=10)
                else:
                    def concurrent_publish(_source, target):
                        Path(target).write_text(sentinel, encoding="utf-8")
                        raise FileExistsError("injected concurrent publisher")

                    with mock.patch.object(training_dataset.os, "rename", concurrent_publish):
                        with self.assertRaisesRegex(TrainingDatasetError, "already exists"):
                            prepare_training_dataset(
                                sessions, output, window_ms=10, step_ms=10
                            )
                self.assertEqual(output.read_text(encoding="utf-8"), sentinel)
                self.assertEqual(list(root.glob(f".{output.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
