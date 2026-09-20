import csv
import hashlib
import json
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from acquisition_pipeline import (
    AcquisitionPipeline,
    PipelineCleanupError,
    PipelineEventType,
    PipelineQualitySnapshot,
    RawAuditPolicy,
    RawNotification,
    RawPacketAudit,
)
from bleak_ble import BleNotification
from data_recorder import DataRecorder, generate_subject_key
from emg_protocol import (
    AcquisitionMetadata,
    DeviceKey,
    HandSide,
    NotificationPacketProtocol,
    PADDED28_PROTOCOL,
    QualityFlags,
    SignalChain,
)
from recording_context import RecordingContext
from shared_memory_v2 import (
    FLAG_DISCONNECTED,
    FLAG_OVERFLOW,
    FLAG_STALE,
    SharedMemoryReader,
    SharedMemoryWriter,
)


class FakeWriter:
    def __init__(self):
        self.calls = []
        self.generation = 42

    def write_frame(self, payload, **kwargs):
        self.calls.append((bytes(payload), kwargs))


class FakeRecorder:
    def __init__(self, session_dir=None):
        self.frames = []
        self.session_sample_indexes = []
        self.boundary_updates = []
        self.closed = []
        self._temporary = None if session_dir is not None else tempfile.TemporaryDirectory()
        self.session_dir = Path(session_dir) if session_dir is not None else Path(self._temporary.name)

    def configure_session_boundary(self, **kwargs):
        self.boundary_updates.append(kwargs)

    def update_session_boundary(self, **kwargs):
        self.boundary_updates.append(kwargs)

    def record(self, frame, *, session_sample_index=None):
        self.frames.append(frame)
        self.session_sample_indexes.append(session_sample_index)

    def close(self, **kwargs):
        self.closed.append(kwargs)
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None


def recording_context(
    *, action_label="fist", action_phase="hold", experiment_id="pipeline-test"
):
    return RecordingContext(
        subject_id="sub-0123456789abcdef0123456789abcdef",
        action_label=action_label,
        action_phase=action_phase,
        experiment_id=experiment_id,
        hand_side=HandSide.LEFT,
    )


def attach_recorder(pipeline, recorder, context=None):
    return pipeline.set_recorder(
        recorder, recording_context=context or recording_context()
    )


class PipelineTests(unittest.TestCase):
    def test_quality_snapshot_tracks_watchdog_stale_and_fresh_recovery(self):
        observed_stale = []
        pipeline = AcquisitionPipeline(
            FakeWriter(),
            stale_after_seconds=0.01,
            event_callback=lambda event: (
                observed_stale.append(pipeline.quality_snapshot())
                if event.event_type is PipelineEventType.STALE
                else None
            ),
        )
        pipeline.start()
        pipeline.arm_stream_watchdog(7)
        with pipeline._lock:
            pipeline._watchdog_baseline_ns = time.monotonic_ns() - 20_000_000
        pipeline._check_stale()

        stale = pipeline.quality_snapshot()
        self.assertTrue(stale.stream_expected)
        self.assertTrue(stale.watchdog_stale)
        self.assertGreaterEqual(stale.watchdog_age_seconds, 0.01)
        self.assertEqual(len(observed_stale), 1)
        self.assertTrue(observed_stale[0].watchdog_stale)

        now = time.monotonic_ns()
        pipeline._process(RawNotification(time.time_ns(), now, 1, bytes(range(16)), 7))
        fresh = pipeline.quality_snapshot()
        self.assertFalse(fresh.watchdog_stale)
        self.assertEqual(fresh.last_raw_received_monotonic_ns, None)
        self.assertEqual(fresh.last_parsed_monotonic_ns, now)
        self.assertEqual(fresh.last_published_monotonic_ns, now)
        self.assertEqual(fresh.watchdog_reference_monotonic_ns, now)
        pipeline.close()

    def test_quality_snapshot_defines_unstarted_disconnect_and_restart_states(self):
        pipeline = AcquisitionPipeline(FakeWriter())
        initial = pipeline.quality_snapshot()
        self.assertIsInstance(initial, PipelineQualitySnapshot)
        self.assertFalse(initial.pipeline_started)
        self.assertFalse(initial.accepting_notifications)
        self.assertFalse(initial.stream_expected)
        self.assertFalse(initial.watchdog_stale)
        self.assertIsNone(initial.watchdog_reference_monotonic_ns)
        self.assertIsNone(initial.watchdog_age_seconds)

        pipeline.start()
        pipeline.arm_stream_watchdog(11)
        pipeline.mark_disconnected(connection_generation=11)
        disconnected = pipeline.quality_snapshot()
        self.assertTrue(disconnected.pipeline_started)
        self.assertFalse(disconnected.accepting_notifications)
        self.assertFalse(disconnected.stream_expected)
        self.assertFalse(disconnected.watchdog_stale)
        self.assertIsNone(disconnected.connection_generation)
        self.assertIsNone(disconnected.watchdog_age_seconds)

        pipeline.resume()
        pipeline.arm_stream_watchdog(12)
        restarted = pipeline.quality_snapshot()
        self.assertTrue(restarted.accepting_notifications)
        self.assertTrue(restarted.stream_expected)
        self.assertFalse(restarted.watchdog_stale)
        self.assertEqual(restarted.connection_generation, 12)
        self.assertIsNotNone(restarted.watchdog_age_seconds)
        pipeline.close()

    def test_quality_snapshot_is_safe_during_concurrent_updates(self):
        pipeline = AcquisitionPipeline(FakeWriter())
        pipeline.start()
        pipeline.arm_stream_watchdog(3)
        failures = []
        finished = threading.Event()

        def read_snapshots():
            try:
                while not finished.is_set():
                    snapshot = pipeline.quality_snapshot()
                    self.assertGreaterEqual(snapshot.queue_depth, 0)
                    self.assertGreaterEqual(snapshot.host_queue_dropped_count, 0)
                    self.assertEqual(snapshot.stale_after_seconds, 1.0)
                    if not snapshot.stream_expected:
                        self.assertFalse(snapshot.watchdog_stale)
                        self.assertIsNone(snapshot.watchdog_age_seconds)
            except BaseException as exc:
                failures.append(exc)

        reader = threading.Thread(target=read_snapshots)
        reader.start()
        for index in range(1, 101):
            now = time.monotonic_ns()
            pipeline.notification_callback(
                BleNotification(
                    None,
                    bytes(range(16)),
                    3,
                    host_wall_timestamp_ns=time.time_ns(),
                    host_monotonic_ns=now,
                )
            )
        pipeline._queue.join()
        finished.set()
        reader.join(1.0)
        pipeline.close()
        self.assertFalse(reader.is_alive())
        self.assertEqual(failures, [])

    def test_quality_snapshot_keeps_legacy_quality_properties_compatible(self):
        pipeline = AcquisitionPipeline(FakeWriter(), queue_capacity=1)
        pipeline.start()
        now = time.monotonic_ns()
        pipeline._process(RawNotification(time.time_ns(), now, 1, bytes(range(16))))
        snapshot = pipeline.quality_snapshot()
        self.assertEqual(snapshot.queue_depth, pipeline.queue_depth)
        self.assertEqual(snapshot.host_queue_dropped_count, pipeline.dropped_count)
        self.assertEqual(
            (
                snapshot.last_raw_received_monotonic_ns,
                snapshot.last_parsed_monotonic_ns,
                snapshot.last_published_monotonic_ns,
            ),
            pipeline.freshness,
        )
        self.assertEqual(pipeline.last_frame.host_receive_index, 1)
        pipeline.close()

    def test_parser_uses_odd_bytes_and_does_not_invent_device_fields(self):
        item = RawNotification(10, 20, 3, bytes(range(16)), 17)
        frame = AcquisitionPipeline.parse_notification(item, 7, False, 42)
        self.assertEqual(frame.channel_values, (1, 3, 5, 7, 9, 11, 13, 15))
        self.assertEqual(frame.host_receive_index, 3)
        self.assertEqual(frame.connection_generation, 17)
        self.assertIsNone(frame.device_packet_sequence)
        self.assertIsNone(frame.device_sample_counter)
        self.assertIsNone(frame.device_time_ticks)

    def test_recorder_attachment_requires_explicit_valid_context(self):
        pipeline = AcquisitionPipeline(FakeWriter())
        recorder = FakeRecorder()
        try:
            with self.assertRaisesRegex(ValueError, "recording_context"):
                pipeline.set_recorder(recorder)
            with self.assertRaisesRegex(ValueError, "recording_context"):
                pipeline.set_recorder(recorder, recording_context=object())
            self.assertEqual(recorder.frames, [])
            self.assertFalse(pipeline.recorder_cleanup_pending)
        finally:
            recorder.close(complete=False, error="test_cleanup")

    def test_recorder_attachment_after_pipeline_close_is_rejected(self):
        pipeline = AcquisitionPipeline(FakeWriter())
        pipeline.close()
        recorder = FakeRecorder()
        try:
            with self.assertRaisesRegex(RuntimeError, "pipeline is closed"):
                attach_recorder(pipeline, recorder)
            self.assertEqual(recorder.frames, [])
            self.assertFalse(pipeline.recorder_cleanup_pending)
        finally:
            recorder.close(complete=False, error="unattached")

    def test_pending_audit_close_blocks_new_session_until_same_audit_retries(self):
        audits = []

        class RetryAudit:
            def __init__(self, path, **kwargs):
                self.close_attempts = 0
                audits.append(self)

            def write_policy(self, policy):
                pass

            def write_protocol(self, protocol):
                pass

            def write(self, item, parse_error):
                pass

            def close(self):
                self.close_attempts += 1
                if self is audits[0] and self.close_attempts == 1:
                    raise OSError("audit busy")

        pipeline = AcquisitionPipeline(
            FakeWriter(),
            raw_audit_factory=RetryAudit,
            raw_audit_policy=RawAuditPolicy(enabled=True),
        )
        first = FakeRecorder()
        attach_recorder(pipeline, first)
        with self.assertRaises(PipelineCleanupError):
            pipeline.stop_recorder(complete=True)
        self.assertEqual(first.closed, [{"complete": True, "error": None}])
        self.assertTrue(pipeline.recorder_cleanup_pending)
        self.assertEqual(len(audits), 1)

        second = FakeRecorder()
        try:
            with self.assertRaisesRegex(RuntimeError, "awaiting close retry"):
                attach_recorder(
                    pipeline, second, recording_context(action_label="open_hand")
                )
            self.assertEqual(len(audits), 1)
            self.assertEqual(audits[0].close_attempts, 1)

            pipeline.stop_recorder(complete=True)
            self.assertEqual(audits[0].close_attempts, 2)
            self.assertFalse(pipeline.recorder_cleanup_pending)

            attach_recorder(
                pipeline, second, recording_context(action_label="open_hand")
            )
            self.assertEqual(len(audits), 2)
            pipeline.stop_recorder(complete=True)
        finally:
            if not second.closed:
                second.close(complete=False, error="test_cleanup")

    def test_raw_audit_header_failures_close_audit_and_preserve_root_error(self):
        for failing_stage in ("policy", "protocol"):
            with self.subTest(failing_stage=failing_stage):
                audits = []

                class HeaderFailureAudit:
                    def __init__(self, path, **kwargs):
                        self.close_attempts = 0
                        audits.append(self)

                    def write_policy(self, policy):
                        if failing_stage == "policy":
                            raise RuntimeError("policy root failure")

                    def write_protocol(self, protocol):
                        if failing_stage == "protocol":
                            raise RuntimeError("protocol root failure")

                    def close(self):
                        self.close_attempts += 1

                pipeline = AcquisitionPipeline(
                    FakeWriter(),
                    raw_audit_factory=HeaderFailureAudit,
                    raw_audit_policy=RawAuditPolicy(enabled=True),
                )
                recorder = FakeRecorder()
                try:
                    with self.assertRaisesRegex(
                        RuntimeError, f"{failing_stage} root failure"
                    ):
                        attach_recorder(pipeline, recorder)
                    self.assertEqual(len(audits), 1)
                    self.assertEqual(audits[0].close_attempts, 1)
                    self.assertFalse(pipeline.recorder_cleanup_pending)
                    self.assertEqual(recorder.frames, [])
                finally:
                    recorder.close(complete=False, error="unattached")

    def test_raw_audit_requires_callable_policy_and_protocol_headers(self):
        class AuditBase:
            def __init__(self, path, **kwargs):
                self.close_attempts = 0

            def close(self):
                self.close_attempts += 1

        class MissingPolicyAudit(AuditBase):
            def write_protocol(self, protocol):
                pass

        class MissingProtocolAudit(AuditBase):
            def write_policy(self, policy):
                pass

        class NonCallablePolicyAudit(AuditBase):
            write_policy = None

            def write_protocol(self, protocol):
                pass

        class NonCallableProtocolAudit(AuditBase):
            def write_policy(self, policy):
                pass

            write_protocol = "not callable"

        for audit_type, missing_method in (
            (MissingPolicyAudit, "write_policy"),
            (MissingProtocolAudit, "write_protocol"),
            (NonCallablePolicyAudit, "write_policy"),
            (NonCallableProtocolAudit, "write_protocol"),
        ):
            with self.subTest(audit_type=audit_type.__name__):
                audits = []

                def factory(path, **kwargs):
                    audit = audit_type(path, **kwargs)
                    audits.append(audit)
                    return audit

                pipeline = AcquisitionPipeline(
                    FakeWriter(),
                    raw_audit_factory=factory,
                    raw_audit_policy=RawAuditPolicy(enabled=True),
                )
                recorder = FakeRecorder()
                try:
                    with self.assertRaisesRegex(
                        TypeError, f"raw audit {missing_method} must be callable"
                    ):
                        attach_recorder(pipeline, recorder)
                    self.assertEqual(audits[0].close_attempts, 1)
                    self.assertFalse(pipeline.recorder_cleanup_pending)
                    self.assertEqual(recorder.frames, [])
                finally:
                    recorder.close(complete=False, error="unattached")

    def test_missing_audit_header_and_close_failure_retains_retry_gate(self):
        audits = []

        class MissingProtocolAudit:
            def __init__(self, path, **kwargs):
                self.close_attempts = 0
                audits.append(self)

            def write_policy(self, policy):
                pass

            def close(self):
                self.close_attempts += 1
                if self.close_attempts == 1:
                    raise OSError("header audit busy")

        pipeline = AcquisitionPipeline(
            FakeWriter(),
            raw_audit_factory=MissingProtocolAudit,
            raw_audit_policy=RawAuditPolicy(enabled=True),
        )
        failed_recorder = FakeRecorder()
        with self.assertRaises(PipelineCleanupError) as captured:
            attach_recorder(pipeline, failed_recorder)
        self.assertIsInstance(captured.exception.__cause__, TypeError)
        self.assertIn("write_protocol must be callable", str(captured.exception.__cause__))
        self.assertEqual(
            [stage for stage, _error in captured.exception.failures],
            ["recorder_setup", "raw_audit_close"],
        )
        self.assertTrue(pipeline.recorder_cleanup_pending)
        self.assertEqual(audits[0].close_attempts, 1)

        blocked_recorder = FakeRecorder()
        try:
            with self.assertRaisesRegex(RuntimeError, "awaiting close retry"):
                attach_recorder(pipeline, blocked_recorder)
            self.assertEqual(len(audits), 1)
            pipeline.stop_recorder(complete=False, error="header_setup_failed")
            self.assertEqual(audits[0].close_attempts, 2)
            self.assertFalse(pipeline.recorder_cleanup_pending)
        finally:
            failed_recorder.close(complete=False, error="unattached")
            blocked_recorder.close(complete=False, error="unattached")

    def test_configure_and_audit_close_double_failure_is_retained_for_retry(self):
        audits = []

        class RetryAudit:
            def __init__(self, path, **kwargs):
                self.close_attempts = 0
                audits.append(self)

            def write_policy(self, policy):
                pass

            def write_protocol(self, protocol):
                pass

            def close(self):
                self.close_attempts += 1
                if self is audits[0] and self.close_attempts == 1:
                    raise OSError("audit close failure")

        class ConfigureFailureRecorder(FakeRecorder):
            def configure_session_boundary(self, **kwargs):
                raise RuntimeError("configure root failure")

        pipeline = AcquisitionPipeline(
            FakeWriter(),
            raw_audit_factory=RetryAudit,
            raw_audit_policy=RawAuditPolicy(enabled=True),
        )
        failed_recorder = ConfigureFailureRecorder()
        with self.assertRaises(PipelineCleanupError) as captured:
            attach_recorder(pipeline, failed_recorder)
        self.assertIsInstance(captured.exception.__cause__, RuntimeError)
        self.assertEqual(str(captured.exception.__cause__), "configure root failure")
        self.assertEqual(
            [stage for stage, _error in captured.exception.failures],
            ["recorder_setup", "raw_audit_close"],
        )
        self.assertTrue(pipeline.recorder_cleanup_pending)
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0].close_attempts, 1)

        replacement = FakeRecorder()
        try:
            with self.assertRaisesRegex(RuntimeError, "awaiting close retry"):
                attach_recorder(pipeline, replacement)
            self.assertEqual(len(audits), 1)
            self.assertEqual(audits[0].close_attempts, 1)

            pipeline.stop_recorder(complete=False, error="setup_failed")
            self.assertEqual(audits[0].close_attempts, 2)
            self.assertFalse(pipeline.recorder_cleanup_pending)

            attach_recorder(pipeline, replacement)
            self.assertEqual(len(audits), 2)
            pipeline.stop_recorder(complete=True)
        finally:
            failed_recorder.close(complete=False, error="unattached")
            if not replacement.closed:
                replacement.close(complete=False, error="test_cleanup")

    def test_active_session_rejects_context_switch_and_keeps_original_label(self):
        pipeline = AcquisitionPipeline(FakeWriter())
        first = FakeRecorder()
        second = FakeRecorder()
        fist = recording_context(action_label="fist")
        opened = recording_context(action_label="open_hand")
        attach_recorder(pipeline, first, fist)
        try:
            with self.assertRaisesRegex(RuntimeError, "recorder is active"):
                attach_recorder(pipeline, second, opened)
            pipeline._process(RawNotification(1, 2, 1, bytes(range(16))))
            self.assertEqual(
                [(frame.action_label, frame.action_phase) for frame in first.frames],
                [("fist", "hold")],
            )
        finally:
            pipeline.stop_recorder(complete=False, error="test")
            second.close(complete=False, error="unattached")

    def test_single_session_annotation_is_stable_for_every_recorded_frame(self):
        pipeline = AcquisitionPipeline(FakeWriter())
        recorder = FakeRecorder()
        context = recording_context(action_label="open_hand")
        attach_recorder(pipeline, recorder, context)
        for index in (1, 2, 3):
            pipeline._process(RawNotification(index, index, index, bytes(range(16))))
        self.assertEqual(
            {(frame.action_label, frame.action_phase) for frame in recorder.frames},
            {("open_hand", "hold")},
        )
        pipeline.stop_recorder(complete=True)

    def test_stop_clears_annotation_from_later_display_frames(self):
        displayed = []
        pipeline = AcquisitionPipeline(FakeWriter(), display_callback=displayed.append)
        recorder = FakeRecorder()
        attach_recorder(pipeline, recorder, recording_context(action_label="rest"))
        pipeline._process(RawNotification(1, 1, 1, bytes(range(16))))
        pipeline.stop_recorder(complete=True)
        pipeline._process(RawNotification(2, 2, 2, bytes(range(16))))
        self.assertEqual(displayed[0].action_label, "rest")
        self.assertEqual((displayed[1].action_label, displayed[1].action_phase), ("", ""))

    def test_recording_fault_does_not_leak_context_into_next_session(self):
        class BrokenRecorder(FakeRecorder):
            def record(self, frame, *, session_sample_index=None):
                raise OSError("disk full")

        displayed = []
        pipeline = AcquisitionPipeline(FakeWriter(), display_callback=displayed.append)
        broken = BrokenRecorder()
        attach_recorder(pipeline, broken, recording_context(action_label="fist"))
        pipeline._process(RawNotification(1, 1, 1, bytes(range(16))))
        pipeline._process(RawNotification(2, 2, 2, bytes(range(16))))
        self.assertEqual((displayed[-1].action_label, displayed[-1].action_phase), ("", ""))

        with pipeline._lock:
            pipeline._host_receive_index = 2
        replacement = FakeRecorder()
        attach_recorder(
            pipeline, replacement, recording_context(action_label="open_hand")
        )
        pipeline._process(RawNotification(3, 3, 3, bytes(range(16))))
        self.assertEqual(replacement.frames[0].action_label, "open_hand")
        pipeline.stop_recorder(complete=True)

    def test_session_boundary_labels_only_frames_inside_start_and_stop_watermarks(self):
        displayed = []
        pipeline = AcquisitionPipeline(FakeWriter(), display_callback=displayed.append)
        with pipeline._lock:
            pipeline._host_receive_index = 1
        recorder = FakeRecorder()
        attach_recorder(pipeline, recorder, recording_context(action_label="fist"))

        pipeline._process(RawNotification(1, 1, 1, bytes(range(16))))
        pipeline._process(RawNotification(2, 2, 2, bytes(range(16))))
        with pipeline._lock:
            pipeline._host_receive_index = 2
        token = pipeline.request_recorder_stop(complete=True)
        pipeline._process(RawNotification(3, 3, 3, bytes(range(16))))
        pipeline.finish_recorder_stop(token, timeout=1)

        self.assertEqual(
            [(frame.host_receive_index, frame.action_label) for frame in displayed],
            [(1, ""), (2, "fist"), (3, "")],
        )
        self.assertEqual([frame.host_receive_index for frame in recorder.frames], [2])

    def test_disconnect_and_close_abort_recorders_without_context_leak(self):
        pipeline = AcquisitionPipeline(FakeWriter())
        disconnected = FakeRecorder()
        attach_recorder(
            pipeline, disconnected, recording_context(action_label="open_hand")
        )
        pipeline.mark_disconnected("test_disconnect")
        self.assertEqual(
            disconnected.closed,
            [{"complete": False, "error": "test_disconnect"}],
        )

        replacement = FakeRecorder()
        attach_recorder(pipeline, replacement, recording_context(action_label="rest"))
        pipeline.close()
        self.assertEqual(
            replacement.closed,
            [{"complete": False, "error": "pipeline_closed"}],
        )

    def test_short_notification_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 16"):
            AcquisitionPipeline.parse_notification(RawNotification(1, 2, 3, b"short"), 1, False, 42)

    def test_explicit_padded_28_protocol_matches_logical_16_channels(self):
        logical = bytes(range(16))
        protocol = NotificationPacketProtocol(
            PADDED28_PROTOCOL,
            28,
            16,
            "zero_suffix",
            "live_probe.json",
        )
        frame_16 = AcquisitionPipeline.parse_notification(
            RawNotification(1, 2, 3, logical), 1, False, 42
        )
        frame_28 = AcquisitionPipeline.parse_notification(
            RawNotification(1, 2, 3, logical + bytes(12)),
            1,
            False,
            42,
            protocol,
        )
        self.assertEqual(frame_28.channel_values, frame_16.channel_values)

    def test_padded_28_protocol_rejects_nonzero_padding_and_wrong_lengths(self):
        protocol = NotificationPacketProtocol(
            PADDED28_PROTOCOL, 28, 16, "zero_suffix", "live_probe.json"
        )
        for payload, message in (
            (bytes(16) + bytes(11) + b"\x01", "non-zero suffix"),
            (bytes(27), "exactly 28"),
            (bytes(29), "exactly 28"),
        ):
            with self.subTest(length=len(payload)), self.assertRaisesRegex(
                ValueError, message
            ):
                AcquisitionPipeline.parse_notification(
                    RawNotification(1, 2, 3, payload),
                    1,
                    False,
                    42,
                    protocol,
                )

    def test_packet_protocol_configuration_mismatch_is_rejected(self):
        for kwargs in (
            {"mode": PADDED28_PROTOCOL, "wire_packet_size": 16, "logical_packet_size": 16, "padding_rule": "none", "evidence_ref": "probe"},
            {"mode": PADDED28_PROTOCOL, "wire_packet_size": 28, "logical_packet_size": 16, "padding_rule": "zero_suffix", "evidence_ref": "unknown"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NotificationPacketProtocol(**kwargs)
        for evidence in (None, "built_in_logical16_contract"):
            arguments = {
                "mode": PADDED28_PROTOCOL,
                "wire_packet_size": 28,
                "logical_packet_size": 16,
                "padding_rule": "zero_suffix",
            }
            if evidence is not None:
                arguments["evidence_ref"] = evidence
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                NotificationPacketProtocol(**arguments)
        with self.assertRaisesRegex(ValueError, "exactly 16"):
            AcquisitionPipeline.parse_notification(
                RawNotification(1, 2, 3, bytes(28)), 1, False, 42
            )

    def test_pipeline_enforces_complete_valid_raw_audit_capacity(self):
        protocol = NotificationPacketProtocol(
            PADDED28_PROTOCOL, 28, 16, "zero_suffix", "probe:sha256"
        )
        required = RawPacketAudit.complete_record_size(
            protocol, include_parse_error=True
        )
        invalid = (
            (RawAuditPolicy(enabled=True, max_bytes=required, payload_prefix_bytes=27), "payload_prefix_bytes"),
            (RawAuditPolicy(enabled=True, max_bytes=required - 1, payload_prefix_bytes=28), "max_bytes"),
        )
        for policy, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                AcquisitionPipeline(FakeWriter(), packet_protocol=protocol, raw_audit_policy=policy)
        AcquisitionPipeline(
            FakeWriter(), packet_protocol=protocol,
            raw_audit_policy=RawAuditPolicy(enabled=True, max_bytes=required, payload_prefix_bytes=28),
        )

    def test_invalid_only_raw_audit_keeps_nonzero_padding_packet_complete(self):
        protocol = NotificationPacketProtocol(
            PADDED28_PROTOCOL, 28, 16, "zero_suffix", "probe:sha256"
        )
        required = RawPacketAudit.complete_record_size(
            protocol, include_parse_error=True
        )
        for policy, message in (
            (RawAuditPolicy(enabled=True, max_bytes=required, payload_prefix_bytes=27, record_valid=False, record_invalid=True), "payload_prefix_bytes"),
            (RawAuditPolicy(enabled=True, max_bytes=required - 1, payload_prefix_bytes=28, record_valid=False, record_invalid=True), "max_bytes"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                AcquisitionPipeline(FakeWriter(), packet_protocol=protocol, raw_audit_policy=policy)
        with tempfile.TemporaryDirectory() as directory:
            recorder = FakeRecorder(Path(directory))
            pipeline = AcquisitionPipeline(
                FakeWriter(),
                packet_protocol=protocol,
                raw_audit_policy=RawAuditPolicy(
                    enabled=True,
                    max_bytes=required,
                    backup_count=1,
                    payload_prefix_bytes=28,
                    record_valid=False,
                    record_invalid=True,
                ),
            )
            attach_recorder(pipeline, recorder)
            payload = bytes(27) + b"\x01"
            pipeline._process(RawNotification(1, 2, 3, payload))
            pipeline.stop_recorder(complete=False, error="test")
            records = [
                json.loads(line)
                for file in Path(directory).glob("raw_packets.jsonl*")
                for line in file.read_text(encoding="utf-8").splitlines()
            ]
            packet = next(item for item in records if "host_receive_index" in item)
            self.assertEqual(bytes.fromhex(packet["payload_hex"]), payload)
            self.assertEqual(packet["original_length"], 28)
            self.assertNotIn("truncated", packet)
            self.assertIn("non-zero suffix", packet["parse_error"])

    def test_exact_invalid_record_budget_covers_worst_json_escaping(self):
        protocol = NotificationPacketProtocol(
            PADDED28_PROTOCOL, 28, 16, "zero_suffix", "probe:sha256"
        )
        required = RawPacketAudit.complete_record_size(
            protocol, include_parse_error=True
        )
        payload = bytes(range(16)) + bytes(12)
        hostile_errors = (
            "\x00" * 1024,
            '"' * 1024,
            "\\" * 1024,
            "肌电错误🙂" * 256,
        )
        for error_text in hostile_errors:
            with self.subTest(error=repr(error_text[:8])), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "raw_packets.jsonl"
                audit = RawPacketAudit(path, max_bytes=required, backup_count=0)
                audit.write_protocol(protocol)
                audit.write(RawNotification(1, 2, 3, payload), error_text)
                audit.close()
                records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                packet = records[-1]
                self.assertEqual(bytes.fromhex(packet["payload_hex"]), payload)
                self.assertEqual(packet["original_length"], 28)
                self.assertNotIn("truncated", packet)
                self.assertLessEqual(path.stat().st_size, required)
        policy = RawAuditPolicy(
            enabled=True,
            max_bytes=required - 1,
            payload_prefix_bytes=28,
            record_valid=False,
            record_invalid=True,
        )
        with self.assertRaisesRegex(ValueError, "max_bytes"):
            AcquisitionPipeline(
                FakeWriter(), packet_protocol=protocol, raw_audit_policy=policy
            )

    def test_duplicate_recorder_with_disabled_audit_raises_runtime_error(self):
        pipeline = AcquisitionPipeline(FakeWriter())
        first, second = FakeRecorder(), FakeRecorder()
        attach_recorder(pipeline, first)
        try:
            with self.assertRaisesRegex(RuntimeError, "recorder is active"):
                attach_recorder(pipeline, second)
        finally:
            pipeline.stop_recorder(complete=False, error="test")
            second.close(complete=False, error="not_attached")

    def test_one_frame_is_shared_recorded_and_displayed(self):
        writer = FakeWriter()
        recorder = FakeRecorder()
        displayed = []
        pipeline = AcquisitionPipeline(writer, display_callback=displayed.append)
        attach_recorder(pipeline, recorder)
        pipeline._process(RawNotification(100, 200, 1, bytes(range(16)), 18))
        self.assertIs(recorder.frames[0], displayed[0])
        self.assertEqual(writer.calls[0][0], bytes(recorder.frames[0].channel_values))
        self.assertEqual(writer.calls[0][1]["host_receive_index"], 1)
        self.assertEqual(writer.calls[0][1]["connection_generation"], 18)
        self.assertEqual(recorder.frames[0].connection_generation, 18)
        pipeline.stop_recorder(complete=True)

    def test_callback_is_nonblocking_and_latches_overflow(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingWriter(FakeWriter):
            def write_frame(self, payload, **kwargs):
                entered.set()
                release.wait(2)
                super().write_frame(payload, **kwargs)

        pipeline = AcquisitionPipeline(BlockingWriter(), queue_capacity=1)
        pipeline.start()
        self.assertTrue(
            pipeline.notification_callback(BleNotification(None, bytes(range(16)), 7))
        )
        self.assertTrue(entered.wait(1))
        self.assertTrue(
            pipeline.notification_callback(BleNotification(None, bytes(range(16)), 7))
        )
        started = time.perf_counter()
        self.assertFalse(
            pipeline.notification_callback(BleNotification(None, bytes(range(16)), 7))
        )
        self.assertLess(time.perf_counter() - started, 0.05)
        self.assertEqual(pipeline.dropped_count, 1)
        release.set()
        pipeline.close()
        self.assertTrue(pipeline.last_frame.quality_flags & QualityFlags.OVERFLOW)

    def test_callback_preserves_ble_backend_receive_timestamps(self):
        writer = FakeWriter()
        pipeline = AcquisitionPipeline(writer)
        pipeline.start()
        try:
            self.assertTrue(
                pipeline.notification_callback(
                    BleNotification(
                        None,
                        bytes(range(16)),
                        7,
                        host_wall_timestamp_ns=1_700_000_000_000_000_000,
                        host_monotonic_ns=123_456_789_000,
                    )
                )
            )
            deadline = time.monotonic() + 1.0
            while not writer.calls:
                if time.monotonic() >= deadline:
                    self.fail("pipeline did not process timestamped notification")
                time.sleep(0.001)
            _, kwargs = writer.calls[0]
            self.assertEqual(kwargs["host_wall_timestamp_ns"], 1_700_000_000_000_000_000)
            self.assertEqual(kwargs["host_monotonic_ns"], 123_456_789_000)
        finally:
            pipeline.close()

    def test_concurrent_callbacks_cannot_overtake_host_receive_order(self):
        first_put_entered = threading.Event()
        release_first_put = threading.Event()
        second_put_entered = threading.Event()

        class OrderingQueue:
            def __init__(self):
                self.items = []

            def put_nowait(self, item):
                if item.host_receive_index == 1:
                    first_put_entered.set()
                    release_first_put.wait(1)
                else:
                    second_put_entered.set()
                self.items.append(item)

        pipeline = AcquisitionPipeline(FakeWriter())
        pipeline._accepting = True
        ordering_queue = OrderingQueue()
        pipeline._queue = ordering_queue
        results = []

        first = threading.Thread(
            target=lambda: results.append(
                pipeline.notification_callback(
                    BleNotification(None, bytes(range(16)), 7)
                )
            )
        )
        second = threading.Thread(
            target=lambda: results.append(
                pipeline.notification_callback(
                    BleNotification(None, bytes(range(16)), 7)
                )
            )
        )
        first.start()
        self.assertTrue(first_put_entered.wait(1))
        second.start()
        self.assertFalse(second_put_entered.wait(0.05))
        release_first_put.set()
        first.join(1)
        second.join(1)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(results, [True, True])
        self.assertEqual(
            [item.host_receive_index for item in ordering_queue.items], [1, 2]
        )
        self.assertLessEqual(
            ordering_queue.items[0].host_monotonic_ns,
            ordering_queue.items[1].host_monotonic_ns,
        )

    def test_queue_drop_keeps_host_receive_index_gap(self):
        class FullQueue:
            def put_nowait(self, item):
                raise queue.Full

        class CaptureQueue:
            def __init__(self):
                self.items = []

            def put_nowait(self, item):
                self.items.append(item)

        pipeline = AcquisitionPipeline(FakeWriter())
        pipeline._accepting = True
        pipeline._queue = FullQueue()
        self.assertFalse(
            pipeline.notification_callback(
                BleNotification(None, bytes(range(16)), 7)
            )
        )
        capture = CaptureQueue()
        pipeline._queue = capture
        self.assertTrue(
            pipeline.notification_callback(
                BleNotification(None, bytes(range(16)), 7)
            )
        )
        self.assertEqual(pipeline.dropped_count, 1)
        self.assertEqual(capture.items[0].host_receive_index, 2)

    def test_late_recorder_attach_and_restart_preserve_pipeline_lifetime_indices(self):
        pipeline = AcquisitionPipeline(FakeWriter())
        pipeline.start()

        def submit_and_wait(expected_sample_index):
            self.assertTrue(
                pipeline.notification_callback(
                    BleNotification(None, bytes(range(16)), 11)
                )
            )
            deadline = time.monotonic() + 1
            while (
                pipeline.last_frame is None
                or pipeline.last_frame.sample_index < expected_sample_index
            ):
                if time.monotonic() >= deadline:
                    self.fail("pipeline did not process notification")
                time.sleep(0.001)

        try:
            for expected in range(1, 4):
                submit_and_wait(expected)

            first = FakeRecorder()
            attach_recorder(pipeline, first)
            submit_and_wait(4)
            submit_and_wait(5)
            pipeline.stop_recorder(complete=True)

            second = FakeRecorder()
            attach_recorder(pipeline, second)
            submit_and_wait(6)
            submit_and_wait(7)
            pipeline.stop_recorder(complete=True)

            self.assertEqual(
                [frame.sample_index for frame in first.frames], [4, 5]
            )
            self.assertEqual(
                [frame.host_receive_index for frame in first.frames], [4, 5]
            )
            self.assertEqual(
                [frame.sample_index for frame in second.frames], [6, 7]
            )
            self.assertEqual(
                [frame.host_receive_index for frame in second.frames], [6, 7]
            )
        finally:
            pipeline.close()

    def test_disconnect_aborts_recording_and_publishes_status(self):
        writer = FakeWriter()
        recorder = FakeRecorder()
        pipeline = AcquisitionPipeline(writer)
        attach_recorder(pipeline, recorder)
        pipeline._process(RawNotification(100, 200, 1, bytes(range(16)), 19))
        valid_call = writer.calls[-1]
        pipeline.mark_disconnected("unexpected")
        self.assertEqual(recorder.closed, [{"complete": False, "error": "unexpected"}])
        self.assertEqual(writer.calls[-1][1]["flags"], FLAG_DISCONNECTED | FLAG_STALE)
        self.assertEqual(writer.calls[-1][0], valid_call[0])
        self.assertEqual(writer.calls[-1][1]["host_wall_timestamp_ns"], 100)
        self.assertEqual(writer.calls[-1][1]["host_monotonic_ns"], 200)
        self.assertEqual(writer.calls[-1][1]["host_receive_index"], 1)
        self.assertEqual(writer.calls[-1][1]["connection_generation"], 19)

    def test_disconnect_close_failure_emits_recording_fault_and_retries_aborted_intent(self):
        class RetryRecorder(FakeRecorder):
            def __init__(self):
                super().__init__()
                self.attempts = 0

            def close(self, **kwargs):
                self.attempts += 1
                if self.attempts == 1:
                    raise OSError("disconnect close busy")
                super().close(**kwargs)

        recorder = RetryRecorder()
        events = []
        pipeline = AcquisitionPipeline(FakeWriter(), event_callback=events.append)
        attach_recorder(pipeline, recorder)
        pipeline.mark_disconnected("ble_unexpected")
        fault = next(
            event for event in events
            if event.event_type is PipelineEventType.RECORDING_FAULT
        )
        self.assertEqual(fault.stage, "recorder_disconnect_close")
        self.assertTrue(fault.cleanup_pending)
        self.assertIs(fault.close_error, fault.error)
        pipeline.stop_recorder(complete=True)
        self.assertEqual(
            recorder.closed,
            [{"complete": False, "error": "ble_unexpected"}],
        )
        self.assertFalse(pipeline.recorder_cleanup_pending)

    def test_reconnect_resumes_acceptance_without_starting_second_worker(self):
        pipeline = AcquisitionPipeline(FakeWriter())
        pipeline.start()
        thread = pipeline._thread
        pipeline.mark_disconnected("unexpected")
        self.assertFalse(
            pipeline.notification_callback(BleNotification(None, bytes(range(16)), 8))
        )
        pipeline.resume()
        self.assertTrue(
            pipeline.notification_callback(BleNotification(None, bytes(range(16)), 9))
        )
        pipeline.close()
        self.assertIsNotNone(thread)
        self.assertEqual(pipeline.last_frame.connection_generation, 9)

    def test_one_sink_failure_does_not_block_other_consumers(self):
        errors = []
        displayed = []

        class BrokenWriter(FakeWriter):
            def write_frame(self, payload, **kwargs):
                raise OSError("shared unavailable")

        pipeline = AcquisitionPipeline(
            BrokenWriter(),
            display_callback=displayed.append,
            event_callback=lambda event: errors.append((event.stage, type(event.error))),
        )
        recorder = FakeRecorder()
        attach_recorder(pipeline, recorder)
        pipeline._process(RawNotification(100, 200, 1, bytes(range(16))))
        self.assertEqual(len(recorder.frames), 1)
        self.assertEqual(len(displayed), 1)
        self.assertEqual(errors, [("shared_write", OSError)])
        pipeline.stop_recorder(complete=True)

    def test_pre_arm_queued_frame_cannot_move_watchdog_baseline_backwards(self):
        writer = FakeWriter()
        events = []
        pipeline = AcquisitionPipeline(
            writer, stale_after_seconds=1.0, event_callback=events.append
        )
        pipeline.start()
        try:
            pipeline.arm_stream_watchdog(1)
            baseline = pipeline._watchdog_baseline_ns
            old = baseline - 2_000_000_000
            pipeline._process(RawNotification(1, old, 1, bytes(range(16)), 1))
            self.assertIsNone(pipeline._watchdog_last_published_ns)

            with mock.patch(
                "acquisition_pipeline.time.monotonic_ns",
                return_value=baseline + 999_999_999,
            ):
                pipeline._check_stale()
            self.assertFalse(any(event.event_type is PipelineEventType.STALE for event in events))
            self.assertEqual(len(writer.calls), 1)

            with mock.patch(
                "acquisition_pipeline.time.monotonic_ns",
                return_value=baseline + 1_000_000_000,
            ):
                pipeline._check_stale()
                pipeline._check_stale()
            self.assertEqual(len(writer.calls), 2)
            self.assertEqual(writer.calls[-1][1]["flags"], FLAG_STALE)
            self.assertEqual(
                len([event for event in events if event.event_type is PipelineEventType.STALE]),
                1,
            )
        finally:
            pipeline.close()

    def test_watchdog_accepts_equal_baseline_and_advances_only_for_newer_frames(self):
        pipeline = AcquisitionPipeline(FakeWriter(), stale_after_seconds=1.0)
        pipeline.start()
        try:
            pipeline.arm_stream_watchdog(3)
            baseline = pipeline._watchdog_baseline_ns
            pipeline._process(
                RawNotification(1, baseline - 1, 1, bytes(range(16)), 3)
            )
            self.assertIsNone(pipeline._watchdog_last_published_ns)
            pipeline._process(
                RawNotification(2, baseline, 2, bytes(range(16)), 3)
            )
            self.assertEqual(pipeline._watchdog_last_published_ns, baseline)
            pipeline._process(
                RawNotification(3, baseline + 1, 3, bytes(range(16)), 3)
            )
            self.assertEqual(pipeline._watchdog_last_published_ns, baseline + 1)
        finally:
            pipeline.close()

    def test_disarmed_stream_does_not_report_watchdog_stale_after_active_stop(self):
        writer = FakeWriter()
        events = []
        pipeline = AcquisitionPipeline(
            writer, stale_after_seconds=0.01, event_callback=events.append
        )
        pipeline.start()
        try:
            pipeline.arm_stream_watchdog(7)
            pipeline._process(
                RawNotification(1, time.monotonic_ns(), 1, bytes(range(16)), 7)
            )
            self.assertTrue(pipeline.disarm_stream_watchdog(7))
            time.sleep(0.02)
            pipeline._check_stale()
            self.assertFalse(any(event.stage == "watchdog" for event in events))
            self.assertEqual(len(writer.calls), 1)
        finally:
            pipeline.close()

    def test_armed_stream_with_missing_notifications_reports_watchdog_stale(self):
        writer = FakeWriter()
        events = []
        pipeline = AcquisitionPipeline(
            writer, stale_after_seconds=0.01, event_callback=events.append
        )
        pipeline.start()
        try:
            pipeline.arm_stream_watchdog(3)
            time.sleep(0.02)
            pipeline._check_stale()
            self.assertTrue(pipeline.stream_expected)
            self.assertEqual(
                [event.stage for event in events if event.event_type is PipelineEventType.STALE],
                ["watchdog"],
            )
        finally:
            pipeline.close()

    def test_watchdog_restart_resets_baseline_and_old_generation_cannot_disarm(self):
        events = []
        pipeline = AcquisitionPipeline(
            FakeWriter(), stale_after_seconds=0.01, event_callback=events.append
        )
        pipeline.start()
        try:
            pipeline.arm_stream_watchdog(1)
            time.sleep(0.02)
            pipeline._check_stale()
            self.assertEqual(len(events), 1)

            self.assertTrue(pipeline.disarm_stream_watchdog(1))
            pipeline.arm_stream_watchdog(2)
            self.assertFalse(pipeline.disarm_stream_watchdog(1))
            pipeline.mark_disconnected("late_old_disconnect", connection_generation=1)
            pipeline._check_stale()
            self.assertTrue(pipeline.stream_expected)
            self.assertEqual(pipeline.watchdog_generation, 2)
            self.assertEqual(len(events), 1)

            time.sleep(0.02)
            pipeline._check_stale()
            self.assertEqual(len(events), 2)
            self.assertEqual(events[-1].stage, "watchdog")
            pipeline.mark_disconnected(
                "current_connection", connection_generation=2
            )
            self.assertFalse(pipeline.stream_expected)
            self.assertIsNone(pipeline.watchdog_generation)
        finally:
            pipeline.close()

    def test_status_before_first_published_frame_does_not_invent_timestamps(self):
        writer = FakeWriter()
        pipeline = AcquisitionPipeline(writer)
        self.assertIsNone(pipeline.publish_status(FLAG_STALE))
        self.assertEqual(writer.calls, [])

    def test_overflow_latch_clears_only_after_successful_shared_commit(self):
        class FlakyWriter(FakeWriter):
            def __init__(self):
                super().__init__()
                self.fail = True

            def write_frame(self, payload, **kwargs):
                if self.fail:
                    raise OSError("busy")
                super().write_frame(payload, **kwargs)

        writer = FlakyWriter()
        pipeline = AcquisitionPipeline(writer)
        pipeline._overflow_epoch = 1
        pipeline._process(RawNotification(1, 2, 1, bytes(range(16))))
        writer.fail = False
        pipeline._process(RawNotification(3, 4, 2, bytes(range(16))))
        pipeline._process(RawNotification(5, 6, 3, bytes(range(16))))
        self.assertEqual(writer.calls[0][1]["flags"], FLAG_OVERFLOW)
        self.assertEqual(writer.calls[1][1]["flags"], 0)

    def test_raw_freshness_advances_for_bad_packet_but_valid_freshness_does_not(self):
        parsed = threading.Event()
        pipeline = AcquisitionPipeline(
            FakeWriter(),
            event_callback=lambda event: parsed.set()
            if event.event_type is PipelineEventType.PARSE_ERROR
            else None,
        )
        pipeline.start()
        self.assertTrue(
            pipeline.notification_callback(BleNotification(None, bytes(range(17)), 7))
        )
        self.assertTrue(parsed.wait(1))
        raw, valid, published = pipeline.freshness
        pipeline.close()
        self.assertIsNotNone(raw)
        self.assertIsNone(valid)
        self.assertIsNone(published)

    def test_raw_audit_rotates_and_closes_with_bounded_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw_packets.jsonl"
            audit = RawPacketAudit(path, max_bytes=512, backup_count=2)
            for index in range(10):
                audit.write(RawNotification(index, index, index, bytes(range(16))), None)
            audit.close()
            files = list(Path(directory).glob("raw_packets.jsonl*"))
            self.assertLessEqual(len(files), 3)
            self.assertTrue(all(file.stat().st_size <= 512 for file in files))

    def test_raw_audit_is_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FakeRecorder(Path(directory))
            pipeline = AcquisitionPipeline(FakeWriter())
            attach_recorder(pipeline, recorder)
            pipeline._process(RawNotification(1, 2, 1, bytes(range(16))))
            pipeline.stop_recorder(complete=True)
            self.assertFalse((Path(directory) / "raw_packets.jsonl").exists())

    def test_enabled_raw_audit_writes_policy_metadata_inside_session(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "sub-anonymous" / "session-anonymous"
            session.mkdir(parents=True)
            recorder = FakeRecorder(session)
            policy = RawAuditPolicy(
                enabled=True,
                max_bytes=2048,
                backup_count=1,
                payload_prefix_bytes=16,
                record_valid=False,
                record_invalid=True,
            )
            pipeline = AcquisitionPipeline(FakeWriter(), raw_audit_policy=policy)
            attach_recorder(pipeline, recorder)
            pipeline._process(RawNotification(1, 2, 1, bytes(range(16))))
            pipeline._process(RawNotification(3, 4, 2, bytes(range(17)), 23))
            pipeline.stop_recorder(complete=False, error="test")
            audit_path = session / "raw_packets.jsonl"
            self.assertEqual(audit_path.parent, recorder.session_dir)
            records = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[0]["record_type"], "raw_audit_policy")
            self.assertEqual(records[0]["payload_prefix_bytes"], 16)
            self.assertFalse(records[0]["record_valid"])
            self.assertTrue(records[0]["record_invalid"])
            self.assertEqual(
                records[0]["complete_payload_guarantee"],
                "exact_configured_wire_size",
            )
            self.assertEqual(
                records[0]["non_wire_length_policy"],
                "full_if_limits_allow_else_prefix_length_sha256",
            )
            self.assertEqual(records[1]["record_type"], "notification_packet_protocol")
            self.assertEqual(len(records), 3)
            self.assertEqual(records[2]["original_length"], 17)
            self.assertEqual(records[2]["connection_generation"], 23)
            self.assertEqual(len(bytes.fromhex(records[2]["payload_hex"])), 16)

    def test_raw_audit_preserves_complete_valid_28_byte_wire_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = NotificationPacketProtocol(
                PADDED28_PROTOCOL, 28, 16, "zero_suffix", "live_probe.json"
            )
            recorder = FakeRecorder(Path(directory))
            pipeline = AcquisitionPipeline(
                FakeWriter(),
                packet_protocol=protocol,
                raw_audit_policy=RawAuditPolicy(enabled=True),
            )
            attach_recorder(pipeline, recorder)
            payload = bytes(range(16)) + bytes(12)
            pipeline._process(RawNotification(1, 2, 1, payload))
            pipeline.stop_recorder(complete=True)
            records = [
                json.loads(line)
                for line in (Path(directory) / "raw_packets.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            packet = records[-1]
            self.assertEqual(bytes.fromhex(packet["payload_hex"]), payload)
            self.assertEqual(packet["original_length"], 28)
            self.assertEqual(packet["wire_protocol_mode"], PADDED28_PROTOCOL)

    def test_raw_audit_marks_29_byte_packet_truncated_with_28_byte_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = NotificationPacketProtocol(
                PADDED28_PROTOCOL, 28, 16, "zero_suffix", "live_probe.json"
            )
            required = RawPacketAudit.complete_record_size(
                protocol, include_parse_error=True
            )
            recorder = FakeRecorder(Path(directory))
            pipeline = AcquisitionPipeline(
                FakeWriter(),
                packet_protocol=protocol,
                raw_audit_policy=RawAuditPolicy(
                    enabled=True,
                    max_bytes=required,
                    payload_prefix_bytes=28,
                ),
            )
            attach_recorder(pipeline, recorder)
            payload = bytes(range(29))
            pipeline._process(RawNotification(1, 2, 1, payload))
            pipeline.stop_recorder(complete=False, error="test")
            records = [
                json.loads(line)
                for line in (Path(directory) / "raw_packets.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            packet = records[-1]
            self.assertTrue(packet["truncated"])
            self.assertEqual(packet["original_length"], 29)
            self.assertEqual(bytes.fromhex(packet["payload_hex"]), payload[:28])
            self.assertEqual(packet["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertIn("exactly 28", packet["parse_error"])

    def test_default_256_byte_prefix_can_preserve_short_non_wire_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = NotificationPacketProtocol(
                PADDED28_PROTOCOL, 28, 16, "zero_suffix", "live_probe.json"
            )
            recorder = FakeRecorder(Path(directory))
            pipeline = AcquisitionPipeline(
                FakeWriter(),
                packet_protocol=protocol,
                raw_audit_policy=RawAuditPolicy(enabled=True),
            )
            attach_recorder(pipeline, recorder)
            payloads = (bytes(range(27)), bytes(range(29)))
            for index, payload in enumerate(payloads, start=1):
                pipeline._process(RawNotification(index, index, index, payload))
            pipeline.stop_recorder(complete=False, error="test")
            records = [
                json.loads(line)
                for line in (Path(directory) / "raw_packets.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if "host_receive_index" in line
            ]
            self.assertEqual(len(records), 2)
            for record, payload in zip(records, payloads):
                self.assertEqual(bytes.fromhex(record["payload_hex"]), payload)
                self.assertEqual(record["original_length"], len(payload))
                self.assertNotIn("truncated", record)

    def test_raw_audit_truncates_one_record_and_keeps_every_file_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw_packets.jsonl"
            payload = b"x" * 10_000
            audit = RawPacketAudit(path, max_bytes=256, backup_count=1)
            for index in range(4):
                audit.write(
                    RawNotification(index, index, index, payload),
                    "parse failure " * 1_000,
                )
            audit.close()
            files = list(Path(directory).glob("raw_packets.jsonl*"))
            self.assertTrue(all(file.stat().st_size <= 256 for file in files))
            records = [
                json.loads(line)
                for file in files
                for line in file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all(record["truncated"] for record in records))
            self.assertTrue(all(record["original_length"] == len(payload) for record in records))
            self.assertTrue(all(len(record["sha256"]) == 64 for record in records))

    def test_raw_audit_recovers_from_rotate_and_reopen_failures(self):
        operations = ("unlink", "replace", "open")
        for operation in operations:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "raw_packets.jsonl"
                audit = RawPacketAudit(path, max_bytes=256, backup_count=1)
                audit.write(RawNotification(1, 2, 3, bytes(range(16))), None)
                oldest = path.with_name(path.name + ".1")
                if operation == "unlink":
                    oldest.write_text("old", encoding="utf-8")
                original = getattr(Path, operation)
                failed = []

                def flaky(candidate, *args, **kwargs):
                    relevant = (
                        (operation == "unlink" and candidate == oldest)
                        or (operation == "replace" and candidate == path)
                        or (
                            operation == "open"
                            and candidate == path
                            and args
                            and str(args[0]).startswith("a")
                        )
                    )
                    if relevant and not failed:
                        failed.append(True)
                        raise OSError(f"transient {operation}")
                    return original(candidate, *args, **kwargs)

                with mock.patch.object(Path, operation, flaky):
                    with self.assertRaisesRegex(OSError, f"transient {operation}"):
                        audit.write(RawNotification(4, 5, 6, bytes(range(16))), None)
                    audit.write(RawNotification(7, 8, 9, bytes(range(16))), None)
                audit.close()
                self.assertTrue(all(
                    file.stat().st_size <= 256
                    for file in Path(directory).glob("raw_packets.jsonl*")
                ))

    def test_raw_audit_failure_does_not_detach_recorder_or_block_fanout(self):
        class FlakyAudit:
            def __init__(self, path, **kwargs):
                self.calls = 0

            def write_policy(self, policy):
                pass

            def write_protocol(self, protocol):
                pass

            def write(self, item, parse_error):
                self.calls += 1
                if self.calls == 1:
                    raise OSError("audit unavailable")

            def close(self):
                pass

        events = []
        displayed = []
        recorder = FakeRecorder()
        pipeline = AcquisitionPipeline(
            FakeWriter(),
            display_callback=displayed.append,
            event_callback=events.append,
            raw_audit_factory=FlakyAudit,
            raw_audit_policy=RawAuditPolicy(enabled=True),
        )
        attach_recorder(pipeline, recorder)
        pipeline._process(RawNotification(1, 2, 1, bytes(range(16))))
        pipeline._process(RawNotification(3, 4, 2, bytes(range(16))))
        self.assertEqual(len(recorder.frames), 2)
        self.assertEqual(len(displayed), 2)
        self.assertEqual([event.stage for event in events], ["raw_audit"])
        pipeline.stop_recorder(complete=True)

    def test_raw_audit_close_is_idempotent_and_retryable(self):
        class RetryStream:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.fail = True

            @property
            def closed(self):
                return self.wrapped.closed

            def flush(self):
                if self.fail:
                    self.fail = False
                    raise OSError("flush busy")
                self.wrapped.flush()

            def close(self):
                self.wrapped.close()

        with tempfile.TemporaryDirectory() as directory:
            audit = RawPacketAudit(Path(directory) / "raw.jsonl", max_bytes=256)
            audit._stream = RetryStream(audit._stream)
            with self.assertRaisesRegex(OSError, "flush busy"):
                audit.close()
            self.assertFalse(audit._closed)
            audit.close()
            audit.close()
            self.assertTrue(audit._closed)

    def test_recorder_close_failure_retains_resource_for_real_retry(self):
        class RetryRecorder(FakeRecorder):
            def __init__(self):
                super().__init__()
                self.attempts = 0

            def close(self, **kwargs):
                self.attempts += 1
                if self.attempts == 1:
                    raise OSError("busy")
                super().close(**kwargs)

        recorder = RetryRecorder()
        pipeline = AcquisitionPipeline(FakeWriter())
        attach_recorder(pipeline, recorder)
        with self.assertRaises(PipelineCleanupError):
            pipeline.stop_recorder(complete=False, error="shutdown")
        pipeline.stop_recorder(complete=False, error="shutdown")
        self.assertEqual(recorder.attempts, 2)
        self.assertEqual(recorder.closed, [{"complete": False, "error": "shutdown"}])

    def test_unattached_startup_recorder_close_is_retained_for_retry(self):
        class RetryRecorder(FakeRecorder):
            def __init__(self):
                super().__init__()
                self.attempts = 0

            def close(self, **kwargs):
                self.attempts += 1
                if self.attempts == 1:
                    raise OSError("busy")
                super().close(**kwargs)

        recorder = RetryRecorder()
        pipeline = AcquisitionPipeline(FakeWriter())
        with self.assertRaises(PipelineCleanupError):
            pipeline.close_unattached_recorder(
                recorder, complete=False, error="start_failed"
            )
        self.assertTrue(pipeline.recorder_cleanup_pending)
        pipeline.stop_recorder(complete=True)
        self.assertFalse(pipeline.recorder_cleanup_pending)
        self.assertEqual(
            recorder.closed,
            [{"complete": False, "error": "start_failed"}],
        )

    def test_bad_and_oversize_packets_are_audited_without_refreshing_valid_freshness(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FakeRecorder(Path(directory))
            events = []
            pipeline = AcquisitionPipeline(
                FakeWriter(),
                event_callback=events.append,
                raw_audit_policy=RawAuditPolicy(enabled=True),
            )
            attach_recorder(pipeline, recorder)
            pipeline._process(RawNotification(1, 2, 1, bytes(range(17))))
            self.assertEqual(pipeline.freshness, (None, None, None))
            audit = Path(directory) / "raw_packets.jsonl"
            pipeline.stop_recorder(complete=False, error="test")
            text = audit.read_text(encoding="utf-8")
            self.assertIn(bytes(range(17)).hex(), text)
            self.assertIn("exactly 16", text)
            self.assertEqual(events[0].event_type, PipelineEventType.PARSE_ERROR)

    def test_first_recorder_failure_detaches_and_emits_one_typed_root_event(self):
        class BrokenRecorder(FakeRecorder):
            def record(self, frame, *, session_sample_index=None):
                raise OSError("disk full")

        recorder = BrokenRecorder()
        events = []
        pipeline = AcquisitionPipeline(FakeWriter(), event_callback=events.append)
        attach_recorder(pipeline, recorder)
        pipeline._process(RawNotification(1, 2, 1, bytes(range(16))))
        pipeline._process(RawNotification(3, 4, 2, bytes(range(16))))
        faults = [event for event in events if event.event_type is PipelineEventType.RECORDING_FAULT]
        self.assertEqual(len(faults), 1)
        self.assertIsInstance(faults[0].error, OSError)
        self.assertEqual(recorder.boundary_updates[-1]["incomplete_reason"], "OSError: disk full")
        self.assertEqual(recorder.closed, [{"complete": False, "error": "OSError: disk full"}])

    def test_recorder_fault_reports_pending_cleanup_and_public_retry(self):
        class BrokenRecorder(FakeRecorder):
            def __init__(self):
                super().__init__()
                self.close_attempts = 0

            def record(self, frame, *, session_sample_index=None):
                raise OSError("disk full")

            def close(self, **kwargs):
                self.close_attempts += 1
                if self.close_attempts == 1:
                    raise OSError("file busy")
                super().close(**kwargs)

        recorder = BrokenRecorder()
        events = []
        pipeline = AcquisitionPipeline(FakeWriter(), event_callback=events.append)
        attach_recorder(pipeline, recorder)
        pipeline._process(RawNotification(1, 2, 1, bytes(range(16))))
        fault = next(
            event for event in events
            if event.event_type is PipelineEventType.RECORDING_FAULT
        )
        self.assertTrue(fault.cleanup_pending)
        self.assertIsInstance(fault.error, OSError)
        self.assertIsInstance(fault.close_error, PipelineCleanupError)
        self.assertTrue(pipeline.recorder_cleanup_pending)
        pipeline.stop_recorder(complete=False, error="ignored_retry_intent")
        self.assertFalse(pipeline.recorder_cleanup_pending)
        self.assertEqual(recorder.close_attempts, 2)
        self.assertEqual(
            recorder.closed,
            [{"complete": False, "error": "OSError: disk full"}],
        )

    def test_real_csv_and_shared_file_represent_same_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared_path = root / "emg_shared_data_v2.bin"
            writer = SharedMemoryWriter(shared_path)
            subject_id = generate_subject_key()
            context = RecordingContext(
                subject_id=subject_id,
                action_label="fist",
                action_phase="hold",
                experiment_id="pipeline-test",
                hand_side=HandSide.LEFT,
            )
            recorder = DataRecorder(
                root / "data",
                subject_id=subject_id,
                device_id=DeviceKey("dev-" + "1" * 32),
                acquisition=AcquisitionMetadata(signal_chain=SignalChain(sample_format="uint8")),
                channels=8,
                flush_every=1,
                side=HandSide.LEFT,
                recording_context=context,
            )
            try:
                pipeline = AcquisitionPipeline(writer)
                attach_recorder(pipeline, recorder, context)
                pipeline._process(
                    RawNotification(123, 456, 9, bytes(range(16)), 73)
                )
                pipeline.stop_recorder(complete=True)
                writer.flush()
                with SharedMemoryReader(shared_path) as reader:
                    shared = reader.read()
                with recorder.csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                    row = next(csv.DictReader(stream))
                self.assertEqual(tuple(shared.unpack_samples()), tuple(int(row[f"channel_{i}"]) for i in range(1, 9)))
                self.assertEqual(shared.host_wall_timestamp_ns, int(row["host_wall_timestamp_ns"]))
                self.assertEqual(shared.host_monotonic_ns, int(row["host_monotonic_ns"]))
                self.assertEqual(shared.host_receive_index, int(row["host_receive_index"]))
                self.assertEqual(shared.generation, int(row["generation"]))
                self.assertEqual(shared.generation, writer.generation)
                self.assertEqual(shared.connection_generation, 73)
                self.assertEqual(int(row["connection_generation"]), 73)
                self.assertEqual(row["device_packet_sequence"], "")
                self.assertEqual(shared.flags & FLAG_OVERFLOW, 0)
            finally:
                writer.close()

    def test_session_boundary_excludes_pre_attach_queue_and_resets_session_index(self):
        pipeline = AcquisitionPipeline(FakeWriter())
        old = RawNotification(1, 1, 1, bytes(range(16)))
        pipeline._queue.put_nowait(old)
        pipeline._host_receive_index = 1
        pipeline._accepted_count = 1
        first = FakeRecorder()
        boundary = attach_recorder(pipeline, first)
        self.assertEqual(boundary.start_host_receive_index, 1)
        pipeline.start()
        self.assertTrue(pipeline.notification_callback(BleNotification(None, bytes(range(16)), 1)))
        token = pipeline.request_recorder_stop(complete=True)
        result = pipeline.finish_recorder_stop(token, timeout=2)
        self.assertEqual([frame.host_receive_index for frame in first.frames], [2])
        self.assertEqual(first.session_sample_indexes, [0])
        self.assertEqual((result.received_count, result.eligible_count, result.written_count), (1, 1, 1))

        second = FakeRecorder()
        attach_recorder(pipeline, second)
        self.assertTrue(pipeline.notification_callback(BleNotification(None, bytes(range(16)), 1)))
        pipeline.stop_recorder(complete=True, timeout=2)
        self.assertEqual(second.session_sample_indexes, [0])
        self.assertGreater(second.frames[0].sample_index, first.frames[0].sample_index)
        pipeline.close()

    def test_stop_watermark_is_retryable_after_timeout_and_excludes_later_frame(self):
        entered = threading.Event()
        release = threading.Event()
        displayed = []

        class BlockingRecorder(FakeRecorder):
            def record(self, frame, *, session_sample_index=None):
                entered.set()
                release.wait(2)
                super().record(frame, session_sample_index=session_sample_index)

        pipeline = AcquisitionPipeline(FakeWriter(), display_callback=displayed.append)
        recorder = BlockingRecorder()
        pipeline.start()
        attach_recorder(pipeline, recorder)
        self.assertTrue(pipeline.notification_callback(BleNotification(None, bytes(range(16)), 1)))
        self.assertTrue(entered.wait(1))
        started = time.monotonic()
        token = pipeline.request_recorder_stop(complete=True)
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertTrue(pipeline.notification_callback(BleNotification(None, bytes(range(16)), 1)))
        with self.assertRaisesRegex(TimeoutError, "tail drain"):
            pipeline.finish_recorder_stop(token, timeout=0.01)
        self.assertEqual(recorder.boundary_updates[-1]["incomplete_reason"], "tail_drain_timeout")
        release.set()
        result = pipeline.finish_recorder_stop(token, timeout=2)
        self.assertEqual([frame.host_receive_index for frame in recorder.frames], [1])
        self.assertEqual(result.tail_pending_count, 0)
        pipeline.close()
        self.assertEqual(
            [(frame.host_receive_index, frame.action_label) for frame in displayed],
            [(1, "fist"), (2, "")],
        )

    def test_tail_queue_drop_is_audited_without_a_following_frame(self):
        pipeline = AcquisitionPipeline(FakeWriter(), queue_capacity=1)
        recorder = FakeRecorder()
        attach_recorder(pipeline, recorder)
        pipeline.resume = lambda: None
        pipeline._accepting = True
        self.assertTrue(pipeline.notification_callback(BleNotification(None, bytes(range(16)), 1)))
        self.assertFalse(pipeline.notification_callback(BleNotification(None, bytes(range(16)), 1)))
        pipeline._process(pipeline._queue.get_nowait())
        pipeline._queue.task_done()
        with pipeline._progress:
            pipeline._processed_accepted_count = 1
            pipeline._processed_host_receive_index = 1
        result = pipeline.stop_recorder(complete=True)
        self.assertEqual(result.queue_drop_session, 1)
        self.assertEqual(result.tail_loss_count, 1)
        self.assertEqual(recorder.boundary_updates[-1]["incomplete_reason"], "queue_drop")

    def test_finish_stop_token_retries_a_transient_recorder_close(self):
        class RetryRecorder(FakeRecorder):
            def __init__(self):
                super().__init__()
                self.attempts = 0

            def close(self, **kwargs):
                self.attempts += 1
                if self.attempts == 1:
                    raise OSError("busy")
                super().close(**kwargs)

        pipeline = AcquisitionPipeline(FakeWriter())
        recorder = RetryRecorder()
        attach_recorder(pipeline, recorder)
        token = pipeline.request_recorder_stop(complete=True)
        with self.assertRaises(PipelineCleanupError):
            pipeline.finish_recorder_stop(token, timeout=1)
        result = pipeline.finish_recorder_stop(token, timeout=1)
        self.assertEqual(result.written_count, 0)
        self.assertEqual(recorder.attempts, 2)


if __name__ == "__main__":
    unittest.main()
