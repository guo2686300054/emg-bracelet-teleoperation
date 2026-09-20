"""Non-blocking BLE notification ingestion and deterministic EMG fan-out."""

from __future__ import annotations

import json
import hashlib
import inspect
import logging
import queue
import threading
import time
import traceback as traceback_module
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from bleak_ble import BleNotification
from emg_protocol import (
    EmgFrame,
    NotificationPacketProtocol,
    QualityFlags,
    to_shared_v2_flags,
)
from recording_context import RecordingContext
from shared_memory_v2 import (
    FLAG_DISCONNECTED,
    FLAG_OVERFLOW,
    FLAG_STALE,
    SAMPLE_UINT8,
    SharedMemoryWriter,
)

_fallback_logger = logging.getLogger("emg_app.pipeline_fallback")
RAW_AUDIT_PARSE_ERROR_CHARS = 256


@dataclass(frozen=True)
class RawNotification:
    host_wall_timestamp_ns: int
    host_monotonic_ns: int
    host_receive_index: int
    payload: bytes
    connection_generation: int = 0


class PipelineEventType(str, Enum):
    PARSE_ERROR = "parse_error"
    SINK_ERROR = "sink_error"
    RECORDING_FAULT = "recording_fault"
    STALE = "stale"
    DISCONNECTED = "disconnected"


@dataclass(frozen=True)
class PipelineEvent:
    event_type: PipelineEventType
    stage: str
    host_receive_index: Optional[int] = None
    error: Optional[BaseException] = None
    traceback_text: str = ""
    cleanup_pending: bool = False
    close_error: Optional[BaseException] = None


@dataclass(frozen=True)
class PipelineQualitySnapshot:
    """One lock-consistent, read-only view of live pipeline quality state.

    ``watchdog_stale`` is meaningful together with ``stream_expected``: it is
    false before the watchdog is armed and after disconnect/disarm, becomes
    true when the watchdog publishes STALE, and returns to false after a valid
    frame for the armed connection is published.
    """

    snapshot_monotonic_ns: int
    pipeline_started: bool
    accepting_notifications: bool
    stream_expected: bool
    connection_generation: Optional[int]
    queue_depth: int
    host_queue_dropped_count: int
    last_raw_received_monotonic_ns: Optional[int]
    last_parsed_monotonic_ns: Optional[int]
    last_published_monotonic_ns: Optional[int]
    watchdog_reference_monotonic_ns: Optional[int]
    watchdog_age_seconds: Optional[float]
    watchdog_stale: bool
    stale_after_seconds: float


class PipelineCleanupError(RuntimeError):
    def __init__(self, failures):
        self.failures = tuple(failures)
        super().__init__(
            "; ".join(f"{stage}: {type(error).__name__}: {error}" for stage, error in failures)
        )


@dataclass(frozen=True)
class RecordingBoundary:
    session_key: int
    start_host_receive_index: int
    start_accepted_count: int
    start_drop_total: int
    start_discarded_count: int


@dataclass(frozen=True)
class RecorderStopToken:
    session_key: int
    complete: bool
    error: Optional[str]
    end_host_receive_index: int
    end_accepted_count: int
    end_drop_total: int
    end_discarded_count: int


@dataclass(frozen=True)
class RecorderStopResult:
    start_host_receive_index: int
    end_host_receive_index: int
    received_count: int
    eligible_count: int
    written_count: int
    queue_drop_total: int
    queue_drop_session: int
    tail_pending_count: int
    tail_loss_count: int
    discarded_accepted_count: int


@dataclass
class _RecordingSession:
    key: int
    recorder: object
    recording_context: RecordingContext
    start_host_receive_index: int
    start_accepted_count: int
    start_drop_total: int
    start_discarded_count: int
    supports_session_sample_index: bool
    end_host_receive_index: Optional[int] = None
    end_accepted_count: Optional[int] = None
    end_drop_total: Optional[int] = None
    end_discarded_count: Optional[int] = None
    eligible_count: int = 0
    written_count: int = 0


@dataclass(frozen=True)
class RawAuditPolicy:
    """Bounded raw-notification retention policy.

    Pipeline startup guarantees complete payload capture only for notifications
    whose length equals the configured protocol ``wire_packet_size``.  This
    covers both valid packets and same-length packets rejected for content (for
    example, non-zero padding).  Other lengths are retained whole only when the
    configured prefix and file-size limits happen to permit it; otherwise the
    record contains a payload prefix, ``original_length`` and SHA-256 digest.
    """

    enabled: bool = False
    max_bytes: int = 2 * 1024 * 1024
    backup_count: int = 2
    payload_prefix_bytes: int = 256
    record_valid: bool = True
    record_invalid: bool = True

    def __post_init__(self):
        if self.max_bytes <= 0 or self.backup_count < 0 or self.payload_prefix_bytes < 0:
            raise ValueError("invalid raw audit policy")
        if self.enabled and not (self.record_valid or self.record_invalid):
            raise ValueError("enabled raw audit must record valid or invalid packets")


class RawPacketAudit:
    """Bounded JSONL sidecar stored within an opaque recording session path."""

    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int = 2 * 1024 * 1024,
        backup_count: int = 2,
        payload_prefix_bytes: int = 256,
    ):
        if max_bytes <= 0 or backup_count < 0 or payload_prefix_bytes < 0:
            raise ValueError("invalid raw audit rotation settings")
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.payload_prefix_bytes = payload_prefix_bytes
        self._stream = None
        self._closed = False
        self._protocol_mode = "unknown"
        self._bound_existing_files()
        self._ensure_stream()

    @staticmethod
    def complete_record_size(
        protocol: NotificationPacketProtocol, *, include_parse_error: bool = False
    ) -> int:
        maximum_uint64 = (1 << 64) - 1
        record = {
            "host_wall_timestamp_ns": maximum_uint64,
            "host_monotonic_ns": maximum_uint64,
            "host_receive_index": maximum_uint64,
            "connection_generation": maximum_uint64,
            "wire_protocol_mode": protocol.mode,
            "original_length": protocol.wire_packet_size,
            "payload_hex": (b"\xff" * protocol.wire_packet_size).hex(),
            "parse_error": (
                "\x00" * RAW_AUDIT_PARSE_ERROR_CHARS
                if include_parse_error
                else None
            ),
        }
        return len(
            (json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )

    def _bound_existing_files(self) -> None:
        for candidate in (
            self.path,
            *(self.path.with_name(f"{self.path.name}.{index}") for index in range(1, self.backup_count + 1)),
        ):
            if candidate.is_file() and candidate.stat().st_size > self.max_bytes:
                with candidate.open("r+b") as stream:
                    stream.truncate(self.max_bytes)

    def _ensure_stream(self) -> None:
        if self._stream is None or self._stream.closed:
            self._stream = self.path.open("a", encoding="utf-8", newline="\n")

    def _encode_record(self, item: RawNotification, parse_error: Optional[str]) -> str:
        bounded_error = None
        if parse_error is not None:
            bounded_error = str(parse_error).encode(
                "ascii", "backslashreplace"
            ).decode("ascii")[:RAW_AUDIT_PARSE_ERROR_CHARS]
        record = {
            "host_wall_timestamp_ns": item.host_wall_timestamp_ns,
            "host_monotonic_ns": item.host_monotonic_ns,
            "host_receive_index": item.host_receive_index,
            "connection_generation": item.connection_generation,
            "wire_protocol_mode": self._protocol_mode,
            "original_length": len(item.payload),
            "payload_hex": item.payload.hex(),
            "parse_error": bounded_error,
        }

        def encode(value):
            return json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n"

        line = encode(record)
        if (
            len(item.payload) <= self.payload_prefix_bytes
            and len(line.encode("utf-8")) <= self.max_bytes
        ):
            return line
        error_text = "" if bounded_error is None else bounded_error
        record = {
            "host_wall_timestamp_ns": item.host_wall_timestamp_ns,
            "host_monotonic_ns": item.host_monotonic_ns,
            "host_receive_index": item.host_receive_index,
            "connection_generation": item.connection_generation,
            "wire_protocol_mode": self._protocol_mode,
            "payload_hex": "",
            "parse_error": error_text,
            "original_length": len(item.payload),
            "sha256": hashlib.sha256(item.payload).hexdigest(),
            "truncated": True,
        }
        if self._protocol_mode == "unknown":
            record.pop("wire_protocol_mode")
        while error_text and len(encode(record).encode("utf-8")) > self.max_bytes:
            error_text = error_text[: len(error_text) // 2]
            record["parse_error"] = error_text
        available = min(
            self.payload_prefix_bytes,
            max(0, (self.max_bytes - len(encode(record).encode("utf-8"))) // 2),
        )
        record["payload_hex"] = item.payload[:available].hex()
        while len(encode(record).encode("utf-8")) > self.max_bytes and record["payload_hex"]:
            record["payload_hex"] = record["payload_hex"][:-2]
        line = encode(record)
        if len(line.encode("utf-8")) > self.max_bytes:
            raise ValueError("raw audit max_bytes cannot hold required diagnostic fields")
        return line

    def _rotate(self) -> None:
        if self._stream is not None:
            stream = self._stream
            try:
                stream.close()
            finally:
                if stream.closed:
                    self._stream = None
        if self.backup_count:
            oldest = self.path.with_name(f"{self.path.name}.{self.backup_count}")
            oldest.unlink(missing_ok=True)
            for index in range(self.backup_count - 1, 0, -1):
                source = self.path.with_name(f"{self.path.name}.{index}")
                if source.exists():
                    source.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
            if self.path.exists():
                self.path.replace(self.path.with_name(f"{self.path.name}.1"))
        else:
            self.path.unlink(missing_ok=True)
        self._ensure_stream()

    def _append_line(self, line: str) -> None:
        encoded_size = len(line.encode("utf-8"))
        if encoded_size > self.max_bytes:
            raise ValueError("raw audit record exceeds max_bytes")
        self._ensure_stream()
        self._stream.flush()
        if self._stream.tell() and self._stream.tell() + encoded_size > self.max_bytes:
            self._rotate()
        self._stream.write(line)

    def write_policy(self, policy: RawAuditPolicy) -> None:
        record = {
            "record_type": "raw_audit_policy",
            "max_bytes": policy.max_bytes,
            "backup_count": policy.backup_count,
            "payload_prefix_bytes": policy.payload_prefix_bytes,
            "record_valid": policy.record_valid,
            "record_invalid": policy.record_invalid,
            "retention_policy": "bounded_rotating_files",
            "complete_payload_guarantee": "exact_configured_wire_size",
            "non_wire_length_policy": "full_if_limits_allow_else_prefix_length_sha256",
        }
        self._append_line(
            json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
        )

    def write_protocol(self, protocol: NotificationPacketProtocol) -> None:
        self._protocol_mode = protocol.mode
        record = {
            "record_type": "notification_packet_protocol",
            "mode": protocol.mode,
            "wire_packet_size": protocol.wire_packet_size,
            "logical_packet_size": protocol.logical_packet_size,
            "padding_rule": protocol.padding_rule,
            "evidence_ref": protocol.evidence_ref,
        }
        self._append_line(
            json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
        )

    def write(self, item: RawNotification, parse_error: Optional[str]) -> None:
        if self._closed:
            raise RuntimeError("raw packet audit is closed")
        line = self._encode_record(item, parse_error)
        self._append_line(line)

    def close(self) -> None:
        if self._closed:
            return
        if self._stream is None or self._stream.closed:
            self._stream = None
            self._closed = True
            return
        stream = self._stream
        try:
            stream.flush()
            stream.close()
        finally:
            if stream.closed:
                self._stream = None
                self._closed = True


class AcquisitionPipeline:
    """Move work out of the Bleak callback and fan out one canonical frame."""

    def __init__(
        self,
        shared_writer: SharedMemoryWriter,
        *,
        channels: int = 8,
        queue_capacity: int = 256,
        stale_after_seconds: float = 1.0,
        display_callback: Optional[Callable[[EmgFrame], None]] = None,
        event_callback: Optional[Callable[[PipelineEvent], None]] = None,
        frame_callback: Optional[Callable[[EmgFrame], None]] = None,
        shared_callback: Optional[Callable[[EmgFrame], None]] = None,
        raw_audit_factory: Callable[[Path], RawPacketAudit] = RawPacketAudit,
        raw_audit_policy: Optional[RawAuditPolicy] = None,
        packet_protocol: Optional[NotificationPacketProtocol] = None,
    ) -> None:
        if channels != 8:
            raise ValueError("the current odd-byte logical protocol requires 8 channels")
        if queue_capacity <= 0 or stale_after_seconds <= 0:
            raise ValueError("queue capacity and stale timeout must be positive")
        generation = getattr(shared_writer, "generation", None)
        if not isinstance(generation, int) or isinstance(generation, bool) or generation <= 0:
            raise ValueError("shared writer must expose a positive generation")
        self.shared_writer = shared_writer
        self.generation = generation
        self.channels = channels
        self.stale_after_seconds = stale_after_seconds
        self.display_callback = display_callback
        self.event_callback = event_callback
        self.frame_callback = frame_callback
        self.shared_callback = shared_callback
        self.raw_audit_factory = raw_audit_factory
        self.raw_audit_policy = raw_audit_policy or RawAuditPolicy()
        self.packet_protocol = packet_protocol or NotificationPacketProtocol()
        if self.raw_audit_policy.enabled and (
            self.raw_audit_policy.record_valid or self.raw_audit_policy.record_invalid
        ):
            if (
                self.raw_audit_policy.payload_prefix_bytes
                < self.packet_protocol.wire_packet_size
            ):
                raise ValueError(
                    "enabled valid RawAudit payload_prefix_bytes must cover wire packet"
                )
            required = max(
                RawPacketAudit.complete_record_size(
                    self.packet_protocol,
                    include_parse_error=self.raw_audit_policy.record_invalid,
                ),
                RawPacketAudit.complete_record_size(self.packet_protocol),
            )
            if self.raw_audit_policy.max_bytes < required:
                raise ValueError(
                    f"enabled valid RawAudit max_bytes must be at least {required}"
                )
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_capacity)
        self._sentinel = object()
        self._lock = threading.RLock()
        self._fanout_lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._accepting = False
        self._closed = False
        self._host_receive_index = 0
        self._accepted_count = 0
        self._processed_accepted_count = 0
        self._processed_host_receive_index = 0
        self._discarded_accepted_count = 0
        self._progress = threading.Condition(self._lock)
        self._sample_index = 0
        self._dropped_count = 0
        self._overflow_epoch = 0
        self._overflow_committed_epoch = 0
        self._last_raw_received_ns: Optional[int] = None
        self._last_parsed_ns: Optional[int] = None
        self._last_published_ns: Optional[int] = None
        self._last_frame: Optional[EmgFrame] = None
        self._last_published_frame: Optional[EmgFrame] = None
        self._stale_published = False
        self._stream_expected = False
        self._watchdog_generation: Optional[int] = None
        self._watchdog_baseline_ns: Optional[int] = None
        self._watchdog_last_published_ns: Optional[int] = None
        self._recorder = None
        self._recording_session: Optional[_RecordingSession] = None
        self._next_recording_session_key = 1
        self._raw_audit: Optional[RawPacketAudit] = None
        self._closing_recorder = None
        self._closing_audit: Optional[RawPacketAudit] = None
        self._closing_intent = None
        self._closing_session: Optional[_RecordingSession] = None
        self._closing_stop_result: Optional[RecorderStopResult] = None
        self._recorder_stop_token: Optional[RecorderStopToken] = None

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def last_frame(self) -> Optional[EmgFrame]:
        with self._lock:
            return self._last_frame

    @property
    def freshness(self):
        with self._lock:
            return self._last_raw_received_ns, self._last_parsed_ns, self._last_published_ns

    def quality_snapshot(self) -> PipelineQualitySnapshot:
        """Return an immutable quality snapshot without exposing mutable frames."""
        snapshot_ns = time.monotonic_ns()
        with self._lock:
            reference_ns = self._watchdog_baseline_ns
            if self._watchdog_last_published_ns is not None and (
                reference_ns is None
                or self._watchdog_last_published_ns > reference_ns
            ):
                reference_ns = self._watchdog_last_published_ns
            if not self._stream_expected:
                reference_ns = None
            watchdog_age_seconds = (
                None
                if reference_ns is None
                else max(0, snapshot_ns - reference_ns) / 1_000_000_000
            )
            return PipelineQualitySnapshot(
                snapshot_monotonic_ns=snapshot_ns,
                pipeline_started=self._thread is not None,
                accepting_notifications=self._accepting,
                stream_expected=self._stream_expected,
                connection_generation=self._watchdog_generation,
                queue_depth=self._queue.qsize(),
                host_queue_dropped_count=self._dropped_count,
                last_raw_received_monotonic_ns=self._last_raw_received_ns,
                last_parsed_monotonic_ns=self._last_parsed_ns,
                last_published_monotonic_ns=self._last_published_ns,
                watchdog_reference_monotonic_ns=reference_ns,
                watchdog_age_seconds=watchdog_age_seconds,
                watchdog_stale=self._stream_expected and self._stale_published,
                stale_after_seconds=self.stale_after_seconds,
            )

    @property
    def recorder_cleanup_pending(self) -> bool:
        with self._lock:
            return self._closing_recorder is not None or self._closing_audit is not None

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("pipeline is closed")
            if self._thread is not None:
                return
            self._accepting = True
            self._thread = threading.Thread(
                target=self._run, name="emg-acquisition-consumer", daemon=True
            )
            self._thread.start()

    def resume(self) -> None:
        with self._lock:
            if self._closed or self._thread is None:
                raise RuntimeError("pipeline is not running")
            self._accepting = True

    @property
    def stream_expected(self) -> bool:
        with self._lock:
            return self._stream_expected

    @property
    def watchdog_generation(self) -> Optional[int]:
        with self._lock:
            return self._watchdog_generation

    def arm_stream_watchdog(self, connection_generation: int) -> None:
        if (
            not isinstance(connection_generation, int)
            or isinstance(connection_generation, bool)
            or connection_generation <= 0
        ):
            raise ValueError("connection_generation must be a positive integer")
        with self._lock:
            if self._closed or self._thread is None:
                raise RuntimeError("pipeline is not running")
            self._stream_expected = True
            self._watchdog_generation = connection_generation
            self._watchdog_baseline_ns = time.monotonic_ns()
            self._watchdog_last_published_ns = None
            self._stale_published = False

    def disarm_stream_watchdog(
        self, connection_generation: Optional[int] = None
    ) -> bool:
        with self._lock:
            if (
                connection_generation is not None
                and self._watchdog_generation != connection_generation
            ):
                return False
            was_armed = self._stream_expected
            self._stream_expected = False
            self._watchdog_generation = None
            self._watchdog_baseline_ns = None
            self._watchdog_last_published_ns = None
            self._stale_published = True
            return was_armed

    def stop_accepting(self) -> None:
        """Close the receive gate without touching sinks used by the consumer."""
        with self._lock:
            self._accepting = False
            self._stream_expected = False
            self._watchdog_generation = None
            self._watchdog_baseline_ns = None
            self._watchdog_last_published_ns = None
            self._stale_published = True

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def notification_callback(self, notification: BleNotification) -> bool:
        payload = bytes(notification.payload)
        with self._lock:
            if not self._accepting:
                return False
            # Assign the host ordering fields and publish to the FIFO as one
            # transaction. BLE backends may invoke callbacks concurrently;
            # assigning an index before releasing the lock but enqueueing
            # afterwards allowed frame N+1 to overtake frame N.
            wall_ns = notification.host_wall_timestamp_ns
            monotonic_ns = notification.host_monotonic_ns
            if not isinstance(wall_ns, int) or wall_ns <= 0:
                wall_ns = time.time_ns()
            if not isinstance(monotonic_ns, int) or monotonic_ns <= 0:
                monotonic_ns = time.monotonic_ns()
            self._host_receive_index += 1
            receive_index = self._host_receive_index
            self._last_raw_received_ns = monotonic_ns
            item = RawNotification(
                wall_ns,
                monotonic_ns,
                receive_index,
                payload,
                notification.connection_generation,
            )
            try:
                self._queue.put_nowait(item)
                self._accepted_count += 1
                return True
            except queue.Full:
                self._dropped_count += 1
                self._overflow_epoch += 1
                return False

    def set_recorder(
        self, recorder, *, recording_context: Optional[RecordingContext] = None
    ) -> RecordingBoundary:
        with self._fanout_lock:
            with self._lock:
                if not isinstance(recording_context, RecordingContext):
                    raise ValueError("recording_context must be a RecordingContext")
                if self._closed:
                    raise RuntimeError("pipeline is closed")
                if (
                    self._recorder is not None
                    or self._closing_recorder is not None
                    or self._closing_audit is not None
                ):
                    raise RuntimeError("a recorder is active or awaiting close retry")
            audit = None
            try:
                if self.raw_audit_policy.enabled:
                    audit = self.raw_audit_factory(
                        Path(recorder.session_dir) / "raw_packets.jsonl",
                        max_bytes=self.raw_audit_policy.max_bytes,
                        backup_count=self.raw_audit_policy.backup_count,
                        payload_prefix_bytes=self.raw_audit_policy.payload_prefix_bytes,
                    )
                    for method_name, argument in (
                        ("write_policy", self.raw_audit_policy),
                        ("write_protocol", self.packet_protocol),
                    ):
                        method = getattr(audit, method_name, None)
                        if not callable(method):
                            raise TypeError(
                                f"raw audit {method_name} must be callable"
                            )
                        method(argument)
                with self._lock:
                    key = self._next_recording_session_key
                    self._next_recording_session_key += 1
                    session = _RecordingSession(
                        key,
                        recorder,
                        recording_context,
                        self._host_receive_index,
                        self._accepted_count,
                        self._dropped_count,
                        self._discarded_accepted_count,
                        any(
                            parameter.kind is inspect.Parameter.VAR_KEYWORD
                            or parameter.name == "session_sample_index"
                            for parameter in inspect.signature(recorder.record).parameters.values()
                        ),
                    )
                    boundary = RecordingBoundary(
                        key,
                        session.start_host_receive_index,
                        session.start_accepted_count,
                        session.start_drop_total,
                        session.start_discarded_count,
                    )
                configure = getattr(recorder, "configure_session_boundary", None)
                if callable(configure):
                    configure(
                        start_host_receive_index=session.start_host_receive_index,
                        start_drop_total=session.start_drop_total,
                    )
            except BaseException as setup_error:
                if audit is not None:
                    try:
                        audit.close()
                    except BaseException as cleanup_error:
                        with self._lock:
                            self._closing_audit = audit
                        raise PipelineCleanupError(
                            (
                                ("recorder_setup", setup_error),
                                ("raw_audit_close", cleanup_error),
                            )
                        ) from setup_error
                raise
            with self._lock:
                self._recorder = recorder
                self._raw_audit = audit
                self._recording_session = session
            return boundary

    def _detach_recorder(self):
        with self._lock:
            if self._recorder is not None:
                self._closing_recorder = self._recorder
                self._closing_audit = self._raw_audit
                self._recorder = None
                self._raw_audit = None
                self._closing_session = self._recording_session
                self._recording_session = None
            return self._closing_recorder, self._closing_audit

    def close_unattached_recorder(
        self, recorder, *, complete: bool, error: Optional[str] = None
    ) -> None:
        """Make rollback cleanup retryable before a recorder reaches fan-out."""
        with self._fanout_lock:
            with self._lock:
                if (
                    self._recorder is not None
                    or self._closing_recorder is not None
                    or self._closing_audit is not None
                ):
                    raise RuntimeError("a recorder is active or awaiting close retry")
                self._closing_recorder = recorder
                self._closing_intent = (complete, error)
        self.stop_recorder(complete=complete, error=error)

    def _close_detached_recorder(self, *, complete: bool, error: Optional[str] = None) -> None:
        failures = []
        with self._fanout_lock:
            recorder, audit = self._detach_recorder()
            requested = (complete, error)
            with self._lock:
                if self._closing_intent is None and recorder is not None:
                    self._closing_intent = requested
                elif recorder is not None:
                    requested = self._closing_intent
            if audit is not None:
                try:
                    audit.close()
                    with self._lock:
                        self._closing_audit = None
                except BaseException as exc:
                    failures.append(("raw_audit_close", exc))
            if recorder is not None:
                try:
                    recorder.close(complete=requested[0], error=requested[1])
                    with self._lock:
                        self._closing_recorder = None
                        self._closing_intent = None
                except BaseException as exc:
                    failures.append(("recorder_close", exc))
        if failures:
            raise PipelineCleanupError(failures)

    def request_recorder_stop(
        self, *, complete: bool, error: Optional[str] = None
    ) -> RecorderStopToken:
        if complete and error is not None:
            raise ValueError("a complete session cannot have an error")
        with self._lock:
            session = self._recording_session
            if session is None:
                if self._recorder_stop_token is not None:
                    return self._recorder_stop_token
                raise RuntimeError("no active recorder")
            if self._recorder_stop_token is not None:
                token = self._recorder_stop_token
                if (token.complete, token.error) != (complete, error):
                    raise ValueError("conflicting recorder stop intent")
                return token
            session.end_host_receive_index = self._host_receive_index
            session.end_accepted_count = self._accepted_count
            session.end_drop_total = self._dropped_count
            session.end_discarded_count = self._discarded_accepted_count
            token = RecorderStopToken(
                session.key,
                complete,
                error,
                session.end_host_receive_index,
                session.end_accepted_count,
                session.end_drop_total,
                session.end_discarded_count,
            )
            self._recorder_stop_token = token
            return token

    def _stop_result(self, session: _RecordingSession) -> RecorderStopResult:
        assert session.end_host_receive_index is not None
        assert session.end_accepted_count is not None
        assert session.end_drop_total is not None
        assert session.end_discarded_count is not None
        pending = max(0, session.end_accepted_count - self._processed_accepted_count)
        discarded = max(0, session.end_discarded_count - session.start_discarded_count)
        trailing = max(
            0,
            session.end_host_receive_index
            - max(session.start_host_receive_index, self._processed_host_receive_index),
        )
        return RecorderStopResult(
            session.start_host_receive_index,
            session.end_host_receive_index,
            max(0, session.end_host_receive_index - session.start_host_receive_index),
            session.eligible_count,
            session.written_count,
            session.end_drop_total,
            max(0, session.end_drop_total - session.start_drop_total),
            pending,
            max(
                trailing,
                max(0, session.end_drop_total - session.start_drop_total) + discarded,
            ),
            discarded,
        )

    @staticmethod
    def _incomplete_reason(token: RecorderStopToken, result: RecorderStopResult) -> Optional[str]:
        if not token.complete:
            return token.error or "recording_aborted"
        if result.tail_pending_count:
            return "tail_drain_timeout"
        if result.discarded_accepted_count:
            return "accepted_frames_discarded"
        if result.queue_drop_session or result.tail_loss_count:
            return "queue_drop"
        return None

    def _persist_stop_result(
        self, recorder, token: RecorderStopToken, result: RecorderStopResult
    ) -> None:
        update = getattr(recorder, "update_session_boundary", None)
        if callable(update):
            update(
                end_host_receive_index=result.end_host_receive_index,
                received_count=result.received_count,
                eligible_count=result.eligible_count,
                written_count=result.written_count,
                queue_drop_total=result.queue_drop_total,
                queue_drop_session=result.queue_drop_session,
                tail_pending_count=result.tail_pending_count,
                tail_loss_count=result.tail_loss_count,
                incomplete_reason=self._incomplete_reason(token, result),
            )

    def finish_recorder_stop(
        self, token: RecorderStopToken, *, timeout: float
    ) -> RecorderStopResult:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        with self._lock:
            retry_result = (
                self._closing_stop_result
                if token is self._recorder_stop_token
                and self._recording_session is None
                and (self._closing_recorder is not None or self._closing_audit is not None)
                else None
            )
        if retry_result is not None:
            self._close_detached_recorder(complete=token.complete, error=token.error)
            with self._lock:
                self._recorder_stop_token = None
                self._closing_session = None
                self._closing_stop_result = None
            return retry_result
        deadline = time.monotonic() + timeout
        with self._progress:
            if token is not self._recorder_stop_token:
                raise ValueError("recorder stop token is not active")
            while self._processed_accepted_count < token.end_accepted_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    session = self._recording_session
                    if session is None:
                        raise RuntimeError("recording session disappeared during stop")
                    result = self._stop_result(session)
                    recorder = session.recorder
                    break
                self._progress.wait(remaining)
            else:
                result = None
                recorder = None
        if result is not None:
            self._persist_stop_result(recorder, token, result)
            raise TimeoutError(
                f"recorder tail drain timed out with {result.tail_pending_count} accepted frame(s) pending"
            )
        with self._fanout_lock:
            with self._lock:
                if token is not self._recorder_stop_token:
                    raise ValueError("recorder stop token is not active")
                session = self._recording_session
                if session is None:
                    raise RuntimeError("recording session disappeared during stop")
                result = self._stop_result(session)
                recorder = session.recorder
            self._persist_stop_result(recorder, token, result)
            self._detach_recorder()
            with self._lock:
                self._closing_stop_result = result
        self._close_detached_recorder(complete=token.complete, error=token.error)
        with self._lock:
            self._recorder_stop_token = None
            self._closing_session = None
            self._closing_stop_result = None
        return result

    def stop_recorder(
        self, *, complete: bool, error: Optional[str] = None, timeout: float = 5.0
    ) -> Optional[RecorderStopResult]:
        with self._lock:
            active = self._recording_session is not None
            token = self._recorder_stop_token
            closing = self._closing_recorder is not None or self._closing_audit is not None
        if active:
            if token is None:
                token = self.request_recorder_stop(complete=complete, error=error)
            return self.finish_recorder_stop(token, timeout=timeout)
        if closing:
            self._close_detached_recorder(complete=complete, error=error)
            with self._lock:
                self._recorder_stop_token = None
                self._closing_session = None
                self._closing_stop_result = None
            return None
        return None

    def _emit(self, event: PipelineEvent) -> None:
        if self.event_callback is None:
            return
        try:
            self.event_callback(event)
        except Exception:
            _fallback_logger.exception("pipeline event callback failed")

    @staticmethod
    def _event(event_type, stage, item=None, error=None):
        return PipelineEvent(
            event_type,
            stage,
            None if item is None else item.host_receive_index,
            error,
            "" if error is None else "".join(traceback_module.format_exception(type(error), error, error.__traceback__)),
        )

    @staticmethod
    def parse_notification(
        item: RawNotification,
        sample_index: int,
        overflow: bool,
        generation: int,
        packet_protocol: Optional[NotificationPacketProtocol] = None,
        *,
        action_label: str = "",
        action_phase: str = "",
    ) -> EmgFrame:
        protocol = packet_protocol or NotificationPacketProtocol()
        logical_payload = protocol.extract_logical_payload(item.payload)
        values = tuple(logical_payload[index] for index in range(1, 16, 2))
        flags = QualityFlags.VALID | QualityFlags.HOST_WALL_TIME_VALID | QualityFlags.HOST_MONOTONIC_VALID | QualityFlags.HOST_RECEIVE_INDEX_VALID
        if overflow:
            flags |= QualityFlags.OVERFLOW
        return EmgFrame(
            channel_values=values,
            host_wall_timestamp_ns=item.host_wall_timestamp_ns,
            host_monotonic_ns=item.host_monotonic_ns,
            host_receive_index=item.host_receive_index,
            sample_index=sample_index,
            generation=generation,
            connection_generation=item.connection_generation,
            device_packet_sequence=None,
            device_sample_counter=None,
            device_time_ticks=None,
            action_label=action_label,
            action_phase=action_phase,
            quality_flags=flags,
        )

    def _recording_annotation(self, item: RawNotification) -> tuple[str, str]:
        """Return the latched annotation only for frames inside the session boundary."""
        with self._lock:
            session = self._recording_session
            eligible = (
                session is not None
                and item.host_receive_index > session.start_host_receive_index
                and (
                    session.end_host_receive_index is None
                    or item.host_receive_index <= session.end_host_receive_index
                )
            )
            if not eligible:
                return "", ""
            context = session.recording_context
            return context.action_label, context.action_phase

    def _recording_fault(self, error: BaseException, item: RawNotification) -> None:
        metadata_failure = None
        with self._lock:
            session = self._recording_session
            if session is not None:
                session.end_host_receive_index = self._host_receive_index
                session.end_accepted_count = self._accepted_count
                session.end_drop_total = self._dropped_count
                session.end_discarded_count = self._discarded_accepted_count
                fault_token = RecorderStopToken(
                    session.key,
                    False,
                    f"{type(error).__name__}: {error}",
                    session.end_host_receive_index,
                    session.end_accepted_count,
                    session.end_drop_total,
                    session.end_discarded_count,
                )
                fault_result = self._stop_result(session)
            else:
                fault_token = None
                fault_result = None
        if fault_token is not None and fault_result is not None:
            try:
                self._persist_stop_result(session.recorder, fault_token, fault_result)
            except BaseException as exc:
                metadata_failure = exc
        self._detach_recorder()
        close_error = None
        try:
            self.stop_recorder(complete=False, error=f"{type(error).__name__}: {error}")
        except PipelineCleanupError as cleanup_error:
            close_error = cleanup_error
            if hasattr(error, "add_note"):
                error.add_note(f"recording cleanup pending retry: {cleanup_error}")
        if metadata_failure is not None:
            failures = [("session_boundary_update", metadata_failure)]
            if close_error is not None:
                failures.extend(close_error.failures)
            close_error = PipelineCleanupError(failures)
            if hasattr(error, "add_note"):
                error.add_note(f"recording boundary metadata update failed: {metadata_failure}")
        event = self._event(PipelineEventType.RECORDING_FAULT, "recorder_write", item, error)
        self._emit(
            PipelineEvent(
                event.event_type,
                event.stage,
                event.host_receive_index,
                event.error,
                event.traceback_text,
                self.recorder_cleanup_pending,
                close_error,
            )
        )

    def _write_audit(self, item: RawNotification, parse_error: Optional[str]) -> bool:
        with self._lock:
            audit = self._raw_audit
            session = self._recording_session
            eligible = (
                audit is not None
                and session is not None
                and item.host_receive_index > session.start_host_receive_index
                and (
                    session.end_host_receive_index is None
                    or item.host_receive_index <= session.end_host_receive_index
                )
            )
        if not eligible:
            return True
        if parse_error is None and not self.raw_audit_policy.record_valid:
            return True
        if parse_error is not None and not self.raw_audit_policy.record_invalid:
            return True
        try:
            audit.write(item, parse_error)
            return True
        except Exception as exc:
            self._emit(self._event(PipelineEventType.SINK_ERROR, "raw_audit", item, exc))
            return False

    def _publish_frame(self, frame: EmgFrame, item: RawNotification) -> bool:
        shared_ok = False
        status_flags = to_shared_v2_flags(frame.quality_flags) & FLAG_OVERFLOW
        try:
            self.shared_writer.write_frame(
                bytes(int(value) for value in frame.channel_values),
                host_wall_timestamp_ns=frame.host_wall_timestamp_ns,
                host_monotonic_ns=frame.host_monotonic_ns,
                host_receive_index=frame.host_receive_index,
                connection_generation=frame.connection_generation,
                flags=status_flags,
                channel_count=self.channels,
                sample_count=1,
                sample_format=SAMPLE_UINT8,
            )
            shared_ok = True
            if self.shared_callback is not None:
                self.shared_callback(frame)
        except Exception as exc:
            self._emit(self._event(PipelineEventType.SINK_ERROR, "shared_write", item, exc))
        with self._lock:
            recorder = self._recorder
            session = self._recording_session
            eligible = (
                recorder is not None
                and session is not None
                and frame.host_receive_index > session.start_host_receive_index
                and (
                    session.end_host_receive_index is None
                    or frame.host_receive_index <= session.end_host_receive_index
                )
            )
            session_sample_index = session.eligible_count if eligible else None
            if eligible:
                session.eligible_count += 1
        if eligible:
            try:
                if session.supports_session_sample_index:
                    recorder.record(frame, session_sample_index=session_sample_index)
                else:
                    recorder.record(frame)
                with self._lock:
                    session.written_count += 1
            except Exception as exc:
                self._recording_fault(exc, item)
        for stage, callback in (("frame_observer", self.frame_callback), ("display", self.display_callback)):
            if callback is not None:
                try:
                    callback(frame)
                except Exception as exc:
                    self._emit(self._event(PipelineEventType.SINK_ERROR, stage, item, exc))
        return shared_ok

    def _process(self, item: RawNotification) -> None:
        with self._fanout_lock:
            with self._lock:
                next_index = self._sample_index + 1
                overflow_epoch = self._overflow_epoch
                overflow = overflow_epoch > self._overflow_committed_epoch
            action_label, action_phase = self._recording_annotation(item)
            try:
                frame = self.parse_notification(
                    item,
                    next_index,
                    overflow,
                    self.generation,
                    self.packet_protocol,
                    action_label=action_label,
                    action_phase=action_phase,
                )
            except Exception as exc:
                self._write_audit(item, f"{type(exc).__name__}: {exc}")
                self._emit(self._event(PipelineEventType.PARSE_ERROR, "parse", item, exc))
                return
            with self._lock:
                self._last_parsed_ns = item.host_monotonic_ns
            self._write_audit(item, None)
            shared_ok = self._publish_frame(frame, item)
            with self._lock:
                self._sample_index = next_index
                self._last_frame = frame
                if shared_ok:
                    self._last_published_ns = item.host_monotonic_ns
                    self._last_published_frame = frame
                    if (
                        self._stream_expected
                        and self._watchdog_generation == frame.connection_generation
                        and self._watchdog_baseline_ns is not None
                        and item.host_monotonic_ns >= self._watchdog_baseline_ns
                    ):
                        self._watchdog_last_published_ns = max(
                            self._watchdog_last_published_ns
                            or self._watchdog_baseline_ns,
                            item.host_monotonic_ns,
                        )
                        self._stale_published = False
                    if self._overflow_epoch == overflow_epoch:
                        self._overflow_committed_epoch = overflow_epoch

    def publish_status(self, flags: int):
        with self._lock:
            frame = self._last_published_frame
        if frame is None:
            return None
        return self.shared_writer.write_frame(
            bytes(int(value) for value in frame.channel_values),
            host_wall_timestamp_ns=frame.host_wall_timestamp_ns,
            host_monotonic_ns=frame.host_monotonic_ns,
            host_receive_index=frame.host_receive_index,
            connection_generation=frame.connection_generation,
            flags=flags,
            channel_count=self.channels,
            sample_count=1,
            sample_format=SAMPLE_UINT8,
        )

    def _discard_pending(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
                if isinstance(item, RawNotification):
                    with self._progress:
                        self._processed_accepted_count += 1
                        self._discarded_accepted_count += 1
                        self._processed_host_receive_index = max(
                            self._processed_host_receive_index, item.host_receive_index
                        )
                        self._progress.notify_all()
                self._queue.task_done()
            except queue.Empty:
                return

    def mark_disconnected(
        self,
        reason: str = "ble_disconnected",
        *,
        connection_generation: Optional[int] = None,
    ) -> None:
        with self._lock:
            if (
                connection_generation is not None
                and self._watchdog_generation is not None
                and self._watchdog_generation != connection_generation
            ):
                return
            self._accepting = False
            if (
                connection_generation is None
                or self._watchdog_generation == connection_generation
            ):
                self._stream_expected = False
                self._watchdog_generation = None
                self._watchdog_baseline_ns = None
                self._watchdog_last_published_ns = None
                self._stale_published = True
        self._discard_pending()
        with self._fanout_lock:
            try:
                self.stop_recorder(complete=False, error=reason)
            except Exception as exc:
                event = self._event(
                    PipelineEventType.RECORDING_FAULT,
                    "recorder_disconnect_close",
                    error=exc,
                )
                self._emit(
                    PipelineEvent(
                        event.event_type,
                        event.stage,
                        event.host_receive_index,
                        event.error,
                        event.traceback_text,
                        self.recorder_cleanup_pending,
                        exc,
                    )
                )
            try:
                self.publish_status(FLAG_DISCONNECTED | FLAG_STALE)
            except Exception as exc:
                self._emit(self._event(PipelineEventType.SINK_ERROR, "shared_disconnect", error=exc))
        self._emit(self._event(PipelineEventType.DISCONNECTED, reason))

    def _check_stale(self) -> None:
        with self._lock:
            reference = self._watchdog_baseline_ns
            if self._watchdog_last_published_ns is not None and (
                reference is None or self._watchdog_last_published_ns > reference
            ):
                reference = self._watchdog_last_published_ns
            if (
                not self._stream_expected
                or reference is None
                or self._stale_published
                or time.monotonic_ns() - reference
                < int(self.stale_after_seconds * 1_000_000_000)
            ):
                return
            try:
                self.publish_status(FLAG_STALE)
                self._stale_published = True
            except Exception as exc:
                error = exc
            else:
                error = None
        if error is None:
            self._emit(self._event(PipelineEventType.STALE, "watchdog"))
        else:
            self._emit(self._event(PipelineEventType.SINK_ERROR, "shared_stale", error=error))

    def _run(self) -> None:
        timeout = min(0.25, self.stale_after_seconds / 2)
        while True:
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                self._check_stale()
                continue
            try:
                if item is self._sentinel:
                    return
                self._process(item)
            except Exception as exc:
                self._emit(self._event(PipelineEventType.SINK_ERROR, "consumer", error=exc))
            finally:
                if isinstance(item, RawNotification):
                    with self._progress:
                        self._processed_accepted_count += 1
                        self._processed_host_receive_index = max(
                            self._processed_host_receive_index, item.host_receive_index
                        )
                        self._progress.notify_all()
                self._queue.task_done()

    def close(self, *, drain: bool = True, timeout: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._accepting = False
            self._stream_expected = False
            self._watchdog_generation = None
            self._watchdog_baseline_ns = None
            self._watchdog_last_published_ns = None
            self._stale_published = True
            thread = self._thread
        if thread is not None:
            if not drain:
                self._discard_pending()
            try:
                self._queue.put(self._sentinel, timeout=timeout)
            except queue.Full as exc:
                raise TimeoutError("could not enqueue acquisition shutdown") from exc
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("acquisition consumer did not stop")
        with self._fanout_lock:
            self.stop_recorder(complete=False, error="pipeline_closed", timeout=timeout)
            with self._lock:
                self._thread = None
                self._closed = True
