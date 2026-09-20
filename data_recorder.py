"""Transactional, non-overwriting session recording for EMG samples."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union
from uuid import uuid4
from recording_context import RecordingContext
from training_contract import validate_training_provenance
from subject_identity import validate_subject_id
from emg_protocol import (
    AcquisitionMetadata,
    DeviceKey,
    EmgFrame,
    HandSide,
    NotificationPacketProtocol,
    QualityFlags,
    SEQUENCE_QUALITY_FLAGS,
    SequenceTracker,
    validate_quality_flags,
)

PathLike = Union[str, Path]
METADATA_SCHEMA_ID = "emg.session.metadata"
METADATA_SCHEMA_MAJOR, METADATA_SCHEMA_MINOR = 1, 8
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z", re.ASCII)
_METADATA_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,95}\Z", re.ASCII)
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL", "CLOCK$",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class RecorderState(str, Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    FAULTED = "FAULTED"


class ResourceState(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CLOSE_FAILED = "CLOSE_FAILED"


@dataclass(frozen=True, slots=True)
class RecorderQualitySnapshot:
    """Immutable, point-in-time recorder counters for operator displays."""

    sequence_detection_available: bool
    duplicate_count: int
    out_of_order_count: int
    gap_count: int
    connection_generation: Optional[int]
    recorded_rows: int


class RecorderCloseError(RuntimeError):
    def __init__(self, failures: list[tuple[str, BaseException]]) -> None:
        self.failures = tuple(failures)
        detail = "; ".join(f"{stage}: {type(exc).__name__}: {exc}" for stage, exc in failures)
        super().__init__(f"recorder close incomplete: {detail}")

class SessionLock:
    def __init__(self,path:Path): self.path=path; self.file=None
    def acquire(self)->bool:
        try:
            self.file=self.path.open("a+b");self.file.seek(0)
            if self.file.read(1)==b"": self.file.write(b"0");self.file.flush()
            self.file.seek(0)
            if os.name=="nt":
                import msvcrt;msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl;fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            return True
        except OSError:
            if self.file is not None:self.file.close()
            self.file=None;return False
    def release(self):
        if self.file is None:return
        self.file.seek(0)
        if os.name=="nt":
            import msvcrt;msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl;fcntl.flock(self.file.fileno(),fcntl.LOCK_UN)
        self.file.close();self.file=None

def generate_subject_key() -> str:
    return "sub-" + secrets.token_hex(16)

def validate_subject_key(value: str) -> str:
    """Compatibility export for the persistence-free subject-id validator."""
    return validate_subject_id(value)


def _validated_metadata_version(value: object, field: str) -> str:
    if not isinstance(value, str) or not _METADATA_VERSION.fullmatch(value):
        raise ValueError(f"{field} must be a bounded safe ASCII version identifier")
    return value


def validate_session_id(value: str) -> str:
    """Validate and canonicalize the sole public session-id contract."""
    field = "session_id"
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} must be a safe ASCII identifier")
    # Win32 strips trailing spaces and dots from path components. Reject them
    # instead of allowing multiple user IDs to alias the same directory.
    if value.endswith((".", " ")):
        raise ValueError(f"{field} must not end with a dot or space")
    canonical = value.lower()
    if canonical in {".", ".."} or canonical.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        raise ValueError(f"{field} is reserved")
    return canonical


def _require_direct_child(parent_resolved: Path, child_resolved: Path, label: str) -> Path:
    """Validate a fully resolved child, including symlink/junction resolution."""
    if child_resolved.parent != parent_resolved:
        raise ValueError(f"{label} directory escapes its configured parent")
    return child_resolved


def _context_identifier(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 256 or any(ord(char) < 32 for char in cleaned):
        raise ValueError(f"{field} must be non-empty text without control characters")
    return cleaned


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        current = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return current.astimezone(timezone.utc).isoformat()
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("timestamp must be finite")
        return f"{numeric:.9f}"
    text = str(value).strip()
    if not text:
        raise ValueError("timestamp must not be empty")
    return text

def _validated_sample(value:Any,sample_format:str)->Any:
    if sample_format=="uint8":
        if not isinstance(value,int) or isinstance(value,bool) or not 0<=value<=255: raise ValueError("uint8 samples must be integers from 0 through 255")
    elif sample_format=="int16":
        if not isinstance(value,int) or isinstance(value,bool) or not -32768<=value<=32767: raise ValueError("int16 samples must be integers from -32768 through 32767")
    elif sample_format=="float32":
        if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(float(value)) or abs(float(value))>3.4028235e38: raise ValueError("float32 samples must be finite representable numbers")
    else:raise ValueError("recording requires a known sample_format")
    return value


class DataRecorder:
    LEGACY_BASE_COLUMNS = (
        "host_wall_timestamp_ns","host_monotonic_ns","host_receive_index","sample_index",
        "device_packet_sequence","device_sample_counter","device_time_ticks","sample_in_packet",
        "action_label","action_phase","sample_rate_hz","device_id","session_id","quality_flags",
        "sample_rate_source_kind","sample_rate_evidence_ref","sample_rate_confirmed",
    )
    V1_2_BASE_COLUMNS = LEGACY_BASE_COLUMNS[:4] + ("generation",) + LEGACY_BASE_COLUMNS[4:]
    V1_5_BASE_COLUMNS = V1_2_BASE_COLUMNS[:5] + ("connection_generation",) + V1_2_BASE_COLUMNS[5:]
    BASE_COLUMNS = V1_5_BASE_COLUMNS[:4] + ("session_sample_index",) + V1_5_BASE_COLUMNS[4:]

    def __init__(
        self, root_path: PathLike, *, subject_id: str, device_id: DeviceKey,
        acquisition: AcquisitionMetadata, channels: int,
        session_id: Optional[str] = None, flush_every: int = 100,
        metadata_extra: Optional[Mapping[str, Any]] = None,
        side: HandSide = HandSide.UNKNOWN,
        recording_context: RecordingContext,
    ) -> None:
        if not isinstance(channels, int) or isinstance(channels, bool) or not 1 <= channels <= 256:
            raise ValueError("channels must be an integer from 1 through 256")
        if not isinstance(acquisition,AcquisitionMetadata): raise ValueError("acquisition must be AcquisitionMetadata")
        if not isinstance(side, HandSide):
            raise ValueError("side must be HandSide")
        if not isinstance(flush_every, int) or isinstance(flush_every, bool) or flush_every <= 0:
            raise ValueError("flush_every must be a positive integer")
        self.root_path = Path(root_path).expanduser().resolve()
        self.subject_id = validate_subject_key(subject_id)
        if not isinstance(device_id,DeviceKey): raise ValueError("device_id must be DeviceKey")
        self.device_id = str(device_id)
        self.side = side
        self.sample_rate_hz = acquisition.sample_rate.value_hz
        self._sample_rate=acquisition.sample_rate
        self._acquisition_metadata=asdict(acquisition)
        self.channels = channels
        requested = validate_session_id(
            session_id or f"{_utc_now():%Y%m%dT%H%M%S.%fZ}-{uuid4().hex[:8]}"
        )
        self.flush_every = flush_every
        self._lock = threading.RLock()
        self._state = RecorderState.OPEN
        self._resource_state = ResourceState.OPEN
        self._row_count = 0
        self._last_persisted_row = 0
        self._created_at = _utc_now()
        self._closed_at: Optional[datetime] = None
        self._fault: Optional[dict[str, str]] = None
        self._failures: list[dict[str, str]] = []
        self._abort_reason: Optional[str] = None
        self._terminal_metadata_written = False
        self._last_host_receive_index = -1
        self._last_sample_index = -1
        self._next_session_sample_index = 0
        self._last_host_monotonic_ns = -1
        self._host_monotonic_tie_count = 0
        self._sequence_tracker = (
            None
            if acquisition.device_sequence_modulus is None
            else SequenceTracker(acquisition.device_sequence_modulus)
        )
        self._sequence_duplicate_count = 0
        self._sequence_out_of_order_count = 0
        self._sequence_gap_count = 0
        self._current_device_packet:Optional[int]=None
        self._current_generation:Optional[int]=None
        self._current_connection_generation:Optional[int]=None
        self._last_sample_in_packet=-1
        self._current_packet_wall=None;self._current_packet_monotonic=None;self._current_packet_duplicate=None
        self._terminal_intent=None
        self._session_boundary = {
            "start_host_receive_index": None,
            "end_host_receive_index": None,
            "received_count": 0,
            "eligible_count": 0,
            "written_count": 0,
            "queue_drop_total": 0,
            "queue_drop_session": 0,
            "tail_pending_count": 0,
            "tail_loss_count": 0,
            "host_monotonic_tie_count": 0,
            "incomplete_reason": None,
        }
        if not isinstance(recording_context, RecordingContext):
            raise ValueError("recording_context is required for a new recording")
        if recording_context.subject_id != self.subject_id:
            raise ValueError("recording_context subject_id does not match recorder subject_id")
        if self.side not in {HandSide.LEFT, HandSide.RIGHT}:
            raise ValueError("new recordings require HandSide.LEFT or HandSide.RIGHT")
        if recording_context.hand_side is not self.side:
            raise ValueError("recording_context hand_side does not match recorder side")
        self._recording_context = recording_context
        self._training_provenance_metadata = (
            recording_context.to_training_provenance_metadata()
        )
        self._expected_action_annotation = (
            recording_context.action_label, recording_context.action_phase
        )
        allowed_extra={"firmware_version","protocol_version"}
        try:
            raw_extra=dict(metadata_extra or {})
            if set(raw_extra)-allowed_extra: raise ValueError("metadata_extra contains non-whitelisted keys")
            for key, value in raw_extra.items():
                raw_extra[key] = _validated_metadata_version(value, key)
            raw_extra.update(recording_context.to_metadata_extra())
            self._extra_metadata = copy.deepcopy(raw_extra)
            json.dumps(self._extra_metadata, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metadata must be finite JSON data: {exc}") from exc

        self.root_path.mkdir(parents=True, exist_ok=True)
        root_resolved = self.root_path.resolve()
        subject_dir = root_resolved / self.subject_id
        subject_dir.mkdir(exist_ok=True)
        subject_resolved = _require_direct_child(
            root_resolved, subject_dir.resolve(), "subject"
        )
        self.session_dir = self._create_unique_session_dir(subject_resolved, requested)
        _require_direct_child(subject_resolved, self.session_dir.resolve(), "session")
        self.session_id = self.session_dir.name
        self._session_lock=SessionLock(self.session_dir/".session.lock")
        if not self._session_lock.acquire(): raise RuntimeError("session is already active")
        self._lock_released=False
        self.csv_path = self.session_dir / "samples.csv"
        self.metadata_path = self.session_dir / "metadata.json"
        self.columns = self.BASE_COLUMNS + tuple(f"channel_{i}" for i in range(1, channels + 1))
        self._csv_file = None
        try:
            self._csv_file = self.csv_path.open("x", newline="", encoding="utf-8-sig")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=self.columns)
            self._writer.writeheader()
            self._sync_csv()
            self._write_metadata("recording")
        except Exception as exc:
            self._state = RecorderState.FAULTED
            self._fault = self._exception_details("initialization", exc)
            cleanup_failures: list[tuple[str, BaseException]] = []
            if self._csv_file is not None:
                try:
                    self._close_csv()
                    self._resource_state = ResourceState.CLOSED
                except BaseException as cleanup_error:
                    self._resource_state = ResourceState.CLOSE_FAILED
                    cleanup_failures.append(("close_csv",cleanup_error))
            try:
                self._session_lock.release()
                self._lock_released=True
            except BaseException as cleanup_error:
                cleanup_failures.append(("release_session_lock",cleanup_error))
            setattr(exc,"cleanup_failures",tuple(cleanup_failures))
            setattr(exc,"initialization_resource_state",self._resource_state.value)
            setattr(exc,"initialization_lock_released",self._lock_released)
            if cleanup_failures and hasattr(exc,"add_note"):
                exc.add_note("recorder initialization cleanup incomplete: "+"; ".join(f"{stage}: {type(error).__name__}: {error}" for stage,error in cleanup_failures))
            raise

    @property
    def state(self) -> RecorderState:
        return self._state

    @property
    def resource_state(self) -> ResourceState:
        return self._resource_state

    def quality_snapshot(self) -> RecorderQualitySnapshot:
        """Return an immutable snapshot without exposing mutable recorder state.

        Device-sequence counters are meaningful only when the acquisition
        contract supplies a trusted sequence modulus.  Payload equality is
        deliberately not used as a retransmission heuristic.
        """
        with self._lock:
            return RecorderQualitySnapshot(
                sequence_detection_available=self._sequence_tracker is not None,
                duplicate_count=self._sequence_duplicate_count,
                out_of_order_count=self._sequence_out_of_order_count,
                gap_count=self._sequence_gap_count,
                connection_generation=self._current_connection_generation,
                recorded_rows=self._row_count,
            )

    @staticmethod
    def _create_unique_session_dir(subject_dir: Path, requested: str) -> Path:
        for suffix in range(10000):
            candidate = subject_dir / (requested if suffix == 0 else f"{requested}-{suffix:03d}")
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                continue
        raise FileExistsError(f"too many sessions named {requested!r}")

    @staticmethod
    def _exception_details(stage: str, exc: BaseException) -> dict[str, str]:
        return {"stage": stage, "type": type(exc).__name__, "message": str(exc)}

    def _metadata(self, status: str) -> dict[str, Any]:
        metadata_state = self._state
        if status in {"complete", "aborted"} and self._state is RecorderState.CLOSING:
            metadata_state = RecorderState.CLOSED
        result: dict[str, Any] = {
            "schema_id":METADATA_SCHEMA_ID,"schema_major":METADATA_SCHEMA_MAJOR,"schema_minor":METADATA_SCHEMA_MINOR,
            "status": status, "recorder_state": metadata_state.value,
            "resource_state": self._resource_state.value, "session_id": self.session_id,
            "subject_id": self.subject_id, "device_id": self.device_id,
            "hand_side": self.side.value,
            "writer_generation_semantics": "producer_instance_generation",
            "connection_generation_semantics": "ble_connection_lifecycle_generation",
            "sample_rate_hz":self.sample_rate_hz,"sample_rate_semantics":"sample_time_base_only","acquisition":self._acquisition_metadata,"channels": self.channels,
            "created_at": self._created_at.isoformat(),
            "closed_at": self._closed_at.isoformat() if self._closed_at else None,
            "row_count": self._row_count, "last_persisted_row": self._last_persisted_row,
            "csv_file": self.csv_path.name, "extra": self._extra_metadata,
            "session_boundary": copy.deepcopy(self._session_boundary),
        }
        if self._fault is not None:
            result["fault"] = self._fault
        if self._failures:
            result["failures"] = list(self._failures)
        if self._abort_reason is not None:
            result["abort_reason"] = self._abort_reason
        if self._training_provenance_metadata is not None:
            result["training_provenance"] = copy.deepcopy(
                self._training_provenance_metadata
            )
        return result

    def _replace_metadata(self, temporary: Path) -> None:
        temporary.replace(self.metadata_path)

    def _write_metadata(self, status: str) -> None:
        temporary = self.metadata_path.with_suffix(".json.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(self._metadata(status), stream, ensure_ascii=False, indent=2,
                          sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            self._replace_metadata(temporary)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _sync_csv(self) -> None:
        self._csv_file.flush()
        os.fsync(self._csv_file.fileno())
        self._last_persisted_row = self._row_count
        if hasattr(self,"metadata_path"):
            self._write_metadata("recording" if self._state is RecorderState.OPEN else "faulted" if self._state is RecorderState.FAULTED else "recording")

    def _close_csv(self) -> None:
        self._csv_file.close()

    def _latch_fault(self, stage: str, exc: BaseException) -> None:
        self._state = RecorderState.FAULTED
        details = self._exception_details(stage, exc)
        self._failures.append(details)
        if self._fault is None:
            self._fault = details
        try:
            self._write_metadata("faulted")
        except Exception as secondary:
            self._failures.append(self._exception_details("fault_metadata", secondary))
            if hasattr(exc,"add_note"):
                exc.add_note(f"fault metadata persistence also failed: {secondary}")

    @staticmethod
    def _quality_flags(value: Union[str, int, Iterable[str], None]) -> str:
        if value is None:
            return ""
        if isinstance(value, (str, int)):
            return str(value)
        return "|".join(str(item) for item in value)

    @staticmethod
    def _boundary_integer(value: Any, field: str, *, optional: bool = False) -> Optional[int]:
        if optional and value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        return value

    def configure_session_boundary(
        self, *, start_host_receive_index: int, start_drop_total: int
    ) -> None:
        with self._lock:
            if self._state is not RecorderState.OPEN or self._row_count:
                raise RuntimeError("session boundary must be configured before recording")
            self._session_boundary["start_host_receive_index"] = self._boundary_integer(
                start_host_receive_index, "start_host_receive_index"
            )
            self._session_boundary["queue_drop_total"] = self._boundary_integer(
                start_drop_total, "start_drop_total"
            )
            self._write_metadata("recording")

    def update_session_boundary(
        self,
        *,
        end_host_receive_index: int,
        received_count: int,
        eligible_count: int,
        written_count: int,
        queue_drop_total: int,
        queue_drop_session: int,
        tail_pending_count: int,
        tail_loss_count: int,
        incomplete_reason: Optional[str],
    ) -> None:
        with self._lock:
            if self._state not in {RecorderState.OPEN, RecorderState.FAULTED}:
                raise RuntimeError("session boundary cannot be updated after close")
            values = {
                "end_host_receive_index": end_host_receive_index,
                "received_count": received_count,
                "eligible_count": eligible_count,
                "written_count": written_count,
                "queue_drop_total": queue_drop_total,
                "queue_drop_session": queue_drop_session,
                "tail_pending_count": tail_pending_count,
                "tail_loss_count": tail_loss_count,
            }
            for field, value in values.items():
                self._session_boundary[field] = self._boundary_integer(value, field)
            if incomplete_reason is not None and (
                not isinstance(incomplete_reason, str) or not incomplete_reason.strip()
            ):
                raise ValueError("incomplete_reason must be non-empty text or None")
            self._session_boundary["incomplete_reason"] = incomplete_reason
            self._write_metadata("recording" if self._state is RecorderState.OPEN else "faulted")

    def record(self, frame:EmgFrame, *, session_sample_index: Optional[int] = None) -> None:
        with self._lock:
            if self._state is not RecorderState.OPEN:
                raise RuntimeError(f"recorder is not open: {self._state.value}")
            if not isinstance(frame,EmgFrame): raise ValueError("record requires EmgFrame")
            if len(frame.channel_values) != self.channels:
                raise ValueError(f"expected {self.channels} channel values, got {len(frame.channel_values)}")
            values = []
            sample_format=self._acquisition_metadata["signal_chain"]["sample_format"]
            for value in frame.channel_values:
                values.append(_validated_sample(value,sample_format))
            if self._expected_action_annotation is not None and (
                frame.action_label,
                frame.action_phase,
            ) != self._expected_action_annotation:
                raise ValueError("frame action annotation does not match metadata action context")
            flags=validate_quality_flags(frame.quality_flags)
            required=QualityFlags.VALID|QualityFlags.HOST_WALL_TIME_VALID|QualityFlags.HOST_MONOTONIC_VALID|QualityFlags.HOST_RECEIVE_INDEX_VALID
            if flags&required!=required or flags&(QualityFlags.DISCONNECTED|QualityFlags.STALE|QualityFlags.CRC_ERROR): raise ValueError("frame validity flags are inconsistent with a recordable sample")
            for field,flag in ((frame.device_packet_sequence,QualityFlags.DEVICE_PACKET_SEQUENCE_VALID),(frame.device_sample_counter,QualityFlags.DEVICE_SAMPLE_COUNTER_VALID),(frame.device_time_ticks,QualityFlags.DEVICE_TIME_VALID)):
                if bool(field is not None)!=bool(flags&flag): raise ValueError("optional field validity flag mismatch")
            if frame.device_packet_sequence is not None and self._sequence_tracker is None:
                raise ValueError("device packet sequence requires a configured modulus and provenance")
            if flags&SEQUENCE_QUALITY_FLAGS: raise ValueError("sequence quality flags are computed by SequenceTracker")
            if frame.device_time_ticks is not None:
                tick=self._acquisition_metadata["device_tick_rate"]
                if not tick["confirmed"] or self._acquisition_metadata["device_tick_modulus"] is None: raise ValueError("device ticks require confirmed timebase and modulus provenance")
                if frame.device_time_ticks>=self._acquisition_metadata["device_tick_modulus"]: raise ValueError("device ticks exceed configured modulus")
            same_packet=frame.host_receive_index==self._last_host_receive_index
            if frame.host_receive_index < self._last_host_receive_index:
                raise ValueError("host_receive_index moved backwards")
            if frame.sample_index <= self._last_sample_index:
                raise ValueError("sample_index is not strictly increasing")
            if session_sample_index is None:
                session_sample_index = self._next_session_sample_index
            if (
                not isinstance(session_sample_index, int)
                or isinstance(session_sample_index, bool)
                or session_sample_index != self._next_session_sample_index
            ):
                raise ValueError("session_sample_index is not continuous")
            if frame.host_monotonic_ns < self._last_host_monotonic_ns:
                raise ValueError("host_monotonic_ns moved backwards")
            monotonic_tie = (
                not same_packet
                and self._last_host_receive_index >= 0
                and frame.host_monotonic_ns == self._last_host_monotonic_ns
            )
            if self._current_connection_generation is not None and frame.connection_generation < self._current_connection_generation:
                raise ValueError("connection_generation moved backwards")
            if not same_packet and frame.sample_in_packet != 0:
                raise ValueError("a packet must begin with sample_in_packet zero")
            if same_packet and (frame.generation!=self._current_generation or frame.connection_generation!=self._current_connection_generation or frame.device_packet_sequence!=self._current_device_packet or frame.sample_in_packet!=self._last_sample_in_packet+1): raise ValueError("same host_receive_index requires same generations, packet and consecutive sample_in_packet")
            if same_packet and (frame.host_wall_timestamp_ns!=self._current_packet_wall or frame.host_monotonic_ns!=self._current_packet_monotonic):raise ValueError("all samples in one packet must share host timestamps")
            sequence_checkpoint = self._sequence_tracker.checkpoint() if self._sequence_tracker else None
            sequence_observation = (
                None
                if same_packet or self._sequence_tracker is None
                else self._sequence_tracker.observe(
                    frame.device_packet_sequence, frame.connection_generation
                )
            )
            sequence_flags = (
                self._current_packet_duplicate
                if same_packet
                else sequence_observation.flags
                if sequence_observation is not None
                else QualityFlags.NONE
            )
            quality_flags=flags|sequence_flags
            row: dict[str, Any] = {
                "host_wall_timestamp_ns":frame.host_wall_timestamp_ns,"host_monotonic_ns":frame.host_monotonic_ns,"host_receive_index":frame.host_receive_index,"sample_index":frame.sample_index,
                "session_sample_index": session_sample_index,
                "generation": frame.generation,
                "connection_generation": frame.connection_generation,
                "device_packet_sequence":"" if frame.device_packet_sequence is None else frame.device_packet_sequence,"device_sample_counter":"" if frame.device_sample_counter is None else frame.device_sample_counter,"device_time_ticks":"" if frame.device_time_ticks is None else frame.device_time_ticks,
                "sample_in_packet": frame.sample_in_packet, "action_label": frame.action_label,
                "action_phase": frame.action_phase,
                "sample_rate_hz":self.sample_rate_hz if self._sample_rate.confirmed else "",
                "sample_rate_source_kind":self._sample_rate.source_kind,"sample_rate_evidence_ref":self._sample_rate.evidence_ref,"sample_rate_confirmed":self._sample_rate.confirmed,
                "device_id": self.device_id, "session_id": self.session_id,
                "quality_flags": int(quality_flags),
            }
            row.update({f"channel_{i}": value for i, value in enumerate(values, 1)})
            try:
                self._writer.writerow(row)
                self._row_count += 1
                self._next_session_sample_index += 1
                if monotonic_tie:
                    self._host_monotonic_tie_count += 1
                    self._session_boundary["host_monotonic_tie_count"] = (
                        self._host_monotonic_tie_count
                    )
                self._last_host_receive_index=frame.host_receive_index; self._last_sample_index=frame.sample_index; self._last_host_monotonic_ns=frame.host_monotonic_ns
                self._current_device_packet=frame.device_packet_sequence;self._current_generation=frame.generation;self._current_connection_generation=frame.connection_generation;self._last_sample_in_packet=frame.sample_in_packet
                self._current_packet_wall=frame.host_wall_timestamp_ns;self._current_packet_monotonic=frame.host_monotonic_ns;self._current_packet_duplicate=sequence_flags
                if sequence_observation is not None:
                    if sequence_observation.flags & QualityFlags.DUPLICATE_PACKET:
                        self._sequence_duplicate_count += 1
                    if sequence_observation.flags & QualityFlags.OUT_OF_ORDER_PACKET:
                        self._sequence_out_of_order_count += 1
                    self._sequence_gap_count += sequence_observation.gap_count
            except Exception as exc:
                if self._sequence_tracker is not None and sequence_checkpoint is not None:
                    self._sequence_tracker.restore(sequence_checkpoint)
                self._latch_fault("writerow", exc)
                raise
            if self._row_count % self.flush_every == 0:
                try:
                    self._sync_csv()
                except Exception as exc:
                    self._latch_fault("flush", exc)
                    raise

    def close(self, *, complete: bool = True, error: Optional[str] = None) -> None:
        if complete and error is not None:
            raise ValueError("a complete session cannot have an error")
        failures: list[tuple[str, BaseException]] = []
        with self._lock:
            requested=("complete" if complete else "aborted",error)
            if self._terminal_intent is None:self._terminal_intent=requested
            elif requested!=("complete",None) and requested!=self._terminal_intent:raise ValueError("conflicting terminal intent")
            complete=self._terminal_intent[0]=="complete";error=self._terminal_intent[1]
            if self._resource_state is ResourceState.CLOSED and self._terminal_metadata_written and self._lock_released:
                return
            if self._closed_at is None:
                self._closed_at = _utc_now()
            was_faulted = self._state is RecorderState.FAULTED
            if not was_faulted:
                self._state = RecorderState.CLOSING
                if not complete:
                    self._abort_reason = error or "recording_aborted"

            if self._resource_state is not ResourceState.CLOSED:
                if not was_faulted:
                    try:
                        self._sync_csv()
                    except Exception as exc:
                        self._latch_fault("close_flush", exc)
                        failures.append(("close_flush", exc))
                try:
                    self._close_csv()
                    self._resource_state = ResourceState.CLOSED
                except Exception as exc:
                    self._resource_state = ResourceState.CLOSE_FAILED
                    self._latch_fault("file_close", exc)
                    failures.append(("file_close", exc))

            terminal_status = "faulted" if self._state is RecorderState.FAULTED else (
                "complete" if complete else "aborted"
            )
            if self._resource_state is ResourceState.CLOSED:
                try:
                    self._write_metadata(terminal_status)
                    self._terminal_metadata_written = True
                except Exception as exc:
                    self._latch_fault("metadata_commit", exc)
                    failures.append(("metadata_commit", exc))

            if not failures and self._state is not RecorderState.FAULTED:
                self._state = RecorderState.CLOSED
            if self._resource_state is ResourceState.CLOSED and self._terminal_metadata_written:
                try:
                    self._session_lock.release();self._lock_released=True
                except Exception as exc:
                    failures.append(("lock_release",exc))
            if failures:
                raise RecorderCloseError(failures)

    def __enter__(self) -> "DataRecorder":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close(complete=True)
            return
        try:
            self.close(complete=False, error=f"{exc_type.__name__}: {exc_value}")
        except Exception as close_error:
            if hasattr(exc_value, "add_note"):
                exc_value.add_note(f"DataRecorder cleanup also failed: {close_error}")


@dataclass(frozen=True)
class RecoveryRecord:
    session_dir: Path
    csv_rows: int
    checkpoint_rows: int
    status: str
@dataclass(frozen=True)
class RecoveryIssue:
    session_dir:Path; code:str; detail:str
@dataclass(frozen=True)
class RecoveryReport:
    recovered:list[RecoveryRecord]; issues:list[RecoveryIssue]

def _csv_integer(row: dict[str, str], name: str, *, optional: bool = False) -> Optional[int]:
    raw = row[name]
    if optional and raw == "":
        return None
    if not raw or not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"invalid {name}")
    value = int(raw)
    if value > (1 << 64) - 1:
        raise ValueError(f"invalid {name}")
    return value


def _legacy_device_identity(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("invalid legacy device identity")
    return value


def _decode_recovery_acquisition(
    value: Any, schema_minor: int
) -> tuple[AcquisitionMetadata, list[dict[str, Any]]]:
    if not isinstance(value, Mapping):
        raise ValueError("acquisition must be an object")
    decoded = copy.deepcopy(dict(value))
    migrations: list[dict[str, Any]] = []
    current_fields = {
        "sample_rate", "adc_rate", "device_output_rate", "host_observed_rate",
        "device_tick_rate", "device_tick_modulus", "device_tick_modulus_source",
        "device_sequence_modulus", "device_sequence_modulus_source", "signal_chain",
        "notification_packet_protocol",
    }
    current_signal_fields = {
        "sample_format", "unit", "adc_bit_width", "scaling", "offset", "filter",
        "rectification", "normalization", "quantization",
    }
    signal_chain = decoded.get("signal_chain")
    if not isinstance(signal_chain, Mapping):
        raise ValueError("signal_chain must be an object")
    signal_chain = copy.deepcopy(dict(signal_chain))
    decoded["signal_chain"] = signal_chain
    historical_fields = current_fields - {"notification_packet_protocol"}
    if schema_minor >= 5:
        if set(decoded) != current_fields or set(signal_chain) != current_signal_fields:
            raise ValueError("schema 1.5 acquisition fields are invalid")
    elif schema_minor >= 3:
        if set(decoded) != historical_fields or set(signal_chain) != current_signal_fields:
            raise ValueError("historical acquisition fields are invalid")
    else:
        defaults = {
            "device_sequence_modulus": None,
            "device_sequence_modulus_source": "unknown",
        }
        for field_name, default in defaults.items():
            if field_name not in decoded:
                decoded[field_name] = default
                migrations.append({
                    "field": f"acquisition.{field_name}",
                    "action": "filled_absent_historical_field",
                    "value": default,
                })
        if (
            decoded.get("device_sequence_modulus") == 1 << 64
            and decoded.get("device_sequence_modulus_source") == "wire_uint64_contract"
        ):
            decoded["device_sequence_modulus"] = None
            decoded["device_sequence_modulus_source"] = "unknown"
            migrations.extend((
                {
                    "field": "acquisition.device_sequence_modulus",
                    "action": "removed_historical_container_width_assumption",
                    "previous_value": 1 << 64,
                    "value": None,
                },
                {
                    "field": "acquisition.device_sequence_modulus_source",
                    "action": "removed_historical_container_width_assumption",
                    "previous_value": "wire_uint64_contract",
                    "value": "unknown",
                },
            ))
        if "quantization" not in signal_chain:
            signal_chain["quantization"] = "unknown"
            migrations.append({
                "field": "acquisition.signal_chain.quantization",
                "action": "filled_absent_historical_field",
                "value": "unknown",
            })
    if schema_minor < 5:
        protocol = asdict(NotificationPacketProtocol())
        decoded["notification_packet_protocol"] = protocol
        migrations.append({
            "field": "acquisition.notification_packet_protocol",
            "action": "filled_absent_historical_field",
            "value": protocol,
        })
    return AcquisitionMetadata.from_mapping(decoded), migrations


class _RecoveryStreamValidator:
    """Validate each CSV row while retaining only packet/sequence state."""

    def __init__(
        self,
        *,
        channels: int,
        schema_minor: int,
        acquisition: AcquisitionMetadata,
        device_id: str,
        session_id: str,
        expected_annotation: Optional[tuple[str, str]] = None,
    ) -> None:
        self.channels = channels
        self.schema_minor = schema_minor
        self.acquisition = acquisition
        self.legacy_identity = schema_minor == 0
        self.device_id = (
            _legacy_device_identity(device_id)
            if self.legacy_identity
            else str(DeviceKey(device_id))
        )
        self.session_id = validate_session_id(session_id)
        self.expected_annotation = expected_annotation
        self.tracker = (
            None
            if acquisition.device_sequence_modulus is None
            else SequenceTracker(acquisition.device_sequence_modulus)
        )
        self.previous: Optional[EmgFrame] = None
        self.packet_sequence_flags = QualityFlags.NONE
        self.next_session_sample_index = 0
        self.host_monotonic_tie_count = 0

    @property
    def expected_columns(self) -> list[str]:
        if self.schema_minor >= 6:
            base = DataRecorder.BASE_COLUMNS
        elif self.schema_minor >= 4:
            base = DataRecorder.V1_5_BASE_COLUMNS
        elif self.schema_minor >= 2:
            base = DataRecorder.V1_2_BASE_COLUMNS
        else:
            base = DataRecorder.LEGACY_BASE_COLUMNS
        return list(base) + [f"channel_{index}" for index in range(1, self.channels + 1)]

    def _validate_rate(self, row: dict[str, str]) -> None:
        rate = self.acquisition.sample_rate
        if row["sample_rate_source_kind"] != rate.source_kind:
            raise ValueError("sample rate source differs from metadata")
        if row["sample_rate_evidence_ref"] != rate.evidence_ref:
            raise ValueError("sample rate evidence differs from metadata")
        if row["sample_rate_confirmed"] != str(rate.confirmed):
            raise ValueError("sample rate confirmation differs from metadata")
        if not rate.confirmed:
            if row["sample_rate_hz"] != "":
                raise ValueError("unconfirmed sample rate must not be authoritative in CSV")
            return
        if row["sample_rate_hz"] == "":
            raise ValueError("confirmed sample rate is missing from CSV")
        try:
            csv_rate = float(row["sample_rate_hz"])
        except ValueError as exc:
            raise ValueError("invalid sample_rate_hz") from exc
        if not math.isfinite(csv_rate) or csv_rate != float(rate.value_hz):
            raise ValueError("sample rate differs from metadata")

    def validate(self, row: dict[str, str]) -> None:
        if any(value is None for value in row.values()):
            raise ValueError("partial CSV row")
        if row["device_id"] != self.device_id or row["session_id"] != self.session_id:
            raise ValueError("CSV identity differs from metadata")
        if self.expected_annotation is not None and (
            row["action_label"], row["action_phase"]
        ) != self.expected_annotation:
            raise ValueError("CSV action annotation differs from metadata")
        self._validate_rate(row)
        if self.schema_minor >= 6:
            session_sample_index = _csv_integer(row, "session_sample_index")
            if session_sample_index != self.next_session_sample_index:
                raise ValueError("session_sample_index is not continuous")
        sample_format = self.acquisition.signal_chain.sample_format
        channel_values = []
        for index in range(1, self.channels + 1):
            raw = row[f"channel_{index}"]
            try:
                value = int(raw) if sample_format in {"uint8", "int16"} else float(raw)
            except ValueError as exc:
                raise ValueError(f"invalid channel_{index}") from exc
            channel_values.append(_validated_sample(value, sample_format))
        try:
            flags = validate_quality_flags(QualityFlags(int(row["quality_flags"])))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid quality_flags") from exc
        generation = _csv_integer(row, "generation") if self.schema_minor >= 2 else 0
        connection_generation = (
            _csv_integer(row, "connection_generation") if self.schema_minor >= 4 else 0
        )
        frame = EmgFrame(
            tuple(channel_values),
            _csv_integer(row, "host_wall_timestamp_ns"),
            _csv_integer(row, "host_monotonic_ns"),
            _csv_integer(row, "host_receive_index"),
            _csv_integer(row, "sample_index"),
            generation=generation,
            connection_generation=connection_generation,
            device_packet_sequence=_csv_integer(row, "device_packet_sequence", optional=True),
            device_sample_counter=_csv_integer(row, "device_sample_counter", optional=True),
            device_time_ticks=_csv_integer(row, "device_time_ticks", optional=True),
            sample_in_packet=_csv_integer(row, "sample_in_packet"),
            action_label=row["action_label"],
            action_phase=row["action_phase"],
            quality_flags=flags,
        )
        required = (
            QualityFlags.VALID
            | QualityFlags.HOST_WALL_TIME_VALID
            | QualityFlags.HOST_MONOTONIC_VALID
            | QualityFlags.HOST_RECEIVE_INDEX_VALID
        )
        invalid = QualityFlags.DISCONNECTED | QualityFlags.STALE | QualityFlags.CRC_ERROR
        if flags & required != required or flags & invalid:
            raise ValueError("recorded row has inconsistent validity flags")
        for value, flag in (
            (frame.device_packet_sequence, QualityFlags.DEVICE_PACKET_SEQUENCE_VALID),
            (frame.device_sample_counter, QualityFlags.DEVICE_SAMPLE_COUNTER_VALID),
            (frame.device_time_ticks, QualityFlags.DEVICE_TIME_VALID),
        ):
            if bool(value is not None) != bool(flags & flag):
                raise ValueError("optional field validity flag mismatch")
        if frame.device_packet_sequence is not None and self.tracker is None:
            raise ValueError("device packet sequence requires a configured modulus and provenance")
        if frame.device_time_ticks is not None:
            tick = self.acquisition.device_tick_rate
            modulus = self.acquisition.device_tick_modulus
            if not tick.confirmed or modulus is None or frame.device_time_ticks >= modulus:
                raise ValueError("device ticks lack valid timebase metadata")

        previous = self.previous
        same_packet = previous is not None and frame.host_receive_index == previous.host_receive_index
        if previous is not None:
            if frame.host_receive_index < previous.host_receive_index:
                raise ValueError("host_receive_index moved backwards")
            if frame.sample_index <= previous.sample_index:
                raise ValueError("sample_index is not strictly increasing")
            if frame.host_monotonic_ns < previous.host_monotonic_ns:
                raise ValueError("host monotonic time moved backwards")
            if frame.connection_generation < previous.connection_generation:
                raise ValueError("connection_generation moved backwards")
        if same_packet:
            if (
                frame.generation != previous.generation
                or frame.connection_generation != previous.connection_generation
                or frame.device_packet_sequence != previous.device_packet_sequence
                or frame.host_wall_timestamp_ns != previous.host_wall_timestamp_ns
                or frame.host_monotonic_ns != previous.host_monotonic_ns
                or frame.sample_in_packet != previous.sample_in_packet + 1
            ):
                raise ValueError("same packet rows have inconsistent packet identity")
            expected_sequence_flags = self.packet_sequence_flags
        else:
            if previous is not None and frame.host_monotonic_ns == previous.host_monotonic_ns:
                self.host_monotonic_tie_count += 1
            if frame.sample_in_packet != 0:
                raise ValueError("packet does not begin at sample zero")
            expected_sequence_flags = (
                self.tracker.observe(
                    frame.device_packet_sequence, frame.connection_generation
                ).flags
                if self.tracker else QualityFlags.NONE
            )
            self.packet_sequence_flags = expected_sequence_flags
        if flags & SEQUENCE_QUALITY_FLAGS != expected_sequence_flags:
            raise ValueError("packet sequence flags do not match SequenceTracker")
        self.previous = frame
        if self.schema_minor >= 6:
            self.next_session_sample_index += 1

def _replace_recovery_metadata(temporary:Path,metadata_path:Path)->None:
    temporary.replace(metadata_path)

def _validated_session_boundary(value: Any, schema_minor: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("session_boundary must be an object")
    expected = {
        "start_host_receive_index", "end_host_receive_index", "received_count",
        "eligible_count", "written_count", "queue_drop_total", "queue_drop_session",
        "tail_pending_count", "tail_loss_count", "incomplete_reason",
    }
    if schema_minor >= 7:
        expected.add("host_monotonic_tie_count")
    if set(value) != expected:
        raise ValueError("session_boundary fields are invalid")
    result = copy.deepcopy(dict(value))
    for field in expected - {"start_host_receive_index", "end_host_receive_index", "incomplete_reason"}:
        DataRecorder._boundary_integer(result[field], field)
    DataRecorder._boundary_integer(
        result["start_host_receive_index"], "start_host_receive_index", optional=True
    )
    DataRecorder._boundary_integer(
        result["end_host_receive_index"], "end_host_receive_index", optional=True
    )
    reason = result["incomplete_reason"]
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("invalid session boundary incomplete_reason")
    start = result["start_host_receive_index"]
    end = result["end_host_receive_index"]
    if start is not None and end is not None and end < start:
        raise ValueError("session boundary end precedes start")
    if result["written_count"] > result["eligible_count"]:
        raise ValueError("session boundary written_count exceeds eligible_count")
    if result["queue_drop_session"] > result["queue_drop_total"]:
        raise ValueError("session boundary queue drop delta exceeds total")
    return result

def recover_incomplete_sessions(root_path: PathLike) -> RecoveryReport:
    """Recover from CSV facts: every complete valid row becomes the terminal count."""
    root=Path(root_path).expanduser().resolve(); recovered=[];issues=[]
    if not root.exists(): return RecoveryReport(recovered,issues)
    for metadata_path in _iter_recovery_metadata(root,issues):
        lock=SessionLock(metadata_path.parent/".session.lock")
        if not lock.acquire(): issues.append(RecoveryIssue(metadata_path.parent,"active","session lock held"));continue
        try:
            payload=json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(payload,dict):raise ValueError("metadata root must be an object")
            for key in ("schema_major","schema_minor"):
                value=payload.get(key)
                if not isinstance(value,int) or isinstance(value,bool) or value<0:raise ValueError(f"invalid {key}")
            if payload.get("schema_id")!=METADATA_SCHEMA_ID or payload.get("schema_major")!=METADATA_SCHEMA_MAJOR or payload.get("schema_minor",0)>METADATA_SCHEMA_MINOR: raise ValueError("unsupported metadata schema")
            if payload.get("schema_id")!=METADATA_SCHEMA_ID or payload.get("status")!="recording": continue
            csv_name=payload["csv_file"]
            if csv_name!="samples.csv" or ":" in csv_name: raise ValueError("csv_file must be exactly samples.csv")
            channels=payload.get("channels");checkpoint=payload.get("last_persisted_row")
            if not isinstance(channels,int) or isinstance(channels,bool) or not 1<=channels<=256: raise ValueError("invalid channels")
            if not isinstance(checkpoint,int) or isinstance(checkpoint,bool) or checkpoint<0: raise ValueError("invalid checkpoint")
            relative_parts = metadata_path.relative_to(root).parts
            if len(relative_parts) != 3:
                raise ValueError("metadata is not in a subject/session directory")
            subject_id, actual_session_id, metadata_name = relative_parts
            if metadata_name != "metadata.json":
                raise ValueError("invalid metadata filename")
            if validate_subject_key(payload.get("subject_id")) != subject_id:
                raise ValueError("metadata subject identity differs from its path")
            if validate_session_id(payload.get("session_id")) != actual_session_id:
                raise ValueError("metadata session identity differs from its path")
            legacy_identity = payload["schema_minor"] == 0
            device_id = (
                _legacy_device_identity(payload.get("device_id"))
                if legacy_identity
                else str(DeviceKey(payload.get("device_id")))
            )
            session_migrations: list[dict[str, Any]] = []
            expected_annotation: Optional[tuple[str, str]] = None
            if payload["schema_minor"] >= 4:
                try:
                    recovered_side = HandSide(payload.get("hand_side"))
                except (TypeError, ValueError) as exc:
                    raise ValueError("invalid hand_side") from exc
                if payload.get("writer_generation_semantics") != "producer_instance_generation":
                    raise ValueError("invalid writer generation semantics")
                if payload.get("connection_generation_semantics") != "ble_connection_lifecycle_generation":
                    raise ValueError("invalid connection generation semantics")
            else:
                payload["hand_side"] = HandSide.UNKNOWN.value
                payload["writer_generation_semantics"] = "legacy_generation_semantics_unspecified"
                payload["connection_generation_semantics"] = "unknown_assumed_zero_for_validation"
                session_migrations.extend((
                    {
                        "field": "hand_side",
                        "action": "filled_unknown_historical_field",
                        "value": HandSide.UNKNOWN.value,
                    },
                    {
                        "field": "connection_generation",
                        "action": "validated_legacy_rows_as_single_unknown_connection",
                        "value": 0,
                    },
                ))
            if payload["schema_minor"] >= 8:
                extra = payload.get("extra")
                if not isinstance(extra, Mapping):
                    raise ValueError("schema 1.8 metadata.extra must be an object")
                required_context = {"action_label", "action_phase", "experiment_id"}
                optional_versions = {"firmware_version", "protocol_version"}
                if not required_context.issubset(extra) or set(extra) - required_context - optional_versions:
                    raise ValueError("schema 1.8 metadata.extra action context is incomplete or invalid")
                for field in optional_versions.intersection(extra):
                    _validated_metadata_version(extra[field], field)
                provenance = None
                if "training_provenance" in payload:
                    provenance = validate_training_provenance(payload["training_provenance"])
                recovered_context = RecordingContext(
                    subject_id=subject_id,
                    action_label=extra["action_label"],
                    action_phase=extra["action_phase"],
                    experiment_id=extra["experiment_id"],
                    hand_side=recovered_side,
                    training_provenance=provenance,
                )
                expected_annotation = (
                    recovered_context.action_label, recovered_context.action_phase
                )
            acquisition, acquisition_migrations = _decode_recovery_acquisition(
                payload.get("acquisition"), payload["schema_minor"]
            )
            if acquisition.signal_chain.sample_format == "unknown":
                raise ValueError("recovery requires an explicit sample_format")
            top_rate = payload.get("sample_rate_hz")
            if isinstance(top_rate, bool) or top_rate != acquisition.sample_rate.value_hz:
                raise ValueError("metadata sample rate differs from acquisition descriptor")
            if payload.get("sample_rate_semantics") != "sample_time_base_only":
                raise ValueError("invalid sample rate semantics")
            validator = _RecoveryStreamValidator(
                channels=channels,
                schema_minor=payload["schema_minor"],
                acquisition=acquisition,
                device_id=device_id,
                session_id=actual_session_id,
                expected_annotation=expected_annotation,
            )
            csv_path=(metadata_path.parent/csv_name).resolve()
            if csv_path.parent!=metadata_path.parent.resolve(): raise ValueError("csv escapes session")
            with csv_path.open("r",encoding="utf-8-sig",newline="") as stream:
                reader=csv.DictReader(stream)
                if reader.fieldnames != validator.expected_columns:
                    raise ValueError("CSV schema mismatch")
                csv_rows=0
                for row in reader:
                    validator.validate(row)
                    csv_rows+=1
            if payload["schema_minor"] >= 6:
                boundary = _validated_session_boundary(
                    payload.get("session_boundary"), payload["schema_minor"]
                )
                if payload["schema_minor"] >= 7:
                    persisted_ties = boundary["host_monotonic_tie_count"]
                    if persisted_ties > validator.host_monotonic_tie_count or (
                        csv_rows == checkpoint
                        and persisted_ties != validator.host_monotonic_tie_count
                    ):
                        raise ValueError("host monotonic tie count differs from CSV")
            else:
                boundary = {
                    "start_host_receive_index": None,
                    "end_host_receive_index": None,
                    "received_count": csv_rows,
                    "eligible_count": csv_rows,
                    "written_count": csv_rows,
                    "queue_drop_total": 0,
                    "queue_drop_session": 0,
                    "tail_pending_count": 0,
                    "tail_loss_count": 0,
                    "incomplete_reason": "process_interrupted",
                }
            if payload["schema_minor"] <= 6:
                session_migrations.append({
                    "field": "session_boundary.host_monotonic_tie_count",
                    "action": "computed_from_validated_csv_for_current_schema",
                    "source_schema_minor": payload["schema_minor"],
                    "target_schema_minor": METADATA_SCHEMA_MINOR,
                    "value": validator.host_monotonic_tie_count,
                })
            boundary["host_monotonic_tie_count"] = validator.host_monotonic_tie_count
            boundary["written_count"] = csv_rows
            boundary["eligible_count"] = max(boundary["eligible_count"], csv_rows)
            if boundary["end_host_receive_index"] is None and validator.previous is not None:
                boundary["end_host_receive_index"] = validator.previous.host_receive_index
            start_index = boundary["start_host_receive_index"]
            end_index = boundary["end_host_receive_index"]
            if start_index is not None and end_index is not None:
                boundary["received_count"] = max(
                    boundary["received_count"], end_index - start_index
                )
            boundary["incomplete_reason"] = "process_interrupted"
            payload["session_boundary"] = boundary
            if acquisition_migrations:
                payload["acquisition"] = asdict(acquisition)
            payload["status"]="incomplete"; payload["recovery"]={"strategy":"validated_csv_rows_are_fact","csv_rows":csv_rows,"checkpoint_rows":checkpoint,"consistent":csv_rows==checkpoint,"legacy_identity":legacy_identity,"source_schema_minor":payload["schema_minor"],"target_schema_minor":METADATA_SCHEMA_MINOR,"metadata_migrations":acquisition_migrations,"session_migrations":session_migrations}
            payload["recorder_state"]="CLOSED";payload["resource_state"]="CLOSED";payload["incomplete_reason"]="process_interrupted"
            payload["row_count"]=csv_rows;payload["last_persisted_row"]=csv_rows
            payload["closed_at"]=datetime.now(timezone.utc).isoformat()
            temporary=metadata_path.with_suffix(".json.recovery.tmp")
            with temporary.open("w",encoding="utf-8") as stream:
                json.dump(payload,stream,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False); stream.flush(); os.fsync(stream.fileno())
            try:_replace_recovery_metadata(temporary,metadata_path)
            except Exception as exc:
                try:temporary.unlink(missing_ok=True)
                except OSError:pass
                issues.append(RecoveryIssue(metadata_path.parent,"metadata_commit",f"{type(exc).__name__}: {exc}"))
                continue
            recovered.append(RecoveryRecord(metadata_path.parent,csv_rows,checkpoint,"incomplete"))
        except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
            issues.append(RecoveryIssue(metadata_path.parent,"invalid_session",f"{type(exc).__name__}: {exc}"))
        finally:
            try:lock.release()
            except Exception as exc:issues.append(RecoveryIssue(metadata_path.parent,"lock_release",f"{type(exc).__name__}: {exc}"))
    return RecoveryReport(recovered,issues)

def _iter_recovery_metadata(root:Path,issues:list[RecoveryIssue]):
    """Walk without following symlinks, junctions, or other reparse points."""
    stack=[root]
    while stack:
        directory=stack.pop()
        try:
            entries=list(os.scandir(directory))
        except OSError as exc:
            issues.append(RecoveryIssue(directory,"scan_failed",f"{type(exc).__name__}: {exc}"))
            continue
        for entry in entries:
            candidate=Path(entry.path)
            try:
                stat_result=entry.stat(follow_symlinks=False)
            except OSError as exc:
                issues.append(RecoveryIssue(candidate,"scan_failed",f"{type(exc).__name__}: {exc}"))
                continue
            is_reparse=entry.is_symlink() or bool(getattr(stat_result,"st_file_attributes",0)&0x400)
            if is_reparse:
                issues.append(RecoveryIssue(candidate,"unsafe_path","symlink or reparse point was not traversed"))
                continue
            if entry.is_dir(follow_symlinks=False):
                try:
                    resolved=candidate.resolve(strict=True)
                    resolved.relative_to(root)
                except (OSError,ValueError) as exc:
                    issues.append(RecoveryIssue(candidate,"unsafe_path",f"{type(exc).__name__}: directory escapes recovery root"))
                    continue
                stack.append(resolved)
                continue
            if entry.name!="metadata.json" or not entry.is_file(follow_symlinks=False):
                continue
            try:
                session_dir=candidate.parent.resolve(strict=True)
                session_dir.relative_to(root)
            except (OSError,ValueError) as exc:
                issues.append(RecoveryIssue(candidate.parent,"unsafe_path",f"{type(exc).__name__}: metadata session escapes recovery root"))
                continue
            yield session_dir/"metadata.json"
