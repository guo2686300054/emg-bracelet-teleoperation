# -*- coding: utf-8 -*-
"""Frozen candidate for the little-endian EMG SharedMemory v2 wire contract.

``host_wall_timestamp_ns`` is Unix epoch time. ``host_monotonic_ns`` is from
the host monotonic clock. ``device_time_ticks`` is an opaque device counter,
not nanoseconds. Optional counters/times are distinguished from a real zero by
their corresponding validity flags.

``generation`` identifies one shared-memory writer lifetime.  It is distinct
from ``connection_generation``, which identifies the BLE connection that
produced a frame and may change without replacing the shared-memory writer.

CRC32 covers stable header bytes 0..95 and exactly ``valid_length`` payload
bytes. CRC, commit sequence and padding are excluded. A seqlock-style commit
marker makes partially written snapshots detectable without flushing each
frame. ``flush()`` is for durability, not inter-process visibility.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import mmap
import os
import secrets
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Optional, Protocol, Union, cast

FRAME_SIZE = 1024
HEADER_SIZE = 128
PAYLOAD_OFFSET = HEADER_SIZE
PAYLOAD_CAPACITY = FRAME_SIZE - PAYLOAD_OFFSET
DEFAULT_SHARED_FILENAME = "emg_shared_data_v2.bin"
LEGACY_SHARED_FILENAME = "emg_shared_data.bin"
MAGIC = b"EMGSHM2\0"
VERSION_MAJOR = 2
VERSION_MINOR = 1

FLAG_VALID = 1 << 0
FLAG_INITIALIZING = 1 << 1
FLAG_HOST_WALL_TIME_VALID = 1 << 2
FLAG_HOST_MONOTONIC_VALID = 1 << 3
FLAG_HOST_RECEIVE_INDEX_VALID = 1 << 4
FLAG_DEVICE_PACKET_SEQUENCE_VALID = 1 << 5
FLAG_DEVICE_SAMPLE_COUNTER_VALID = 1 << 6
FLAG_DEVICE_TIME_VALID = 1 << 7
FLAG_DISCONNECTED = 1 << 8
FLAG_OVERFLOW = 1 << 9
FLAG_CRC_ERROR = 1 << 10
FLAG_STALE = 1 << 11
FLAG_CONNECTION_GENERATION_VALID = 1 << 12
FLAG_REPLAY = 1 << 13
FLAG_SYNTHETIC_TIME = 1 << 14
_VALIDITY_FLAGS = (
    FLAG_HOST_WALL_TIME_VALID
    | FLAG_HOST_MONOTONIC_VALID
    | FLAG_HOST_RECEIVE_INDEX_VALID
    | FLAG_DEVICE_PACKET_SEQUENCE_VALID
    | FLAG_DEVICE_SAMPLE_COUNTER_VALID
    | FLAG_DEVICE_TIME_VALID
    | FLAG_CONNECTION_GENERATION_VALID
)
_STATUS_FLAGS = (
    FLAG_DISCONNECTED
    | FLAG_OVERFLOW
    | FLAG_CRC_ERROR
    | FLAG_STALE
    | FLAG_REPLAY
    | FLAG_SYNTHETIC_TIME
)
_CALLER_FLAGS = _STATUS_FLAGS
_KNOWN_FLAGS = _VALIDITY_FLAGS | _STATUS_FLAGS | FLAG_VALID | FLAG_INITIALIZING

SAMPLE_UINT8 = 1
SAMPLE_INT16 = 2
SAMPLE_FLOAT32 = 3
_FORMAT_DETAILS = {SAMPLE_UINT8: ("B", 1), SAMPLE_INT16: ("h", 2), SAMPLE_FLOAT32: ("f", 4)}

HOST_WALL_TIMESTAMP_OFFSET = 16
HOST_MONOTONIC_OFFSET = 24
GENERATION_OFFSET = 32
SEQUENCE_OFFSET = 40
HOST_RECEIVE_INDEX_OFFSET = 48
DEVICE_PACKET_SEQUENCE_OFFSET = 56
DEVICE_SAMPLE_COUNTER_OFFSET = 64
DEVICE_TIME_TICKS_OFFSET = 72
VALID_LENGTH_OFFSET = 80
FLAGS_OFFSET = 84
CHANNEL_COUNT_OFFSET = 88
SAMPLE_COUNT_OFFSET = 90
SAMPLE_FORMAT_OFFSET = 92
CONTENT_CRC32_OFFSET = 96
COMMIT_SEQUENCE_OFFSET = 100
CONNECTION_GENERATION_OFFSET = 108
CRC_HEADER_END = CONTENT_CRC32_OFFSET

_HEADER = struct.Struct("<8sHHHHQQQQQQQQIIHHHHIQQ12x")
assert _HEADER.size == HEADER_SIZE
_UINT64_MAX = (1 << 64) - 1

_ACTIVE_WRITERS = set()
_ACTIVE_WRITERS_LOCK = threading.Lock()
_WAIT_OBJECT_0 = 0
_WAIT_ABANDONED = 0x80
_WAIT_TIMEOUT = 0x102
_READER_MAGIC_READ_HOOK = None


class SharedMemoryProtocolError(ValueError):
    pass


class SharedMemoryWriterAlreadyActive(RuntimeError):
    pass


class _Kernel32MutexApi(Protocol):
    def ReleaseMutex(self, handle: object, /) -> int: ...

    def CloseHandle(self, handle: object, /) -> int: ...


@dataclass
class _ProducerToken:
    key: str
    kernel32: Optional[_Kernel32MutexApi] = None
    handle: object = None
    mutex_released: bool = False
    handle_closed: bool = False
    registry_cleared: bool = False

    @property
    def has_native_mutex(self):
        return self.kernel32 is not None and self.handle is not None


class SharedMemoryProducerCleanup:
    """Reachable, owner-thread guard for retrying failed constructor cleanup."""

    def __init__(self, token, owner_thread_id):
        self._token = token
        self._owner_thread_id = owner_thread_id
        self._lock = threading.Lock()

    @property
    def pending(self):
        return self._token is not None

    @property
    def token(self):
        """Expose cleanup state for diagnostics without transferring ownership."""
        return self._token

    def retry(self):
        with self._lock:
            token = self._token
            if token is None:
                return
            if (
                token.has_native_mutex
                and self._owner_thread_id is not None
                and threading.get_ident() != self._owner_thread_id
            ):
                raise RuntimeError(
                    "Windows producer cleanup must run on the mutex creator thread"
                )
            _release_producer(token)
            self._token = None


@dataclass(frozen=True)
class SharedMemoryFrame:
    host_wall_timestamp_ns: int
    host_monotonic_ns: int
    generation: int
    connection_generation: int
    sequence: int
    host_receive_index: int
    device_packet_sequence: int
    device_sample_counter: int
    device_time_ticks: int
    valid_length: int
    flags: int
    channel_count: int
    sample_count: int
    sample_format: int
    payload: bytes

    def has(self, flag):
        return bool(self.flags & flag)

    def unpack_samples(self):
        try:
            format_char, item_size = _FORMAT_DETAILS[self.sample_format]
        except KeyError as exc:
            raise SharedMemoryProtocolError(f"unsupported sample format: {self.sample_format}") from exc
        count = self.valid_length // item_size
        return list(struct.unpack(f"<{count}{format_char}", self.payload))

    def is_live_control_eligible(self, expected_connection_generation):
        """Return eligibility within an explicitly bound connection lifetime."""

        if (
            not isinstance(expected_connection_generation, int)
            or isinstance(expected_connection_generation, bool)
            or expected_connection_generation <= 0
        ):
            return False
        unsafe = (
            FLAG_REPLAY | FLAG_SYNTHETIC_TIME | FLAG_STALE | FLAG_DISCONNECTED
            | FLAG_OVERFLOW | FLAG_CRC_ERROR
        )
        return (
            not bool(self.flags & unsafe)
            and self.channel_count == 8
            and self.has(FLAG_CONNECTION_GENERATION_VALID)
            and self.connection_generation == expected_connection_generation
        )

    def require_live_control_eligible(self, expected_connection_generation):
        if not self.is_live_control_eligible(expected_connection_generation):
            raise SharedMemoryProtocolError(
                "frame is not eligible for live control within the expected connection generation"
            )
        return self


def discover_shared_file(explicit_path=None, *, filename=DEFAULT_SHARED_FILENAME, script_dir=None, cwd=None):
    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser().resolve()
        return candidate if candidate.is_file() else None
    script_candidate = (Path(script_dir) if script_dir else Path(__file__).parent) / filename
    cwd_candidate = (Path(cwd) if cwd else Path.cwd()) / filename
    for candidate in (script_candidate.resolve(), cwd_candidate.resolve()):
        if candidate.is_file():
            return candidate
    return None


def _serialized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        lock = getattr(self, "_write_lock", None)
        with lock if lock is not None else contextlib.nullcontext():
            return method(self, *args, **kwargs)
    return wrapper


class SharedMemoryWriter:
    """Single-producer writer; sequence never wraps within a generation."""

    def __init__(self, shared_file_path, *, generation=None, flush_interval_frames=0):
        self.shared_file_path = Path(shared_file_path).expanduser().resolve()
        self._file = None
        self._mmap = None
        self._write_lock = threading.Lock()
        self._producer_token = None
        self._producer_owner_thread_id = None
        self._sequence = 0
        self._dirty = False
        self._writes_since_flush = 0
        self.flush_interval_frames = _uint("flush_interval_frames", flush_interval_frames, 32)
        generated = (secrets.randbits(64) or 1) if generation is None else generation
        self.generation = _uint("generation", generated, 64, nonzero=True)
        try:
            self._producer_token = _acquire_producer(self.shared_file_path)
            if self._producer_token.has_native_mutex:
                self._producer_owner_thread_id = threading.get_ident()
            self.shared_file_path.parent.mkdir(parents=True, exist_ok=True)
            if self.shared_file_path.exists():
                size = self.shared_file_path.stat().st_size
                if size != FRAME_SIZE:
                    raise SharedMemoryProtocolError(f"shared file size must be {FRAME_SIZE}, got {size}")
                self._file = open(self.shared_file_path, "r+b")
            else:
                self._file = open(self.shared_file_path, "x+b")
                self._file.truncate(FRAME_SIZE)
                self._file.flush()
            self._mmap = mmap.mmap(self._file.fileno(), FRAME_SIZE, access=mmap.ACCESS_WRITE)
            image = _build_header(generation=self.generation, flags=FLAG_INITIALIZING)
            self._mmap[:] = image + bytes(PAYLOAD_CAPACITY)
            self._dirty = True
        except BaseException as original_error:
            _close_without_masking(self, original_error)
            token = self._producer_token
            cleanup = SharedMemoryProducerCleanup(
                token, self._producer_owner_thread_id
            )
            try:
                cleanup.retry()
            except BaseException as cleanup_error:
                extras = (cleanup_error,) + tuple(
                    getattr(cleanup_error, "cleanup_errors", ())
                )
                _attach_cleanup_errors(original_error, extras)
                try:
                    setattr(original_error, "producer_cleanup", cleanup)
                except BaseException as attachment_error:
                    _attach_cleanup_errors(original_error, (attachment_error,))
            else:
                self._producer_token = None
                self._producer_owner_thread_id = None
            raise

    @_serialized
    def write_frame(
        self,
        payload,
        *,
        host_wall_timestamp_ns=None,
        host_monotonic_ns=None,
        host_receive_index=None,
        device_packet_sequence=None,
        device_sample_counter=None,
        device_time_ticks=None,
        connection_generation=None,
        sequence=None,
        flags=0,
        channel_count=8,
        sample_count=1,
        sample_format=SAMPLE_UINT8,
    ):
        if self._mmap is None or self._producer_token is None:
            raise RuntimeError("writer is closed")
        payload = bytes(payload)
        if len(payload) > PAYLOAD_CAPACITY:
            raise ValueError(f"payload exceeds {PAYLOAD_CAPACITY} bytes")
        wall = time.time_ns() if host_wall_timestamp_ns is None else _uint("host_wall_timestamp_ns", host_wall_timestamp_ns, 64)
        monotonic = time.monotonic_ns() if host_monotonic_ns is None else _uint("host_monotonic_ns", host_monotonic_ns, 64)
        receive, receive_flag = _optional_uint("host_receive_index", host_receive_index)
        packet, packet_flag = _optional_uint("device_packet_sequence", device_packet_sequence)
        sample_counter, sample_flag = _optional_uint("device_sample_counter", device_sample_counter)
        ticks, ticks_flag = _optional_uint("device_time_ticks", device_time_ticks)
        connection, connection_flag = _optional_uint(
            "connection_generation", connection_generation
        )
        channels = _uint("channel_count", channel_count, 16, nonzero=True)
        samples = _uint("sample_count", sample_count, 16, nonzero=True)
        sample_fmt = _uint("sample_format", sample_format, 16, nonzero=True)
        caller_flags = _uint("flags", flags, 32)
        if caller_flags & ~_CALLER_FLAGS:
            raise ValueError(f"flags contain unsupported or writer-owned bits: {caller_flags:#x}")
        _validate_payload(payload, channels, samples, sample_fmt, ValueError)
        if sequence is None:
            if self._sequence == _UINT64_MAX:
                raise OverflowError("sequence exhausted; create a writer with a new generation")
            next_sequence = self._sequence + 1
        else:
            next_sequence = _uint("sequence", sequence, 64, nonzero=True)
        if next_sequence <= self._sequence:
            raise ValueError(f"sequence must increase: {next_sequence} <= {self._sequence}")
        stable_flags = (
            caller_flags | FLAG_VALID | FLAG_HOST_WALL_TIME_VALID | FLAG_HOST_MONOTONIC_VALID
            | receive_flag | packet_flag | sample_flag | ticks_flag
            | connection_flag
        )
        values = dict(
            host_wall_timestamp_ns=wall,
            host_monotonic_ns=monotonic,
            generation=self.generation,
            sequence=next_sequence,
            host_receive_index=receive,
            device_packet_sequence=packet,
            device_sample_counter=sample_counter,
            device_time_ticks=ticks,
            connection_generation=connection,
            valid_length=len(payload),
            channel_count=channels,
            sample_count=samples,
            sample_format=sample_fmt,
        )
        initializing = _build_header(**values, flags=FLAG_INITIALIZING)
        crc_source = _build_header(**values, flags=stable_flags)
        crc = _content_crc(crc_source, payload, VERSION_MINOR)
        stable = _build_header(**values, flags=stable_flags, content_crc32=crc)
        image = stable + payload + bytes(PAYLOAD_CAPACITY - len(payload))
        struct.pack_into("<Q", self._mmap, COMMIT_SEQUENCE_OFFSET, 0)
        self._mmap[:HEADER_SIZE] = initializing
        self._mmap[:] = image
        struct.pack_into("<Q", self._mmap, COMMIT_SEQUENCE_OFFSET, next_sequence)
        self._sequence = next_sequence
        self._dirty = True
        self._writes_since_flush += 1
        if self.flush_interval_frames and self._writes_since_flush >= self.flush_interval_frames:
            self._flush_unlocked()
        return SharedMemoryFrame(
            wall, monotonic, self.generation, connection, next_sequence, receive, packet,
            sample_counter, ticks, len(payload), stable_flags, channels, samples,
            sample_fmt, payload,
        )

    def _flush_unlocked(self):
        if self._mmap is None and self._file is None:
            raise RuntimeError("writer is closed")
        if self._mmap is not None:
            self._mmap.flush()
        if self._file is not None:
            self._file.flush()
        self._dirty = False
        self._writes_since_flush = 0

    @_serialized
    def flush(self):
        self._flush_unlocked()

    def _close_resources_unlocked(self):
        flush_error = None
        if getattr(self, "_dirty", False) and self._mmap is not None:
            try:
                self._flush_unlocked()
            except BaseException as error:
                flush_error = error
        close_error = None
        try:
            _close_resources(self)
        except BaseException as error:
            close_error = error
        if self._mmap is None and self._file is None:
            self._dirty = False
            self._writes_since_flush = 0
        if flush_error is not None:
            _attach_cleanup_errors(flush_error, _error_tree(close_error))
            raise flush_error
        if close_error is not None:
            raise close_error

    def _release_producer_unlocked(self, *, require_resources_closed):
        token = getattr(self, "_producer_token", None)
        owner_thread_id = getattr(self, "_producer_owner_thread_id", None)
        if (
            token is not None
            and token.has_native_mutex
            and owner_thread_id is not None
            and threading.get_ident() != owner_thread_id
        ):
            raise RuntimeError(
                "Windows shared-memory writer producer lock must be released on its creator thread"
            )
        if require_resources_closed and (self._mmap is not None or self._file is not None):
            raise RuntimeError(
                "shared-memory resources must close before releasing the producer lock"
            )
        producer_error = None
        try:
            _release_producer(token)
        except BaseException as error:
            producer_error = error
        else:
            self._producer_token = None
            self._producer_owner_thread_id = None
        if producer_error is not None:
            raise producer_error

    @_serialized
    def close_resources(self):
        """Flush and close mmap/file resources; safe to run on a worker thread."""
        self._close_resources_unlocked()

    @_serialized
    def release_producer(self):
        """Release the thread-affine producer lock after resources are closed."""
        self._release_producer_unlocked(require_resources_closed=True)

    @_serialized
    def close(self):
        owner_thread_id = getattr(self, "_producer_owner_thread_id", None)
        token = getattr(self, "_producer_token", None)
        if (
            token is not None
            and token.has_native_mutex
            and owner_thread_id is not None
            and threading.get_ident() != owner_thread_id
        ):
            raise RuntimeError(
                "Windows shared-memory writer must be closed on its creator thread"
            )
        resource_error = None
        try:
            self._close_resources_unlocked()
        except BaseException as error:
            resource_error = error
        producer_error = None
        try:
            self._release_producer_unlocked(require_resources_closed=False)
        except BaseException as error:
            producer_error = error
        if resource_error is not None:
            _attach_cleanup_errors(resource_error, _error_tree(producer_error))
            raise resource_error
        if producer_error is not None:
            raise producer_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            try:
                self.close()
            except BaseException as cleanup_error:
                extras = (cleanup_error,) + tuple(
                    getattr(cleanup_error, "cleanup_errors", ())
                )
                _attach_cleanup_errors(exc_value, extras)
        else:
            self.close()
        return False


class SharedMemoryReader:
    def __init__(self, shared_file_path=None, *, wait_timeout=0.0, poll_interval=0.1, filename=DEFAULT_SHARED_FILENAME, script_dir=None, cwd=None, allow_legacy_minor=False, expected_connection_generation=None):
        self.shared_file_path = None
        self._file = None
        self._mmap = None
        self._last_generation = None
        self._last_sequence = None
        self.allow_legacy_minor = bool(allow_legacy_minor)
        if expected_connection_generation is not None and (
            not isinstance(expected_connection_generation, int)
            or isinstance(expected_connection_generation, bool)
            or expected_connection_generation <= 0
        ):
            raise ValueError("expected_connection_generation must be a positive integer")
        self.expected_connection_generation = expected_connection_generation
        deadline = time.monotonic() + max(0.0, wait_timeout)
        try:
            while True:
                located = discover_shared_file(shared_file_path, filename=filename, script_dir=script_dir, cwd=cwd)
                if located is not None:
                    self.shared_file_path = located
                    break
                if time.monotonic() >= deadline:
                    target = f"explicit path {shared_file_path!s}" if shared_file_path else "auto-probe paths"
                    migration = ""
                    if shared_file_path is None:
                        legacy = _find_legacy_files(script_dir=script_dir, cwd=cwd)
                        if legacy:
                            migration = (
                                f"; legacy {LEGACY_SHARED_FILENAME} found at {legacy[0]}, "
                                f"but v2 requires {filename}; migrate the writer and reader together"
                            )
                    raise FileNotFoundError(f"{filename} not found in {target}{migration}")
                time.sleep(max(0.001, poll_interval))
            while True:
                size = self.shared_file_path.stat().st_size
                if size == FRAME_SIZE:
                    break
                if size > FRAME_SIZE or wait_timeout <= 0:
                    raise SharedMemoryProtocolError(
                        f"shared file size must be {FRAME_SIZE}, got {size}"
                    )
                if time.monotonic() >= deadline:
                    raise SharedMemoryProtocolError(
                        f"shared file size initialization timed out at {size}; "
                        f"expected {FRAME_SIZE}"
                    )
                time.sleep(max(0.001, poll_interval))
            self._file = open(self.shared_file_path, "rb")
            self._mmap = mmap.mmap(self._file.fileno(), FRAME_SIZE, access=mmap.ACCESS_READ)
            while True:
                commit_before = struct.unpack_from(
                    "<Q", self._mmap, COMMIT_SEQUENCE_OFFSET
                )[0]
                prefix = self._mmap[: len(MAGIC)]
                hook = _READER_MAGIC_READ_HOOK
                if hook is not None:
                    hook()
                commit_after = struct.unpack_from(
                    "<Q", self._mmap, COMMIT_SEQUENCE_OFFSET
                )[0]
                if commit_before != commit_after:
                    if time.monotonic() >= deadline:
                        raise SharedMemoryProtocolError(
                            "v2 magic initialization timed out during concurrent commit"
                        )
                    time.sleep(max(0.001, poll_interval))
                    continue
                if prefix == MAGIC:
                    break
                if commit_before != 0 or not _is_initializing_magic_prefix(prefix):
                    raise SharedMemoryProtocolError(f"invalid v2 magic: {prefix!r}")
                if time.monotonic() >= deadline:
                    raise SharedMemoryProtocolError(
                        f"v2 magic initialization timed out: {prefix!r}"
                    )
                time.sleep(max(0.001, poll_interval))
        except BaseException as original_error:
            _close_without_masking(self, original_error)
            raise

    def read(self, *, only_new=False):
        if self._mmap is None:
            raise RuntimeError("reader is closed")
        before = struct.unpack_from("<Q", self._mmap, COMMIT_SEQUENCE_OFFSET)[0]
        if before == 0:
            return None
        snapshot = self._mmap[:FRAME_SIZE]
        after = struct.unpack_from("<Q", self._mmap, COMMIT_SEQUENCE_OFFSET)[0]
        if before != after or after == 0:
            return None
        fields = _HEADER.unpack_from(snapshot)
        (
            magic, major, minor, header_size, frame_size, wall, monotonic,
            generation, sequence, receive, packet, sample_counter, ticks,
            valid_length, flags, channels, samples, sample_fmt, reserved,
            content_crc32, commit,
            connection,
        ) = fields
        _validate_header(magic, major, minor, header_size, frame_size, generation, sequence, valid_length, flags, reserved, self.allow_legacy_minor)
        if commit != before or sequence != before:
            return None
        if flags & FLAG_INITIALIZING or not flags & FLAG_VALID:
            return None
        payload = snapshot[PAYLOAD_OFFSET : PAYLOAD_OFFSET + valid_length]
        crc = _content_crc(snapshot, payload, minor)
        if crc != content_crc32:
            raise SharedMemoryProtocolError("stable header/payload CRC mismatch")
        _validate_payload(payload, channels, samples, sample_fmt, SharedMemoryProtocolError)
        if only_new and self._last_generation == generation and sequence <= self._last_sequence:
            return None
        frame = SharedMemoryFrame(
            wall, monotonic, generation, connection, sequence, receive, packet,
            sample_counter, ticks, valid_length, flags, channels, samples,
            sample_fmt, payload,
        )
        self._last_generation, self._last_sequence = generation, sequence
        return frame

    def read_data(self, *, only_new=False):
        return self.read(only_new=only_new)

    def read_for_live_control(self, *, only_new=False):
        if self.expected_connection_generation is None:
            raise SharedMemoryProtocolError(
                "live control reader requires expected_connection_generation"
            )
        frame = self.read(only_new=only_new)
        if frame is not None:
            frame.require_live_control_eligible(self.expected_connection_generation)
        return frame

    def close(self):
        _close_resources(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            _close_without_masking(self, exc_value)
        else:
            self.close()
        return False


def _build_header(**values):
    return _HEADER.pack(
        MAGIC, VERSION_MAJOR, VERSION_MINOR, HEADER_SIZE, FRAME_SIZE,
        values.get("host_wall_timestamp_ns", 0), values.get("host_monotonic_ns", 0),
        values.get("generation", 0), values.get("sequence", 0),
        values.get("host_receive_index", 0), values.get("device_packet_sequence", 0),
        values.get("device_sample_counter", 0), values.get("device_time_ticks", 0),
        values.get("valid_length", 0), values.get("flags", 0),
        values.get("channel_count", 0), values.get("sample_count", 0),
        values.get("sample_format", 0), 0, values.get("content_crc32", 0),
        values.get("commit_sequence", 0),
        values.get("connection_generation", 0),
    )


def _content_crc(header, payload, minor):
    checksum = zlib.crc32(header[:CRC_HEADER_END])
    if minor >= 1:
        checksum = zlib.crc32(
            header[CONNECTION_GENERATION_OFFSET : CONNECTION_GENERATION_OFFSET + 8],
            checksum,
        )
    return zlib.crc32(payload, checksum) & 0xFFFFFFFF


def _find_legacy_files(*, script_dir=None, cwd=None):
    script_candidate = (
        Path(script_dir) if script_dir else Path(__file__).parent
    ) / LEGACY_SHARED_FILENAME
    cwd_candidate = (Path(cwd) if cwd else Path.cwd()) / LEGACY_SHARED_FILENAME
    found = []
    for candidate in (script_candidate.resolve(), cwd_candidate.resolve()):
        if candidate.is_file() and candidate not in found:
            found.append(candidate)
    return found


def _is_initializing_magic_prefix(prefix):
    """Return true only for an all-zero or sequentially written MAGIC prefix."""
    initialized = prefix.rstrip(b"\0")
    return len(initialized) < len(MAGIC) and MAGIC.startswith(initialized)


def _validate_header(magic, major, minor, header_size, frame_size, generation, sequence, valid_length, flags, reserved, allow_legacy_minor=False):
    if magic != MAGIC:
        raise SharedMemoryProtocolError(f"invalid magic: {magic!r}")
    if major != VERSION_MAJOR:
        raise SharedMemoryProtocolError(f"unsupported major version: {major}")
    if minor == 0 and not allow_legacy_minor:
        raise SharedMemoryProtocolError(
            "shared memory minor version 0 requires allow_legacy_minor=True"
        )
    if minor not in (0, VERSION_MINOR):
        raise SharedMemoryProtocolError(f"unsupported minor version: {minor}")
    if minor == 0 and flags & FLAG_CONNECTION_GENERATION_VALID:
        raise SharedMemoryProtocolError("minor version 0 cannot carry connection generation")
    if header_size != HEADER_SIZE:
        raise SharedMemoryProtocolError(f"invalid header size: {header_size}")
    if frame_size != FRAME_SIZE:
        raise SharedMemoryProtocolError(f"invalid frame size: {frame_size}")
    if not generation or not sequence:
        raise SharedMemoryProtocolError("stable generation and sequence must be nonzero")
    if valid_length > PAYLOAD_CAPACITY:
        raise SharedMemoryProtocolError(f"invalid payload length: {valid_length}")
    if flags & ~_KNOWN_FLAGS:
        raise SharedMemoryProtocolError(f"unknown flags: {flags:#x}")
    if flags & FLAG_VALID and flags & FLAG_INITIALIZING:
        raise SharedMemoryProtocolError("VALID and INITIALIZING cannot both be set")
    if reserved:
        raise SharedMemoryProtocolError("reserved header field must be zero")


def _validate_payload(payload, channels, samples, sample_format, error_type):
    try:
        item_size = _FORMAT_DETAILS[sample_format][1]
    except KeyError as exc:
        raise error_type(f"unsupported sample format: {sample_format}") from exc
    if channels <= 0 or samples <= 0:
        raise error_type("channel_count and sample_count must be positive")
    expected = channels * samples * item_size
    if len(payload) != expected:
        raise error_type(f"payload length {len(payload)} does not match expected {expected}")


def _optional_uint(name, value):
    if value is None:
        return 0, 0
    flag = {
        "host_receive_index": FLAG_HOST_RECEIVE_INDEX_VALID,
        "device_packet_sequence": FLAG_DEVICE_PACKET_SEQUENCE_VALID,
        "device_sample_counter": FLAG_DEVICE_SAMPLE_COUNTER_VALID,
        "device_time_ticks": FLAG_DEVICE_TIME_VALID,
        "connection_generation": FLAG_CONNECTION_GENERATION_VALID,
    }[name]
    return _uint(name, value, 64), flag


def _uint(name, value, bits, *, nonzero=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    maximum = (1 << bits) - 1
    if value < 0 or value > maximum or (nonzero and value == 0):
        raise ValueError(f"{name} must be {'nonzero ' if nonzero else ''}uint{bits}")
    return value


def _mutex_name(path):
    digest = hashlib.sha256(os.path.normcase(str(path)).encode("utf-8")).hexdigest()
    return f"Local\\EMGSharedMemoryV2-{digest}"


def _acquire_producer(path):
    key = os.path.normcase(str(path))
    with _ACTIVE_WRITERS_LOCK:
        if key in _ACTIVE_WRITERS:
            raise SharedMemoryWriterAlreadyActive(f"writer already active for {path}")
        _ACTIVE_WRITERS.add(key)
    if os.name != "nt":
        return _ProducerToken(key)
    kernel32 = None
    handle = None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        kernel32.WaitForSingleObject.restype = ctypes.c_ulong
        kernel32.ReleaseMutex.argtypes = (ctypes.c_void_p,)
        kernel32.ReleaseMutex.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.CreateMutexW(None, False, _mutex_name(path))
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateMutexW failed")
        result = kernel32.WaitForSingleObject(handle, 0)
        if result == _WAIT_TIMEOUT:
            raise SharedMemoryWriterAlreadyActive(f"writer already active for {path}")
        if result not in (_WAIT_OBJECT_0, _WAIT_ABANDONED):
            raise OSError(ctypes.get_last_error(), "WaitForSingleObject failed")
        return _ProducerToken(key, cast(_Kernel32MutexApi, kernel32), handle)
    except BaseException as original_error:
        try:
            if kernel32 is not None and handle is not None:
                if not kernel32.CloseHandle(handle):
                    _attach_cleanup_errors(
                        original_error,
                        (ctypes.WinError(ctypes.get_last_error()),),
                    )
        except BaseException as cleanup_error:
            _attach_cleanup_errors(original_error, (cleanup_error,))
        finally:
            with _ACTIVE_WRITERS_LOCK:
                _ACTIVE_WRITERS.discard(key)
        raise


def _release_producer(token):
    if token is None:
        return
    if not isinstance(token, _ProducerToken):
        raise TypeError("producer token must be an internal _ProducerToken")
    if token.registry_cleared:
        return
    if token.has_native_mutex and not token.mutex_released:
        kernel32 = token.kernel32
        if kernel32 is None or not kernel32.ReleaseMutex(token.handle):
            raise ctypes.WinError(ctypes.get_last_error())
        token.mutex_released = True
    if token.has_native_mutex and not token.handle_closed:
        kernel32 = token.kernel32
        if kernel32 is None or not kernel32.CloseHandle(token.handle):
            raise ctypes.WinError(ctypes.get_last_error())
        token.handle_closed = True
    if not token.registry_cleared:
        with _ACTIVE_WRITERS_LOCK:
            _ACTIVE_WRITERS.discard(token.key)
        token.registry_cleared = True


def _close_resources(owner):
    failures = []
    mapping, file_object = owner._mmap, owner._file
    try:
        if mapping is not None:
            try:
                mapping.close()
            except BaseException as error:
                failures.append(error)
            else:
                if owner._mmap is mapping:
                    owner._mmap = None
    finally:
        if file_object is not None:
            try:
                file_object.close()
            except BaseException as error:
                failures.append(error)
            else:
                if owner._file is file_object:
                    owner._file = None
    if failures:
        _attach_cleanup_errors(failures[0], failures[1:])
        raise failures[0]


def _close_without_masking(owner, original_error):
    try:
        _close_resources(owner)
    except BaseException as cleanup_error:
        extras = (cleanup_error,) + tuple(getattr(cleanup_error, "cleanup_errors", ()))
        _attach_cleanup_errors(original_error, extras)


def _attach_cleanup_errors(error, cleanup_errors):
    try:
        existing = tuple(getattr(error, "cleanup_errors", ()))
        error.cleanup_errors = existing + tuple(cleanup_errors)
    except BaseException:
        pass


def _error_tree(error):
    if error is None:
        return ()
    return (error,) + tuple(getattr(error, "cleanup_errors", ()))


__all__ = [name for name in globals() if name.startswith("FLAG_") or name.startswith("SAMPLE_")] + [
    "FRAME_SIZE", "HEADER_SIZE", "PAYLOAD_CAPACITY", "MAGIC", "VERSION_MAJOR",
    "VERSION_MINOR", "DEFAULT_SHARED_FILENAME", "LEGACY_SHARED_FILENAME",
    "SharedMemoryFrame", "SharedMemoryProtocolError",
    "SharedMemoryWriterAlreadyActive", "SharedMemoryProducerCleanup",
    "SharedMemoryReader", "SharedMemoryWriter",
    "discover_shared_file",
]
