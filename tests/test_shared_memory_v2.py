import contextlib
import io
import mmap
import multiprocessing
import os
import subprocess
import struct
import sys
import tempfile
import threading
import time
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import test as reader_cli
import shared_memory_v2 as shared_memory_module
from test import FrameDiagnostics
from shared_memory_v2 import (
    COMMIT_SEQUENCE_OFFSET,
    CONNECTION_GENERATION_OFFSET,
    CONTENT_CRC32_OFFSET,
    CRC_HEADER_END,
    DEFAULT_SHARED_FILENAME,
    FLAG_CONNECTION_GENERATION_VALID,
    FLAG_DEVICE_PACKET_SEQUENCE_VALID,
    FLAG_DEVICE_SAMPLE_COUNTER_VALID,
    FLAG_DEVICE_TIME_VALID,
    FLAG_DISCONNECTED,
    FLAG_HOST_RECEIVE_INDEX_VALID,
    FLAG_OVERFLOW,
    FLAG_REPLAY,
    FLAG_STALE,
    FLAG_SYNTHETIC_TIME,
    FLAG_VALID,
    FRAME_SIZE,
    HEADER_SIZE,
    LEGACY_SHARED_FILENAME,
    MAGIC,
    DEVICE_PACKET_SEQUENCE_OFFSET,
    PAYLOAD_CAPACITY,
    PAYLOAD_OFFSET,
    SAMPLE_FLOAT32,
    SAMPLE_FORMAT_OFFSET,
    SAMPLE_INT16,
    SAMPLE_UINT8,
    SEQUENCE_OFFSET,
    SharedMemoryProtocolError,
    SharedMemoryReader,
    SharedMemoryWriter,
    SharedMemoryWriterAlreadyActive,
    VERSION_MAJOR,
    VERSION_MINOR,
    discover_shared_file,
)


def _concurrent_writer(path, ready, start, count):
    with SharedMemoryWriter(path, generation=7001) as writer:
        ready.set()
        start.wait(5)
        for index in range(1, count + 1):
            writer.write_frame(
                bytes(((index + channel) & 0xFF for channel in range(8))),
                sequence=index,
                host_receive_index=index,
                device_packet_sequence=index,
                device_sample_counter=index,
            )
            time.sleep(0.0005)


def _attempt_second_writer(path, result_queue):
    try:
        with SharedMemoryWriter(path, generation=8002):
            result_queue.put("opened")
    except BaseException as error:
        result_queue.put(type(error).__name__)


class _CloseProbe:
    def __init__(self, error=None, *, failures_before_success=None):
        self.error = error
        self.failures_before_success = failures_before_success
        self.close_calls = 0
        self.closed = False

    @property
    def close_called(self):
        return self.close_calls > 0

    def close(self):
        self.close_calls += 1
        should_fail = self.error is not None and (
            self.failures_before_success is None or self.close_calls <= self.failures_before_success
        )
        if should_fail:
            raise self.error
        self.closed = True


class _InitFailMapping(_CloseProbe):
    def __setitem__(self, key, value):
        raise RuntimeError("initializing image failed")

    def flush(self):
        pass


class _FileProbe(_CloseProbe):
    def truncate(self, size):
        pass

    def flush(self):
        pass

    def fileno(self):
        return 123


class _NativeFunction:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.argtypes = None
        self.restype = None
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class _FakeKernel:
    def __init__(self, *, wait_result=0, release_result=1, close_result=1):
        self.CreateMutexW = _NativeFunction(123)
        self.WaitForSingleObject = _NativeFunction(wait_result)
        self.ReleaseMutex = _NativeFunction(release_result)
        self.CloseHandle = _NativeFunction(close_result)


class _SignatureRejectFunction(_NativeFunction):
    @property
    def argtypes(self):
        return None

    @argtypes.setter
    def argtypes(self, value):
        if value is not None:
            raise RuntimeError("signature setup failed")


class _FlushFailMapping:
    def __init__(self, mapping):
        self.mapping = mapping

    def flush(self):
        raise OSError("flush failed")

    def close(self):
        self.mapping.close()


class _FlushCloseFailMapping(_CloseProbe):
    def flush(self):
        raise OSError("flush failed")


class _TransientFlushCloseMapping:
    def __init__(self):
        self.flush_calls = 0
        self.close_calls = 0
        self.closed = False

    def flush(self):
        self.flush_calls += 1
        if self.flush_calls == 1:
            raise OSError("transient mmap flush failure")

    def close(self):
        self.close_calls += 1
        if self.close_calls == 1:
            raise OSError("transient mmap close failure")
        self.closed = True


class SharedMemoryV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / DEFAULT_SHARED_FILENAME

    def tearDown(self):
        self.temporary.cleanup()

    def test_round_trip_and_fixed_wire_vector(self):
        payload = bytes(range(8))
        with SharedMemoryWriter(self.path, generation=3) as writer:
            frame = writer.write_frame(
                payload,
                host_wall_timestamp_ns=11,
                host_monotonic_ns=12,
                sequence=13,
                host_receive_index=14,
                device_packet_sequence=15,
                device_sample_counter=16,
                device_time_ticks=17,
                connection_generation=18,
            )
        raw = self.path.read_bytes()
        expected_prefix = struct.pack(
            "<8sHHHHQQQQQQQQIIHHHH",
            MAGIC,
            VERSION_MAJOR,
            VERSION_MINOR,
            128,
            1024,
            11,
            12,
            3,
            13,
            14,
            15,
            16,
            17,
            8,
            frame.flags,
            8,
            1,
            SAMPLE_UINT8,
            0,
        )
        self.assertEqual(HEADER_SIZE, 128)
        self.assertEqual(len(raw), FRAME_SIZE)
        self.assertEqual(raw[:CRC_HEADER_END], expected_prefix)
        self.assertEqual(struct.unpack_from("<Q", raw, COMMIT_SEQUENCE_OFFSET)[0], 13)
        self.assertEqual(
            struct.unpack_from("<Q", raw, CONNECTION_GENERATION_OFFSET)[0], 18
        )
        self.assertEqual(raw[PAYLOAD_OFFSET : PAYLOAD_OFFSET + 8], payload)
        with SharedMemoryReader(self.path) as reader:
            read = reader.read()
        self.assertEqual(read, frame)
        self.assertEqual(read.unpack_samples(), list(range(8)))

    def test_minor_one_crc_has_an_independent_fixed_vector(self):
        payload = bytes(range(8))
        flags = 0x10FD
        prefix = struct.pack(
            "<8sHHHHQQQQQQQQIIHHHH",
            MAGIC,
            2,
            1,
            128,
            1024,
            11,
            12,
            3,
            13,
            14,
            15,
            16,
            17,
            8,
            flags,
            8,
            1,
            SAMPLE_UINT8,
            0,
        )
        checksum = zlib.crc32(prefix)
        checksum = zlib.crc32(struct.pack("<Q", 18), checksum)
        checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
        self.assertEqual(checksum, 0x8414EBFC)
        with SharedMemoryWriter(self.path, generation=3) as writer:
            frame = writer.write_frame(
                payload,
                host_wall_timestamp_ns=11,
                host_monotonic_ns=12,
                sequence=13,
                host_receive_index=14,
                device_packet_sequence=15,
                device_sample_counter=16,
                device_time_ticks=17,
                connection_generation=18,
            )
        raw = self.path.read_bytes()
        self.assertEqual(frame.flags, flags)
        self.assertEqual(
            struct.unpack_from("<I", raw, CONTENT_CRC32_OFFSET)[0], checksum
        )

    def test_round_trip_supported_sample_formats(self):
        cases = (
            (SAMPLE_INT16, struct.pack("<4h", -2, -1, 1, 2), [-2, -1, 1, 2]),
            (SAMPLE_FLOAT32, struct.pack("<2f", 1.5, -2.5), [1.5, -2.5]),
        )
        for sample_format, payload, expected in cases:
            with self.subTest(sample_format=sample_format):
                with SharedMemoryWriter(self.path, generation=5) as writer:
                    writer.write_frame(
                        payload,
                        channel_count=2,
                        sample_count=len(expected) // 2,
                        sample_format=sample_format,
                    )
                    with SharedMemoryReader(self.path) as reader:
                        self.assertEqual(reader.read().unpack_samples(), expected)

    def test_only_new_rejects_duplicate_and_regression_but_accepts_generation_change(self):
        with SharedMemoryWriter(self.path, generation=10) as writer:
            writer.write_frame(b"12345678", sequence=5)
            with SharedMemoryReader(self.path) as reader:
                self.assertEqual(reader.read(only_new=True).sequence, 5)
                self.assertIsNone(reader.read(only_new=True))
                self._set_sequence_and_generation(sequence=4, generation=10)
                self.assertIsNone(reader.read(only_new=True))
                self._set_sequence_and_generation(sequence=1, generation=11)
                restarted = reader.read(only_new=True)
                self.assertEqual((restarted.generation, restarted.sequence), (11, 1))

    def test_windows_reader_survives_real_writer_restart_on_same_file(self):
        writer_one = SharedMemoryWriter(self.path, generation=101)
        writer_one.write_frame(b"12345678", sequence=1)
        with SharedMemoryReader(self.path) as reader:
            first = reader.read(only_new=True)
            self.assertEqual((first.generation, first.sequence), (101, 1))
            writer_one.close()

            writer_two = SharedMemoryWriter(self.path, generation=102)
            try:
                # Constructor publishes the new generation's commit=0 state.
                self.assertIsNone(reader.read(only_new=True))
                writer_two.write_frame(b"abcdefgh", sequence=1)
                second = reader.read(only_new=True)
                self.assertEqual((second.generation, second.sequence), (102, 1))
                self.assertEqual(second.payload, b"abcdefgh")
            finally:
                writer_two.close()

    @unittest.skipUnless(os.name == "nt", "named mutex assertion is Windows-specific")
    def test_named_mutex_rejects_second_writer_in_another_process(self):
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        with SharedMemoryWriter(self.path, generation=801) as writer:
            process = context.Process(
                target=_attempt_second_writer, args=(str(self.path), result_queue)
            )
            process.start()
            process.join(5)
            self.assertEqual(process.exitcode, 0)
            self.assertEqual(result_queue.get(timeout=1), "SharedMemoryWriterAlreadyActive")
            writer.write_frame(b"12345678")

    def test_same_process_second_writer_is_rejected(self):
        with SharedMemoryWriter(self.path, generation=802):
            with self.assertRaises(SharedMemoryWriterAlreadyActive):
                SharedMemoryWriter(self.path, generation=803)

    def test_concurrent_calls_on_one_writer_are_serialized(self):
        sequences = []
        errors = []
        with SharedMemoryWriter(self.path, generation=804) as writer:
            def write_many():
                try:
                    for _ in range(50):
                        sequences.append(writer.write_frame(b"12345678").sequence)
                except BaseException as error:
                    errors.append(error)

            threads = [threading.Thread(target=write_many) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(sorted(sequences), list(range(1, 201)))

    def test_optional_counter_zero_is_distinct_from_unknown(self):
        with SharedMemoryWriter(self.path, generation=805) as writer:
            unknown = writer.write_frame(b"12345678")
            known_zero = writer.write_frame(
                b"abcdefgh",
                host_receive_index=0,
                device_packet_sequence=0,
                device_sample_counter=0,
                device_time_ticks=0,
                connection_generation=0,
            )
        validity = (
            FLAG_HOST_RECEIVE_INDEX_VALID
            | FLAG_DEVICE_PACKET_SEQUENCE_VALID
            | FLAG_DEVICE_SAMPLE_COUNTER_VALID
            | FLAG_DEVICE_TIME_VALID
            | FLAG_CONNECTION_GENERATION_VALID
        )
        self.assertEqual(unknown.flags & validity, 0)
        self.assertEqual(known_zero.flags & validity, validity)

    def test_default_writes_do_not_flush_every_frame_and_explicit_flush_resets_period(self):
        with SharedMemoryWriter(self.path, generation=806) as writer:
            writer.write_frame(b"12345678")
            writer.write_frame(b"abcdefgh")
            self.assertEqual(writer._writes_since_flush, 2)
            self.assertTrue(writer._dirty)
            writer.flush()
            self.assertEqual(writer._writes_since_flush, 0)
            self.assertFalse(writer._dirty)

    def test_configured_flush_period_flushes_on_boundary(self):
        with SharedMemoryWriter(
            self.path, generation=807, flush_interval_frames=2
        ) as writer:
            writer.write_frame(b"12345678")
            self.assertEqual(writer._writes_since_flush, 1)
            self.assertTrue(writer._dirty)
            writer.write_frame(b"abcdefgh")
            self.assertEqual(writer._writes_since_flush, 0)
            self.assertFalse(writer._dirty)

    def test_writer_rejects_sequence_zero_duplicate_regression_and_wrap(self):
        with SharedMemoryWriter(self.path, generation=20) as writer:
            writer.write_frame(b"12345678", sequence=2)
            baseline = self.path.read_bytes()
            for invalid in (0, 2, 1):
                with self.subTest(sequence=invalid):
                    with self.assertRaises(ValueError):
                        writer.write_frame(b"abcdefgh", sequence=invalid)
                    self.assertEqual(self.path.read_bytes(), baseline)
            writer.write_frame(b"abcdefgh", sequence=(1 << 64) - 1)
            maximum_snapshot = self.path.read_bytes()
            with self.assertRaises(OverflowError):
                writer.write_frame(b"ABCDEFGH")
            self.assertEqual(self.path.read_bytes(), maximum_snapshot)

    def test_all_invalid_write_fields_preserve_previous_snapshot(self):
        with SharedMemoryWriter(self.path, generation=30) as writer:
            writer.write_frame(b"12345678")
            baseline = self.path.read_bytes()
            invalid_arguments = (
                {"host_wall_timestamp_ns": -1},
                {"host_monotonic_ns": 1 << 64},
                {"host_receive_index": -1},
                {"device_packet_sequence": 1 << 64},
                {"device_sample_counter": -1},
                {"device_time_ticks": 1 << 64},
                {"connection_generation": -1},
                {"channel_count": 1 << 16},
                {"sample_count": -1},
                {"sample_format": 999},
                {"flags": 1 << 31},
                {"flags": FLAG_VALID},
            )
            for arguments in invalid_arguments:
                with self.subTest(arguments=arguments):
                    with self.assertRaises((TypeError, ValueError)):
                        writer.write_frame(b"abcdefgh", **arguments)
                    self.assertEqual(self.path.read_bytes(), baseline)

    def test_invalid_generation_is_rejected_before_existing_file_is_opened(self):
        self.path.write_bytes(b"preserve-existing-content")
        baseline = self.path.read_bytes()
        for generation in (0, -1, 1 << 64):
            with self.subTest(generation=generation):
                with self.assertRaises((TypeError, ValueError)):
                    SharedMemoryWriter(self.path, generation=generation)
                self.assertEqual(self.path.read_bytes(), baseline)

    def test_oversize_payload_preserves_previous_snapshot(self):
        with SharedMemoryWriter(self.path, generation=31) as writer:
            writer.write_frame(b"12345678")
            baseline = self.path.read_bytes()
            with self.assertRaisesRegex(ValueError, "exceeds"):
                writer.write_frame(bytes(PAYLOAD_CAPACITY + 1))
            self.assertEqual(self.path.read_bytes(), baseline)

    def test_zero_commit_startup_times_out_with_protocol_error(self):
        self.path.write_bytes(bytes(FRAME_SIZE))
        with self.assertRaisesRegex(SharedMemoryProtocolError, "initialization timed out"):
            SharedMemoryReader(self.path, wait_timeout=0)

    def test_initializing_or_inconsistent_commit_is_not_returned(self):
        with SharedMemoryWriter(self.path, generation=40) as writer:
            with SharedMemoryReader(self.path) as reader:
                self.assertIsNone(reader.read())
            writer.write_frame(b"12345678")
        self._overwrite(COMMIT_SEQUENCE_OFFSET, struct.pack("<Q", 0))
        with SharedMemoryReader(self.path) as reader:
            self.assertIsNone(reader.read())

    def test_corrupt_magic_version_minor_sizes_and_flags(self):
        corruptions = (
            (0, b"BADMAGIC", "magic"),
            (8, struct.pack("<H", VERSION_MAJOR + 1), "major version"),
            (10, struct.pack("<H", VERSION_MINOR + 1), "minor version"),
            (12, struct.pack("<H", HEADER_SIZE - 1), "header size"),
            (14, struct.pack("<H", FRAME_SIZE - 1), "frame size"),
            (84, struct.pack("<I", 1 << 31), "unknown flags"),
        )
        for offset, value, message in corruptions:
            with self.subTest(message=message):
                self._write_valid()
                self._overwrite(offset, value)
                with self.assertRaisesRegex(SharedMemoryProtocolError, message):
                    with SharedMemoryReader(self.path) as reader:
                        reader.read()

    def test_minor_zero_requires_explicit_compatibility_mode(self):
        with SharedMemoryWriter(self.path, generation=32) as writer:
            writer.write_frame(b"12345678", sequence=1)
        self._overwrite(10, struct.pack("<H", 0))
        self._recalculate_crc()
        with SharedMemoryReader(self.path) as reader:
            with self.assertRaisesRegex(
                SharedMemoryProtocolError,
                "allow_legacy_minor=True",
            ):
                reader.read()
        with SharedMemoryReader(self.path, allow_legacy_minor=True) as reader:
            frame = reader.read()
        self.assertEqual(frame.generation, 32)
        self.assertEqual(frame.connection_generation, 0)
        self.assertFalse(frame.flags & FLAG_CONNECTION_GENERATION_VALID)

    def test_exact_file_size_required(self):
        for size in (FRAME_SIZE - 1, FRAME_SIZE + 1):
            with self.subTest(size=size):
                self.path.write_bytes(bytes(size))
                with self.assertRaisesRegex(SharedMemoryProtocolError, "size must"):
                    SharedMemoryReader(self.path)

    def test_short_file_waits_for_truncate_and_initialization_explicit_and_auto(self):
        for explicit in (True, False):
            with self.subTest(explicit=explicit):
                self.path.write_bytes(b"")

                def initialize_file():
                    time.sleep(0.02)
                    with open(self.path, "r+b", buffering=0) as stream:
                        stream.truncate(FRAME_SIZE)
                        stream.seek(0)
                        stream.write(
                            shared_memory_module._build_header(
                                generation=1201,
                                flags=shared_memory_module.FLAG_INITIALIZING,
                            )
                        )
                        stream.flush()

                worker = threading.Thread(target=initialize_file)
                worker.start()
                try:
                    arguments = (self.path,) if explicit else ()
                    with SharedMemoryReader(
                        *arguments,
                        wait_timeout=1,
                        poll_interval=0.002,
                        script_dir=self.root,
                        cwd=self.root,
                    ) as reader:
                        self.assertIsNone(reader.read())
                finally:
                    worker.join(1)
                self.assertFalse(worker.is_alive())

    def test_short_file_initialization_timeout_and_oversize_are_protocol_errors(self):
        for size in (0, 1, FRAME_SIZE - 1):
            with self.subTest(size=size):
                self.path.write_bytes(bytes(size))
                with self.assertRaisesRegex(
                    SharedMemoryProtocolError,
                    "size initialization timed out",
                ):
                    SharedMemoryReader(
                        self.path,
                        wait_timeout=0.005,
                        poll_interval=0.001,
                    )
        self.path.write_bytes(bytes(FRAME_SIZE + 1))
        started = time.monotonic()
        with self.assertRaisesRegex(SharedMemoryProtocolError, "size must"):
            SharedMemoryReader(self.path, wait_timeout=1)
        self.assertLess(time.monotonic() - started, 0.5)

    def test_crc_covers_payload_and_stable_metadata(self):
        for offset in (
            PAYLOAD_OFFSET,
            DEVICE_PACKET_SEQUENCE_OFFSET,
            CONNECTION_GENERATION_OFFSET,
        ):
            with self.subTest(offset=offset):
                self._write_valid()
                raw = bytearray(self.path.read_bytes())
                raw[offset] ^= 0x01
                self.path.write_bytes(raw)
                with SharedMemoryReader(self.path) as reader:
                    with self.assertRaisesRegex(SharedMemoryProtocolError, "CRC"):
                        reader.read()

    def test_reader_shape_and_format_errors_are_protocol_errors(self):
        for offset, value, message in (
            (88, struct.pack("<H", 7), "payload length"),
            (SAMPLE_FORMAT_OFFSET, struct.pack("<H", 99), "sample format"),
        ):
            with self.subTest(message=message):
                self._write_valid()
                self._overwrite(offset, value)
                self._recalculate_crc()
                with SharedMemoryReader(self.path) as reader:
                    with self.assertRaisesRegex(SharedMemoryProtocolError, message):
                        reader.read()

    def test_crc_and_shape_are_checked_before_only_new(self):
        self._write_valid()
        with SharedMemoryReader(self.path) as reader:
            self.assertIsNotNone(reader.read(only_new=True))
            current = self.path.read_bytes()[PAYLOAD_OFFSET]
            self._overwrite(PAYLOAD_OFFSET, bytes((current ^ 1,)))
            with self.assertRaisesRegex(SharedMemoryProtocolError, "CRC"):
                reader.read(only_new=True)

    def test_explicit_missing_path_does_not_fall_back(self):
        automatic = self.root / "auto" / DEFAULT_SHARED_FILENAME
        automatic.parent.mkdir()
        automatic.write_bytes(bytes(FRAME_SIZE))
        missing = self.root / "missing.bin"
        self.assertIsNone(
            discover_shared_file(missing, script_dir=automatic.parent, cwd=automatic.parent)
        )
        with self.assertRaises(FileNotFoundError):
            SharedMemoryReader(
                missing,
                wait_timeout=0,
                script_dir=automatic.parent,
                cwd=automatic.parent,
            )

    def test_auto_detection_prefers_script_then_cwd(self):
        script = self.root / "script"
        cwd = self.root / "cwd"
        script.mkdir()
        cwd.mkdir()
        script_file = script / DEFAULT_SHARED_FILENAME
        cwd_file = cwd / DEFAULT_SHARED_FILENAME
        script_file.write_bytes(b"script")
        cwd_file.write_bytes(b"cwd")
        self.assertEqual(discover_shared_file(script_dir=script, cwd=cwd), script_file.resolve())
        script_file.unlink()
        self.assertEqual(discover_shared_file(script_dir=script, cwd=cwd), cwd_file.resolve())

    def test_auto_detection_selects_v2_when_legacy_and_v2_both_exist(self):
        legacy = self.root / LEGACY_SHARED_FILENAME
        legacy.write_bytes(struct.pack("<II", 8, 2) + bytes(FRAME_SIZE - 8))
        with SharedMemoryWriter(self.path, generation=1001) as writer:
            writer.write_frame(b"12345678")
        self.assertEqual(
            discover_shared_file(script_dir=self.root, cwd=self.root),
            self.path.resolve(),
        )
        with SharedMemoryReader(script_dir=self.root, cwd=self.root) as reader:
            self.assertEqual(reader.read().generation, 1001)

    def test_auto_detected_v2_name_rejects_wrong_magic_with_zero_commit(self):
        self.path.write_bytes(b"BADMAGIC" + bytes(FRAME_SIZE - len(MAGIC)))
        with self.assertRaisesRegex(SharedMemoryProtocolError, "invalid v2 magic"):
            SharedMemoryReader(
                wait_timeout=1,
                script_dir=self.root,
                cwd=self.root,
            )

    def test_explicit_and_auto_readers_wait_for_partial_magic_to_complete(self):
        for explicit in (True, False):
            with self.subTest(explicit=explicit):
                self.path.write_bytes(b"E" + bytes(FRAME_SIZE - 1))

                def complete_magic():
                    with open(self.path, "r+b", buffering=0) as stream:
                        for end in (3, 6, len(MAGIC)):
                            time.sleep(0.015)
                            stream.seek(0)
                            stream.write(MAGIC[:end])
                            stream.flush()

                worker = threading.Thread(target=complete_magic)
                worker.start()
                try:
                    arguments = (self.path,) if explicit else ()
                    with SharedMemoryReader(
                        *arguments,
                        wait_timeout=1,
                        poll_interval=0.002,
                        script_dir=self.root,
                        cwd=self.root,
                    ) as reader:
                        self.assertIsNone(reader.read())
                finally:
                    worker.join(1)
                self.assertFalse(worker.is_alive())

    def test_magic_decision_retries_when_writer_commits_after_magic_read(self):
        self.path.write_bytes(b"BADMAGIC" + bytes(FRAME_SIZE - len(MAGIC)))
        submitted = []

        def submit_valid_frame_once():
            if submitted:
                return
            submitted.append(True)
            with SharedMemoryWriter(self.path, generation=1301) as writer:
                writer.write_frame(b"12345678")

        with patch.object(
            shared_memory_module,
            "_READER_MAGIC_READ_HOOK",
            submit_valid_frame_once,
        ):
            with SharedMemoryReader(
                self.path,
                wait_timeout=1,
                poll_interval=0.001,
            ) as reader:
                frame = reader.read()
        self.assertEqual((frame.generation, frame.sequence), (1301, 1))

    def test_partial_magic_times_out_instead_of_returning_no_frame_forever(self):
        self.path.write_bytes(b"EMG" + bytes(FRAME_SIZE - 3))
        with self.assertRaisesRegex(
            SharedMemoryProtocolError,
            "v2 magic initialization timed out",
        ):
            SharedMemoryReader(self.path, wait_timeout=0.01, poll_interval=0.002)

    def test_auto_detected_v2_name_rejects_legacy_header_immediately(self):
        self.path.write_bytes(struct.pack("<II", 8, 2) + bytes(FRAME_SIZE - 8))
        started = time.monotonic()
        with self.assertRaisesRegex(SharedMemoryProtocolError, "invalid v2 magic"):
            SharedMemoryReader(
                wait_timeout=1,
                script_dir=self.root,
                cwd=self.root,
            )
        self.assertLess(time.monotonic() - started, 0.5)

    def test_only_legacy_file_reports_migration_instead_of_parsing_as_v2(self):
        legacy = self.root / LEGACY_SHARED_FILENAME
        legacy.write_bytes(struct.pack("<II", 8, 2) + bytes(FRAME_SIZE - 8))
        with self.assertRaisesRegex(
            FileNotFoundError,
            f"legacy {LEGACY_SHARED_FILENAME}.*v2 requires {DEFAULT_SHARED_FILENAME}.*migrate",
        ):
            SharedMemoryReader(
                wait_timeout=0.01,
                poll_interval=0.002,
                script_dir=self.root,
                cwd=self.root,
            )

    def test_explicit_legacy_file_is_strictly_validated_as_v2(self):
        legacy = self.root / LEGACY_SHARED_FILENAME
        legacy.write_bytes(struct.pack("<II", 8, 2) + bytes(FRAME_SIZE - 8))
        with self.assertRaisesRegex(SharedMemoryProtocolError, "invalid v2 magic"):
            SharedMemoryReader(legacy, wait_timeout=0)

    def test_close_is_idempotent_and_partial_initialization_cleans_up(self):
        with SharedMemoryWriter(self.path, generation=50) as writer:
            writer.write_frame(b"12345678")
        reader = SharedMemoryReader(self.path)
        reader.close()
        reader.close()
        writer.close()
        writer.close()
        with patch("shared_memory_v2.mmap.mmap", side_effect=OSError("mapping failed")):
            with self.assertRaisesRegex(OSError, "mapping failed"):
                SharedMemoryReader(self.path)
        self.path.unlink()

    def test_close_attempts_both_resources_and_retains_all_failures(self):
        for owner_type in (SharedMemoryWriter, SharedMemoryReader):
            with self.subTest(owner_type=owner_type.__name__):
                mapping_error = OSError("mapping close failed")
                file_error = OSError("file close failed")
                mapping = _CloseProbe(mapping_error)
                file_object = _CloseProbe(file_error)
                owner = owner_type.__new__(owner_type)
                owner._mmap = mapping
                owner._file = file_object
                with self.assertRaisesRegex(OSError, "mapping close failed") as caught:
                    owner.close()
                self.assertTrue(mapping.close_called)
                self.assertTrue(file_object.close_called)
                self.assertEqual(len(caught.exception.cleanup_errors), 1)
                self.assertRegex(str(caught.exception.cleanup_errors[0]), "file close failed")
                self.assertIs(owner._mmap, mapping)
                self.assertIs(owner._file, file_object)
                with self.assertRaisesRegex(OSError, "mapping close failed"):
                    owner.close()
                self.assertEqual(mapping.close_calls, 2)
                self.assertEqual(file_object.close_calls, 2)

    def test_transient_close_failures_are_retried_until_resources_close(self):
        for owner_type in (SharedMemoryWriter, SharedMemoryReader):
            with self.subTest(owner_type=owner_type.__name__):
                mapping = _CloseProbe(
                    OSError("transient mapping close"), failures_before_success=1
                )
                file_object = _CloseProbe(
                    OSError("transient file close"), failures_before_success=1
                )
                owner = owner_type.__new__(owner_type)
                owner._mmap = mapping
                owner._file = file_object
                with self.assertRaisesRegex(OSError, "transient mapping close") as caught:
                    owner.close()
                self.assertEqual(len(caught.exception.cleanup_errors), 1)
                self.assertIs(owner._mmap, mapping)
                self.assertIs(owner._file, file_object)

                owner.close()
                self.assertEqual(mapping.close_calls, 2)
                self.assertEqual(file_object.close_calls, 2)
                self.assertTrue(mapping.closed)
                self.assertTrue(file_object.closed)
                self.assertIsNone(owner._mmap)
                self.assertIsNone(owner._file)
                owner.close()
                self.assertEqual(mapping.close_calls, 2)
                self.assertEqual(file_object.close_calls, 2)

    def test_file_close_failure_is_reported_when_mapping_close_succeeds(self):
        owner = SharedMemoryReader.__new__(SharedMemoryReader)
        owner._mmap = _CloseProbe()
        owner._file = _CloseProbe(OSError("file close failed"))
        with self.assertRaisesRegex(OSError, "file close failed"):
            owner.close()

    def test_business_error_is_not_masked_by_context_cleanup_failures(self):
        owner = SharedMemoryWriter.__new__(SharedMemoryWriter)
        mapping = _CloseProbe(OSError("mapping close failed"))
        file_object = _CloseProbe(OSError("file close failed"))
        owner._mmap = mapping
        owner._file = file_object
        business_error = RuntimeError("business failed")
        self.assertFalse(owner.__exit__(RuntimeError, business_error, None))
        self.assertTrue(mapping.close_called)
        self.assertTrue(file_object.close_called)
        self.assertEqual(len(business_error.cleanup_errors), 2)
        self.assertEqual(str(business_error), "business failed")

    def test_mapping_then_initialization_failure_releases_both_and_preserves_original(self):
        mapping = _InitFailMapping(OSError("mapping close failed"))
        file_object = _FileProbe(OSError("file close failed"))
        with patch("builtins.open", return_value=file_object), patch(
            "shared_memory_v2.mmap.mmap", return_value=mapping
        ):
            with self.assertRaisesRegex(RuntimeError, "initializing image failed") as caught:
                SharedMemoryWriter(self.path, generation=501)
        self.assertTrue(mapping.close_called)
        self.assertTrue(file_object.close_called)
        self.assertEqual(len(caught.exception.cleanup_errors), 2)
        # Producer ownership is released even though both fake closes failed.
        with SharedMemoryWriter(self.path, generation=502) as next_writer:
            next_writer.write_frame(b"12345678")

    def test_constructor_original_error_retains_producer_release_failure_context(self):
        mapping = _InitFailMapping()
        file_object = _FileProbe()
        real_release = shared_memory_module._release_producer

        def release_then_report_failure(token):
            real_release(token)
            raise OSError("producer release reported failure")

        with patch("builtins.open", return_value=file_object), patch(
            "shared_memory_v2.mmap.mmap", return_value=mapping
        ), patch(
            "shared_memory_v2._release_producer", side_effect=release_then_report_failure
        ):
            with self.assertRaisesRegex(RuntimeError, "initializing image failed") as caught:
                SharedMemoryWriter(self.path, generation=503)
        self.assertTrue(
            any(
                "producer release reported failure" in str(error)
                for error in caught.exception.cleanup_errors
            )
        )
        with SharedMemoryWriter(self.path, generation=504) as next_writer:
            next_writer.write_frame(b"12345678")

    def test_constructor_combines_init_double_close_and_producer_release_failures(self):
        mapping = _InitFailMapping(OSError("mapping close failed"))
        file_object = _FileProbe(OSError("file close failed"))
        real_release = shared_memory_module._release_producer

        def release_then_fail(token):
            real_release(token)
            release_error = OSError("producer release failed")
            release_error.cleanup_errors = (OSError("native handle close failed"),)
            raise release_error

        with patch("builtins.open", return_value=file_object), patch(
            "shared_memory_v2.mmap.mmap", return_value=mapping
        ), patch("shared_memory_v2._release_producer", side_effect=release_then_fail):
            with self.assertRaisesRegex(RuntimeError, "initializing image failed") as caught:
                SharedMemoryWriter(self.path, generation=505)
        messages = [str(error) for error in caught.exception.cleanup_errors]
        self.assertEqual(
            messages,
            [
                "mapping close failed",
                "file close failed",
                "producer release failed",
                "native handle close failed",
            ],
        )
        with SharedMemoryWriter(self.path, generation=506) as next_writer:
            next_writer.write_frame(b"12345678")

    @unittest.skipUnless(os.name == "nt", "native constructor cleanup is Windows-specific")
    def test_constructor_release_mutex_failure_exposes_retryable_cleanup_owner(self):
        kernel = _FakeKernel(wait_result=0, release_result=0, close_result=1)
        with patch("shared_memory_v2.ctypes.WinDLL", return_value=kernel), patch(
            "shared_memory_v2.mmap.mmap", side_effect=RuntimeError("mmap init failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "mmap init failed") as caught:
                SharedMemoryWriter(self.path, generation=507)

        cleanup = caught.exception.producer_cleanup
        token = cleanup.token
        key = os.path.normcase(str(self.path.resolve()))
        self.assertTrue(cleanup.pending)
        self.assertEqual(token.handle, 123)
        self.assertFalse(token.mutex_released)
        self.assertFalse(token.handle_closed)
        self.assertEqual(kernel.ReleaseMutex.calls, 1)
        self.assertEqual(kernel.CloseHandle.calls, 0)
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            self.assertIn(key, shared_memory_module._ACTIVE_WRITERS)

        kernel.ReleaseMutex.result = 1
        cleanup.retry()
        cleanup.retry()
        self.assertFalse(cleanup.pending)
        self.assertEqual(kernel.ReleaseMutex.calls, 2)
        self.assertEqual(kernel.CloseHandle.calls, 1)
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            self.assertNotIn(key, shared_memory_module._ACTIVE_WRITERS)
        with SharedMemoryWriter(self.path, generation=508) as retry:
            retry.write_frame(b"12345678")

    @unittest.skipUnless(os.name == "nt", "native constructor cleanup is Windows-specific")
    def test_constructor_close_handle_failure_retries_without_second_mutex_release(self):
        kernel = _FakeKernel(wait_result=0, release_result=1, close_result=0)
        with patch("shared_memory_v2.ctypes.WinDLL", return_value=kernel), patch(
            "shared_memory_v2.mmap.mmap", side_effect=RuntimeError("mmap init failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "mmap init failed") as caught:
                SharedMemoryWriter(self.path, generation=509)

        cleanup = caught.exception.producer_cleanup
        token = cleanup.token
        key = os.path.normcase(str(self.path.resolve()))
        self.assertTrue(token.mutex_released)
        self.assertFalse(token.handle_closed)
        self.assertEqual(kernel.ReleaseMutex.calls, 1)
        self.assertEqual(kernel.CloseHandle.calls, 1)
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            self.assertIn(key, shared_memory_module._ACTIVE_WRITERS)

        kernel.CloseHandle.result = 1
        cleanup.retry()
        self.assertFalse(cleanup.pending)
        self.assertEqual(kernel.ReleaseMutex.calls, 1)
        self.assertEqual(kernel.CloseHandle.calls, 2)
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            self.assertNotIn(key, shared_memory_module._ACTIVE_WRITERS)
        with SharedMemoryWriter(self.path, generation=510) as retry:
            retry.write_frame(b"12345678")

    @unittest.skipUnless(os.name == "nt", "native release assertions are Windows-specific")
    def test_native_mutex_release_failure_retains_state_and_retry_completes(self):
        kernel = _FakeKernel(release_result=0, close_result=1)
        key = "native-release-failure-test"
        token = shared_memory_module._ProducerToken(key, kernel, 123)
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            shared_memory_module._ACTIVE_WRITERS.add(key)
        with self.assertRaises(OSError):
            shared_memory_module._release_producer(token)
        self.assertFalse(token.mutex_released)
        self.assertFalse(token.handle_closed)
        self.assertEqual(kernel.ReleaseMutex.calls, 1)
        self.assertEqual(kernel.CloseHandle.calls, 0)
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            self.assertIn(key, shared_memory_module._ACTIVE_WRITERS)

        kernel.ReleaseMutex.result = 1
        shared_memory_module._release_producer(token)
        self.assertTrue(token.mutex_released)
        self.assertTrue(token.handle_closed)
        self.assertEqual(kernel.ReleaseMutex.calls, 2)
        self.assertEqual(kernel.CloseHandle.calls, 1)
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            self.assertNotIn(key, shared_memory_module._ACTIVE_WRITERS)

    @unittest.skipUnless(os.name == "nt", "native acquisition assertions are Windows-specific")
    def test_native_acquisition_failures_always_clear_registry_and_allow_retry(self):
        scenarios = []
        scenarios.append((OSError("WinDLL failed"), None, "WinDLL failed"))
        signature_kernel = _FakeKernel()
        signature_kernel.CreateMutexW = _SignatureRejectFunction(123)
        scenarios.append((None, signature_kernel, "signature setup failed"))
        scenarios.append((None, _FakeKernel(wait_result=0x102), "writer already active"))
        scenarios.append((None, _FakeKernel(wait_result=0xFFFFFFFF), "WaitForSingleObject"))
        wait_raises_kernel = _FakeKernel()
        wait_raises_kernel.WaitForSingleObject = _NativeFunction(
            error=RuntimeError("wait call raised")
        )
        scenarios.append((None, wait_raises_kernel, "wait call raised"))

        for index, (dll_error, kernel, message) in enumerate(scenarios, start=1):
            path = self.root / f"native-{index}.bin"
            replacement = patch(
                "shared_memory_v2.ctypes.WinDLL",
                side_effect=dll_error if dll_error is not None else None,
                return_value=kernel,
            )
            with self.subTest(message=message), replacement:
                with self.assertRaisesRegex((OSError, RuntimeError), message):
                    SharedMemoryWriter(path, generation=600 + index)
            key = os.path.normcase(str(path.resolve()))
            with shared_memory_module._ACTIVE_WRITERS_LOCK:
                self.assertNotIn(key, shared_memory_module._ACTIVE_WRITERS)
            with SharedMemoryWriter(path, generation=700 + index) as retry:
                retry.write_frame(b"12345678")

    @unittest.skipUnless(os.name == "nt", "native release assertions are Windows-specific")
    def test_native_close_handle_failure_retries_without_releasing_mutex_twice(self):
        kernel = _FakeKernel(release_result=1, close_result=0)
        key = "native-close-retry"
        token = shared_memory_module._ProducerToken(key, kernel, 123)
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            shared_memory_module._ACTIVE_WRITERS.add(key)
        with self.assertRaises(OSError):
            shared_memory_module._release_producer(token)
        self.assertTrue(token.mutex_released)
        self.assertFalse(token.handle_closed)
        self.assertEqual(kernel.ReleaseMutex.calls, 1)
        self.assertEqual(kernel.CloseHandle.calls, 1)
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            self.assertIn(key, shared_memory_module._ACTIVE_WRITERS)

        kernel.CloseHandle.result = 1
        shared_memory_module._release_producer(token)
        self.assertEqual(kernel.ReleaseMutex.calls, 1)
        self.assertEqual(kernel.CloseHandle.calls, 2)
        self.assertTrue(token.registry_cleared)
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            self.assertNotIn(key, shared_memory_module._ACTIVE_WRITERS)

    @unittest.skipUnless(os.name == "nt", "native release assertions are Windows-specific")
    def test_native_release_api_exception_retains_state_until_retry(self):
        kernel = _FakeKernel()
        kernel.ReleaseMutex = _NativeFunction(error=RuntimeError("release call raised"))
        kernel.CloseHandle = _NativeFunction(error=RuntimeError("close call raised"))
        key = "native-release-exceptions"
        token = shared_memory_module._ProducerToken(key, kernel, 123)
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            shared_memory_module._ACTIVE_WRITERS.add(key)
        with self.assertRaisesRegex(RuntimeError, "release call raised"):
            shared_memory_module._release_producer(token)
        self.assertEqual(kernel.CloseHandle.calls, 0)
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            self.assertIn(key, shared_memory_module._ACTIVE_WRITERS)

        kernel.ReleaseMutex = _NativeFunction(result=1)
        kernel.CloseHandle = _NativeFunction(result=1)
        shared_memory_module._release_producer(token)
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            self.assertNotIn(key, shared_memory_module._ACTIVE_WRITERS)

    @unittest.skipUnless(os.name == "nt", "native acquisition assertions are Windows-specific")
    def test_wait_failure_retains_close_handle_failure(self):
        kernel = _FakeKernel(wait_result=0xFFFFFFFF, close_result=0)
        with patch("shared_memory_v2.ctypes.WinDLL", return_value=kernel):
            with self.assertRaisesRegex(OSError, "WaitForSingleObject") as caught:
                SharedMemoryWriter(self.path, generation=710)
        self.assertEqual(kernel.CloseHandle.calls, 1)
        self.assertEqual(len(caught.exception.cleanup_errors), 1)
        key = os.path.normcase(str(self.path.resolve()))
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            self.assertNotIn(key, shared_memory_module._ACTIVE_WRITERS)

    @unittest.skipUnless(os.name == "nt", "native release assertions are Windows-specific")
    def test_close_flush_and_native_release_failures_are_aggregated_in_priority_order(self):
        writer = SharedMemoryWriter(self.path, generation=720)
        writer.write_frame(b"12345678")
        original_mapping = writer._mmap
        writer._mmap = _FlushFailMapping(original_mapping)
        real_token = writer._producer_token
        writer._producer_token = None
        shared_memory_module._release_producer(real_token)

        fake_key = "flush-release-combination"
        kernel = _FakeKernel(release_result=0, close_result=0)
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            shared_memory_module._ACTIVE_WRITERS.add(fake_key)
        writer._producer_token = shared_memory_module._ProducerToken(
            fake_key, kernel, 123
        )
        with self.assertRaisesRegex(OSError, "flush failed") as caught:
            writer.close()
        self.assertIsNone(writer._mmap)
        self.assertIsNone(writer._file)
        self.assertEqual(len(caught.exception.cleanup_errors), 1)
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            self.assertIn(fake_key, shared_memory_module._ACTIVE_WRITERS)
        kernel.ReleaseMutex.result = 1
        kernel.CloseHandle.result = 1
        shared_memory_module._release_producer(writer._producer_token)

    @unittest.skipUnless(os.name == "nt", "Windows mutex ownership is thread-specific")
    def test_writer_cross_thread_close_is_rejected_without_partial_cleanup(self):
        writer = SharedMemoryWriter(self.path, generation=721)
        writer.write_frame(b"12345678")
        errors = []

        def close_from_non_owner_thread():
            try:
                writer.close()
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=close_from_non_owner_thread)
        worker.start()
        worker.join()

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertRegex(str(errors[0]), "creator thread")
        self.assertIsNotNone(writer._mmap)
        self.assertIsNotNone(writer._file)
        self.assertIsNotNone(writer._producer_token)

        writer.close()
        self.assertIsNone(writer._mmap)
        self.assertIsNone(writer._file)
        self.assertIsNone(writer._producer_token)

    @unittest.skipUnless(os.name == "nt", "Windows mutex ownership is thread-specific")
    def test_two_phase_close_allows_worker_resources_but_owner_only_mutex_release(self):
        writer = SharedMemoryWriter(self.path, generation=722)
        writer.write_frame(b"12345678")
        resource_errors = []

        def close_resources_from_worker():
            try:
                writer.close_resources()
            except BaseException as error:
                resource_errors.append(error)

        worker = threading.Thread(target=close_resources_from_worker)
        worker.start()
        worker.join()

        self.assertEqual(resource_errors, [])
        self.assertIsNone(writer._mmap)
        self.assertIsNone(writer._file)
        self.assertIsNotNone(writer._producer_token)

        release_errors = []

        def release_mutex_from_worker():
            try:
                writer.release_producer()
            except BaseException as error:
                release_errors.append(error)

        worker = threading.Thread(target=release_mutex_from_worker)
        worker.start()
        worker.join()

        self.assertEqual(len(release_errors), 1)
        self.assertRegex(str(release_errors[0]), "creator thread")
        self.assertIsNotNone(writer._producer_token)

        writer.release_producer()
        self.assertIsNone(writer._producer_token)

    def test_close_combines_flush_resource_and_producer_failures(self):
        mapping = _FlushCloseFailMapping(OSError("mapping close failed"))
        file_object = _FileProbe(OSError("file close failed"))
        kernel = _FakeKernel(release_result=0, close_result=0)
        key = "all-close-failures"
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            shared_memory_module._ACTIVE_WRITERS.add(key)
        writer = SharedMemoryWriter.__new__(SharedMemoryWriter)
        writer._write_lock = threading.Lock()
        writer._mmap = mapping
        writer._file = file_object
        writer._dirty = True
        writer._producer_token = shared_memory_module._ProducerToken(key, kernel, 123)
        with self.assertRaisesRegex(OSError, "flush failed") as caught:
            writer.close()
        messages = [str(error) for error in caught.exception.cleanup_errors]
        self.assertEqual(len(messages), 3)
        self.assertIn("mapping close failed", messages[0])
        self.assertIn("file close failed", messages[1])
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            self.assertIn(key, shared_memory_module._ACTIVE_WRITERS)
        kernel.ReleaseMutex.result = 1
        kernel.CloseHandle.result = 1
        shared_memory_module._release_producer(writer._producer_token)

    def test_producer_release_is_primary_when_flush_and_resource_close_succeed(self):
        kernel = _FakeKernel(release_result=0, close_result=0)
        key = "producer-primary-failure"
        with shared_memory_module._ACTIVE_WRITERS_LOCK:
            shared_memory_module._ACTIVE_WRITERS.add(key)
        writer = SharedMemoryWriter.__new__(SharedMemoryWriter)
        writer._write_lock = threading.Lock()
        writer._mmap = _CloseProbe()
        writer._file = _CloseProbe()
        writer._dirty = False
        writer._producer_token = shared_memory_module._ProducerToken(key, kernel, 123)
        with self.assertRaises(OSError) as caught:
            writer.close()
        self.assertEqual(getattr(caught.exception, "cleanup_errors", ()), ())
        self.assertIsNone(writer._mmap)
        self.assertIsNone(writer._file)
        self.assertIsNotNone(writer._producer_token)
        kernel.ReleaseMutex.result = 1
        kernel.CloseHandle.result = 1
        writer.release_producer()
        self.assertIsNone(writer._producer_token)

    def test_close_retry_flushes_only_remaining_mmap_after_file_already_closed(self):
        mapping = _TransientFlushCloseMapping()
        file_object = _FileProbe()
        writer = SharedMemoryWriter.__new__(SharedMemoryWriter)
        writer._write_lock = threading.Lock()
        writer._mmap = mapping
        writer._file = file_object
        writer._dirty = True
        writer._writes_since_flush = 1
        writer._producer_token = None

        with self.assertRaisesRegex(OSError, "transient mmap flush failure") as caught:
            writer.close()
        self.assertEqual(len(caught.exception.cleanup_errors), 1)
        self.assertRegex(
            str(caught.exception.cleanup_errors[0]), "transient mmap close failure"
        )
        self.assertIs(writer._mmap, mapping)
        self.assertIsNone(writer._file)
        self.assertTrue(file_object.closed)
        self.assertTrue(writer._dirty)

        writer.close()
        self.assertEqual(mapping.flush_calls, 2)
        self.assertEqual(mapping.close_calls, 2)
        self.assertTrue(mapping.closed)
        self.assertIsNone(writer._mmap)
        self.assertIsNone(writer._file)
        self.assertFalse(writer._dirty)
        self.assertEqual(writer._writes_since_flush, 0)

    def test_independent_process_concurrent_stress_has_no_torn_frames(self):
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        start = context.Event()
        process = context.Process(
            target=_concurrent_writer,
            args=(str(self.path), ready, start, 300),
        )
        process.start()
        self.assertTrue(ready.wait(5))
        sequences = []
        with SharedMemoryReader(self.path, wait_timeout=5) as reader:
            start.set()
            deadline = time.monotonic() + 10
            while process.is_alive() and time.monotonic() < deadline:
                frame = reader.read(only_new=True)
                if frame is not None:
                    sequences.append(frame.sequence)
            frame = reader.read(only_new=True)
            if frame is not None:
                sequences.append(frame.sequence)
        process.join(5)
        self.assertEqual(process.exitcode, 0)
        self.assertGreater(len(sequences), 5)
        self.assertEqual(sequences, sorted(set(sequences)))

    def test_cli_missing_path_and_protocol_error_exit_codes(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(reader_cli.main([str(self.root / "missing"), "--wait", "0"]), 2)
        self.assertIn("未找到共享文件", output.getvalue())
        self.path.write_bytes(bytes(FRAME_SIZE))
        self._overwrite(COMMIT_SEQUENCE_OFFSET, struct.pack("<Q", 1))
        with contextlib.redirect_stdout(output):
            self.assertEqual(reader_cli.main([str(self.path), "--wait", "0"]), 3)
        self.assertIn("共享文件协议错误", output.getvalue())
        self._assert_no_mojibake(output.getvalue())

    def test_cli_help_and_keyboard_interrupt_messages_are_valid_utf8_chinese(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as exit_info:
            reader_cli._arguments(["--help"])
        self.assertEqual(exit_info.exception.code, 0)
        self.assertIn("读取 EMG SharedMemory v2 共享文件", output.getvalue())
        self.assertIn("自动探测", output.getvalue())
        self._assert_no_mojibake(output.getvalue())

        output = io.StringIO()
        with patch.object(reader_cli, "SharedMemoryReader", side_effect=KeyboardInterrupt):
            with contextlib.redirect_stdout(output):
                self.assertEqual(reader_cli.main([str(self.path), "--wait", "0"]), 0)
        self.assertIn("已停止读取。", output.getvalue())
        self._assert_no_mojibake(output.getvalue())

    def test_cli_process_emits_utf8_help_when_stdout_is_redirected(self):
        result = subprocess.run(
            [sys.executable, str(Path(reader_cli.__file__)), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout.decode("utf-8")
        self.assertIn("读取 EMG SharedMemory v2 共享文件", output)
        self._assert_no_mojibake(output)

    def test_configure_console_encoding_skips_stringio_streams(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(reader_cli.sys, "stdout", stdout), patch.object(
            reader_cli.sys, "stderr", stderr
        ):
            reader_cli._configure_console_encoding()
            print("标准输出中文", file=reader_cli.sys.stdout)
            print("错误输出中文", file=reader_cli.sys.stderr)
        self.assertEqual(stdout.getvalue(), "标准输出中文\n")
        self.assertEqual(stderr.getvalue(), "错误输出中文\n")

    def test_configure_console_encoding_reconfigures_real_text_capture_streams(self):
        stdout_bytes = io.BytesIO()
        stderr_bytes = io.BytesIO()
        stdout = io.TextIOWrapper(stdout_bytes, encoding="gbk")
        stderr = io.TextIOWrapper(stderr_bytes, encoding="gbk")
        with patch.object(reader_cli.sys, "stdout", stdout), patch.object(
            reader_cli.sys, "stderr", stderr
        ):
            reader_cli._configure_console_encoding()
            self.assertEqual(reader_cli.sys.stdout.encoding.lower(), "utf-8")
            self.assertEqual(reader_cli.sys.stderr.encoding.lower(), "utf-8")
            print("捕获输出中文", file=reader_cli.sys.stdout)
            print("捕获错误中文", file=reader_cli.sys.stderr)
            stdout.flush()
            stderr.flush()
        self.assertEqual(stdout_bytes.getvalue().decode("utf-8").splitlines(), ["捕获输出中文"])
        self.assertEqual(stderr_bytes.getvalue().decode("utf-8").splitlines(), ["捕获错误中文"])

    def _assert_no_mojibake(self, text):
        for marker in ("\ufffd", "锛", "杈", "锷", "鍏"):
            self.assertNotIn(marker, text)

    def test_cli_formats_device_time_as_raw_not_wall_clock(self):
        with SharedMemoryWriter(self.path, generation=60) as writer:
            frame = writer.write_frame(
                b"12345678",
                host_wall_timestamp_ns=1_700_000_000_000_000_000,
                device_time_ticks=1234,
                connection_generation=0,
            )
        rendered = reader_cli._format_frame(frame)
        self.assertIn("device_ticks=1234", rendered)
        self.assertIn("connection_generation=0", rendered)
        self.assertNotIn("1970-", rendered)

    def test_cli_diagnostics_reports_transport_anomalies_and_negative_age(self):
        diagnostics = FrameDiagnostics()
        with SharedMemoryWriter(self.path, generation=901) as writer:
            first = writer.write_frame(
                b"12345678",
                host_wall_timestamp_ns=2_000_000_000,
                host_receive_index=1,
                device_packet_sequence=5,
            )
            duplicate = writer.write_frame(
                b"abcdefgh",
                host_wall_timestamp_ns=2_000_000_100,
                host_receive_index=3,
                device_packet_sequence=5,
                flags=FLAG_DISCONNECTED | FLAG_STALE | FLAG_OVERFLOW,
            )
        self.assertEqual(diagnostics.observe(first, monotonic_ns=10), [])
        events = diagnostics.observe(duplicate, monotonic_ns=20)
        self.assertIn("device_packet_duplicate:5", events)
        self.assertIn("host_receive_gap:1", events)
        summary = diagnostics.summary(wall_time_ns=1_000_000_000, monotonic_ns=30)
        self.assertIn("age_state=clock_skew", summary)
        self.assertIn("age_ms=-1000.0", summary)
        self.assertIn("stall_ms=0.0", summary)
        self.assertIn("disconnected=1", summary)
        self.assertIn("stale=1", summary)
        self.assertIn("overflow=1", summary)

    def test_cli_diagnostics_reports_gap_late_fill_and_generation_restart(self):
        diagnostics = FrameDiagnostics()
        frames = []
        with SharedMemoryWriter(self.path, generation=910) as writer:
            for packet in (1, 4, 2):
                frames.append(writer.write_frame(b"12345678", device_packet_sequence=packet))
        for frame in frames:
            diagnostics.observe(frame)
        with SharedMemoryWriter(self.path, generation=911) as writer:
            restarted = writer.write_frame(b"abcdefgh", device_packet_sequence=0)
        events = diagnostics.observe(restarted)
        self.assertEqual(diagnostics.device_packets.gaps, 2)
        self.assertEqual(diagnostics.device_packets.late_fills, 1)
        self.assertEqual(diagnostics.device_packets.unrecovered_gaps, 1)
        self.assertIn("generation_restart:910->911", events)

    def test_connection_generation_change_resets_device_sequence_diagnostics(self):
        diagnostics = FrameDiagnostics()
        with SharedMemoryWriter(self.path, generation=912) as writer:
            before = writer.write_frame(
                b"12345678",
                connection_generation=40,
                device_packet_sequence=900,
            )
            after = writer.write_frame(
                b"abcdefgh",
                connection_generation=41,
                device_packet_sequence=0,
            )
        diagnostics.observe(before)
        events = diagnostics.observe(after)
        self.assertIn("connection_generation_change:40->41", events)
        self.assertNotIn("device_packet_out_of_order:900->0", events)
        self.assertEqual(after.generation, before.generation)
        self.assertNotEqual(after.connection_generation, before.connection_generation)

    def test_connection_generation_validity_transitions_reset_device_diagnostics(self):
        diagnostics = FrameDiagnostics()
        with SharedMemoryWriter(self.path, generation=913) as writer:
            known = writer.write_frame(
                b"12345678",
                connection_generation=41,
                device_packet_sequence=900,
            )
            unknown = writer.write_frame(
                b"abcdefgh",
                connection_generation=None,
                device_packet_sequence=0,
            )
            known_again = writer.write_frame(
                b"ABCDEFGH",
                connection_generation=42,
                device_packet_sequence=0,
            )
        diagnostics.observe(known)
        known_to_unknown = diagnostics.observe(unknown)
        unknown_to_known = diagnostics.observe(known_again)
        self.assertIn(
            "connection_generation_change:41->unknown", known_to_unknown
        )
        self.assertNotIn("device_packet_out_of_order:900->0", known_to_unknown)
        self.assertIn(
            "connection_generation_change:unknown->42", unknown_to_known
        )
        self.assertNotIn("device_packet_duplicate:0", unknown_to_known)

    def test_uint64_device_counter_wrap_and_cross_wrap_gap(self):
        diagnostics = FrameDiagnostics()
        with SharedMemoryWriter(self.path, generation=920) as writer:
            maximum = writer.write_frame(
                b"12345678", device_packet_sequence=(1 << 64) - 1
            )
            zero = writer.write_frame(b"abcdefgh", device_packet_sequence=0)
        diagnostics.observe(maximum)
        events = diagnostics.observe(zero)
        self.assertIn(
            f"device_packet_wrap:{(1 << 64) - 1}->0", events
        )
        self.assertEqual(diagnostics.device_packets.gaps, 0)

        second = FrameDiagnostics()
        with SharedMemoryWriter(self.path, generation=921) as writer:
            near_max = writer.write_frame(
                b"12345678", device_packet_sequence=(1 << 64) - 2
            )
            one = writer.write_frame(b"abcdefgh", device_packet_sequence=1)
        second.observe(near_max)
        events = second.observe(one)
        self.assertIn("device_packet_gap:2", events)
        self.assertEqual(second.device_packets.unrecovered_gaps, 2)

    def test_nonconsecutive_duplicate_and_late_gap_fill_are_distinct(self):
        duplicate_diagnostics = FrameDiagnostics()
        with SharedMemoryWriter(self.path, generation=930) as writer:
            duplicate_frames = [
                writer.write_frame(b"12345678", device_packet_sequence=value)
                for value in (1, 2, 1)
            ]
        duplicate_events = []
        for frame in duplicate_frames:
            duplicate_events.extend(duplicate_diagnostics.observe(frame))
        self.assertIn("device_packet_duplicate:1", duplicate_events)
        self.assertEqual(duplicate_diagnostics.device_packets.out_of_order, 0)

        late_diagnostics = FrameDiagnostics()
        with SharedMemoryWriter(self.path, generation=931) as writer:
            late_frames = [
                writer.write_frame(b"abcdefgh", device_packet_sequence=value)
                for value in (1, 3, 2, 0)
            ]
        late_events = []
        for frame in late_frames:
            late_events.extend(late_diagnostics.observe(frame))
        self.assertIn("device_packet_gap:1", late_events)
        self.assertIn("device_packet_late_fill:2", late_events)
        self.assertIn("device_packet_out_of_order:3->0", late_events)
        self.assertEqual(late_diagnostics.device_packets.unrecovered_gaps, 0)

    def test_host_receive_duplicate_and_backward_and_generation_reset(self):
        diagnostics = FrameDiagnostics()
        with SharedMemoryWriter(self.path, generation=940) as writer:
            frames = [
                writer.write_frame(b"12345678", host_receive_index=value)
                for value in (1, 2, 1, 0)
            ]
        events = []
        for frame in frames:
            events.extend(diagnostics.observe(frame))
        self.assertIn("host_receive_duplicate:1", events)
        self.assertIn("host_receive_out_of_order:2->0", events)

        with SharedMemoryWriter(self.path, generation=941) as writer:
            restarted = writer.write_frame(b"abcdefgh", host_receive_index=0)
        restart_events = diagnostics.observe(restarted)
        self.assertIn("generation_restart:940->941", restart_events)
        self.assertNotIn("host_receive_out_of_order:2->0", restart_events)

    def _write_valid(self):
        with SharedMemoryWriter(self.path, generation=99) as writer:
            writer.write_frame(b"12345678", sequence=7, device_packet_sequence=8)

    def _set_sequence_and_generation(self, *, sequence, generation):
        self._overwrite(32, struct.pack("<Q", generation))
        self._overwrite(SEQUENCE_OFFSET, struct.pack("<Q", sequence))
        self._overwrite(COMMIT_SEQUENCE_OFFSET, struct.pack("<Q", sequence))
        self._recalculate_crc()

    def _recalculate_crc(self):
        raw = bytearray(self.path.read_bytes())
        valid_length = struct.unpack_from("<I", raw, 80)[0]
        minor = struct.unpack_from("<H", raw, 10)[0]
        checksum = shared_memory_module._content_crc(
            raw,
            raw[PAYLOAD_OFFSET : PAYLOAD_OFFSET + valid_length],
            minor,
        )
        self._overwrite(CONTENT_CRC32_OFFSET, struct.pack("<I", checksum))

    def _overwrite(self, offset, value):
        with open(self.path, "r+b") as file_object:
            with mmap.mmap(file_object.fileno(), FRAME_SIZE) as mapping:
                mapping[offset : offset + len(value)] = value
                mapping.flush()


def test_live_control_reader_rejects_replay_and_accepts_real_frame(tmp_path):
    path = tmp_path / "control.bin"
    with SharedMemoryWriter(path) as writer:
        writer.write_frame(
            bytes(range(8)), flags=FLAG_REPLAY | FLAG_SYNTHETIC_TIME
        )
        with SharedMemoryReader(path, expected_connection_generation=1) as reader:
            with unittest.TestCase().assertRaisesRegex(
                SharedMemoryProtocolError, "live control"
            ):
                reader.read_for_live_control()
    with SharedMemoryWriter(path) as writer:
        writer.write_frame(bytes(range(8)), connection_generation=1)
        with SharedMemoryReader(path, expected_connection_generation=1) as reader:
            frame = reader.read_for_live_control()
            assert frame is not None and frame.is_live_control_eligible(1)


def test_live_control_gate_rejects_every_quality_and_connection_boundary(tmp_path):
    unsafe_flags = (
        FLAG_OVERFLOW,
        shared_memory_module.FLAG_CRC_ERROR,
        FLAG_STALE,
        FLAG_DISCONNECTED,
        FLAG_REPLAY,
        FLAG_SYNTHETIC_TIME,
    )
    for index, flags in enumerate(unsafe_flags):
        path = tmp_path / f"unsafe-{index}.bin"
        with SharedMemoryWriter(path) as writer:
            writer.write_frame(bytes(range(8)), flags=flags, connection_generation=9)
            with SharedMemoryReader(path, expected_connection_generation=9) as reader:
                with unittest.TestCase().assertRaises(SharedMemoryProtocolError):
                    reader.read_for_live_control()

    missing = tmp_path / "missing-generation.bin"
    with SharedMemoryWriter(missing) as writer:
        writer.write_frame(bytes(range(8)))
        with SharedMemoryReader(missing, expected_connection_generation=9) as reader:
            with unittest.TestCase().assertRaises(SharedMemoryProtocolError):
                reader.read_for_live_control()

    wrong = tmp_path / "wrong-generation.bin"
    with SharedMemoryWriter(wrong) as writer:
        writer.write_frame(bytes(range(8)), connection_generation=8)
        with SharedMemoryReader(wrong, expected_connection_generation=9) as reader:
            with unittest.TestCase().assertRaises(SharedMemoryProtocolError):
                reader.read_for_live_control()

    channels = tmp_path / "wrong-channels.bin"
    with SharedMemoryWriter(channels) as writer:
        writer.write_frame(bytes(range(7)), channel_count=7, connection_generation=9)
        with SharedMemoryReader(channels, expected_connection_generation=9) as reader:
            with unittest.TestCase().assertRaises(SharedMemoryProtocolError):
                reader.read_for_live_control()

    unbound = tmp_path / "unbound.bin"
    with SharedMemoryWriter(unbound) as writer:
        writer.write_frame(bytes(range(8)), connection_generation=9)
        with SharedMemoryReader(unbound) as reader:
            with unittest.TestCase().assertRaisesRegex(
                SharedMemoryProtocolError, "expected_connection_generation"
            ):
                reader.read_for_live_control()


if __name__ == "__main__":
    unittest.main()
