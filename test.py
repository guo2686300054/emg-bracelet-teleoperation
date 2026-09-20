# -*- coding: utf-8 -*-
"""连续读取 EMG SharedMemory v2，并周期输出链路诊断。"""

import argparse
import sys
import time
from collections import deque
from datetime import datetime, timezone

from shared_memory_v2 import (
    DEFAULT_SHARED_FILENAME,
    FLAG_CONNECTION_GENERATION_VALID,
    FLAG_DEVICE_PACKET_SEQUENCE_VALID,
    FLAG_DEVICE_SAMPLE_COUNTER_VALID,
    FLAG_DEVICE_TIME_VALID,
    FLAG_DISCONNECTED,
    FLAG_HOST_RECEIVE_INDEX_VALID,
    FLAG_OVERFLOW,
    FLAG_STALE,
    SharedMemoryProtocolError,
    SharedMemoryReader,
)


UINT64_MODULUS = 1 << 64
UINT64_HALF_RANGE = 1 << 63


def _configure_console_encoding():
    """Emit stable UTF-8 text when Windows stdout/stderr are redirected."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


class _ModularCounterTracker:
    """Track a uint64 counter with bounded duplicate and missing histories."""

    def __init__(self, label, history_limit=4096):
        self.label = label
        self.history_limit = history_limit
        self.high_water = None
        self._seen_queue = deque()
        self._seen = set()
        self._missing_queue = deque()
        self._missing = set()
        self.duplicates = 0
        self.gaps = 0
        self.late_fills = 0
        self.out_of_order = 0
        self.wraps = 0

    @property
    def unrecovered_gaps(self):
        return self.gaps - self.late_fills

    def reset_generation(self):
        self.high_water = None
        self._seen_queue.clear()
        self._seen.clear()
        self._missing_queue.clear()
        self._missing.clear()

    def observe(self, value):
        events = []
        if value in self._missing:
            self._missing.remove(value)
            self._missing_queue.remove(value)
            self.late_fills += 1
            self._remember_seen(value)
            return [f"{self.label}_late_fill:{value}"]
        if value in self._seen:
            self.duplicates += 1
            return [f"{self.label}_duplicate:{value}"]
        if self.high_water is None:
            self.high_water = value
            self._remember_seen(value)
            return events
        previous = self.high_water
        delta = (value - previous) % UINT64_MODULUS
        if 0 < delta < UINT64_HALF_RANGE:
            if value < previous:
                self.wraps += 1
                events.append(f"{self.label}_wrap:{previous}->{value}")
            if delta > 1:
                gap = delta - 1
                self.gaps += gap
                events.append(f"{self.label}_gap:{gap}")
                bounded = min(gap, self.history_limit)
                for step in range(1, bounded + 1):
                    self._remember_missing((previous + step) % UINT64_MODULUS)
            self.high_water = value
            self._remember_seen(value)
            return events
        self.out_of_order += 1
        self._remember_seen(value)
        return [f"{self.label}_out_of_order:{previous}->{value}"]

    def _remember_seen(self, value):
        if value in self._seen:
            return
        self._seen.add(value)
        self._seen_queue.append(value)
        while len(self._seen_queue) > self.history_limit:
            self._seen.discard(self._seen_queue.popleft())

    def _remember_missing(self, value):
        if value in self._missing:
            return
        self._missing.add(value)
        self._missing_queue.append(value)
        while len(self._missing) > self.history_limit:
            self._missing.discard(self._missing_queue.popleft())


class FrameDiagnostics:
    """Accumulate transport diagnostics without treating equal payloads as duplicates."""

    def __init__(self):
        self.frames = 0
        self.generation_restarts = 0
        self.disconnected_frames = 0
        self.stale_frames = 0
        self.overflow_frames = 0
        self._generation = None
        self._connection_generation = None
        self._shared_sequence = None
        self.device_packets = _ModularCounterTracker("device_packet")
        self.host_receives = _ModularCounterTracker("host_receive")
        self._last_frame = None
        self._last_observed_monotonic_ns = None

    def observe(self, frame, *, monotonic_ns=None):
        events = []
        now_mono = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
        if self._generation is not None and frame.generation != self._generation:
            self.generation_restarts += 1
            events.append(f"generation_restart:{self._generation}->{frame.generation}")
            self.device_packets.reset_generation()
            self.host_receives.reset_generation()
        connection_generation = (
            frame.connection_generation
            if frame.flags & FLAG_CONNECTION_GENERATION_VALID
            else None
        )
        if (
            self._last_frame is not None
            and connection_generation != self._connection_generation
        ):
            previous_label = (
                self._connection_generation
                if self._connection_generation is not None
                else "unknown"
            )
            current_label = (
                connection_generation
                if connection_generation is not None
                else "unknown"
            )
            events.append(
                "connection_generation_change:"
                f"{previous_label}->{current_label}"
            )
            self.device_packets.reset_generation()
        if frame.flags & FLAG_DEVICE_PACKET_SEQUENCE_VALID:
            events.extend(self.device_packets.observe(frame.device_packet_sequence))
        if frame.flags & FLAG_HOST_RECEIVE_INDEX_VALID:
            events.extend(self.host_receives.observe(frame.host_receive_index))
        for flag, attribute, label in (
            (FLAG_DISCONNECTED, "disconnected_frames", "DISCONNECTED"),
            (FLAG_STALE, "stale_frames", "STALE"),
            (FLAG_OVERFLOW, "overflow_frames", "OVERFLOW"),
        ):
            if frame.flags & flag:
                setattr(self, attribute, getattr(self, attribute) + 1)
                events.append(label)
        self.frames += 1
        self._generation = frame.generation
        self._connection_generation = connection_generation
        self._shared_sequence = frame.sequence
        self._last_frame = frame
        self._last_observed_monotonic_ns = now_mono
        return events

    def summary(self, *, wall_time_ns=None, monotonic_ns=None):
        now_wall = time.time_ns() if wall_time_ns is None else wall_time_ns
        now_mono = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
        if self._last_frame is None:
            return "status frames=0 shared=waiting"
        age_ms = (now_wall - self._last_frame.host_wall_timestamp_ns) / 1_000_000
        stall_ms = (now_mono - self._last_observed_monotonic_ns) / 1_000_000
        clock_state = "clock_skew" if age_ms < 0 else "ok"
        return (
            f"status frames={self.frames} generation={self._generation} "
            f"connection_generation={self._connection_generation if self._connection_generation is not None else 'unknown'} "
            f"shared_seq={self._shared_sequence} stall_ms={stall_ms:.1f} "
            f"age_ms={age_ms:.1f} age_state={clock_state} "
            f"restarts={self.generation_restarts} device_duplicates={self.device_packets.duplicates} "
            f"device_gaps={self.device_packets.gaps} device_late={self.device_packets.late_fills} "
            f"device_unrecovered={self.device_packets.unrecovered_gaps} "
            f"device_out_of_order={self.device_packets.out_of_order} device_wraps={self.device_packets.wraps} "
            f"host_gaps={self.host_receives.gaps} host_duplicates={self.host_receives.duplicates} "
            f"host_out_of_order={self.host_receives.out_of_order} "
            f"disconnected={self.disconnected_frames} "
            f"stale={self.stale_frames} overflow={self.overflow_frames}"
        )


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description="读取 EMG SharedMemory v2 共享文件")
    parser.add_argument(
        "path",
        nargs="?",
        help=f"v2 共享文件路径；省略时自动探测 {DEFAULT_SHARED_FILENAME}",
    )
    parser.add_argument("--wait", type=float, default=10.0, help="等待共享文件出现的秒数")
    parser.add_argument("--interval", type=float, default=0.1, help="轮询间隔秒数")
    parser.add_argument("--summary-interval", type=float, default=5.0, help="状态汇总间隔秒数")
    return parser.parse_args(argv)


def _optional(frame, flag, value):
    return str(value) if frame.flags & flag else "unknown"


def _format_frame(frame, *, now_ns=None):
    wall_time = datetime.fromtimestamp(frame.host_wall_timestamp_ns / 1_000_000_000, timezone.utc)
    now = time.time_ns() if now_ns is None else now_ns
    age_ms = (now - frame.host_wall_timestamp_ns) / 1_000_000
    return (
        f"[{wall_time.astimezone():%Y-%m-%d %H:%M:%S.%f}] generation={frame.generation} "
        f"connection_generation={_optional(frame, FLAG_CONNECTION_GENERATION_VALID, frame.connection_generation)} "
        f"seq={frame.sequence} host_receive="
        f"{_optional(frame, FLAG_HOST_RECEIVE_INDEX_VALID, frame.host_receive_index)} "
        f"device_packet={_optional(frame, FLAG_DEVICE_PACKET_SEQUENCE_VALID, frame.device_packet_sequence)} "
        f"device_sample={_optional(frame, FLAG_DEVICE_SAMPLE_COUNTER_VALID, frame.device_sample_counter)} "
        f"device_ticks={_optional(frame, FLAG_DEVICE_TIME_VALID, frame.device_time_ticks)} "
        f"age_ms={age_ms:.1f} channels={frame.channel_count} values={frame.unpack_samples()}"
    )


def main(argv=None):
    args = _arguments(argv)
    diagnostics = FrameDiagnostics()
    try:
        with SharedMemoryReader(args.path, wait_timeout=args.wait) as reader:
            print(f"已打开共享文件：{reader.shared_file_path}")
            print("开始读取新帧，按 Ctrl+C 停止。")
            next_summary = time.monotonic() + max(0.001, args.summary_interval)
            while True:
                frame = reader.read(only_new=True)
                if frame is not None:
                    print(_format_frame(frame))
                    events = diagnostics.observe(frame)
                    if events:
                        print("diagnostic " + " ".join(events))
                now = time.monotonic()
                if now >= next_summary:
                    print(diagnostics.summary())
                    next_summary = now + max(0.001, args.summary_interval)
                time.sleep(max(0.001, args.interval))
    except FileNotFoundError as exc:
        print(f"未找到共享文件：{exc}")
        return 2
    except SharedMemoryProtocolError as exc:
        print(f"共享文件协议错误：{exc}")
        return 3
    except KeyboardInterrupt:
        print("\n已停止读取。")
        return 0


if __name__ == "__main__":
    _configure_console_encoding()
    raise SystemExit(main())
