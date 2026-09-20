"""Transport-independent domain contracts for EMG acquisition."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum, IntFlag
from numbers import Real
from typing import Any, Mapping, Optional, Sequence

UINT64_MAX = (1 << 64) - 1

LOGICAL16_PROTOCOL = "logical16_odd_bytes_v1"
PADDED28_PROTOCOL = "wire28_logical16_zero_suffix_v1"


class HandSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


def _contract_text(name: str, value: Any, *, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"invalid {name}")
    return cleaned


class QualityFlags(IntFlag):
    NONE = 0
    VALID = 1 << 0
    HOST_WALL_TIME_VALID = 1 << 2
    HOST_MONOTONIC_VALID = 1 << 3
    HOST_RECEIVE_INDEX_VALID = 1 << 4
    DEVICE_PACKET_SEQUENCE_VALID = 1 << 5
    DEVICE_SAMPLE_COUNTER_VALID = 1 << 6
    DEVICE_TIME_VALID = 1 << 7
    DISCONNECTED = 1 << 8
    OVERFLOW = 1 << 9
    CRC_ERROR = 1 << 10
    STALE = 1 << 11
    DUPLICATE_PACKET = 1 << 16
    OUT_OF_ORDER_PACKET = 1 << 17
    PACKET_GAP = 1 << 18


KNOWN_QUALITY_MASK = sum(int(flag) for flag in QualityFlags)
SEQUENCE_QUALITY_FLAGS = (
    QualityFlags.DUPLICATE_PACKET
    | QualityFlags.OUT_OF_ORDER_PACKET
    | QualityFlags.PACKET_GAP
)

# Explicit adapter contract. These are wire-v2 values, not imports from transport code.
SHARED_V2_FLAG_MAP = {
    QualityFlags.VALID: 1 << 0,
    QualityFlags.HOST_WALL_TIME_VALID: 1 << 2,
    QualityFlags.HOST_MONOTONIC_VALID: 1 << 3,
    QualityFlags.HOST_RECEIVE_INDEX_VALID: 1 << 4,
    QualityFlags.DEVICE_PACKET_SEQUENCE_VALID: 1 << 5,
    QualityFlags.DEVICE_SAMPLE_COUNTER_VALID: 1 << 6,
    QualityFlags.DEVICE_TIME_VALID: 1 << 7,
    QualityFlags.DISCONNECTED: 1 << 8,
    QualityFlags.OVERFLOW: 1 << 9,
    QualityFlags.CRC_ERROR: 1 << 10,
    QualityFlags.STALE: 1 << 11,
}


def validate_quality_flags(value: QualityFlags) -> QualityFlags:
    if not isinstance(value, QualityFlags):
        raise ValueError("quality_flags must be QualityFlags")
    if int(value) & ~KNOWN_QUALITY_MASK:
        raise ValueError("quality_flags contains unknown bits")
    return value


def to_shared_v2_flags(value: QualityFlags) -> int:
    """Map semantic flags to the subset represented by shared-memory v2."""
    validate_quality_flags(value)
    result = 0
    for semantic, wire_value in SHARED_V2_FLAG_MAP.items():
        if value & semantic:
            result |= wire_value
    return result


def from_shared_v2_flags(value:int)->QualityFlags:
    if not isinstance(value,int) or isinstance(value,bool) or value<0: raise ValueError("wire flags must be a non-negative integer")
    wire_mask=sum(SHARED_V2_FLAG_MAP.values())
    if value&~wire_mask: raise ValueError("wire flags contain unsupported bits")
    result=QualityFlags.NONE
    for semantic,wire_value in SHARED_V2_FLAG_MAP.items():
        if value&wire_value:result|=semantic
    return result


_DEVICE_KEY = re.compile(r"dev-[0-9a-f]{32}\Z", re.ASCII)


@dataclass(frozen=True)
class DeviceKey:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not _DEVICE_KEY.fullmatch(self.value):
            raise ValueError("device key must be opaque dev-<32 lowercase hex>")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class RateDescriptor:
    value_hz: Optional[float] = None
    source_kind: str = "unknown"
    evidence_ref: str = "unknown"
    confirmed: bool = False

    def __post_init__(self) -> None:
        if self.value_hz is not None and (
            isinstance(self.value_hz, bool)
            or not isinstance(self.value_hz, (int, float))
            or not math.isfinite(self.value_hz)
            or self.value_hz <= 0
        ):
            raise ValueError("rate must be a finite positive number")
        source_kind = _contract_text("source_kind", self.source_kind)
        evidence_ref = _contract_text("evidence_ref", self.evidence_ref)
        if not isinstance(self.confirmed, bool):
            raise ValueError("confirmed must be bool")
        if self.confirmed and (
            self.value_hz is None
            or source_kind.casefold() == "unknown"
            or evidence_ref.casefold() == "unknown"
        ):
            raise ValueError("confirmed rate requires evidence")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RateDescriptor":
        if not isinstance(value, Mapping):
            raise ValueError("rate descriptor must be an object")
        expected = {"value_hz", "source_kind", "evidence_ref", "confirmed"}
        if set(value) != expected:
            raise ValueError("rate descriptor fields are invalid")
        return cls(
            value_hz=value["value_hz"],
            source_kind=value["source_kind"],
            evidence_ref=value["evidence_ref"],
            confirmed=value["confirmed"],
        )


@dataclass(frozen=True)
class NotificationPacketProtocol:
    """Versioned BLE notification envelope; never guesses or truncates a packet."""

    mode: str = LOGICAL16_PROTOCOL
    wire_packet_size: int = 16
    logical_packet_size: int = 16
    padding_rule: str = "none"
    evidence_ref: str = "unknown"

    def __post_init__(self) -> None:
        mode = _contract_text("packet protocol mode", self.mode)
        evidence = _contract_text("packet protocol evidence_ref", self.evidence_ref)
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (self.wire_packet_size, self.logical_packet_size)
        ):
            raise ValueError("packet sizes must be integers")
        contracts = {
            LOGICAL16_PROTOCOL: (16, 16, "none"),
            PADDED28_PROTOCOL: (28, 16, "zero_suffix"),
        }
        expected = contracts.get(mode)
        actual = (self.wire_packet_size, self.logical_packet_size, self.padding_rule)
        if expected is None or actual != expected:
            raise ValueError("unsupported or inconsistent packet protocol contract")
        if mode == PADDED28_PROTOCOL and evidence.casefold() in {
            "unknown",
            "built_in_logical16_contract",
        }:
            raise ValueError("28-byte packet protocol requires evidence_ref")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NotificationPacketProtocol":
        if not isinstance(value, Mapping):
            raise ValueError("notification_packet_protocol must be an object")
        expected = {
            "mode",
            "wire_packet_size",
            "logical_packet_size",
            "padding_rule",
            "evidence_ref",
        }
        if set(value) != expected:
            raise ValueError("notification_packet_protocol fields are invalid")
        return cls(**{name: value[name] for name in expected})

    def extract_logical_payload(self, payload: bytes) -> bytes:
        packet = bytes(payload)
        if len(packet) != self.wire_packet_size:
            raise ValueError(
                f"{self.mode} notification must be exactly "
                f"{self.wire_packet_size} bytes: {len(packet)}"
            )
        if self.padding_rule == "zero_suffix" and any(
            packet[self.logical_packet_size :]
        ):
            raise ValueError("28-byte notification has non-zero suffix padding")
        return packet[: self.logical_packet_size]


@dataclass(frozen=True)
class SignalChain:
    sample_format: str = "unknown"
    unit: str = "unknown"
    adc_bit_width: Optional[int] = None
    scaling: str = "unknown"
    offset: str = "unknown"
    filter: str = "unknown"
    rectification: str = "unknown"
    normalization: str = "unknown"
    quantization: str = "unknown"

    def __post_init__(self) -> None:
        for name in (
            "sample_format",
            "unit",
            "scaling",
            "offset",
            "filter",
            "rectification",
            "normalization",
            "quantization",
        ):
            _contract_text(name, getattr(self, name))
        if self.sample_format not in {"unknown", "uint8", "int16", "float32"}:
            raise ValueError("unsupported sample_format")
        if self.adc_bit_width is not None and (
            not isinstance(self.adc_bit_width, int)
            or isinstance(self.adc_bit_width, bool)
            or not 1 <= self.adc_bit_width <= 64
        ):
            raise ValueError("invalid adc_bit_width")
        output_width = {"uint8": 8, "int16": 16}.get(self.sample_format)
        if (
            self.adc_bit_width is not None
            and output_width is not None
            and self.adc_bit_width > output_width
            and self.quantization.strip().casefold() == "unknown"
        ):
            raise ValueError("narrower output format requires an explicit quantization rule")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SignalChain":
        if not isinstance(value, Mapping):
            raise ValueError("signal_chain must be an object")
        required = {
            "sample_format", "unit", "adc_bit_width", "scaling", "offset",
            "filter", "rectification", "normalization",
        }
        allowed = required | {"quantization"}
        if not required.issubset(value) or set(value) - allowed:
            raise ValueError("signal_chain fields are invalid")
        return cls(**{name: value[name] for name in required}, quantization=value.get("quantization", "unknown"))


@dataclass(frozen=True)
class AcquisitionMetadata:
    sample_rate: RateDescriptor = field(default_factory=RateDescriptor)
    adc_rate: RateDescriptor = field(default_factory=RateDescriptor)
    device_output_rate: RateDescriptor = field(default_factory=RateDescriptor)
    host_observed_rate: RateDescriptor = field(default_factory=RateDescriptor)
    device_tick_rate: RateDescriptor = field(default_factory=RateDescriptor)
    device_tick_modulus: Optional[int] = None
    device_tick_modulus_source: str = "unknown"
    device_sequence_modulus: Optional[int] = None
    device_sequence_modulus_source: str = "unknown"
    signal_chain: SignalChain = field(default_factory=SignalChain)
    notification_packet_protocol: NotificationPacketProtocol = field(
        default_factory=NotificationPacketProtocol
    )

    def __post_init__(self) -> None:
        for name in (
            "sample_rate",
            "adc_rate",
            "device_output_rate",
            "host_observed_rate",
            "device_tick_rate",
        ):
            if not isinstance(getattr(self, name), RateDescriptor):
                raise ValueError(f"{name} must be RateDescriptor")
        if not isinstance(self.signal_chain, SignalChain):
            raise ValueError("signal_chain must be SignalChain")
        if not isinstance(self.notification_packet_protocol, NotificationPacketProtocol):
            raise ValueError(
                "notification_packet_protocol must be NotificationPacketProtocol"
            )
        if self.device_tick_modulus is not None and (
            not isinstance(self.device_tick_modulus, int)
            or isinstance(self.device_tick_modulus, bool)
            or self.device_tick_modulus < 2
            or self.device_tick_modulus > 1 << 64
        ):
            raise ValueError("invalid device_tick_modulus")
        tick_source = _contract_text(
            "device_tick_modulus_source", self.device_tick_modulus_source
        )
        if self.device_tick_modulus is not None and tick_source.casefold() == "unknown":
            raise ValueError("device tick modulus requires provenance")
        if self.device_sequence_modulus is not None and (
            not isinstance(self.device_sequence_modulus, int)
            or isinstance(self.device_sequence_modulus, bool)
            or not 2 <= self.device_sequence_modulus <= 1 << 64
        ):
            raise ValueError("invalid device_sequence_modulus")
        sequence_source = _contract_text(
            "device_sequence_modulus_source", self.device_sequence_modulus_source
        )
        if self.device_sequence_modulus is None and sequence_source.casefold() != "unknown":
            raise ValueError("unknown device sequence modulus cannot claim provenance")
        if self.device_sequence_modulus is not None and sequence_source.casefold() == "unknown":
            raise ValueError("device sequence modulus requires provenance")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AcquisitionMetadata":
        if not isinstance(value, Mapping):
            raise ValueError("acquisition must be an object")
        expected = {
            "sample_rate", "adc_rate", "device_output_rate", "host_observed_rate",
            "device_tick_rate", "device_tick_modulus", "device_tick_modulus_source",
            "device_sequence_modulus", "device_sequence_modulus_source", "signal_chain",
            "notification_packet_protocol",
        }
        if set(value) != expected:
            raise ValueError("acquisition fields are invalid")
        return cls(
            sample_rate=RateDescriptor.from_mapping(value["sample_rate"]),
            adc_rate=RateDescriptor.from_mapping(value["adc_rate"]),
            device_output_rate=RateDescriptor.from_mapping(value["device_output_rate"]),
            host_observed_rate=RateDescriptor.from_mapping(value["host_observed_rate"]),
            device_tick_rate=RateDescriptor.from_mapping(value["device_tick_rate"]),
            device_tick_modulus=value["device_tick_modulus"],
            device_tick_modulus_source=value["device_tick_modulus_source"],
            device_sequence_modulus=value["device_sequence_modulus"],
            device_sequence_modulus_source=value["device_sequence_modulus_source"],
            signal_chain=SignalChain.from_mapping(value["signal_chain"]),
            notification_packet_protocol=NotificationPacketProtocol.from_mapping(
                value["notification_packet_protocol"]
            ),
        )


def _uint64(name: str, value: Optional[int], *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= UINT64_MAX:
        raise ValueError(f"{name} must be uint64")


@dataclass(frozen=True)
class EmgFrame:
    channel_values: tuple[Real, ...]
    host_wall_timestamp_ns: int
    host_monotonic_ns: int
    host_receive_index: int
    sample_index: int
    generation: int = 0  # Producer/writer instance generation.
    connection_generation: int = 0  # BLE connection lifecycle generation.
    device_packet_sequence: Optional[int] = None
    device_sample_counter: Optional[int] = None
    device_time_ticks: Optional[int] = None
    sample_in_packet: int = 0
    action_label: str = ""
    action_phase: str = ""
    quality_flags: QualityFlags = (
        QualityFlags.VALID
        | QualityFlags.HOST_WALL_TIME_VALID
        | QualityFlags.HOST_MONOTONIC_VALID
        | QualityFlags.HOST_RECEIVE_INDEX_VALID
    )

    def __post_init__(self) -> None:
        if not isinstance(self.channel_values, tuple) or not self.channel_values:
            raise ValueError("channel_values must be a non-empty tuple")
        for value in self.channel_values:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError("channel values must be finite numbers")
        for name in (
            "host_wall_timestamp_ns",
            "host_monotonic_ns",
            "host_receive_index",
            "sample_index",
            "generation",
            "connection_generation",
        ):
            _uint64(name, getattr(self, name))
        for name in (
            "device_packet_sequence",
            "device_sample_counter",
            "device_time_ticks",
        ):
            _uint64(name, getattr(self, name), optional=True)
        _uint64("sample_in_packet", self.sample_in_packet)
        if not isinstance(self.action_label, str) or not isinstance(self.action_phase, str):
            raise ValueError("action fields must be text")
        validate_quality_flags(self.quality_flags)


@dataclass(frozen=True)
class SequenceObservation:
    flags: QualityFlags = QualityFlags.NONE
    gap_count: int = 0


class SequenceTracker:
    """O(1)-memory modular sequence classifier."""

    __slots__ = ("modulus", "_generation", "_last")

    def __init__(self, modulus: int = 1 << 64) -> None:
        if not isinstance(modulus, int) or isinstance(modulus, bool) or modulus < 2 or modulus > 1 << 64:
            raise ValueError("modulus must be an integer from 2 through 2**64")
        self.modulus = modulus
        self._generation: Optional[int] = None
        self._last: Optional[int] = None

    def observe(self, sequence: Optional[int], generation: int = 0) -> SequenceObservation:
        _uint64("generation", generation)
        if sequence is None:
            if self._generation != generation:
                self._generation = generation
            self._last = None
            return SequenceObservation()
        if not isinstance(sequence, int) or isinstance(sequence, bool) or not 0 <= sequence < self.modulus:
            raise ValueError("sequence is outside configured modulus")
        if self._generation != generation:
            self._generation, self._last = generation, sequence
            return SequenceObservation()
        if self._last is None:
            self._last = sequence
            return SequenceObservation()
        delta = (sequence - self._last) % self.modulus
        if delta == 0:
            return SequenceObservation(QualityFlags.DUPLICATE_PACKET)
        if delta < (self.modulus + 1) // 2:
            self._last = sequence
            if delta > 1:
                return SequenceObservation(QualityFlags.PACKET_GAP, delta - 1)
            return SequenceObservation()
        return SequenceObservation(QualityFlags.OUT_OF_ORDER_PACKET)

    def checkpoint(self) -> tuple[Optional[int], Optional[int]]:
        return self._generation, self._last

    def restore(self, checkpoint: tuple[Optional[int], Optional[int]]) -> None:
        if not isinstance(checkpoint,tuple) or len(checkpoint)!=2: raise ValueError("invalid checkpoint")
        generation, last = checkpoint
        if generation is not None:_uint64("generation",generation)
        if last is not None and (not isinstance(last,int) or isinstance(last,bool) or not 0<=last<self.modulus): raise ValueError("invalid checkpoint sequence")
        self._generation, self._last = generation, last
