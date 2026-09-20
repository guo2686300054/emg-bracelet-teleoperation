"""Versioned, strict application configuration."""

from __future__ import annotations

import configparser
import logging
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from emg_protocol import (
    AcquisitionMetadata,
    DeviceKey,
    LOGICAL16_PROTOCOL,
    NotificationPacketProtocol,
    RateDescriptor,
    SignalChain,
)

CONFIG_MAJOR, CONFIG_MINOR = 1, 4
MIN_RAW_AUDIT_FILE_BYTES = 8192
RAW_AUDIT_MAX_BYTES_LIMIT = 1 << 30
RAW_AUDIT_TOTAL_BUDGET_LIMIT = 2 << 30
PathLike = Union[str, Path]
RateConfig = RateDescriptor  # Compatibility alias; one canonical rate type.


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SerialConfig:
    port: str = "COM3"
    baudrate: int = 115200


@dataclass(frozen=True)
class BleConfig:
    device_name: str
    device_address: str
    device_key: DeviceKey


@dataclass(frozen=True)
class DataConfig:
    sample_time_rate: RateDescriptor
    adc_sampling_rate: RateDescriptor
    device_output_rate: RateDescriptor
    host_observed_rate: RateDescriptor
    device_tick_rate: RateDescriptor
    legacy_unclassified_rate: RateDescriptor
    device_tick_modulus: Optional[int]
    device_tick_modulus_source: str
    device_sequence_modulus: Optional[int]
    device_sequence_modulus_source: str
    channels: int
    data_path: Path
    signal_chain: SignalChain


@dataclass(frozen=True)
class LoggingConfig:
    path: Path
    level: str
    max_bytes: int
    backup_count: int
    queue_capacity: int


@dataclass(frozen=True)
class RawAuditConfig:
    enabled: bool = False
    max_bytes: int = 64 * 1024 * 1024
    backup_count: int = 3
    payload_prefix_bytes: int = 256
    record_valid: bool = True
    record_invalid: bool = True
    retention_policy: str = "bounded_rotating_files"

    @property
    def total_budget_bytes(self) -> int:
        return self.max_bytes * (self.backup_count + 1)

    def __post_init__(self) -> None:
        for name in ("enabled", "record_valid", "record_invalid"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        if not isinstance(self.max_bytes, int) or isinstance(self.max_bytes, bool) or not 1 <= self.max_bytes <= RAW_AUDIT_MAX_BYTES_LIMIT:
            raise ValueError("max_bytes out of range")
        if not isinstance(self.backup_count, int) or isinstance(self.backup_count, bool) or not 0 <= self.backup_count <= 100:
            raise ValueError("backup_count out of range")
        if not isinstance(self.payload_prefix_bytes, int) or isinstance(self.payload_prefix_bytes, bool) or not 0 <= self.payload_prefix_bytes <= 65536:
            raise ValueError("payload_prefix_bytes out of range")
        if self.retention_policy != "bounded_rotating_files":
            raise ValueError("unsupported retention_policy")
        if self.enabled and self.max_bytes < MIN_RAW_AUDIT_FILE_BYTES:
            raise ValueError(f"enabled max_bytes must be at least {MIN_RAW_AUDIT_FILE_BYTES}")
        if self.total_budget_bytes > RAW_AUDIT_TOTAL_BUDGET_LIMIT:
            raise ValueError("raw audit total budget exceeds hard limit")
        if self.enabled and not (self.record_valid or self.record_invalid):
            raise ValueError("enabled raw audit must record valid or invalid packets")


@dataclass(frozen=True)
class AppConfig:
    serial: SerialConfig
    ble: BleConfig
    data: DataConfig
    logging: LoggingConfig
    raw_audit: RawAuditConfig
    packet_protocol: NotificationPacketProtocol
    source_path: Path
    config_version: str = "1.0"

    @property
    def device_id(self) -> str:
        return str(self.ble.device_key)

    @property
    def channels(self) -> int:
        return self.data.channels

    @property
    def data_path(self) -> Path:
        return self.data.data_path

    @property
    def log_path(self) -> Path:
        return self.logging.path

    def to_acquisition_metadata(self) -> AcquisitionMetadata:
        return AcquisitionMetadata(
            sample_rate=self.data.sample_time_rate,
            adc_rate=self.data.adc_sampling_rate,
            device_output_rate=self.data.device_output_rate,
            host_observed_rate=self.data.host_observed_rate,
            device_tick_rate=self.data.device_tick_rate,
            device_tick_modulus=self.data.device_tick_modulus,
            device_tick_modulus_source=self.data.device_tick_modulus_source,
            device_sequence_modulus=self.data.device_sequence_modulus,
            device_sequence_modulus_source=self.data.device_sequence_modulus_source,
            signal_chain=self.data.signal_chain,
            notification_packet_protocol=self.packet_protocol,
        )


_RATE_NAMES = (
    "sample_time_rate",
    "adc_sampling_rate",
    "device_output_rate",
    "host_observed_rate",
    "device_tick_rate",
)
_SIGNAL_KEYS = {
    "sample_format",
    "unit",
    "adc_bit_width",
    "scaling",
    "offset",
    "filter",
    "rectification",
    "normalization",
    "device_tick_modulus",
    "device_tick_modulus_source",
    "device_sequence_modulus",
    "device_sequence_modulus_source",
    "quantization",
}
_ALLOWED = {
    "App": {"config_version"},
    "Serial": {"port", "baudrate"},
    "BLE": {"device_name", "device_address", "device_id"},
    "Data": {
        "channels",
        "data_path",
        *(
            item
            for rate in _RATE_NAMES
            for item in (
                rate,
                f"{rate}_source_kind",
                f"{rate}_evidence_ref",
                f"{rate}_confirmed",
            )
        ),
        *_SIGNAL_KEYS,
    },
    "Logging": {"path", "level", "max_bytes", "backup_count", "queue_capacity"},
    "RawAudit": {"enabled", "max_bytes", "backup_count", "payload_prefix_bytes", "record_valid", "record_invalid", "retention_policy"},
    "Protocol": {"mode", "wire_packet_size", "logical_packet_size", "padding_rule", "evidence_ref"},
}


def _scalar(parser, section, key, fallback, *, allow_empty=False):
    value = parser.get(section, key, fallback=fallback)
    if not isinstance(value, str) or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ConfigError(f"[{section}] {key} contains control characters")
    cleaned = value.strip()
    if not cleaned and not allow_empty:
        raise ConfigError(f"[{section}] {key} must not be empty")
    return cleaned


def _integer(parser, section, key, default, zero=False):
    try:
        value = int(_scalar(parser, section, key, str(default)))
    except ValueError as exc:
        raise ConfigError(f"[{section}] {key} must be an integer") from exc
    if value < (0 if zero else 1):
        raise ConfigError(f"[{section}] {key} out of range")
    return value


def _boolean(parser, section, key, default):
    _scalar(parser, section, key, "true" if default else "false")
    try:
        return parser.getboolean(section, key, fallback=default)
    except ValueError as exc:
        raise ConfigError(f"[{section}] {key} must be boolean") from exc


def _optional_integer(parser, section, key):
    raw = _scalar(parser, section, key, "unknown").casefold()
    if raw in {"", "unknown", "none", "null"}:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"[{section}] {key} must be an integer or unknown") from exc
    if not 2 <= value <= 1 << 64:
        raise ConfigError(f"[{section}] {key} out of range")
    return value


def _rate(parser, key):
    raw = _scalar(parser, "Data", key, "unknown").casefold()
    if raw in {"unknown", "none", "null", ""}:
        value = None
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigError(f"[Data] {key} invalid") from exc
        if not math.isfinite(value) or value <= 0:
            raise ConfigError(f"[Data] {key} must be finite positive")
    try:
        confirmed = parser.getboolean("Data", f"{key}_confirmed", fallback=False)
    except ValueError as exc:
        raise ConfigError(f"[Data] {key}_confirmed invalid") from exc
    source = _scalar(parser, "Data", f"{key}_source_kind", "unknown")
    evidence = _scalar(parser, "Data", f"{key}_evidence_ref", "unknown")
    try:
        return RateDescriptor(value, source, evidence, confirmed)
    except ValueError as exc:
        raise ConfigError(f"[Data] {key}: {exc}") from exc


def _path(raw, base):
    value = Path(raw).expanduser()
    return (value if value.is_absolute() else base / value).resolve()


def load_config(path: Optional[PathLike] = None) -> AppConfig:
    source = (Path(path) if path else Path(__file__).with_name("config.ini")).expanduser().resolve()
    parser = configparser.ConfigParser(interpolation=None)
    if source.exists():
        try:
            with source.open("r", encoding="utf-8-sig") as stream:
                parser.read_file(stream)
        except (OSError, configparser.Error) as exc:
            raise ConfigError(f"cannot read {source}: {exc}") from exc
    if parser.defaults():
        raise ConfigError(f"unknown [DEFAULT] keys: {sorted(parser.defaults())}")
    version = _scalar(parser, "App", "config_version", "0.0")
    try:
        major, minor = (int(item) for item in version.split(".", 1))
    except (ValueError, TypeError) as exc:
        raise ConfigError("[App] config_version must be major.minor") from exc
    if major < 0 or minor < 0:
        raise ConfigError("config version cannot be negative")
    if major == CONFIG_MAJOR and minor > CONFIG_MINOR:
        raise ConfigError(f"unsupported future config minor {major}.{minor}")
    if major > CONFIG_MAJOR:
        raise ConfigError(f"unsupported future config major {major}")
    if major == 0 and minor != 0:
        raise ConfigError(f"unsupported legacy config version {major}.{minor}")
    legacy = major == 0
    if (major, minor) < (1, 2) and parser.has_section("RawAudit"):
        raise ConfigError("[RawAudit] requires config_version 1.2 or newer")
    if (major, minor) < (1, 4) and parser.has_section("Protocol"):
        raise ConfigError("[Protocol] requires config_version 1.4 or newer")

    unknown_sections = set(parser.sections()) - set(_ALLOWED)
    if unknown_sections:
        raise ConfigError(f"unknown sections: {sorted(unknown_sections)}")
    for section, allowed in _ALLOWED.items():
        if parser.has_section(section):
            permitted = set(allowed)
            if legacy and section == "Data":
                permitted |= {"save_path", "sampling_rate"}
            unknown = set(parser.options(section)) - permitted
            if unknown:
                raise ConfigError(f"unknown keys in [{section}]: {sorted(unknown)}")

    legacy_rate = RateDescriptor()
    if legacy and parser.has_option("Data", "sampling_rate"):
        raw = _scalar(parser, "Data", "sampling_rate", "unknown")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigError("legacy sampling_rate invalid") from exc
        if not math.isfinite(value) or value <= 0:
            raise ConfigError("legacy sampling_rate invalid")
        warnings.warn("migrated v0 sampling_rate as unclassified", UserWarning, stacklevel=2)
        legacy_rate = RateDescriptor(value, "legacy_key", "config.ini:[Data].sampling_rate", False)

    legacy_path = _scalar(parser, "Data", "save_path", "data") if legacy else "data"
    data_raw = _scalar(parser, "Data", "data_path", legacy_path)
    channels = _integer(parser, "Data", "channels", 8)
    if channels > 256:
        raise ConfigError("channels > 256")
    level = _scalar(parser, "Logging", "level", "INFO").upper()
    if not isinstance(logging._nameToLevel.get(level), int):
        raise ConfigError("invalid log level")
    port = _scalar(parser, "Serial", "port", "COM3")
    try:
        device_key = DeviceKey(_scalar(parser, "BLE", "device_id", "dev-00000000000000000000000000000000"))
        signal_chain = SignalChain(
            sample_format=_scalar(parser, "Data", "sample_format", "unknown"),
            unit=_scalar(parser, "Data", "unit", "unknown"),
            adc_bit_width=_optional_integer(parser, "Data", "adc_bit_width"),
            scaling=_scalar(parser, "Data", "scaling", "unknown"),
            offset=_scalar(parser, "Data", "offset", "unknown"),
            filter=_scalar(parser, "Data", "filter", "unknown"),
            rectification=_scalar(parser, "Data", "rectification", "unknown"),
            normalization=_scalar(parser, "Data", "normalization", "unknown"),
            quantization=_scalar(parser, "Data", "quantization", "unknown"),
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    tick_modulus = _optional_integer(parser, "Data", "device_tick_modulus")
    tick_source = _scalar(parser, "Data", "device_tick_modulus_source", "unknown")
    if tick_modulus is not None and tick_source.casefold() == "unknown":
        raise ConfigError("device_tick_modulus requires provenance")
    sequence_modulus = _optional_integer(parser, "Data", "device_sequence_modulus")
    sequence_source = _scalar(parser, "Data", "device_sequence_modulus_source", "unknown")
    if sequence_modulus is None and sequence_source.casefold() != "unknown":
        raise ConfigError("unknown device_sequence_modulus cannot claim provenance")
    if sequence_modulus is not None and sequence_source.casefold() == "unknown":
        raise ConfigError("device_sequence_modulus requires provenance")
    try:
        raw_audit = RawAuditConfig(
            enabled=_boolean(parser, "RawAudit", "enabled", False),
            max_bytes=_integer(parser, "RawAudit", "max_bytes", 64 * 1024 * 1024),
            backup_count=_integer(parser, "RawAudit", "backup_count", 3, True),
            payload_prefix_bytes=_integer(parser, "RawAudit", "payload_prefix_bytes", 256, True),
            record_valid=_boolean(parser, "RawAudit", "record_valid", True),
            record_invalid=_boolean(parser, "RawAudit", "record_invalid", True),
            retention_policy=_scalar(parser, "RawAudit", "retention_policy", "bounded_rotating_files"),
        )
    except ValueError as exc:
        raise ConfigError(f"[RawAudit] {exc}") from exc
    try:
        packet_protocol = NotificationPacketProtocol(
            mode=_scalar(parser, "Protocol", "mode", LOGICAL16_PROTOCOL),
            wire_packet_size=_integer(parser, "Protocol", "wire_packet_size", 16),
            logical_packet_size=_integer(parser, "Protocol", "logical_packet_size", 16),
            padding_rule=_scalar(parser, "Protocol", "padding_rule", "none"),
            evidence_ref=_scalar(
                parser,
                "Protocol",
                "evidence_ref",
                "unknown",
            ),
        )
    except ValueError as exc:
        raise ConfigError(f"[Protocol] {exc}") from exc
    if (
        raw_audit.enabled
        and (raw_audit.record_valid or raw_audit.record_invalid)
        and raw_audit.payload_prefix_bytes < packet_protocol.wire_packet_size
    ):
        raise ConfigError(
            "[RawAudit] payload_prefix_bytes must cover configured wire packet"
        )
    return AppConfig(
        SerialConfig(port, _integer(parser, "Serial", "baudrate", 115200)),
        BleConfig(
            _scalar(parser, "BLE", "device_name", "MyEMGBracelet"),
            _scalar(parser, "BLE", "device_address", "", allow_empty=True),
            device_key,
        ),
        DataConfig(
            _rate(parser, "sample_time_rate"),
            _rate(parser, "adc_sampling_rate"),
            _rate(parser, "device_output_rate"),
            _rate(parser, "host_observed_rate"),
            _rate(parser, "device_tick_rate"),
            legacy_rate,
            tick_modulus,
            tick_source,
            sequence_modulus,
            sequence_source,
            channels,
            _path(data_raw, source.parent),
            signal_chain,
        ),
        LoggingConfig(
            _path(_scalar(parser, "Logging", "path", "logs"), source.parent),
            level,
            _integer(parser, "Logging", "max_bytes", 5242880),
            _integer(parser, "Logging", "backup_count", 5, True),
            _integer(parser, "Logging", "queue_capacity", 2048),
        ),
        raw_audit,
        packet_protocol,
        source,
        f"{CONFIG_MAJOR}.{CONFIG_MINOR}",
    )
