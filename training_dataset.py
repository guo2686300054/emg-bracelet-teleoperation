"""Fail-closed preparation of modern EMG sessions for grouped training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import random
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union, cast

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer

from data_recorder import (
    METADATA_SCHEMA_ID,
    METADATA_SCHEMA_MAJOR,
    METADATA_SCHEMA_MINOR,
    _RecoveryStreamValidator,
    _decode_recovery_acquisition,
    _validated_session_boundary,
    validate_subject_key,
)
from emg_protocol import AcquisitionMetadata, DeviceKey, HandSide
from legacy_dataset import SPLIT_POLICY
import session_quality
from training_contract import (
    EMG_CHANNEL_COUNT,
    validate_training_provenance,
    validate_sample_rate_source_kind,
    validate_training_annotation,
)


PathLike = Union[str, Path]
MANIFEST_SCHEMA = "emg.training.dataset_manifest"
MANIFEST_VERSION = "1.1"
SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS = (0.7, 0.15, 0.15)
MAX_SESSIONS = 1_000
MAX_METADATA_BYTES = 1024 * 1024
MAX_CSV_BYTES = 256 * 1024 * 1024
MAX_CSV_ROWS_PER_SESSION = 250_000
MAX_DATASET_ROWS = 1_000_000
MAX_MANIFEST_WINDOWS = 100_000
_REPARSE_POINT = 0x400
_WINDOWS_NO_REPLACE_RENAME = os.name == "nt"
_LOGGER = logging.getLogger(__name__)


class TrainingDatasetError(ValueError):
    """Raised when training inputs or output paths violate the data contract."""


def _reject_nonfinite_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


@dataclass(frozen=True)
class LoadedSession:
    session_dir: Path
    subject_id: str
    session_id: str
    sample_rate_hz: float
    sample_rate_descriptor: Mapping[str, Any]
    row_count: int
    action_runs: tuple[tuple[str, str, int, int], ...]
    source_labels: frozenset[str]
    metadata_sha256: str
    metadata_size_bytes: int
    samples_sha256: str
    samples_size_bytes: int
    channel_count: int
    sample_format: str
    signal_chain: Mapping[str, Any]
    notification_protocol: Mapping[str, Any]
    hand_side: str
    quality_snapshot_version: int
    quality_source_snapshot: tuple[tuple[str, str, int, str], ...]
    provenance: str

    @property
    def split_group_id(self) -> str:
        return self.subject_id

    @property
    def full_session_group_id(self) -> str:
        return f"{self.subject_id}:{self.session_id}"


def _safe_existing_path(path: PathLike, *, kind: str) -> Path:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
        stat_result = candidate.lstat()
    except (OSError, RuntimeError) as exc:
        raise TrainingDatasetError(
            f"cannot access {kind} {candidate.name!r}: {type(exc).__name__}"
        ) from exc
    if candidate.is_symlink() or bool(getattr(stat_result, "st_file_attributes", 0) & _REPARSE_POINT):
        raise TrainingDatasetError(f"{kind} must not be a symlink or reparse point")
    return resolved


def _direct_file(session_dir: Path, name: str, maximum_bytes: int) -> tuple[Path, int]:
    candidate = session_dir / name
    resolved = _safe_existing_path(candidate, kind=name)
    if resolved.parent != session_dir or resolved.name != name or not resolved.is_file():
        raise TrainingDatasetError(f"{name} must be a regular direct child of the session directory")
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise TrainingDatasetError(f"cannot stat {name}: {type(exc).__name__}") from exc
    if size > maximum_bytes:
        raise TrainingDatasetError(f"{name} exceeds the {maximum_bytes}-byte resource limit")
    return resolved, size


def _read_bounded_bytes(path: Path, maximum_bytes: int) -> tuple[bytes, str]:
    try:
        with path.open("rb") as stream:
            content = stream.read(maximum_bytes + 1)
    except OSError as exc:
        raise TrainingDatasetError(f"cannot read {path.name}: {type(exc).__name__}") from exc
    if len(content) > maximum_bytes:
        raise TrainingDatasetError(f"{path.name} exceeds the {maximum_bytes}-byte resource limit")
    return content, hashlib.sha256(content).hexdigest()


def _assert_source_unchanged(path: Path, expected_sha256: str, maximum_bytes: int) -> None:
    verified_path, _ = _direct_file(path.parent, path.name, maximum_bytes)
    _, current_sha256 = _read_bounded_bytes(verified_path, maximum_bytes)
    if current_sha256 != expected_sha256:
        raise TrainingDatasetError(f"{path.name} changed during validation")


def _validate_quality_snapshot(
    session_dir: Path,
    snapshot: object,
    *,
    metadata_sha256: str,
    metadata_size: int,
    samples_sha256: str,
    samples_size: int,
) -> tuple[tuple[str, str, int, str], ...]:
    if not isinstance(snapshot, Mapping) or snapshot.get("version") != 1:
        raise TrainingDatasetError("session quality source snapshot is missing or unsupported")
    files = snapshot.get("files")
    if not isinstance(files, list):
        raise TrainingDatasetError("session quality source snapshot file list is invalid")
    expected: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {
            "relative_name", "role", "size_bytes", "sha256"
        }:
            raise TrainingDatasetError("session quality source snapshot entry is invalid")
        name = item["relative_name"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in expected
        ):
            raise TrainingDatasetError("session quality source snapshot filename is invalid")
        role, size, digest = item["role"], item["size_bytes"], item["sha256"]
        if role not in {"metadata", "samples", "raw_audit"}:
            raise TrainingDatasetError("session quality source snapshot role is invalid")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise TrainingDatasetError("session quality source snapshot size is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise TrainingDatasetError("session quality source snapshot hash is invalid")
        expected[name] = dict(item)

    current: dict[str, dict[str, Any]] = {}
    raw_bytes = 0
    for path in session_quality._ordered_raw_files(session_dir):
        content, digest = _read_bounded_bytes(path, session_quality.MAX_RAW_BYTES)
        raw_bytes += len(content)
        if raw_bytes > session_quality.MAX_RAW_BYTES:
            raise TrainingDatasetError("RawAudit snapshot exceeds the hard byte limit")
        current[path.name] = {
            "relative_name": path.name,
            "role": "raw_audit",
            "size_bytes": len(content),
            "sha256": digest,
        }
    for name, role, validated_size, validated_digest, limit in (
        ("metadata.json", "metadata", metadata_size, metadata_sha256, MAX_METADATA_BYTES),
        ("samples.csv", "samples", samples_size, samples_sha256, MAX_CSV_BYTES),
    ):
        path, _ = _direct_file(session_dir, name, limit)
        content, digest = _read_bounded_bytes(path, limit)
        if len(content) != validated_size or digest != validated_digest:
            raise TrainingDatasetError(f"{name} changed before training admission")
        current[name] = {
            "relative_name": name,
            "role": role,
            "size_bytes": len(content),
            "sha256": digest,
        }
    if expected != current:
        raise TrainingDatasetError("session sources changed after quality analysis")
    return tuple(
        sorted(
            (
                item["relative_name"],
                item["role"],
                item["size_bytes"],
                item["sha256"],
            )
            for item in expected.values()
        )
    )


def _revalidate_loaded_session_snapshot(session: LoadedSession) -> None:
    snapshot = {
        "version": session.quality_snapshot_version,
        "files": [
            {
                "relative_name": name,
                "role": role,
                "size_bytes": size,
                "sha256": digest,
            }
            for name, role, size, digest in session.quality_source_snapshot
        ],
    }
    _validate_quality_snapshot(
        session.session_dir,
        snapshot,
        metadata_sha256=session.metadata_sha256,
        metadata_size=session.metadata_size_bytes,
        samples_sha256=session.samples_sha256,
        samples_size=session.samples_size_bytes,
    )


def _load_json(content: bytes) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            content.decode("utf-8-sig"),
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise TrainingDatasetError(f"cannot read metadata.json: {type(exc).__name__}") from exc
    if not isinstance(payload, Mapping):
        raise TrainingDatasetError("metadata.json root must be an object")
    return payload


class _HashingRawReader(io.RawIOBase):
    """Hash and bound exactly the bytes consumed by CSV validation."""

    def __init__(self, stream: io.BufferedReader, maximum_bytes: int) -> None:
        super().__init__()
        self._stream = stream
        self._maximum_bytes = maximum_bytes
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: WriteableBuffer) -> int:
        buffer_view = memoryview(buffer).cast("B")
        data = self._stream.read(len(buffer_view))
        if not data:
            return 0
        self.byte_count += len(data)
        if self.byte_count > self._maximum_bytes:
            raise TrainingDatasetError("samples.csv exceeds the streaming byte resource limit")
        self.digest.update(data)
        buffer_view[: len(data)] = data
        return len(data)


def _confirmed_sample_rate(
    metadata: Mapping[str, Any], schema_minor: int
) -> tuple[AcquisitionMetadata, float]:
    try:
        acquisition, migrations = _decode_recovery_acquisition(
            metadata.get("acquisition"), schema_minor
        )
    except (TypeError, ValueError) as exc:
        raise TrainingDatasetError(f"invalid canonical acquisition metadata: {exc}") from exc
    if migrations:
        raise TrainingDatasetError("training requires acquisition metadata without migrations")
    descriptor = acquisition.sample_rate
    if not descriptor.confirmed or descriptor.value_hz is None:
        raise TrainingDatasetError(
            "a confirmed sample-time rate is required; host_observed_rate is not a substitute"
        )
    try:
        validate_sample_rate_source_kind(descriptor.source_kind)
    except ValueError:
        raise TrainingDatasetError(
            f"sample-time rate source {descriptor.source_kind!r} is not in the trusted allowlist"
        )
    if descriptor.evidence_ref.strip().casefold() == "unknown":
        raise TrainingDatasetError("sample-time rate requires a specific evidence_ref")
    evidence_ref = descriptor.evidence_ref.strip()
    if evidence_ref.startswith(("/", "\\")) or (
        len(evidence_ref) >= 3
        and evidence_ref[1] == ":"
        and evidence_ref[2] in {"/", "\\"}
    ):
        raise TrainingDatasetError("sample-time rate evidence_ref must not be an absolute path")
    top_level = metadata.get("sample_rate_hz")
    if isinstance(top_level, bool) or not isinstance(top_level, (int, float)):
        raise TrainingDatasetError("metadata sample_rate_hz must contain the confirmed sample-time rate")
    if not math.isfinite(float(top_level)) or float(top_level) != float(descriptor.value_hz):
        raise TrainingDatasetError("metadata sample_rate_hz disagrees with acquisition.sample_rate")
    if metadata.get("sample_rate_semantics") != "sample_time_base_only":
        raise TrainingDatasetError("sample_rate_semantics must be sample_time_base_only")
    return acquisition, float(descriptor.value_hz)


def load_session(session_dir: PathLike) -> LoadedSession:
    """Load and fully validate one current canonical session."""
    directory = _safe_existing_path(session_dir, kind="session directory")
    if not directory.is_dir():
        raise TrainingDatasetError("session path is not a directory")
    try:
        quality_report = session_quality.analyze_session(directory)
    except Exception as exc:
        raise TrainingDatasetError(
            f"session quality gate could not complete: {type(exc).__name__}"
        ) from exc
    quality_usable = quality_report.get("training_usable") is True
    metadata_path, _ = _direct_file(directory, "metadata.json", MAX_METADATA_BYTES)
    metadata_bytes, metadata_sha256 = _read_bounded_bytes(metadata_path, MAX_METADATA_BYTES)
    metadata_size = len(metadata_bytes)
    metadata = _load_json(metadata_bytes)
    if metadata.get("schema_id") != METADATA_SCHEMA_ID:
        raise TrainingDatasetError("unsupported metadata schema_id")
    if metadata.get("schema_major") != METADATA_SCHEMA_MAJOR or metadata.get("schema_minor") != METADATA_SCHEMA_MINOR:
        raise TrainingDatasetError(f"training requires canonical metadata schema 1.{METADATA_SCHEMA_MINOR}")
    minor = METADATA_SCHEMA_MINOR
    if metadata.get("status") != "complete":
        raise TrainingDatasetError(f"session is not complete: status={metadata.get('status')!r}")
    extra = metadata.get("extra")
    if not isinstance(extra, Mapping):
        raise TrainingDatasetError("metadata extra must be an object")
    try:
        provenance = validate_training_provenance(metadata.get("training_provenance"))
    except ValueError as exc:
        raise TrainingDatasetError(f"invalid or missing training provenance: {exc}") from exc
    if metadata.get("recorder_state") != "CLOSED" or metadata.get("resource_state") != "CLOSED":
        raise TrainingDatasetError("complete session must have closed recorder and resource state")
    if metadata.get("csv_file") != "samples.csv":
        raise TrainingDatasetError("csv_file must be exactly samples.csv")
    subject_id, session_id = metadata.get("subject_id"), metadata.get("session_id")
    try:
        subject_id = validate_subject_key(cast(str, subject_id))
        device_id = str(DeviceKey(cast(str, metadata.get("device_id"))))
        hand_side = HandSide(metadata.get("hand_side")).value
    except (TypeError, ValueError) as exc:
        raise TrainingDatasetError(f"invalid session identity or hand_side: {exc}") from exc
    if hand_side == HandSide.UNKNOWN.value:
        raise TrainingDatasetError("hand_side must be known for training")
    if not isinstance(session_id, str) or not session_id or directory.name != session_id:
        raise TrainingDatasetError("metadata session_id differs from its directory")
    if directory.parent.name != subject_id:
        raise TrainingDatasetError("metadata subject_id differs from its directory")
    if metadata.get("writer_generation_semantics") != "producer_instance_generation":
        raise TrainingDatasetError("invalid writer generation semantics")
    if metadata.get("connection_generation_semantics") != "ble_connection_lifecycle_generation":
        raise TrainingDatasetError("invalid connection generation semantics")
    acquisition, sample_rate_hz = _confirmed_sample_rate(metadata, minor)
    if acquisition.signal_chain.sample_format == "unknown":
        raise TrainingDatasetError("training requires an explicit sample_format")
    channels, row_count = metadata.get("channels"), metadata.get("row_count")
    if channels != EMG_CHANNEL_COUNT or isinstance(channels, bool):
        raise TrainingDatasetError(
            f"formal EMG training requires exactly {EMG_CHANNEL_COUNT} channels"
        )
    if not isinstance(row_count, int) or isinstance(row_count, bool) or not 0 < row_count <= MAX_CSV_ROWS_PER_SESSION:
        raise TrainingDatasetError("complete session row_count is invalid or exceeds the resource limit")
    if metadata.get("last_persisted_row") != row_count:
        raise TrainingDatasetError("last_persisted_row must equal row_count")
    try:
        boundary = _validated_session_boundary(metadata.get("session_boundary"), minor)
    except (TypeError, ValueError) as exc:
        raise TrainingDatasetError(f"invalid canonical session_boundary: {exc}") from exc
    for field in ("received_count", "eligible_count", "written_count"):
        if boundary[field] != row_count:
            raise TrainingDatasetError(f"session_boundary {field} must equal CSV row_count")
    for field in ("queue_drop_total", "queue_drop_session", "tail_pending_count", "tail_loss_count"):
        if boundary[field] != 0:
            raise TrainingDatasetError(f"session_boundary {field} must be zero for training")
    if boundary["incomplete_reason"] is not None:
        raise TrainingDatasetError("session_boundary incomplete_reason must be null for training")
    start_host_receive_index = boundary["start_host_receive_index"]
    end_host_receive_index = boundary["end_host_receive_index"]
    if start_host_receive_index is None or end_host_receive_index is None:
        raise TrainingDatasetError("session_boundary start/end host indexes are required")
    if end_host_receive_index - start_host_receive_index != boundary["received_count"]:
        raise TrainingDatasetError(
            "session_boundary open-start host span must equal received_count"
        )
    csv_path, _ = _direct_file(directory, "samples.csv", MAX_CSV_BYTES)
    validator = _RecoveryStreamValidator(
        channels=channels, schema_minor=minor, acquisition=acquisition,
        device_id=device_id, session_id=session_id,
    )
    action_runs: list[tuple[str, str, int, int]] = []
    current_action: tuple[str, str] | None = None
    run_start = 0
    csv_rows = 0
    first_host_receive_index: int | None = None
    last_host_receive_index: int | None = None
    csv_source: _HashingRawReader | None = None
    try:
        with csv_path.open("rb") as binary_stream:
            csv_source = _HashingRawReader(binary_stream, MAX_CSV_BYTES)
            with io.BufferedReader(csv_source) as buffered_stream:
                with io.TextIOWrapper(
                    buffered_stream, encoding="utf-8-sig", newline=""
                ) as stream:
                    reader = csv.DictReader(stream)
                    if reader.fieldnames != validator.expected_columns:
                        raise TrainingDatasetError(
                            "samples.csv schema does not match canonical recorder schema"
                        )
                    for row_number, row in enumerate(reader, 1):
                        if row_number > MAX_CSV_ROWS_PER_SESSION:
                            raise TrainingDatasetError(
                                "samples.csv exceeds the per-session row limit"
                            )
                        try:
                            validator.validate(row)
                        except (TypeError, ValueError) as exc:
                            raise TrainingDatasetError(
                                f"row {row_number}: canonical stream validation failed: {exc}"
                            ) from exc
                        host_receive_index = int(row["host_receive_index"])
                        if int(row["sample_in_packet"]) != 0:
                            raise TrainingDatasetError(
                                f"row {row_number}: sample_in_packet must be zero "
                                "for one-sample notifications"
                            )
                        if host_receive_index != start_host_receive_index + row_number:
                            raise TrainingDatasetError(
                                f"row {row_number}: host_receive_index is not strictly continuous"
                            )
                        label = row["action_label"]
                        phase = row["action_phase"]
                        try:
                            validate_training_annotation(label, phase)
                        except ValueError as exc:
                            raise TrainingDatasetError(
                                f"row {row_number}: {exc}"
                            ) from exc
                        action = (label, phase)
                        if current_action is None:
                            current_action = action
                            run_start = row_number - 1
                        elif action != current_action:
                            action_runs.append(
                                (*current_action, run_start, row_number - 1)
                            )
                            current_action = action
                            run_start = row_number - 1
                        if first_host_receive_index is None:
                            first_host_receive_index = host_receive_index
                        last_host_receive_index = host_receive_index
                        csv_rows = row_number
    except TrainingDatasetError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise TrainingDatasetError(
            f"cannot read samples.csv: {type(exc).__name__}"
        ) from exc
    if current_action is not None:
        action_runs.append((*current_action, run_start, csv_rows))
    if csv_rows != row_count:
        raise TrainingDatasetError(f"samples.csv row count {csv_rows} differs from metadata row_count {row_count}")
    if first_host_receive_index != start_host_receive_index + 1:
        raise TrainingDatasetError(
            "CSV first host_receive_index must immediately follow session boundary start"
        )
    if last_host_receive_index != end_host_receive_index:
        raise TrainingDatasetError("session_boundary end_host_receive_index differs from CSV")
    if boundary["host_monotonic_tie_count"] != validator.host_monotonic_tie_count:
        raise TrainingDatasetError("session_boundary host_monotonic_tie_count differs from CSV")
    if csv_source is None:
        raise TrainingDatasetError("samples.csv was not validated")
    samples_size = csv_source.byte_count
    samples_sha256 = csv_source.digest.hexdigest()
    _assert_source_unchanged(metadata_path, metadata_sha256, MAX_METADATA_BYTES)
    _assert_source_unchanged(csv_path, samples_sha256, MAX_CSV_BYTES)
    if not quality_usable:
        failed_checks = sorted(
            name
            for name, check in quality_report.get("checks", {}).items()
            if isinstance(check, Mapping) and check.get("status") in {"fail", "warning"}
        )
        summary = ", ".join(failed_checks[:12]) or "unknown_quality_failure"
        raise TrainingDatasetError(f"session quality gate rejected input: {summary}")
    quality_source_snapshot = _validate_quality_snapshot(
        directory,
        quality_report.get("source_snapshot"),
        metadata_sha256=metadata_sha256,
        metadata_size=metadata_size,
        samples_sha256=samples_sha256,
        samples_size=samples_size,
    )
    return LoadedSession(
        session_dir=directory, subject_id=subject_id, session_id=session_id,
        sample_rate_hz=sample_rate_hz,
        sample_rate_descriptor=asdict(acquisition.sample_rate),
        row_count=row_count,
        action_runs=tuple(action_runs),
        source_labels=frozenset(run[0] for run in action_runs),
        metadata_sha256=metadata_sha256, metadata_size_bytes=metadata_size,
        samples_sha256=samples_sha256, samples_size_bytes=samples_size,
        channel_count=channels, sample_format=acquisition.signal_chain.sample_format,
        signal_chain=asdict(acquisition.signal_chain),
        notification_protocol=asdict(acquisition.notification_packet_protocol),
        hand_side=hand_side,
        quality_snapshot_version=1,
        quality_source_snapshot=quality_source_snapshot,
        provenance=provenance,
    )

def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingDatasetError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise TrainingDatasetError(f"{name} must be a finite positive number")
    return result


def _duration_to_samples(milliseconds: float, rate_hz: float, name: str) -> int:
    exact = milliseconds * rate_hz / 1000.0
    rounded = int(math.floor(exact + 0.5))
    if rounded < 1:
        raise TrainingDatasetError(f"{name} is shorter than one sample at {rate_hz:g} Hz")
    return rounded


def _windows(session: LoadedSession, size: int, step: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label, phase, run_start, run_end in session.action_runs:
        for start in range(run_start, run_end - size + 1, step):
            end = start + size
            window_fact = (
                f"{session.samples_sha256}:{label}:{phase}:{start}:{end}".encode("utf-8")
            )
            result.append(
                {
                    "window_id": hashlib.sha256(window_fact).hexdigest(),
                    "session_id": session.session_id,
                    "subject_id": session.subject_id,
                    "label": label,
                    "action_phase": phase,
                    "start_row": start,
                    "end_row_exclusive": end,
                    "samples_sha256": session.samples_sha256,
                }
            )
            if len(result) > MAX_MANIFEST_WINDOWS:
                raise TrainingDatasetError("one session exceeds the manifest window resource limit")
    return result


def _split_counts(group_count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if group_count < 3:
        raise TrainingDatasetError("at least three distinct subjects are required for non-empty splits")
    counts = [1, 1, 1]
    remaining = group_count - 3
    if remaining:
        quotas = [remaining * ratio / sum(ratios) for ratio in ratios]
        floors = [int(math.floor(value)) for value in quotas]
        counts = [count + floor for count, floor in zip(counts, floors)]
        for index in sorted(
            range(3), key=lambda item: (-(quotas[item] - floors[item]), item)
        )[: group_count - sum(counts)]:
            counts[index] += 1
    return tuple(counts)  # type: ignore[return-value]


def _validate_ratios(split_ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(split_ratios) != 3:
        raise TrainingDatasetError("split_ratios must contain train, validation, and test ratios")
    ratios = tuple(_positive_number(value, "split ratio") for value in split_ratios)
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise TrainingDatasetError("split ratios must sum to 1")
    return ratios  # type: ignore[return-value]


def _output_path(output_path: PathLike, source_dirs: Sequence[Path]) -> Path:
    candidate = Path(output_path).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise TrainingDatasetError(
            f"invalid output path {candidate.name!r}: {type(exc).__name__}"
        ) from exc
    for directory in source_dirs:
        if resolved == directory or directory in resolved.parents:
            raise TrainingDatasetError("output must not be inside or overwrite source data")
    if resolved.exists():
        raise TrainingDatasetError("output already exists and will not be overwritten")
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        parent = resolved.parent.resolve(strict=True)
    except OSError as exc:
        raise TrainingDatasetError(
            f"cannot create output directory: {type(exc).__name__}"
        ) from exc
    if parent != resolved.parent:
        raise TrainingDatasetError("output parent resolves through an unsafe path")
    return resolved


def _contract_key(session: LoadedSession) -> tuple[Any, ...]:
    return (
        session.channel_count,
        json.dumps(session.signal_chain, sort_keys=True, separators=(",", ":")),
        json.dumps(session.notification_protocol, sort_keys=True, separators=(",", ":")),
        session.hand_side,
        json.dumps(session.sample_rate_descriptor, sort_keys=True, separators=(",", ":")),
    )


def _publish_manifest(
    destination: Path,
    manifest: Mapping[str, Any],
    sessions: Sequence[LoadedSession],
) -> None:
    """Durably stage and atomically publish a manifest without overwriting."""
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                manifest,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        # This is the final admission check.  Windows rename is atomic and does not
        # replace an existing destination.  POSIX rename would replace, so use a
        # same-directory hard link there as the atomic no-replace primitive.
        for session in sessions:
            _revalidate_loaded_session_snapshot(session)
        if _WINDOWS_NO_REPLACE_RENAME:
            os.rename(temporary, destination)
            temporary = None
            return
        os.link(temporary, destination)
    except Exception as failure:
        cleanup_failure: OSError | None = None
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_failure = exc

        if isinstance(failure, TrainingDatasetError):
            message = str(failure)
        elif isinstance(failure, FileExistsError):
            message = "output already exists and will not be overwritten"
        else:
            message = (
                f"cannot publish output manifest {destination.name!r}: "
                f"{type(failure).__name__}"
            )
        if cleanup_failure is not None:
            message += (
                "; residual temporary manifest cleanup failed: "
                f"{type(cleanup_failure).__name__}"
            )
        raise TrainingDatasetError(message) from failure

    # The POSIX hard link is now the committed output.  Never remove or roll it
    # back based on a path comparison: another process may replace that path
    # between the comparison and unlink.  A failed staging-file cleanup is only
    # an orphan warning; the valid final manifest remains published.
    assert temporary is not None
    try:
        temporary.unlink()
    except OSError as cleanup_failure:
        try:
            _LOGGER.warning(
                "manifest was published successfully, but orphaned temporary file "
                "%r could not be removed: %s",
                temporary.name,
                type(cleanup_failure).__name__,
            )
        except Exception:
            # Publication is already committed.  Diagnostics must never turn a
            # successful no-replace commit into a caller-visible failure.
            pass


def prepare_training_dataset(
    session_dirs: Sequence[PathLike],
    output_path: PathLike,
    *,
    window_ms: float,
    step_ms: float,
    seed: int = 0,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    group_by: str = "subject",
) -> dict[str, Any]:
    """Validate sessions and write a subject-grouped, sample-free window manifest."""
    if not session_dirs:
        raise TrainingDatasetError("at least one session directory is required")
    if len(session_dirs) > MAX_SESSIONS:
        raise TrainingDatasetError(f"session count exceeds the {MAX_SESSIONS}-session resource limit")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TrainingDatasetError("seed must be an integer")
    if group_by != "subject":
        raise TrainingDatasetError("only stable subject grouping is permitted")
    ratios = _validate_ratios(split_ratios)
    window_ms_value = _positive_number(window_ms, "window_ms")
    step_ms_value = _positive_number(step_ms, "step_ms")

    source_dirs: list[Path] = []
    for raw_path in session_dirs:
        resolved = _safe_existing_path(raw_path, kind="session directory")
        if resolved in source_dirs:
            raise TrainingDatasetError(f"duplicate input session directory: {resolved.name}")
        source_dirs.append(resolved)
    destination = _output_path(output_path, source_dirs)

    loaded: list[LoadedSession] = []
    dataset_rows = 0
    for directory in sorted(source_dirs, key=lambda item: (item.parent.name, item.name)):
        try:
            session = load_session(directory)
        except TrainingDatasetError as exc:
            raise TrainingDatasetError(
                f"explicit input {directory.name!r} is not training-eligible: {exc}"
            ) from exc
        dataset_rows += session.row_count
        if dataset_rows > MAX_DATASET_ROWS:
            raise TrainingDatasetError("dataset exceeds the total row resource limit")
        loaded.append(session)
    if len({_contract_key(session) for session in loaded}) != 1:
        raise TrainingDatasetError(
            "all sessions must share channel_count, sample_format, notification protocol, "
            "hand_side, and trusted sampling-rate contract"
        )
    if len({session.provenance for session in loaded}) != 1:
        raise TrainingDatasetError("all sessions must share one derived training provenance")

    rate_hz = loaded[0].sample_rate_hz
    window_size = _duration_to_samples(window_ms_value, rate_hz, "window_ms")
    step_size = _duration_to_samples(step_ms_value, rate_hz, "step_ms")
    sessions_and_windows: list[tuple[LoadedSession, list[dict[str, Any]]]] = []
    total_windows = 0
    for session in sorted(loaded, key=lambda item: item.full_session_group_id):
        windows = _windows(session, window_size, step_size)
        if not windows:
            raise TrainingDatasetError(
                f"explicit input {session.session_id!r} has no complete label+phase window"
            )
        total_windows += len(windows)
        if total_windows > MAX_MANIFEST_WINDOWS:
            raise TrainingDatasetError("dataset exceeds the manifest window resource limit")
        sessions_and_windows.append((session, windows))
    if not sessions_and_windows:
        raise TrainingDatasetError("no valid session contains a complete fixed window")

    grouped: dict[str, list[tuple[LoadedSession, list[dict[str, Any]]]]] = {}
    seen_sessions: set[str] = set()
    for session, windows in sessions_and_windows:
        if session.full_session_group_id in seen_sessions:
            raise TrainingDatasetError(f"duplicate full session group: {session.full_session_group_id}")
        seen_sessions.add(session.full_session_group_id)
        grouped.setdefault(session.subject_id, []).append((session, windows))
    subject_ids = sorted(grouped)
    random.Random(seed).shuffle(subject_ids)
    split_counts = _split_counts(len(subject_ids), ratios)
    assigned: dict[str, str] = {}
    cursor = 0
    for split_name, count in zip(SPLIT_NAMES, split_counts):
        for subject_id in subject_ids[cursor:cursor + count]:
            assigned[subject_id] = split_name
        cursor += count

    splits: dict[str, dict[str, Any]] = {
        name: {"subject_ids": [], "session_ids": [], "windows": []} for name in SPLIT_NAMES
    }
    inputs: list[dict[str, Any]] = []
    for session, windows in sessions_and_windows:
        split = splits[assigned[session.subject_id]]
        if session.subject_id not in split["subject_ids"]:
            split["subject_ids"].append(session.subject_id)
        split["session_ids"].append(session.session_id)
        split["windows"].extend(windows)
        inputs.append({
            "session_locator": f"{session.subject_id}/{session.session_id}",
            "subject_id": session.subject_id,
            "session_id": session.session_id,
            "row_count": session.row_count,
            "window_count": len(windows),
            "source_files": {
                "metadata.json": {"sha256": session.metadata_sha256, "size_bytes": session.metadata_size_bytes},
                "samples.csv": {"sha256": session.samples_sha256, "size_bytes": session.samples_size_bytes},
            },
        })

    source_labels = set().union(*(session.source_labels for session, _ in sessions_and_windows))
    window_labels = {window["label"] for _, windows in sessions_and_windows for window in windows}
    if window_labels != source_labels:
        missing = ", ".join(sorted(source_labels - window_labels))
        raise TrainingDatasetError(f"source labels lack complete windows: {missing}")
    for split_name, split in splits.items():
        split["subject_ids"].sort()
        split["session_ids"].sort()
        split["windows"].sort(key=lambda item: (item["subject_id"], item["session_id"], item["start_row"]))
        labels = {window["label"] for window in split["windows"]}
        if not split["windows"]:
            raise TrainingDatasetError(f"{split_name} split is empty")
        if labels != source_labels:
            missing = ", ".join(sorted(source_labels - labels))
            raise TrainingDatasetError(f"{split_name} split lacks required label coverage: {missing}")

    statistics: dict[str, Any] = {}
    total = Counter()
    for name, split in splits.items():
        counts_by_label = Counter(window["label"] for window in split["windows"])
        total.update(counts_by_label)
        statistics[name] = {
            label: {"window_count": count, "sample_count": count * window_size}
            for label, count in sorted(counts_by_label.items())
        }
    statistics["total"] = {
        label: {"window_count": count, "sample_count": count * window_size}
        for label, count in sorted(total.items())
    }

    first = loaded[0]
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "version": MANIFEST_VERSION,
        "seed": seed,
        "partition_key": "subject_id",
        "group_by": "subject",
        "split_policy": SPLIT_POLICY,
        "split_ratios": dict(zip(SPLIT_NAMES, ratios)),
        "data_contract": {
            "channel_count": first.channel_count,
            "sample_format": first.sample_format,
            "signal_chain": first.signal_chain,
            "notification_packet_protocol": first.notification_protocol,
            "hand_side": first.hand_side,
            "sample_rate": first.sample_rate_descriptor,
        },
        "window": {
            "requested_window_ms": window_ms_value,
            "requested_step_ms": step_ms_value,
            "window_size_samples": window_size,
            "step_size_samples": step_size,
            "effective_window_ms": window_size * 1000.0 / rate_hz,
            "effective_step_ms": step_size * 1000.0 / rate_hz,
            "boundary_policy": "one canonical session and one contiguous action_label+action_phase run",
        },
        "inputs": sorted(inputs, key=lambda item: item["session_locator"]),
        "invalid_input_policy": "reject_entire_dataset",
        "excluded_inputs": [],
        "splits": splits,
        "label_statistics": statistics,
    }
    _publish_manifest(
        destination,
        manifest,
        [session for session, _ in sessions_and_windows],
    )
    return manifest

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare verified modern EMG sessions for leakage-safe training."
    )
    parser.add_argument("sessions", nargs="+", help="modern session directories")
    parser.add_argument("--output", required=True, help="new JSON manifest path")
    parser.add_argument("--window-ms", required=True, type=float)
    parser.add_argument("--step-ms", required=True, type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-ratios",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=DEFAULT_SPLIT_RATIOS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = prepare_training_dataset(
            args.sessions,
            args.output,
            window_ms=args.window_ms,
            step_ms=args.step_ms,
            seed=args.seed,
            split_ratios=args.split_ratios,
        )
    except TrainingDatasetError as exc:
        print(f"training dataset error: {exc}", file=sys.stderr)
        return 2
    window_count = sum(len(split["windows"]) for split in manifest["splits"].values())
    print(
        f"wrote training dataset manifest: {Path(args.output).resolve()} "
        f"({len(manifest['inputs'])} sessions, {window_count} windows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
