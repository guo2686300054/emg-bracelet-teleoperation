"""Fail-closed import and analysis loading for legacy eight-channel EMG data."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

PathLike = Union[str, Path]
ANNOTATION_SCHEMA = "emg_session_annotations"
MEASUREMENT_MANIFEST_SCHEMA = "emg_measurement_manifest"
TRANSFORM_SCHEMA = "emg_legacy_transform"
ANNOTATION_VERSION = "1.1"
MEASUREMENT_MANIFEST_VERSION = "1.1"
TRANSFORM_VERSION = "1.2"
LEGACY_TRANSFORM_VERSION = "1.1"
BATCH_SCHEMA = "emg_legacy_batch_transform"
BATCH_VERSION = "1.1"
LEGACY_BATCH_VERSION = "1.0"
ALGORITHM_ID = "legacy_host_write_factor_run_recovery"
ALGORITHM_VERSION = "1.0"
LEGACY_TRAINING_PROVENANCE = {
    "schema": "emg.training.provenance",
    "version": "1.0",
    "kind": "legacy_experimental",
}
CHANNEL_COLUMNS = tuple(f"channel_{index}" for index in range(1, 9))
DERIVED_COLUMNS = (
    "canonical_index", "raw_start_row", "raw_end_row_exclusive",
    "action_label", "action_phase", "confidence", "training_usable",
    "experimental_analysis_usable", "analysis_scope", "timestamp",
    "sampling_rate_hz", "sampling_rate_status", "device_id", "session_id",
    "split_group_id", "full_session_group_id", "eligible_for_future_training_review",
) + CHANNEL_COLUMNS
SPLIT_POLICY = "group_only_by_split_group_id_and_full_session; random row/window split forbidden"


class LegacyDatasetError(ValueError):
    """Raised when source provenance or a legacy-data contract is invalid."""


@dataclass(frozen=True)
class ImportResult:
    output_dir: Path
    derived_csv: Path
    transform_manifest: Path
    input_rows: int
    output_rows: int
    transform_sha256: str


@dataclass(frozen=True)
class AnalysisSession:
    session_id: str
    split_group_id: str
    full_session_group_id: str
    rows: tuple[Mapping[str, str], ...]


@dataclass(frozen=True)
class ValidatedTransform:
    manifest: Mapping[str, Any]
    rows: tuple[Mapping[str, str], ...]
    source_path: Path
    source_sha256: str
    source_size_bytes: int
    source_row_count: int
    annotation_path: Path
    annotation_sha256: str
    annotation_size_bytes: int
    annotation_status: str


def _validated_measurement_annotations(manifest_path: Path) -> tuple[Path, ...]:
    manifest = _load_json(manifest_path)
    _exact_schema(manifest, MEASUREMENT_MANIFEST_SCHEMA, MEASUREMENT_MANIFEST_VERSION)
    split_group = _text(manifest.get("split_group_id"), "split_group_id")
    raw_policy = _mapping(manifest.get("raw_training_policy"), "raw_training_policy")
    if raw_policy.get("training_usable_raw") is not False:
        raise LegacyDatasetError("measurement manifest must reject raw training use")
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise LegacyDatasetError("measurement manifest sessions must be a non-empty array")
    if manifest.get("session_count") != len(sessions):
        raise LegacyDatasetError("measurement manifest session_count mismatch")
    patient_root = manifest_path.parent.resolve()
    annotations: list[Path] = []
    seen: set[str] = set()
    for index, raw in enumerate(sessions):
        entry = _mapping(raw, f"sessions[{index}]")
        directory = _text(entry.get("relative_directory"), f"sessions[{index}].relative_directory")
        if Path(directory).name != directory:
            raise LegacyDatasetError("relative_directory must be one direct child")
        if entry.get("annotation_file") != "annotations.json":
            raise LegacyDatasetError("annotation_file must be annotations.json")
        annotation_path = (patient_root / directory / "annotations.json").resolve()
        if annotation_path.parent.parent != patient_root:
            raise LegacyDatasetError("annotation path escapes patient_data")
        annotation, _, _ = _validate_annotation(annotation_path)
        session_id = _text(entry.get("session_id"), f"sessions[{index}].session_id")
        if session_id in seen:
            raise LegacyDatasetError(f"duplicate manifest session_id: {session_id}")
        seen.add(session_id)
        expected = {
            "session_id": session_id,
            "status": entry.get("status"),
            "split_group_id": split_group,
            "full_session_group_id": entry.get("full_session_group_id"),
            "source_sha256": entry.get("source_sha256"),
            "source_size_bytes": entry.get("source_size_bytes"),
            "source_row_count": entry.get("source_row_count"),
            "legacy_write_factor": entry.get("legacy_write_factor"),
        }
        if any(annotation.get(field) != value for field, value in expected.items()):
            raise LegacyDatasetError(f"measurement manifest disagrees with annotation for {session_id}")
        if entry.get("training_usable_raw") is not False:
            raise LegacyDatasetError("every manifest session must reject raw training use")
        annotations.append(annotation_path)
    return tuple(annotations)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LegacyDatasetError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 for char in value):
        raise LegacyDatasetError(f"{field} must be non-empty text without control characters")
    return value.strip()


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise LegacyDatasetError(f"{field} must be an integer >= {minimum}")
    return value


def _exact_schema(payload: Mapping[str, Any], schema: str, version: str) -> None:
    if payload.get("schema") != schema or payload.get("version") != version:
        raise LegacyDatasetError(
            f"unsupported schema/version: {payload.get('schema')!r} {payload.get('version')!r}"
        )


def _load_json(path: PathLike) -> Mapping[str, Any]:
    candidate = Path(path).expanduser().resolve()
    try:
        return _mapping(json.loads(candidate.read_text(encoding="utf-8-sig")), "document")
    except LegacyDatasetError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LegacyDatasetError(f"cannot read JSON {candidate}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise LegacyDatasetError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise LegacyDatasetError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_annotation(path: Path) -> tuple[Mapping[str, Any], Path, list[Mapping[str, Any]]]:
    payload = _load_json(path)
    _exact_schema(payload, ANNOTATION_SCHEMA, ANNOTATION_VERSION)
    if payload.get("training_usable_raw") is not False:
        raise LegacyDatasetError("training_usable_raw must be false")
    session_id = _text(payload.get("session_id"), "session_id")
    split_group_id = _text(payload.get("split_group_id"), "split_group_id")
    full_group = _text(payload.get("full_session_group_id"), "full_session_group_id")
    if full_group != f"{split_group_id}:{session_id}":
        raise LegacyDatasetError("full_session_group_id must bind split_group_id to session_id")
    if payload.get("status") not in {"completed", "excluded", "completed_non_target"}:
        raise LegacyDatasetError("unsupported session status")
    factor = _integer(payload.get("legacy_write_factor"), "legacy_write_factor", 1)
    if factor > 1_000_000:
        raise LegacyDatasetError("legacy_write_factor is unreasonably large")

    source_name = _text(payload.get("source_file"), "source_file")
    if source_name != "client_data.csv":
        raise LegacyDatasetError("source_file must be client_data.csv")
    if _text(payload.get("source_directory_name"), "source_directory_name") != path.parent.name:
        raise LegacyDatasetError("source_directory_name does not match annotation directory")
    source = (path.parent / source_name).resolve()
    if source.parent != path.parent.resolve():
        raise LegacyDatasetError("source_file escapes the annotation directory")
    expected_hash = _digest(payload.get("source_sha256"), "source_sha256")
    expected_size = _integer(payload.get("source_size_bytes"), "source_size_bytes")
    expected_rows = _integer(payload.get("source_row_count"), "source_row_count", 1)
    try:
        actual_size = source.stat().st_size
    except OSError as exc:
        raise LegacyDatasetError(f"cannot stat source CSV {source}: {exc}") from exc
    if actual_size != expected_size or _sha256(source) != expected_hash:
        raise LegacyDatasetError("source CSV hash/size does not match annotation")

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise LegacyDatasetError("segments must be a non-empty array")
    segments: list[Mapping[str, Any]] = []
    cursor = 0
    for index, raw in enumerate(raw_segments):
        segment = _mapping(raw, f"segments[{index}]")
        start = _integer(segment.get("start_row"), f"segments[{index}].start_row")
        end = _integer(segment.get("end_row_exclusive"), f"segments[{index}].end_row_exclusive")
        if start != cursor or end <= start or end > expected_rows:
            raise LegacyDatasetError("segments must be non-empty, ordered, contiguous, and in bounds")
        for field in ("action_label", "action_phase", "confidence", "basis"):
            _text(segment.get(field), f"segments[{index}].{field}")
        if segment.get("training_usable_raw") is not False:
            raise LegacyDatasetError("every legacy segment must have training_usable_raw=false")
        if not isinstance(segment.get("eligible_for_future_training_review"), bool):
            raise LegacyDatasetError("eligible_for_future_training_review must be boolean")
        segments.append(segment)
        cursor = end
    if cursor != expected_rows:
        raise LegacyDatasetError("segments must cover every source row")
    return payload, source, segments


def _read_source(path: Path, expected_rows: int) -> list[tuple[str, ...]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            if tuple(next(reader, ())) != CHANNEL_COLUMNS:
                raise LegacyDatasetError("source CSV header must be channel_1 through channel_8")
            rows: list[tuple[str, ...]] = []
            for row_number, row in enumerate(reader, 1):
                if len(row) != 8:
                    raise LegacyDatasetError(f"source row {row_number} must contain exactly 8 channels")
                parsed: list[str] = []
                for column, value in enumerate(row, 1):
                    try:
                        number = int(value)
                    except ValueError as exc:
                        raise LegacyDatasetError(f"source row {row_number} channel {column} is not uint8") from exc
                    if not 0 <= number <= 255 or value.strip() != str(number):
                        raise LegacyDatasetError(f"source row {row_number} channel {column} is not canonical uint8")
                    parsed.append(str(number))
                rows.append(tuple(parsed))
    except LegacyDatasetError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise LegacyDatasetError(f"cannot read source CSV {path}: {exc}") from exc
    if len(rows) != expected_rows:
        raise LegacyDatasetError("source CSV row count does not match annotation")
    return rows


def _segment_at(segments: Sequence[Mapping[str, Any]], index: int, cursor: int) -> int:
    while not (segments[cursor]["start_row"] <= index < segments[cursor]["end_row_exclusive"]):
        cursor += 1
    return cursor


def _derive(payload: Mapping[str, Any], rows: Sequence[tuple[str, ...]], segments: Sequence[Mapping[str, Any]]):
    factor = int(payload["legacy_write_factor"])
    status = str(payload["status"])
    session_scope = "excluded" if status == "excluded" else (
        "artifact_only" if status == "completed_non_target" else "diagnostic_only"
    )
    output: list[dict[str, str]] = []
    mapping: list[dict[str, int]] = []
    run_start = 0
    segment_cursor = 0
    while run_start < len(rows):
        run_end = run_start + 1
        while run_end < len(rows) and rows[run_end] == rows[run_start]:
            run_end += 1
        for start in range(run_start, run_end, factor):
            end = min(start + factor, run_end)
            segment_cursor = _segment_at(segments, start, segment_cursor)
            covered_indexes = {_segment_at(segments, raw_index, segment_cursor) for raw_index in range(start, end)}
            homogeneous = len(covered_indexes) == 1
            segment = segments[min(covered_indexes)]
            future_review = bool(segment["eligible_for_future_training_review"]) and homogeneous
            item = {
                "canonical_index": str(len(output)), "raw_start_row": str(start),
                "raw_end_row_exclusive": str(end),
                "action_label": str(segment["action_label"]) if homogeneous else "unknown",
                "action_phase": str(segment["action_phase"]) if homogeneous else "transition",
                "confidence": str(segment["confidence"]) if homogeneous else "mixed_boundary",
                "training_usable": "false",
                "experimental_analysis_usable": "true" if status != "excluded" else "false",
                "analysis_scope": session_scope,
                "timestamp": "", "sampling_rate_hz": "", "sampling_rate_status": "unknown",
                "device_id": "", "session_id": str(payload["session_id"]),
                "split_group_id": str(payload["split_group_id"]),
                "full_session_group_id": str(payload["full_session_group_id"]),
                "eligible_for_future_training_review": "true" if future_review else "false",
            }
            item.update(dict(zip(CHANNEL_COLUMNS, rows[run_start])))
            output.append(item)
            mapping.append({"raw_start_row": start, "raw_end_row_exclusive": end, "canonical_index": len(output) - 1})
        run_start = run_end
    return output, mapping


def _write_csv(path: Path, rows: Iterable[Mapping[str, str]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DERIVED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    return count


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def import_legacy_dataset(annotation_path: PathLike, output_dir: PathLike) -> ImportResult:
    """Create a non-training, experimental analysis derivative of one legacy session."""
    annotation = Path(annotation_path).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {target}")
    payload, source, segments = _validate_annotation(annotation)
    rows = _read_source(source, int(payload["source_row_count"]))
    derived, row_mapping = _derive(payload, rows, segments)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    try:
        csv_path = temporary / "derived_samples.csv"
        output_count = _write_csv(csv_path, derived)
        manifest = {
            "schema": TRANSFORM_SCHEMA, "version": TRANSFORM_VERSION,
            "algorithm_id": ALGORITHM_ID, "algorithm_version": ALGORITHM_VERSION,
            "legacy_write_factor": payload["legacy_write_factor"],
            "session_id": payload["session_id"], "split_group_id": payload["split_group_id"],
            "full_session_group_id": payload["full_session_group_id"],
            "training_usable": False, "experimental_analysis_usable": payload["status"] != "excluded",
            "training_provenance": LEGACY_TRAINING_PROVENANCE,
            "analysis_scope": "excluded" if payload["status"] == "excluded" else (
                "artifact_only" if payload["status"] == "completed_non_target" else "diagnostic_only"
            ),
            "training_split_constraint": SPLIT_POLICY,
            "source": {"path": str(source), "sha256": payload["source_sha256"],
                       "size_bytes": payload["source_size_bytes"], "row_count": len(rows),
                       "header": list(CHANNEL_COLUMNS)},
            "annotation": {"path": str(annotation), "sha256": _sha256(annotation)},
            "output": {"file": "derived_samples.csv", "sha256": _sha256(csv_path),
                       "size_bytes": csv_path.stat().st_size, "row_count": output_count,
                       "header": list(DERIVED_COLUMNS)},
            "raw_to_canonical_ranges": row_mapping,
            "signal_metadata": {"timestamp": "unknown", "sampling_rate_hz": None,
                                "sampling_rate_status": "unknown", "device_id": "unknown"},
            "limitations": ["packet identity unavailable", "device timestamps unavailable",
                            "sampling rate unconfirmed", "run recovery is experimental only",
                            "not approved for supervised training"],
        }
        manifest_path = temporary / "transform_manifest.json"
        _write_json(manifest_path, manifest)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite output directory: {target}")
        os.rename(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    final_manifest = target / "transform_manifest.json"
    return ImportResult(target, target / "derived_samples.csv", final_manifest,
                        len(rows), len(derived), _sha256(final_manifest))


def import_legacy_measurement_manifest(
    measurement_manifest_path: PathLike, output_root: PathLike
) -> tuple[ImportResult, ...]:
    """Validate a measurement manifest and atomically import all of its sessions."""
    manifest_path = Path(measurement_manifest_path).expanduser().resolve()
    target = Path(output_root).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {target}")
    annotations = _validated_measurement_annotations(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    staged: list[ImportResult] = []
    try:
        for annotation_path in annotations:
            annotation = _load_json(annotation_path)
            staged.append(import_legacy_dataset(annotation_path, temporary / str(annotation["session_id"])))
        source_manifest = _load_json(manifest_path)
        source_entries = {entry["session_id"]: entry for entry in source_manifest["sessions"]}
        batch_entries = []
        for item, annotation_path in zip(staged, annotations):
            annotation = _load_json(annotation_path)
            session_id = str(annotation["session_id"])
            source_path = annotation_path.parent / "client_data.csv"
            child_manifest = item.transform_manifest
            child_csv = item.derived_csv
            batch_entries.append({
                "session_id": session_id,
                "full_session_group_id": annotation["full_session_group_id"],
                "legacy_write_factor": annotation["legacy_write_factor"],
                "source_csv": {"relative_path": os.path.relpath(source_path, target),
                               "sha256": _sha256(source_path), "size_bytes": source_path.stat().st_size,
                               "row_count": annotation["source_row_count"]},
                "annotation": {"relative_path": os.path.relpath(annotation_path, target),
                               "sha256": _sha256(annotation_path), "size_bytes": annotation_path.stat().st_size},
                "transform_manifest": {"relative_path": f"{session_id}/transform_manifest.json",
                                       "sha256": _sha256(child_manifest), "size_bytes": child_manifest.stat().st_size},
                "derived_csv": {"relative_path": f"{session_id}/derived_samples.csv",
                                "sha256": _sha256(child_csv), "size_bytes": child_csv.stat().st_size,
                                "row_count": item.output_rows},
                "status": source_entries[session_id]["status"],
            })
        _write_json(temporary / "dataset_manifest.json", {
            "schema": BATCH_SCHEMA, "version": BATCH_VERSION,
            "training_usable": False,
            "training_provenance": LEGACY_TRAINING_PROVENANCE,
            "source_measurement_manifest": {
                "relative_path": os.path.relpath(manifest_path, target),
                "sha256": _sha256(manifest_path), "size_bytes": manifest_path.stat().st_size,
            },
            "session_count": len(staged),
            "session_ids": [item.output_dir.name for item in staged],
            "split_group_id": source_manifest["split_group_id"],
            "partition_key": "split_group_id",
            "full_session_group_id_role": "nested_integrity_only",
            "split_policy": "subject-level split_group_id only; random row/window/full-session top-level partition forbidden",
            "allowed_legacy_write_factors": list(range(1, 8)),
            "legacy_write_factor_sequence": [entry["legacy_write_factor"] for entry in batch_entries],
            "sessions": batch_entries,
        })
        if target.exists():
            raise FileExistsError(f"refusing to overwrite output directory: {target}")
        os.rename(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    results = []
    for item in staged:
        final_dir = target / item.output_dir.name
        final_manifest = final_dir / "transform_manifest.json"
        results.append(ImportResult(final_dir, final_dir / "derived_samples.csv", final_manifest,
                                    item.input_rows, item.output_rows, _sha256(final_manifest)))
    return tuple(results)


def _verified_file(reference: Mapping[str, Any], base: Path, field: str) -> Path:
    relative = _text(reference.get("relative_path"), f"{field}.relative_path")
    if Path(relative).is_absolute():
        raise LegacyDatasetError(f"{field}.relative_path must be relative")
    path = (base / relative).resolve()
    size = _integer(reference.get("size_bytes"), f"{field}.size_bytes")
    digest = _digest(reference.get("sha256"), f"{field}.sha256")
    if not path.is_file() or path.stat().st_size != size or _sha256(path) != digest:
        raise LegacyDatasetError(f"{field} is missing or its hash/size does not match")
    return path


def _validate_transform(path: Path) -> ValidatedTransform:
    manifest = _load_json(path)
    if manifest.get("schema") != TRANSFORM_SCHEMA or manifest.get("version") not in {
        LEGACY_TRANSFORM_VERSION, TRANSFORM_VERSION,
    }:
        raise LegacyDatasetError("unsupported transform schema/version")
    if manifest.get("training_usable") is not False:
        raise LegacyDatasetError("legacy transform must not claim training usability")
    if (manifest.get("version") == TRANSFORM_VERSION
            and manifest.get("training_provenance") != LEGACY_TRAINING_PROVENANCE):
        raise LegacyDatasetError("legacy transform must declare legacy_experimental provenance")
    if (manifest.get("version") == LEGACY_TRANSFORM_VERSION
            and manifest.get("training_provenance") not in (None, LEGACY_TRAINING_PROVENANCE)):
        raise LegacyDatasetError("legacy transform contains incompatible provenance")
    if manifest.get("algorithm_id") != ALGORITHM_ID or manifest.get("algorithm_version") != ALGORITHM_VERSION:
        raise LegacyDatasetError("unsupported transform algorithm")
    if manifest.get("training_split_constraint") != SPLIT_POLICY:
        raise LegacyDatasetError("transform does not enforce grouped splits")
    for field in ("session_id", "split_group_id", "full_session_group_id"):
        _text(manifest.get(field), field)
    source = _mapping(manifest.get("source"), "source")
    source_path = Path(_text(source.get("path"), "source.path")).resolve()
    if source.get("header") != list(CHANNEL_COLUMNS):
        raise LegacyDatasetError("source header contract mismatch")
    if (not source_path.is_file() or source_path.stat().st_size != _integer(source.get("size_bytes"), "source.size_bytes")
            or _sha256(source_path) != _digest(source.get("sha256"), "source.sha256")):
        raise LegacyDatasetError("source CSV is missing or its hash/size does not match")
    annotation_ref = _mapping(manifest.get("annotation"), "annotation")
    annotation_path = Path(_text(annotation_ref.get("path"), "annotation.path")).resolve()
    if not annotation_path.is_file() or _sha256(annotation_path) != _digest(annotation_ref.get("sha256"), "annotation.sha256"):
        raise LegacyDatasetError("annotation is missing or its hash does not match")
    annotation, validated_source, segments = _validate_annotation(annotation_path)
    if validated_source != source_path:
        raise LegacyDatasetError("transform source path disagrees with annotation")
    for field in ("session_id", "split_group_id", "full_session_group_id", "legacy_write_factor"):
        if manifest.get(field) != annotation.get(field):
            raise LegacyDatasetError(f"transform {field} disagrees with annotation")
    source_row_count = _integer(source.get("row_count"), "source.row_count", 1)
    if (source_row_count != annotation.get("source_row_count")
            or source.get("sha256") != annotation.get("source_sha256")
            or source.get("size_bytes") != annotation.get("source_size_bytes")):
        raise LegacyDatasetError("transform source facts disagree with annotation")
    output = _mapping(manifest.get("output"), "output")
    if output.get("file") != "derived_samples.csv" or output.get("header") != list(DERIVED_COLUMNS):
        raise LegacyDatasetError("derived output contract mismatch")
    csv_path = path.parent / "derived_samples.csv"
    if (not csv_path.is_file() or csv_path.stat().st_size != _integer(output.get("size_bytes"), "output.size_bytes")
            or _sha256(csv_path) != _digest(output.get("sha256"), "output.sha256")):
        raise LegacyDatasetError("derived CSV is missing or its hash/size does not match")
    mapping_value = manifest.get("raw_to_canonical_ranges")
    if not isinstance(mapping_value, list) or len(mapping_value) != output.get("row_count"):
        raise LegacyDatasetError("row mapping count mismatch")
    factor = _integer(manifest.get("legacy_write_factor"), "legacy_write_factor", 1)
    expected_scope = "excluded" if annotation["status"] == "excluded" else (
        "artifact_only" if annotation["status"] == "completed_non_target" else "diagnostic_only"
    )
    if manifest.get("analysis_scope") != expected_scope or manifest.get("experimental_analysis_usable") is not (annotation["status"] != "excluded"):
        raise LegacyDatasetError("transform analysis policy disagrees with session status")
    signal_metadata = _mapping(manifest.get("signal_metadata"), "signal_metadata")
    if signal_metadata != {"timestamp": "unknown", "sampling_rate_hz": None,
                           "sampling_rate_status": "unknown", "device_id": "unknown"}:
        raise LegacyDatasetError("transform must preserve unknown signal metadata")
    output_row_count = _integer(output.get("row_count"), "output.row_count", 1)
    loaded_rows: list[Mapping[str, str]] = []
    try:
        with source_path.open("r", encoding="utf-8-sig", newline="") as source_stream, csv_path.open("r", encoding="utf-8", newline="") as derived_stream:
            source_reader = csv.reader(source_stream)
            derived_reader = csv.DictReader(derived_stream)
            if tuple(next(source_reader, ())) != CHANNEL_COLUMNS or tuple(derived_reader.fieldnames or ()) != DERIVED_COLUMNS:
                raise LegacyDatasetError("source or derived CSV header mismatch")
            raw_cursor = 0
            segment_cursor = 0
            previous_values: tuple[str, ...] | None = None
            previous_span = factor
            for canonical_index, (row, row_map) in enumerate(zip(derived_reader, mapping_value)):
                if not isinstance(row_map, Mapping):
                    raise LegacyDatasetError("row mapping entry must be an object")
                start = _integer(row_map.get("raw_start_row"), "raw_start_row")
                end = _integer(row_map.get("raw_end_row_exclusive"), "raw_end_row_exclusive")
                if row_map.get("canonical_index") != canonical_index or start != raw_cursor or not start < end or end - start > factor:
                    raise LegacyDatasetError("row mapping is not contiguous, bounded, and canonical")
                if row["canonical_index"] != str(canonical_index) or row["raw_start_row"] != str(start) or row["raw_end_row_exclusive"] != str(end):
                    raise LegacyDatasetError("derived canonical/raw range fields disagree with mapping")
                raw_values = []
                for _ in range(end - start):
                    raw = next(source_reader, None)
                    if raw is None or len(raw) != 8:
                        raise LegacyDatasetError("source row count or channel count disagrees with mapping")
                    if any(not value.isdigit() or value != str(int(value)) or not 0 <= int(value) <= 255 for value in raw):
                        raise LegacyDatasetError("source contains a non-canonical uint8 value")
                    raw_values.append(tuple(raw))
                if not raw_values or any(values != raw_values[0] for values in raw_values):
                    raise LegacyDatasetError("mapping combines different source vectors")
                derived_values = tuple(row[column] for column in CHANNEL_COLUMNS)
                if derived_values != raw_values[0] or any(not value.isdigit() or not 0 <= int(value) <= 255 for value in derived_values):
                    raise LegacyDatasetError("derived uint8 values disagree with source")
                if previous_values == derived_values and previous_span < factor:
                    raise LegacyDatasetError("mapping splits a maximal run after an incomplete factor chunk")
                if (row["session_id"], row["split_group_id"], row["full_session_group_id"]) != (
                    manifest["session_id"], manifest["split_group_id"], manifest["full_session_group_id"]
                ) or row["training_usable"] != "false" or row["experimental_analysis_usable"] != (
                    "false" if annotation["status"] == "excluded" else "true"
                ):
                    raise LegacyDatasetError("derived row identity or training policy mismatch")
                if row["analysis_scope"] != expected_scope or row["timestamp"] or row["sampling_rate_hz"] or row["sampling_rate_status"] != "unknown" or row["device_id"]:
                    raise LegacyDatasetError("derived scope or unknown signal metadata mismatch")
                segment_cursor = _segment_at(segments, start, segment_cursor)
                covered_segments = {_segment_at(segments, raw_index, segment_cursor) for raw_index in range(start, end)}
                segment = segments[min(covered_segments)]
                homogeneous = len(covered_segments) == 1
                expected_labels = (
                    str(segment["action_label"]) if homogeneous else "unknown",
                    str(segment["action_phase"]) if homogeneous else "transition",
                    str(segment["confidence"]) if homogeneous else "mixed_boundary",
                    "true" if homogeneous and segment["eligible_for_future_training_review"] else "false",
                )
                if (row["action_label"], row["action_phase"], row["confidence"],
                    row["eligible_for_future_training_review"]) != expected_labels:
                    raise LegacyDatasetError("derived labels disagree with annotation segments")
                loaded_rows.append(row)
                previous_values, previous_span, raw_cursor = derived_values, end - start, end
            if next(derived_reader, None) is not None or next(source_reader, None) is not None:
                raise LegacyDatasetError("source/derived row count exceeds manifest or mapping")
    except LegacyDatasetError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise LegacyDatasetError(f"cannot stream-validate transform: {exc}") from exc
    if raw_cursor != source_row_count or len(loaded_rows) != output_row_count:
        raise LegacyDatasetError("source/output row count mismatch")
    normalized_manifest = dict(manifest)
    normalized_manifest["training_provenance"] = dict(LEGACY_TRAINING_PROVENANCE)
    return ValidatedTransform(
        manifest=normalized_manifest,
        rows=tuple(loaded_rows),
        source_path=source_path,
        source_sha256=str(source["sha256"]),
        source_size_bytes=int(source["size_bytes"]),
        source_row_count=source_row_count,
        annotation_path=annotation_path,
        annotation_sha256=str(annotation_ref["sha256"]),
        annotation_size_bytes=annotation_path.stat().st_size,
        annotation_status=str(annotation["status"]),
    )


def validate_legacy_transform(manifest_path: PathLike) -> ValidatedTransform:
    """Strictly validate a transform and normalize fully proven legacy v1.1 provenance."""
    return _validate_transform(Path(manifest_path).expanduser().resolve())


def _validate_batch_manifest(path: Path) -> tuple[Mapping[str, Any], tuple[ValidatedTransform, ...]]:
    batch = _load_json(path)
    if batch.get("schema") != BATCH_SCHEMA or batch.get("version") not in {
        LEGACY_BATCH_VERSION, BATCH_VERSION,
    }:
        raise LegacyDatasetError("unsupported batch transform schema/version")
    if (batch.get("version") == BATCH_VERSION and (
            batch.get("training_usable") is not False
            or batch.get("training_provenance") != LEGACY_TRAINING_PROVENANCE)):
        raise LegacyDatasetError("legacy batch must be non-training legacy_experimental data")
    if (batch.get("version") == LEGACY_BATCH_VERSION and (
            batch.get("training_usable") not in (None, False)
            or batch.get("training_provenance") not in (None, LEGACY_TRAINING_PROVENANCE))):
        raise LegacyDatasetError("legacy batch contains incompatible training claims")
    if path.name != "dataset_manifest.json":
        raise LegacyDatasetError("official loader requires dataset_manifest.json")
    if batch.get("partition_key") != "split_group_id" or batch.get("full_session_group_id_role") != "nested_integrity_only":
        raise LegacyDatasetError("top-level partition must use split_group_id only")
    if batch.get("allowed_legacy_write_factors") != list(range(1, 8)) or batch.get("legacy_write_factor_sequence") != list(range(1, 8)):
        raise LegacyDatasetError("this batch must explicitly bind legacy write factors 1 through 7")
    source_manifest_ref = _mapping(batch.get("source_measurement_manifest"), "source_measurement_manifest")
    source_manifest_path = _verified_file(source_manifest_ref, path.parent, "source_measurement_manifest")
    source_annotations = _validated_measurement_annotations(source_manifest_path)
    source_by_session = {_load_json(item)["session_id"]: item for item in source_annotations}
    sessions = batch.get("sessions")
    if not isinstance(sessions, list) or batch.get("session_count") != 7 or len(sessions) != 7:
        raise LegacyDatasetError("batch must contain exactly seven sessions")
    session_ids = batch.get("session_ids")
    if not isinstance(session_ids, list) or len(set(session_ids)) != 7 or set(session_ids) != set(source_by_session):
        raise LegacyDatasetError("batch session set does not match source measurement manifest")
    if [entry.get("session_id") for entry in sessions if isinstance(entry, Mapping)] != session_ids:
        raise LegacyDatasetError("batch session_ids must exactly match ordered session entries")
    if [entry.get("legacy_write_factor") for entry in sessions if isinstance(entry, Mapping)] != batch["legacy_write_factor_sequence"]:
        raise LegacyDatasetError("batch session factors must match the declared factor sequence")
    expected_files = {"dataset_manifest.json"}
    validated = []
    seen: set[str] = set()
    for index, raw in enumerate(sessions):
        entry = _mapping(raw, f"sessions[{index}]")
        session_id = _text(entry.get("session_id"), f"sessions[{index}].session_id")
        if session_id in seen or session_id not in source_by_session:
            raise LegacyDatasetError("batch contains duplicate or unknown session")
        seen.add(session_id)
        source_csv = _verified_file(_mapping(entry.get("source_csv"), "source_csv"), path.parent, "source_csv")
        annotation_path = _verified_file(_mapping(entry.get("annotation"), "annotation"), path.parent, "annotation")
        child_manifest = _verified_file(_mapping(entry.get("transform_manifest"), "transform_manifest"), path.parent, "transform_manifest")
        child_csv = _verified_file(_mapping(entry.get("derived_csv"), "derived_csv"), path.parent, "derived_csv")
        if annotation_path != source_by_session[session_id] or source_csv != annotation_path.parent / "client_data.csv":
            raise LegacyDatasetError("batch source/annotation path disagrees with source measurement manifest")
        validated_child = _validate_transform(child_manifest)
        child = validated_child.manifest
        rows = validated_child.rows
        source_ref = _mapping(entry.get("source_csv"), "source_csv")
        annotation_ref = _mapping(entry.get("annotation"), "annotation")
        if child_csv != child_manifest.parent / "derived_samples.csv" or child.get("session_id") != session_id:
            raise LegacyDatasetError("batch child paths or session identity disagree")
        derived_ref = _mapping(entry.get("derived_csv"), "derived_csv")
        if (validated_child.source_path != source_csv
                or validated_child.source_sha256 != source_ref.get("sha256")
                or validated_child.source_size_bytes != source_ref.get("size_bytes")
                or validated_child.source_row_count != source_ref.get("row_count")
                or validated_child.annotation_path != annotation_path
                or validated_child.annotation_sha256 != annotation_ref.get("sha256")
                or validated_child.annotation_size_bytes != annotation_ref.get("size_bytes")
                or entry.get("legacy_write_factor") != child.get("legacy_write_factor")
                or entry.get("full_session_group_id") != child.get("full_session_group_id")
                or entry.get("status") != validated_child.annotation_status
                or _integer(source_ref.get("row_count"), "source_csv.row_count", 1) != child["source"]["row_count"]
                or _integer(derived_ref.get("row_count"), "derived_csv.row_count", 1) != len(rows)):
            raise LegacyDatasetError("batch/root source, annotation, session, or factor identity disagrees with child")
        for internal in (child_manifest, child_csv):
            try:
                expected_files.add(internal.relative_to(path.parent).as_posix())
            except ValueError as exc:
                raise LegacyDatasetError("batch child artifacts must stay inside dataset root") from exc
        validated.append(validated_child)
    actual_files = {item.relative_to(path.parent).as_posix() for item in path.parent.rglob("*") if item.is_file()}
    if actual_files != expected_files:
        raise LegacyDatasetError("batch artifact closure has missing or unexpected files")
    actual_directories = {item.relative_to(path.parent).as_posix() for item in path.parent.iterdir() if item.is_dir()}
    if actual_directories != set(session_ids):
        raise LegacyDatasetError("batch artifact closure has missing or unexpected session directories")
    normalized_batch = dict(batch)
    normalized_batch["training_usable"] = False
    normalized_batch["training_provenance"] = dict(LEGACY_TRAINING_PROVENANCE)
    return normalized_batch, tuple(validated)


def load_experimental_analysis_groups(dataset_manifest_path: PathLike) -> Mapping[str, tuple[AnalysisSession, ...]]:
    """Load verified diagnostic derivatives, preserving subject/session grouping."""
    dataset_path = Path(dataset_manifest_path).expanduser().resolve()
    batch, transforms = _validate_batch_manifest(dataset_path)
    groups: dict[str, list[AnalysisSession]] = {}
    for validated in transforms:
        manifest, all_rows = validated.manifest, validated.rows
        if manifest.get("experimental_analysis_usable") is not True:
            continue
        usable = tuple(row for row in all_rows
                       if row["experimental_analysis_usable"] == "true"
                       and row["action_phase"].lower() not in {"transition", "excluded"}
                       and row["action_label"].lower() not in {"transition", "excluded", "unknown", ""})
        session_id = str(manifest["session_id"])
        split_group = str(manifest["split_group_id"])
        full_group = str(manifest["full_session_group_id"])
        if split_group != batch["split_group_id"]:
            raise LegacyDatasetError("child split_group_id disagrees with batch")
        groups.setdefault(split_group, []).append(AnalysisSession(session_id, split_group, full_group, usable))
    return {key: tuple(value) for key, value in groups.items()}


def load_training_groups(*args: Any, **kwargs: Any) -> Mapping[str, tuple[AnalysisSession, ...]]:
    """Fail closed: current legacy derivatives are not approved training inputs."""
    raise LegacyDatasetError(
        "legacy raw and run-recovered derivatives are not training usable; "
        "use load_experimental_analysis_groups for diagnostics only"
    )
