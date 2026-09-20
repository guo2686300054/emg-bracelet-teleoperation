"""Fail-closed loading of samples named by a trusted training manifest."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import weakref
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Union

import numpy as np

from training_contract import CANONICAL_LABELS, validate_training_annotation
from training_dataset import (
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    MAX_CSV_BYTES,
    MAX_MANIFEST_WINDOWS,
    MAX_METADATA_BYTES,
    SPLIT_NAMES,
    LoadedSession,
    TrainingDatasetError,
    load_session,
)
from legacy_dataset import SPLIT_POLICY


PathLike = Union[str, Path]
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_SESSION_MATRIX_BYTES = 128 * 1024 * 1024
MAX_TRAINING_WORKING_SET_BYTES = 256 * 1024 * 1024
_REPARSE_POINT = 0x400
_ADMISSION_TOKEN = object()
_ADMITTED_CORPORA: weakref.WeakValueDictionary[int, "TrainingCorpus"] = (
    weakref.WeakValueDictionary()
)


class TrainingSamplesError(TrainingDatasetError):
    """Raised when a manifest cannot safely name a training corpus."""


@dataclass(frozen=True)
class WindowExample:
    samples: np.ndarray
    label: str
    subject_id: str
    session_id: str
    window_id: str
    start_row: int
    end_row_exclusive: int


@dataclass(frozen=True)
class TrainingCorpus:
    manifest_path: Path
    session_root: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    sessions: Mapping[str, LoadedSession]
    provenance: str
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _ADMISSION_TOKEN:
            raise TypeError("TrainingCorpus instances must come from load_training_manifest")

    @property
    def data_contract(self) -> Mapping[str, Any]:
        return self.manifest["data_contract"]

    @property
    def window_contract(self) -> Mapping[str, Any]:
        return self.manifest["window"]


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _freeze_session(session: LoadedSession) -> LoadedSession:
    return replace(
        session,
        sample_rate_descriptor=_deep_freeze(session.sample_rate_descriptor),
        signal_chain=_deep_freeze(session.signal_chain),
        notification_protocol=_deep_freeze(session.notification_protocol),
    )


def _require_admitted(corpus: object) -> TrainingCorpus:
    if (
        not isinstance(corpus, TrainingCorpus)
        or corpus._token is not _ADMISSION_TOKEN
        or _ADMITTED_CORPORA.get(id(corpus)) is not corpus
    ):
        raise TrainingSamplesError("corpus must come from load_training_manifest")
    return corpus


def reload_training_corpus(corpus: object) -> TrainingCorpus:
    """Re-admit disk sources and reject stale or forged corpus handles."""
    admitted = _require_admitted(corpus)
    current = load_training_manifest(admitted.manifest_path, admitted.session_root)
    if current.manifest_sha256 != admitted.manifest_sha256:
        raise TrainingSamplesError("training manifest changed after admission")
    return current


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _safe_existing(path: PathLike, *, kind: str) -> Path:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
        stat_result = candidate.lstat()
    except (OSError, RuntimeError) as exc:
        raise TrainingSamplesError(f"cannot access {kind}: {type(exc).__name__}") from exc
    if candidate.is_symlink() or bool(
        getattr(stat_result, "st_file_attributes", 0) & _REPARSE_POINT
    ):
        raise TrainingSamplesError(f"{kind} must not be a symlink or reparse point")
    return resolved


def _bounded_bytes(path: Path, limit: int) -> tuple[bytes, str]:
    try:
        with path.open("rb") as stream:
            content = stream.read(limit + 1)
    except OSError as exc:
        raise TrainingSamplesError(f"cannot read {path.name}: {type(exc).__name__}") from exc
    if len(content) > limit:
        raise TrainingSamplesError(f"{path.name} exceeds the {limit}-byte resource limit")
    return content, hashlib.sha256(content).hexdigest()


def _bounded_digest(path: Path, limit: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(min(1024 * 1024, limit + 1 - byte_count))
                if not chunk:
                    break
                byte_count += len(chunk)
                if byte_count > limit:
                    raise TrainingSamplesError(
                        f"{path.name} exceeds the {limit}-byte resource limit"
                    )
                digest.update(chunk)
    except TrainingSamplesError:
        raise
    except OSError as exc:
        raise TrainingSamplesError(f"cannot read {path.name}: {type(exc).__name__}") from exc
    return byte_count, digest.hexdigest()


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingSamplesError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], name: str) -> None:
    if set(value) != keys:
        raise TrainingSamplesError(f"{name} has missing or unsupported fields")


def _plain_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TrainingSamplesError(f"{name} must be a positive integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingSamplesError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TrainingSamplesError(f"{name} must be a finite number")
    return result


def _hex_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingSamplesError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _resolve_locator(root: Path, locator: Any) -> Path:
    if not isinstance(locator, str) or not locator or "\\" in locator:
        raise TrainingSamplesError("session_locator must use subject/session POSIX form")
    relative = Path(locator)
    if relative.is_absolute() or len(relative.parts) != 2 or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise TrainingSamplesError("session_locator must be exactly subject/session")
    subject_dir = _safe_existing(root / relative.parts[0], kind="subject directory")
    if subject_dir.parent != root or not subject_dir.is_dir():
        raise TrainingSamplesError("session_locator escapes the explicit session root")
    session_dir = _safe_existing(subject_dir / relative.parts[1], kind="session directory")
    if session_dir.parent != subject_dir or not session_dir.is_dir():
        raise TrainingSamplesError("session_locator escapes the explicit session root")
    return session_dir


def _source_entry(value: Any, name: str) -> tuple[str, int]:
    entry = _object(value, name)
    _exact_keys(entry, {"sha256", "size_bytes"}, name)
    return _hex_digest(entry["sha256"], f"{name}.sha256"), _plain_positive_int(
        entry["size_bytes"], f"{name}.size_bytes"
    )


def _validate_input(
    raw: Any,
    root: Path,
    seen_locators: set[str],
    seen_sessions: set[tuple[str, str]],
    seen_session_ids: set[str],
) -> tuple[str, LoadedSession]:
    item = _object(raw, "manifest input")
    _exact_keys(
        item,
        {"session_locator", "subject_id", "session_id", "row_count", "window_count", "source_files"},
        "manifest input",
    )
    locator = item["session_locator"]
    if not isinstance(locator, str) or locator in seen_locators:
        raise TrainingSamplesError("manifest session_locator is invalid or duplicated")
    seen_locators.add(locator)
    session_dir = _resolve_locator(root, locator)
    try:
        session = load_session(session_dir)
    except TrainingDatasetError as exc:
        raise TrainingSamplesError(f"session {locator!r} failed canonical admission: {exc}") from exc
    identity = (session.subject_id, session.session_id)
    if identity in seen_sessions:
        raise TrainingSamplesError("manifest contains a duplicate session identity")
    seen_sessions.add(identity)
    if session.session_id in seen_session_ids:
        raise TrainingSamplesError("manifest contains a duplicate session_id")
    seen_session_ids.add(session.session_id)
    if item["subject_id"] != session.subject_id or item["session_id"] != session.session_id:
        raise TrainingSamplesError("manifest input identity differs from canonical session")
    if item["row_count"] != session.row_count:
        raise TrainingSamplesError("manifest input row_count differs from canonical session")
    _plain_positive_int(item["window_count"], "manifest input window_count")
    sources = _object(item["source_files"], "manifest input source_files")
    _exact_keys(sources, {"metadata.json", "samples.csv"}, "manifest input source_files")
    metadata_hash, metadata_size = _source_entry(sources["metadata.json"], "metadata.json")
    samples_hash, samples_size = _source_entry(sources["samples.csv"], "samples.csv")
    if (metadata_hash, metadata_size) != (session.metadata_sha256, session.metadata_size_bytes):
        raise TrainingSamplesError("manifest metadata.json snapshot differs from canonical session")
    if (samples_hash, samples_size) != (session.samples_sha256, session.samples_size_bytes):
        raise TrainingSamplesError("manifest samples.csv snapshot differs from canonical session")
    return locator, session


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _validate_data_contract(contract: Any, sessions: Mapping[str, LoadedSession]) -> None:
    value = _object(contract, "data_contract")
    _exact_keys(
        value,
        {
            "channel_count", "sample_format", "signal_chain",
            "notification_packet_protocol", "hand_side", "sample_rate",
        },
        "data_contract",
    )
    for session in sessions.values():
        expected = {
            "channel_count": session.channel_count,
            "sample_format": session.sample_format,
            "signal_chain": session.signal_chain,
            "notification_packet_protocol": session.notification_protocol,
            "hand_side": session.hand_side,
            "sample_rate": session.sample_rate_descriptor,
        }
        if _canonical_json(value) != _canonical_json(expected):
            raise TrainingSamplesError("manifest data_contract differs from canonical session")


def _validate_window_contract(value: Any) -> tuple[int, int]:
    window = _object(value, "window")
    _exact_keys(
        window,
        {
            "requested_window_ms", "requested_step_ms", "window_size_samples",
            "step_size_samples", "effective_window_ms", "effective_step_ms",
            "boundary_policy",
        },
        "window",
    )
    size = _plain_positive_int(window["window_size_samples"], "window_size_samples")
    step = _plain_positive_int(window["step_size_samples"], "step_size_samples")
    for name in ("requested_window_ms", "requested_step_ms", "effective_window_ms", "effective_step_ms"):
        if _finite_number(window[name], name) <= 0:
            raise TrainingSamplesError(f"{name} must be positive")
    if window["boundary_policy"] != "one canonical session and one contiguous action_label+action_phase run":
        raise TrainingSamplesError("unsupported window boundary_policy")
    return size, step


def _validate_splits(
    raw_splits: Any,
    sessions: Mapping[str, LoadedSession],
    window_size: int,
) -> None:
    splits = _object(raw_splits, "splits")
    _exact_keys(splits, set(SPLIT_NAMES), "splits")
    input_identities = {(session.subject_id, session.session_id) for session in sessions.values()}
    subject_sets: dict[str, set[str]] = {}
    window_sets: dict[str, set[str]] = {}
    declared_counts: dict[tuple[str, str], int] = {}
    total_windows = 0
    for split_name in SPLIT_NAMES:
        split = _object(splits[split_name], f"splits.{split_name}")
        _exact_keys(split, {"subject_ids", "session_ids", "windows"}, f"splits.{split_name}")
        subjects = split["subject_ids"]
        session_ids = split["session_ids"]
        windows = split["windows"]
        if not isinstance(subjects, list) or not subjects or len(subjects) != len(set(subjects)):
            raise TrainingSamplesError(f"{split_name} subject_ids are empty or duplicated")
        if not isinstance(session_ids, list) or not session_ids or len(session_ids) != len(set(session_ids)):
            raise TrainingSamplesError(f"{split_name} session_ids are empty or duplicated")
        if not isinstance(windows, list) or not windows:
            raise TrainingSamplesError(f"{split_name} windows are empty")
        subject_sets[split_name] = set(subjects)
        ids: set[str] = set()
        labels: set[str] = set()
        canonical_keys: list[tuple[str, str, int]] = []
        closed_locators: set[str] = set()
        current_locator: str | None = None
        for raw_window in windows:
            window = _object(raw_window, f"{split_name} window")
            _exact_keys(
                window,
                {"window_id", "session_id", "subject_id", "label", "action_phase", "start_row", "end_row_exclusive", "samples_sha256"},
                f"{split_name} window",
            )
            window_id = _hex_digest(window["window_id"], "window_id")
            if window_id in ids:
                raise TrainingSamplesError(f"{split_name} contains duplicate window IDs")
            ids.add(window_id)
            subject_id, session_id = window["subject_id"], window["session_id"]
            if (subject_id, session_id) not in input_identities:
                raise TrainingSamplesError("window names an undeclared session")
            if subject_id not in subject_sets[split_name] or session_id not in session_ids:
                raise TrainingSamplesError("window identity differs from split declarations")
            locator = f"{subject_id}/{session_id}"
            if locator != current_locator:
                if locator in closed_locators:
                    raise TrainingSamplesError(
                        f"{split_name} windows re-enter a previously closed session locator"
                    )
                if current_locator is not None:
                    closed_locators.add(current_locator)
                current_locator = locator
            session = sessions.get(locator)
            if session is None or window["samples_sha256"] != session.samples_sha256:
                raise TrainingSamplesError("window source hash differs from canonical session")
            start = window["start_row"]
            end = window["end_row_exclusive"]
            canonical_keys.append((subject_id, session_id, start))
            if (
                not isinstance(start, int) or isinstance(start, bool) or start < 0
                or not isinstance(end, int) or isinstance(end, bool)
                or end - start != window_size or end > session.row_count
            ):
                raise TrainingSamplesError("window row range is invalid")
            try:
                label, phase = validate_training_annotation(window["label"], window["action_phase"])
            except ValueError as exc:
                raise TrainingSamplesError(f"window annotation is invalid: {exc}") from exc
            if not any(
                run_label == label
                and run_phase == phase
                and run_start <= start
                and end <= run_end
                for run_label, run_phase, run_start, run_end in session.action_runs
            ):
                raise TrainingSamplesError("window crosses or mislabels a canonical action run")
            expected_id = hashlib.sha256(
                f"{session.samples_sha256}:{label}:{phase}:{start}:{end}".encode("utf-8")
            ).hexdigest()
            if window_id != expected_id:
                raise TrainingSamplesError("window_id differs from its canonical window facts")
            labels.add(label)
            declared_counts[(subject_id, session_id)] = declared_counts.get((subject_id, session_id), 0) + 1
        if canonical_keys != sorted(canonical_keys):
            raise TrainingSamplesError(
                f"{split_name} windows are not in canonical subject/session/start_row order"
            )
        if labels != set(CANONICAL_LABELS):
            raise TrainingSamplesError(f"{split_name} lacks canonical label coverage")
        window_sets[split_name] = ids
        total_windows += len(ids)
    if total_windows > MAX_MANIFEST_WINDOWS:
        raise TrainingSamplesError("manifest exceeds the window resource limit")
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            if subject_sets[left] & subject_sets[right]:
                raise TrainingSamplesError("subject IDs overlap across splits")
            if window_sets[left] & window_sets[right]:
                raise TrainingSamplesError("window IDs overlap across splits")
def _validate_complete_window_sets(
    manifest: Mapping[str, Any], sessions: Mapping[str, LoadedSession], size: int, step: int
) -> None:
    actual: dict[str, list[tuple[str, str, int, int, str]]] = {}
    for split in manifest["splits"].values():
        for window in split["windows"]:
            locator = f"{window['subject_id']}/{window['session_id']}"
            actual.setdefault(locator, []).append(
                (
                    window["label"], window["action_phase"], window["start_row"],
                    window["end_row_exclusive"], window["window_id"],
                )
            )
    for locator, session in sessions.items():
        expected: list[tuple[str, str, int, int, str]] = []
        for label, phase, run_start, run_end in session.action_runs:
            for start in range(run_start, run_end - size + 1, step):
                end = start + size
                window_id = hashlib.sha256(
                    f"{session.samples_sha256}:{label}:{phase}:{start}:{end}".encode("utf-8")
                ).hexdigest()
                expected.append((label, phase, start, end, window_id))
        if not expected or actual.get(locator, []) != expected:
            raise TrainingSamplesError("manifest windows are not the complete canonical window set")


def load_training_manifest(manifest_path: PathLike, session_root: PathLike) -> TrainingCorpus:
    """Validate a manifest and every canonical session that it names."""
    manifest_file = _safe_existing(manifest_path, kind="training manifest")
    if not manifest_file.is_file():
        raise TrainingSamplesError("training manifest must be a regular file")
    root = _safe_existing(session_root, kind="session root")
    if not root.is_dir():
        raise TrainingSamplesError("session_root must be a directory")
    content, digest = _bounded_bytes(manifest_file, MAX_MANIFEST_BYTES)
    try:
        manifest = json.loads(
            content.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_object_without_duplicates,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise TrainingSamplesError(f"cannot parse training manifest: {type(exc).__name__}") from exc
    manifest = _object(manifest, "training manifest")
    _exact_keys(
        manifest,
        {
            "schema", "version", "seed", "partition_key", "group_by", "split_policy",
            "split_ratios", "data_contract", "window", "inputs", "invalid_input_policy",
            "excluded_inputs", "splits", "label_statistics",
        },
        "training manifest",
    )
    if manifest["schema"] != MANIFEST_SCHEMA or manifest["version"] != MANIFEST_VERSION:
        raise TrainingSamplesError("unsupported training manifest schema or version")
    if manifest["partition_key"] != "subject_id" or manifest["group_by"] != "subject":
        raise TrainingSamplesError("manifest must be partitioned and grouped by subject")
    if manifest["split_policy"] != SPLIT_POLICY:
        raise TrainingSamplesError("manifest split_policy is not canonical")
    if not isinstance(manifest["seed"], int) or isinstance(manifest["seed"], bool):
        raise TrainingSamplesError("manifest seed must be an integer")
    ratios = _object(manifest["split_ratios"], "split_ratios")
    _exact_keys(ratios, set(SPLIT_NAMES), "split_ratios")
    ratio_values = [_finite_number(ratios[name], f"split_ratios.{name}") for name in SPLIT_NAMES]
    if any(value <= 0 for value in ratio_values) or not math.isclose(
        sum(ratio_values), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise TrainingSamplesError("split_ratios must be positive and sum to one")
    if manifest["invalid_input_policy"] != "reject_entire_dataset" or manifest["excluded_inputs"] != []:
        raise TrainingSamplesError("manifest does not use fail-closed input admission")
    inputs = manifest["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise TrainingSamplesError("manifest inputs must be non-empty")
    sessions: dict[str, LoadedSession] = {}
    seen_locators: set[str] = set()
    seen_sessions: set[tuple[str, str]] = set()
    seen_session_ids: set[str] = set()
    for raw_input in inputs:
        locator, session = _validate_input(
            raw_input, root, seen_locators, seen_sessions, seen_session_ids
        )
        sessions[locator] = session
    _validate_data_contract(manifest["data_contract"], sessions)
    window_size, window_step = _validate_window_contract(manifest["window"])
    _validate_splits(manifest["splits"], sessions, window_size)
    _validate_complete_window_sets(manifest, sessions, window_size, window_step)
    expected_counts = {
        (item["subject_id"], item["session_id"]): item["window_count"] for item in inputs
    }
    actual_counts: dict[tuple[str, str], int] = {}
    for split in manifest["splits"].values():
        for window in split["windows"]:
            key = (window["subject_id"], window["session_id"])
            actual_counts[key] = actual_counts.get(key, 0) + 1
    if actual_counts != expected_counts:
        raise TrainingSamplesError("manifest input window counts disagree with split windows")
    # Detect a mutation of the manifest itself while its sessions were admitted.
    _, final_digest = _bounded_bytes(manifest_file, MAX_MANIFEST_BYTES)
    if final_digest != digest:
        raise TrainingSamplesError("training manifest changed during validation")
    provenances = {session.provenance for session in sessions.values()}
    if len(provenances) != 1:
        raise TrainingSamplesError("manifest mixes incompatible training provenance")
    frozen_sessions = {
        locator: _freeze_session(session) for locator, session in sessions.items()
    }
    corpus = TrainingCorpus(
        manifest_path=manifest_file,
        session_root=root,
        manifest_sha256=digest,
        manifest=_deep_freeze(manifest),
        sessions=MappingProxyType(frozen_sessions),
        provenance=next(iter(provenances)),
        _token=_ADMISSION_TOKEN,
    )
    _ADMITTED_CORPORA[id(corpus)] = corpus
    return corpus


def _load_session_values(
    session: LoadedSession,
    *,
    reserved_bytes: int,
    window_bytes: int,
) -> np.ndarray:
    path = _safe_existing(session.session_dir / "samples.csv", kind="samples.csv")
    if path.parent != session.session_dir or not path.is_file():
        raise TrainingSamplesError("samples.csv must remain a direct regular session child")
    matrix_bytes = session.row_count * session.channel_count * np.dtype(np.float64).itemsize
    if matrix_bytes > MAX_SESSION_MATRIX_BYTES:
        raise TrainingSamplesError("session sample matrix exceeds the memory budget")
    peak_bytes = _session_iteration_peak_bytes(
        session, reserved_bytes=reserved_bytes, window_bytes=window_bytes
    )
    if peak_bytes > MAX_TRAINING_WORKING_SET_BYTES:
        raise TrainingSamplesError("session loading exceeds the training working-set budget")
    before, before_digest = _bounded_bytes(path, session.samples_size_bytes)
    if before_digest != session.samples_sha256 or len(before) != session.samples_size_bytes:
        raise TrainingSamplesError("samples.csv changed after manifest admission")
    rows = np.empty((session.row_count, session.channel_count), dtype=np.float64)
    row_count = 0
    try:
        with io.TextIOWrapper(io.BytesIO(before), encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for row_index, row in enumerate(reader):
                if row_index >= session.row_count:
                    raise TrainingSamplesError("samples.csv grew after manifest admission")
                for channel in range(1, session.channel_count + 1):
                    value = float(row[f"channel_{channel}"])
                    if not math.isfinite(value):
                        raise TrainingSamplesError(
                            "canonical samples contain non-finite values"
                        )
                    rows[row_index, channel - 1] = value
                row_count = row_index + 1
    except TrainingSamplesError:
        raise
    except (OSError, UnicodeError, csv.Error, KeyError, TypeError, ValueError) as exc:
        raise TrainingSamplesError(f"cannot stream canonical samples: {type(exc).__name__}") from exc
    if row_count != session.row_count:
        raise TrainingSamplesError("samples.csv shrank after manifest admission")
    del before
    final_path = _safe_existing(session.session_dir / "samples.csv", kind="samples.csv")
    if final_path != path or final_path.parent != session.session_dir or not final_path.is_file():
        raise TrainingSamplesError("samples.csv path changed while streaming windows")
    after_size, after_digest = _bounded_digest(
        final_path, session.samples_size_bytes
    )
    if after_size != session.samples_size_bytes or after_digest != before_digest:
        raise TrainingSamplesError("samples.csv changed while streaming windows")
    return rows


def _session_iteration_peak_bytes(
    session: LoadedSession,
    *,
    reserved_bytes: int,
    window_bytes: int,
) -> int:
    matrix_bytes = session.row_count * session.channel_count * np.dtype(np.float64).itemsize
    # Public iterator callers commonly retain the previously returned example
    # while requesting the next one.  During that next() call the peak is the
    # session matrix plus either CSV snapshot + old window, or old + new window.
    return reserved_bytes + matrix_bytes + max(
        session.samples_size_bytes + window_bytes,
        2 * window_bytes,
    )


def iter_split_windows(
    corpus: TrainingCorpus,
    split: str,
    *,
    reserved_bytes: int = 0,
) -> Iterator[WindowExample]:
    """Yield only manifest-declared sample ranges for one trusted split."""
    corpus = reload_training_corpus(corpus)
    if split not in SPLIT_NAMES:
        raise TrainingSamplesError("split must be train, validation, or test")
    if not isinstance(reserved_bytes, int) or isinstance(reserved_bytes, bool) or reserved_bytes < 0:
        raise TrainingSamplesError("reserved_bytes must be a non-negative integer")
    window_size = corpus.window_contract["window_size_samples"]
    channel_count = corpus.data_contract["channel_count"]
    window_bytes = window_size * channel_count * np.dtype(np.float64).itemsize
    locators = {
        f"{window['subject_id']}/{window['session_id']}"
        for window in corpus.manifest["splits"][split]["windows"]
    }
    if any(
        _session_iteration_peak_bytes(
            corpus.sessions[locator],
            reserved_bytes=reserved_bytes,
            window_bytes=window_bytes,
        )
        > MAX_TRAINING_WORKING_SET_BYTES
        for locator in locators
    ):
        raise TrainingSamplesError(
            "split iteration exceeds the training working-set budget"
        )
    current_locator: str | None = None
    rows: np.ndarray | None = None
    for window in corpus.manifest["splits"][split]["windows"]:
        locator = f"{window['subject_id']}/{window['session_id']}"
        session = corpus.sessions[locator]
        if locator != current_locator:
            # Release the previous session before evaluating/loading the RHS;
            # otherwise Python keeps the old matrix alive during the call.
            rows = None
            current_locator = None
            rows = _load_session_values(
                session,
                reserved_bytes=reserved_bytes,
                window_bytes=window_bytes,
            )
            current_locator = locator
        assert rows is not None
        start, end = window["start_row"], window["end_row_exclusive"]
        samples = rows[start:end].copy()
        samples.setflags(write=False)
        yield WindowExample(
            samples=samples,
            label=window["label"],
            subject_id=window["subject_id"],
            session_id=window["session_id"],
            window_id=window["window_id"],
            start_row=start,
            end_row_exclusive=end,
        )
        # Do not retain the previous yielded copy when the generator resumes.
        del samples
