"""Strict JSON contract and NumPy inference for the EMG shrinkage-LDA model."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import warnings
from dataclasses import asdict
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from emg_features import FEATURE_NAMES, FeatureSpec
from emg_protocol import HandSide, NotificationPacketProtocol, RateDescriptor, SignalChain
from training_contract import (
    CANONICAL_LABELS,
    EMG_CHANNEL_COUNT,
    validate_sample_rate_source_kind,
)


BUNDLE_SCHEMA = "emg.lda.bundle"
BUNDLE_VERSION = "1.0"
DIGEST_SCHEMA = "emg.lda.bundle.digests"
DIGEST_VERSION = "1.0"
MODEL_FILENAME = "model.json"
EVALUATION_FILENAME = "evaluation.json"
DIGEST_FILENAME = "digests.json"
_EXPECTED_FILES = frozenset({MODEL_FILENAME, EVALUATION_FILENAME, DIGEST_FILENAME})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_BUNDLE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MAX_MODEL_BYTES = 4 * 1024 * 1024
_MAX_EVALUATION_BYTES = 2 * 1024 * 1024
_MAX_DIGEST_BYTES = 64 * 1024
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

EVALUATION_SCHEMA = "emg.training.evaluation"
EVALUATION_VERSION = "1.1"
_SPLIT_NAMES = ("train", "validation", "test")
MAX_REPORTED_ERROR_WINDOWS = 50
_METRIC_FIELDS = {
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "per_class",
    "confusion_matrix",
    "window_count",
    "error_count",
    "error_windows",
}
_ERROR_WINDOW_FIELDS = {
    "true_label",
    "predicted_label",
    "confidence",
    "subject_id",
    "session_id",
    "window_id",
    "start_row",
    "end_row_exclusive",
}
_CLASS_METRIC_FIELDS = {"precision", "recall", "f1", "support"}

_TOP_LEVEL_FIELDS = {
    "schema",
    "version",
    "bundle_id",
    "labels",
    "channel_count",
    "sample_rate",
    "hand_side",
    "window",
    "feature_spec",
    "sample_format",
    "signal_chain",
    "notification_protocol",
    "scaler",
    "lda",
    "package_versions",
    "manifest_sha256",
    "source_hashes",
    "seed",
    "provenance",
    "deployment_status",
}


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _validate_json_tree(value: Any, *, depth: int = 0, budget: list[int]) -> None:
    if depth > 20:
        raise ValueError("JSON nesting is too deep")
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("JSON contains too many values")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        if len(value) > 16_384:
            raise ValueError("JSON string is too long")
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_tree(item, depth=depth + 1, budget=budget)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            _validate_json_tree(key, depth=depth + 1, budget=budget)
            _validate_json_tree(item, depth=depth + 1, budget=budget)
        return
    raise ValueError(f"value of type {type(value).__name__} is not JSON-safe")


def _parse_json_bytes(raw: bytes, name: str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid bounded UTF-8 JSON in {name!r}: {exc}") from exc
    _validate_json_tree(value, budget=[200_000])
    return value


def _is_reparse_point(file_stat: os.stat_result) -> bool:
    return bool(getattr(file_stat, "st_file_attributes", 0) & _REPARSE_POINT)


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _read_bounded_regular_file(
    path: Path, max_bytes: int, expected_stat: os.stat_result
) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot safely open bundle file {path.name!r}: {exc}") from exc
    try:
        opened_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or _is_reparse_point(opened_stat)
            or not _same_file(expected_stat, opened_stat)
            or expected_stat.st_size != opened_stat.st_size
            or expected_stat.st_mtime_ns != opened_stat.st_mtime_ns
        ):
            raise ValueError(f"bundle file {path.name!r} was replaced or is not a regular file")
        if opened_stat.st_size > max_bytes:
            raise ValueError(f"bundle file {path.name!r} exceeds the size limit")
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            raw = stream.read(max_bytes + 1)
            final_stat = os.fstat(stream.fileno())
        if len(raw) > max_bytes:
            raise ValueError(f"bundle file {path.name!r} exceeds the size limit")
        if (
            not _same_file(opened_stat, final_stat)
            or opened_stat.st_size != final_stat.st_size
            or opened_stat.st_mtime_ns != final_stat.st_mtime_ns
            or len(raw) != opened_stat.st_size
        ):
            raise ValueError(f"bundle file {path.name!r} changed while it was being read")
        return raw, opened_stat
    except OSError as exc:
        raise ValueError(f"cannot read bundle file {path.name!r}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _canonical_json_bytes(value: Any) -> bytes:
    _validate_json_tree(value, budget=[200_000])
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload is not finite JSON: {exc}") from exc
    return encoded


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_exact_keys(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} has missing or unknown fields")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _unit_interval(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _string(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _plain_json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    _validate_json_tree(value, budget=[20_000])
    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


def _finite_vector(value: Any, length: int, name: str, *, positive: bool = False) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if any(isinstance(item, (bool, np.bool_)) for item in value):
        raise ValueError(f"{name} cannot contain booleans")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite numeric values")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} values must be strictly positive")
    result.setflags(write=False)
    return result


def _finite_matrix(value: Any, rows: int, columns: int, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{name} must have {rows} rows")
    if any(not isinstance(row, list) or len(row) != columns for row in value):
        raise ValueError(f"{name} must have shape ({rows}, {columns})")
    if any(isinstance(item, (bool, np.bool_)) for row in value for item in row):
        raise ValueError(f"{name} cannot contain booleans")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if result.shape != (rows, columns) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite numeric values")
    result.setflags(write=False)
    return result


def _validate_evaluation(value: Any) -> dict[str, Any]:
    evaluation = _require_exact_keys(
        value,
        {"schema", "version", "labels", "validation_role", "splits"},
        "evaluation",
    )
    if (
        evaluation["schema"] != EVALUATION_SCHEMA
        or evaluation["version"] != EVALUATION_VERSION
    ):
        raise ValueError("unsupported evaluation schema or version")
    if not isinstance(evaluation["labels"], list) or tuple(evaluation["labels"]) != CANONICAL_LABELS:
        raise ValueError(f"evaluation labels must be exactly {CANONICAL_LABELS!r}")
    validation_role = _string(evaluation["validation_role"], "evaluation.validation_role")
    if validation_role != "informational_no_hyperparameter_tuning":
        raise ValueError("unsupported evaluation validation_role")
    splits = _require_exact_keys(evaluation["splits"], set(_SPLIT_NAMES), "evaluation.splits")
    normalized_splits: dict[str, Any] = {}
    for split_name in _SPLIT_NAMES:
        metrics = _require_exact_keys(
            splits[split_name], _METRIC_FIELDS, f"evaluation.splits.{split_name}"
        )
        window_count = _positive_int(
            metrics["window_count"], f"evaluation.splits.{split_name}.window_count"
        )
        per_class = _require_exact_keys(
            metrics["per_class"], set(CANONICAL_LABELS),
            f"evaluation.splits.{split_name}.per_class",
        )
        normalized_per_class: dict[str, Any] = {}
        supports: list[int] = []
        for label in CANONICAL_LABELS:
            class_metrics = _require_exact_keys(
                per_class[label], _CLASS_METRIC_FIELDS,
                f"evaluation.splits.{split_name}.per_class.{label}",
            )
            support = _positive_int(
                class_metrics["support"],
                f"evaluation.splits.{split_name}.per_class.{label}.support",
            )
            supports.append(support)
            normalized_per_class[label] = {
                metric: _unit_interval(
                    class_metrics[metric],
                    f"evaluation.splits.{split_name}.per_class.{label}.{metric}",
                )
                for metric in ("precision", "recall", "f1")
            }
            normalized_per_class[label]["support"] = support
        if sum(supports) != window_count:
            raise ValueError(f"evaluation {split_name} supports do not sum to window_count")

        matrix_value = metrics["confusion_matrix"]
        if (
            not isinstance(matrix_value, list)
            or len(matrix_value) != len(CANONICAL_LABELS)
            or any(not isinstance(row, list) or len(row) != len(CANONICAL_LABELS) for row in matrix_value)
        ):
            raise ValueError(f"evaluation {split_name} confusion_matrix has invalid shape")
        matrix = [
            [
                _nonnegative_int(cell, f"evaluation.splits.{split_name}.confusion_matrix")
                for cell in row
            ]
            for row in matrix_value
        ]
        if sum(sum(row) for row in matrix) != window_count:
            raise ValueError(f"evaluation {split_name} confusion_matrix total is inconsistent")
        if [sum(row) for row in matrix] != supports:
            raise ValueError(f"evaluation {split_name} confusion_matrix rows disagree with support")

        expected_error_count = window_count - sum(
            matrix[index][index] for index in range(len(matrix))
        )
        error_count = _nonnegative_int(
            metrics["error_count"], f"evaluation.splits.{split_name}.error_count"
        )
        if error_count != expected_error_count:
            raise ValueError(
                f"evaluation {split_name} error_count disagrees with confusion_matrix"
            )
        error_windows_value = metrics["error_windows"]
        if not isinstance(error_windows_value, list):
            raise ValueError(f"evaluation {split_name} error_windows must be a list")
        expected_reported_count = min(error_count, MAX_REPORTED_ERROR_WINDOWS)
        if len(error_windows_value) != expected_reported_count:
            raise ValueError(
                f"evaluation {split_name} error_windows must contain exactly "
                f"{expected_reported_count} bounded entries"
            )
        normalized_error_windows: list[dict[str, Any]] = []
        seen_window_ids: set[str] = set()
        for index, raw_error in enumerate(error_windows_value):
            name = f"evaluation.splits.{split_name}.error_windows[{index}]"
            error = _require_exact_keys(raw_error, _ERROR_WINDOW_FIELDS, name)
            true_label = _string(error["true_label"], f"{name}.true_label")
            predicted_label = _string(
                error["predicted_label"], f"{name}.predicted_label"
            )
            if true_label not in CANONICAL_LABELS or predicted_label not in CANONICAL_LABELS:
                raise ValueError(f"{name} labels must be canonical")
            if true_label == predicted_label:
                raise ValueError(f"{name} must describe a misclassification")
            subject_id = _string(error["subject_id"], f"{name}.subject_id")
            session_id = _string(error["session_id"], f"{name}.session_id")
            window_id = _string(error["window_id"], f"{name}.window_id")
            if any(len(value) > 256 for value in (subject_id, session_id, window_id)):
                raise ValueError(f"{name} locator exceeds 256 characters")
            if window_id in seen_window_ids:
                raise ValueError(f"{name}.window_id is duplicated")
            seen_window_ids.add(window_id)
            start_row = _nonnegative_int(error["start_row"], f"{name}.start_row")
            end_row = _positive_int(
                error["end_row_exclusive"], f"{name}.end_row_exclusive"
            )
            if end_row <= start_row:
                raise ValueError(f"{name} row bounds are invalid")
            normalized_error_windows.append(
                {
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": _unit_interval(
                        error["confidence"], f"{name}.confidence"
                    ),
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "window_id": window_id,
                    "start_row": start_row,
                    "end_row_exclusive": end_row,
                }
            )

        for index, label in enumerate(CANONICAL_LABELS):
            true_positive = matrix[index][index]
            predicted = sum(row[index] for row in matrix)
            expected_precision = true_positive / predicted if predicted else 0.0
            expected_recall = true_positive / supports[index]
            expected_f1 = (
                2.0 * expected_precision * expected_recall
                / (expected_precision + expected_recall)
                if expected_precision + expected_recall
                else 0.0
            )
            for metric_name, expected in (
                ("precision", expected_precision),
                ("recall", expected_recall),
                ("f1", expected_f1),
            ):
                if not math.isclose(
                    normalized_per_class[label][metric_name],
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"evaluation {split_name} {label} {metric_name} is inconsistent"
                    )

        accuracy = _unit_interval(metrics["accuracy"], f"evaluation.splits.{split_name}.accuracy")
        balanced = _unit_interval(
            metrics["balanced_accuracy"],
            f"evaluation.splits.{split_name}.balanced_accuracy",
        )
        macro_f1 = _unit_interval(metrics["macro_f1"], f"evaluation.splits.{split_name}.macro_f1")
        expected_accuracy = sum(matrix[index][index] for index in range(len(matrix))) / window_count
        expected_balanced = sum(
            matrix[index][index] / supports[index] if supports[index] else 0.0
            for index in range(len(matrix))
        ) / len(matrix)
        expected_macro_f1 = sum(normalized_per_class[label]["f1"] for label in CANONICAL_LABELS) / len(CANONICAL_LABELS)
        if not math.isclose(accuracy, expected_accuracy, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"evaluation {split_name} accuracy disagrees with confusion_matrix")
        if not math.isclose(balanced, expected_balanced, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"evaluation {split_name} balanced_accuracy is inconsistent")
        if not math.isclose(macro_f1, expected_macro_f1, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"evaluation {split_name} macro_f1 is inconsistent")
        normalized_splits[split_name] = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "macro_f1": macro_f1,
            "per_class": normalized_per_class,
            "confusion_matrix": matrix,
            "window_count": window_count,
            "error_count": error_count,
            "error_windows": normalized_error_windows,
        }
    return {
        "schema": EVALUATION_SCHEMA,
        "version": EVALUATION_VERSION,
        "labels": list(CANONICAL_LABELS),
        "validation_role": validation_role,
        "splits": normalized_splits,
    }


class EmgLdaBundle:
    """Validated, immutable model metadata with NumPy-only inference."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != _TOP_LEVEL_FIELDS:
            raise ValueError("model bundle has missing or unknown fields")
        if payload["schema"] != BUNDLE_SCHEMA or payload["version"] != BUNDLE_VERSION:
            raise ValueError("unsupported model bundle schema or version")

        bundle_id = _string(payload["bundle_id"], "bundle_id")
        if _BUNDLE_ID_PATTERN.fullmatch(bundle_id) is None:
            raise ValueError("bundle_id contains forbidden characters or has invalid length")

        labels_value = payload["labels"]
        if not isinstance(labels_value, list) or tuple(labels_value) != CANONICAL_LABELS:
            raise ValueError(f"labels must be exactly {CANONICAL_LABELS!r}")
        if len(set(labels_value)) != len(labels_value):
            raise ValueError("labels must be unique")
        labels = tuple(labels_value)

        channel_count = _positive_int(payload["channel_count"], "channel_count")
        if channel_count != EMG_CHANNEL_COUNT:
            raise ValueError(
                f"formal EMG model bundles require exactly {EMG_CHANNEL_COUNT} channels"
            )
        feature_spec = FeatureSpec.from_dict(payload["feature_spec"])
        if feature_spec.channel_count != channel_count:
            raise ValueError("feature_spec channel_count disagrees with bundle channel_count")
        feature_count = channel_count * len(FEATURE_NAMES)

        rate_descriptor = RateDescriptor.from_mapping(payload["sample_rate"])
        if rate_descriptor.confirmed is not True:
            raise ValueError("sample rate must be confirmed")
        source_kind = validate_sample_rate_source_kind(rate_descriptor.source_kind)
        rate_descriptor = RateDescriptor(
            value_hz=rate_descriptor.value_hz,
            source_kind=source_kind,
            evidence_ref=rate_descriptor.evidence_ref,
            confirmed=True,
        )
        value_hz = rate_descriptor.value_hz
        assert value_hz is not None
        sample_rate = asdict(rate_descriptor)

        try:
            hand_side_contract = HandSide(payload["hand_side"])
        except (TypeError, ValueError) as exc:
            raise ValueError("hand_side must be left or right")
        if hand_side_contract is HandSide.UNKNOWN:
            raise ValueError("hand_side must be left or right")
        hand_side = hand_side_contract.value

        window = _require_exact_keys(
            payload["window"],
            {"window_ms", "step_ms", "window_samples", "step_samples"},
            "window",
        )
        if window["window_ms"] != 200 or window["step_ms"] != 50:
            raise ValueError("bundle window contract must be exactly 200/50 ms")
        window_samples = _positive_int(window["window_samples"], "window.window_samples")
        step_samples = _positive_int(window["step_samples"], "window.step_samples")
        expected_window = value_hz * 0.2
        expected_step = value_hz * 0.05
        if not expected_window.is_integer() or not expected_step.is_integer():
            raise ValueError("sample rate does not produce integral 200/50 ms sample counts")
        if window_samples != int(expected_window) or step_samples != int(expected_step):
            raise ValueError("window sample counts disagree with the confirmed sample rate")
        normalized_window = {
            "window_ms": 200,
            "step_ms": 50,
            "window_samples": window_samples,
            "step_samples": step_samples,
        }

        scaler = _require_exact_keys(payload["scaler"], {"mean", "scale"}, "scaler")
        scaler_mean = _finite_vector(scaler["mean"], feature_count, "scaler.mean")
        scaler_scale = _finite_vector(
            scaler["scale"], feature_count, "scaler.scale", positive=True
        )

        lda = _require_exact_keys(payload["lda"], {"classes", "coef", "intercept"}, "lda")
        if not isinstance(lda["classes"], list) or tuple(lda["classes"]) != labels:
            raise ValueError("lda.classes must exactly match the ordered labels")
        lda_coef = _finite_matrix(
            lda["coef"], len(labels), feature_count, "lda.coef"
        )
        lda_intercept = _finite_vector(lda["intercept"], len(labels), "lda.intercept")

        package_versions = _require_exact_keys(
            payload["package_versions"],
            {"python", "numpy", "scikit-learn"},
            "package_versions",
        )
        normalized_versions = {
            key: _string(value, f"package_versions.{key}")
            for key, value in package_versions.items()
        }

        manifest_sha256 = _string(payload["manifest_sha256"], "manifest_sha256")
        if _SHA256_PATTERN.fullmatch(manifest_sha256) is None:
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
        source_hashes_value = payload["source_hashes"]
        if not isinstance(source_hashes_value, dict) or not source_hashes_value:
            raise ValueError("source_hashes must be a non-empty object")
        source_hashes: dict[str, str] = {}
        for name, digest in source_hashes_value.items():
            _string(name, "source_hashes key")
            if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                raise ValueError("every source hash must be a lowercase SHA-256 digest")
            source_hashes[name] = digest

        seed = payload["seed"]
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        seed = int(seed)
        if seed < 0 or seed > 2**32 - 1:
            raise ValueError("seed is outside the uint32 range")
        provenance = payload["provenance"]
        if provenance not in {"canonical_session", "synthetic_test", "external_benchmark"}:
            raise ValueError("unsupported provenance")
        if payload["deployment_status"] != "engineering_only":
            raise ValueError("this baseline only permits engineering_only deployment status")

        signal_chain_contract = SignalChain.from_mapping(payload["signal_chain"])
        if signal_chain_contract.sample_format == "unknown":
            raise ValueError("signal_chain sample_format must be explicit")
        sample_format = _string(payload["sample_format"], "sample_format")
        if sample_format != signal_chain_contract.sample_format:
            raise ValueError("sample_format must match signal_chain.sample_format")
        notification_contract = NotificationPacketProtocol.from_mapping(
            payload["notification_protocol"]
        )
        signal_chain = asdict(signal_chain_contract)
        notification_protocol = asdict(notification_contract)

        normalized = {
            "schema": BUNDLE_SCHEMA,
            "version": BUNDLE_VERSION,
            "bundle_id": bundle_id,
            "labels": list(labels),
            "channel_count": channel_count,
            "sample_rate": sample_rate,
            "hand_side": hand_side,
            "window": normalized_window,
            "feature_spec": feature_spec.to_dict(),
            "sample_format": sample_format,
            "signal_chain": signal_chain,
            "notification_protocol": notification_protocol,
            "scaler": {
                "mean": scaler_mean.tolist(),
                "scale": scaler_scale.tolist(),
            },
            "lda": {
                "classes": list(labels),
                "coef": lda_coef.tolist(),
                "intercept": lda_intercept.tolist(),
            },
            "package_versions": normalized_versions,
            "manifest_sha256": manifest_sha256,
            "source_hashes": dict(sorted(source_hashes.items())),
            "seed": seed,
            "provenance": provenance,
            "deployment_status": "engineering_only",
        }

        self._payload = normalized
        self.bundle_id = bundle_id
        self.labels = labels
        self.channel_count = channel_count
        self.sample_rate = MappingProxyType(sample_rate)
        self.hand_side = hand_side
        self.window = MappingProxyType(normalized_window)
        self.feature_spec = feature_spec
        self.sample_format = sample_format
        self.signal_chain = MappingProxyType(signal_chain)
        self.notification_protocol = MappingProxyType(notification_protocol)
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        self.lda_classes = labels
        self.lda_coef = lda_coef
        self.lda_intercept = lda_intercept
        self.package_versions = MappingProxyType(normalized_versions)
        self.manifest_sha256 = manifest_sha256
        self.source_hashes = MappingProxyType(dict(sorted(source_hashes.items())))
        self.seed = seed
        self.provenance = provenance
        self.deployment_status = "engineering_only"
        self._evaluation: Any = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmgLdaBundle":
        return cls(payload)

    @property
    def window_samples(self) -> int:
        return self.window["window_samples"]

    @property
    def step_samples(self) -> int:
        return self.window["step_samples"]

    @property
    def acquisition_contract(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "sample_rate": dict(self.sample_rate),
                "sample_format": self.sample_format,
                "signal_chain": dict(self.signal_chain),
                "notification_protocol": dict(self.notification_protocol),
            }
        )

    @property
    def evaluation(self) -> Any:
        return self._evaluation

    def to_dict(self) -> dict[str, Any]:
        return json.loads(_canonical_json_bytes(self._payload).decode("utf-8"))

    def _feature_batch(self, features: object) -> np.ndarray:
        if not isinstance(features, np.ndarray):
            raise ValueError("features must be a numeric NumPy array")
        if features.dtype.kind not in "iuf":
            raise ValueError("features must have a real numeric dtype")
        if features.ndim == 1:
            batch = np.asarray(features, dtype=np.float64).reshape(1, -1)
        elif features.ndim == 2:
            batch = np.asarray(features, dtype=np.float64)
        else:
            raise ValueError("features must be one- or two-dimensional")
        if batch.shape[1] != self.scaler_mean.size:
            raise ValueError(f"features must have width {self.scaler_mean.size}")
        if batch.shape[0] < 1 or not np.isfinite(batch).all():
            raise ValueError("features must be a non-empty finite batch")
        return batch

    def decision_function(self, features: object) -> np.ndarray:
        batch = self._feature_batch(features)
        standardized = (batch - self.scaler_mean) / self.scaler_scale
        return standardized @ self.lda_coef.T + self.lda_intercept

    def predict_proba(self, features: object) -> np.ndarray:
        decision = self.decision_function(features)
        shifted = decision - np.max(decision, axis=1, keepdims=True)
        exponential = np.exp(shifted)
        probabilities = exponential / np.sum(exponential, axis=1, keepdims=True)
        return probabilities

    def predict(self, features: object) -> np.ndarray:
        decision = self.decision_function(features)
        indices = np.argmax(decision, axis=-1)
        labels = np.asarray(self.labels, dtype=str)
        return labels[indices]

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "EmgLdaBundle":
        directory = Path(path)
        try:
            directory_stat = os.lstat(directory)
        except OSError as exc:
            raise ValueError(f"cannot inspect bundle directory: {exc}") from exc
        if not stat.S_ISDIR(directory_stat.st_mode) or _is_reparse_point(directory_stat):
            raise ValueError("bundle path must be an existing directory")
        try:
            with os.scandir(directory) as iterator:
                entries = list(iterator)
        except OSError as exc:
            raise ValueError(f"cannot inspect bundle directory: {exc}") from exc
        if {entry.name for entry in entries} != _EXPECTED_FILES:
            raise ValueError("bundle directory contains missing, unknown, or non-file entries")
        expected_stats: dict[str, os.stat_result] = {}
        for entry in entries:
            try:
                entry_stat = os.lstat(directory / entry.name)
            except OSError as exc:
                raise ValueError(f"cannot inspect bundle file {entry.name!r}: {exc}") from exc
            if entry.is_symlink() or not stat.S_ISREG(entry_stat.st_mode) or _is_reparse_point(entry_stat):
                raise ValueError("bundle directory contains missing, unknown, or non-file entries")
            expected_stats[entry.name] = entry_stat

        limits = {
            MODEL_FILENAME: _MAX_MODEL_BYTES,
            EVALUATION_FILENAME: _MAX_EVALUATION_BYTES,
            DIGEST_FILENAME: _MAX_DIGEST_BYTES,
        }
        raw_files: dict[str, bytes] = {}
        opened_stats: dict[str, os.stat_result] = {}
        for filename in (DIGEST_FILENAME, MODEL_FILENAME, EVALUATION_FILENAME):
            raw_files[filename], opened_stats[filename] = _read_bounded_regular_file(
                directory / filename, limits[filename], expected_stats[filename]
            )

        digests = _parse_json_bytes(raw_files[DIGEST_FILENAME], DIGEST_FILENAME)
        digest_object = _require_exact_keys(
            digests, {"schema", "version", "files"}, "digests"
        )
        if (
            digest_object["schema"] != DIGEST_SCHEMA
            or digest_object["version"] != DIGEST_VERSION
        ):
            raise ValueError("unsupported digest schema or version")
        files = _require_exact_keys(
            digest_object["files"],
            {MODEL_FILENAME, EVALUATION_FILENAME},
            "digests.files",
        )
        for filename in (MODEL_FILENAME, EVALUATION_FILENAME):
            expected = files[filename]
            if not isinstance(expected, str) or _SHA256_PATTERN.fullmatch(expected) is None:
                raise ValueError("invalid SHA-256 digest entry")
            actual = _sha256(raw_files[filename])
            if actual != expected:
                raise ValueError(f"digest mismatch for {filename}")

        payload = _parse_json_bytes(raw_files[MODEL_FILENAME], MODEL_FILENAME)
        evaluation = _validate_evaluation(
            _parse_json_bytes(raw_files[EVALUATION_FILENAME], EVALUATION_FILENAME)
        )
        bundle = cls.from_dict(payload)
        bundle._evaluation = evaluation

        try:
            if not _same_file(directory_stat, os.lstat(directory)):
                raise ValueError("bundle directory was replaced while loading")
            for filename, opened_stat in opened_stats.items():
                current = os.lstat(directory / filename)
                if (
                    not _same_file(opened_stat, current)
                    or opened_stat.st_size != current.st_size
                    or opened_stat.st_mtime_ns != current.st_mtime_ns
                    or _is_reparse_point(current)
                ):
                    raise ValueError(f"bundle file {filename!r} was replaced while loading")
        except OSError as exc:
            raise ValueError(f"bundle changed while loading: {exc}") from exc
        return bundle


class BundlePublishError(RuntimeError):
    """Publication failed and one or more task-owned paths could not be cleaned."""


class BundleCleanupWarning(RuntimeWarning):
    """The final bundle was committed, but a publication lock remains."""


def publish_bundle(
    path: str | os.PathLike[str],
    bundle: EmgLdaBundle,
    evaluation: Mapping[str, Any],
) -> Path:
    """Publish a complete bundle directory, atomically and without overwrite."""

    if not isinstance(bundle, EmgLdaBundle):
        raise TypeError("bundle must be an EmgLdaBundle")
    evaluation_copy = _validate_evaluation(evaluation)
    model_bytes = _canonical_json_bytes(bundle.to_dict())
    evaluation_bytes = _canonical_json_bytes(evaluation_copy)
    if len(model_bytes) > _MAX_MODEL_BYTES or len(evaluation_bytes) > _MAX_EVALUATION_BYTES:
        raise ValueError("bundle payload exceeds the size limit")

    destination = Path(path)
    parent = destination.parent
    if not destination.name:
        raise ValueError("bundle path must name a directory")
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".{destination.name}.publish.lock"
    lock_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    try:
        lock_descriptor = os.open(lock_path, lock_flags, 0o600)
    except FileExistsError as exc:
        stale_lock_note = ""
        if destination.exists() or destination.is_symlink():
            try:
                lock_path.unlink()
                stale_lock_note = "; removed stale publication lock"
            except FileNotFoundError:
                stale_lock_note = ""
            except OSError as cleanup_exc:
                stale_lock_note = f"; stale publication lock remains: {cleanup_exc}"
        raise FileExistsError(
            f"bundle destination already exists: {destination}{stale_lock_note}"
        ) from exc
    except OSError as exc:
        raise BundlePublishError(f"cannot acquire publication lock {lock_path}: {exc}") from exc
    os.close(lock_descriptor)

    staging: Path | None = None
    published = False
    primary_error: BaseException | None = None
    try:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"bundle destination already exists: {destination}")
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent))
        files = {
            MODEL_FILENAME: model_bytes,
            EVALUATION_FILENAME: evaluation_bytes,
        }
        digest_bytes = _canonical_json_bytes(
            {
                "schema": DIGEST_SCHEMA,
                "version": DIGEST_VERSION,
                "files": {name: _sha256(content) for name, content in sorted(files.items())},
            }
        )
        files[DIGEST_FILENAME] = digest_bytes
        for filename, content in files.items():
            with (staging / filename).open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"bundle destination already exists: {destination}")
        os.rename(staging, destination)
        published = True
        return destination
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_failures: list[str] = []
        if staging is not None and not published and staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError as exc:
                cleanup_failures.append(f"staging {staging}: {exc}")
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            cleanup_failures.append(f"publication lock {lock_path}: {exc}")
        if cleanup_failures:
            details = "; ".join(cleanup_failures)
            if published:
                with warnings.catch_warnings():
                    warnings.simplefilter("always", BundleCleanupWarning)
                    warnings.warn(
                        f"bundle published successfully at {destination}; cleanup issue: {details}",
                        BundleCleanupWarning,
                        stacklevel=2,
                    )
                return destination
            primary = (
                f"; original error: {type(primary_error).__name__}: {primary_error}"
                if primary_error is not None
                else ""
            )
            raise BundlePublishError(
                "bundle publication cleanup failed; residual paths: "
                + details
                + primary
            )


__all__ = [
    "BUNDLE_SCHEMA",
    "BUNDLE_VERSION",
    "BundleCleanupWarning",
    "BundlePublishError",
    "CANONICAL_LABELS",
    "EmgLdaBundle",
    "publish_bundle",
]
