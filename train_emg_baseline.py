"""Train and safely publish the fixed StandardScaler + shrinkage-LDA baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import sklearn
import training_samples as training_samples_module
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

from emg_features import (
    FeatureSpec,
    extract_time_domain_features,
    feature_extraction_scratch_bytes,
)
from emg_model_bundle import (
    BUNDLE_SCHEMA,
    BUNDLE_VERSION,
    EVALUATION_VERSION,
    MAX_REPORTED_ERROR_WINDOWS,
    EmgLdaBundle,
    publish_bundle,
)
from training_contract import (
    CANONICAL_LABELS,
    EMG_CHANNEL_COUNT,
    validate_sample_rate_source_kind,
)
from training_dataset import SPLIT_NAMES, TrainingDatasetError
from training_samples import (
    TrainingCorpus,
    TrainingSamplesError,
    iter_split_windows,
    load_training_manifest,
    reload_training_corpus,
)


MAX_FEATURE_MATRIX_BYTES = 128 * 1024 * 1024


class BaselineTrainingError(ValueError):
    """Raised when fixed-baseline preconditions are not satisfied."""


@dataclass(frozen=True)
class TrainingResult:
    bundle: EmgLdaBundle
    evaluation: Mapping[str, Any]
    output_dir: Path


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineTrainingError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise BaselineTrainingError(f"{name} must be a finite number")
    return result


def _validate_fixed_contract(corpus: TrainingCorpus) -> tuple[float, int, int]:
    window = corpus.window_contract
    if (
        _finite_number(window.get("requested_window_ms"), "requested_window_ms") != 200.0
        or _finite_number(window.get("requested_step_ms"), "requested_step_ms") != 50.0
    ):
        raise BaselineTrainingError("baseline requires requested window/step of exactly 200/50 ms")
    rate = corpus.data_contract.get("sample_rate")
    if not isinstance(rate, Mapping) or rate.get("confirmed") is not True:
        raise BaselineTrainingError("baseline requires a confirmed sample-time rate")
    rate_hz = _finite_number(rate.get("value_hz"), "sample_rate.value_hz")
    if rate_hz <= 0:
        raise BaselineTrainingError("sample_rate.value_hz must be positive")
    try:
        validate_sample_rate_source_kind(rate.get("source_kind"))
    except ValueError as exc:
        raise BaselineTrainingError(str(exc)) from exc
    exact_window = rate_hz * 0.2
    exact_step = rate_hz * 0.05
    if not exact_window.is_integer() or not exact_step.is_integer():
        raise BaselineTrainingError("confirmed sample rate does not produce integral 200/50 ms windows")
    window_samples, step_samples = int(exact_window), int(exact_step)
    if window.get("window_size_samples") != window_samples or window.get("step_size_samples") != step_samples:
        raise BaselineTrainingError("manifest sample counts disagree with confirmed rate")
    if not math.isclose(
        _finite_number(window.get("effective_window_ms"), "effective_window_ms"),
        200.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        _finite_number(window.get("effective_step_ms"), "effective_step_ms"),
        50.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise BaselineTrainingError("manifest effective window/step is not 200/50 ms")
    return rate_hz, window_samples, step_samples


def _assert_disjoint(corpus: TrainingCorpus) -> None:
    subject_sets = {
        name: set(corpus.manifest["splits"][name]["subject_ids"]) for name in SPLIT_NAMES
    }
    window_sets = {
        name: {window["window_id"] for window in corpus.manifest["splits"][name]["windows"]}
        for name in SPLIT_NAMES
    }
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            if subject_sets[left] & subject_sets[right]:
                raise BaselineTrainingError("subject IDs overlap across splits")
            if window_sets[left] & window_sets[right]:
                raise BaselineTrainingError("window IDs overlap across splits")


def _split_memory_bytes(
    corpus: TrainingCorpus, split: str, spec: FeatureSpec
) -> dict[str, int]:
    assert spec.channel_count is not None
    split_contract = corpus.manifest["splits"][split]
    window_count = len(split_contract["windows"])
    feature_count = spec.channel_count * 5
    feature_bytes = window_count * feature_count * np.dtype(np.float64).itemsize
    label_bytes = window_count * max(map(len, CANONICAL_LABELS)) * 4
    prediction_bytes = label_bytes
    class_output_bytes = window_count * len(CANONICAL_LABELS) * np.dtype(np.float64).itemsize
    window_bytes = (
        corpus.window_contract["window_size_samples"]
        * spec.channel_count
        * np.dtype(np.float64).itemsize
    )
    locators = {
        f"{window['subject_id']}/{window['session_id']}"
        for window in split_contract["windows"]
    }
    loader_bytes = max(
        (
            training_samples_module._session_iteration_peak_bytes(
                corpus.sessions[locator],
                reserved_bytes=0,
                window_bytes=window_bytes,
            )
            for locator in locators
        ),
        default=0,
    )
    feature_phase_bytes = max(
        (
            corpus.sessions[locator].row_count
            * corpus.sessions[locator].channel_count
            * np.dtype(np.float64).itemsize
            + window_bytes
            + feature_extraction_scratch_bytes(
                corpus.window_contract["window_size_samples"], spec.channel_count
            )
            for locator in locators
        ),
        default=0,
    )
    model_scratch_bytes = feature_count * feature_count * np.dtype(np.float64).itemsize * 6
    return {
        "feature": feature_bytes,
        "extraction_peak": feature_bytes + label_bytes + max(
            loader_bytes, feature_phase_bytes
        ),
        "fit_peak": feature_bytes * 2 + label_bytes + model_scratch_bytes,
        # sklearn scaled input, expected output, and bundle-side scaling/output
        # can coexist during parity proof.
        "proof_peak": (
            feature_bytes * 3
            + label_bytes
            + prediction_bytes
            + class_output_bytes * 2
        ),
        "reserved_for_loader": feature_bytes + label_bytes,
    }


def _assert_training_memory_budget(
    corpus: TrainingCorpus, spec: FeatureSpec
) -> dict[str, dict[str, int]]:
    plans = {split: _split_memory_bytes(corpus, split, spec) for split in SPLIT_NAMES}
    budget = training_samples_module.MAX_TRAINING_WORKING_SET_BYTES
    for split, plan in plans.items():
        if plan["feature"] > MAX_FEATURE_MATRIX_BYTES:
            raise BaselineTrainingError(f"{split} feature matrix exceeds the memory budget")
        phases = ("extraction_peak", "proof_peak")
        if split == "train":
            phases += ("fit_peak",)
        for phase in phases:
            if plan[phase] > budget:
                raise BaselineTrainingError(
                    f"{split} {phase} working set exceeds the {budget}-byte memory budget"
                )
    return plans


def _extract_split(
    corpus: TrainingCorpus,
    split: str,
    spec: FeatureSpec,
    *,
    reserved_for_loader: int,
) -> tuple[np.ndarray, np.ndarray]:
    assert spec.channel_count is not None
    split_contract = corpus.manifest["splits"][split]
    window_count = len(split_contract["windows"])
    feature_count = spec.channel_count * 5
    if window_count == 0:
        raise BaselineTrainingError(f"{split} split is empty")
    matrix = np.empty((window_count, feature_count), dtype=np.float64)
    labels = np.empty(window_count, dtype=f"<U{max(map(len, CANONICAL_LABELS))}")
    observed = 0
    for observed, example in enumerate(
        iter_split_windows(corpus, split, reserved_bytes=reserved_for_loader), 1
    ):
        if observed > window_count:
            raise BaselineTrainingError(f"{split} yielded more windows than declared")
        matrix[observed - 1] = extract_time_domain_features(example.samples, spec)
        labels[observed - 1] = example.label
    if observed != window_count:
        raise BaselineTrainingError(f"{split} yielded fewer windows than declared")
    if set(labels.tolist()) != set(CANONICAL_LABELS):
        raise BaselineTrainingError(f"{split} split lacks canonical class coverage")
    if not np.isfinite(matrix).all():
        raise BaselineTrainingError(f"{split} features contain non-finite values")
    return matrix, labels


def _evaluate_and_prove_split(
    *,
    split: str,
    features: np.ndarray,
    labels: np.ndarray,
    scaled: np.ndarray,
    estimator: LinearDiscriminantAnalysis,
    bundle: EmgLdaBundle,
    sklearn_order: list[int],
    window_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_predictions = estimator.predict(scaled)
    expected_decisions = estimator.decision_function(scaled)[:, sklearn_order]
    actual_decisions = bundle.decision_function(features)
    np.testing.assert_allclose(actual_decisions, expected_decisions, rtol=1e-10, atol=1e-12)
    del actual_decisions, expected_decisions
    expected_probabilities = estimator.predict_proba(scaled)[:, sklearn_order]
    actual_probabilities = bundle.predict_proba(features)
    np.testing.assert_allclose(
        actual_probabilities, expected_probabilities, rtol=1e-10, atol=1e-12
    )
    del actual_probabilities
    if not np.array_equal(bundle.predict(features), expected_predictions):
        raise BaselineTrainingError(
            f"exported {split} predictions differ from fitted sklearn model"
        )
    return _metrics(
        labels,
        expected_predictions,
        expected_probabilities,
        window_records=window_records,
    )


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    *,
    window_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(window_records) != y_true.size or probabilities.shape != (
        y_true.size,
        len(CANONICAL_LABELS),
    ):
        raise BaselineTrainingError("evaluation evidence is not aligned with predictions")
    precision, recall, f1, support = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CANONICAL_LABELS,
            zero_division=cast(str, 0),
        ),
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(CANONICAL_LABELS)
    }
    error_indices = np.flatnonzero(y_true != y_pred)
    error_windows = []
    for row_index in error_indices[:MAX_REPORTED_ERROR_WINDOWS]:
        record = window_records[int(row_index)]
        error_windows.append(
            {
                "true_label": str(y_true[row_index]),
                "predicted_label": str(y_pred[row_index]),
                "confidence": float(np.max(probabilities[row_index])),
                "subject_id": record["subject_id"],
                "session_id": record["session_id"],
                "window_id": record["window_id"],
                "start_row": record["start_row"],
                "end_row_exclusive": record["end_row_exclusive"],
            }
        )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=CANONICAL_LABELS,
                average="macro",
                zero_division=cast(str, 0),
            )
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=CANONICAL_LABELS
        ).astype(int).tolist(),
        "window_count": int(y_true.size),
        "error_count": int(error_indices.size),
        "error_windows": error_windows,
    }


def _source_hashes(corpus: TrainingCorpus) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in corpus.manifest["inputs"]:
        locator = item["session_locator"]
        for filename in ("metadata.json", "samples.csv"):
            result[f"{locator}/{filename}"] = item["source_files"][filename]["sha256"]
    return result


def _bundle_id(payload_without_id: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload_without_id,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"emg-lda-{hashlib.sha256(encoded).hexdigest()[:24]}"


def _ordered_lda_parameters(
    estimator: LinearDiscriminantAnalysis,
) -> tuple[np.ndarray, np.ndarray]:
    classes = [str(value) for value in estimator.classes_]
    if set(classes) != set(CANONICAL_LABELS) or len(classes) != len(CANONICAL_LABELS):
        raise BaselineTrainingError("fitted LDA classes differ from the canonical label set")
    order = [classes.index(label) for label in CANONICAL_LABELS]
    coef = np.asarray(estimator.coef_, dtype=np.float64)[order]
    intercept = np.asarray(estimator.intercept_, dtype=np.float64)[order]
    if not np.isfinite(coef).all() or not np.isfinite(intercept).all():
        raise BaselineTrainingError("fitted LDA parameters are non-finite")
    return coef, intercept


def train_baseline(
    corpus: TrainingCorpus,
    output_dir: str | Path,
    *,
    seed: int = 0,
) -> TrainingResult:
    """Fit once on train subjects, evaluate fixed parameters, and atomically publish."""
    try:
        corpus = reload_training_corpus(corpus)
    except TrainingSamplesError as exc:
        raise BaselineTrainingError(f"corpus failed fresh canonical admission: {exc}") from exc
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise BaselineTrainingError("seed must be a uint32 integer")
    rate_hz, window_samples, step_samples = _validate_fixed_contract(corpus)
    _assert_disjoint(corpus)
    channel_count = corpus.data_contract["channel_count"]
    if channel_count != EMG_CHANNEL_COUNT or isinstance(channel_count, bool):
        raise BaselineTrainingError(
            f"formal EMG training requires exactly {EMG_CHANNEL_COUNT} channels"
        )
    spec = FeatureSpec(channel_count=channel_count, zc_threshold=0.0, ssc_threshold=0.0)
    memory_plans = _assert_training_memory_budget(corpus, spec)
    scaler = StandardScaler()
    train_features, train_labels = _extract_split(
        corpus,
        "train",
        spec,
        reserved_for_loader=memory_plans["train"]["reserved_for_loader"],
    )
    scaler.fit(train_features)
    scaled_train = cast(np.ndarray, scaler.transform(train_features))
    estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    estimator.fit(scaled_train, train_labels)

    coef, intercept = _ordered_lda_parameters(estimator)
    rate_descriptor = dict(corpus.data_contract["sample_rate"])
    rate_payload = {
        "value_hz": rate_hz,
        "source_kind": rate_descriptor["source_kind"],
        "evidence_ref": rate_descriptor["evidence_ref"],
        "confirmed": True,
    }
    payload: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "version": BUNDLE_VERSION,
        "labels": list(CANONICAL_LABELS),
        "channel_count": channel_count,
        "sample_rate": rate_payload,
        "hand_side": corpus.data_contract["hand_side"],
        "window": {
            "window_ms": 200,
            "step_ms": 50,
            "window_samples": window_samples,
            "step_samples": step_samples,
        },
        "feature_spec": spec.to_dict(),
        "sample_format": corpus.data_contract["sample_format"],
        "signal_chain": dict(corpus.data_contract["signal_chain"]),
        "notification_protocol": dict(corpus.data_contract["notification_packet_protocol"]),
        "scaler": {
            "mean": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
            "scale": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
        },
        "lda": {
            "classes": list(CANONICAL_LABELS),
            "coef": coef.tolist(),
            "intercept": intercept.tolist(),
        },
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "manifest_sha256": corpus.manifest_sha256,
        "source_hashes": _source_hashes(corpus),
        "seed": seed,
        "provenance": corpus.provenance,
        "deployment_status": "engineering_only",
    }
    payload["bundle_id"] = _bundle_id(payload)
    bundle = EmgLdaBundle.from_dict(payload)

    evaluation: dict[str, Any] = {
        "schema": "emg.training.evaluation",
        "version": EVALUATION_VERSION,
        "labels": list(CANONICAL_LABELS),
        "validation_role": "informational_no_hyperparameter_tuning",
        "splits": {},
    }

    # Evaluate and prove one split at a time so transformed matrices do not
    # remain resident across train/validation/test.
    sklearn_order = [
        list(estimator.classes_).index(label) for label in CANONICAL_LABELS
    ]
    evaluation["splits"]["train"] = _evaluate_and_prove_split(
        split="train",
        features=train_features,
        labels=train_labels,
        scaled=scaled_train,
        estimator=estimator,
        bundle=bundle,
        sklearn_order=sklearn_order,
        window_records=corpus.manifest["splits"]["train"]["windows"],
    )
    del train_features, train_labels, scaled_train

    for split in ("validation", "test"):
        features, labels = _extract_split(
            corpus,
            split,
            spec,
            reserved_for_loader=memory_plans[split]["reserved_for_loader"],
        )
        scaled = cast(np.ndarray, scaler.transform(features))
        evaluation["splits"][split] = _evaluate_and_prove_split(
            split=split,
            features=features,
            labels=labels,
            scaled=scaled,
            estimator=estimator,
            bundle=bundle,
            sklearn_order=sklearn_order,
            window_records=corpus.manifest["splits"][split]["windows"],
        )
        del features, labels, scaled

    # Repeat canonical admission after all reads to catch source/manifest mutation.
    try:
        reload_training_corpus(corpus)
    except TrainingSamplesError as exc:
        raise BaselineTrainingError(f"training sources changed during training: {exc}") from exc
    destination = publish_bundle(output_dir, bundle, evaluation)
    return TrainingResult(bundle=bundle, evaluation=evaluation, output_dir=destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the fixed leakage-safe EMG baseline.")
    parser.add_argument("--manifest", required=True, help="trusted dataset manifest v1.1")
    parser.add_argument("--session-root", required=True, help="explicit canonical session root")
    parser.add_argument("--output", required=True, help="new model bundle directory")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        corpus = load_training_manifest(args.manifest, args.session_root)
        result = train_baseline(corpus, args.output, seed=args.seed)
    except (TrainingDatasetError, TrainingSamplesError, BaselineTrainingError, ValueError, FileExistsError) as exc:
        print(f"training rejected: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote EMG baseline bundle: {result.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
