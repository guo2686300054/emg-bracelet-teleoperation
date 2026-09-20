import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from emg_model_bundle import (
    BundleCleanupWarning,
    BundlePublishError,
    EmgLdaBundle,
    publish_bundle,
)
from emg_protocol import NotificationPacketProtocol, SignalChain


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload() -> dict:
    feature_count = 40
    return {
        "schema": "emg.lda.bundle",
        "version": "1.0",
        "bundle_id": "synthetic-baseline-v1",
        "labels": ["rest", "fist", "open_hand"],
        "channel_count": 8,
        "sample_rate": {
            "value_hz": 200.0,
            "source_kind": "device-spec",
            "evidence_ref": "synthetic-fixture-rate",
            "confirmed": True,
        },
        "hand_side": "right",
        "window": {
            "window_ms": 200,
            "step_ms": 50,
            "window_samples": 40,
            "step_samples": 10,
        },
        "feature_spec": {
            "version": "time_domain_v1",
            "names": ["MAV", "RMS", "WL", "ZC", "SSC"],
            "zc_threshold": 0.0,
            "ssc_threshold": 0.0,
            "channel_count": 8,
        },
        "sample_format": "float32",
        "signal_chain": asdict(SignalChain(sample_format="float32")),
        "notification_protocol": asdict(NotificationPacketProtocol()),
        "scaler": {
            "mean": [float(value) for value in range(feature_count)],
            "scale": [float(value + 1) for value in range(feature_count)],
        },
        "lda": {
            "classes": ["rest", "fist", "open_hand"],
            "coef": [
                [0.1 * (index + 1) for index in range(feature_count)],
                [-0.05 * (index + 1) for index in range(feature_count)],
                [0.02 * ((index % 3) - 1) for index in range(feature_count)],
            ],
            "intercept": [0.25, -0.4, 0.05],
        },
        "package_versions": {
            "python": "3.10.11",
            "numpy": np.__version__,
            "scikit-learn": "1.7.2",
        },
        "manifest_sha256": _digest("manifest"),
        "source_hashes": {"fixture.csv": _digest("source")},
        "seed": 0,
        "provenance": "synthetic_test",
        "deployment_status": "engineering_only",
    }


def _deep_copy(payload: dict) -> dict:
    return json.loads(json.dumps(payload))


def _evaluation() -> dict:
    metrics = {
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "macro_f1": 1.0,
        "per_class": {
            label: {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}
            for label in ("rest", "fist", "open_hand")
        },
        "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "window_count": 3,
        "error_count": 0,
        "error_windows": [],
    }
    return {
        "schema": "emg.training.evaluation",
        "version": "1.1",
        "labels": ["rest", "fist", "open_hand"],
        "validation_role": "informational_no_hyperparameter_tuning",
        "splits": {
            split: _deep_copy(metrics) for split in ("train", "validation", "test")
        },
    }


def test_numpy_decisions_probabilities_and_classes_match_exported_affine_model():
    payload = _payload()
    bundle = EmgLdaBundle.from_dict(payload)
    features = np.array(
        [
            np.linspace(-3.0, 4.0, 40),
            np.linspace(10.0, -5.0, 40),
        ]
    )

    scaled = (features - np.array(payload["scaler"]["mean"])) / np.array(
        payload["scaler"]["scale"]
    )
    expected_decisions = scaled @ np.array(payload["lda"]["coef"]).T + np.array(
        payload["lda"]["intercept"]
    )
    exp = np.exp(expected_decisions - expected_decisions.max(axis=1, keepdims=True))
    expected_probabilities = exp / exp.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(
        bundle.decision_function(features), expected_decisions, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        bundle.predict_proba(features), expected_probabilities, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_array_equal(
        bundle.predict(features),
        np.array(payload["labels"])[np.argmax(expected_decisions, axis=1)],
    )
    assert bundle.decision_function(features[0]).shape == (1, 3)
    assert bundle.predict_proba(features[0]).shape == (1, 3)
    assert bundle.predict(features[0]).shape == (1,)


def test_exported_parameters_match_sklearn_lda_to_release_tolerance():
    sklearn_discriminant = pytest.importorskip("sklearn.discriminant_analysis")
    sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")
    rng = np.random.default_rng(71)
    labels = np.repeat(np.array(["rest", "fist", "open_hand"]), 24)
    centers = {
        "rest": np.linspace(-2.0, 0.0, 40),
        "fist": np.linspace(0.5, 2.5, 40),
        "open_hand": np.linspace(3.0, 5.0, 40),
    }
    features = np.vstack(
        [centers[label] + rng.normal(0.0, 0.15, 40) for label in labels]
    )
    scaler = sklearn_preprocessing.StandardScaler().fit(features)
    lda = sklearn_discriminant.LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage="auto"
    ).fit(scaler.transform(features), labels)

    payload = _payload()
    order = [int(np.flatnonzero(lda.classes_ == label)[0]) for label in payload["labels"]]
    payload["scaler"] = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    payload["lda"] = {
        "classes": payload["labels"],
        "coef": lda.coef_[order].tolist(),
        "intercept": lda.intercept_[order].tolist(),
    }
    bundle = EmgLdaBundle(payload)
    held_out = features[[2, 27, 53, 70]]
    sklearn_decision = lda.decision_function(scaler.transform(held_out))[:, order]
    sklearn_probability = lda.predict_proba(scaler.transform(held_out))[:, order]

    np.testing.assert_allclose(
        bundle.decision_function(held_out), sklearn_decision, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        bundle.predict_proba(held_out), sklearn_probability, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_array_equal(bundle.predict(held_out), lda.predict(scaler.transform(held_out)))


def test_publish_round_trip_is_json_only_deterministic_and_no_overwrite(tmp_path):
    bundle = EmgLdaBundle.from_dict(_payload())
    evaluation = _evaluation()
    first = publish_bundle(tmp_path / "first", bundle, evaluation)
    second = publish_bundle(tmp_path / "second", bundle, evaluation)

    assert {item.name for item in first.iterdir()} == {
        "model.json",
        "evaluation.json",
        "digests.json",
    }
    for filename in ("model.json", "evaluation.json", "digests.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    loaded = EmgLdaBundle.load(first)
    assert loaded.to_dict() == bundle.to_dict()
    assert loaded.evaluation == evaluation
    assert loaded.package_versions["python"] == "3.10.11"
    assert loaded.manifest_sha256 == _digest("manifest")
    with pytest.raises(FileExistsError):
        publish_bundle(first, bundle, evaluation)
    assert not list(tmp_path.glob(".first.staging-*"))


def test_digest_tampering_is_rejected(tmp_path):
    output = publish_bundle(tmp_path / "bundle", EmgLdaBundle(_payload()), _evaluation())
    with (output / "model.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        EmgLdaBundle.load(output)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(schema="wrong"),
        lambda p: p.update(version="2.0"),
        lambda p: p.update(extra=True),
        lambda p: p.pop("seed"),
        lambda p: p.update(labels=["rest", "fist", "fist"]),
        lambda p: p.update(channel_count=1),
        lambda p: p.update(channel_count=7),
        lambda p: p.update(channel_count=9),
        lambda p: p["lda"].update(classes=["fist", "open_hand", "rest"]),
        lambda p: p["scaler"].update(mean=[0.0] * 39),
        lambda p: p["scaler"].update(scale=[1.0] * 39 + [0.0]),
        lambda p: p["lda"].update(coef=[[0.0] * 40] * 2),
        lambda p: p["lda"].update(intercept=[0.0, np.inf, 0.0]),
        lambda p: p["sample_rate"].update(source_kind="observed"),
        lambda p: p["sample_rate"].update(confirmed=False),
        lambda p: p["window"].update(window_ms=201),
        lambda p: p["window"].update(step_samples=11),
        lambda p: p.update(manifest_sha256="not-a-hash"),
        lambda p: p.update(deployment_status="deployable"),
    ],
)
def test_invalid_bundle_contracts_fail_closed(mutate):
    payload = _deep_copy(_payload())
    mutate(payload)
    with pytest.raises((ValueError, TypeError)):
        EmgLdaBundle.from_dict(payload)


def test_prediction_rejects_malformed_or_non_finite_features():
    bundle = EmgLdaBundle(_payload())
    for value in (
        np.zeros(39),
        np.zeros((1, 40, 1)),
        np.array([True] * 40),
        np.array([0.0] * 39 + [np.nan]),
        [0.0] * 40,
    ):
        with pytest.raises(ValueError):
            bundle.predict_proba(value)


def test_publish_failure_cleans_staging_and_never_creates_final(tmp_path, monkeypatch):
    bundle = EmgLdaBundle(_payload())
    destination = tmp_path / "bundle"

    def fail_rename(source: Path, target: Path) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr("emg_model_bundle.os.rename", fail_rename)
    with pytest.raises(OSError, match="simulated interruption"):
        publish_bundle(destination, bundle, _evaluation())
    assert not destination.exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_unknown_files_and_duplicate_json_keys_are_rejected(tmp_path):
    output = publish_bundle(tmp_path / "bundle", EmgLdaBundle(_payload()), _evaluation())
    (output / "model.pkl").write_bytes(b"not executable but forbidden")
    with pytest.raises(ValueError, match="unknown"):
        EmgLdaBundle.load(output)

    (output / "model.pkl").unlink()
    (output / "digests.json").write_text(
        '{"schema":"a","schema":"b","version":"1.0","files":{}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        EmgLdaBundle.load(output)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(version="2.0"),
        lambda value: value.update(labels=["fist", "rest", "open_hand"]),
        lambda value: value["splits"].pop("test"),
        lambda value: value["splits"]["test"].update(window_count=4),
        lambda value: value["splits"]["test"]["per_class"]["rest"].update(support=2),
        lambda value: value["splits"]["test"]["per_class"]["rest"].update(precision=0.5),
        lambda value: value["splits"]["test"].update(confusion_matrix=[[0, 1], [1, 0]]),
        lambda value: value["splits"]["test"].update(accuracy=0.5),
        lambda value: value["splits"]["test"].update(error_count=1),
        lambda value: value["splits"]["test"].update(error_windows=[{}]),
        lambda value: value["splits"]["test"].update(extra=True),
    ],
)
def test_evaluation_schema_metrics_support_and_matrix_fail_closed(tmp_path, mutate):
    evaluation = _evaluation()
    mutate(evaluation)
    with pytest.raises(ValueError):
        publish_bundle(tmp_path / "bundle", EmgLdaBundle(_payload()), evaluation)
    assert not (tmp_path / "bundle").exists()


def test_error_window_contract_accepts_only_bounded_locator_evidence(tmp_path):
    evaluation = _evaluation()
    evaluation["splits"]["test"] = {
        "accuracy": 2 / 3,
        "balanced_accuracy": 2 / 3,
        "macro_f1": 5 / 9,
        "per_class": {
            "rest": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1},
            "fist": {"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "support": 1},
            "open_hand": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
        },
        "confusion_matrix": [[0, 1, 0], [0, 1, 0], [0, 0, 1]],
        "window_count": 3,
        "error_count": 1,
        "error_windows": [
            {
                "true_label": "rest",
                "predicted_label": "fist",
                "confidence": 0.75,
                "subject_id": "sub-1",
                "session_id": "session-1",
                "window_id": "window-1",
                "start_row": 10,
                "end_row_exclusive": 50,
            }
        ],
    }
    output = publish_bundle(tmp_path / "valid", EmgLdaBundle(_payload()), evaluation)
    stored = EmgLdaBundle.load(output).evaluation["splits"]["test"]
    assert stored["error_count"] == 1
    assert stored["error_windows"][0]["window_id"] == "window-1"

    for mutation in (
        lambda item: item.update(confidence=1.1),
        lambda item: item.update(predicted_label="rest"),
        lambda item: item.update(raw_samples=[1, 2, 3]),
    ):
        malformed = _deep_copy(evaluation)
        mutation(malformed["splits"]["test"]["error_windows"][0])
        with pytest.raises(ValueError):
            publish_bundle(
                tmp_path / f"invalid-{len(list(tmp_path.iterdir()))}",
                EmgLdaBundle(_payload()),
                malformed,
            )


def test_signal_chain_protocol_and_sample_format_use_strict_domain_contracts():
    malformed = _payload()
    malformed["signal_chain"]["sample_format"] = "int16"
    with pytest.raises(ValueError, match="sample_format"):
        EmgLdaBundle(malformed)

    malformed = _payload()
    malformed["notification_protocol"]["wire_packet_size"] = 17
    with pytest.raises(ValueError, match="packet protocol"):
        EmgLdaBundle(malformed)


def test_load_rejects_file_growth_between_directory_stat_and_open(tmp_path, monkeypatch):
    output = publish_bundle(tmp_path / "bundle", EmgLdaBundle(_payload()), _evaluation())
    real_open = os.open
    changed = False

    def grow_before_open(path, flags, *args, **kwargs):
        nonlocal changed
        candidate = Path(path)
        if candidate.name == "model.json" and not changed:
            changed = True
            with candidate.open("ab") as stream:
                stream.write(b" ")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr("emg_model_bundle.os.open", grow_before_open)
    with pytest.raises(ValueError, match="replaced|changed"):
        EmgLdaBundle.load(output)


def test_load_rejects_replacement_between_hash_and_parse(tmp_path, monkeypatch):
    output = publish_bundle(tmp_path / "bundle", EmgLdaBundle(_payload()), _evaluation())
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes((output / "model.json").read_bytes())
    import emg_model_bundle as module

    real_sha256 = module._sha256
    replaced = False

    def replace_after_hash(value: bytes) -> str:
        nonlocal replaced
        digest = real_sha256(value)
        if not replaced:
            replaced = True
            os.replace(replacement, output / "model.json")
        return digest

    monkeypatch.setattr(module, "_sha256", replace_after_hash)
    with pytest.raises(ValueError, match="replaced while loading"):
        EmgLdaBundle.load(output)


def test_load_rejects_symlink_artifact(tmp_path):
    output = publish_bundle(tmp_path / "bundle", EmgLdaBundle(_payload()), _evaluation())
    target = tmp_path / "model-copy.json"
    target.write_bytes((output / "model.json").read_bytes())
    (output / "model.json").unlink()
    try:
        (output / "model.json").symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(ValueError, match="non-file"):
        EmgLdaBundle.load(output)


def test_concurrent_publish_has_one_winner_and_never_overwrites(tmp_path):
    destination = tmp_path / "bundle"
    bundle = EmgLdaBundle(_payload())

    def publish():
        try:
            return publish_bundle(destination, bundle, _evaluation())
        except FileExistsError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: publish(), range(2)))
    assert sum(isinstance(result, Path) for result in results) == 1
    assert sum(isinstance(result, FileExistsError) for result in results) == 1
    assert EmgLdaBundle.load(destination).bundle_id == bundle.bundle_id


def test_cleanup_failure_reports_residual_staging_path(tmp_path, monkeypatch):
    destination = tmp_path / "bundle"

    def fail_rename(source: Path, target: Path) -> None:
        raise OSError("publish failed")

    def fail_cleanup(path: Path) -> None:
        raise OSError("cleanup denied")

    monkeypatch.setattr("emg_model_bundle.os.rename", fail_rename)
    monkeypatch.setattr("emg_model_bundle.shutil.rmtree", fail_cleanup)
    with pytest.raises(BundlePublishError, match=r"residual paths: staging .*cleanup denied") as caught:
        publish_bundle(destination, EmgLdaBundle(_payload()), _evaluation())
    staging = next(tmp_path.glob(".bundle.staging-*"))
    assert str(staging) in str(caught.value)
    assert not destination.exists()


def test_committed_bundle_returns_success_when_lock_cleanup_fails_and_recovers_stale_lock(
    tmp_path, monkeypatch
):
    destination = tmp_path / "bundle"
    lock_path = tmp_path / ".bundle.publish.lock"
    real_unlink = Path.unlink
    failed_once = False

    def fail_first_lock_cleanup(path: Path, *args, **kwargs):
        nonlocal failed_once
        if path == lock_path and not failed_once:
            failed_once = True
            raise PermissionError("lock cleanup denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_first_lock_cleanup)
    with pytest.warns(BundleCleanupWarning, match="published successfully"):
        result = publish_bundle(destination, EmgLdaBundle(_payload()), _evaluation())

    assert result == destination
    assert destination.is_dir()
    assert lock_path.is_file()
    assert EmgLdaBundle.load(result).bundle_id == "synthetic-baseline-v1"

    with pytest.raises(FileExistsError, match="removed stale publication lock"):
        publish_bundle(destination, EmgLdaBundle(_payload()), _evaluation())
    assert not lock_path.exists()
