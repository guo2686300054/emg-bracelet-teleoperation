import math
import json
import tempfile
from pathlib import Path

import pytest
import numpy as np
import train_emg_baseline as train_module
from unittest import mock

from data_recorder import DataRecorder
from emg_model_bundle import EmgLdaBundle
from emg_features import extract_time_domain_features
from emg_protocol import (
    AcquisitionMetadata,
    DeviceKey,
    EmgFrame,
    HandSide,
    NotificationPacketProtocol,
    QualityFlags,
    RateDescriptor,
    SignalChain,
)
from recording_context import RecordingContext
from realtime_emg_inference import RealtimeEmgClassifier
from train_emg_baseline import BaselineTrainingError, main, train_baseline
from training_contract import CANONICAL_LABELS
from training_dataset import prepare_training_dataset
from training_samples import iter_split_windows, load_training_manifest
from test_training_dataset import TrainingDatasetTests as _DatasetFixture

_DatasetFixture.__test__ = False


_DEVICE = DeviceKey("dev-0123456789abcdef0123456789abcdef")
_FLAGS = (
    QualityFlags.VALID
    | QualityFlags.HOST_WALL_TIME_VALID
    | QualityFlags.HOST_MONOTONIC_VALID
    | QualityFlags.HOST_RECEIVE_INDEX_VALID
)


def _make_session(root: Path, subject_number: int, label: str) -> Path:
    subject_id = f"sub-{subject_number:032x}"
    session_id = f"session-{subject_number}-{label}"
    acquisition = AcquisitionMetadata(
        sample_rate=RateDescriptor(200.0, "protocol", "synthetic-device-spec-v1", True),
        host_observed_rate=RateDescriptor(200.0, "host_observed", "synthetic-clock", True),
        signal_chain=SignalChain(sample_format="float32"),
        notification_packet_protocol=NotificationPacketProtocol(),
    )
    recorder = DataRecorder(
        root,
        subject_id=subject_id,
        device_id=_DEVICE,
        acquisition=acquisition,
        channels=8,
        side=HandSide.LEFT,
        session_id=session_id,
        flush_every=1,
        recording_context=RecordingContext(
            subject_id,
            label,
            "hold",
            "synthetic_baseline_test_v1",
            HandSide.LEFT,
            "synthetic_test",
        ),
    )
    recorder.configure_session_boundary(start_host_receive_index=0, start_drop_total=0)
    try:
        for index in range(100):
            if label == "rest":
                base = 0.15 * math.sin(index / 4.0)
            elif label == "fist":
                base = (6.0 if index % 2 else -6.0) + 0.2 * math.sin(index)
            else:
                base = float(index % 20) * 0.35
            offset = subject_number * 0.001
            recorder.record(
                EmgFrame(
                    tuple(
                        base * (1.0 + channel * 0.02) + 0.05 * channel + offset
                        for channel in range(8)
                    ),
                    1_000_000 + index,
                    2_000_000 + index,
                    index + 1,
                    index,
                    action_label=label,
                    action_phase="hold",
                    quality_flags=_FLAGS,
                )
            )
        recorder.update_session_boundary(
            end_host_receive_index=100,
            received_count=100,
            eligible_count=100,
            written_count=100,
            queue_drop_total=0,
            queue_drop_session=0,
            tail_pending_count=0,
            tail_loss_count=0,
            incomplete_reason=None,
        )
    finally:
        recorder.close()
    return recorder.session_dir


def _prepare(root: Path, *, window_ms: float = 200, step_ms: float = 50) -> Path:
    sessions = [
        _make_session(root, subject, label)
        for subject in range(1, 4)
        for label in ("rest", "fist", "open_hand")
    ]
    manifest = root / "manifest.json"
    prepare_training_dataset(
        sessions,
        manifest,
        window_ms=window_ms,
        step_ms=step_ms,
        seed=11,
    )
    return manifest


def test_fixed_training_metrics_bundle_and_determinism():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _prepare(root)
        corpus = load_training_manifest(manifest, root)
        first = train_baseline(corpus, root / "bundle-a", seed=3)
        second = train_baseline(corpus, root / "bundle-b", seed=3)

        assert first.bundle.provenance == "synthetic_test"
        assert first.bundle.deployment_status == "engineering_only"
        assert first.bundle.window_samples == 40
        assert first.bundle.step_samples == 10
        for split in ("train", "validation", "test"):
            metrics = first.evaluation["splits"][split]
            assert metrics["macro_f1"] == pytest.approx(1.0)
            assert len(metrics["confusion_matrix"]) == 3
            assert set(metrics["per_class"]) == {"rest", "fist", "open_hand"}
            assert metrics["error_count"] == 0
            assert metrics["error_windows"] == []
        for filename in ("model.json", "evaluation.json", "digests.json"):
            assert (first.output_dir / filename).read_bytes() == (second.output_dir / filename).read_bytes()
        loaded = EmgLdaBundle.load(first.output_dir)
        assert loaded.bundle_id == first.bundle.bundle_id

        example = next(iter_split_windows(corpus, "test"))
        acquisition = AcquisitionMetadata(
            sample_rate=RateDescriptor(200.0, "protocol", "synthetic-device-spec-v1", True),
            host_observed_rate=RateDescriptor(200.0, "host_observed", "synthetic-clock", True),
            signal_chain=SignalChain(sample_format="float32"),
            notification_packet_protocol=NotificationPacketProtocol(),
        )
        classifier = RealtimeEmgClassifier(
            loaded, acquisition, HandSide.LEFT, evaluation_mode=True
        )
        predictions = classifier.push_many(example.samples, range(1, len(example.samples) + 1))
        assert len(predictions) == 1
        offline_features = extract_time_domain_features(example.samples, loaded.feature_spec)
        expected_probabilities = np.asarray(loaded.predict_proba(offline_features)).reshape(-1)
        np.testing.assert_allclose(predictions[0].probabilities, expected_probabilities)
        assert predictions[0].label == np.asarray(loaded.predict(offline_features)).reshape(-1)[0]


def test_non_fixed_window_and_existing_output_reject_without_partial_publish():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _prepare(root, window_ms=100, step_ms=50)
        corpus = load_training_manifest(manifest, root)
        output = root / "rejected-bundle"
        with pytest.raises(BaselineTrainingError, match="200/50"):
            train_baseline(corpus, output)
        assert not output.exists()


def test_cli_return_codes_and_no_bypass_options():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _prepare(root)
        output = root / "cli-bundle"
        args = [
            "--manifest", str(manifest),
            "--session-root", str(root),
            "--output", str(output),
            "--seed", "5",
        ]
        assert main(args) == 0
        assert main(args) == 2
        assert output.is_dir()
        assert EmgLdaBundle.load(output).provenance == "synthetic_test"
        with pytest.raises(SystemExit):
            main(args + ["--skip-quality-gate"])


def test_caller_cannot_override_provenance():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = load_training_manifest(_prepare(root), root)
        with pytest.raises(TypeError, match="provenance"):
            train_baseline(corpus, root / "bundle", provenance="canonical_session")


def test_fit_is_train_only_and_internal_failure_does_not_publish():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = load_training_manifest(_prepare(root), root)
        output = root / "bundle"
        with mock.patch(
            "train_emg_baseline.LinearDiscriminantAnalysis.fit",
            side_effect=RuntimeError("fit failed"),
        ) as fit:
            with pytest.raises(RuntimeError, match="fit failed"):
                train_baseline(corpus, output)
        assert fit.call_count == 1
        assert not output.exists()


def test_fit_spy_proves_only_train_rows_are_used(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = load_training_manifest(_prepare(root), root)
        expected_train_rows = len(corpus.manifest["splits"]["train"]["windows"])
        total_rows = sum(
            len(corpus.manifest["splits"][split]["windows"])
            for split in ("train", "validation", "test")
        )
        observed: list[tuple[int, frozenset[str]]] = []
        original_fit = train_module.LinearDiscriminantAnalysis.fit

        def fit_spy(estimator, features, labels):
            observed.append((features.shape[0], frozenset(labels.tolist())))
            return original_fit(estimator, features, labels)

        monkeypatch.setattr(train_module.LinearDiscriminantAnalysis, "fit", fit_spy)
        train_baseline(corpus, root / "bundle")
        assert observed == [(expected_train_rows, frozenset(CANONICAL_LABELS))]
        assert expected_train_rows < total_rows


def test_cli_unexpected_internal_failure_returns_one_without_publish(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _prepare(root)
        output = root / "bundle"
        monkeypatch.setattr(
            train_module,
            "train_baseline",
            mock.Mock(side_effect=RuntimeError("unexpected internal failure")),
        )
        assert main(
            [
                "--manifest", str(manifest),
                "--session-root", str(root),
                "--output", str(output),
            ]
        ) == 1
        assert not output.exists()


def test_stale_corpus_source_mutation_cannot_publish():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _prepare(root)
        corpus = load_training_manifest(manifest, root)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["splits"]["train"]["windows"][0]["window_id"] = "0" * 64
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        output = root / "bundle"
        with pytest.raises(BaselineTrainingError, match="fresh canonical admission"):
            train_baseline(corpus, output)
        assert not output.exists()


def test_feature_matrix_memory_budget_fails_without_publish(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = load_training_manifest(_prepare(root), root)
        output = root / "bundle"
        monkeypatch.setattr(train_module, "MAX_FEATURE_MATRIX_BYTES", 1)
        with pytest.raises(BaselineTrainingError, match="memory budget"):
            train_baseline(corpus, output)
        assert not output.exists()


def test_aggregate_working_set_rejects_when_one_matrix_would_fit(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = load_training_manifest(_prepare(root), root)
        channel_count = corpus.data_contract["channel_count"]
        train_windows = len(corpus.manifest["splits"]["train"]["windows"])
        feature_bytes = train_windows * channel_count * 5 * 8
        budget = feature_bytes + 1
        assert feature_bytes < budget
        monkeypatch.setattr(
            train_module.training_samples_module,
            "MAX_TRAINING_WORKING_SET_BYTES",
            budget,
        )
        output = root / "bundle"
        with pytest.raises(BaselineTrainingError, match="working set"):
            train_baseline(corpus, output)
        assert not output.exists()


def test_feature_scratch_peak_rejects_before_feature_extraction(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _DatasetFixture()
        sessions = [
            fixture.make_session(
                root,
                subject,
                f"wide-{subject}-{label}",
                [label] * 40,
                channels=8,
            )
            for subject in range(1, 4)
            for label in ("rest", "fist", "open_hand")
        ]
        manifest = root / "manifest.json"
        prepare_training_dataset(
            sessions, manifest, window_ms=200, step_ms=50, seed=11
        )
        corpus = load_training_manifest(manifest, root)
        spec = train_module.FeatureSpec(
            channel_count=8, zc_threshold=0.0, ssc_threshold=0.0
        )
        split = corpus.manifest["splits"]["train"]
        feature_bytes = len(split["windows"]) * 8 * 5 * 8
        label_bytes = len(split["windows"]) * max(map(len, CANONICAL_LABELS)) * 4
        window_bytes = 40 * 8 * 8
        locators = {
            f"{window['subject_id']}/{window['session_id']}"
            for window in split["windows"]
        }
        iterator_only_budget = feature_bytes + label_bytes + max(
            train_module.training_samples_module._session_iteration_peak_bytes(
                corpus.sessions[locator],
                reserved_bytes=0,
                window_bytes=window_bytes,
            )
            for locator in locators
        )
        assert iterator_only_budget < train_module._split_memory_bytes(
            corpus, "train", spec
        )["extraction_peak"]
        monkeypatch.setattr(
            train_module.training_samples_module,
            "MAX_TRAINING_WORKING_SET_BYTES",
            iterator_only_budget,
        )
        extractor = mock.Mock(side_effect=AssertionError("extractor must not run"))
        monkeypatch.setattr(train_module, "extract_time_domain_features", extractor)
        output = root / "bundle"
        with pytest.raises(BaselineTrainingError, match="train extraction_peak"):
            train_baseline(corpus, output)
        extractor.assert_not_called()
        assert not output.exists()


def test_error_window_evidence_is_bounded_and_contains_only_safe_locators():
    count = 60
    labels = np.array((["rest", "fist", "open_hand"] * 20))
    predictions = np.array((["fist", "open_hand", "rest"] * 20))
    probabilities = np.tile(
        np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]),
        (20, 1),
    )
    records = [
        {
            "subject_id": "sub-safe",
            "session_id": "session-safe",
            "window_id": f"window-{index}",
            "start_row": index,
            "end_row_exclusive": index + 40,
            "raw_samples": "must-not-leak",
        }
        for index in range(count)
    ]

    metrics = train_module._metrics(
        labels, predictions, probabilities, window_records=records
    )

    assert metrics["error_count"] == count
    assert len(metrics["error_windows"]) == 50
    assert metrics["error_windows"][0] == {
        "true_label": "rest",
        "predicted_label": "fist",
        "confidence": pytest.approx(0.8),
        "subject_id": "sub-safe",
        "session_id": "session-safe",
        "window_id": "window-0",
        "start_row": 0,
        "end_row_exclusive": 40,
    }
    assert all("raw_samples" not in item for item in metrics["error_windows"])
