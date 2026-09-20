from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest

from emg_features import extract_time_domain_features
from emg_model_bundle import EmgLdaBundle
from emg_protocol import (
    AcquisitionMetadata,
    HandSide,
    NotificationPacketProtocol,
    RateDescriptor,
    SignalChain,
)
from realtime_emg_inference import (
    DecisionPolicy,
    Prediction,
    RealtimeDecisionFilter,
    RealtimeEmgClassifier,
)


def make_acquisition(*, observed_hz: float | None = None) -> AcquisitionMetadata:
    observed = RateDescriptor()
    if observed_hz is not None:
        observed = RateDescriptor(
            value_hz=observed_hz,
            source_kind="host-observed",
            evidence_ref="test clock",
            confirmed=False,
        )
    return AcquisitionMetadata(
        sample_rate=RateDescriptor(200.0, "protocol", "synthetic fixture", True),
        host_observed_rate=observed,
        signal_chain=SignalChain(sample_format="float32"),
        notification_packet_protocol=NotificationPacketProtocol(),
    )


def make_bundle() -> EmgLdaBundle:
    acquisition = make_acquisition()
    feature_count = 8 * 5
    return EmgLdaBundle.from_dict(
        {
            "schema": "emg.lda.bundle",
            "version": "1.0",
            "bundle_id": "stream-test",
            "labels": ["rest", "fist", "open_hand"],
            "channel_count": 8,
            "sample_rate": asdict(acquisition.sample_rate),
            "hand_side": "left",
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
            "signal_chain": asdict(acquisition.signal_chain),
            "notification_protocol": asdict(
                acquisition.notification_packet_protocol
            ),
            "scaler": {"mean": [0.0] * feature_count, "scale": [1.0] * feature_count},
            "lda": {
                "classes": ["rest", "fist", "open_hand"],
                "coef": [
                    [-1.0] + [0.0] * (feature_count - 1),
                    [1.0] + [0.0] * (feature_count - 1),
                    [0.0] * feature_count,
                ],
                "intercept": [0.0, 0.0, 0.25],
            },
            "package_versions": {
                "python": "3.10",
                "numpy": np.__version__,
                "scikit-learn": "1.7.2",
            },
            "manifest_sha256": "0" * 64,
            "source_hashes": {"fixture": "1" * 64},
            "seed": 0,
            "provenance": "synthetic_test",
            "deployment_status": "engineering_only",
        }
    )


def make_classifier(acquisition: AcquisitionMetadata | None = None) -> RealtimeEmgClassifier:
    return RealtimeEmgClassifier(
        make_bundle(),
        acquisition or make_acquisition(),
        HandSide.LEFT,
        evaluation_mode=True,
    )


def test_fixed_ring_cadence_reset_and_prediction_fields():
    classifier = make_classifier()
    outputs = [classifier.push([index] * 8, index + 1) for index in range(50)]
    assert all(item is None for item in outputs[:39])
    assert isinstance(outputs[39], Prediction)
    assert all(item is None for item in outputs[40:49])
    assert isinstance(outputs[49], Prediction)
    prediction = outputs[49]
    assert prediction is not None
    assert prediction.window_end_timestamp_ns == 50
    assert prediction.bundle_id == "stream-test"
    assert len(prediction.probabilities) == 3
    assert prediction.confidence == max(prediction.probabilities)
    assert prediction.feature_ns >= 0
    assert prediction.predict_ns >= 0
    assert prediction.total_ns == prediction.feature_ns + prediction.predict_ns

    ring_identity = id(classifier._ring)
    classifier.reset()
    assert id(classifier._ring) == ring_identity
    assert all(classifier.push([index] * 8, 100 + index) is None for index in range(39))
    assert classifier.push([39] * 8, 139) is not None


def test_chunked_matches_single_push_and_offline_windows():
    samples = np.repeat(
        np.linspace(-2.0, 2.0, 65, dtype=np.float64).reshape(-1, 1), 8, axis=1
    )
    timestamps = np.arange(1, 66, dtype=np.int64)
    single = make_classifier()
    one_by_one = [
        output
        for sample, timestamp in zip(samples, timestamps)
        if (output := single.push(sample, int(timestamp))) is not None
    ]
    chunked = make_classifier().push_many(samples, (int(value) for value in timestamps))
    assert len(one_by_one) == len(chunked) == 3
    for index, (streamed, batched) in enumerate(zip(one_by_one, chunked)):
        np.testing.assert_allclose(streamed.probabilities, batched.probabilities)
        assert streamed.label == batched.label
        start = index * 10
        features = extract_time_domain_features(
            samples[start : start + 40], single.bundle.feature_spec
        )
        expected = np.asarray(single.bundle.predict_proba(features)).reshape(-1)
        np.testing.assert_allclose(streamed.probabilities, expected, rtol=1e-12, atol=1e-12)
        assert streamed.label == np.asarray(single.bundle.predict(features)).reshape(-1)[0]


def test_contract_binding_and_engineering_mode_fail_closed():
    bundle = make_bundle()
    with pytest.raises(ValueError, match="evaluation_mode"):
        RealtimeEmgClassifier(bundle, make_acquisition(), HandSide.LEFT)
    with pytest.raises(ValueError, match="hand_side"):
        RealtimeEmgClassifier(
            bundle, make_acquisition(), HandSide.RIGHT, evaluation_mode=True
        )

    wrong_rate = replace(
        make_acquisition(),
        sample_rate=RateDescriptor(100.0, "protocol", "synthetic fixture", True),
    )
    with pytest.raises(ValueError, match="sample_rate"):
        RealtimeEmgClassifier(bundle, wrong_rate, HandSide.LEFT, evaluation_mode=True)
    wrong_signal = replace(
        make_acquisition(), signal_chain=SignalChain(sample_format="int16")
    )
    with pytest.raises(ValueError, match="signal_chain"):
        RealtimeEmgClassifier(bundle, wrong_signal, HandSide.LEFT, evaluation_mode=True)

    # Pure observation changes are intentionally outside the model contract.
    make_classifier(make_acquisition(observed_hz=199.5))


def test_push_rejects_invalid_values_without_advancing_state():
    classifier = make_classifier()
    with pytest.raises(ValueError, match="channel count"):
        classifier.push([1.0] * 7, 1)
    with pytest.raises(ValueError, match="finite"):
        classifier.push([np.nan] * 8, 1)
    with pytest.raises(ValueError, match="strictly increasing"):
        classifier.push([1.0] * 8, 1)
        classifier.push([2.0] * 8, 1)
    assert classifier._sample_count == 1
    with pytest.raises(ValueError, match="equal lengths"):
        make_classifier().push_many([[1.0] * 8, [2.0] * 8], [1])


def make_prediction(label: str, confidence: float, timestamp: int = 1) -> Prediction:
    probabilities = {
        "rest": (confidence, 1.0 - confidence, 0.0),
        "fist": (1.0 - confidence, confidence, 0.0),
        "open_hand": (1.0 - confidence, 0.0, confidence),
    }[label]
    return Prediction(label, probabilities, confidence, timestamp, "stream-test", 1, 1, 2)


def test_decision_policy_validation():
    with pytest.raises(ValueError, match="between 0 and 1"):
        DecisionPolicy(confidence_threshold=1.1)
    with pytest.raises(ValueError, match="unknown or hold"):
        DecisionPolicy(low_confidence_policy="unsafe")
    with pytest.raises(ValueError, match="consecutive or majority"):
        DecisionPolicy(smoothing_mode="average")
    with pytest.raises(ValueError, match="positive integer"):
        DecisionPolicy(smoothing_windows=0)


def test_consecutive_debounce_and_low_confidence_unknown_fail_closed():
    decision = RealtimeDecisionFilter(
        make_classifier(), DecisionPolicy(confidence_threshold=0.8, smoothing_windows=2)
    )
    first = decision.apply(make_prediction("fist", 0.9))
    assert first.output_label == "unknown"
    assert first.reason == "debouncing"
    second = decision.apply(make_prediction("fist", 0.9, 2))
    assert second.output_label == "fist"
    assert second.reason == "accepted"
    uncertain = decision.apply(make_prediction("rest", 0.79, 3))
    assert uncertain.output_label == "unknown"
    assert uncertain.reason == "low_confidence_unknown"
    assert uncertain.stable_label == "unknown"
    # Unknown revokes all previously accumulated authorization state.
    after_unknown = decision.apply(make_prediction("fist", 0.9, 4))
    assert after_unknown.output_label == "unknown"
    assert after_unknown.reason == "debouncing"


def test_hold_policy_majority_and_disconnect_reset():
    decision = RealtimeDecisionFilter(
        make_classifier(),
        DecisionPolicy(
            confidence_threshold=0.8,
            low_confidence_policy="hold",
            smoothing_mode="majority",
            smoothing_windows=3,
        ),
    )
    assert decision.apply(make_prediction("rest", 0.9)).output_label == "unknown"
    assert decision.apply(make_prediction("fist", 0.9, 2)).output_label == "unknown"
    assert decision.apply(make_prediction("rest", 0.9, 3)).output_label == "rest"
    held = decision.apply(make_prediction("fist", 0.4, 4))
    assert held.output_label == "rest"
    assert held.reason == "low_confidence_hold"
    decision.disconnect()
    assert decision.apply(make_prediction("fist", 0.4, 5)).output_label == "unknown"


def test_decision_push_many_uses_classifier_cadence_and_reset():
    decision = RealtimeDecisionFilter(
        make_classifier(), DecisionPolicy(confidence_threshold=0.0, smoothing_windows=1)
    )
    samples = np.ones((50, 8))
    outputs = decision.push_many(samples, range(1, 51))
    assert len(outputs) == 2
    assert all(item.output_label in {"rest", "fist", "open_hand"} for item in outputs)
    decision.reset()
    assert decision.push_many(np.ones((39, 8)), range(100, 139)) == []
