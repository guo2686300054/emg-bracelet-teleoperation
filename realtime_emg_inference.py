"""Bounded, NumPy-only streaming inference for versioned EMG bundles."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import Field, asdict, dataclass, is_dataclass
from time import perf_counter_ns
from typing import ClassVar, Iterable, Mapping, Optional, Protocol, Sequence, cast

import numpy as np

from emg_features import extract_time_domain_features
from emg_model_bundle import EmgLdaBundle
from emg_protocol import AcquisitionMetadata, HandSide


class _DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[object]]]


@dataclass(frozen=True)
class Prediction:
    """One prediction emitted at the end of a complete streaming window."""

    label: str
    probabilities: tuple[float, ...]
    confidence: float
    window_end_timestamp_ns: int
    bundle_id: str
    feature_ns: int
    predict_ns: int
    total_ns: int


@dataclass(frozen=True)
class DecisionPolicy:
    """Safety policy applied after model inference.

    Defaults are intentionally conservative.  The threshold and debounce length
    are engineering defaults and must be calibrated with canonical recordings
    before the output is allowed to drive hardware.
    """

    confidence_threshold: float = 0.70
    low_confidence_policy: str = "unknown"
    smoothing_mode: str = "consecutive"
    smoothing_windows: int = 3
    unknown_label: str = "unknown"

    def __post_init__(self) -> None:
        if (
            isinstance(self.confidence_threshold, bool)
            or not isinstance(self.confidence_threshold, (int, float))
            or not 0.0 <= float(self.confidence_threshold) <= 1.0
        ):
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.low_confidence_policy not in {"unknown", "hold"}:
            raise ValueError("low_confidence_policy must be unknown or hold")
        if self.smoothing_mode not in {"consecutive", "majority"}:
            raise ValueError("smoothing_mode must be consecutive or majority")
        if (
            isinstance(self.smoothing_windows, bool)
            or not isinstance(self.smoothing_windows, int)
            or self.smoothing_windows < 1
        ):
            raise ValueError("smoothing_windows must be a positive integer")
        if not isinstance(self.unknown_label, str) or not self.unknown_label.strip():
            raise ValueError("unknown_label must be non-empty")


@dataclass(frozen=True)
class ControlledPrediction:
    """Raw model evidence plus the safe label exposed to a controller."""

    output_label: str
    raw_label: str
    confidence: float
    accepted: bool
    reason: str
    stable_label: str
    window_end_timestamp_ns: int
    prediction: Prediction


class RealtimeDecisionFilter:
    """Confidence gate and temporal debounce around a realtime classifier."""

    def __init__(
        self,
        classifier: "RealtimeEmgClassifier",
        policy: DecisionPolicy | None = None,
    ) -> None:
        if not isinstance(classifier, RealtimeEmgClassifier):
            raise TypeError("classifier must be RealtimeEmgClassifier")
        self.classifier = classifier
        self.policy = policy or DecisionPolicy()
        self.reset()

    def reset(self) -> None:
        """Clear model window and all decision history after a stream boundary."""

        self.classifier.reset()
        self._stable_label: str | None = None
        self._candidate_label: str | None = None
        self._candidate_count = 0
        self._votes: deque[str] = deque(maxlen=self.policy.smoothing_windows)

    def disconnect(self) -> None:
        """Fail closed on a connection break; the next stream starts cold."""

        self.reset()

    def _smooth(self, candidate: str) -> str:
        if self.policy.smoothing_mode == "consecutive":
            if candidate == self._candidate_label:
                self._candidate_count += 1
            else:
                self._candidate_label = candidate
                self._candidate_count = 1
            if self._candidate_count >= self.policy.smoothing_windows:
                self._stable_label = candidate
            return self._stable_label or self.policy.unknown_label

        self._votes.append(candidate)
        counts = Counter(self._votes)
        winner, count = counts.most_common(1)[0]
        # Require a strict majority of the configured full window.  A partial
        # startup buffer cannot prematurely authorize motion.
        if len(self._votes) == self.policy.smoothing_windows and count > len(self._votes) // 2:
            self._stable_label = winner
        return self._stable_label or self.policy.unknown_label

    def apply(self, prediction: Prediction) -> ControlledPrediction:
        if not isinstance(prediction, Prediction):
            raise TypeError("prediction must be Prediction")
        accepted = prediction.confidence >= self.policy.confidence_threshold
        if accepted:
            output = self._smooth(prediction.label)
            reason = "accepted" if output == prediction.label else "debouncing"
        elif self.policy.low_confidence_policy == "hold" and self._stable_label is not None:
            output = self._stable_label
            reason = "low_confidence_hold"
        else:
            # Unknown is immediate and bypasses smoothing: uncertainty must not
            # leave a previously authorized motion active.
            self._stable_label = None
            self._candidate_label = None
            self._candidate_count = 0
            self._votes.clear()
            output = self.policy.unknown_label
            reason = "low_confidence_unknown"
        return ControlledPrediction(
            output_label=output,
            raw_label=prediction.label,
            confidence=prediction.confidence,
            accepted=accepted,
            reason=reason,
            stable_label=self._stable_label or self.policy.unknown_label,
            window_end_timestamp_ns=prediction.window_end_timestamp_ns,
            prediction=prediction,
        )

    def push(self, sample: Sequence[float], timestamp_ns: int) -> ControlledPrediction | None:
        prediction = self.classifier.push(sample, timestamp_ns)
        return None if prediction is None else self.apply(prediction)

    def push_many(
        self,
        samples: Iterable[Sequence[float]],
        timestamps_ns: Iterable[int],
    ) -> list[ControlledPrediction]:
        sentinel = object()
        outputs: list[ControlledPrediction] = []
        sample_iterator = iter(samples)
        timestamp_iterator = iter(timestamps_ns)
        while True:
            sample = next(sample_iterator, sentinel)
            timestamp = next(timestamp_iterator, sentinel)
            if sample is sentinel and timestamp is sentinel:
                return outputs
            if sample is sentinel or timestamp is sentinel:
                raise ValueError("samples and timestamps_ns must have equal lengths")
            output = self.push(sample, timestamp)  # type: ignore[arg-type]
            if output is not None:
                outputs.append(output)


def _plain_mapping(value: object, name: str) -> dict[str, object]:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(cast(_DataclassInstance, value))
    if not isinstance(value, Mapping):
        raise ValueError(f"bundle {name} must be an object")
    return dict(value)


def _mapping_int(value: object) -> int:
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        raise TypeError("value is not integer-convertible")
    return int(value)


class RealtimeEmgClassifier:
    """Apply one bundle to a strictly bound acquisition stream.

    The ring and chronological-window scratch arrays are allocated at
    construction time. ``push`` performs no disk, network, BLE, or GUI work.
    """

    def __init__(
        self,
        bundle: EmgLdaBundle,
        acquisition: AcquisitionMetadata,
        hand_side: HandSide,
        *,
        evaluation_mode: bool = False,
    ) -> None:
        if not isinstance(bundle, EmgLdaBundle):
            raise TypeError("bundle must be EmgLdaBundle")
        if not isinstance(acquisition, AcquisitionMetadata):
            raise TypeError("acquisition must be AcquisitionMetadata")
        if not isinstance(hand_side, HandSide):
            raise TypeError("hand_side must be HandSide")
        if not isinstance(evaluation_mode, bool):
            raise TypeError("evaluation_mode must be bool")
        if bundle.deployment_status == "engineering_only" and not evaluation_mode:
            raise ValueError(
                "engineering-only bundle requires explicit evaluation_mode=True"
            )

        window = _plain_mapping(bundle.window, "window")
        try:
            window_ms = _mapping_int(window["window_ms"])
            step_ms = _mapping_int(window["step_ms"])
            window_samples = _mapping_int(window["window_samples"])
            step_samples = _mapping_int(window["step_samples"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("bundle window contract is invalid") from exc
        if (window_ms, step_ms) != (200, 50):
            raise ValueError("bundle must use the 200/50 ms inference contract")
        if window_samples <= 0 or step_samples <= 0 or step_samples > window_samples:
            raise ValueError("bundle sample window contract is invalid")

        sample_rate = _plain_mapping(bundle.sample_rate, "sample_rate")
        if sample_rate != asdict(acquisition.sample_rate):
            raise ValueError("acquisition sample_rate does not match bundle")
        if not acquisition.sample_rate.confirmed:
            raise ValueError("acquisition sample_rate must be confirmed")
        rate_hz = acquisition.sample_rate.value_hz
        assert rate_hz is not None
        expected_window = rate_hz * window_ms / 1000.0
        expected_step = rate_hz * step_ms / 1000.0
        if (
            not expected_window.is_integer()
            or not expected_step.is_integer()
            or window_samples != int(expected_window)
            or step_samples != int(expected_step)
        ):
            raise ValueError("bundle window samples do not match confirmed sample_rate")

        signal_chain = _plain_mapping(bundle.signal_chain, "signal_chain")
        if signal_chain != asdict(acquisition.signal_chain):
            raise ValueError("acquisition signal_chain does not match bundle")
        if bundle.sample_format != acquisition.signal_chain.sample_format:
            raise ValueError("acquisition sample_format does not match bundle")
        notification = _plain_mapping(
            bundle.notification_protocol, "notification_protocol"
        )
        if notification != asdict(acquisition.notification_packet_protocol):
            raise ValueError(
                "acquisition notification_packet_protocol does not match bundle"
            )
        if bundle.hand_side != hand_side.value:
            raise ValueError("hand_side does not match bundle")

        channel_count = bundle.channel_count
        if (
            not isinstance(channel_count, int)
            or isinstance(channel_count, bool)
            or channel_count <= 0
        ):
            raise ValueError("bundle channel_count is invalid")

        self.bundle = bundle
        self.acquisition = acquisition
        self.hand_side = hand_side
        self.window_samples = window_samples
        self.step_samples = step_samples
        self.channel_count = channel_count
        self._ring = np.empty((window_samples, channel_count), dtype=np.float64)
        self._window = np.empty_like(self._ring)
        self.reset()

    def reset(self) -> None:
        """Discard all buffered samples, timestamps, and cadence state."""

        self._write_index = 0
        self._sample_count = 0
        self._next_emit = self.window_samples
        self._last_timestamp_ns: Optional[int] = None

    def _ordered_window(self) -> np.ndarray:
        split = self.window_samples - self._write_index
        self._window[:split] = self._ring[self._write_index :]
        if self._write_index:
            self._window[split:] = self._ring[: self._write_index]
        return self._window

    def push(self, sample: Sequence[float], timestamp_ns: int) -> Prediction | None:
        """Add one sample and emit on the fixed 200/50 ms cadence."""

        if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
            raise ValueError("timestamp_ns must be an integer")
        if timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self._last_timestamp_ns is not None and timestamp_ns <= self._last_timestamp_ns:
            raise ValueError("timestamps must be strictly increasing")

        values = np.asarray(sample)
        if values.ndim != 1 or values.shape[0] != self.channel_count:
            raise ValueError("sample channel count does not match bundle")
        if values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
            raise ValueError("sample values must be finite real numbers")
        values64 = values.astype(np.float64, copy=False)

        self._ring[self._write_index] = values64
        self._write_index = (self._write_index + 1) % self.window_samples
        self._sample_count += 1
        self._last_timestamp_ns = timestamp_ns
        if self._sample_count < self._next_emit:
            return None
        self._next_emit += self.step_samples

        total_start = perf_counter_ns()
        feature_start = total_start
        features = extract_time_domain_features(
            self._ordered_window(), self.bundle.feature_spec
        )
        feature_end = perf_counter_ns()
        probabilities_raw = np.asarray(self.bundle.predict_proba(features))
        predict_end = perf_counter_ns()
        if probabilities_raw.shape == (1, len(self.bundle.labels)):
            probabilities_raw = probabilities_raw[0]
        if probabilities_raw.shape != (len(self.bundle.labels),):
            raise ValueError("bundle predict_proba returned an invalid shape")
        probabilities = tuple(float(value) for value in probabilities_raw)
        if not np.all(np.isfinite(probabilities_raw)):
            raise ValueError("bundle predict_proba returned non-finite values")
        label_index = int(np.argmax(probabilities_raw))
        return Prediction(
            label=self.bundle.labels[label_index],
            probabilities=probabilities,
            confidence=probabilities[label_index],
            window_end_timestamp_ns=timestamp_ns,
            bundle_id=self.bundle.bundle_id,
            feature_ns=feature_end - feature_start,
            predict_ns=predict_end - feature_end,
            total_ns=predict_end - total_start,
        )

    def push_many(
        self,
        samples: Iterable[Sequence[float]],
        timestamps_ns: Iterable[int],
    ) -> list[Prediction]:
        """Push paired sample/timestamp iterables and return emitted predictions."""

        sentinel = object()
        predictions: list[Prediction] = []
        sample_iterator = iter(samples)
        timestamp_iterator = iter(timestamps_ns)
        while True:
            sample = next(sample_iterator, sentinel)
            timestamp = next(timestamp_iterator, sentinel)
            if sample is sentinel and timestamp is sentinel:
                return predictions
            if sample is sentinel or timestamp is sentinel:
                raise ValueError("samples and timestamps_ns must have equal lengths")
            prediction = self.push(sample, timestamp)  # type: ignore[arg-type]
            if prediction is not None:
                predictions.append(prediction)
