"""Latency benchmark for the NumPy-only EMG streaming inference path."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from emg_model_bundle import EmgLdaBundle
from emg_protocol import (
    AcquisitionMetadata,
    HandSide,
    NotificationPacketProtocol,
    RateDescriptor,
    SignalChain,
)
from realtime_emg_inference import Prediction, RealtimeEmgClassifier


MIN_WARMUP = 100
MIN_MEASURED = 2_000
INFERENCE_COMPUTE_P99_LIMIT_NS = 50_000_000


def latency_percentiles(values: Sequence[int]) -> dict[str, float]:
    """Return deterministic linear p50/p95/p99 statistics in nanoseconds."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("latency values must be a non-empty finite sequence")
    percentiles = np.percentile(array, [50, 95, 99], method="linear")
    return {
        "p50": float(percentiles[0]),
        "p95": float(percentiles[1]),
        "p99": float(percentiles[2]),
    }


def _acquisition_from_bundle(bundle: EmgLdaBundle) -> AcquisitionMetadata:
    return AcquisitionMetadata(
        sample_rate=RateDescriptor.from_mapping(bundle.sample_rate),
        signal_chain=SignalChain.from_mapping(bundle.signal_chain),
        notification_packet_protocol=NotificationPacketProtocol.from_mapping(
            bundle.notification_protocol
        ),
    )


def _next_prediction(
    classifier: RealtimeEmgClassifier,
    rng: np.random.Generator,
    timestamp_ns: int,
) -> tuple[Prediction, int]:
    while True:
        timestamp_ns += 1
        sample = rng.standard_normal(classifier.channel_count).tolist()
        prediction = classifier.push(sample, timestamp_ns)
        if prediction is not None:
            return prediction, timestamp_ns


def benchmark_classifier(
    classifier: RealtimeEmgClassifier,
    *,
    warmup: int = MIN_WARMUP,
    iterations: int = MIN_MEASURED,
    seed: int = 0,
) -> dict[str, object]:
    """Warm the classifier, then summarize exactly ``iterations`` predictions."""

    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(seed)
    timestamp_ns = 0

    def produce() -> Prediction:
        nonlocal timestamp_ns
        prediction, timestamp_ns = _next_prediction(classifier, rng, timestamp_ns)
        return prediction

    for _ in range(warmup):
        produce()
    measured = [produce() for _ in range(iterations)]
    feature = latency_percentiles([item.feature_ns for item in measured])
    prediction = latency_percentiles([item.predict_ns for item in measured])
    total = latency_percentiles([item.total_ns for item in measured])
    return {
        "schema": "emg.inference_latency",
        "version": "1.1",
        "bundle_id": classifier.bundle.bundle_id,
        "warmup_count": warmup,
        "measured_count": iterations,
        "seed": seed,
        "window_samples": classifier.window_samples,
        "step_samples": classifier.step_samples,
        "latency_ns": {
            "feature": feature,
            "prediction": prediction,
            "total": total,
        },
        "inference_compute_p99_limit_ns": INFERENCE_COMPUTE_P99_LIMIT_NS,
        "inference_compute_latency_budget_passed": (
            total["p99"] < INFERENCE_COMPUTE_P99_LIMIT_NS
        ),
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }


def benchmark_bundle(
    bundle: EmgLdaBundle,
    *,
    warmup: int = MIN_WARMUP,
    iterations: int = MIN_MEASURED,
    seed: int = 0,
) -> dict[str, object]:
    acquisition = _acquisition_from_bundle(bundle)
    classifier = RealtimeEmgClassifier(
        bundle,
        acquisition,
        HandSide(bundle.hand_side),
        evaluation_mode=True,
    )
    return benchmark_classifier(
        classifier, warmup=warmup, iterations=iterations, seed=seed
    )


def _minimum_count(name: str, minimum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"{name} must be at least {minimum}")
        return parsed

    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument(
        "--warmup", type=_minimum_count("warmup", MIN_WARMUP), default=MIN_WARMUP
    )
    parser.add_argument(
        "--iterations",
        type=_minimum_count("iterations", MIN_MEASURED),
        default=MIN_MEASURED,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = benchmark_bundle(
            EmgLdaBundle.load(args.bundle),
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed,
        )
        encoded = json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n"
        if args.json is not None:
            args.json.write_text(encoded, encoding="utf-8")
        sys.stdout.write(encoded)
        return 0
    except (OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"benchmark rejected: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
