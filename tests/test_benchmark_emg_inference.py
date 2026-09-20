from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmark_emg_inference import (
    INFERENCE_COMPUTE_P99_LIMIT_NS,
    MIN_MEASURED,
    MIN_WARMUP,
    _parser,
    benchmark_classifier,
    latency_percentiles,
    main,
)
from emg_model_bundle import publish_bundle
from test_emg_model_bundle import _evaluation
from test_realtime_emg_inference import make_bundle, make_classifier


def test_latency_percentiles_and_warmup_exclusion_are_deterministic(monkeypatch):
    classifier = make_classifier()
    clock_values = []
    base = 0
    for prediction_number in range(1, 7):
        clock_values.extend(
            [
                base,
                base + prediction_number,
                base + prediction_number * 11,
            ]
        )
        base += 1_000
    clock = iter(clock_values)
    monkeypatch.setattr(
        "realtime_emg_inference.perf_counter_ns", lambda: next(clock)
    )

    report = benchmark_classifier(classifier, warmup=2, iterations=4)
    assert report["warmup_count"] == 2
    assert report["measured_count"] == 4
    assert report["latency_ns"]["feature"] == latency_percentiles([3, 4, 5, 6])
    assert report["latency_ns"]["prediction"] == latency_percentiles(
        [30, 40, 50, 60]
    )
    assert report["latency_ns"]["total"] == latency_percentiles([33, 44, 55, 66])
    assert report["inference_compute_latency_budget_passed"] is True
    assert "live_control_gate_passed" not in report


def test_gate_is_recorded_but_does_not_fail_benchmark(monkeypatch):
    classifier = make_classifier()
    clock_values = []
    for base in (0, 100_000_000, 200_000_000):
        clock_values.extend([base, base + 1, base + INFERENCE_COMPUTE_P99_LIMIT_NS])
    clock = iter(clock_values)
    monkeypatch.setattr(
        "realtime_emg_inference.perf_counter_ns", lambda: next(clock)
    )
    report = benchmark_classifier(classifier, warmup=0, iterations=3)
    assert report["inference_compute_latency_budget_passed"] is False


def test_cli_enforces_minimum_counts():
    defaults = _parser().parse_args(["--bundle", "unused"])
    assert defaults.warmup == MIN_WARMUP
    assert defaults.iterations == MIN_MEASURED
    with pytest.raises(SystemExit):
        _parser().parse_args(["--bundle", "unused", "--warmup", "99"])
    with pytest.raises(SystemExit):
        _parser().parse_args(["--bundle", "unused", "--iterations", "1999"])


def test_real_cli_writes_machine_readable_report(tmp_path: Path, capsys):
    bundle_dir = tmp_path / "bundle"
    report_path = tmp_path / "latency.json"
    publish_bundle(bundle_dir, make_bundle(), _evaluation())
    result = main(
        [
            "--bundle",
            str(bundle_dir),
            "--warmup",
            str(MIN_WARMUP),
            "--iterations",
            str(MIN_MEASURED),
            "--json",
            str(report_path),
        ]
    )
    assert result == 0
    written = json.loads(report_path.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert written == printed
    assert written["warmup_count"] == 100
    assert written["measured_count"] == 2000
    assert set(written["latency_ns"]) == {"feature", "prediction", "total"}
    assert all(
        set(values) == {"p50", "p95", "p99"}
        for values in written["latency_ns"].values()
    )
    assert isinstance(written["inference_compute_latency_budget_passed"], bool)
    assert "live_control_gate_passed" not in written
    assert written["environment"]["numpy"] == np.__version__
