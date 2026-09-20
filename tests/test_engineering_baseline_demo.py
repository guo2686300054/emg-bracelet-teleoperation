import json

import pytest

import engineering_baseline_demo as demo
from emg_model_bundle import EmgLdaBundle


def test_demo_is_reproducible_engineering_only_and_reports_required_evidence(
    tmp_path, monkeypatch
):
    original = demo.benchmark_bundle

    def faster_benchmark(bundle, *, warmup, iterations, seed):
        assert warmup == 100
        assert iterations == 2_000
        return original(bundle, warmup=2, iterations=5, seed=seed)

    monkeypatch.setattr(demo, "benchmark_bundle", faster_benchmark)
    output = demo.run_demo(tmp_path / "baseline", seed=7)
    report = json.loads((output / "engineering_report.json").read_text(encoding="utf-8"))
    bundle = EmgLdaBundle.load(output / "bundle")

    assert report["provenance"] == bundle.provenance == "synthetic_test"
    assert report["deployment_status"] == bundle.deployment_status == "engineering_only"
    assert bundle.channel_count == demo.EMG_CHANNEL_COUNT == 8
    assert report["real_accuracy_claim_allowed"] is False
    assert report["live_control_eligibility"] == "not_evaluated"
    assert report["deployment_gate_passed"] is False
    assert "live_control_gate_passed" not in json.dumps(report)
    assert "do not represent" in report["warning"]
    for split in ("train", "validation", "test"):
        metrics = report["evaluation"]["splits"][split]
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert len(metrics["confusion_matrix"]) == 3
        assert metrics["error_count"] >= len(metrics["error_windows"])
        assert len(metrics["error_windows"]) <= 50
    assert "p99" in report["latency_benchmark"]["latency_ns"]["total"]

    with pytest.raises(FileExistsError):
        demo.run_demo(output, seed=7)
