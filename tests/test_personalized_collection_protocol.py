import copy
import hashlib
import json

import pytest

from collection_protocol import build_provisional_collection_plan
from personalized_collection_protocol import (
    ALL_FINGER_SETS,
    REQUIRED_SAMPLE_FIELDS,
    build_personalized_collection_plan,
    validate_personalized_collection_plan,
    write_personalized_collection_plan,
)


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _set_path(value, path, replacement):
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement


def test_exact_split_coverage_counts_and_unique_duration():
    plan = build_personalized_collection_plan()
    blocks = plan["blocks"]
    split = plan["split_and_evaluation"]["assignments"]
    assert blocks["structured_trajectory"]["conditions_per_split"] == 132
    assert blocks["structured_trajectory"]["trial_count"] == 396
    assert blocks["structured_trajectory"]["planned_effective_seconds"] == 4752
    assert blocks["activation_hold"]["conditions_per_split"] == 44
    assert blocks["activation_hold"]["trial_count"] == 132
    assert blocks["activation_hold"]["planned_effective_seconds"] == 1056
    assert blocks["rest_no_action"]["planned_effective_seconds"] == 240
    assert blocks["free_continuous"]["planned_effective_seconds"] == 600
    assert blocks["closed_loop_hard_mining"]["planned_effective_seconds"] == 600
    assert blocks["robustness"]["trial_count"] == 198
    assert blocks["robustness"]["planned_effective_seconds"] == 2376
    for name in ("train", "validation", "sealed_test"):
        assert split[name]["structured_trajectory_trials"] == 132
        assert split[name]["hold_trials"] == 44
        assert split[name]["robustness_trials"] == 66
    assert [split[name]["donning_id"] for name in ("train", "validation", "sealed_test")] == [
        "donning-01", "donning-02", "donning-03"
    ]
    assert split["sealed_test"]["sealed"] is True
    assert split["sealed_test"]["closed_loop_hard_mining_trials"] == 0
    assert plan["split_and_evaluation"]["minimum_independent_donning_count"] == 5
    assert blocks["transition"]["split_trial_counts"] == {
        "train": 8, "validation": 8, "sealed_test": 8,
    }
    assert "phase metrics" in blocks["transition"]["evaluation_rule"]
    hard_assignments = blocks["closed_loop_hard_mining"]["independent_assignments"]
    assert hard_assignments["train"]["donning_id"] == "donning-04"
    assert hard_assignments["validation"]["donning_id"] == "donning-05"
    assert hard_assignments["sealed_test"] is None
    timing = plan["timing_and_quota"]
    assert timing["quota_trial_count"] == 764
    assert timing["unique_quota_effective_seconds"] == 9624
    assert timing["unique_quota_effective_minutes"] == 160.4
    assert timing["transition_trial_count_not_credited"] == 24
    assert timing["transition_seconds_collected_not_credited"] == 240
    assert timing["total_trial_count_including_transition"] == 788
    assert timing["total_accepted_recorded_seconds_including_transition"] == 9864
    assert timing["total_accepted_recorded_minutes_including_transition"] == 164.4


def test_stage_zero_is_bound_to_existing_canonical_plan():
    plan = build_personalized_collection_plan()
    stage0 = build_provisional_collection_plan(subject_count=10)
    reference = plan["stage_0_reference"]
    assert reference["schema"] == stage0["schema"]
    assert reference["version"] == stage0["version"]
    assert reference["canonical_plan_sha256"] == hashlib.sha256(_canonical_json(stage0)).hexdigest()
    assert reference["labels"] == list(stage0["actions"])
    assert reference["training_provenance"] == "canonical_session"
    assert reference["quality_gate"] == "session_quality.training_usable must be true"


def test_truth_gate_and_classification_fallback_are_explicit():
    plan = build_personalized_collection_plan()
    assert plan["target_model"]["commanded_phases"] == ["flexion", "extension", "hold", "transition"]
    assert plan["annotation_schema"]["required_sample_fields"] == list(REQUIRED_SAMPLE_FIELDS)
    rule = plan["annotation_schema"]["commanded_vs_measured_rule"]
    assert "otherwise they are null" in rule and "must never be copied" in rule
    gate = plan["regression_truth_gate"]
    assert gate["fail_closed"] is True
    assert gate["common_synchronization"] == {
        "p95_absolute_error_ms_max": 20,
        "absolute_error_ms_max": 50,
        "clock_alignment_evidence_required": True,
    }
    assert gate["common_calibration"]["required_each_session_and_after_each_redonning"] is True
    assert gate["pre_fit_decision"]["readable_splits"] == ["train", "validation"]
    assert gate["pre_fit_decision"]["forbidden_split"] == "sealed_test"
    assert gate["rom_head"]["calibration_error_percentage_points_max"] == 5
    assert gate["rom_head"]["valid_synchronized_truth_fraction_per_trial_min"] == 0.95
    assert gate["rom_head"]["valid_truth_fraction_each_train_validation_finger_rom_phase_cell_min"] == 0.90
    assert gate["speed_head"]["calibration_absolute_error_hz_max"] == 0.05
    assert gate["speed_head"]["calibration_relative_error_fraction_max"] == 0.10
    assert gate["speed_head"]["required_factor_axes"] == [
        "split", "finger_set", "speed", "measured_phase"
    ]
    assert "permanently disable" in gate["speed_head"]["failure_effect"]
    assert "classification only" in gate["classification_fallback"]
    assert "once" in gate["sealed_test_truth_evaluation"]["access_time"]
    assert "Do not feed" in gate["sealed_test_truth_evaluation"]["failure_effect"]
    assert "test action labels" in gate["capture_qc_boundary"]["forbidden_before_test_unseal"]


def test_factor_hold_hard_mining_robustness_and_fatigue_contracts():
    plan = build_personalized_collection_plan()
    blocks = plan["blocks"]
    assert blocks["structured_trajectory"]["finger_sets"] == list(ALL_FINGER_SETS)
    assert blocks["structured_trajectory"]["speed_targets_hz"] == {"slow": 0.25, "normal": 0.5, "fast": 1.0}
    assert blocks["structured_trajectory"]["rom_targets_percent"] == [20, 50, 80, 100]
    assert blocks["activation_hold"]["commanded_phase"] == "hold"
    assert blocks["activation_hold"]["finger_sets"] == list(ALL_FINGER_SETS)
    assert blocks["closed_loop_hard_mining"]["split_trial_counts"] == {
        "train": 10, "validation": 10, "sealed_test": 0,
    }
    assert "permanently unavailable" in blocks["closed_loop_hard_mining"]["leakage_rule"]
    assert blocks["robustness"]["forearm_poses"] == ["neutral", "pronated", "supinated"]
    assert blocks["robustness"]["fatigue_states"] == ["rested", "fatigued"]
    fatigue = plan["fatigue_protocol"]
    assert fatigue["hard_limits"] == {
        "induction_repetitions_max": 60,
        "induction_duration_seconds_max": 300,
        "continuous_active_work_seconds_max": 120,
        "mandatory_recovery_seconds": 300,
        "rpe_recheck_every_repetitions": 10,
        "rpe_recheck_every_seconds": 60,
        "mvc_recheck_every_repetitions_if_used": 20,
    }
    assert "terminate" in fatigue["recovery_rule"]
    assert "not medical advice" in fatigue["fatigued_target"]


def test_writer_emits_equivalent_json_and_complete_markdown(tmp_path):
    destination = write_personalized_collection_plan(tmp_path / "plan")
    stored = json.loads((destination / "personalized_collection_plan.json").read_text(encoding="utf-8"))
    assert stored == build_personalized_collection_plan()
    markdown = (destination / "personalized_collection_plan.md").read_text(encoding="utf-8")
    for text in (
        "160.4 分钟", "sealed-test", "全部 132 条件", "全部 44 条件",
        "p95 ≤ 20 ms", "measured 字段必须为 null", "764 trials",
    ):
        assert text in markdown


MUTATIONS = [
    (("schema",), "bad"),
    (("version",), "bad"),
    (("stage_0_reference", "canonical_plan_sha256"), "0" * 64),
    (("stage_0_reference", "training_provenance"), "legacy_experimental"),
    (("stage_0_reference", "quality_gate"), "disabled"),
    (("blocks", "structured_trajectory", "finger_sets"), ["thumb"]),
    (("blocks", "structured_trajectory", "speed_targets_hz", "slow"), 0.5),
    (("blocks", "structured_trajectory", "rom_targets_percent"), [50]),
    (("blocks", "structured_trajectory", "trial_count"), 1),
    (("blocks", "activation_hold", "commanded_phase"), "transition"),
    (("blocks", "activation_hold", "trial_count"), 1),
    (("blocks", "transition", "quota_credit_seconds"), 1),
    (("blocks", "closed_loop_hard_mining", "split_trial_counts", "sealed_test"), 1),
    (("blocks", "robustness", "forearm_poses"), ["neutral"]),
    (("timing_and_quota", "unique_quota_effective_seconds"), 1),
    (("annotation_schema", "required_sample_fields"), ["finger_set"]),
    (("annotation_schema", "commanded_vs_measured_rule"), "copy cue"),
    (("regression_truth_gate", "fail_closed"), False),
    (("regression_truth_gate", "common_synchronization", "p95_absolute_error_ms_max"), 100),
    (("regression_truth_gate", "common_calibration", "required_each_session_and_after_each_redonning"), False),
    (("regression_truth_gate", "rom_head", "valid_synchronized_truth_fraction_per_trial_min"), 0.1),
    (("regression_truth_gate", "speed_head", "calibration_absolute_error_hz_max"), 1.0),
    (("regression_truth_gate", "pre_fit_decision", "readable_splits"), ["train", "validation", "sealed_test"]),
    (("regression_truth_gate", "sealed_test_truth_evaluation", "access_time"), "before fitting"),
    (("blocks", "transition", "split_trial_counts", "sealed_test"), 0),
    (("blocks", "closed_loop_hard_mining", "independent_assignments", "train", "donning_id"), "donning-01"),
    (("split_and_evaluation", "minimum_independent_donning_count"), 3),
    (("split_and_evaluation", "assignments", "sealed_test", "sealed"), False),
    (("split_and_evaluation", "assignments", "sealed_test", "structured_trajectory_trials"), 0),
    (("split_and_evaluation", "cross_subject_claim_allowed"), True),
    (("fatigue_protocol", "hard_limits", "induction_repetitions_max"), 1000),
    (("fatigue_protocol", "recovery_rule"), "continue"),
]


@pytest.mark.parametrize(("path", "replacement"), MUTATIONS)
def test_validator_rejects_every_core_contract_mutation(path, replacement):
    plan = copy.deepcopy(build_personalized_collection_plan())
    _set_path(plan, path, replacement)
    with pytest.raises(ValueError, match="canonical fail-closed contract"):
        validate_personalized_collection_plan(plan)


def test_validator_accepts_only_canonical_plan():
    validate_personalized_collection_plan(build_personalized_collection_plan())


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("subject_scope", "subject_count"), True),
        (("regression_truth_gate", "common_synchronization", "p95_absolute_error_ms_max"), 20.0),
        (("timing_and_quota", "quota_trial_count"), 764.0),
    ],
)
def test_validator_rejects_bool_int_and_int_float_type_confusion(path, replacement):
    plan = copy.deepcopy(build_personalized_collection_plan())
    _set_path(plan, path, replacement)
    with pytest.raises(ValueError, match="canonical fail-closed contract"):
        validate_personalized_collection_plan(plan)
