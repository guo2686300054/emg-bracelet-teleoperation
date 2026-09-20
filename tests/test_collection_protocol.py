import json

import pytest

from collection_protocol import build_provisional_collection_plan, write_collection_plan


def test_provisional_plan_is_explicit_quantified_and_requires_learning_curve(tmp_path):
    plan = build_provisional_collection_plan(subject_count=10)
    assert plan["status"] == "provisional_pending_hardware_learning_curve"
    assert plan["deployment_decision_allowed"] is False
    assert set(plan["actions"]) == {"rest", "fist", "open_hand"}
    for action in plan["actions"].values():
        assert action == {
            "visits_per_subject": 3,
            "recordings_per_visit": 10,
            "hold_seconds_per_recording": 5,
            "recordings_per_subject_total": 30,
            "hold_seconds_per_subject_total": 150,
        }
    assert plan["recordings_per_subject_total"] == 90
    assert "one training-capture GUI recording" in plan["gui_recording_unit"]
    assert "experiment_batch" in plan["identifier_templates"]
    assert "session_id" in plan["identifier_templates"]
    assert plan["rest_intervals"]["between_recordings_seconds"] == 10
    assert "_retry-01" in plan["missed_or_rejected_recording_rule"]
    assert plan["learning_curve_update"]["required_after_hardware_returns"] is True
    assert "training_usable=false" in plan["known_blockers"][-1]

    destination = write_collection_plan(tmp_path / "plan", subject_count=10)
    stored = json.loads((destination / "collection_plan.json").read_text(encoding="utf-8"))
    assert stored == plan
    assert "provisional" in (destination / "collection_plan.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("value", [True, 0, 2, 3.0])
def test_subject_count_fails_closed(value):
    with pytest.raises(ValueError, match="subject_count"):
        build_provisional_collection_plan(subject_count=value)
