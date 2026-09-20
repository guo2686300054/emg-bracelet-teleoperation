"""Generate the fail-closed single-subject dexterous-hand EMG protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from collection_protocol import (
    PLAN_SCHEMA as STAGE0_PLAN_SCHEMA,
    PLAN_VERSION as STAGE0_PLAN_VERSION,
    build_provisional_collection_plan,
)


PLAN_SCHEMA = "emg.personalized.collection.plan"
PLAN_VERSION = "2.0"
SINGLE_FINGER_SETS = ("thumb", "index", "middle", "ring", "little")
MULTI_FINGER_SETS = (
    "index+middle", "ring+little", "thumb+index", "thumb+middle",
    "thumb+index+middle", "thumb+index+middle+ring+little",
)
ALL_FINGER_SETS = SINGLE_FINGER_SETS + MULTI_FINGER_SETS
SPEEDS_HZ = {"slow": 0.25, "normal": 0.5, "fast": 1.0}
ROM_LEVELS_PERCENT = (20, 50, 80, 100)
FOREARM_POSES = ("neutral", "pronated", "supinated")
FATIGUE_STATES = ("rested", "fatigued")
SPLITS = ("train", "validation", "sealed_test")
REQUIRED_SAMPLE_FIELDS = (
    "action_family", "finger_set", "speed", "commanded_rom_percent",
    "measured_rom_percent", "forearm_pose", "donning_id", "fatigue_state",
    "label_source",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _stage0_reference() -> dict[str, Any]:
    stage0 = build_provisional_collection_plan(subject_count=10)
    acceptance = stage0["acceptance"]
    return {
        "status": "retained_unchanged",
        "source_module": "collection_protocol.py",
        "schema": STAGE0_PLAN_SCHEMA,
        "version": STAGE0_PLAN_VERSION,
        "canonical_plan_sha256": hashlib.sha256(_canonical_json(stage0)).hexdigest(),
        "canonical_subject_count": stage0["subject_count"],
        "labels": list(stage0["actions"]),
        "training_provenance": acceptance["provenance"],
        "quality_gate": acceptance["quality_gate"],
        "purpose": "pipeline and coarse three-class engineering baseline only",
        "not_sufficient_for": "continuous dexterous-hand teleoperation",
    }


def build_personalized_collection_plan() -> dict[str, Any]:
    """Return a fresh deterministic copy of the complete protocol contract."""
    conditions_per_split = len(ALL_FINGER_SETS) * len(SPEEDS_HZ) * len(ROM_LEVELS_PERCENT)
    structured_trial_seconds = 12
    structured_trials = conditions_per_split * len(SPLITS)
    structured_seconds = structured_trials * structured_trial_seconds

    hold_conditions_per_split = len(ALL_FINGER_SETS) * len(ROM_LEVELS_PERCENT)
    hold_trial_seconds = 8
    hold_trials = hold_conditions_per_split * len(SPLITS)
    hold_seconds = hold_trials * hold_trial_seconds

    rest_trials, rest_trial_seconds = 12, 20
    transition_trials, transition_trial_seconds = 24, 10
    free_trials, free_trial_seconds = 6, 100
    closed_trials, closed_trial_seconds = 20, 30
    robustness_trials = len(SPLITS) * len(FOREARM_POSES) * len(FATIGUE_STATES) * len(ALL_FINGER_SETS)
    robustness_trial_seconds = 12

    rest_seconds = rest_trials * rest_trial_seconds
    transition_seconds = transition_trials * transition_trial_seconds
    free_seconds = free_trials * free_trial_seconds
    closed_seconds = closed_trials * closed_trial_seconds
    robustness_seconds = robustness_trials * robustness_trial_seconds
    quota_trials = structured_trials + hold_trials + rest_trials + free_trials + closed_trials + robustness_trials
    quota_seconds = structured_seconds + hold_seconds + rest_seconds + free_seconds + closed_seconds + robustness_seconds

    split_assignments = {
        "train": {
            "capture_date": "date-train (unique date)",
            "session_id_prefix": "personal-train-",
            "donning_id": "donning-01",
            "structured_trajectory_trials": conditions_per_split,
            "hold_trials": hold_conditions_per_split,
            "rest_no_action_trials": 4,
            "free_trials": 2,
            "closed_loop_hard_mining_trials": 0,
            "robustness_trials": 3 * 2 * len(ALL_FINGER_SETS),
            "sealed": False,
        },
        "validation": {
            "capture_date": "date-validation (different date)",
            "session_id_prefix": "personal-validation-",
            "donning_id": "donning-02",
            "structured_trajectory_trials": conditions_per_split,
            "hold_trials": hold_conditions_per_split,
            "rest_no_action_trials": 4,
            "free_trials": 2,
            "closed_loop_hard_mining_trials": 0,
            "robustness_trials": 3 * 2 * len(ALL_FINGER_SETS),
            "sealed": False,
        },
        "sealed_test": {
            "capture_date": "date-test (later untouched date)",
            "session_id_prefix": "personal-test-",
            "donning_id": "donning-03",
            "structured_trajectory_trials": conditions_per_split,
            "hold_trials": hold_conditions_per_split,
            "rest_no_action_trials": 4,
            "free_trials": 2,
            "closed_loop_hard_mining_trials": 0,
            "robustness_trials": 3 * 2 * len(ALL_FINGER_SETS),
            "sealed": True,
        },
    }

    return {
        "schema": PLAN_SCHEMA,
        "version": PLAN_VERSION,
        "subject_scope": {
            "subject_count": 1,
            "purpose": "personalized control for the recorded wearer",
            "generalization_limit": (
                "One subject cannot establish cross-subject generalization; report only "
                "within-subject unseen-date/session/donning results."
            ),
        },
        "stage_0_reference": _stage0_reference(),
        "target_model": {
            "recommended": "multi_task_when_truth_gate_passes_otherwise_classification_only",
            "classification_outputs": ["action family", "finger set"],
            "conditional_regression_outputs": ["continuous ROM", "optional speed"],
            "commanded_phases": ["flexion", "extension", "hold", "transition"],
            "measured_phases": ["flexion", "extension", "hold", "transition", None],
            "trajectory_rule": (
                "Trajectory trials repeat flexion-extension; sample-level commanded_phase is "
                "the cue, while measured_phase comes only from synchronized external truth."
            ),
        },
        "timing_and_quota": {
            "quota_definition": (
                "Only quality-gate-passing labeled signal earns quota credit. Countdowns, "
                "breaks, rejected frames, and transition samples are excluded. No recording "
                "segment may be credited to more than one block."
            ),
            "quota_trial_count": quota_trials,
            "unique_quota_effective_seconds": quota_seconds,
            "unique_quota_effective_minutes": quota_seconds / 60,
            "transition_trial_count_not_credited": transition_trials,
            "transition_seconds_collected_not_credited": transition_seconds,
            "total_trial_count_including_transition": quota_trials + transition_trials,
            "total_accepted_recorded_seconds_including_transition": quota_seconds + transition_seconds,
            "total_accepted_recorded_minutes_including_transition": (quota_seconds + transition_seconds) / 60,
        },
        "blocks": {
            "structured_trajectory": {
                "minimum_required_seconds": 20 * 60,
                "planned_effective_seconds": structured_seconds,
                "trial_count": structured_trials,
                "conditions_per_split": conditions_per_split,
                "trials_per_condition_per_split": 1,
                "seconds_per_trial": structured_trial_seconds,
                "finger_sets": list(ALL_FINGER_SETS),
                "speed_targets_hz": SPEEDS_HZ,
                "speed_definition": "complete flexion-extension cycles per second",
                "rom_targets_percent": list(ROM_LEVELS_PERCENT),
                "factorial_rule": (
                    "Each split independently contains all 11 finger sets x 3 commanded "
                    "speeds x 4 commanded ROM levels = 132 conditions."
                ),
                "commanded_phase_sequence": ["flexion", "extension"],
            },
            "activation_hold": {
                "planned_effective_seconds": hold_seconds,
                "trial_count": hold_trials,
                "conditions_per_split": hold_conditions_per_split,
                "trials_per_condition_per_split": 1,
                "seconds_per_trial": hold_trial_seconds,
                "finger_sets": list(ALL_FINGER_SETS),
                "rom_targets_percent": list(ROM_LEVELS_PERCENT),
                "commanded_phase": "hold",
                "factorial_rule": (
                    "Each split independently contains all 11 finger sets x 4 commanded "
                    "ROM levels = 44 sustained activation conditions."
                ),
            },
            "rest_no_action": {
                "planned_effective_seconds": rest_seconds,
                "trial_count": rest_trials,
                "trials_per_split": 4,
                "seconds_per_trial": rest_trial_seconds,
                "labels": ["rest", "no_action"],
                "purpose": "negative evidence for false-activation suppression",
            },
            "transition": {
                "planned_recorded_seconds": transition_seconds,
                "quota_credit_seconds": 0,
                "trial_count": transition_trials,
                "split_trial_counts": {"train": 8, "validation": 8, "sealed_test": 8},
                "seconds_per_trial": transition_trial_seconds,
                "label": "transition",
                "purpose": "safe state changes without inflating action quotas",
                "evaluation_rule": (
                    "Transition is a positive class for commanded/measured phase classification "
                    "and phase metrics. Its eight trials per split remain isolated inside that "
                    "split's base date/session/donning and earn zero duration quota credit."
                ),
            },
            "free_continuous": {
                "minimum_required_seconds": 10 * 60,
                "planned_effective_seconds": free_seconds,
                "trial_count": free_trials,
                "trials_per_split": 2,
                "seconds_per_trial": free_trial_seconds,
                "instruction": "randomize fingers, cadence, amplitude, and order; do not repeat a script",
            },
            "closed_loop_hard_mining": {
                "minimum_required_seconds": 10 * 60,
                "planned_effective_seconds": closed_seconds,
                "trial_count": closed_trials,
                "split_trial_counts": {"train": 10, "validation": 10, "sealed_test": 0},
                "independent_assignments": {
                    "train": {
                        "capture_date": "date-train-hard (after initial model freeze)",
                        "session_id_prefix": "personal-train-hard-",
                        "donning_id": "donning-04",
                        "trial_count": 10,
                    },
                    "validation": {
                        "capture_date": "date-validation-hard (different later date)",
                        "session_id_prefix": "personal-validation-hard-",
                        "donning_id": "donning-05",
                        "trial_count": 10,
                    },
                    "sealed_test": None,
                },
                "seconds_per_trial": closed_trial_seconds,
                "enabled_initially": False,
                "enable_condition": "only after the first personalized model is frozen",
                "leakage_rule": (
                    "Hard-mined samples use new post-freeze dates/sessions/donnings and may enter "
                    "train/validation only. Sealed-test data, "
                    "predictions, errors, labels, and metadata remain permanently unavailable "
                    "to mining, selection, relabeling, tuning, and retraining."
                ),
            },
            "robustness": {
                "planned_effective_seconds": robustness_seconds,
                "trial_count": robustness_trials,
                "trials_per_split": 3 * 2 * len(ALL_FINGER_SETS),
                "seconds_per_trial": robustness_trial_seconds,
                "factorial_rule": (
                    "Within each split/donning: 3 forearm poses x rested/fatigued x 11 "
                    "finger sets at commanded normal 0.5 Hz and 50% ROM."
                ),
                "forearm_poses": list(FOREARM_POSES),
                "fatigue_states": list(FATIGUE_STATES),
                "fatigue_pairing": "record rested and fatigued trials for every matched condition",
            },
        },
        "split_and_evaluation": {
            "unit": "date/session/donning-disjoint",
            "minimum_independent_donning_count": 5,
            "assignments": split_assignments,
            "factor_coverage_rule": (
                "Train, validation, and sealed test each independently cover all 132 trajectory "
                "conditions and all 44 hold conditions before any window extraction."
            ),
            "grouping_rule": (
                "No recording, overlapping window, session, capture date, or donning_id may "
                "appear in multiple splits. Assign immutable groups before window extraction."
            ),
            "sealed_test_rule": (
                "Seal test before first model fitting; access it once for final evaluation. "
                "Never feed test-derived choices back into collection, training, or tuning."
            ),
            "chronology": [
                "collect base train on donning-01 and base validation on donning-02",
                "collect base sealed test on donning-03; run target-blind capture QC then seal labels and truth",
                "decide pre-fit heads from train/validation truth and freeze the initial model",
                "collect train-hard on donning-04 and validation-hard on donning-05 on later independent dates",
                "re-run train/validation truth gates, train/tune, and freeze the final model",
                "unseal test once, independently validate test truth, and perform final evaluation",
            ],
            "cross_subject_claim_allowed": False,
        },
        "annotation_schema": {
            "required_sample_fields": list(REQUIRED_SAMPLE_FIELDS),
            "additional_required_fields": [
                "commanded_phase", "measured_phase", "commanded_speed_hz",
                "measured_speed_hz", "session_id", "recording_id", "capture_date",
                "split_assignment", "truth_valid",
            ],
            "commanded_vs_measured_rule": (
                "Commanded phase/ROM/speed describe cues only. Measured phase/ROM/speed come "
                "only from synchronized calibrated external truth; otherwise they are null. "
                "Commanded values must never be copied, aliased, or scored as measured values."
            ),
            "label_source_examples": [
                "scripted_cue", "camera_truth", "data_glove_truth", "encoder_truth",
                "closed_loop_hard_mining_with_external_truth",
            ],
        },
        "regression_truth_gate": {
            "fail_closed": True,
            "allowed_external_sources": ["camera", "data_glove", "encoder"],
            "capture_qc_boundary": {
                "allowed_before_test_unseal": [
                    "file hash and schema", "channel count", "packet and timestamp integrity",
                    "loss/duplicate/out-of-order/stale counters", "sample-rate evidence",
                    "duration", "device/electrode quality flags",
                ],
                "forbidden_before_test_unseal": [
                    "test action labels", "test measured phase", "test measured ROM",
                    "test measured speed", "test predictions", "test errors",
                ],
                "rule": (
                    "Model-independent capture QC may verify transport and file integrity only; "
                    "it must not decrypt, inspect, summarize, or select using test targets."
                ),
            },
            "common_synchronization": {
                "p95_absolute_error_ms_max": 20,
                "absolute_error_ms_max": 50,
                "clock_alignment_evidence_required": True,
            },
            "common_calibration": {
                "required_each_session_and_after_each_redonning": True,
                "calibration_record_required": True,
            },
            "pre_fit_decision": {
                "readable_splits": ["train", "validation"],
                "forbidden_split": "sealed_test",
                "decision_time": "before fitting any regression head",
                "rule": (
                    "Head eligibility is decided only from train/validation truth. Test labels, "
                    "truth, predictions, and errors remain sealed and cannot influence fitting."
                ),
            },
            "rom_head": {
                "calibration_error_percentage_points_max": 5,
                "valid_synchronized_truth_fraction_per_trial_min": 0.95,
                "valid_truth_fraction_each_train_validation_finger_rom_phase_cell_min": 0.90,
                "required_factor_axes": ["split", "finger_set", "rom", "measured_phase"],
                "failure_effect": "disable ROM regression head and all ROM regression metrics",
            },
            "speed_head": {
                "calibration_absolute_error_hz_max": 0.05,
                "calibration_relative_error_fraction_max": 0.10,
                "valid_synchronized_truth_fraction_per_trial_min": 0.95,
                "valid_truth_fraction_each_train_validation_finger_speed_phase_cell_min": 0.90,
                "required_factor_axes": ["split", "finger_set", "speed", "measured_phase"],
                "failure_effect": (
                    "permanently disable speed regression for this model version and publish no speed regression metrics"
                ),
            },
            "classification_fallback": (
                "If neither regression head passes, train/evaluate classification only. "
                "Commanded targets never substitute for measured targets."
            ),
            "sealed_test_truth_evaluation": {
                "access_time": "once, after final model and analysis code are frozen",
                "independent_gate": (
                    "Apply the same common, ROM-head, and speed-head truth thresholds to sealed test without refitting."
                ),
                "failure_effect": (
                    "Mark the affected test regression result invalid and publish no metric for "
                    "that head. Do not feed the failure, labels, truth, predictions, or errors "
                    "back into collection, feature design, fitting, tuning, or retraining."
                ),
            },
        },
        "fatigue_protocol": {
            "rested_definition": "RPE <= 2/10 after at least 5 minutes without contraction",
            "fatigued_target": (
                "RPE 6-7/10 after repeated contractions; if MVC is available, record its "
                "decline and target 15-30%. This is an engineering protocol, not medical advice."
            ),
            "hard_limits": {
                "induction_repetitions_max": 60,
                "induction_duration_seconds_max": 300,
                "continuous_active_work_seconds_max": 120,
                "mandatory_recovery_seconds": 300,
                "rpe_recheck_every_repetitions": 10,
                "rpe_recheck_every_seconds": 60,
                "mvc_recheck_every_repetitions_if_used": 20,
            },
            "required_fields": [
                "rpe_before", "rpe_during_checks", "rpe_after",
                "fatigue_induction_repetitions", "fatigue_induction_seconds",
                "mvc_decline_percent_if_available", "recovery_check_passed",
            ],
            "safety_stop": (
                "Stop immediately for pain, numbness, cramp, dizziness, abnormal weakness, "
                "RPE >= 8/10, participant request, or any hard limit. Never force ROM."
            ),
            "recovery_rule": (
                "After mandatory recovery, resume only when RPE <= 2/10 and symptoms are absent. "
                "If recovery check fails, terminate fatigue collection for that day."
            ),
        },
        "execution_order": [
            "verify stage-0 canonical provenance and quality gate",
            "assign immutable date/session/donning split groups",
            "collect complete trajectory, hold, negative, free, and robustness coverage per split",
            "seal test before first model fitting",
            "apply pre-fit regression truth gates using train/validation only",
            "train the first personalized model using train and validation only",
            "freeze it, then collect independent donning-04/05 hard examples into train/validation only",
            "freeze the final model and analysis code",
            "unseal test once, validate its truth independently, and evaluate without feedback",
        ],
        "assessment_of_original_request": (
            "Directionally correct, with minimum durations rather than caps. Execution also "
            "requires replicated split coverage, hold/negative/transition evidence, numeric "
            "cadence, commanded/measured separation, truth gates, fatigue limits, and sealing."
        ),
    }


def validate_personalized_collection_plan(plan: object) -> None:
    """Require type-sensitive canonical-JSON equality with the protocol."""
    if not isinstance(plan, dict):
        raise ValueError("plan must be an object")
    expected = build_personalized_collection_plan()
    try:
        actual_bytes = _canonical_json(plan)
    except (TypeError, ValueError) as exc:
        raise ValueError("personalized collection plan must contain canonical JSON values") from exc
    if actual_bytes != _canonical_json(expected):
        raise ValueError(
            "personalized collection plan differs from the canonical fail-closed contract"
        )


def _render_markdown(plan: dict[str, Any]) -> str:
    timing = plan["timing_and_quota"]
    return f"""# 单受试者个性化灵巧手采集协议 v2

> 原始方向正确，但用户给的是最低时长，不是上限。为了让 train/validation/sealed-test 都完整覆盖动作因子，并增加 hold 正样本和硬真值门禁，本版有效量提高到 **{timing['unique_quota_effective_minutes']:.1f} 分钟**。

## 采集总量

| 数据块 | 数量 | 唯一有效时长 |
|---|---:|---:|
| 结构化连续屈伸 | 3 splits × 132 条件 × 12 秒 = 396 trials | 79.2 分钟 |
| 激活 hold 正样本 | 3 splits × 44 条件 × 8 秒 = 132 trials | 17.6 分钟 |
| rest/no-action | 12 × 20 秒 | 4 分钟 |
| 自由连续动作 | 6 × 100 秒 | 10 分钟 |
| 闭环困难样本 | 独立 train-hard 10 + validation-hard 10 + test 0；每个 30 秒 | 10 分钟 |
| 鲁棒性 | 3 splits × 3 姿态 × 2 疲劳状态 × 11 指组 × 12 秒 | 39.6 分钟 |
| **配额合计** | **{timing['quota_trial_count']} trials** | **{timing['unique_quota_effective_minutes']:.1f} 分钟** |

另采 24 个 transition trials，共 4 分钟，不计最低配额；train/validation/sealed-test 各 8 个，进入 phase 分类和 phase 指标，但严格留在各自 base date/session/donning。全部合计 {timing['total_trial_count_including_transition']} trials、{timing['total_accepted_recorded_minutes_including_transition']:.1f} 分钟合格信号。

## 三个切分必须各自完整

| split | 日期/session/佩戴 | 连续屈伸 | hold | 闭环困难样本 | 状态 |
|---|---|---:|---:|---:|---|
| train | 独立 date-train/session/donning-01 | 全部 132 条件 | 全部 44 条件 | 0 | 可训练 |
| validation | 独立 date-validation/session/donning-02 | 全部 132 条件 | 全部 44 条件 | 0 | 只调参 |
| sealed-test | 后续 date-test/session/donning-03 | 全部 132 条件 | 全部 44 条件 | 0 | 首次训练前永久封存 |

初模冻结后，困难样本使用新的 `date-train-hard/session/donning-04` 和 `date-validation-hard/session/donning-05`，绝不复用已结束的 01/02。因此本协议最少需要 5 次独立佩戴。132 条件 = 11 指组 × slow/normal/fast（0.25/0.5/1.0 Hz）× 20/50/80/100% commanded ROM。44 个 hold 条件 = 11 指组 × 4 ROM。必须先按 date/session/donning 分组，再切窗口；重叠窗口不能跨 split。单人结果不能证明跨人泛化。

## commanded 和 measured 绝不混用

`commanded_phase/ROM/speed` 只是提示；`measured_phase/ROM/speed` 只能来自同步且校准过的摄像头、数据手套或编码器。没有真值时 measured 字段必须为 null，不能复制 commanded。

拟合前只能读取 train/validation 真值，sealed-test 的标签和真值不可提前读取。公共门限为同步误差 p95 ≤ 20 ms、最大值 ≤ 50 ms，且每个 session/重新佩戴后校准。ROM 头还要求校准误差 ≤ 5 个百分点、每 trial 真值 ≥ 95%、每个 train/validation×finger×ROM×phase 单元 ≥ 90%。速度头独立要求校准绝对误差 ≤ 0.05 Hz、相对误差 ≤ 10%、每 trial 真值 ≥ 95%、每个 train/validation×finger×speed×phase 单元 ≥ 90%；失败则该模型版本永久禁用速度回归且不得发布速度回归指标。

封存前只允许执行与模型无关的文件、通道、数据包、时间戳、丢包和设备质量 QC，不得解密或汇总 test 标签/真值。最终模型和分析代码冻结后只解封一次，并对 test truth 独立执行相同门禁；失败则相应测试结果无效，且不得把失败、标签、真值、预测或错误反馈给采集、调参或重训练。

## 疲劳安全硬限制

疲劳诱导最多 60 次或 300 秒；连续主动工作最多 120 秒；每 10 次或 60 秒复查 RPE，使用 MVC 时每 20 次复测；之后强制恢复 300 秒。疼痛、麻木、抽筋、眩晕、异常无力、RPE ≥ 8、受试者要求或触及任一硬限制时立即停止。恢复后仅当 RPE ≤ 2 且无症状才继续，否则当天终止。本协议是工程限制，不构成医疗建议。

## 模型路线

现有 `collection_protocol.py` 的 stage 0 以 schema/version/SHA-256、`canonical_session` provenance 和 `session_quality.training_usable=true` 门禁绑定，继续只做 `rest/fist/open_hand` 粗分类链路验证。精细控制在真值门禁通过后采用动作/手指集合分类 + ROM 连续回归，必要时增加速度回归；门禁失败则严格退化为分类模型。
"""


def write_personalized_collection_plan(output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plan = build_personalized_collection_plan()
    validate_personalized_collection_plan(plan)
    (destination / "personalized_collection_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "personalized_collection_plan.md").write_text(
        _render_markdown(plan), encoding="utf-8"
    )
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    destination = write_personalized_collection_plan(args.output)
    print(f"wrote personalized collection plan: {destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
