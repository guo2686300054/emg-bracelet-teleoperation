"""Generate the provisional canonical EMG collection quantity plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from training_contract import CANONICAL_LABELS


PLAN_SCHEMA = "emg.collection.plan"
PLAN_VERSION = "1.0"


def build_provisional_collection_plan(*, subject_count: int = 10) -> dict[str, Any]:
    """Return a conservative starting plan pending hardware learning curves."""

    if not isinstance(subject_count, int) or isinstance(subject_count, bool) or subject_count < 3:
        raise ValueError("subject_count must be an integer of at least 3")
    visits_per_subject = 3
    recordings_per_action_per_visit = 10
    hold_seconds = 5
    recordings_per_action = visits_per_subject * recordings_per_action_per_visit
    seconds_per_action = recordings_per_action * hold_seconds
    action_plan = {
        label: {
            "visits_per_subject": visits_per_subject,
            "recordings_per_visit": recordings_per_action_per_visit,
            "hold_seconds_per_recording": hold_seconds,
            "recordings_per_subject_total": recordings_per_action,
            "hold_seconds_per_subject_total": seconds_per_action,
        }
        for label in CANONICAL_LABELS
    }
    return {
        "schema": PLAN_SCHEMA,
        "version": PLAN_VERSION,
        "status": "provisional_pending_hardware_learning_curve",
        "deployment_decision_allowed": False,
        "subject_count": subject_count,
        "actions": action_plan,
        "recordings_per_subject_total": (
            visits_per_subject * recordings_per_action_per_visit * len(CANONICAL_LABELS)
        ),
        "gui_recording_unit": (
            "one training-capture GUI recording/session_id equals one fixed-action trial"
        ),
        "visit_definition": "one independent donning; prefer different days",
        "randomization": "counterbalance action trial order within each visit",
        "identifier_templates": {
            "experiment_batch": "exp-YYYYMMDD-protocol-v1",
            "visit_id": "visit-01 through visit-03",
            "session_id": (
                "{subject_id}_{visit_id}_{action_label}_trial-{trial_number:02d}"
            ),
        },
        "rest_intervals": {
            "between_recordings_seconds": 10,
            "between_action_blocks_seconds": 60,
            "instruction": "extend rest when fatigue or unstable baseline is observed",
        },
        "missed_or_rejected_recording_rule": (
            "Never overwrite or relabel a failed trial. Retain it as rejected evidence, then "
            "record a new session_id with suffix _retry-01 (increment for further retries) "
            "until ten quality-gate-passing recordings exist for that action in that visit."
        ),
        "acceptance": {
            "provenance": "canonical_session",
            "required_labels": list(CANONICAL_LABELS),
            "action_phase": "hold",
            "quality_gate": "session_quality.training_usable must be true",
            "split_rule": "subject-disjoint train/validation/test",
        },
        "rationale": [
            "Multiple sessions expose re-donning and day-to-day electrode variation.",
            "Five-second recordings provide many 200 ms windows without making one trial excessively long.",
            "Ten recordings per action/visit distribute evidence across repeated contractions instead of one correlated recording.",
            "Ten subjects are a practical engineering starting point, not a population-level validation claim.",
        ],
        "learning_curve_update": {
            "required_after_hardware_returns": True,
            "checkpoints_subjects": [3, 5, 8, 10],
            "checkpoints_recordings_per_action": [5, 10, 20, 30],
            "metric": "subject-disjoint validation macro_f1 with per-class recall",
            "stop_or_expand_rule": (
                "Freeze the final quantity only after the last two checkpoints improve macro_f1 "
                "by less than 0.02 and every class recall is at least 0.85; otherwise add subjects "
                "before adding more correlated windows from the same session."
            ),
        },
        "known_blockers": [
            "real sample rate is not yet confirmed",
            "real BLE loss/duplicate/out-of-order rates are not yet measured",
            "electrode wearing repeatability is not yet validated",
            "the historical experimental sessions are explicitly training_usable=false",
        ],
    }


def write_collection_plan(output_dir: str | Path, *, subject_count: int = 10) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plan = build_provisional_collection_plan(subject_count=subject_count)
    (destination / "collection_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    action = plan["actions"][CANONICAL_LABELS[0]]
    markdown = f"""# 肌电手环正式采集数量方案（临时版）

> 状态：**provisional**。当前历史数据明确不可用于正式训练；本方案只能作为硬件恢复后的首轮采集起点，不能据此宣称模型达到真实控制精度。

## 立即照此采集

- 受试者：{plan['subject_count']} 人。
- 每人：{action['visits_per_subject']} 次 visit；每次 visit 都要重新独立佩戴，优先跨天。
- 每个 visit：`rest`、`fist`、`open_hand` 各 {action['recordings_per_visit']} 次 recording/trial。
- 上位机一次“开始—停止”录制就是一个固定动作 trial，对应唯一 `session_id`，保持 {action['hold_seconds_per_recording']} 秒，阶段固定为 `hold`。
- 每人每动作合计：{action['recordings_per_subject_total']} 次录制、{action['hold_seconds_per_subject_total']} 秒；三动作共 {plan['recordings_per_subject_total']} 次录制。
- 只接收 `canonical_session` 且质量门禁通过的数据；数据集按受试者隔离切分。

命名：`experiment_batch=exp-YYYYMMDD-protocol-v1`；visit 使用 `visit-01`～`visit-03`；`session_id={{subject_id}}_{{visit_id}}_{{action_label}}_trial-{{trial_number:02d}}`。

录制间休息至少10秒，动作块之间至少60秒。漏采或质量拒绝的 trial 不覆盖、不改标签，保留拒绝证据，并用 `_retry-01` 新建录制，直到该 visit/动作累计10条通过质量门禁的录制。

## 为什么先这样定

三次会话用于覆盖重复佩戴差异；短时多组比一次长录制更能覆盖收缩变化。10 人是工程起点，不是最终统计结论。

## 何时修改数量

硬件恢复后，在 3/5/8/10 人与每动作 5/10/20/30 次录制处绘制学习曲线。最后两个检查点的 macro-F1 提升小于 0.02，且每类召回率均不低于 0.85，才冻结数量；否则优先增加受试者，而不是继续堆同一录制的重叠窗口。

## 当前不能完成的验证

- 真实采样率、BLE 丢包/重复/乱序；
- 电极佩戴重复性；
- 新训练采集程序的真实硬件验收。
"""
    (destination / "collection_plan.md").write_text(markdown, encoding="utf-8")
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--subjects", type=int, default=10)
    args = parser.parse_args(argv)
    write_collection_plan(args.output, subject_count=args.subjects)
    print(f"wrote provisional collection plan: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
