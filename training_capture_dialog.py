"""Controlled data-entry dialog for canonical training recordings."""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtWidgets

from data_recorder import validate_session_id
from emg_protocol import DeviceKey, HandSide
from training_contract import CANONICAL_LABELS


_ACTION_DISPLAY = {
    "rest": "静息（rest）",
    "fist": "握拳（fist）",
    "open_hand": "张手（open_hand）",
}
ACTION_OPTIONS = tuple(
    (_ACTION_DISPLAY.get(action, action), action) for action in CANONICAL_LABELS
)


class TrainingCaptureDialog(QtWidgets.QDialog):
    """Collect only fields permitted by the canonical training protocol."""

    def __init__(self, parent=None, *, device_id: Optional[DeviceKey] = None):
        super().__init__(parent)
        self._device_id = device_id
        self.setWindowTitle("录入训练采集信息")
        self.setModal(True)

        form = QtWidgets.QFormLayout()
        self.subject_identifier_edit = QtWidgets.QLineEdit()
        self.subject_identifier_edit.setObjectName("subject_identifier")
        self.session_id_edit = QtWidgets.QLineEdit()
        self.session_id_edit.setObjectName("session_id")
        self.device_id_edit = QtWidgets.QLineEdit(
            str(device_id) if device_id is not None else "未连接"
        )
        self.device_id_edit.setObjectName("device_id")
        self.device_id_edit.setReadOnly(True)
        self.hand_side_combo = QtWidgets.QComboBox()
        self.hand_side_combo.setObjectName("hand_side")
        self.hand_side_combo.addItem("左手", HandSide.LEFT)
        self.hand_side_combo.addItem("右手", HandSide.RIGHT)
        self.experiment_batch_edit = QtWidgets.QLineEdit()
        self.experiment_batch_edit.setObjectName("experiment_batch")
        self.action_combo = QtWidgets.QComboBox()
        self.action_combo.setObjectName("action_label")
        for display, value in ACTION_OPTIONS:
            self.action_combo.addItem(display, value)
        self.action_phase_edit = QtWidgets.QLineEdit("hold")
        self.action_phase_edit.setObjectName("action_phase")
        self.action_phase_edit.setReadOnly(True)
        self.countdown_spin = QtWidgets.QSpinBox()
        self.countdown_spin.setObjectName("countdown_seconds")
        self.countdown_spin.setRange(0, 60)
        self.countdown_spin.setValue(3)
        self.countdown_spin.setSuffix(" 秒")
        self.duration_spin = QtWidgets.QSpinBox()
        self.duration_spin.setObjectName("duration_seconds")
        self.duration_spin.setRange(1, 3600)
        self.duration_spin.setValue(30)
        self.duration_spin.setSuffix(" 秒")

        form.addRow("受试者编号", self.subject_identifier_edit)
        form.addRow("会话编号", self.session_id_edit)
        form.addRow("设备编号", self.device_id_edit)
        form.addRow("佩戴侧", self.hand_side_combo)
        form.addRow("实验批次", self.experiment_batch_edit)
        form.addRow("固定动作", self.action_combo)
        form.addRow("动作阶段", self.action_phase_edit)
        form.addRow("开始倒计时", self.countdown_spin)
        form.addRow("采集时长", self.duration_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _accept_if_valid(self) -> None:
        missing = []
        for label, widget in (
            ("受试者编号", self.subject_identifier_edit),
            ("会话编号", self.session_id_edit),
            ("实验批次", self.experiment_batch_edit),
        ):
            if not widget.text().strip():
                missing.append(label)
        if self._device_id is None:
            missing.append("已连接设备")
        if missing:
            QtWidgets.QMessageBox.warning(
                self, "信息不完整", "请补充：" + "、".join(missing)
            )
            return
        try:
            validate_session_id(self.session_id_edit.text().strip())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "会话编号无效", str(exc))
            return
        self.accept()

    def getuserInfo(self) -> dict:
        return {
            "subject_identifier": self.subject_identifier_edit.text().strip(),
            "session_id": self.session_id_edit.text().strip(),
            "device_id": self._device_id,
            "side": self.hand_side_combo.currentData(),
            "experiment_batch": self.experiment_batch_edit.text().strip(),
            "action_label": self.action_combo.currentData(),
            "action_phase": "hold",
            "countdown_seconds": self.countdown_spin.value(),
            "duration_seconds": self.duration_spin.value(),
        }


__all__ = ["ACTION_OPTIONS", "TrainingCaptureDialog"]
