"""Recording-context input dialog with canonical internal values."""

from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QMessageBox

from UI_user_info import Ui_Dialog
from emg_protocol import HandSide
from recording_context import validate_experiment_id


ACTION_OPTIONS = (("静息", "rest"), ("握拳", "fist"), ("张手", "open_hand"))
DEFAULT_EXPERIMENT_ID = "discrete_hand_v1"


class user_ui_dialog(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.user_info_dict = {}
        self.side.clear()
        self.side.addItem("请选择", HandSide.UNKNOWN)
        self.side.addItem("左手", HandSide.LEFT)
        self.side.addItem("右手", HandSide.RIGHT)
        self.action.clear()
        self.action.addItem("请选择", "")
        for display, value in ACTION_OPTIONS:
            self.action.addItem(display, value)
        self.experiment_id.setText(DEFAULT_EXPERIMENT_ID)
        self.collect_time.setDateTime(QtCore.QDateTime.currentDateTime())

    def save_click(self):
        subject_identifier = self.user_name.text().strip()
        side = self.side.currentData()
        action_label = str(self.action.currentData() or "")
        experiment_id = self.experiment_id.text().strip()
        if not subject_identifier:
            QMessageBox.warning(self, "信息不完整", "请输入受试者标识。")
            return
        if side not in (HandSide.LEFT, HandSide.RIGHT):
            QMessageBox.warning(self, "信息不完整", "请选择左手或右手。")
            return
        if action_label not in {value for _, value in ACTION_OPTIONS}:
            QMessageBox.warning(self, "信息不完整", "请选择本次采集动作。")
            return
        try:
            validate_experiment_id(experiment_id)
        except ValueError:
            QMessageBox.warning(
                self,
                "信息不完整",
                "实验编号仅允许小写字母、数字、点、下划线和连字符，最长 64 位。",
            )
            return
        self.user_info_dict = {
            "subject_identifier": subject_identifier,
            "side": side,
            "action_label": action_label,
            "action_phase": "hold",
            "experiment_id": experiment_id,
        }
        self.accept()

    def getuserInfo(self):
        return dict(self.user_info_dict)
