# -*- coding: utf-8 -*-
"""Independent, controlled GUI for canonical sEMG training capture."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, cast

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from qasync import QEventLoop

import qt5_bleak
from acquisition_pipeline import PipelineEventType
from app_logging import EventCode, configure_logging
from data_recorder import DataRecorder, RecorderQualitySnapshot
from emg_protocol import DeviceKey, HandSide
from qt5_bleak import RecordingState
from recording_context import RecordingContext
from shared_memory_v2 import SharedMemoryWriter
from replay_emg_session import decision_filter_from_bundle, replay_session
from training_capture_dialog import TrainingCaptureDialog
from training_capture_support import (
    TrainingCaptureSettings,
    format_live_quality,
    format_stop_report,
)


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.ini"
TRAINING_SHARED_FILE_PATH = BASE_DIR / "emg_training_shared_data_v2.bin"
REPLAY_SHARED_FILE_PATH = BASE_DIR / "emg_replay_shared_data_v2.bin"


def _add_exception_note(error: BaseException, note: str) -> None:
    """Use Python 3.11 exception notes when available; remain compatible with 3.10."""

    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)
WINDOW_TITLE = "肌电手环训练采集程序"
QUALITY_ERROR_LOG_INTERVAL_SECONDS = 10.0


class TrainingCapturePhase(str, Enum):
    IDLE = "IDLE"
    PREPARING = "PREPARING"
    COUNTDOWN = "COUNTDOWN"
    RECORDING = "RECORDING"
    STOPPING = "STOPPING"
    REPLAYING = "REPLAYING"
    REPLAY_STOPPING = "REPLAY_STOPPING"


class TrainingCaptureWindow(qt5_bleak.window):
    """Training-only UI layered over the existing BLE acquisition window."""

    replay_progress_signal = QtCore.pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        self._pending_training_settings: Optional[TrainingCaptureSettings] = None
        self._frozen_training_settings: Optional[TrainingCaptureSettings] = None
        self._frozen_recording_context: Optional[RecordingContext] = None
        self._training_recorder = None
        self._timed_capture_task = None
        self._capture_started_monotonic = None
        self._capture_cancel_requested = False
        self._inherited_start_in_progress = False
        self.training_phase = TrainingCapturePhase.IDLE
        self._last_quality_error_log_monotonic = float("-inf")
        self._replay_task = None
        self._replay_worker_future = None
        self._replay_cancel_event = None
        self._replay_shared_file = REPLAY_SHARED_FILE_PATH
        self._last_replay_report = None
        created_writer = None
        if kwargs.get("shared_writer") is None:
            created_writer = SharedMemoryWriter(
                TRAINING_SHARED_FILE_PATH, flush_interval_frames=100
            )
            kwargs["shared_writer"] = created_writer
        try:
            super().__init__(*args, **kwargs)
        except BaseException as exc:
            if created_writer is not None:
                try:
                    created_writer.close()
                except BaseException as cleanup_error:
                    _add_exception_note(
                        exc,
                        "training shared writer cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}",
                    )
            raise
        self.setWindowTitle(WINDOW_TITLE)
        self.user_info.setText("录入训练采集信息")
        self.start_recording.setText("开始定时采集")
        self.stop_recording.setText("停止采集")
        self._install_training_panel()
        self.replay_progress_signal.connect(self._update_replay_progress)
        self._quality_timer = QtCore.QTimer(self)
        self._quality_timer.setInterval(500)
        self._quality_timer.timeout.connect(self._refresh_live_quality)
        self._quality_timer.start()

    def _install_training_panel(self) -> None:
        group = QtWidgets.QGroupBox("实时训练质量证据", self.centralwidget)
        group.setObjectName("training_quality_panel")
        layout = QtWidgets.QFormLayout(group)
        self.live_quality_labels = {}
        for key, title in (
            ("dropped", "上位机队列丢弃"),
            ("sequence_gap", "设备序列缺口"),
            ("duplicate", "设备序列重复"),
            ("out_of_order", "设备序列乱序"),
            ("stale", "看门狗过期"),
            ("connection_generation", "连接代次"),
            ("effective_sample_rate", "有效采样率证据"),
        ):
            label = QtWidgets.QLabel("不可检测")
            label.setObjectName(f"quality_{key}")
            label.setWordWrap(True)
            self.live_quality_labels[key] = label
            layout.addRow(title, label)
        self.capture_instruction_label = QtWidgets.QLabel("等待录入训练采集信息")
        self.capture_instruction_label.setObjectName("capture_instruction")
        self.capture_instruction_label.setWordWrap(True)
        layout.addRow("动作提示", self.capture_instruction_label)
        self.live_quality_summary = QtWidgets.QLabel("尚无实时质量证据")
        self.live_quality_summary.setObjectName("quality_evidence_summary")
        self.live_quality_summary.setWordWrap(True)
        layout.addRow("证据摘要", self.live_quality_summary)
        self.replay_progress = QtWidgets.QProgressBar(group)
        self.replay_progress.setRange(0, 100)
        self.replay_progress.setValue(0)
        layout.addRow("离线回放进度", self.replay_progress)
        replay_row = QtWidgets.QWidget(group)
        replay_layout = QtWidgets.QHBoxLayout(replay_row)
        replay_layout.setContentsMargins(0, 0, 0, 0)
        self.replay_start_button = QtWidgets.QPushButton("离线回放历史数据", replay_row)
        self.replay_cancel_button = QtWidgets.QPushButton("取消回放", replay_row)
        self.replay_cancel_button.setEnabled(False)
        self.replay_start_button.clicked.connect(self._start_replay_clicked)
        self.replay_cancel_button.clicked.connect(self._cancel_replay)
        replay_layout.addWidget(self.replay_start_button)
        replay_layout.addWidget(self.replay_cancel_button)
        layout.addRow("工程回放", replay_row)
        self.verticalLayout.insertWidget(4, group)

    def _connected_device_key(self) -> Optional[DeviceKey]:
        profile = getattr(self, "WG", None)
        generation = getattr(profile, "client_generation", None)
        if (
            profile is None
            or not getattr(profile, "is_connected", False)
            or generation is None
            or generation != getattr(self, "_accepted_connection_generation", None)
            or not isinstance(getattr(self, "device_key", None), DeviceKey)
        ):
            return None
        return self.device_key

    def _create_user_info_dialog(self):
        return TrainingCaptureDialog(self, device_id=self._connected_device_key())

    def _build_recording_context(self, subject_id, user_info):
        device_id = user_info.get("device_id")
        if device_id is None or device_id != self._connected_device_key():
            raise ValueError("device_id must be the currently connected device")
        settings = TrainingCaptureSettings(
            subject_id=subject_id,
            session_id=str(user_info.get("session_id", "")),
            device_id=device_id,
            hand_side=user_info.get("side", HandSide.UNKNOWN),
            action_label=str(user_info.get("action_label", "")),
            experiment_batch=str(user_info.get("experiment_batch", "")),
            countdown_seconds=user_info.get("countdown_seconds"),
            duration_seconds=user_info.get("duration_seconds"),
        )
        self._pending_training_settings = settings
        return settings.to_recording_context()

    def _after_recording_context_registered(self, context, user_info):
        settings = self._pending_training_settings
        if settings is None or settings.to_recording_context() != context:
            raise RuntimeError("training capture settings were not registered")
        self.capture_instruction_label.setText(
            f"已就绪：{settings.action_label} / hold，"
            f"倒计时 {settings.countdown_seconds} 秒，采集 {settings.duration_seconds} 秒"
        )

    def _create_data_recorder(self, context):
        settings = self._frozen_training_settings
        if (
            settings is None
            or self._frozen_recording_context is None
            or self._frozen_recording_context != context
        ):
            raise RuntimeError("frozen training settings do not match the context")
        self._assert_frozen_capture_unchanged()
        actual_device = self._connected_device_key()
        if actual_device is None or actual_device != settings.device_id:
            raise RuntimeError("connected device changed after training information entry")
        recorder = DataRecorder(
            self.config.data.data_path,
            subject_id=context.subject_id,
            device_id=actual_device,
            acquisition=self.config.to_acquisition_metadata(),
            channels=self.config.data.channels,
            session_id=settings.session_id,
            side=context.hand_side,
            recording_context=context,
        )
        self._training_recorder = recorder
        return recorder

    def _set_training_phase(self, phase: TrainingCapturePhase) -> None:
        self.training_phase = phase
        editing = phase is TrainingCapturePhase.IDLE
        qt5_bleak.window._set_context_editing_enabled(self, editing)
        if getattr(self, "start_recording", None) is not None:
            self.start_recording.setEnabled(
                editing and getattr(self, "_controls_enabled", True)
            )
        if getattr(self, "replay_start_button", None) is not None:
            self.replay_start_button.setEnabled(editing)
            self.replay_cancel_button.setEnabled(
                phase in {
                    TrainingCapturePhase.REPLAYING,
                    TrainingCapturePhase.REPLAY_STOPPING,
                }
            )

    def _update_replay_progress(self, completed: int, total: int) -> None:
        percent = 0 if total <= 0 else min(100, round(completed * 100 / total))
        self.replay_progress.setValue(percent)
        self.capture_instruction_label.setText(
            f"离线工程回放：{completed}/{total}；禁止用于训练或真实控制"
        )

    def _start_replay_clicked(self) -> None:
        source, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择历史 derived_samples.csv", str(BASE_DIR), "CSV (*.csv)"
        )
        if not source:
            return
        bundle_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择工程模型目录中的 model.json", str(BASE_DIR), "JSON (*.json)"
        )
        if not bundle_file:
            return
        try:
            decision_filter = decision_filter_from_bundle(
                Path(bundle_file).resolve().parent, hand_side=HandSide.LEFT
            )
            rate = decision_filter.classifier.acquisition.sample_rate.value_hz
            assert rate is not None
            self._begin_replay(Path(source), rate, decision_filter)
        except Exception as exc:
            QMessageBox.critical(self, "离线回放失败", f"{type(exc).__name__}: {exc}")

    def _begin_replay(
        self,
        source_csv,
        assumed_rate_hz,
        decision_filter=None,
        *,
        shared_file=REPLAY_SHARED_FILE_PATH,
    ) -> None:
        if (
            self.training_phase is not TrainingCapturePhase.IDLE
            or self._replay_task is not None
            or self.recording_state is not RecordingState.IDLE
        ):
            raise RuntimeError("capture or replay is already active")
        self._replay_cancel_event = threading.Event()
        self._replay_shared_file = Path(shared_file).expanduser().resolve()
        self._last_replay_report = None
        self.replay_progress.setValue(0)
        self._set_training_phase(TrainingCapturePhase.REPLAYING)
        task = self.loop.create_task(
            self._run_replay(Path(source_csv), float(assumed_rate_hz), decision_filter)
        )
        self._replay_task = task
        self._ble_ui_tasks.add(task)
        task.add_done_callback(self._replay_finished)

    async def _run_replay(self, source_csv, assumed_rate_hz, decision_filter):
        cancel_event = self._replay_cancel_event
        if cancel_event is None:
            raise RuntimeError("replay cancellation state is missing")
        worker = asyncio.create_task(
            asyncio.to_thread(
                replay_session,
                source_csv,
                self._replay_shared_file,
                assumed_sample_rate_hz=assumed_rate_hz,
                decision_filter=decision_filter,
                realtime=True,
                cancel_event=cancel_event,
                progress_callback=self.replay_progress_signal.emit,
            )
        )
        self._replay_worker_future = worker
        cancellation_requested = False
        try:
            while True:
                try:
                    result = await asyncio.shield(worker)
                except asyncio.CancelledError:
                    cancellation_requested = True
                    cancel_event.set()
                    # Keep shielding until the executor future is truly done.
                    # Shutdown may cancel this wrapper more than once.
                    continue
                if cancellation_requested:
                    raise asyncio.CancelledError
                return result
        finally:
            self._replay_worker_future = None

    def _replay_finished(self, task) -> None:
        self._ble_ui_tasks.discard(task)
        if self._replay_task is task:
            self._replay_task = None
        try:
            report = None if task.cancelled() else task.result()
        except Exception as exc:
            message = f"离线回放异常：{type(exc).__name__}: {exc}"
            self.capture_instruction_label.setText(message)
            if self._controls_enabled:
                QMessageBox.critical(self, "离线回放失败", message)
        else:
            self._last_replay_report = report
            if report is not None:
                message = (
                    f"离线回放{report.status}：{report.frames_written}/{report.source_rows}，"
                    f"预测 {report.predictions_emitted}；训练资格：否；"
                    f"错误：{'；'.join(report.errors) if report.errors else '无'}"
                )
                self.capture_instruction_label.setText(message)
                self.live_quality_summary.setText(message)
        finally:
            self._replay_cancel_event = None
            self._set_training_phase(TrainingCapturePhase.IDLE)

    def _cancel_replay(self) -> None:
        event = self._replay_cancel_event
        if event is not None:
            self._set_training_phase(TrainingCapturePhase.REPLAY_STOPPING)
            event.set()

    def _finish_training_cycle(self) -> None:
        self._training_recorder = None
        self._capture_started_monotonic = None
        self._frozen_training_settings = None
        self._frozen_recording_context = None
        self._capture_cancel_requested = False
        self._set_training_phase(TrainingCapturePhase.IDLE)

    def _assert_frozen_capture_unchanged(self) -> None:
        settings = self._frozen_training_settings
        context = self._frozen_recording_context
        if settings is None or context is None:
            raise RuntimeError("training capture was not frozen before start")
        if self._pending_training_settings != settings:
            raise RuntimeError("training settings changed during countdown")
        if self._pending_recording_context != context:
            raise RuntimeError("recording context changed during countdown")
        if settings.to_recording_context() != context:
            raise RuntimeError("frozen settings and context do not match")
        if self._connected_device_key() != settings.device_id:
            raise RuntimeError("connected device changed during countdown")

    def _start_recording_clicked(self):
        if (
            not self._controls_enabled
            or self.training_phase is not TrainingCapturePhase.IDLE
            or self._timed_capture_task is not None
            or self.recording_state is not RecordingState.IDLE
        ):
            return
        settings = self._pending_training_settings
        context = self._pending_recording_context
        if not isinstance(settings, TrainingCaptureSettings) or not isinstance(
            context, RecordingContext
        ):
            QMessageBox.warning(self, "无法开始", "请先录入训练采集信息。")
            return
        if settings.to_recording_context() != context:
            QMessageBox.warning(self, "无法开始", "训练设置与采集上下文不一致。")
            return
        if settings.device_id != self._connected_device_key():
            QMessageBox.warning(self, "无法开始", "当前连接设备与录入设备不一致。")
            return
        self._training_recorder = None
        self._capture_started_monotonic = None
        self._frozen_training_settings = settings
        self._frozen_recording_context = context
        self._capture_cancel_requested = False
        self._set_training_phase(TrainingCapturePhase.PREPARING)
        task = asyncio.ensure_future(self._run_timed_capture())
        self._timed_capture_task = task
        self._ble_ui_tasks.add(task)
        task.add_done_callback(self._timed_capture_finished)

    def _timed_capture_finished(self, task) -> None:
        self._ble_ui_tasks.discard(task)
        if self._timed_capture_task is task:
            self._timed_capture_task = None
        if task.cancelled():
            if self.recording_state is RecordingState.IDLE:
                self._finish_training_cycle()
            return
        try:
            task.result()
        except Exception as exc:
            if self.recording_state is RecordingState.IDLE:
                self._finish_training_cycle()
            if self._controls_enabled:
                QMessageBox.critical(
                    self,
                    "训练采集失败",
                    f"定时采集失败：{type(exc).__name__}: {exc}",
                )

    async def _wait_capture_second(self) -> None:
        await asyncio.sleep(1)

    async def _run_timed_capture(self) -> None:
        settings = self._frozen_training_settings
        if settings is None:
            raise RuntimeError("training settings were not frozen")
        self._set_training_phase(TrainingCapturePhase.COUNTDOWN)
        try:
            for remaining in range(settings.countdown_seconds, 0, -1):
                if self._capture_cancel_requested:
                    return
                self.capture_instruction_label.setText(
                    f"准备：{remaining} 秒后开始 {settings.action_label}（hold）"
                )
                await self._wait_capture_second()
            if self._capture_cancel_requested:
                return
            self._assert_frozen_capture_unchanged()
            self._set_training_phase(TrainingCapturePhase.PREPARING)
            self.capture_instruction_label.setText(
                f"动作开始：{settings.action_label}（hold）"
            )
            self._inherited_start_in_progress = True
            try:
                await super()._start_recording()
            finally:
                self._inherited_start_in_progress = False
            if self.recording_state is not RecordingState.RECORDING:
                return
            if self._capture_cancel_requested:
                self._set_training_phase(TrainingCapturePhase.STOPPING)
                await self._stop_recording()
                return
            self._set_training_phase(TrainingCapturePhase.RECORDING)
            self._capture_started_monotonic = time.monotonic()
            for remaining in range(settings.duration_seconds, 0, -1):
                if self._capture_cancel_requested:
                    return
                self.capture_instruction_label.setText(
                    f"保持 {settings.action_label}，剩余 {remaining} 秒"
                )
                await self._wait_capture_second()
            self.capture_instruction_label.setText(
                "动作结束，正在自动停止并检查训练资格……"
            )
            self._set_training_phase(TrainingCapturePhase.STOPPING)
            await self._stop_recording()
        finally:
            if (
                self.recording_state is RecordingState.IDLE
                and not self._inherited_start_in_progress
            ):
                self._finish_training_cycle()

    def _cancel_timed_capture(self) -> None:
        self._capture_cancel_requested = True
        task = self._timed_capture_task
        if (
            task is not None
            and not task.done()
            and not self._inherited_start_in_progress
        ):
            task.cancel()

    def _stop_recording_clicked(self):
        prior_phase = self.training_phase
        self._cancel_timed_capture()
        if prior_phase is not TrainingCapturePhase.IDLE:
            self._set_training_phase(TrainingCapturePhase.STOPPING)
        if (
            prior_phase in {
                TrainingCapturePhase.PREPARING,
                TrainingCapturePhase.COUNTDOWN,
            }
            and not self._inherited_start_in_progress
            and self.recording_state is RecordingState.IDLE
        ):
            self._finish_training_cycle()
            return
        super()._stop_recording_clicked()

    async def _stop_recording(self):
        await super()._stop_recording()
        if self.recording_state is RecordingState.IDLE:
            self._finish_training_cycle()

    def _on_ble_state(self, event) -> None:
        if (
            self._controls_enabled
            and self.training_phase not in {
                TrainingCapturePhase.REPLAYING,
                TrainingCapturePhase.REPLAY_STOPPING,
            }
            and getattr(event, "event_type", None) == "disconnected"
            and getattr(event, "generation", None)
            == getattr(self, "_accepted_connection_generation", None)
        ):
            self._cancel_timed_capture()
            self._set_training_phase(TrainingCapturePhase.STOPPING)
            self.capture_instruction_label.setText("设备已断开，定时采集已取消")
        super()._on_ble_state(event)

    def _handle_pipeline_event(self, event) -> None:
        if getattr(event, "event_type", None) is PipelineEventType.RECORDING_FAULT:
            self._cancel_timed_capture()
            self._set_training_phase(TrainingCapturePhase.STOPPING)
        super()._handle_pipeline_event(event)
        self._reconcile_stopping_phase()

    def closeEvent(self, event):
        self._cancel_replay()
        self._cancel_timed_capture()
        if self.training_phase is not TrainingCapturePhase.IDLE:
            self._set_training_phase(TrainingCapturePhase.STOPPING)
        timer = getattr(self, "_quality_timer", None)
        if timer is not None:
            timer.stop()
        super().closeEvent(event)

    def _live_quality_values(self):
        if self.training_phase in {
            TrainingCapturePhase.REPLAYING,
            TrainingCapturePhase.REPLAY_STOPPING,
        }:
            values = {
                "dropped": "回放不适用",
                "sequence_gap": "回放不适用",
                "duplicate": "回放不适用",
                "out_of_order": "回放不适用",
                "stale": "回放不适用",
                "connection_generation": "独立回放流",
                "effective_sample_rate": "合成时间轴（非设备证据）",
            }
            return values, "历史工程回放：非 BLE、非 canonical、禁止训练及真实控制"
        recorder = (
            self._training_recorder
            if self.training_phase is TrainingCapturePhase.RECORDING
            else None
        )
        recorder_snapshot: Optional[RecorderQualitySnapshot] = None
        if recorder is not None:
            snapshot = getattr(recorder, "quality_snapshot", None)
            if callable(snapshot):
                recorder_snapshot = cast(RecorderQualitySnapshot, snapshot())

        pipeline_quality = self.pipeline.quality_snapshot()
        snapshot_ns = pipeline_quality.snapshot_monotonic_ns
        published_ns = pipeline_quality.last_published_monotonic_ns
        freshness_seconds = (
            None
            if published_ns is None
            else max(0.0, (snapshot_ns - published_ns) / 1_000_000_000)
        )
        pipeline_snapshot = {
            "dropped_count": pipeline_quality.host_queue_dropped_count,
            "queue_depth": pipeline_quality.queue_depth,
            "freshness_seconds": freshness_seconds,
        }
        elapsed_seconds = (
            None
            if self.training_phase is not TrainingCapturePhase.RECORDING
            or self._capture_started_monotonic is None
            else max(0.0, time.monotonic() - self._capture_started_monotonic)
        )

        host_rate = None
        if recorder_snapshot is not None and elapsed_seconds:
            host_rate = recorder_snapshot.recorded_rows / elapsed_seconds
        formatter_snapshot: RecorderQualitySnapshot | Mapping[str, object] = (
            recorder_snapshot
            if recorder_snapshot is not None
            else {
            "sequence_detection_available": False,
            "duplicate_count": 0,
            "out_of_order_count": 0,
            "gap_count": 0,
            "connection_generation": pipeline_quality.connection_generation,
            "recorded_rows": 0,
            }
        )
        summary = format_live_quality(
            pipeline_snapshot=pipeline_snapshot,
            recorder_snapshot=formatter_snapshot,
            sample_rate=self.config.to_acquisition_metadata().sample_rate,
            host_observed_rate_hz=host_rate,
        )
        generation = (
            recorder_snapshot.connection_generation
            if recorder_snapshot is not None
            else pipeline_quality.connection_generation
        )
        watchdog_stale = pipeline_quality.watchdog_stale
        stream_expected = pipeline_quality.stream_expected
        values = {
            "dropped": pipeline_snapshot["dropped_count"],
            "sequence_gap": "不可检测",
            "duplicate": "不可检测",
            "out_of_order": "不可检测",
            "stale": (
                "不可检测"
                if stream_expected is not True or watchdog_stale is None
                else ("是" if watchdog_stale else "否")
            ),
            "connection_generation": (
                "不可检测" if generation is None else generation
            ),
            "effective_sample_rate": (
                "不可检测"
                if host_rate is None
                else f"{host_rate:.2f} Hz（主机观测值）"
            ),
        }
        if recorder_snapshot is not None and recorder_snapshot.sequence_detection_available:
            values["sequence_gap"] = recorder_snapshot.gap_count
            values["duplicate"] = recorder_snapshot.duplicate_count
            values["out_of_order"] = recorder_snapshot.out_of_order_count
        return values, summary

    def _refresh_live_quality(self) -> None:
        self._reconcile_stopping_phase()
        try:
            values, summary = self._live_quality_values()
        except Exception as exc:
            values = {key: "不可检测" for key in self.live_quality_labels}
            summary = "实时质量证据不可用"
            now = time.monotonic()
            if (
                now - self._last_quality_error_log_monotonic
                >= QUALITY_ERROR_LOG_INTERVAL_SECONDS
            ):
                self._last_quality_error_log_monotonic = now
                self._log(
                    EventCode.APP_EXCEPTION,
                    "training live quality read failed",
                    level="error",
                    exc_info=(type(exc), exc, exc.__traceback__),
                    error_count=1,
                )
        for key, label in self.live_quality_labels.items():
            value = values.get(key, "不可检测")
            label.setText(str(value) if value not in (None, "") else "不可检测")
        self.live_quality_summary.setText(summary)

    def _reconcile_stopping_phase(self) -> None:
        task = self._timed_capture_task
        if (
            self.training_phase is TrainingCapturePhase.STOPPING
            and self.recording_state is RecordingState.IDLE
            and (task is None or task.done())
            and getattr(self, "recording_stop_pending", None) is None
            and not getattr(self.pipeline, "recorder_cleanup_pending", False)
        ):
            self._finish_training_cycle()

    def _format_quality_report_message(self, report, report_path):
        return format_stop_report(report)

    def _after_quality_report(self, report, report_path):
        message = format_stop_report(report)
        self._last_training_quality_report = (report, report_path, message)
        self.quality_status_label.setText(message)
        if not self._controls_enabled:
            return
        if report.get("training_usable") is True:
            QMessageBox.information(self, "训练资格检查", message)
        else:
            QMessageBox.warning(self, "训练资格检查", message)


def _configure_training_runtime(config):
    return configure_logging(
        config.logging.path,
        level=config.logging.level,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
        queue_capacity=config.logging.queue_capacity,
        logger_name="emg_training_capture",
        file_name="training_capture.log",
    )


def main() -> int:
    config = qt5_bleak.load_config(CONFIG_PATH)
    runtime = _configure_training_runtime(config)
    app = QtWidgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    form = TrainingCaptureWindow(loop, config=config, logging_runtime=runtime)
    form.show()
    qt5_bleak.install_shutdown_hooks(app, form, runtime)
    try:
        with loop:
            qt5_bleak._run_application_loop(loop, form)
    except qt5_bleak.ShutdownError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REPLAY_SHARED_FILE_PATH",
    "TrainingCapturePhase",
    "TrainingCaptureWindow",
    "WINDOW_TITLE",
    "main",
]
