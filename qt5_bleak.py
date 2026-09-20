# -*- coding: utf-8 -*-
"""PyQt5/qasync desktop application for trustworthy EMG acquisition."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import os
import stat
import sys
import threading
import time
import traceback as traceback_module
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

import pandas as pd
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox
from qasync import QEventLoop, asyncSlot

import bleak_ble
from acquisition_pipeline import (
    AcquisitionPipeline,
    PipelineEvent,
    PipelineEventType,
    RawAuditPolicy,
)
from app_config import AppConfig, load_config
from app_logging import EventCode, LoggingRuntime, configure_logging
from data_recorder import DataRecorder, recover_incomplete_sessions
from device_identity import DeviceIdentityStore
from emg_protocol import DeviceKey, EmgFrame, HandSide
from on_time_show_dialog import MultiChannelShowDialog, OnTimeShowDialog
from Qt5_MainWindow import Ui_MainWindow
from shared_memory_v2 import FLAG_DISCONNECTED, FLAG_STALE, SharedMemoryWriter
from recording_context import RecordingContext
from session_quality import analyze_session
from subject_identity import SubjectIdentityStore
from user_info import user_ui_dialog


os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_PLUGIN_PATH", None)
if getattr(sys, "frozen", False):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
        sys._MEIPASS, "PyQt5", "Qt5", "plugins", "platforms"
    )

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.ini"
SHARED_FILE_PATH = BASE_DIR / "emg_shared_data_v2.bin"
BLE_UI_TASK_WAIT_TIMEOUT_SECONDS = 2.0
RECORDING_STOP_BLE_TIMEOUT_SECONDS = 2.0
RECORDING_STOP_RECORDER_TIMEOUT_SECONDS = 5.0
RECORDING_STOP_WORK_WAIT_SECONDS = 5.2
ASYNC_TASK_DRAIN_MAX_SECONDS = 0.1
WINDOW_SHUTDOWN_TIMEOUT_SECONDS = 5.0
SHUTDOWN_STAGE_MAX_SECONDS = 0.4
_SHUTDOWN_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="emg-shutdown"
)
SUBJECT_DATASET_DOMAIN = "dexterous_hand_v1"


def _subject_identity_directory() -> Path:
    """Return a stable per-user location outside the source/release directory."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        base = Path(local_app_data).expanduser()
    else:
        writable = QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.AppLocalDataLocation
        )
        if not writable:
            raise RuntimeError("no per-user local application data directory is available")
        base = Path(writable).expanduser()
    if not base.is_absolute():
        raise RuntimeError("per-user local application data directory must be absolute")
    return base.resolve() / "EMGBracelet" / "subject_identity"


def _entry_address(entry) -> Optional[str]:
    try:
        address = entry[2]
    except (IndexError, KeyError, TypeError):
        return None
    if address is None:
        return None
    normalized = str(address).strip()
    return normalized or None


def _resolve_matched_entry(matched, devices):
    """Locate the scan-confirmed target without relying on display order/name."""
    if matched is None:
        return None, None

    for position, entry in enumerate(devices):
        if entry is matched:
            return position, entry

    for position, entry in enumerate(devices):
        if entry == matched:
            return position, entry

    matched_address = _entry_address(matched)
    if matched_address is not None:
        normalized_address = matched_address.casefold()
        for position, entry in enumerate(devices):
            entry_address = _entry_address(entry)
            if entry_address is not None and entry_address.casefold() == normalized_address:
                return position, entry
    return None, None


def _resolve_scan_matches(matched, devices):
    """Normalize legacy single-match stubs and canonicalize all scan matches."""
    if matched is None:
        candidates = []
    elif isinstance(matched, (list, tuple)):
        if not matched:
            candidates = []
        elif isinstance(matched[0], (list, tuple)):
            candidates = list(matched)
        else:
            candidates = [matched]
    else:
        candidates = [matched]

    resolved = []
    seen_addresses = set()
    seen_objects = set()
    for candidate in candidates:
        _, entry = _resolve_matched_entry(candidate, devices)
        if entry is None:
            continue
        address = _entry_address(entry)
        if address is not None:
            identity = address.casefold()
            if identity in seen_addresses:
                continue
            seen_addresses.add(identity)
        else:
            identity = id(entry)
            if identity in seen_objects:
                continue
            seen_objects.add(identity)
        resolved.append(entry)
    return tuple(resolved)


def _coerce_scan_snapshot(result) -> bleak_ble.ScanSnapshot:
    """Isolate the legacy ``(matched, entries)`` test/profile adapter."""
    if isinstance(result, bleak_ble.ScanSnapshot):
        return result
    matched, entries = result
    resolved_matches = _resolve_scan_matches(matched, entries)
    matched_positions = {
        position
        for resolved in resolved_matches
        for position, entry in enumerate(entries)
        if entry is resolved
    }
    devices = []
    for position, entry in enumerate(entries):
        address = _entry_address(entry)
        devices.append(
            bleak_ble.DiscoveredDevice(
                candidate_id=f"legacy-candidate-{position + 1}",
                index=int(entry[0]),
                name=entry[1],
                address=address,
                rssi=entry[3],
                matches_service=position in matched_positions,
                native_device=address,
            )
        )
    matches = tuple(device for device in devices if device.matches_service)
    return bleak_ble.ScanSnapshot(tuple(devices), matches)


def _raw_audit_policy(config: AppConfig) -> RawAuditPolicy:
    audit = config.raw_audit
    return RawAuditPolicy(
        enabled=audit.enabled,
        max_bytes=audit.max_bytes,
        backup_count=audit.backup_count,
        payload_prefix_bytes=audit.payload_prefix_bytes,
        record_valid=audit.record_valid,
        record_invalid=audit.record_invalid,
    )


class RecordingState(str, Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    RECORDING = "RECORDING"
    STOPPING = "STOPPING"


class ShutdownError(RuntimeError):
    def __init__(self, failures):
        self.failures = tuple(failures)
        super().__init__(
            "; ".join(f"{stage}: {type(error).__name__}: {error}" for stage, error in failures)
        )


class _BlockingShutdownWork:
    def __init__(self, stage, operation):
        self.stage = stage
        self.operation = operation
        self.completed = threading.Event()
        self.error = None
        self.result = None
        self.future = None

    def run(self):
        try:
            self.result = self.operation()
            return self.result
        except BaseException as exc:
            self.error = exc
            raise
        finally:
            self.completed.set()


class window(QMainWindow, Ui_MainWindow):
    """Main window; every BLE coroutine is owned by the qasync event loop."""

    wg_show_signal = QtCore.pyqtSignal(list)
    pipeline_event_signal = QtCore.pyqtSignal(object)

    @staticmethod
    def _call_extension_hook(instance, name, *args):
        """Call a bound override, with a fallback for lightweight test doubles."""
        hook = getattr(instance, name, None)
        if callable(hook):
            return hook(*args)
        return getattr(window, name)(instance, *args)

    def _create_user_info_dialog(self):
        return user_ui_dialog(self)

    def _build_recording_context(self, subject_id, user_info):
        return RecordingContext(
            subject_id=subject_id,
            hand_side=user_info.get("side", HandSide.UNKNOWN),
            action_label=str(user_info.get("action_label", "")),
            action_phase=str(user_info.get("action_phase", "")),
            experiment_id=str(user_info.get("experiment_id", "")),
        )

    def _after_recording_context_registered(self, context, user_info):
        pass

    def _create_data_recorder(self, context):
        return DataRecorder(
            self.config.data.data_path,
            subject_id=context.subject_id,
            device_id=self.device_key,
            acquisition=self.config.to_acquisition_metadata(),
            channels=self.config.data.channels,
            side=context.hand_side,
            recording_context=context,
        )

    def _format_quality_report_message(self, report, report_path):
        status = report.get("overall_status", "fail")
        usable = report.get("training_usable") is True
        if status == "pass" and usable:
            return f"质量检查：通过（可训练） | {report_path}"
        if status == "warning":
            return f"质量检查：警告（不可训练） | {report_path}"
        return f"质量检查：失败（不可训练） | {report_path}"

    def _after_quality_report(self, report, report_path):
        pass

    def __init__(
        self,
        loop=None,
        *,
        config: Optional[AppConfig] = None,
        logging_runtime: Optional[LoggingRuntime] = None,
        profile=None,
        shared_writer=None,
        pipeline=None,
        identity_store=None,
        subject_identity_store=None,
    ):
        super().__init__()
        self.setupUi(self)
        self.loop = loop or asyncio.get_event_loop()
        self.config = config or load_config(CONFIG_PATH)
        self.logging_runtime = logging_runtime or configure_logging(
            self.config.logging.path,
            level=self.config.logging.level,
            max_bytes=self.config.logging.max_bytes,
            backup_count=self.config.logging.backup_count,
            queue_capacity=self.config.logging.queue_capacity,
        )
        self.device_key = self.config.ble.device_key
        self.identity_store = identity_store or DeviceIdentityStore(
            identity_path=self.config.source_path.parent / ".emg_identity",
            data_root=self.config.data.data_path,
        )
        self.subject_identity_store = subject_identity_store or SubjectIdentityStore(
            _subject_identity_directory(),
            dataset_domain=SUBJECT_DATASET_DOMAIN,
        )
        self.subject_key: Optional[str] = None
        self.temp_info = {"side": HandSide.UNKNOWN}
        self._pending_recording_context: Optional[RecordingContext] = None
        self._active_recording_context: Optional[RecordingContext] = None
        self._active_session_dir: Optional[Path] = None
        self._quality_tasks = set()
        self._scan_snapshot = None
        self._scan_epoch = 0
        self._completed_scan_epoch = None
        self._scan_in_progress = False
        self._connect_epoch = 0
        self._connect_in_progress = False
        self._accepted_connection_generation = None
        self._ble_ui_tasks = set()
        self._controls_enabled = True
        self._shutdown_task = None
        self._shutdown_complete = False
        self._shutdown_stages = set()
        self._shutdown_blocking_work = {}
        self._last_shutdown_error = None
        self._allow_close = False
        self._quit_reason = None
        self._recording_start_token = None
        self.is_recording = False
        self.recording_state = RecordingState.IDLE
        self.recording_stop_pending = None
        self._recorder_stop_token = None
        self._recorder_stop_work = None
        self._unattached_recorder = None
        self._unattached_cleanup_intent = None
        self._unattached_cleanup_work = None
        self._pipeline_cleanup_tasks = set()
        self._pipeline_resume_task = None
        self._pipeline_resume_generation = None
        self._recording_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self.current_session_id: Optional[str] = None

        recovery = recover_incomplete_sessions(self.config.data.data_path)
        if recovery.issues:
            self._log(
                EventCode.APP_EXCEPTION,
                "incomplete session recovery reported issues",
                level="warning",
                error_count=len(recovery.issues),
            )
        else:
            self._log(EventCode.APP_OK, "session recovery completed")

        self.WG = profile or bleak_ble.BleakProfile(
            bleak_ble.service_uuid_2,
            bleak_ble.service_uuid_2,
            bleak_ble.characteristic_uuid_2,
            bleak_ble.cmd_uuid_2,
        )
        self.WG.add_state_listener(self._on_ble_state)
        self.shared_writer = shared_writer or SharedMemoryWriter(
            SHARED_FILE_PATH, flush_interval_frames=100
        )
        self.pipeline = pipeline or AcquisitionPipeline(
            self.shared_writer,
            channels=self.config.data.channels,
            queue_capacity=256,
            stale_after_seconds=1.0,
            display_callback=self._display_frame,
            event_callback=self.pipeline_event_signal.emit,
            frame_callback=self._frame_processed,
            shared_callback=self._shared_written,
            raw_audit_policy=_raw_audit_policy(self.config),
            packet_protocol=self.config.packet_protocol,
        )
        self.pipeline.start()
        self.WG.add_notification_listener(self.pipeline.notification_callback)

        self.show_window = OnTimeShowDialog(wg_repeat_time=2, wg_channel=8)
        self.multichannel_window = MultiChannelShowDialog(wg_repeat_time=2, wg_channel=8)
        self.wg_show_signal.connect(self.show_window.receive_waveguide_signal)
        self.wg_show_signal.connect(self.multichannel_window.receive_waveguide_signal)
        self.pipeline_event_signal.connect(self._handle_pipeline_event)
        self.recording_status = QtWidgets.QLabel("未采集")
        self.statusBar().addWidget(self.recording_status)
        # Do not use Qt's ``on_<object>_<signal>`` naming convention here.
        # setupUi() calls connectSlotsByName(), so convention-named handlers
        # would be connected once automatically and once explicitly below.
        self.start_recording.clicked.connect(self._start_recording_clicked)
        self.stop_recording.clicked.connect(self._stop_recording_clicked)
        self.start_playback = self.findChild(QtWidgets.QPushButton, "start_playback")
        if self.start_playback:
            self.start_playback.clicked.connect(self.on_start_playback)

    def _log(self, event: EventCode, message: str, *, level: str = "info", exc_info=None, **context) -> None:
        getattr(self.logging_runtime.get_logger(event=event, **context), level)(
            message, exc_info=exc_info
        )

    def _handle_pipeline_event(self, event: PipelineEvent) -> None:
        if event.event_type is PipelineEventType.RECORDING_FAULT:
            self.is_recording = False
            retrying_close = event.cleanup_pending or event.stage == "recorder_close"
            self.recording_state = (
                RecordingState.STOPPING if retrying_close else RecordingState.IDLE
            )
            self.current_session_id = None
            self.recording_status.setText(
                "停止失败（请重试）" if retrying_close else "未采集（写入失败）"
            )
            if not retrying_close:
                window._release_active_recording_context(
                    self, schedule_quality=True
                )
        error = event.error
        self._log(
            EventCode.APP_EXCEPTION,
            f"acquisition pipeline event at {event.stage}: "
            f"{type(error).__name__ if error is not None else event.event_type.value}",
            level="error" if error is not None else "warning",
            exc_info=(type(error), error, error.__traceback__) if error is not None else None,
            device_id=self.device_key,
            error_count=1 if error is not None else 0,
        )

    def _frame_processed(self, frame: EmgFrame) -> None:
        if self.is_recording and self.current_session_id and frame.sample_index % 500 == 0:
            self._log(
                EventCode.DATA_SAVE,
                "recording progress",
                device_id=self.device_key,
                session_id=self.current_session_id,
                packet_index=frame.host_receive_index,
                dropped_count=self.pipeline.dropped_count,
                queue_depth=self.pipeline.queue_depth,
            )

    def _shared_written(self, frame: EmgFrame) -> None:
        if frame.sample_index == 1 or frame.sample_index % 500 == 0:
            self._log(
                EventCode.SHARED_WRITE,
                "shared memory progress",
                device_id=self.device_key,
                packet_index=frame.host_receive_index,
                dropped_count=self.pipeline.dropped_count,
                queue_depth=self.pipeline.queue_depth,
            )

    def _display_frame(self, frame: EmgFrame) -> None:
        self.wg_show_signal.emit(list(frame.channel_values))

    def _on_ble_state(self, event) -> None:
        if not self._controls_enabled:
            return
        if event.event_type == "connected":
            if event.generation != self._accepted_connection_generation:
                return
            window._resume_pipeline_after_prior_cleanup(self, event.generation)
            self._log(EventCode.BLE_CONNECT, "BLE connected", device_id=self.device_key)
            self.statusBar().showMessage("设备已连接", 3000)
        elif event.event_type == "disconnected":
            if event.generation != self._accepted_connection_generation:
                return
            self._accepted_connection_generation = None
            self._recording_start_token = None
            self.is_recording = False
            pending = {
                "generation": event.generation,
                "stage": "disconnect_cleanup",
            }
            self.recording_stop_pending = pending
            self.recording_state = RecordingState.STOPPING
            self.recording_status.setText("未采集（设备已断开）")
            window._schedule_pipeline_disconnect_cleanup(
                self, "ble_" + (event.reason or "disconnected"), event.generation, pending
            )
            self._log(
                EventCode.BLE_DISCONNECT,
                "BLE disconnected",
                device_id=self.device_key,
                error_count=1 if event.reason == "unexpected" else 0,
            )

    def _resume_pipeline_after_prior_cleanup(self, generation) -> None:
        prior_tasks = tuple(
            task
            for task in getattr(self, "_pipeline_cleanup_tasks", ())
            if not task.done()
        )
        if not prior_tasks:
            self.pipeline.resume()
            return
        existing = getattr(self, "_pipeline_resume_task", None)
        if (
            existing is not None
            and not existing.done()
            and getattr(self, "_pipeline_resume_generation", None) == generation
        ):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = getattr(self, "loop", None)
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(
                    window._resume_pipeline_after_prior_cleanup, self, generation
                )
            return
        task = loop.create_task(
            window._complete_pipeline_resume(self, generation, prior_tasks)
        )
        self._pipeline_resume_task = task
        self._pipeline_resume_generation = generation
        tasks = getattr(self, "_pipeline_cleanup_tasks", None)
        if tasks is None:
            tasks = set()
            self._pipeline_cleanup_tasks = tasks
        tasks.add(task)

        def finished(completed):
            tasks.discard(completed)
            if getattr(self, "_pipeline_resume_task", None) is completed:
                self._pipeline_resume_task = None
                self._pipeline_resume_generation = None
            window._consume_ble_ui_task_result(completed)

        task.add_done_callback(finished)

    async def _complete_pipeline_resume(self, generation, prior_tasks) -> None:
        await asyncio.gather(*prior_tasks, return_exceptions=True)
        if (
            self._controls_enabled
            and self._accepted_connection_generation == generation
            and getattr(self.WG, "is_connected", False)
        ):
            self.pipeline.resume()

    def _schedule_pipeline_disconnect_cleanup(
        self, reason, generation, transaction
    ) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = getattr(self, "loop", None)
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(
                    window._schedule_pipeline_disconnect_cleanup,
                    self,
                    reason,
                    generation,
                    transaction,
                )
            return
        task = loop.create_task(
            window._complete_pipeline_disconnect_cleanup(
                self, reason, generation, transaction
            )
        )
        tasks = getattr(self, "_pipeline_cleanup_tasks", None)
        if tasks is None:
            tasks = set()
            self._pipeline_cleanup_tasks = tasks
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        task.add_done_callback(window._consume_ble_ui_task_result)

    async def _complete_pipeline_disconnect_cleanup(
        self, reason, generation, transaction
    ) -> None:
        async with self._recording_lock:
            try:
                cleaned_unattached = await asyncio.get_running_loop().run_in_executor(
                    _SHUTDOWN_EXECUTOR,
                    window._disconnect_cleanup_operation,
                    self,
                    reason,
                    generation,
                )
            except Exception as exc:
                if getattr(self, "recording_stop_pending", None) is transaction:
                    transaction["stage"] = "recorder_stop"
                    self.recording_state = RecordingState.STOPPING
                    self.recording_status.setText("断连清理失败（请重试停止）")
                self._log(
                    EventCode.APP_EXCEPTION,
                    "BLE disconnect cleanup failed",
                    level="error",
                    exc_info=(type(exc), exc, exc.__traceback__),
                    device_id=self.device_key,
                    error_count=1,
                )
                return
            if (
                cleaned_unattached is not None
                and getattr(self, "_unattached_recorder", None) is cleaned_unattached
            ):
                self._unattached_recorder = None
                self._unattached_cleanup_intent = None
            if getattr(self, "recording_stop_pending", None) is not transaction:
                return
            if getattr(self.pipeline, "recorder_cleanup_pending", False) or getattr(
                self, "_unattached_recorder", None
            ) is not None:
                transaction["stage"] = "recorder_stop"
                self.recording_state = RecordingState.STOPPING
                return
            self.recording_stop_pending = None
            self.current_session_id = None
            self.recording_state = RecordingState.IDLE
            window._release_active_recording_context(self, schedule_quality=True)

    def _disconnect_cleanup_operation(self, reason, generation):
        window._mark_pipeline_disconnected(self, reason, generation)
        recorder = getattr(self, "_unattached_recorder", None)
        if recorder is not None:
            window._retry_unattached_recorder_cleanup(self)
        return recorder

    async def _wait_pipeline_cleanup_tasks(self, deadline=None) -> None:
        current_task = asyncio.current_task()
        tasks = tuple(
            task
            for task in getattr(self, "_pipeline_cleanup_tasks", ())
            if task is not current_task and not task.done()
        )
        if not tasks:
            return
        if deadline is None:
            deadline = time.monotonic() + BLE_UI_TASK_WAIT_TIMEOUT_SECONDS
        done, pending = await asyncio.wait(
            tasks, timeout=max(0.0, deadline - time.monotonic())
        )
        for task in done:
            task.result()
        if pending:
            raise TimeoutError("pipeline disconnect cleanup is still running")

    def _invalidate_ble_ui_transactions(self) -> None:
        self._scan_epoch = getattr(self, "_scan_epoch", 0) + 1
        self._completed_scan_epoch = None
        self._scan_snapshot = None
        self._connect_epoch = getattr(self, "_connect_epoch", 0) + 1
        self._accepted_connection_generation = None

    def _cancel_ble_ui_tasks(self) -> None:
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        for task in tuple(getattr(self, "_ble_ui_tasks", ())):
            if task is not current_task and not task.done():
                task.cancel()

    async def _wait_ble_ui_tasks(self, deadline=None) -> None:
        current_task = asyncio.current_task()
        tasks = tuple(
            task
            for task in getattr(self, "_ble_ui_tasks", ())
            if task is not current_task and not task.done()
        )
        if tasks:
            loop = asyncio.get_running_loop()
            if deadline is None:
                deadline = time.monotonic() + BLE_UI_TASK_WAIT_TIMEOUT_SECONDS
            available = max(0.0, deadline - time.monotonic())
            drain_budget = min(
                ASYNC_TASK_DRAIN_MAX_SECONDS,
                available / 2.0,
            )
            done, pending = await asyncio.wait(
                tasks, timeout=max(0.0, deadline - drain_budget - time.monotonic())
            )
            for task in done:
                try:
                    task.result()
                except BaseException:
                    pass
            if pending:
                for task in pending:
                    task.cancel()
                drained, pending = await asyncio.wait(
                    pending, timeout=max(0.0, deadline - time.monotonic())
                )
                for task in drained:
                    window._consume_ble_ui_task_result(task)
                for task in pending:
                    task.add_done_callback(window._consume_ble_ui_task_result)
                raise TimeoutError(
                    f"BLE UI tasks did not stop within "
                    f"{BLE_UI_TASK_WAIT_TIMEOUT_SECONDS:.1f}s"
                    + (" and did not terminate" if pending else "")
                )

    @staticmethod
    def _consume_ble_ui_task_result(task) -> None:
        try:
            task.result()
        except BaseException:
            pass

    async def _run_shutdown_async_operation(self, operation, deadline):
        task = asyncio.ensure_future(operation())
        available = max(0.0, deadline - time.monotonic())
        drain_budget = min(ASYNC_TASK_DRAIN_MAX_SECONDS, available / 2.0)
        cancel_at = deadline - drain_budget
        cancellation = None
        while not task.done() and time.monotonic() < cancel_at:
            try:
                await asyncio.wait(
                    {task}, timeout=min(cancel_at - time.monotonic(), 0.05)
                )
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc

        timed_out = not task.done()
        if timed_out:
            task.cancel()
        while not task.done() and time.monotonic() < deadline:
            try:
                await asyncio.wait(
                    {task}, timeout=min(deadline - time.monotonic(), 0.01)
                )
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
                task.cancel()

        if not task.done():
            pending_tasks = getattr(self, "_shutdown_pending_tasks", None)
            if pending_tasks is None:
                pending_tasks = set()
                self._shutdown_pending_tasks = pending_tasks
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)
            task.add_done_callback(window._consume_ble_ui_task_result)
            return (
                TimeoutError("shutdown async stage timed out and did not terminate"),
                cancellation,
            )
        if timed_out:
            window._consume_ble_ui_task_result(task)
            return TimeoutError("shutdown async stage timed out"), cancellation
        try:
            task.result()
        except asyncio.CancelledError as exc:
            # Cancellation never proves that the cleanup operation completed.
            # Report the stage as failed so dependent teardown (notably the
            # producer-lock release) cannot run against active work.
            return exc, cancellation or exc
        except BaseException as exc:
            return exc, cancellation
        return None, cancellation

    async def _run_shutdown_blocking_work(self, stage, operation):
        registry = getattr(self, "_shutdown_blocking_work", None)
        if registry is None:
            registry = {}
            self._shutdown_blocking_work = registry
        work = registry.get(stage)
        if work is not None and work.completed.is_set() and work.error is not None:
            registry.pop(stage, None)
            work = None
        if work is None:
            work = _BlockingShutdownWork(stage, operation)
            try:
                # qasync's default QThread executor may terminate a callable
                # when its asyncio Future is cancelled. Resource cleanup must
                # keep running to a real completion boundary, so use the
                # standard executor and track the underlying work explicitly.
                future = asyncio.get_running_loop().run_in_executor(
                    _SHUTDOWN_EXECUTOR, work.run
                )
            except BaseException:
                # Executor submission and registry publication are one
                # transaction; rejected work must not become a ghost entry.
                raise
            work.future = future
            registry[stage] = work
            work.future.add_done_callback(window._consume_ble_ui_task_result)
        while not work.completed.is_set():
            await asyncio.sleep(0.01)
        if work.error is not None:
            raise work.error

    @staticmethod
    def _pipeline_shutdown_protocol_error(pipeline):
        try:
            stop_accepting = getattr(pipeline, "stop_accepting", None)
            close = getattr(pipeline, "close", None)
            is_closed = getattr(pipeline, "is_closed", None)
        except BaseException as exc:
            return TypeError(f"pipeline shutdown protocol inspection failed: {exc}")
        if not callable(stop_accepting):
            return TypeError("pipeline must implement stop_accepting()")
        if not callable(close):
            return TypeError("pipeline must implement close(timeout=...)")
        try:
            parameters = inspect.signature(close).parameters.values()
        except (TypeError, ValueError) as exc:
            return TypeError(f"pipeline close() signature is unavailable: {exc}")
        supports_timeout = any(
            parameter.name == "timeout"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if not supports_timeout:
            return TypeError("pipeline close() must accept timeout")
        if not isinstance(is_closed, bool):
            return TypeError("pipeline must expose boolean is_closed")
        return None

    @staticmethod
    def _annotate_shutdown_cancellation(cancellation, failures) -> None:
        try:
            cancellation.cleanup_failures = tuple(failures)
        except Exception:
            pass
        if hasattr(cancellation, "add_note"):
            for stage, error in failures:
                cancellation.add_note(
                    f"shutdown cleanup {stage} failed: {type(error).__name__}: {error}"
                )

    @asyncSlot()
    async def search_click(self):
        if not self._controls_enabled:
            return
        if getattr(self, "_connect_in_progress", False):
            QMessageBox.warning(self, "搜索提示", "设备连接进行中，请等待连接完成")
            return
        current_task = asyncio.current_task()
        operation_tasks = getattr(self, "_ble_ui_tasks", None)
        if operation_tasks is None:
            operation_tasks = set()
            self._ble_ui_tasks = operation_tasks
        operation_tasks.add(current_task)
        scan_epoch = getattr(self, "_scan_epoch", 0) + 1
        self._scan_epoch = scan_epoch
        self._completed_scan_epoch = None
        self._scan_in_progress = True
        self._scan_snapshot = None
        self.device_combo.clear()
        self.devices_list.clear()
        try:
            snapshot = _coerce_scan_snapshot(await self.WG.scan(5))
            if scan_epoch != self._scan_epoch or not self._controls_enabled:
                return
            self._scan_snapshot = snapshot
            match_ids = {entry.candidate_id for entry in snapshot.matches}
            for entry in snapshot.devices:
                if entry.candidate_id not in match_ids:
                    target_marker = ""
                elif len(snapshot.matches) == 1:
                    target_marker = " [目标手环]"
                else:
                    target_marker = " [候选手环]"
                self.devices_list.append(
                    f"{entry.index}: {entry.name or 'Unknown'} "
                    f"{entry.address or 'Unknown'} Rssi={entry.rssi}{target_marker}"
                )
            if not snapshot.matches:
                self.devices_list.append("未找到目标手环")
            else:
                for matched_entry in snapshot.matches:
                    self.device_combo.addItem(
                        str(matched_entry.index), matched_entry.candidate_id
                    )
                if len(snapshot.matches) > 1:
                    self.device_combo.setCurrentIndex(-1)
                    self.devices_list.append("找到多个目标手环，请在下拉框中明确选择")
            self.devices_list.append("搜索结束")
            self._completed_scan_epoch = scan_epoch
            self._log(
                EventCode.BLE_SEARCH,
                "BLE search completed",
                device_count=len(snapshot.devices),
            )
        except asyncio.CancelledError:
            if scan_epoch == self._scan_epoch:
                self._scan_snapshot = None
                self.device_combo.clear()
                self.devices_list.clear()
                if self._controls_enabled:
                    self.devices_list.append("搜索已取消")
            raise
        except Exception as exc:
            if scan_epoch != self._scan_epoch:
                return
            self._scan_snapshot = None
            self.device_combo.clear()
            self.devices_list.clear()
            if not self._controls_enabled:
                return
            self.devices_list.append("搜索失败")
            self._log(
                EventCode.BLE_SEARCH,
                "BLE search failed",
                level="error",
                exc_info=(type(exc), exc, exc.__traceback__),
                error_count=1,
            )
            QMessageBox.critical(self, "错误", f"蓝牙搜索失败: {type(exc).__name__}")
        finally:
            if scan_epoch == self._scan_epoch:
                self._scan_in_progress = False
            operation_tasks.discard(current_task)

    @asyncSlot()
    async def connect_semg_click(self):
        if not self._controls_enabled or self.WG.is_connected:
            return
        if getattr(self, "_connect_in_progress", False):
            QMessageBox.warning(self, "连接提示", "设备连接进行中，请勿重复操作")
            return
        scan_epoch = getattr(self, "_scan_epoch", 0)
        completed_scan_epoch = getattr(self, "_completed_scan_epoch", None)
        if getattr(self, "_scan_in_progress", False):
            message = "蓝牙搜索进行中，请等待搜索完成"
        elif completed_scan_epoch != scan_epoch:
            message = "请先重新搜索并确认目标手环"
        else:
            snapshot = getattr(self, "_scan_snapshot", None)
            matches = snapshot.matches if snapshot is not None else ()
            if len(matches) == 1:
                entry = matches[0]
                message = None
            elif len(matches) > 1:
                candidate_id = self.device_combo.currentData()
                selected = tuple(
                    match for match in matches if match.candidate_id == candidate_id
                )
                if len(selected) == 1:
                    entry = selected[0]
                    message = None
                else:
                    message = "找到多个目标手环，请在下拉框中明确选择"
            else:
                message = "未搜索到目标手环，请开启手环后重新搜索"
        if message is not None:
            self._log(
                EventCode.BLE_CONNECT,
                "BLE connection blocked: scan did not confirm target",
                level="warning",
                device_id=self.device_key,
                error_count=1,
            )
            QMessageBox.warning(self, "连接提示", message)
            return
        current_task = asyncio.current_task()
        operation_tasks = getattr(self, "_ble_ui_tasks", None)
        if operation_tasks is None:
            operation_tasks = set()
            self._ble_ui_tasks = operation_tasks
        operation_tasks.add(current_task)
        self._connect_in_progress = True
        connect_epoch = getattr(self, "_connect_epoch", 0) + 1
        self._connect_epoch = connect_epoch
        try:
            address = entry.address
            if not address:
                raise ValueError("设备没有可用地址")
            self.logging_runtime.register_sensitive_token(str(address))
            connected = await self.WG.connect(entry.native_device)
            transaction_current = (
                self._controls_enabled
                and connect_epoch == self._connect_epoch
                and scan_epoch == self._scan_epoch
                and completed_scan_epoch == self._completed_scan_epoch
                and snapshot is self._scan_snapshot
                and entry in snapshot.matches
            )
            if connected is not True or not self.WG.is_connected:
                raise ConnectionError("BLE backend did not establish a new connection")
            if not transaction_current:
                if self.WG.is_connected:
                    await self.WG.disconnect()
                return
            device_key = self.identity_store.device_key(str(address))
            if not (
                self._controls_enabled
                and connect_epoch == self._connect_epoch
                and snapshot is self._scan_snapshot
            ):
                if self.WG.is_connected:
                    await self.WG.disconnect()
                return
            self.device_key = device_key
            self._accepted_connection_generation = getattr(
                self.WG, "client_generation", None
            )
            QMessageBox.information(self, "连接结果", "连接成功")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._controls_enabled or connect_epoch != self._connect_epoch:
                return
            if self.WG.is_connected:
                try:
                    await self.WG.disconnect()
                except Exception as cleanup_exc:
                    if hasattr(exc, "add_note"):
                        exc.add_note(
                            "connection rollback disconnect failed: "
                            f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                        )
            self._log(
                EventCode.BLE_CONNECT,
                "BLE connection failed",
                level="error",
                exc_info=(type(exc), exc, exc.__traceback__),
                device_id=self.device_key,
                error_count=1,
            )
            QMessageBox.critical(self, "连接失败", f"无法连接设备: {type(exc).__name__}")
        finally:
            self._connect_in_progress = False
            operation_tasks.discard(current_task)

    def user_info_click(self):
        if (
            self.recording_state is not RecordingState.IDLE
            or self.is_recording
            or getattr(self, "_active_recording_context", None) is not None
        ):
            QMessageBox.warning(self, "采集中", "采集期间不能修改受试者或动作信息。")
            return
        dialog = window._call_extension_hook(self, "_create_user_info_dialog")
        dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        user_info = dialog.getuserInfo()
        if not user_info:
            return
        subject_input = str(
            user_info.get("subject_identifier", user_info.get("collect_name", ""))
        ).strip()
        if subject_input:
            self.logging_runtime.register_sensitive_token(subject_input)
        try:
            subject_id = self.subject_identity_store.derive_subject_id(subject_input)
            context = window._call_extension_hook(
                self, "_build_recording_context", subject_id, user_info
            )
        except (OSError, TypeError, ValueError) as exc:
            self._log(
                EventCode.APP_EXCEPTION,
                "recording context registration failed",
                level="error",
                exc_info=(type(exc), exc, exc.__traceback__),
                error_count=1,
            )
            QMessageBox.critical(self, "录入失败", "采集信息无效或匿名编号存储不可用。")
            return
        self.subject_key = context.subject_id
        self.temp_info = {
            "side": context.hand_side,
            "action_label": context.action_label,
            "action_phase": context.action_phase,
            "experiment_id": context.experiment_id,
        }
        self._pending_recording_context = context
        window._call_extension_hook(
            self, "_after_recording_context_registered", context, user_info
        )
        action_display = {"rest": "静息", "fist": "握拳", "open_hand": "张手"}[
            context.action_label
        ]
        side_display = "左手" if context.hand_side is HandSide.LEFT else "右手"
        self.context_status_label.setText(
            f"采集信息：{context.subject_id} / {side_display} / {action_display}"
        )
        self.statusBar().showMessage("采集信息已录入", 3000)

    def _set_context_editing_enabled(self, enabled: bool) -> None:
        button = getattr(self, "user_info", None)
        if button is not None:
            button.setEnabled(enabled)

    def _release_active_recording_context(self, *, schedule_quality: bool) -> None:
        session_dir = getattr(self, "_active_session_dir", None)
        self._active_recording_context = None
        self._active_session_dir = None
        window._set_context_editing_enabled(self, True)
        if schedule_quality and session_dir is not None:
            window._schedule_session_quality(self, session_dir)

    def _schedule_session_quality(self, session_dir: Path) -> None:
        label = getattr(self, "quality_status_label", None)
        if label is not None:
            label.setText(f"质量检查：检查中… {session_dir}")
        task = asyncio.ensure_future(
            window._analyze_closed_session(self, session_dir)
        )
        tasks = getattr(self, "_quality_tasks", None)
        if tasks is None:
            tasks = set()
            self._quality_tasks = tasks
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def _wait_quality_tasks(self, deadline=None) -> None:
        current_task = asyncio.current_task()
        tasks = tuple(
            task
            for task in getattr(self, "_quality_tasks", ())
            if task is not current_task and not task.done()
        )
        if not tasks:
            return
        if deadline is None:
            deadline = time.monotonic() + BLE_UI_TASK_WAIT_TIMEOUT_SECONDS
        done, pending = await asyncio.wait(
            tasks, timeout=max(0.0, deadline - time.monotonic())
        )
        for task in done:
            task.result()
        if pending:
            raise TimeoutError("session quality analysis is still running")

    @staticmethod
    def _analyze_and_write_quality_report(session_dir: Path):
        report_path = session_dir / "quality_report.json"
        if report_path.exists() or report_path.is_symlink():
            return window._read_existing_quality_report(report_path), report_path
        report = analyze_session(session_dir)
        temporary_path = session_dir / f".quality_report.{uuid4().hex}.tmp"
        encoded = json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8") + b"\n"
        try:
            with temporary_path.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, report_path)
            except FileExistsError:
                report = window._read_existing_quality_report(report_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return report, report_path

    @staticmethod
    def _read_existing_quality_report(report_path: Path):
        info = report_path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError("existing quality report is not a regular file")
        if info.st_size > 4 * 1024 * 1024:
            raise ValueError("existing quality report exceeds the size limit")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("existing quality report must contain a JSON object")
        return report

    async def _analyze_closed_session(self, session_dir: Path) -> None:
        try:
            report, report_path = await asyncio.get_running_loop().run_in_executor(
                _SHUTDOWN_EXECUTOR,
                window._analyze_and_write_quality_report,
                session_dir,
            )
            usable = report.get("training_usable") is True
            message = window._call_extension_hook(
                self, "_format_quality_report_message", report, report_path
            )
            if self._controls_enabled:
                self.quality_status_label.setText(message)
            self._log(
                EventCode.DATA_SAVE,
                "session quality analysis completed",
                level="info" if usable else "warning",
                session_id=session_dir.name,
                error_count=0 if usable else 1,
            )
            window._call_extension_hook(
                self, "_after_quality_report", report, report_path
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._controls_enabled:
                self.quality_status_label.setText(
                    f"质量检查：执行失败（不可训练） | {session_dir}"
                )
            self._log(
                EventCode.APP_EXCEPTION,
                "session quality analysis failed",
                level="error",
                exc_info=(type(exc), exc, exc.__traceback__),
                session_id=session_dir.name,
                error_count=1,
            )

    def _start_recording_clicked(self):
        if self._controls_enabled:
            asyncio.ensure_future(self._start_recording())

    async def _start_recording(self):
        async with self._recording_lock:
            if not self._controls_enabled:
                return
            if self.recording_state is not RecordingState.IDLE:
                return
            if getattr(self.pipeline, "recorder_cleanup_pending", False):
                self.recording_state = RecordingState.STOPPING
                QMessageBox.warning(
                    self,
                    "Recording unavailable",
                    "The previous recording is still awaiting cleanup; retry Stop first.",
                )
                return
            if not self.WG.is_connected:
                QMessageBox.warning(self, "警告", "请先连接设备")
                return
            context = getattr(self, "_pending_recording_context", None)
            if not isinstance(context, RecordingContext):
                QMessageBox.warning(self, "警告", "请先录入受试者、手侧和本次动作。")
                return
            self.recording_state = RecordingState.STARTING
            self._active_recording_context = context
            window._set_context_editing_enabled(self, False)
            quality_label = getattr(self, "quality_status_label", None)
            if quality_label is not None:
                quality_label.setText("质量检查：等待采集结束")
            connection_generation = getattr(self.WG, "client_generation", None)
            start_token = object()
            self._recording_start_token = start_token
            recorder = None
            recorder_attached = False
            notify_started = False
            command_attempted = False
            stop_pending = None
            try:
                recorder = window._call_extension_hook(
                    self, "_create_data_recorder", context
                )
                self._active_session_dir = Path(recorder.session_dir)
                notify_started = bool(
                    await window._profile_operation(
                        self.WG, "setNotify", (1,), connection_generation
                    )
                )
                window._require_current_recording_connection(
                    self, connection_generation, start_token
                )
                window._arm_stream_watchdog(self, connection_generation)
                command_attempted = True
                stop_pending = {
                    "generation": connection_generation,
                    "stage": "device_stop",
                }
                self.recording_stop_pending = stop_pending
                await window._profile_operation(
                    self.WG, "setDataType", (1, 0), connection_generation
                )
                window._require_current_recording_connection(
                    self, connection_generation, start_token
                )
                self.pipeline.set_recorder(recorder, recording_context=context)
                recorder_attached = True
                self.is_recording = True
                self.recording_state = RecordingState.RECORDING
                self.current_session_id = recorder.session_id
                self.recording_status.setText(f"正在采集: {recorder.session_id}")
                self._log(
                    EventCode.BLE_NOTIFY,
                    "BLE notifications active",
                    device_id=self.device_key,
                    packet_index=0,
                    queue_depth=self.pipeline.queue_depth,
                    dropped_count=self.pipeline.dropped_count,
                )
                self._log(
                    EventCode.DATA_SAVE,
                    "recording started",
                    device_id=self.device_key,
                    session_id=recorder.session_id,
                    packet_index=0,
                )
            except Exception as exc:
                rollback_failures = []
                rollback_recorder_closed = False
                same_connection = window._recording_connection_is_current(
                    self, connection_generation, start_token
                )
                if (
                    not same_connection
                    and getattr(self, "recording_stop_pending", None) is stop_pending
                ):
                    self.recording_stop_pending = None
                if command_attempted and same_connection:
                    try:
                        await window._run_bounded_profile_operation(
                            self,
                            "setDataType",
                            (0, 0),
                            connection_generation,
                            timeout=RECORDING_STOP_BLE_TIMEOUT_SECONDS,
                        )
                        if getattr(self, "recording_stop_pending", None) is stop_pending:
                            self.recording_stop_pending = None
                    except Exception as rollback_error:
                        rollback_failures.append(("device_stop", rollback_error))
                same_connection = window._recording_connection_is_current(
                    self, connection_generation, start_token
                )
                if notify_started and same_connection:
                    try:
                        await window._run_bounded_profile_operation(
                            self,
                            "setNotify",
                            (0,),
                            connection_generation,
                            timeout=RECORDING_STOP_BLE_TIMEOUT_SECONDS,
                        )
                        window._disarm_stream_watchdog(
                            self, connection_generation
                        )
                    except Exception as rollback_error:
                        rollback_failures.append(("notify_stop", rollback_error))
                try:
                    if recorder_attached:
                        recorder_token = self.pipeline.request_recorder_stop(
                            complete=False, error="start_failed"
                        )
                        self._recorder_stop_token = recorder_token
                        await window._finish_recorder_stop_in_background(
                            self, recorder_token
                        )
                        rollback_recorder_closed = True
                        if self._recorder_stop_token is recorder_token:
                            self._recorder_stop_token = None
                    elif recorder is not None:
                        close_unattached = getattr(
                            self.pipeline, "close_unattached_recorder", None
                        )
                        if close_unattached is None:
                            recorder.close(complete=False, error="start_failed")
                        else:
                            close_unattached(
                                recorder, complete=False, error="start_failed"
                            )
                        rollback_recorder_closed = True
                except Exception as rollback_error:
                    rollback_failures.append(("recorder_close", rollback_error))
                    if not recorder_attached and recorder is not None:
                        self._unattached_recorder = recorder
                        self._unattached_cleanup_intent = (False, "start_failed")
                        self.recording_stop_pending = {
                            "generation": connection_generation,
                            "stage": (
                                "notify_stop"
                                if getattr(self.WG, "is_notifying", False)
                                else "unattached_cleanup"
                            ),
                        }
                for stage, rollback_error in rollback_failures:
                    if hasattr(exc, "add_note"):
                        exc.add_note(f"rollback {stage}: {type(rollback_error).__name__}: {rollback_error}")
                self.is_recording = False
                self.recording_state = (
                    RecordingState.STOPPING
                    if (
                        getattr(self.pipeline, "recorder_cleanup_pending", False)
                        or getattr(self, "recording_stop_pending", None) is not None
                    )
                    else RecordingState.IDLE
                )
                self.current_session_id = None
                if rollback_recorder_closed:
                    window._release_active_recording_context(
                        self, schedule_quality=True
                    )
                self._log(
                    EventCode.APP_EXCEPTION,
                    "recording start failed",
                    level="error",
                    exc_info=(type(exc), exc, exc.__traceback__),
                    device_id=self.device_key,
                    error_count=1,
                )
                QMessageBox.critical(self, "错误", f"启动采集失败: {type(exc).__name__}")

            finally:
                if self._recording_start_token is start_token:
                    self._recording_start_token = None

    def _recording_connection_is_current(self, generation, token) -> bool:
        return (
            self._controls_enabled
            and self._recording_start_token is token
            and self.WG.is_connected
            and getattr(self.WG, "client_generation", None) == generation
        )

    def _require_current_recording_connection(self, generation, token) -> None:
        if not window._recording_connection_is_current(self, generation, token):
            raise ConnectionError("BLE connection changed while recording was starting")

    def _arm_stream_watchdog(self, generation) -> None:
        arm = getattr(self.pipeline, "arm_stream_watchdog", None)
        if callable(arm):
            arm(generation)

    def _mark_pipeline_disconnected(self, reason, generation) -> None:
        mark_disconnected = self.pipeline.mark_disconnected
        try:
            parameters = inspect.signature(mark_disconnected).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        supports_generation = any(
            parameter.name == "connection_generation"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if supports_generation:
            mark_disconnected(reason, connection_generation=generation)
        else:
            mark_disconnected(reason)

    def _disarm_stream_watchdog(self, generation) -> None:
        disarm = getattr(self.pipeline, "disarm_stream_watchdog", None)
        if callable(disarm):
            disarm(generation)

    def _stop_connection_is_current(self, generation) -> bool:
        profile = getattr(self, "WG", None)
        accepted = getattr(self, "_accepted_connection_generation", generation)
        return bool(
            profile is not None
            and getattr(profile, "is_connected", False)
            and getattr(profile, "client_generation", generation) == generation
            and accepted == generation
        )

    @staticmethod
    def _profile_operation(profile, method_name, args, generation):
        method = getattr(profile, method_name)
        try:
            parameters = inspect.signature(method).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        supports_generation = any(
            parameter.name == "expected_generation"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        kwargs = {"expected_generation": generation} if supports_generation else {}
        return method(*args, **kwargs)

    async def _run_bounded_profile_operation(
        self, method_name, args, generation, *, timeout
    ):
        task = asyncio.ensure_future(
            window._profile_operation(self.WG, method_name, args, generation)
        )
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
        except BaseException:
            task.cancel()
            raise
        if task not in done:
            task.cancel()
            pending_tasks = getattr(self, "_ble_ui_tasks", None)
            if pending_tasks is None:
                pending_tasks = set()
                self._ble_ui_tasks = pending_tasks
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)
            task.add_done_callback(window._consume_ble_ui_task_result)
            raise TimeoutError(f"{method_name} timed out")
        return task.result()

    async def _finish_recorder_stop_in_background(self, token):
        work = getattr(self, "_recorder_stop_work", None)
        if work is not None and getattr(self, "_recorder_stop_work_token", None) is not token:
            raise RuntimeError("another recorder stop is still running")
        if work is None:
            work = _BlockingShutdownWork(
                "recording_stop",
                lambda: self.pipeline.finish_recorder_stop(
                    token, timeout=RECORDING_STOP_RECORDER_TIMEOUT_SECONDS
                ),
            )
            future = asyncio.get_running_loop().run_in_executor(
                _SHUTDOWN_EXECUTOR, work.run
            )
            work.future = future
            future.add_done_callback(window._consume_ble_ui_task_result)
            self._recorder_stop_work = work
            self._recorder_stop_work_token = token
            registry = getattr(self, "_shutdown_blocking_work", None)
            if registry is None:
                registry = {}
                self._shutdown_blocking_work = registry
            registry["recorder_stop"] = work

        deadline = time.monotonic() + RECORDING_STOP_WORK_WAIT_SECONDS
        while not work.completed.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        if not work.completed.is_set():
            raise TimeoutError("recorder stop worker did not finish within its boundary")
        self._recorder_stop_work = None
        self._recorder_stop_work_token = None
        registry = getattr(self, "_shutdown_blocking_work", None)
        if registry is not None and registry.get("recorder_stop") is work:
            registry.pop("recorder_stop", None)
        if work.error is not None:
            raise work.error
        return work.result

    @staticmethod
    def _recorder_resources_are_closed(recorder) -> bool:
        state = getattr(recorder, "resource_state", None)
        value = getattr(state, "value", state)
        return value == "CLOSED" or getattr(recorder, "closed", False) is True

    def _retry_unattached_recorder_cleanup(self) -> None:
        recorder = getattr(self, "_unattached_recorder", None)
        if recorder is None:
            return
        complete, error = self._unattached_cleanup_intent or (False, "start_failed")
        self.pipeline.stop_recorder(
            complete=complete,
            error=error,
            timeout=RECORDING_STOP_RECORDER_TIMEOUT_SECONDS,
        )
        if not window._recorder_resources_are_closed(recorder):
            close_unattached = getattr(self.pipeline, "close_unattached_recorder", None)
            if close_unattached is None:
                recorder.close(complete=complete, error=error)
            else:
                close_unattached(recorder, complete=complete, error=error)
        if not window._recorder_resources_are_closed(recorder) and hasattr(
            recorder, "resource_state"
        ):
            raise RuntimeError("unattached recorder close was not confirmed")

    async def _finish_unattached_cleanup_in_background(self):
        work = getattr(self, "_unattached_cleanup_work", None)
        if work is None:
            work = _BlockingShutdownWork(
                "unattached_recorder_cleanup",
                lambda: window._retry_unattached_recorder_cleanup(self),
            )
            future = asyncio.get_running_loop().run_in_executor(
                _SHUTDOWN_EXECUTOR, work.run
            )
            work.future = future
            future.add_done_callback(window._consume_ble_ui_task_result)
            self._unattached_cleanup_work = work
            registry = getattr(self, "_shutdown_blocking_work", None)
            if registry is None:
                registry = {}
                self._shutdown_blocking_work = registry
            registry["unattached_recorder_cleanup"] = work
        deadline = time.monotonic() + RECORDING_STOP_WORK_WAIT_SECONDS
        while not work.completed.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        if not work.completed.is_set():
            raise TimeoutError(
                "unattached recorder cleanup did not finish within its boundary"
            )
        self._unattached_cleanup_work = None
        registry = getattr(self, "_shutdown_blocking_work", None)
        if registry is not None and registry.get("unattached_recorder_cleanup") is work:
            registry.pop("unattached_recorder_cleanup", None)
        if work.error is not None:
            raise work.error
        self._unattached_recorder = None
        self._unattached_cleanup_intent = None

    def _stop_recording_clicked(self):
        if self._controls_enabled:
            profile = getattr(self, "WG", None)
            if (
                self.recording_state is not RecordingState.IDLE
                or self.is_recording
                or self.current_session_id is not None
                or getattr(self, "recording_stop_pending", None) is not None
                or bool(getattr(profile, "is_notifying", False))
                or bool(getattr(self.pipeline, "recorder_cleanup_pending", False))
                or getattr(self, "_unattached_recorder", None) is not None
            ):
                self.recording_state = RecordingState.STOPPING
                self.recording_status.setText("正在停止…")
            asyncio.ensure_future(self._stop_recording())

    async def _stop_recording(self):
        async with self._recording_lock:
            if not self._controls_enabled:
                return
            profile = getattr(self, "WG", None)
            recorder_cleanup_pending = bool(
                getattr(self.pipeline, "recorder_cleanup_pending", False)
            )
            transmission_active = bool(getattr(profile, "is_notifying", False))
            pending = getattr(self, "recording_stop_pending", None)
            recorder_token = getattr(self, "_recorder_stop_token", None)
            unattached_recorder = getattr(self, "_unattached_recorder", None)
            recording_active = (
                self.recording_state
                in {
                    RecordingState.STARTING,
                    RecordingState.RECORDING,
                }
                or self.is_recording
                or self.current_session_id is not None
                or recorder_cleanup_pending
                or recorder_token is not None
                or unattached_recorder is not None
            )
            if not recording_active and not transmission_active and pending is None:
                return
            self.recording_state = RecordingState.STOPPING
            self.recording_status.setText("正在停止…")

            if pending is None:
                generation = getattr(profile, "client_generation", None)
                pending = {
                    "generation": generation,
                    "stage": (
                        "device_stop"
                        if getattr(profile, "is_connected", False)
                        else "recorder_stop"
                    ),
                }
                self.recording_stop_pending = pending
            else:
                generation = pending["generation"]

            stage = pending["stage"]
            failures = []
            failed_stage = None

            def retain_failure(name, error):
                nonlocal failed_stage
                failures.append((name, error))
                if failed_stage is None:
                    failed_stage = name

            def transaction_is_current():
                return getattr(self, "recording_stop_pending", None) is pending

            if stage == "device_stop":
                if getattr(profile, "is_connected", False):
                    if not window._stop_connection_is_current(self, generation):
                        return
                    try:
                        await window._run_bounded_profile_operation(
                            self,
                            "setDataType",
                            (0, 0),
                            generation,
                            timeout=RECORDING_STOP_BLE_TIMEOUT_SECONDS,
                        )
                    except Exception as exc:
                        retain_failure("device_stop", exc)
                    if not transaction_is_current() or not window._stop_connection_is_current(
                        self, generation
                    ):
                        return
                elif transaction_is_current():
                    stage = "notify_stop"

            if stage in {"device_stop", "notify_stop"}:
                if getattr(profile, "is_connected", False):
                    if not window._stop_connection_is_current(self, generation):
                        return
                    if getattr(profile, "is_notifying", False):
                        notify_stopped = False
                        try:
                            await window._run_bounded_profile_operation(
                                self,
                                "setNotify",
                                (0,),
                                generation,
                                timeout=RECORDING_STOP_BLE_TIMEOUT_SECONDS,
                            )
                            notify_stopped = True
                        except Exception as exc:
                            retain_failure("notify_stop", exc)
                        if not transaction_is_current() or not window._stop_connection_is_current(
                            self, generation
                        ):
                            return
                        if notify_stopped:
                            window._disarm_stream_watchdog(self, generation)
                    else:
                        window._disarm_stream_watchdog(self, generation)

            if transaction_is_current():
                pending["stage"] = failed_stage or "recorder_stop"

            needs_recorder_stop = bool(
                recorder_token is not None
                or self.is_recording
                or self.current_session_id is not None
                or recorder_cleanup_pending
                or unattached_recorder is not None
            )
            if needs_recorder_stop:
                try:
                    if unattached_recorder is not None:
                        await window._finish_unattached_cleanup_in_background(self)
                    elif recorder_token is None:
                        recorder_token = self.pipeline.request_recorder_stop(
                            complete=True
                        )
                        self._recorder_stop_token = recorder_token
                    if unattached_recorder is None:
                        await window._finish_recorder_stop_in_background(
                            self, recorder_token
                        )
                        if self._recorder_stop_token is recorder_token:
                            self._recorder_stop_token = None
                except Exception as exc:
                    retain_failure("recorder_stop", exc)

            if transaction_is_current():
                self.is_recording = False
                self.current_session_id = None
            cleanup_pending = bool(
                getattr(self.pipeline, "recorder_cleanup_pending", False)
            )
            notify_pending = bool(getattr(profile, "is_notifying", False))
            recorder_pending = bool(
                getattr(self, "_recorder_stop_token", None) is not None
                or cleanup_pending
                or getattr(self, "_unattached_recorder", None) is not None
            )
            if transaction_is_current():
                if failed_stage is not None:
                    pending["stage"] = failed_stage
                elif notify_pending:
                    pending["stage"] = "notify_stop"
                elif recorder_pending:
                    pending["stage"] = "recorder_stop"
                else:
                    self.recording_stop_pending = None

            if (
                not failures
                and not notify_pending
                and not recorder_pending
                and getattr(self, "recording_stop_pending", None) is None
            ):
                self.recording_state = RecordingState.IDLE
                self.recording_status.setText("未采集")
                window._release_active_recording_context(
                    self, schedule_quality=True
                )
                QMessageBox.information(self, "完成", "采集数据已保存")
                return

            self.recording_state = (
                RecordingState.STOPPING
                if (
                    getattr(self, "recording_stop_pending", None) is not None
                    or notify_pending
                    or recorder_pending
                )
                else RecordingState.IDLE
            )
            self.recording_status.setText(
                "停止失败（请重试）"
                if self.recording_state is RecordingState.STOPPING
                else "未采集（停止命令异常）"
            )
            error = ShutdownError(failures) if failures else RuntimeError(
                "recording cleanup remains pending"
            )
            self._log(
                EventCode.APP_EXCEPTION,
                "recording stop incomplete",
                level="error",
                exc_info=(type(error), error, error.__traceback__),
                device_id=self.device_key,
                error_count=len(failures) or 1,
            )
            QMessageBox.critical(
                self,
                "错误",
                "停止采集未完全成功，请再次点击停止采集",
            )

    async def _ensure_transmitting(self):
        if not self.WG.is_connected:
            QMessageBox.warning(self, "警告", "请先连接设备")
            return False
        if not self.WG.is_notifying:
            generation = getattr(self.WG, "client_generation", None)
            await window._profile_operation(
                self.WG, "setNotify", (1,), generation
            )
            window._arm_stream_watchdog(self, generation)
            try:
                stop_pending = {
                    "generation": generation,
                    "stage": "device_stop",
                }
                self.recording_stop_pending = stop_pending
                await window._profile_operation(
                    self.WG, "setDataType", (1, 0), generation
                )
            except Exception as primary_error:
                cleanup_failures = []
                try:
                    await self.WG.setNotify(0)
                    window._disarm_stream_watchdog(self, generation)
                except Exception as cleanup_error:
                    cleanup_failures.append(("notify_stop", cleanup_error))
                    if hasattr(primary_error, "add_note"):
                        primary_error.add_note(
                            "display-start rollback notify_stop failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                try:
                    primary_error.cleanup_failures = tuple(cleanup_failures)
                except Exception:
                    if cleanup_failures and hasattr(primary_error, "add_note"):
                        primary_error.add_note("display-start cleanup failures retained in traceback notes")
                raise
            self._log(
                EventCode.BLE_NOTIFY,
                "BLE notifications active",
                device_id=self.device_key,
                packet_index=0,
                queue_depth=self.pipeline.queue_depth,
                dropped_count=self.pipeline.dropped_count,
            )
        return True

    @asyncSlot()
    async def show_window_click(self):
        if await self._ensure_transmitting():
            self.show_window.setWindowModality(QtCore.Qt.NonModal)
            self.show_window.show()

    @asyncSlot()
    async def show_multichannel_window_click(self):
        if await self._ensure_transmitting():
            self.multichannel_window.setWindowModality(QtCore.Qt.NonModal)
            self.multichannel_window.show()

    def on_start_playback(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择肌电数据文件",
            "",
            "CSV files (*.csv)",
        )
        if not file_path:
            return
        try:
            data_frame = pd.read_csv(file_path, encoding="utf-8-sig")
            columns = [f"channel_{index}" for index in range(1, 9)]
            if any(column not in data_frame.columns for column in columns):
                raise ValueError("文件必须包含 channel_1 到 channel_8")
            self.show_replay_dialog(data_frame[columns].values)
        except Exception as exc:
            self._log(
                EventCode.APP_EXCEPTION,
                "playback file read failed",
                level="error",
                exc_info=(type(exc), exc, exc.__traceback__),
                error_count=1,
            )
            QMessageBox.critical(self, "文件读取失败", str(exc))

    def show_replay_dialog(self, data):
        dialog = OnTimeShowDialog(wg_repeat_time=1, wg_channel=8)
        dialog.setWindowTitle("肌电数据回放")

        def feed():
            for offset in range(0, data.shape[0], 200):
                if not dialog.wave_guide_queue.full():
                    dialog.wave_guide_queue.put(data[offset : offset + 200, :])
                time.sleep(0.5)

        threading.Thread(target=feed, daemon=True).start()
        dialog.exec_()

    async def shutdown(self, reason: str = "application_exit", *, deadline=None):
        async with self._shutdown_lock:
            if self._shutdown_complete:
                return
            self._controls_enabled = False
            if deadline is None:
                deadline = time.monotonic() + WINDOW_SHUTDOWN_TIMEOUT_SECONDS
            window._invalidate_ble_ui_transactions(self)
            window._cancel_ble_ui_tasks(self)
            failures = []
            cancellation = None
            pipeline_protocol_error = window._pipeline_shutdown_protocol_error(
                self.pipeline
            )
            if pipeline_protocol_error is not None:
                failures.append(("pipeline_protocol", pipeline_protocol_error))

            async def async_stage(name, operation):
                nonlocal cancellation
                if name in self._shutdown_stages:
                    return
                # A stage gets only a slice of the one application deadline so
                # one unresponsive dependency cannot consume every later
                # best-effort cleanup opportunity.
                stage_deadline = min(
                    deadline, time.monotonic() + SHUTDOWN_STAGE_MAX_SECONDS
                )
                parameters = inspect.signature(operation).parameters
                awaitable = (
                    operation(stage_deadline) if parameters else operation()
                )
                error, stage_cancellation = await window._run_shutdown_async_operation(
                    self, lambda: awaitable, stage_deadline
                )
                if cancellation is None and stage_cancellation is not None:
                    cancellation = stage_cancellation
                if error is None:
                    self._shutdown_stages.add(name)
                elif error is not None:
                    failures.append((name, error))

            def sync_stage(name, operation):
                if name in self._shutdown_stages:
                    return
                try:
                    operation()
                    self._shutdown_stages.add(name)
                except BaseException as exc:
                    failures.append((name, exc))

            await async_stage(
                "ui_ble_tasks",
                lambda stage_deadline: window._wait_ble_ui_tasks(
                    self, stage_deadline
                ),
            )
            await async_stage(
                "pipeline_disconnect_cleanup",
                lambda stage_deadline: window._wait_pipeline_cleanup_tasks(
                    self, stage_deadline
                ),
            )

            async def stop_device():
                if self.WG.is_connected:
                    await self.WG.setDataType(0, 0)
                self.recording_stop_pending = None

            async def stop_notify():
                if self.WG.is_connected and self.WG.is_notifying:
                    await self.WG.setNotify(0)

            if pipeline_protocol_error is None:
                sync_stage("receive_gate", self.pipeline.stop_accepting)
            await async_stage("device_stop", stop_device)
            await async_stage("notify_stop", stop_notify)
            sync_stage(
                "notification_listener_remove",
                lambda: self.WG.remove_notification_listener(
                    self.pipeline.notification_callback
                ),
            )

            async def disconnect_ble(stage_deadline):
                parameters = inspect.signature(self.WG.disconnect).parameters
                if "deadline" in parameters:
                    await self.WG.disconnect(deadline=stage_deadline)
                else:
                    await self.WG.disconnect()

            await async_stage("ble_disconnect", disconnect_ble)

            async def stop_consumer(stage_deadline):
                if pipeline_protocol_error is not None:
                    raise pipeline_protocol_error
                remaining = max(0.0, stage_deadline - time.monotonic())
                if remaining <= 0:
                    raise TimeoutError("shutdown deadline expired before consumer stop")
                await window._run_shutdown_blocking_work(
                    self,
                    "consumer_stop",
                    lambda: self.pipeline.close(drain=True, timeout=remaining),
                )
                if self.pipeline.is_closed is not True:
                    raise RuntimeError("acquisition consumer close was not confirmed")

            await async_stage("consumer_stop", stop_consumer)

            def close_recorder():
                active = self.recording_state in {
                    RecordingState.STARTING,
                    RecordingState.RECORDING,
                    RecordingState.STOPPING,
                }
                try:
                    if getattr(self, "_unattached_recorder", None) is not None:
                        window._retry_unattached_recorder_cleanup(self)
                        self._unattached_recorder = None
                        self._unattached_cleanup_intent = None
                    else:
                        self.pipeline.stop_recorder(
                            complete=not active,
                            error=reason if active else None,
                        )
                finally:
                    self.is_recording = False
                    self.recording_state = RecordingState.IDLE
                    self.current_session_id = None

            if "consumer_stop" in self._shutdown_stages:
                async def wait_recording_operation():
                    async with self._recording_lock:
                        pass

                await async_stage("recording_operation", wait_recording_operation)
                if "recording_operation" in self._shutdown_stages:
                    await async_stage(
                        "recorder_stop",
                        lambda: window._run_shutdown_blocking_work(
                            self, "recorder_stop", close_recorder
                        ),
                    )
                    if "recorder_stop" in self._shutdown_stages:
                        window._release_active_recording_context(
                            self, schedule_quality=True
                        )
                else:
                    failures.append(
                        (
                            "forced_boundary.recording_operation",
                            RuntimeError(
                                "recording operation did not stop; recorder remains open"
                            ),
                        )
                    )
                await async_stage(
                    "status_publish",
                    lambda: window._run_shutdown_blocking_work(
                        self,
                        "status_publish",
                        lambda: self.pipeline.publish_status(
                            FLAG_DISCONNECTED | FLAG_STALE
                        ),
                    ),
                )
                await async_stage(
                    "shared_resources_close",
                    lambda: window._run_shutdown_blocking_work(
                        self,
                        "shared_resources_close",
                        self.shared_writer.close_resources,
                    ),
                )
                if "shared_resources_close" in self._shutdown_stages:
                    # Flush/mapped-file/file close may block and runs above in the
                    # executor. Only the short, thread-affine Windows mutex
                    # release returns to the qasync/GUI owner thread.
                    sync_stage("shared_close", self.shared_writer.release_producer)
            else:
                failures.append(
                    (
                        "forced_boundary.consumer_active",
                        RuntimeError(
                            "consumer did not stop; recorder/audit/shared sinks remain open"
                        ),
                    )
                )
            await async_stage(
                "quality_analysis",
                lambda stage_deadline: window._wait_quality_tasks(
                    self, stage_deadline
                ),
            )
            sync_stage("listener_remove", lambda: self.WG.remove_state_listener(self._on_ble_state))
            if failures:
                sync_stage(
                    "shutdown_log",
                    lambda: self._log(
                        EventCode.APP_EXCEPTION,
                        "application shutdown completed with failures",
                        level="error",
                        error_count=len(failures),
                    ),
                )
            else:
                sync_stage(
                    "shutdown_log",
                    lambda: self._log(
                        EventCode.SYSTEM_SHUTDOWN, "application shutdown completed"
                    ),
                )
            await async_stage(
                "logging_shutdown",
                lambda: window._run_shutdown_blocking_work(
                    self, "logging_shutdown", self.logging_runtime.shutdown
                ),
            )
            if cancellation is not None:
                window._annotate_shutdown_cancellation(
                    cancellation, failures
                )
                self._last_shutdown_error = cancellation
                self._shutdown_cancellation_attempt_token = getattr(
                    self, "_shutdown_attempt_token", None
                )
                raise cancellation
            if failures:
                shutdown_error = ShutdownError(failures)
                self._last_shutdown_error = shutdown_error
                raise shutdown_error
            self._shutdown_complete = True
            self._last_shutdown_error = None

    def request_shutdown(self, reason: str = "application_exit"):
        self._controls_enabled = False
        self._pending_recording_context = None
        self.subject_key = None
        self.temp_info = {"side": HandSide.UNKNOWN}
        window._invalidate_ble_ui_transactions(self)
        window._cancel_ble_ui_tasks(self)
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.ensure_future(self.shutdown(reason))
            self._shutdown_task.add_done_callback(self._finish_close)
        return self._shutdown_task

    def note_shutdown_request(self, reason: str = "about_to_quit"):
        self._controls_enabled = False
        self._pending_recording_context = None
        self.subject_key = None
        self.temp_info = {"side": HandSide.UNKNOWN}
        window._invalidate_ble_ui_transactions(self)
        window._cancel_ble_ui_tasks(self)
        if self._quit_reason is None:
            self._quit_reason = reason

    def _finish_close(self, task):
        try:
            task.result()
        except asyncio.CancelledError as exc:
            self._last_shutdown_error = exc
            self._shutdown_task = None
            return
        except Exception as exc:
            self._last_shutdown_error = exc
            self._shutdown_task = None
            QMessageBox.critical(
                self,
                "退出未完成",
                f"资源释放失败，可再次关闭重试: {type(exc).__name__}: {exc}",
            )
            return
        self._allow_close = True
        super().close()

    def closeEvent(self, event):
        if self._allow_close or self._shutdown_complete:
            event.accept()
            return
        event.ignore()
        self.request_shutdown("window_close")


def _configure_runtime(config: AppConfig) -> LoggingRuntime:
    return configure_logging(
        config.logging.path,
        level=config.logging.level,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
        queue_capacity=config.logging.queue_capacity,
    )


def install_shutdown_hooks(app, form, runtime):
    """Route Qt termination and uncaught exceptions through one shutdown gate."""
    app.aboutToQuit.connect(lambda: form.note_shutdown_request("about_to_quit"))

    def handle_exception(exc_type, exc_value, traceback):
        try:
            runtime.get_logger(event=EventCode.APP_EXCEPTION).error(
                f"uncaught exception: {exc_type.__name__}: {exc_value}",
                exc_info=(exc_type, exc_value, traceback),
            )
        finally:
            form.request_shutdown("uncaught_exception")

    sys.excepthook = handle_exception
    return handle_exception


def _pending_application_tasks(form, *, loop=None, baseline=(), exclude=()):
    excluded = set(exclude)
    tasks = set(getattr(form, "_shutdown_pending_tasks", ()))
    tasks.update(getattr(form, "_ble_ui_tasks", ()))
    shutdown_task = getattr(form, "_shutdown_task", None)
    if shutdown_task is not None:
        tasks.add(shutdown_task)
    profile = getattr(form, "WG", None)
    pending_disconnect = getattr(profile, "_pending_disconnect_task", None)
    if pending_disconnect is not None:
        tasks.add(pending_disconnect)
    if loop is not None:
        baseline_tasks = set(baseline)
        tasks.update(
            task
            for task in asyncio.all_tasks(loop)
            if task not in baseline_tasks and task not in excluded
        )
    return tuple(
        task
        for task in tasks
        if task is not None and task not in excluded and not task.done()
    )


def _active_blocking_shutdown_work(form):
    return tuple(
        work
        for work in getattr(form, "_shutdown_blocking_work", {}).values()
        if not work.completed.is_set()
    )


def _reports_unterminated_cleanup(error):
    """Preserve a cleanup timeout's hard-boundary evidence across task races."""
    cleanup_task = getattr(error, "cleanup_task", None)
    if cleanup_task is not None:
        return not cleanup_task.done()
    return bool(
        getattr(error, "cleanup_pending", False)
        or (
            isinstance(error, TimeoutError)
            and "did not terminate" in str(error).casefold()
        )
    )


def _drain_application_inventory(loop, form, deadline, *, baseline=(), exclude=()):
    excluded = set(exclude)
    try:
        current_task = asyncio.current_task(loop=loop)
    except RuntimeError:
        current_task = None
    if current_task is not None:
        excluded.add(current_task)
    stable_empty_snapshots = 0
    while True:
        current_tasks = set(
            _pending_application_tasks(
                form, loop=loop, baseline=baseline, exclude=excluded
            )
        )
        active_work = _active_blocking_shutdown_work(form)
        if not current_tasks and not active_work:
            stable_empty_snapshots += 1
            if stable_empty_snapshots >= 2:
                return (), (), True
        else:
            stable_empty_snapshots = 0
            for task in current_tasks:
                task.cancel()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        loop.run_until_complete(asyncio.sleep(min(0.01, remaining)))
    pending_tasks = tuple(
        task
        for task in _pending_application_tasks(
            form, loop=loop, baseline=baseline, exclude=excluded
        )
        if not task.done()
    )
    for task in pending_tasks:
        task.cancel()
    return pending_tasks, _active_blocking_shutdown_work(form), False


def _run_application_loop(
    loop,
    form,
    *,
    shutdown_attempts: int = 3,
    shutdown_timeout_seconds: float = 5.0,
    hard_terminate=os._exit,
) -> None:
    if shutdown_attempts <= 0 or shutdown_timeout_seconds <= 0:
        raise ValueError("shutdown retry bounds must be positive")
    owned_task_baseline = tuple(asyncio.all_tasks(loop))
    try:
        loop.run_forever()
    finally:
        reason = getattr(form, "_quit_reason", None) or "event_loop_stopped"
        deadline = time.monotonic() + shutdown_timeout_seconds
        failures = []
        completed = False
        cancellation = None
        for attempt in range(1, shutdown_attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failures.append(("shutdown_deadline", TimeoutError("shutdown deadline expired")))
                break
            form._last_shutdown_error = None
            attempt_token = object()
            form._shutdown_attempt_token = attempt_token
            form._shutdown_cancellation_attempt_token = None
            shutdown_task = getattr(form, "_shutdown_task", None)
            if shutdown_task is None:
                parameters = inspect.signature(form.shutdown).parameters
                shutdown = (
                    form.shutdown(reason, deadline=deadline)
                    if "deadline" in parameters
                    else form.shutdown(reason)
                )
                shutdown_task = loop.create_task(shutdown)
                form._shutdown_task = shutdown_task
            inventory_reserve = min(ASYNC_TASK_DRAIN_MAX_SECONDS, remaining / 2.0)
            done, _ = loop.run_until_complete(
                asyncio.wait(
                    {shutdown_task}, timeout=max(0.0, remaining - inventory_reserve)
                )
            )
            if not done:
                shutdown_task.cancel()
                failures.append(
                    (
                        f"attempt_{attempt}.shutdown_timeout",
                        TimeoutError("shutdown task exceeded application deadline"),
                    )
                )
                break
            try:
                shutdown_task.result()
                completed = True
                break
            except ShutdownError as error:
                failures.extend(
                    (f"attempt_{attempt}.{stage}", stage_error)
                    for stage, stage_error in error.failures
                )
            except asyncio.CancelledError as exc:
                original_cancellation = getattr(form, "_last_shutdown_error", None)
                cancellation = (
                    original_cancellation
                    if (
                        isinstance(original_cancellation, asyncio.CancelledError)
                        and getattr(
                            form, "_shutdown_cancellation_attempt_token", None
                        )
                        is attempt_token
                    )
                    else exc
                )
                break
            except BaseException as error:
                failures.append((f"attempt_{attempt}.shutdown", error))
            finally:
                if getattr(form, "_shutdown_task", None) is shutdown_task:
                    form._shutdown_task = None
        abandoned_tasks, active_work, inventory_stable = _drain_application_inventory(
            loop, form, deadline, baseline=owned_task_baseline
        )
        reported_unterminated = tuple(
            (stage, error)
            for stage, error in failures
            if _reports_unterminated_cleanup(error)
        )
        forced_count = (
            len(abandoned_tasks)
            + len(active_work)
            + (not inventory_stable)
            + bool(reported_unterminated)
        )
        if forced_count:
            completed = False
            failures.append(
                (
                    "forced_process_boundary",
                    TimeoutError(
                        f"{len(abandoned_tasks)} task(s) and {len(active_work)} "
                        "blocking cleanup operation(s) remain active; "
                        f"stable_empty_inventory={inventory_stable}; "
                        f"reported_unterminated={len(reported_unterminated)}"
                    ),
                )
            )
        if cancellation is not None:
            if failures:
                window._annotate_shutdown_cancellation(cancellation, failures)
            form._last_shutdown_error = cancellation
            if forced_count:
                try:
                    print(
                        "EMG shutdown cancelled with active cleanup; forcing process boundary: "
                        f"tasks={len(abandoned_tasks)},blocking={len(active_work)}",
                        file=sys.stderr,
                    )
                finally:
                    hard_terminate(2)
                    raise RuntimeError("hard_terminate returned unexpectedly")
            raise cancellation
        if not completed:
            aggregate = ShutdownError(failures)
            form._last_shutdown_error = aggregate
            stages = ",".join(
                f"{stage}:{type(error).__name__}" for stage, error in failures
            )
            try:
                print(
                    f"EMG shutdown failed after bounded retries: {stages}",
                    file=sys.stderr,
                )
            except BaseException as fallback_error:
                failures.append(("fallback_report", fallback_error))
                aggregate = ShutdownError(failures)
                form._last_shutdown_error = aggregate
            if forced_count:
                hard_terminate(2)
                raise RuntimeError("hard_terminate returned unexpectedly")
            raise aggregate


def main() -> int:
    config = load_config(CONFIG_PATH)
    runtime = _configure_runtime(config)
    app = QtWidgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    form = window(loop, config=config, logging_runtime=runtime)
    form.show()
    install_shutdown_hooks(app, form, runtime)
    try:
        with loop:
            _run_application_loop(loop, form)
    except ShutdownError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
