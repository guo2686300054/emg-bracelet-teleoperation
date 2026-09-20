import asyncio
import os
import runpy
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

import qt5_bleak
import qt5_training_capture
from acquisition_pipeline import PipelineEventType
from app_config import load_config
from data_recorder import RecorderQualitySnapshot
from emg_protocol import DeviceKey, HandSide
from qt5_bleak import RecordingState
from shared_memory_v2 import SharedMemoryWriter
from shared_memory_v2 import FLAG_DISCONNECTED, FLAG_REPLAY, FLAG_SYNTHETIC_TIME, SharedMemoryReader
from replay_emg_session import decision_filter_from_bundle
from test_replay_emg_session import write_derived
from training_capture_dialog import ACTION_OPTIONS, TrainingCaptureDialog
from training_capture_support import TrainingCaptureSettings, format_stop_report
from training_contract import CANONICAL_LABELS


DEVICE = DeviceKey("dev-" + "2" * 32)
SUBJECT = "sub-" + "1" * 32


class FakeProfile:
    is_connected = True
    is_notifying = False
    client_generation = 7

    def add_state_listener(self, listener):
        self.state_listener = listener

    def add_notification_listener(self, listener):
        self.notification_listener = listener


class FakePipeline:
    recorder_cleanup_pending = False

    def __init__(self):
        self.stale = False
        self.stream_expected = False
        self.quality_error = None

    def start(self):
        pass

    def notification_callback(self, notification):
        pass

    def quality_snapshot(self):
        if self.quality_error:
            raise self.quality_error
        return SimpleNamespace(
            snapshot_monotonic_ns=2_000_000_000,
            last_published_monotonic_ns=1_000_000_000,
            host_queue_dropped_count=4,
            queue_depth=2,
            watchdog_stale=self.stale,
            stream_expected=self.stream_expected,
            connection_generation=7,
        )


class TrainingCaptureGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def make_window(self, *, pipeline=None, shared_writer=SimpleNamespace()):
        config = load_config(qt5_bleak.CONFIG_PATH)
        runtime = SimpleNamespace(
            get_logger=lambda **kwargs: SimpleNamespace(
                info=lambda *args, **kwargs: None,
                warning=lambda *args, **kwargs: None,
                error=lambda *args, **kwargs: None,
            ),
            register_sensitive_token=lambda token: None,
            shutdown=lambda: None,
        )
        with mock.patch.object(
            qt5_bleak,
            "recover_incomplete_sessions",
            return_value=SimpleNamespace(issues=()),
        ):
            form = qt5_training_capture.TrainingCaptureWindow(
                asyncio.new_event_loop(),
                config=config,
                logging_runtime=runtime,
                profile=FakeProfile(),
                shared_writer=shared_writer,
                pipeline=pipeline or FakePipeline(),
                identity_store=SimpleNamespace(),
                subject_identity_store=SimpleNamespace(),
            )
        form.device_key = DEVICE
        form._accepted_connection_generation = 7
        return form

    def settings(self, action="fist", countdown=2, duration=3):
        return TrainingCaptureSettings(
            subject_id=SUBJECT,
            session_id=f"subject01-{action}-01",
            device_id=DEVICE,
            hand_side=HandSide.LEFT,
            action_label=action,
            experiment_batch="batch-01",
            countdown_seconds=countdown,
            duration_seconds=duration,
        )

    def prepare(self, form, **kwargs):
        settings = self.settings(**kwargs)
        form._pending_training_settings = settings
        form._pending_recording_context = settings.to_recording_context()
        return settings

    def close_window(self, form):
        form._quality_timer.stop()
        form._allow_close = True
        form.close()
        form.loop.close()

    def test_dialog_uses_canonical_actions_hold_and_shared_session_validation(self):
        dialog = TrainingCaptureDialog(device_id=DEVICE)
        try:
            self.assertEqual(
                tuple(dialog.action_combo.itemData(i) for i in range(3)),
                CANONICAL_LABELS,
            )
            self.assertEqual(tuple(value for _, value in ACTION_OPTIONS), CANONICAL_LABELS)
            self.assertEqual(dialog.action_phase_edit.text(), "hold")
            self.assertTrue(dialog.action_phase_edit.isReadOnly())
            self.assertTrue(dialog.device_id_edit.isReadOnly())
            dialog.subject_identifier_edit.setText("subject01")
            dialog.session_id_edit.setText("INVALID SESSION")
            dialog.experiment_batch_edit.setText("batch-01")
            with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning:
                dialog._accept_if_valid()
            self.assertTrue(warning.called)
            self.assertNotEqual(dialog.result(), QtWidgets.QDialog.Accepted)
        finally:
            dialog.close()

    def test_title_logging_shared_path_and_entry_spec_smoke(self):
        form = self.make_window()
        try:
            self.assertEqual(form.windowTitle(), "肌电手环训练采集程序")
            self.assertIsNotNone(form.findChild(QtWidgets.QGroupBox, "training_quality_panel"))
        finally:
            self.close_window(form)
        config = load_config(qt5_bleak.CONFIG_PATH)
        with mock.patch.object(qt5_training_capture, "configure_logging") as configure:
            configure.return_value = object()
            qt5_training_capture._configure_training_runtime(config)
        self.assertEqual(configure.call_args.kwargs["logger_name"], "emg_training_capture")
        self.assertEqual(configure.call_args.kwargs["file_name"], "training_capture.log")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "training.bin"
            writer = SharedMemoryWriter(target)
            try:
                self.assertEqual(writer.shared_file_path, target.resolve())
                self.assertNotEqual(target.resolve(), qt5_bleak.SHARED_FILE_PATH.resolve())
            finally:
                writer.close()
        runpy.run_path(str(Path("build/qt5_training_capture_entry.py")))
        spec = Path("qt5_training_capture.spec").read_text(encoding="utf-8")
        self.assertIn("qt5_training_capture_entry.py", spec)
        self.assertIn("肌电手环训练采集程序", spec)

    def test_window_injects_training_writer_without_mutating_legacy_global(self):
        sentinel = SimpleNamespace()
        original = qt5_bleak.SHARED_FILE_PATH
        with mock.patch.object(
            qt5_training_capture, "SharedMemoryWriter", return_value=sentinel
        ) as factory:
            form = self.make_window(shared_writer=None)
        try:
            self.assertIs(form.shared_writer, sentinel)
            self.assertEqual(factory.call_args.args[0], qt5_training_capture.TRAINING_SHARED_FILE_PATH)
            self.assertEqual(qt5_bleak.SHARED_FILE_PATH, original)
        finally:
            self.close_window(form)

    def test_context_freezes_immediately_duplicate_start_is_ignored_and_stop_cleans_task(self):
        form = self.make_window()
        self.prepare(form)

        async def flow():
            gate = asyncio.Event()

            async def blocked():
                await gate.wait()

            form._run_timed_capture = blocked
            form._start_recording_clicked()
            first = form._timed_capture_task
            self.assertEqual(
                form.training_phase, qt5_training_capture.TrainingCapturePhase.PREPARING
            )
            self.assertFalse(form.user_info.isEnabled())
            self.assertIsNotNone(form._frozen_training_settings)
            form._start_recording_clicked()
            self.assertIs(form._timed_capture_task, first)
            form._stop_recording_clicked()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertTrue(first.done())
            self.assertIsNone(form._timed_capture_task)
            self.assertEqual(
                form.training_phase, qt5_training_capture.TrainingCapturePhase.IDLE
            )

        try:
            asyncio.run(flow())
        finally:
            self.close_window(form)

    def test_context_change_during_countdown_is_rejected_before_ble_start(self):
        form = self.make_window()
        original = self.prepare(form, countdown=0)
        form._frozen_training_settings = original
        form._frozen_recording_context = original.to_recording_context()
        form._pending_training_settings = self.settings(action="rest", countdown=0)
        try:
            with mock.patch.object(
                qt5_bleak.window, "_start_recording", new=mock.AsyncMock()
            ) as start:
                with self.assertRaisesRegex(RuntimeError, "changed during countdown"):
                    asyncio.run(form._run_timed_capture())
            start.assert_not_awaited()
        finally:
            self.close_window(form)

    def test_countdown_auto_stops_and_clears_old_recorder(self):
        form = self.make_window()
        settings = self.prepare(form)
        form._frozen_training_settings = settings
        form._frozen_recording_context = settings.to_recording_context()
        events = []

        async def no_wait():
            events.append("tick")

        async def start(_self):
            events.append("start")
            form.recording_state = RecordingState.RECORDING

        async def stop(_self):
            events.append("stop")
            form.recording_state = RecordingState.IDLE

        try:
            form._wait_capture_second = no_wait
            form._training_recorder = SimpleNamespace()
            with mock.patch.object(qt5_bleak.window, "_start_recording", start), mock.patch.object(
                qt5_bleak.window, "_stop_recording", stop
            ):
                asyncio.run(form._run_timed_capture())
            self.assertEqual(events, ["tick", "tick", "start", "tick", "tick", "tick", "stop"])
            self.assertEqual(form.training_phase, qt5_training_capture.TrainingCapturePhase.IDLE)
            self.assertIsNone(form._training_recorder)
        finally:
            self.close_window(form)

    def test_disconnect_fault_and_close_cancel_timed_task(self):
        form = self.make_window()
        task = mock.Mock(done=lambda: False, cancel=mock.Mock())
        form._timed_capture_task = task
        form.training_phase = qt5_training_capture.TrainingCapturePhase.RECORDING
        form.recording_state = RecordingState.RECORDING
        try:
            with mock.patch.object(qt5_bleak.window, "_on_ble_state"):
                form._on_ble_state(
                    SimpleNamespace(event_type="disconnected", generation=7)
                )
            task.cancel.assert_called()
            task.cancel.reset_mock()
            with mock.patch.object(qt5_bleak.window, "_handle_pipeline_event"):
                form._handle_pipeline_event(
                    SimpleNamespace(event_type=PipelineEventType.RECORDING_FAULT)
                )
            task.cancel.assert_called()
            task.cancel.reset_mock()
            with mock.patch.object(qt5_bleak.window, "closeEvent"):
                form.closeEvent(SimpleNamespace())
            task.cancel.assert_called()
            form._timed_capture_task = None
            form.recording_state = RecordingState.IDLE
            form.recording_stop_pending = None
            form._reconcile_stopping_phase()
            self.assertEqual(
                form.training_phase, qt5_training_capture.TrainingCapturePhase.IDLE
            )
        finally:
            form._timed_capture_task = None
            self.close_window(form)

    def test_nonfatal_pipeline_events_preserve_preparing_and_countdown(self):
        form = self.make_window()
        settings = self.prepare(form)
        context = settings.to_recording_context()
        task = mock.Mock(done=lambda: False, cancel=mock.Mock())
        form._timed_capture_task = task
        form._frozen_training_settings = settings
        form._frozen_recording_context = context
        form.recording_state = RecordingState.IDLE
        try:
            with mock.patch.object(qt5_bleak.window, "_handle_pipeline_event"):
                for phase in (
                    qt5_training_capture.TrainingCapturePhase.PREPARING,
                    qt5_training_capture.TrainingCapturePhase.COUNTDOWN,
                ):
                    for event_type in (
                        PipelineEventType.PARSE_ERROR,
                        PipelineEventType.SINK_ERROR,
                    ):
                        form.training_phase = phase
                        form._handle_pipeline_event(
                            SimpleNamespace(event_type=event_type)
                        )
                        self.assertEqual(form.training_phase, phase)
                        self.assertIs(form._frozen_training_settings, settings)
                        self.assertIs(form._frozen_recording_context, context)
                        self.assertIs(form._timed_capture_task, task)
                        task.cancel.assert_not_called()
        finally:
            form._timed_capture_task = None
            self.close_window(form)

    def test_stale_generation_disconnect_does_not_cancel_current_recording(self):
        form = self.make_window()
        task = mock.Mock(done=lambda: False, cancel=mock.Mock())
        form._timed_capture_task = task
        form.training_phase = qt5_training_capture.TrainingCapturePhase.RECORDING
        form.recording_state = RecordingState.RECORDING
        try:
            with mock.patch.object(qt5_bleak.window, "_on_ble_state") as parent:
                form._on_ble_state(
                    SimpleNamespace(event_type="disconnected", generation=6)
                )
            task.cancel.assert_not_called()
            self.assertEqual(
                form.training_phase,
                qt5_training_capture.TrainingCapturePhase.RECORDING,
            )
            parent.assert_called_once()
        finally:
            form._timed_capture_task = None
            self.close_window(form)

    def test_quality_distinguishes_host_and_device_and_stale_recovers(self):
        pipeline = FakePipeline()
        form = self.make_window(pipeline=pipeline)
        form.training_phase = qt5_training_capture.TrainingCapturePhase.RECORDING
        form._capture_started_monotonic = 1.0
        form._training_recorder = SimpleNamespace(
            quality_snapshot=lambda: RecorderQualitySnapshot(False, 0, 0, 0, 7, 10)
        )
        try:
            with mock.patch.object(qt5_training_capture.time, "monotonic", return_value=2.0):
                pipeline.stream_expected = True
                pipeline.stale = True
                form._refresh_live_quality()
                self.assertEqual(form.live_quality_labels["stale"].text(), "是")
                self.assertEqual(form.live_quality_labels["dropped"].text(), "4")
                self.assertEqual(form.live_quality_labels["sequence_gap"].text(), "不可检测")
                pipeline.stale = False
                form._refresh_live_quality()
                self.assertEqual(form.live_quality_labels["stale"].text(), "否")
            form.training_phase = qt5_training_capture.TrainingCapturePhase.COUNTDOWN
            pipeline.stream_expected = False
            form._refresh_live_quality()
            self.assertEqual(
                form.live_quality_labels["effective_sample_rate"].text(), "不可检测"
            )
        finally:
            self.close_window(form)

    def test_watchdog_stale_is_unavailable_when_stream_is_not_expected(self):
        pipeline = FakePipeline()
        pipeline.stale = False
        pipeline.stream_expected = False
        form = self.make_window(pipeline=pipeline)
        try:
            for phase in (
                qt5_training_capture.TrainingCapturePhase.IDLE,
                qt5_training_capture.TrainingCapturePhase.COUNTDOWN,
                qt5_training_capture.TrainingCapturePhase.STOPPING,
            ):
                form.training_phase = phase
                form._refresh_live_quality()
                self.assertEqual(form.live_quality_labels["stale"].text(), "不可检测")
            form.training_phase = qt5_training_capture.TrainingCapturePhase.STOPPING
            form.recording_state = RecordingState.IDLE
            form.recording_stop_pending = {"stage": "disconnect_cleanup"}
            form._refresh_live_quality()
            self.assertEqual(form.live_quality_labels["stale"].text(), "不可检测")
        finally:
            self.close_window(form)

    def test_quality_errors_are_rate_limited_and_not_silent(self):
        pipeline = FakePipeline()
        pipeline.quality_error = RuntimeError("snapshot failed")
        form = self.make_window(pipeline=pipeline)
        form._log = mock.Mock()
        try:
            with mock.patch.object(
                qt5_training_capture.time, "monotonic", side_effect=[10.0, 11.0, 21.0]
            ):
                form._refresh_live_quality()
                form._refresh_live_quality()
                form._refresh_live_quality()
            self.assertEqual(form._log.call_count, 2)
            self.assertTrue(form._log.call_args.kwargs["exc_info"])
            self.assertEqual(form.live_quality_summary.text(), "实时质量证据不可用")
        finally:
            self.close_window(form)

    def test_stop_report_shows_every_rejection_reason(self):
        form = self.make_window()
        report = {
            "training_usable": False,
            "checks": {
                "packet_loss": {"status": "fail", "message": "loss too high"},
                "sample_rate": {"status": "warning", "message": "rate unconfirmed"},
            },
        }
        expected = format_stop_report(report)
        try:
            with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning:
                form._after_quality_report(report, "unused.json")
            self.assertIn("loss too high", expected)
            self.assertIn("rate unconfirmed", expected)
            warning.assert_called_once_with(form, "训练资格检查", expected)
        finally:
            self.close_window(form)

    def test_shutdown_quality_report_updates_without_modal_dialog(self):
        form = self.make_window()
        report = {
            "training_usable": False,
            "checks": {
                "packet_loss": {"status": "fail", "message": "loss too high"},
            },
        }
        try:
            form._controls_enabled = False
            with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning, mock.patch.object(
                QtWidgets.QMessageBox, "information"
            ) as information:
                form._after_quality_report(report, "quality.json")
            warning.assert_not_called()
            information.assert_not_called()
            self.assertIn("loss too high", form.quality_status_label.text())
            self.assertEqual(form._last_training_quality_report[1], "quality.json")
        finally:
            form._controls_enabled = True
            self.close_window(form)

    def test_replay_normal_completion_updates_report_and_returns_idle(self):
        form = self.make_window()
        report = SimpleNamespace(
            status="completed", frames_written=10, source_rows=10,
            predictions_emitted=4, errors=(),
        )
        task = mock.Mock()
        task.cancelled.return_value = False
        task.result.return_value = report
        form._replay_task = task
        form._ble_ui_tasks.add(task)
        form.training_phase = qt5_training_capture.TrainingCapturePhase.REPLAYING
        try:
            form._replay_finished(task)
            self.assertIs(form._last_replay_report, report)
            self.assertEqual(form.training_phase, qt5_training_capture.TrainingCapturePhase.IDLE)
            self.assertIn("10/10", form.capture_instruction_label.text())
            self.assertIn("训练资格：否", form.live_quality_summary.text())
        finally:
            self.close_window(form)

    def test_replay_cancel_and_close_signal_worker_without_task_cancel(self):
        form = self.make_window()
        event = __import__("threading").Event()
        form._replay_cancel_event = event
        form.training_phase = qt5_training_capture.TrainingCapturePhase.REPLAYING
        try:
            form._cancel_replay()
            self.assertTrue(event.is_set())
            self.assertEqual(
                form.training_phase,
                qt5_training_capture.TrainingCapturePhase.REPLAY_STOPPING,
            )
            event.clear()
            form._allow_close = True
            form.close()
            self.assertTrue(event.is_set())
        finally:
            form._quality_timer.stop()
            form.loop.close()

    def test_replay_failure_report_and_task_exception_are_visible(self):
        form = self.make_window()
        failed_report = SimpleNamespace(
            status="failed", frames_written=2, source_rows=10,
            predictions_emitted=0, errors=("writer failed",),
        )
        report_task = mock.Mock()
        report_task.cancelled.return_value = False
        report_task.result.return_value = failed_report
        form._replay_task = report_task
        form._ble_ui_tasks.add(report_task)
        try:
            form._replay_finished(report_task)
            self.assertIn("writer failed", form.capture_instruction_label.text())
            boom_task = mock.Mock()
            boom_task.cancelled.return_value = False
            boom_task.result.side_effect = RuntimeError("boom")
            form._replay_task = boom_task
            form._ble_ui_tasks.add(boom_task)
            with mock.patch.object(QtWidgets.QMessageBox, "critical") as critical:
                form._replay_finished(boom_task)
            critical.assert_called_once()
            self.assertIn("boom", form.capture_instruction_label.text())
        finally:
            self.close_window(form)

    def test_ble_disconnect_and_stale_pipeline_do_not_impersonate_replay(self):
        pipeline = FakePipeline()
        pipeline.stale = True
        pipeline.stream_expected = True
        form = self.make_window(pipeline=pipeline)
        form.training_phase = qt5_training_capture.TrainingCapturePhase.REPLAYING
        form._replay_cancel_event = __import__("threading").Event()
        event = SimpleNamespace(event_type="disconnected", generation=7)
        try:
            with mock.patch.object(qt5_bleak.window, "_on_ble_state"):
                form._on_ble_state(event)
            self.assertEqual(
                form.training_phase, qt5_training_capture.TrainingCapturePhase.REPLAYING
            )
            self.assertFalse(form._replay_cancel_event.is_set())
            form._refresh_live_quality()
            self.assertNotEqual(form.training_phase, qt5_training_capture.TrainingCapturePhase.STOPPING)
            self.assertEqual(form.live_quality_labels["stale"].text(), "回放不适用")
            self.assertIn("非 BLE", form.live_quality_summary.text())
        finally:
            self.close_window(form)

    def test_real_async_replay_e2e_reaches_idle_with_prediction_and_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_derived(root / "legacy", rows=80)
            bundle = root / "bundle"
            shutil.copytree(
                Path(qt5_training_capture.BASE_DIR) / "artifacts" / "model_baseline" / "bundle",
                bundle,
            )
            decision = decision_filter_from_bundle(bundle, hand_side=HandSide.LEFT)
            shared = root / "gui-replay.bin"
            form = self.make_window()
            try:
                form._begin_replay(source, 200.0, decision, shared_file=shared)
                task = form._replay_task
                form.loop.run_until_complete(task)
                self.app.processEvents()
                self.assertIsNone(form._replay_task)
                self.assertIsNone(form._replay_worker_future)
                self.assertNotIn(task, form._ble_ui_tasks)
                self.assertEqual(form.training_phase, qt5_training_capture.TrainingCapturePhase.IDLE)
                self.assertEqual(form.replay_progress.value(), 100)
                self.assertGreater(form._last_replay_report.predictions_emitted, 0)
                with SharedMemoryReader(shared) as reader:
                    frame = reader.read()
                self.assertTrue(frame.flags & FLAG_REPLAY)
                self.assertTrue(frame.flags & FLAG_SYNTHETIC_TIME)
                self.assertTrue(frame.flags & FLAG_DISCONNECTED)
            finally:
                self.close_window(form)

    def test_close_during_real_async_replay_waits_for_worker_and_stops_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_derived(root / "legacy", rows=1000)
            shared = root / "gui-cancel.bin"
            form = self.make_window()
            form._begin_replay(source, 200.0, None, shared_file=shared)
            task = form._replay_task

            async def close_while_running():
                await asyncio.sleep(0.05)
                form._allow_close = True
                form.close()
                await task

            form.loop.run_until_complete(close_while_running())
            self.app.processEvents()
            self.assertTrue(task.done())
            self.assertIsNone(form._replay_worker_future)
            self.assertNotIn(task, form._ble_ui_tasks)
            self.assertEqual(form._last_replay_report.status, "cancelled")
            with SharedMemoryReader(shared) as reader:
                sequence = reader.read().sequence
            form.loop.run_until_complete(asyncio.sleep(0.05))
            with SharedMemoryReader(shared) as reader:
                self.assertEqual(reader.read().sequence, sequence)
            form._quality_timer.stop()
            form.loop.close()

    def test_repeated_base_cancellation_cannot_outlive_replay_worker_or_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_derived(root / "legacy", rows=1000)
            shared = root / "gui-double-cancel.bin"
            form = self.make_window()
            form._begin_replay(source, 200.0, None, shared_file=shared)
            task = form._replay_task

            async def cancel_like_shutdown_entrypoints():
                await asyncio.sleep(0.05)
                form._cancel_replay()
                form._cancel_ble_ui_tasks()
                await asyncio.sleep(0)
                form._cancel_ble_ui_tasks()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            form.loop.run_until_complete(cancel_like_shutdown_entrypoints())
            self.app.processEvents()
            self.assertTrue(task.done())
            self.assertIsNone(form._replay_worker_future)
            self.assertNotIn(task, form._ble_ui_tasks)
            with SharedMemoryReader(shared) as reader:
                terminal = reader.read()
            self.assertTrue(terminal.flags & FLAG_DISCONNECTED)
            # Immediate reacquisition proves the worker closed mmap/file and
            # released the single-producer lock before its wrapper completed.
            with SharedMemoryWriter(shared) as writer:
                writer.write_frame(bytes(range(8)), connection_generation=99)
            form._quality_timer.stop()
            form._allow_close = True
            form.close()
            form.loop.close()


if __name__ == "__main__":
    unittest.main()
