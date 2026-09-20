import asyncio
import concurrent.futures
import inspect
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import qt5_bleak
import device_identity
import shared_memory_v2
from app_config import load_config
from app_logging import EventCode, configure_logging
from emg_protocol import DeviceKey, HandSide
from recording_context import RecordingContext


def recording_context(*, side=HandSide.RIGHT, action="fist"):
    return RecordingContext(
        subject_id="sub-" + "1" * 32,
        hand_side=side,
        action_label=action,
        action_phase="hold",
        experiment_id="discrete_hand_v1",
    )


class FakeProfile:
    def __init__(self, command_error=None):
        self.is_connected = True
        self.is_notifying = False
        self.client_generation = 1
        self.command_error = command_error
        self.calls = []
        self.loops = []

    async def setNotify(self, flag, *, expected_generation=None):
        if expected_generation is not None and expected_generation != self.client_generation:
            raise ConnectionError("stale test generation")
        self.calls.append(("notify", flag))
        self.loops.append(asyncio.get_running_loop())
        changed = self.is_notifying != bool(flag)
        self.is_notifying = bool(flag)
        return changed

    async def setDataType(self, flag, wristband, *, expected_generation=None):
        if expected_generation is not None and expected_generation != self.client_generation:
            raise ConnectionError("stale test generation")
        self.calls.append(("command", flag, wristband))
        self.loops.append(asyncio.get_running_loop())
        if self.command_error:
            raise self.command_error


class FakePipeline:
    def __init__(self):
        self.recorder = None

    def set_recorder(self, recorder, *, recording_context=None):
        self.recorder = recorder
        self.recording_context = recording_context

    def stop_recorder(self, **kwargs):
        self.recorder.close(**kwargs)
        self.recorder = None


class SelectionCombo:
    def __init__(self):
        self.items = []
        self.item_data = []
        self.index = -1

    def clear(self):
        self.items.clear()
        self.item_data.clear()
        self.index = -1

    def addItem(self, value, user_data=None):
        self.items.append(value)
        self.item_data.append(user_data)
        if self.index == -1:
            self.index = 0

    def setCurrentIndex(self, index):
        self.index = index

    def currentIndex(self):
        return self.index

    def currentText(self):
        return self.items[self.index] if self.index >= 0 else ""

    def currentData(self):
        return self.item_data[self.index] if self.index >= 0 else None


class SplitSharedShutdownProbe:
    """Test double for the writer's resource-close/owner-release contract."""

    def __init__(self, release_callback=lambda: None, resource_callback=lambda: None):
        self._release_callback = release_callback
        self._resource_callback = resource_callback

    def close_resources(self):
        return self._resource_callback()

    def release_producer(self):
        return self._release_callback()

    def close(self):
        self.close_resources()
        return self.release_producer()


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_no_temporary_event_loop_or_v1_mmap_remains(self):
        source = inspect.getsource(qt5_bleak)
        self.assertNotIn("new_event_loop", source)
        self.assertNotIn("WG_thread", source)
        self.assertNotIn("emg_shared_data.bin", source)
        self.assertNotIn("mmap", source)
        self.assertIn("emg_shared_data_v2.bin", source)

    def test_device_identifier_maps_to_opaque_stable_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_store = qt5_bleak.DeviceIdentityStore(root / "host1.key", root / "data")
            same_store = qt5_bleak.DeviceIdentityStore(root / "host1.key", root / "data")
            other_store = qt5_bleak.DeviceIdentityStore(root / "host2.key", root / "data")
            first = first_store.device_key("AA:BB:CC:DD:EE:FF")
            second = same_store.device_key("aa:bb:cc:dd:ee:ff")
            self.assertIsInstance(first, DeviceKey)
            self.assertEqual(first, second)
            self.assertNotEqual(first, other_store.device_key("AA:BB:CC:DD:EE:FF"))
            self.assertNotIn("aa:bb", str(first))

    def test_qt_uses_secure_identity_module_and_corruption_fails_closed(self):
        self.assertIs(qt5_bleak.DeviceIdentityStore, device_identity.DeviceIdentityStore)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = qt5_bleak.DeviceIdentityStore(
                identity_path=root / "identity",
                data_root=root / "data",
            )
            store.device_key("AA:BB:CC:DD:EE:FF")
            store.key_path.write_bytes(b"corrupt")
            store.backup_path.write_bytes(b"corrupt")
            with self.assertRaises(device_identity.IdentityIntegrityError):
                qt5_bleak.DeviceIdentityStore(
                    identity_path=root / "identity",
                    data_root=root / "data",
                )

    def test_app_config_raw_audit_maps_all_policy_fields(self):
        config = load_config(qt5_bleak.CONFIG_PATH)
        policy = qt5_bleak._raw_audit_policy(config)
        configured = config.raw_audit
        self.assertEqual(policy.enabled, configured.enabled)
        self.assertEqual(policy.max_bytes, configured.max_bytes)
        self.assertEqual(policy.backup_count, configured.backup_count)
        self.assertEqual(policy.payload_prefix_bytes, configured.payload_prefix_bytes)
        self.assertEqual(policy.record_valid, configured.record_valid)
        self.assertEqual(policy.record_invalid, configured.record_invalid)

    def test_window_passes_validated_packet_protocol_to_pipeline(self):
        config = load_config(qt5_bleak.CONFIG_PATH)
        profile = SimpleNamespace(
            is_connected=False,
            is_notifying=False,
            add_state_listener=lambda listener: None,
            remove_state_listener=lambda listener: None,
            add_notification_listener=lambda listener: None,
            remove_notification_listener=lambda listener: None,
        )
        pipeline = SimpleNamespace(
            start=lambda: None,
            notification_callback=lambda notification: None,
        )
        with mock.patch.object(
            qt5_bleak, "AcquisitionPipeline", return_value=pipeline
        ) as factory:
            form = qt5_bleak.window(
                SimpleNamespace(),
                config=config,
                logging_runtime=SimpleNamespace(
                    get_logger=lambda **kwargs: SimpleNamespace(
                        info=lambda *args, **kwargs: None,
                        warning=lambda *args, **kwargs: None,
                        error=lambda *args, **kwargs: None,
                    ),
                    register_sensitive_token=lambda token: None,
                    shutdown=lambda: None,
                ),
                profile=profile,
                shared_writer=SimpleNamespace(),
                identity_store=SimpleNamespace(),
            )
            self.assertIs(factory.call_args.kwargs["packet_protocol"], config.packet_protocol)
            form._allow_close = True
            form.close()

    def test_notify_success_command_failure_rolls_back_on_same_loop(self):
        profile = FakeProfile(RuntimeError("command failed"))
        target = SimpleNamespace(
            _controls_enabled=True,
            is_recording=False,
            recording_state=qt5_bleak.RecordingState.IDLE,
            _recording_lock=asyncio.Lock(),
            current_session_id=None,
            subject_key="sub-" + "1" * 32,
            _pending_recording_context=recording_context(),
            WG=profile,
            pipeline=FakePipeline(),
            config=SimpleNamespace(
                data=SimpleNamespace(data_path="unused", channels=8),
                to_acquisition_metadata=lambda: None,
            ),
            device_key=DeviceKey("dev-" + "2" * 32),
            _log=lambda *args, **kwargs: None,
            recording_status=SimpleNamespace(setText=lambda value: None),
        )
        current_loop = asyncio.new_event_loop()
        try:
            recorder = SimpleNamespace(
                session_id="session",
                session_dir=Path(tempfile.gettempdir()) / "fake-session",
                close=lambda **kwargs: None,
            )
            with mock.patch.object(qt5_bleak, "DataRecorder", return_value=recorder), mock.patch.object(qt5_bleak.QMessageBox, "critical"):
                current_loop.run_until_complete(qt5_bleak.window._start_recording(target))
        finally:
            current_loop.close()
        self.assertEqual(profile.calls, [("notify", 1), ("command", 1, 0), ("command", 0, 0), ("notify", 0)])
        self.assertEqual(len(set(profile.loops)), 1)
        self.assertFalse(profile.is_notifying)
        self.assertFalse(target.is_recording)

    def test_display_start_preserves_command_error_when_notify_rollback_fails(self):
        primary = RuntimeError("command rejected")
        cleanup = OSError("notify stop failed")

        class Profile:
            is_connected = True
            is_notifying = False

            def __init__(self):
                self.calls = []

            async def setNotify(self, flag):
                self.calls.append(("notify", flag))
                if not flag:
                    raise cleanup
                self.is_notifying = True
                return True

            async def setDataType(self, flag, wristband):
                self.calls.append(("command", flag, wristband))
                raise primary

        profile = Profile()
        target = SimpleNamespace(
            WG=profile,
            device_key=DeviceKey("dev-" + "2" * 32),
            pipeline=SimpleNamespace(queue_depth=0, dropped_count=0),
            _log=lambda *args, **kwargs: None,
        )
        with self.assertRaises(RuntimeError) as caught:
            asyncio.run(qt5_bleak.window._ensure_transmitting(target))
        self.assertIs(caught.exception, primary)
        self.assertEqual(primary.cleanup_failures, (("notify_stop", cleanup),))
        if hasattr(primary, "__notes__"):
            self.assertTrue(any("notify_stop failed" in note for note in primary.__notes__))
        self.assertEqual(
            profile.calls,
            [("notify", 1), ("command", 1, 0), ("notify", 0)],
        )

    def test_display_callback_is_not_gated_by_recording(self):
        emitted = []
        target = SimpleNamespace(wg_show_signal=SimpleNamespace(emit=emitted.append), is_recording=False)
        frame = SimpleNamespace(channel_values=(1, 2, 3, 4, 5, 6, 7, 8))
        qt5_bleak.window._display_frame(target, frame)
        self.assertEqual(emitted, [[1, 2, 3, 4, 5, 6, 7, 8]])

    def test_typed_recording_fault_drives_gui_state_and_keeps_traceback(self):
        logged = []
        target = SimpleNamespace(
            is_recording=True,
            recording_state=qt5_bleak.RecordingState.RECORDING,
            current_session_id="session",
            recording_status=SimpleNamespace(setText=lambda value: logged.append(("ui", value))),
            device_key=DeviceKey("dev-" + "4" * 32),
            _log=lambda *args, **kwargs: logged.append(("log", kwargs)),
        )
        try:
            raise OSError("disk full")
        except OSError as error:
            event = qt5_bleak.PipelineEvent(
                qt5_bleak.PipelineEventType.RECORDING_FAULT,
                "recorder_write",
                7,
                error,
                "trace",
            )
            qt5_bleak.window._handle_pipeline_event(target, event)
        self.assertFalse(target.is_recording)
        self.assertEqual(target.recording_state, qt5_bleak.RecordingState.IDLE)
        self.assertIs(logged[-1][1]["exc_info"][1], event.error)

    def test_shutdown_order_and_repeated_entry_are_idempotent(self):
        events = []
        shutdown_thread = threading.get_ident()

        class Profile:
            is_connected = True
            is_notifying = True

            async def setDataType(self, flag, wristband):
                events.append(("command", flag, wristband))

            async def setNotify(self, flag):
                events.append(("notify", flag))
                self.is_notifying = False

            async def disconnect(self):
                events.append("disconnect")

            def remove_notification_listener(self, listener):
                events.append("remove_notification_listener")

            def remove_state_listener(self, listener):
                events.append("remove_listener")

        class Pipeline:
            is_closed = False

            def stop_accepting(self):
                pass

            def notification_callback(self, notification):
                pass

            def close(self, **kwargs):
                events.append(("pipeline_close", kwargs))
                self.is_closed = True

            def stop_recorder(self, **kwargs):
                events.append(("recorder_close", kwargs))

            def publish_status(self, flags):
                events.append(("status", flags))

        class Shared:
            def close_resources(self):
                pass

            def release_producer(self):
                self.close_thread = threading.get_ident()
                events.append("shared_close")

            def close(self):
                self.close_resources()
                self.release_producer()

        class Runtime:
            def shutdown(self):
                events.append("logging_shutdown")

        target = SimpleNamespace(
            _shutdown_complete=False,
            _shutdown_lock=asyncio.Lock(),
            _recording_lock=asyncio.Lock(),
            _shutdown_stages=set(),
            _last_shutdown_error=None,
            _controls_enabled=True,
            is_recording=True,
            recording_state=qt5_bleak.RecordingState.RECORDING,
            current_session_id="session",
            WG=Profile(),
            pipeline=Pipeline(),
            shared_writer=Shared(),
            logging_runtime=Runtime(),
            _on_ble_state=lambda event: None,
            _log=lambda *args, **kwargs: events.append("log"),
        )
        asyncio.run(qt5_bleak.window.shutdown(target, "test_exit"))
        asyncio.run(qt5_bleak.window.shutdown(target, "test_exit"))
        self.assertEqual(
            events,
            [
                ("command", 0, 0),
                ("notify", 0),
                "remove_notification_listener",
                "disconnect",
                ("pipeline_close", {"drain": True, "timeout": mock.ANY}),
                ("recorder_close", {"complete": False, "error": "test_exit"}),
                ("status", qt5_bleak.FLAG_DISCONNECTED | qt5_bleak.FLAG_STALE),
                "shared_close",
                "remove_listener",
                "log",
                "logging_shutdown",
            ],
        )
        self.assertTrue(target._shutdown_complete)
        self.assertEqual(target.shared_writer.close_thread, shutdown_thread)

    @unittest.skipUnless(os.name == "nt", "Windows mutex ownership is thread-specific")
    def test_shutdown_closes_real_shared_writer_without_mutex_residue(self):
        class Profile:
            is_connected = False
            is_notifying = False

            async def disconnect(self):
                pass

            def remove_notification_listener(self, listener):
                pass

            def remove_state_listener(self, listener):
                pass

        class Pipeline:
            is_closed = False
            notification_callback = staticmethod(lambda notification: None)

            def stop_accepting(self):
                pass

            def close(self, **kwargs):
                self.is_closed = True

            def stop_recorder(self, **kwargs):
                pass

            def publish_status(self, flags):
                pass

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shutdown-shared.bin"
            writer = qt5_bleak.SharedMemoryWriter(path, generation=801)
            target = SimpleNamespace(
                _shutdown_complete=False,
                _shutdown_lock=asyncio.Lock(),
                _recording_lock=asyncio.Lock(),
                _shutdown_stages=set(),
                _last_shutdown_error=None,
                _controls_enabled=True,
                is_recording=False,
                recording_state=qt5_bleak.RecordingState.IDLE,
                current_session_id=None,
                WG=Profile(),
                pipeline=Pipeline(),
                shared_writer=writer,
                logging_runtime=SimpleNamespace(shutdown=lambda: None),
                _on_ble_state=lambda event: None,
                _log=lambda *args, **kwargs: None,
            )

            asyncio.run(qt5_bleak.window.shutdown(target, "test_exit"))

            self.assertTrue(target._shutdown_complete)
            self.assertIsNone(writer._producer_token)
            with qt5_bleak.SharedMemoryWriter(path, generation=802) as replacement:
                replacement.write_frame(b"12345678")

    @unittest.skipUnless(os.name == "nt", "native release assertions are Windows-specific")
    def test_shutdown_real_writer_retries_native_release_failure(self):
        class Profile:
            is_connected = False
            is_notifying = False

            async def disconnect(self):
                pass

            def remove_notification_listener(self, listener):
                pass

            def remove_state_listener(self, listener):
                pass

        class Pipeline:
            is_closed = False
            notification_callback = staticmethod(lambda notification: None)

            def stop_accepting(self):
                pass

            def close(self, **kwargs):
                self.is_closed = True

            def stop_recorder(self, **kwargs):
                pass

            def publish_status(self, flags):
                pass

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shutdown-native-retry.bin"
            writer = qt5_bleak.SharedMemoryWriter(path, generation=803)
            target = SimpleNamespace(
                _shutdown_complete=False,
                _shutdown_lock=asyncio.Lock(),
                _recording_lock=asyncio.Lock(),
                _shutdown_stages=set(),
                _shutdown_blocking_work={},
                _last_shutdown_error=None,
                _controls_enabled=True,
                is_recording=False,
                recording_state=qt5_bleak.RecordingState.IDLE,
                current_session_id=None,
                WG=Profile(),
                pipeline=Pipeline(),
                shared_writer=writer,
                logging_runtime=SimpleNamespace(shutdown=lambda: None),
                _on_ble_state=lambda event: None,
                _log=lambda *args, **kwargs: None,
            )
            real_release = shared_memory_v2._release_producer
            attempts = []

            def fail_once(token):
                attempts.append(token)
                if len(attempts) == 1:
                    raise OSError("ReleaseMutex injected failure")
                return real_release(token)

            with mock.patch(
                "shared_memory_v2._release_producer", side_effect=fail_once
            ):
                with self.assertRaises(qt5_bleak.ShutdownError):
                    asyncio.run(qt5_bleak.window.shutdown(target, "test_exit"))
                self.assertFalse(target._shutdown_complete)
                self.assertIsNotNone(writer._producer_token)
                self.assertNotIn("shared_close", target._shutdown_stages)

                asyncio.run(qt5_bleak.window.shutdown(target, "test_exit"))

            self.assertTrue(target._shutdown_complete)
            self.assertIsNone(writer._producer_token)
            self.assertEqual(len(attempts), 2)
            self.assertIs(attempts[0], attempts[1])
            with qt5_bleak.SharedMemoryWriter(path, generation=804) as replacement:
                replacement.write_frame(b"12345678")

    def test_blocked_shared_resource_close_honors_deadline_and_defers_owner_release(self):
        resource_started = threading.Event()
        resource_release = threading.Event()
        release_threads = []

        class Profile:
            is_connected = False
            is_notifying = False

            async def disconnect(self):
                pass

            def remove_notification_listener(self, listener):
                pass

            def remove_state_listener(self, listener):
                pass

        class Pipeline:
            is_closed = False
            notification_callback = staticmethod(lambda notification: None)

            def stop_accepting(self):
                pass

            def close(self, **kwargs):
                self.is_closed = True

            def stop_recorder(self, **kwargs):
                pass

            def publish_status(self, flags):
                pass

        def block_resource_close():
            resource_started.set()
            resource_release.wait()

        owner_thread = threading.get_ident()
        shared = SplitSharedShutdownProbe(
            lambda: release_threads.append(threading.get_ident()),
            block_resource_close,
        )
        target = SimpleNamespace(
            _shutdown_complete=False,
            _shutdown_lock=asyncio.Lock(),
            _recording_lock=asyncio.Lock(),
            _shutdown_stages=set(),
            _shutdown_blocking_work={},
            _last_shutdown_error=None,
            _controls_enabled=True,
            is_recording=False,
            recording_state=qt5_bleak.RecordingState.IDLE,
            current_session_id=None,
            WG=Profile(),
            pipeline=Pipeline(),
            shared_writer=shared,
            logging_runtime=SimpleNamespace(shutdown=lambda: None),
            _on_ble_state=lambda event: None,
            _log=lambda *args, **kwargs: None,
        )

        started = time.monotonic()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with self.assertRaises(qt5_bleak.ShutdownError) as raised:
                loop.run_until_complete(
                    qt5_bleak.window.shutdown(
                        target, "test_exit", deadline=time.monotonic() + 0.08
                    )
                )
            elapsed = time.monotonic() - started

            self.assertTrue(resource_started.is_set())
            self.assertLess(elapsed, 0.25)
            self.assertIn(
                "shared_resources_close",
                [name for name, _ in raised.exception.failures],
            )
            self.assertNotIn("shared_resources_close", target._shutdown_stages)
            self.assertNotIn("shared_close", target._shutdown_stages)
            self.assertEqual(release_threads, [])
            self.assertTrue(qt5_bleak._active_blocking_shutdown_work(target))

            resource_release.set()
            for work in target._shutdown_blocking_work.values():
                self.assertTrue(work.completed.wait(1.0))
            loop.run_until_complete(qt5_bleak.window.shutdown(target, "test_exit"))
            self.assertEqual(release_threads, [owner_thread])
            self.assertTrue(target._shutdown_complete)
        finally:
            resource_release.set()
            loop.run_until_complete(asyncio.sleep(0))
            loop.close()
            asyncio.set_event_loop(None)

    def test_shutdown_retries_failed_producer_release_without_false_completion(self):
        release_attempts = []

        class Profile:
            is_connected = False
            is_notifying = False

            async def disconnect(self):
                pass

            def remove_notification_listener(self, listener):
                pass

            def remove_state_listener(self, listener):
                pass

        class Pipeline:
            is_closed = False
            notification_callback = staticmethod(lambda notification: None)

            def stop_accepting(self):
                pass

            def close(self, **kwargs):
                self.is_closed = True

            def stop_recorder(self, **kwargs):
                pass

            def publish_status(self, flags):
                pass

        class Shared:
            resource_close_calls = 0

            def close_resources(self):
                self.resource_close_calls += 1

            def release_producer(self):
                release_attempts.append(threading.get_ident())
                if len(release_attempts) == 1:
                    raise OSError("ReleaseMutex failed")

        owner_thread = threading.get_ident()
        shared = Shared()
        target = SimpleNamespace(
            _shutdown_complete=False,
            _shutdown_lock=asyncio.Lock(),
            _recording_lock=asyncio.Lock(),
            _shutdown_stages=set(),
            _shutdown_blocking_work={},
            _last_shutdown_error=None,
            _controls_enabled=True,
            is_recording=False,
            recording_state=qt5_bleak.RecordingState.IDLE,
            current_session_id=None,
            WG=Profile(),
            pipeline=Pipeline(),
            shared_writer=shared,
            logging_runtime=SimpleNamespace(shutdown=lambda: None),
            _on_ble_state=lambda event: None,
            _log=lambda *args, **kwargs: None,
        )

        with self.assertRaises(qt5_bleak.ShutdownError) as raised:
            asyncio.run(qt5_bleak.window.shutdown(target, "test_exit"))
        self.assertIn("shared_close", [name for name, _ in raised.exception.failures])
        self.assertFalse(target._shutdown_complete)
        self.assertIn("shared_resources_close", target._shutdown_stages)
        self.assertNotIn("shared_close", target._shutdown_stages)

        asyncio.run(qt5_bleak.window.shutdown(target, "test_exit"))
        self.assertTrue(target._shutdown_complete)
        self.assertEqual(shared.resource_close_calls, 1)
        self.assertEqual(release_attempts, [owner_thread, owner_thread])

    def test_shutdown_invalidates_and_waits_for_ui_ble_task_before_teardown(self):
        events = []

        class Profile:
            is_connected = False
            is_notifying = False

            async def disconnect(self):
                events.append("disconnect")

            def remove_notification_listener(self, listener):
                events.append("remove_notification_listener")

            def remove_state_listener(self, listener):
                events.append("remove_state_listener")

        class Pipeline:
            is_closed = False

            def stop_accepting(self):
                pass

            def notification_callback(self, notification):
                pass

            def close(self, **kwargs):
                events.append("pipeline_close")
                self.is_closed = True

            def stop_recorder(self, **kwargs):
                events.append("recorder_close")

            def publish_status(self, flags):
                events.append("status")

        target = SimpleNamespace(
            _shutdown_complete=False,
            _shutdown_lock=asyncio.Lock(),
            _recording_lock=asyncio.Lock(),
            _shutdown_stages=set(),
            _last_shutdown_error=None,
            _controls_enabled=True,
            _scan_epoch=1,
            _completed_scan_epoch=1,
            _scan_snapshot=object(),
            _connect_epoch=1,
            _accepted_connection_generation=1,
            _ble_ui_tasks=set(),
            is_recording=False,
            recording_state=qt5_bleak.RecordingState.IDLE,
            current_session_id=None,
            WG=Profile(),
            pipeline=Pipeline(),
            shared_writer=SplitSharedShutdownProbe(lambda: events.append("shared_close")),
            logging_runtime=SimpleNamespace(shutdown=lambda: events.append("logging")),
            _on_ble_state=lambda event: None,
            _log=lambda *args, **kwargs: events.append("log"),
        )

        async def exercise():
            async def pending_ui_operation():
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    events.append("ui_task_cancelled")
                    if target._controls_enabled:
                        events.append("late_ui_publish")
                    raise

            task = asyncio.create_task(pending_ui_operation())
            target._ble_ui_tasks.add(task)
            await asyncio.sleep(0)
            await qt5_bleak.window.shutdown(target, "test")

        asyncio.run(exercise())
        self.assertEqual(events[0], "ui_task_cancelled")
        self.assertNotIn("late_ui_publish", events)
        self.assertIsNone(target._scan_snapshot)
        self.assertIsNone(target._accepted_connection_generation)

    def test_shutdown_failure_does_not_block_independent_teardown_and_retries(self):
        events = []

        class Profile:
            is_connected = True
            is_notifying = True

            async def setDataType(self, flag, wristband):
                events.append("device_stop")

            async def setNotify(self, flag):
                events.append("notify_stop")
                self.is_notifying = False

            async def disconnect(self):
                events.append("disconnect")

            def remove_notification_listener(self, listener):
                events.append("notification_listener_remove")

            def remove_state_listener(self, listener):
                events.append("listener_remove")

        class Pipeline:
            attempts = 0
            is_closed = False

            def stop_accepting(self):
                pass

            def notification_callback(self, notification):
                pass

            def close(self, **kwargs):
                self.attempts += 1
                events.append(f"consumer_{self.attempts}")
                if self.attempts == 1:
                    raise TimeoutError("worker busy")
                self.is_closed = True

            def stop_recorder(self, **kwargs):
                events.append("recorder")

            def publish_status(self, flags):
                events.append("status")

        target = SimpleNamespace(
            _shutdown_complete=False,
            _shutdown_lock=asyncio.Lock(),
            _recording_lock=asyncio.Lock(),
            _shutdown_stages=set(),
            _last_shutdown_error=None,
            _controls_enabled=True,
            is_recording=True,
            recording_state=qt5_bleak.RecordingState.RECORDING,
            current_session_id="session",
            WG=Profile(),
            pipeline=Pipeline(),
            shared_writer=SplitSharedShutdownProbe(lambda: events.append("shared")),
            logging_runtime=SimpleNamespace(shutdown=lambda: events.append("logging")),
            _on_ble_state=lambda event: None,
            _log=lambda *args, **kwargs: events.append("log"),
        )
        with self.assertRaises(qt5_bleak.ShutdownError):
            asyncio.run(qt5_bleak.window.shutdown(target, "test"))
        self.assertEqual(
            events,
            [
                "device_stop",
                "notify_stop",
                "notification_listener_remove",
                "disconnect",
                "consumer_1",
                "listener_remove",
                "log",
                "logging",
            ],
        )
        asyncio.run(qt5_bleak.window.shutdown(target, "test"))
        self.assertEqual(events[-4:], ["consumer_2", "recorder", "status", "shared"])
        self.assertTrue(target._shutdown_complete)

    def test_shutdown_aggregates_early_failures_and_still_attempts_all_cleanup(self):
        events = []

        class Profile:
            is_connected = True
            is_notifying = True

            async def setDataType(self, flag, wristband):
                events.append("device_stop")
                raise OSError("device failed")

            async def setNotify(self, flag):
                events.append("notify_stop")
                raise OSError("notify failed")

            async def disconnect(self):
                events.append("disconnect")

            def remove_notification_listener(self, listener):
                events.append("notification_listener_remove")

            def remove_state_listener(self, listener):
                events.append("listener_remove")

        class Pipeline:
            is_closed = False

            def stop_accepting(self):
                pass

            def notification_callback(self, notification):
                pass

            def close(self, **kwargs):
                events.append("consumer")
                raise TimeoutError("consumer failed")

            def stop_recorder(self, **kwargs):
                events.append("recorder")
                raise OSError("recorder failed")

            def publish_status(self, flags):
                events.append("status")

        target = SimpleNamespace(
            _shutdown_complete=False,
            _shutdown_lock=asyncio.Lock(),
            _recording_lock=asyncio.Lock(),
            _shutdown_stages=set(),
            _last_shutdown_error=None,
            _controls_enabled=True,
            _ble_ui_tasks=set(),
            is_recording=True,
            recording_state=qt5_bleak.RecordingState.RECORDING,
            current_session_id="session",
            WG=Profile(),
            pipeline=Pipeline(),
            shared_writer=SplitSharedShutdownProbe(lambda: events.append("shared")),
            logging_runtime=SimpleNamespace(shutdown=lambda: events.append("logging")),
            _on_ble_state=lambda event: None,
            _log=lambda *args, **kwargs: events.append("log"),
        )

        with self.assertRaises(qt5_bleak.ShutdownError) as raised:
            asyncio.run(qt5_bleak.window.shutdown(target, "test"))

        self.assertEqual(
            [name for name, _ in raised.exception.failures],
            [
                "device_stop",
                "notify_stop",
                "consumer_stop",
                "forced_boundary.consumer_active",
            ],
        )
        self.assertEqual(
            events,
            [
                "device_stop",
                "notify_stop",
                "notification_listener_remove",
                "disconnect",
                "consumer",
                "listener_remove",
                "log",
                "logging",
            ],
        )
        self.assertTrue(target.is_recording)
        self.assertEqual(target.recording_state, qt5_bleak.RecordingState.RECORDING)
        self.assertEqual(target.current_session_id, "session")

    def test_shutdown_cancels_and_drains_ui_task_before_cleanup(self):
        events = []

        class Profile:
            is_connected = False
            is_notifying = False

            async def disconnect(self):
                events.append("disconnect")

            def remove_notification_listener(self, listener):
                events.append("notification_listener_remove")

            def remove_state_listener(self, listener):
                events.append("listener_remove")

        class Pipeline:
            is_closed = False

            def stop_accepting(self):
                pass

            def notification_callback(self, notification):
                pass

            def close(self, **kwargs):
                events.append("consumer")
                self.is_closed = True

            def stop_recorder(self, **kwargs):
                events.append("recorder")

            def publish_status(self, flags):
                events.append("status")

        target = SimpleNamespace(
            _shutdown_complete=False,
            _shutdown_lock=asyncio.Lock(),
            _recording_lock=asyncio.Lock(),
            _shutdown_stages=set(),
            _last_shutdown_error=None,
            _controls_enabled=True,
            _ble_ui_tasks=set(),
            is_recording=False,
            recording_state=qt5_bleak.RecordingState.IDLE,
            current_session_id=None,
            WG=Profile(),
            pipeline=Pipeline(),
            shared_writer=SplitSharedShutdownProbe(lambda: events.append("shared")),
            logging_runtime=SimpleNamespace(shutdown=lambda: events.append("logging")),
            _on_ble_state=lambda event: None,
            _log=lambda *args, **kwargs: events.append("log"),
        )

        async def exercise():
            async def resistant_operation():
                await asyncio.Event().wait()

            task = asyncio.create_task(resistant_operation())
            target._ble_ui_tasks.add(task)
            await asyncio.sleep(0)
            with mock.patch.object(qt5_bleak, "BLE_UI_TASK_WAIT_TIMEOUT_SECONDS", 0.01):
                await asyncio.wait_for(
                    qt5_bleak.window.shutdown(target, "test"), timeout=0.2
                )
            self.assertTrue(task.done())

        asyncio.run(exercise())
        self.assertIn("disconnect", events)
        self.assertIn("shared", events)
        self.assertIn("logging", events)

    def test_wait_ble_ui_tasks_times_out_then_cancels_and_drains(self):
        async def exercise():
            task = asyncio.create_task(asyncio.Event().wait())
            target = SimpleNamespace(_ble_ui_tasks={task})
            with mock.patch.object(qt5_bleak, "BLE_UI_TASK_WAIT_TIMEOUT_SECONDS", 0.05):
                with self.assertRaisesRegex(TimeoutError, "BLE UI tasks"):
                    await asyncio.wait_for(
                        qt5_bleak.window._wait_ble_ui_tasks(target), timeout=0.2
                    )
            self.assertTrue(task.done())

        asyncio.run(exercise())

    def test_shutdown_async_stage_timeout_covers_recording_lock_and_continues(self):
        events = []

        class Profile:
            is_connected = False
            is_notifying = False

            async def disconnect(self):
                events.append("disconnect")

            def remove_notification_listener(self, listener):
                events.append("notification_listener_remove")

            def remove_state_listener(self, listener):
                events.append("listener_remove")

        class Pipeline:
            notification_callback = staticmethod(lambda notification: None)

            is_closed = False

            def stop_accepting(self):
                pass

            def close(self, **kwargs):
                events.append("consumer")
                self.is_closed = True
            stop_recorder = lambda self, **kwargs: events.append("recorder")
            publish_status = lambda self, flags: events.append("status")

        target = SimpleNamespace(
            _shutdown_complete=False,
            _shutdown_lock=asyncio.Lock(),
            _recording_lock=asyncio.Lock(),
            _shutdown_stages=set(),
            _last_shutdown_error=None,
            _controls_enabled=True,
            _ble_ui_tasks=set(),
            is_recording=False,
            recording_state=qt5_bleak.RecordingState.IDLE,
            current_session_id=None,
            WG=Profile(),
            pipeline=Pipeline(),
            shared_writer=SplitSharedShutdownProbe(lambda: events.append("shared")),
            logging_runtime=SimpleNamespace(shutdown=lambda: events.append("logging")),
            _on_ble_state=lambda event: None,
            _log=lambda *args, **kwargs: events.append("log"),
        )

        async def exercise():
            await target._recording_lock.acquire()
            with self.assertRaises(qt5_bleak.ShutdownError) as raised:
                await qt5_bleak.window.shutdown(
                    target, "test", deadline=time.monotonic() + 0.05
                )
            self.assertIn("recording_operation", [name for name, _ in raised.exception.failures])
            target._recording_lock.release()

        asyncio.run(exercise())
        self.assertIn("disconnect", events)

    def test_cancelled_shutdown_finishes_best_effort_then_reraises_cancel(self):
        events = []
        device_started = None

        class Profile:
            is_connected = True
            is_notifying = False

            async def setDataType(self, flag, wristband):
                device_started.set()
                await asyncio.sleep(0.01)
                events.append("device_stop")

            async def disconnect(self):
                events.append("disconnect")

            def remove_notification_listener(self, listener):
                events.append("notification_listener_remove")

            def remove_state_listener(self, listener):
                events.append("listener_remove")

        class Pipeline:
            notification_callback = staticmethod(lambda notification: None)
            is_closed = False

            def stop_accepting(self):
                pass

            def close(self, **kwargs):
                events.append("consumer")
                raise OSError("consumer cleanup failed")

            stop_recorder = lambda self, **kwargs: events.append("recorder")
            publish_status = lambda self, flags: events.append("status")

        target = SimpleNamespace(
            _shutdown_complete=False,
            _shutdown_lock=asyncio.Lock(),
            _recording_lock=asyncio.Lock(),
            _shutdown_stages=set(),
            _last_shutdown_error=None,
            _controls_enabled=True,
            _ble_ui_tasks=set(),
            is_recording=False,
            recording_state=qt5_bleak.RecordingState.IDLE,
            current_session_id=None,
            WG=Profile(),
            pipeline=Pipeline(),
            shared_writer=SplitSharedShutdownProbe(lambda: events.append("shared")),
            logging_runtime=SimpleNamespace(shutdown=lambda: events.append("logging")),
            _on_ble_state=lambda event: None,
            _log=lambda *args, **kwargs: events.append("log"),
        )

        async def exercise():
            nonlocal device_started
            device_started = asyncio.Event()
            task = asyncio.create_task(qt5_bleak.window.shutdown(target, "test"))
            await device_started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=0.3)

        asyncio.run(exercise())
        self.assertIn("device_stop", events)
        self.assertIn("disconnect", events)
        self.assertIn("logging", events)
        self.assertIsInstance(target._last_shutdown_error, asyncio.CancelledError)
        failure_names = [
            name for name, _ in target._last_shutdown_error.cleanup_failures
        ]
        self.assertIn("consumer_stop", failure_names)
        self.assertIn("forced_boundary.consumer_active", failure_names)

    def test_shutdown_uses_one_absolute_deadline_for_all_async_stages(self):
        class Profile:
            is_connected = True
            is_notifying = True

            async def setDataType(self, flag, wristband):
                await asyncio.Event().wait()

            async def setNotify(self, flag):
                await asyncio.Event().wait()

            async def disconnect(self, *, deadline):
                await asyncio.Event().wait()

            def remove_notification_listener(self, listener):
                pass

            def remove_state_listener(self, listener):
                pass

        class Pipeline:
            notification_callback = staticmethod(lambda notification: None)
            is_closed = False

            def stop_accepting(self):
                pass

            def close(self, **kwargs):
                self.is_closed = True
            stop_recorder = lambda self, **kwargs: None
            publish_status = lambda self, flags: None

        target = SimpleNamespace(
            _shutdown_complete=False,
            _shutdown_lock=asyncio.Lock(),
            _recording_lock=asyncio.Lock(),
            _shutdown_stages=set(),
            _last_shutdown_error=None,
            _controls_enabled=True,
            _ble_ui_tasks=set(),
            is_recording=False,
            recording_state=qt5_bleak.RecordingState.IDLE,
            current_session_id=None,
            WG=Profile(),
            pipeline=Pipeline(),
            shared_writer=SplitSharedShutdownProbe(),
            logging_runtime=SimpleNamespace(shutdown=lambda: None),
            _on_ble_state=lambda event: None,
            _log=lambda *args, **kwargs: None,
        )

        async def exercise():
            deadline = time.monotonic() + 0.05
            started = time.monotonic()
            with self.assertRaises(qt5_bleak.ShutdownError):
                await qt5_bleak.window.shutdown(target, "test", deadline=deadline)
            return time.monotonic() - started

        elapsed = asyncio.run(exercise())
        self.assertLess(elapsed, 0.15)

    def test_consumer_holding_real_fanout_lock_keeps_sinks_open(self):
        import threading

        class Shared:
            generation = 1

            def __init__(self):
                self.closed = False

            def write_frame(self, *args, **kwargs):
                pass

            def close(self):
                self.closed = True

        class Recorder:
            session_dir = Path(tempfile.gettempdir())

            def __init__(self):
                self.entered = threading.Event()
                self.release = threading.Event()
                self.closed = False

            def record(self, frame):
                self.entered.set()
                self.release.wait()

            def close(self, **kwargs):
                self.closed = True

        class Profile:
            is_connected = False
            is_notifying = False

            async def disconnect(self, *, deadline):
                pass

            def remove_notification_listener(self, listener):
                pass

            def remove_state_listener(self, listener):
                pass

        shared = Shared()
        recorder = Recorder()
        pipeline = qt5_bleak.AcquisitionPipeline(shared, queue_capacity=4)
        pipeline.set_recorder(recorder, recording_context=recording_context())
        pipeline.start()
        pipeline.notification_callback(
            qt5_bleak.bleak_ble.BleNotification(None, bytes(16), 1)
        )
        self.assertTrue(recorder.entered.wait(1.0))
        target = SimpleNamespace(
            _shutdown_complete=False,
            _shutdown_lock=asyncio.Lock(),
            _recording_lock=asyncio.Lock(),
            _shutdown_stages=set(),
            _last_shutdown_error=None,
            _controls_enabled=True,
            _ble_ui_tasks=set(),
            is_recording=True,
            recording_state=qt5_bleak.RecordingState.RECORDING,
            current_session_id="active",
            WG=Profile(),
            pipeline=pipeline,
            shared_writer=shared,
            logging_runtime=SimpleNamespace(shutdown=lambda: None),
            _on_ble_state=lambda event: None,
            _log=lambda *args, **kwargs: None,
        )

        async def exercise():
            with self.assertRaises(qt5_bleak.ShutdownError) as raised:
                await qt5_bleak.window.shutdown(
                    target, "test", deadline=time.monotonic() + 0.05
                )
            return [name for name, _ in raised.exception.failures]

        failures = asyncio.run(exercise())
        self.assertIn("consumer_stop", failures)
        self.assertIn("forced_boundary.consumer_active", failures)
        self.assertFalse(recorder.closed)
        self.assertFalse(shared.closed)
        recorder.release.set()
        pipeline._thread.join(0.5)
        pipeline.stop_recorder(complete=False, error="test cleanup")
        shared.close()

    def test_missing_pipeline_shutdown_protocol_never_closes_active_sinks(self):
        events = []

        class Pipeline:
            notification_callback = staticmethod(lambda notification: None)

            def close(self):
                events.append("consumer_close")

            def stop_recorder(self, **kwargs):
                events.append("recorder_close")

            def publish_status(self, flags):
                events.append("status")

        class Profile:
            is_connected = False
            is_notifying = False

            async def disconnect(self, *, deadline):
                events.append("disconnect")

            def remove_notification_listener(self, listener):
                events.append("notification_listener_remove")

            def remove_state_listener(self, listener):
                events.append("listener_remove")

        target = SimpleNamespace(
            _shutdown_complete=False,
            _shutdown_lock=asyncio.Lock(),
            _recording_lock=asyncio.Lock(),
            _shutdown_stages=set(),
            _last_shutdown_error=None,
            _controls_enabled=True,
            _ble_ui_tasks=set(),
            is_recording=True,
            recording_state=qt5_bleak.RecordingState.RECORDING,
            current_session_id="active",
            WG=Profile(),
            pipeline=Pipeline(),
            shared_writer=SplitSharedShutdownProbe(lambda: events.append("shared_close")),
            logging_runtime=SimpleNamespace(shutdown=lambda: events.append("logging")),
            _on_ble_state=lambda event: None,
            _log=lambda *args, **kwargs: events.append("log"),
        )

        with self.assertRaises(qt5_bleak.ShutdownError) as raised:
            asyncio.run(qt5_bleak.window.shutdown(target, "test"))

        failure_names = [name for name, _ in raised.exception.failures]
        self.assertIn("pipeline_protocol", failure_names)
        self.assertIn("forced_boundary.consumer_active", failure_names)
        self.assertNotIn("consumer_close", events)
        self.assertNotIn("recorder_close", events)
        self.assertNotIn("status", events)
        self.assertNotIn("shared_close", events)
        self.assertIn("disconnect", events)
        self.assertIn("logging", events)

    def test_blocking_shutdown_registry_reuses_active_stage_work(self):
        async def exercise():
            target = SimpleNamespace(_shutdown_blocking_work={})
            entered = threading.Event()
            release = threading.Event()
            calls = []

            def blocking_operation():
                calls.append("started")
                entered.set()
                release.wait()

            first = asyncio.create_task(
                qt5_bleak.window._run_shutdown_blocking_work(
                    target, "consumer_stop", blocking_operation
                )
            )
            await asyncio.to_thread(entered.wait, 1.0)
            second = asyncio.create_task(
                qt5_bleak.window._run_shutdown_blocking_work(
                    target, "consumer_stop", blocking_operation
                )
            )
            await asyncio.sleep(0.02)
            self.assertEqual(calls, ["started"])
            release.set()
            await asyncio.gather(first, second)

        import threading
        asyncio.run(exercise())

    def test_blocking_registry_submission_failure_is_atomic(self):
        loop = asyncio.new_event_loop()
        target = SimpleNamespace(_shutdown_blocking_work={})
        try:
            with mock.patch.object(
                qt5_bleak._SHUTDOWN_EXECUTOR,
                "submit",
                side_effect=RuntimeError("Executor shutdown"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Executor shutdown"):
                    loop.run_until_complete(
                        qt5_bleak.window._run_shutdown_blocking_work(
                            target, "consumer_stop", lambda: None
                        )
                    )
            self.assertEqual(target._shutdown_blocking_work, {})
            self.assertEqual(qt5_bleak._active_blocking_shutdown_work(target), ())
        finally:
            loop.close()

    def test_inventory_rescans_tasks_spawned_while_handling_cancellation(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        child_tasks = []

        class Form:
            _shutdown_pending_tasks = set()
            _ble_ui_tasks = set()
            _shutdown_blocking_work = {}
            WG = SimpleNamespace(_pending_disconnect_task=None)

        form = Form()

        async def parent_task():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                child = asyncio.create_task(asyncio.Event().wait())
                child_tasks.append(child)
                form._shutdown_pending_tasks.add(child)
                raise

        form._shutdown_task = loop.create_task(parent_task())
        try:
            loop.run_until_complete(asyncio.sleep(0))
            pending, work, stable = qt5_bleak._drain_application_inventory(
                loop, form, time.monotonic() + 0.1
            )
            self.assertEqual(pending, ())
            self.assertEqual(work, ())
            self.assertTrue(stable)
            self.assertEqual(len(child_tasks), 1)
            self.assertTrue(child_tasks[0].done())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_stale_disconnect_generation_cannot_reset_active_session(self):
        events = []
        target = SimpleNamespace(
            _controls_enabled=True,
            _accepted_connection_generation=9,
            _recording_start_token=object(),
            is_recording=True,
            recording_stop_pending={"generation": 9, "stage": "device_stop"},
            _recorder_stop_token=None,
            _unattached_recorder=None,
            _recording_lock=asyncio.Lock(),
            _pipeline_cleanup_tasks=set(),
            _active_recording_context=None,
            _active_session_dir=None,
            recording_state=qt5_bleak.RecordingState.RECORDING,
            current_session_id="active-session",
            recording_status=SimpleNamespace(
                setText=lambda value: events.append(("status", value))
            ),
            pipeline=SimpleNamespace(
                mark_disconnected=lambda reason: events.append(("pipeline", reason))
            ),
            device_key=DeviceKey("dev-" + "9" * 32),
            _log=lambda *args, **kwargs: events.append(("log", args)),
        )

        qt5_bleak.window._on_ble_state(
            target,
            qt5_bleak.bleak_ble.BleStateEvent(
                "disconnected", False, False, 8, "unexpected"
            ),
        )

        self.assertEqual(events, [])
        self.assertTrue(target.is_recording)
        self.assertEqual(
            target.recording_stop_pending,
            {"generation": 9, "stage": "device_stop"},
        )
        self.assertEqual(target.current_session_id, "active-session")
        self.assertEqual(target._accepted_connection_generation, 9)

        async def current_disconnect():
            qt5_bleak.window._on_ble_state(
                target,
                qt5_bleak.bleak_ble.BleStateEvent(
                    "disconnected", False, False, 9, "unexpected"
                ),
            )
            await qt5_bleak.window._wait_pipeline_cleanup_tasks(target)

        asyncio.run(current_disconnect())
        self.assertFalse(target.is_recording)
        self.assertIsNone(target.recording_stop_pending)
        self.assertIsNone(target.current_session_id)
        self.assertIsNone(target._accepted_connection_generation)
        self.assertTrue(any(event[0] == "pipeline" for event in events))

    def test_real_window_constructs_headlessly_with_injected_resources(self):
        class Runtime:
            def get_logger(self, **kwargs):
                return SimpleNamespace(
                    info=lambda message, **kwargs: None,
                    warning=lambda message, **kwargs: None,
                    error=lambda message, **kwargs: None,
                )

            def register_sensitive_token(self, token):
                pass

            def shutdown(self):
                pass

        class Profile:
            is_connected = False
            is_notifying = False

            def add_state_listener(self, listener):
                self.listener = listener

            def remove_state_listener(self, listener):
                pass

            def add_notification_listener(self, handler):
                self.handler = handler

            def remove_notification_listener(self, handler):
                pass

        class Pipeline:
            def start(self):
                pass

            def notification_callback(self, notification):
                pass

        form = qt5_bleak.window(
            SimpleNamespace(),
            config=load_config(qt5_bleak.CONFIG_PATH),
            logging_runtime=Runtime(),
            profile=Profile(),
            shared_writer=SimpleNamespace(),
            pipeline=Pipeline(),
            identity_store=SimpleNamespace(),
        )
        form._allow_close = True
        form.close()

    def test_main_workflow_keeps_all_ble_operations_on_one_loop(self):
        class Combo:
            def __init__(self):
                self.items = []

            def clear(self):
                self.items.clear()

            def addItem(self, value, user_data=None):
                self.items.append(value)

            def currentText(self):
                return self.items[0]

        class Profile:
            is_connected = False
            is_notifying = False

            def __init__(self):
                self.loops = []

            def _track(self):
                self.loops.append(asyncio.get_running_loop())

            async def scan(self, timeout):
                self._track()
                entry = [1, "bracelet", "AA:BB:CC:DD:EE:FF", -40]
                return entry, [entry]

            async def connect(self, address):
                self._track()
                self.is_connected = True
                return True

            async def setNotify(self, flag):
                self._track()
                changed = self.is_notifying != bool(flag)
                self.is_notifying = bool(flag)
                return changed

            async def setDataType(self, flag, wristband):
                self._track()

            async def disconnect(self):
                self._track()
                self.is_connected = False

            def remove_notification_listener(self, listener):
                pass

            def remove_state_listener(self, listener):
                pass

        class Runtime:
            def register_sensitive_token(self, token):
                pass

            def shutdown(self):
                pass

        class Pipeline:
            queue_depth = 0
            dropped_count = 0
            is_closed = False

            def stop_accepting(self):
                pass

            def notification_callback(self, notification):
                pass

            def close(self, **kwargs):
                self.is_closed = True

            def stop_recorder(self, **kwargs):
                pass

            def publish_status(self, flags):
                pass

        profile = Profile()
        target = SimpleNamespace(
            _controls_enabled=True,
            _shutdown_complete=False,
            is_recording=False,
            WG=profile,
            device_combo=Combo(),
            devices_list=[],
            WG_searched_devices=(None, []),
            logging_runtime=Runtime(),
            identity_store=SimpleNamespace(
                device_key=lambda identifier: DeviceKey("dev-" + "3" * 32)
            ),
            device_key=DeviceKey("dev-" + "0" * 32),
            _log=lambda *args, **kwargs: None,
            pipeline=Pipeline(),
            shared_writer=SplitSharedShutdownProbe(),
            _on_ble_state=lambda event: None,
            _shutdown_lock=asyncio.Lock(),
            _recording_lock=asyncio.Lock(),
            _shutdown_stages=set(),
            _last_shutdown_error=None,
            recording_state=qt5_bleak.RecordingState.IDLE,
            current_session_id=None,
        )

        async def workflow():
            await qt5_bleak.window.search_click.__wrapped__(target)
            with mock.patch.object(qt5_bleak.QMessageBox, "information"):
                await qt5_bleak.window.connect_semg_click.__wrapped__(target)
            self.assertTrue(await qt5_bleak.window._ensure_transmitting(target))
            await qt5_bleak.window.shutdown(target, "test")

        asyncio.run(workflow())
        self.assertGreaterEqual(len(profile.loops), 6)
        self.assertEqual(len(set(profile.loops)), 1)

    def test_search_selects_scan_confirmed_target_at_any_position(self):
        class Combo:
            def __init__(self):
                self.items = []

            def clear(self):
                self.items.clear()

            def addItem(self, value, user_data=None):
                self.items.append(value)

        for target_position in (0, 1, 2):
            with self.subTest(target_position=target_position):
                devices = [
                    [1, "other", "00:00:00:00:00:01", -70],
                    [2, "bracelet", "00:00:00:00:00:02", -40],
                    [3, "other", "00:00:00:00:00:03", -80],
                ]

                class Profile:
                    async def scan(self, timeout):
                        return devices[target_position], devices

                target = SimpleNamespace(
                    _controls_enabled=True,
                    WG=Profile(),
                    WG_searched_devices=(None, []),
                    _matched_device_entry=None,
                    device_combo=Combo(),
                    devices_list=[],
                    _log=lambda *args, **kwargs: None,
                )
                asyncio.run(qt5_bleak.window.search_click.__wrapped__(target))

                self.assertEqual(
                    target._scan_snapshot.matches[0].address,
                    devices[target_position][2],
                )
                self.assertEqual(target.device_combo.items, [str(target_position + 1)])
                target_lines = [line for line in target.devices_list if "[目标手环]" in line]
                self.assertEqual(len(target_lines), 1)
                self.assertTrue(target_lines[0].startswith(f"{target_position + 1}:"))

    def test_target_resolution_uses_identity_then_address_not_display_name(self):
        duplicate_name_devices = [
            [1, "same-name", "00:00:00:00:00:01", -50],
            [2, "same-name", "00:00:00:00:00:02", -50],
        ]
        copied_match = [99, "same-name", "00:00:00:00:00:02", -99]
        position, entry = qt5_bleak._resolve_matched_entry(
            copied_match, duplicate_name_devices
        )
        self.assertEqual(position, 1)
        self.assertIs(entry, duplicate_name_devices[1])

        duplicate_address_devices = [
            [1, "first", "00:00:00:00:00:03", -50],
            [2, "second", "00:00:00:00:00:03", -40],
        ]
        position, entry = qt5_bleak._resolve_matched_entry(
            list(duplicate_address_devices[1]), duplicate_address_devices
        )
        self.assertEqual(position, 1)
        self.assertIs(entry, duplicate_address_devices[1])

    def test_rescan_without_match_clears_old_target_and_blocks_connection(self):
        first_entry = [1, "bracelet", "00:00:00:00:00:01", -40]

        class Combo:
            def __init__(self):
                self.items = []

            def clear(self):
                self.items.clear()

            def addItem(self, value, user_data=None):
                self.items.append(value)

        class Profile:
            is_connected = False

            def __init__(self):
                self.results = [
                    (first_entry, [first_entry]),
                    (None, [[1, "other", "00:00:00:00:00:09", -30]]),
                ]
                self.connect_calls = []

            async def scan(self, timeout):
                return self.results.pop(0)

            async def connect(self, address):
                self.connect_calls.append(address)
                self.is_connected = True
                return True

        profile = Profile()
        target = SimpleNamespace(
            _controls_enabled=True,
            WG=profile,
            WG_searched_devices=(None, []),
            _matched_device_entry=None,
            device_combo=Combo(),
            devices_list=[],
            device_key=DeviceKey("dev-" + "0" * 32),
            _log=lambda *args, **kwargs: None,
        )

        async def exercise():
            await qt5_bleak.window.search_click.__wrapped__(target)
            self.assertEqual(target._scan_snapshot.matches[0].address, first_entry[2])
            await qt5_bleak.window.search_click.__wrapped__(target)
            with mock.patch.object(qt5_bleak.QMessageBox, "warning") as warning:
                await qt5_bleak.window.connect_semg_click.__wrapped__(target)
            warning.assert_called_once()

        asyncio.run(exercise())
        self.assertEqual(target._scan_snapshot.matches, ())
        self.assertEqual(target.device_combo.items, [])
        self.assertEqual(profile.connect_calls, [])

    def test_connect_uses_confirmed_target_not_combo_selection(self):
        wrong_entry = [1, "other", "00:00:00:00:00:01", -30]
        matched_entry = [5, "bracelet", "00:00:00:00:00:05", -60]

        class Profile:
            is_connected = False

            def __init__(self):
                self.connect_calls = []

            async def connect(self, address):
                self.connect_calls.append(address)
                self.is_connected = True
                return True

        class Runtime:
            def __init__(self):
                self.tokens = []

            def register_sensitive_token(self, token):
                self.tokens.append(token)

        profile = Profile()
        runtime = Runtime()
        identity_calls = []
        snapshot = qt5_bleak._coerce_scan_snapshot(
            (matched_entry, [wrong_entry, matched_entry])
        )
        target = SimpleNamespace(
            _controls_enabled=True,
            WG=profile,
            _scan_snapshot=snapshot,
            _scan_epoch=4,
            _completed_scan_epoch=4,
            _scan_in_progress=False,
            _connect_epoch=0,
            _connect_in_progress=False,
            _ble_ui_tasks=set(),
            device_combo=SimpleNamespace(currentText=lambda: "1"),
            logging_runtime=runtime,
            identity_store=SimpleNamespace(
                device_key=lambda address: identity_calls.append(address)
                or DeviceKey("dev-" + "4" * 32)
            ),
            device_key=DeviceKey("dev-" + "0" * 32),
            _log=lambda *args, **kwargs: None,
        )

        with mock.patch.object(qt5_bleak.QMessageBox, "information"):
            asyncio.run(qt5_bleak.window.connect_semg_click.__wrapped__(target))

        self.assertEqual(profile.connect_calls, [matched_entry[2]])
        self.assertEqual(identity_calls, [matched_entry[2]])
        self.assertEqual(runtime.tokens, [matched_entry[2]])

    def test_multiple_matching_devices_require_explicit_selection(self):
        first = [1, "bracelet-a", "00:00:00:00:00:01", -20]
        second = [3, "bracelet-b", "00:00:00:00:00:03", -80]

        class Profile:
            is_connected = False

            def __init__(self):
                self.connect_calls = []

            async def scan(self, timeout):
                return [first, second], [first, [2, "other", "x", -10], second]

            async def connect(self, address):
                self.connect_calls.append(address)
                self.is_connected = True
                return True

        profile = Profile()
        combo = SelectionCombo()
        target = SimpleNamespace(
            _controls_enabled=True,
            WG=profile,
            WG_searched_devices=(None, []),
            _matched_device_entry=None,
            _matched_device_entries=(),
            _scan_epoch=0,
            _completed_scan_epoch=None,
            _scan_in_progress=False,
            device_combo=combo,
            devices_list=[],
            logging_runtime=SimpleNamespace(register_sensitive_token=lambda token: None),
            identity_store=SimpleNamespace(
                device_key=lambda address: DeviceKey("dev-" + "5" * 32)
            ),
            device_key=DeviceKey("dev-" + "0" * 32),
            _log=lambda *args, **kwargs: None,
        )

        async def exercise():
            await qt5_bleak.window.search_click.__wrapped__(target)
            self.assertEqual(combo.items, ["1", "3"])
            self.assertEqual(
                combo.item_data, ["legacy-candidate-1", "legacy-candidate-3"]
            )
            self.assertEqual(combo.currentIndex(), -1)
            with mock.patch.object(qt5_bleak.QMessageBox, "warning") as warning:
                await qt5_bleak.window.connect_semg_click.__wrapped__(target)
            warning.assert_called_once()
            self.assertEqual(profile.connect_calls, [])

            combo.setCurrentIndex(1)
            with mock.patch.object(qt5_bleak.QMessageBox, "information"):
                await qt5_bleak.window.connect_semg_click.__wrapped__(target)

        asyncio.run(exercise())
        self.assertEqual(profile.connect_calls, [second[2]])

    def test_connect_is_single_flight_and_preserves_native_device(self):
        native_device = object()
        discovered = qt5_bleak.bleak_ble.DiscoveredDevice(
            "candidate-1", 1, "bracelet", "00:00:00:00:00:01", -40, True,
            native_device,
        )
        snapshot = qt5_bleak.bleak_ble.ScanSnapshot((discovered,), (discovered,))

        class Profile:
            is_connected = False
            client_generation = 7

            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()
                self.connect_calls = []

            async def connect(self, device):
                self.connect_calls.append(device)
                self.started.set()
                await self.release.wait()
                self.is_connected = True
                return True

        profile = Profile()
        identity_calls = []
        target = SimpleNamespace(
            _controls_enabled=True,
            WG=profile,
            _scan_snapshot=snapshot,
            _scan_epoch=1,
            _completed_scan_epoch=1,
            _scan_in_progress=False,
            _connect_epoch=0,
            _connect_in_progress=False,
            _ble_ui_tasks=set(),
            device_combo=SelectionCombo(),
            logging_runtime=SimpleNamespace(register_sensitive_token=lambda token: None),
            identity_store=SimpleNamespace(
                device_key=lambda address: identity_calls.append(address)
                or DeviceKey("dev-" + "7" * 32)
            ),
            device_key=DeviceKey("dev-" + "0" * 32),
            _log=lambda *args, **kwargs: None,
        )

        async def exercise():
            first = asyncio.create_task(qt5_bleak.window.connect_semg_click.__wrapped__(target))
            await profile.started.wait()
            with mock.patch.object(qt5_bleak.QMessageBox, "warning") as warning:
                await qt5_bleak.window.connect_semg_click.__wrapped__(target)
            warning.assert_called_once()
            profile.release.set()
            with mock.patch.object(qt5_bleak.QMessageBox, "information") as information:
                await first
            information.assert_called_once()

        asyncio.run(exercise())
        self.assertEqual(profile.connect_calls, [native_device])
        self.assertEqual(identity_calls, [discovered.address])
        self.assertEqual(target.device_key, DeviceKey("dev-" + "7" * 32))
        self.assertEqual(target._accepted_connection_generation, 7)

    def test_rescan_is_blocked_during_connection_without_rebinding_identity(self):
        native_device = object()
        discovered = qt5_bleak.bleak_ble.DiscoveredDevice(
            "candidate-1", 1, "bracelet", "00:00:00:00:00:01", -40, True,
            native_device,
        )
        snapshot = qt5_bleak.bleak_ble.ScanSnapshot((discovered,), (discovered,))

        class Profile:
            is_connected = False
            client_generation = 8

            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()
                self.scan_calls = 0

            async def connect(self, device):
                self.started.set()
                await self.release.wait()
                self.is_connected = True
                return True

            async def scan(self, timeout):
                self.scan_calls += 1
                raise AssertionError("scan must be blocked during connection")

        profile = Profile()
        target = SimpleNamespace(
            _controls_enabled=True,
            WG=profile,
            _scan_snapshot=snapshot,
            _scan_epoch=1,
            _completed_scan_epoch=1,
            _scan_in_progress=False,
            _connect_epoch=0,
            _connect_in_progress=False,
            _ble_ui_tasks=set(),
            device_combo=SelectionCombo(),
            devices_list=[],
            logging_runtime=SimpleNamespace(register_sensitive_token=lambda token: None),
            identity_store=SimpleNamespace(
                device_key=lambda address: DeviceKey("dev-" + "8" * 32)
            ),
            device_key=DeviceKey("dev-" + "0" * 32),
            _log=lambda *args, **kwargs: None,
        )

        async def exercise():
            connect_task = asyncio.create_task(
                qt5_bleak.window.connect_semg_click.__wrapped__(target)
            )
            await profile.started.wait()
            with mock.patch.object(qt5_bleak.QMessageBox, "warning") as warning:
                await qt5_bleak.window.search_click.__wrapped__(target)
            warning.assert_called_once()
            self.assertIs(target._scan_snapshot, snapshot)
            profile.release.set()
            with mock.patch.object(qt5_bleak.QMessageBox, "information"):
                await connect_task

        asyncio.run(exercise())
        self.assertEqual(profile.scan_calls, 0)
        self.assertEqual(target.device_key, DeviceKey("dev-" + "8" * 32))

    def test_backend_false_never_commits_success_or_device_identity(self):
        discovered = qt5_bleak.bleak_ble.DiscoveredDevice(
            "candidate-1", 1, "bracelet", "00:00:00:00:00:01", -40, True,
            object(),
        )
        snapshot = qt5_bleak.bleak_ble.ScanSnapshot((discovered,), (discovered,))
        identity_calls = []
        target = SimpleNamespace(
            _controls_enabled=True,
            WG=SimpleNamespace(
                is_connected=False,
                connect=lambda device: asyncio.sleep(0, result=False),
            ),
            _scan_snapshot=snapshot,
            _scan_epoch=1,
            _completed_scan_epoch=1,
            _scan_in_progress=False,
            _connect_epoch=0,
            _connect_in_progress=False,
            _ble_ui_tasks=set(),
            device_combo=SelectionCombo(),
            logging_runtime=SimpleNamespace(register_sensitive_token=lambda token: None),
            identity_store=SimpleNamespace(
                device_key=lambda address: identity_calls.append(address)
            ),
            device_key=DeviceKey("dev-" + "0" * 32),
            _log=lambda *args, **kwargs: None,
        )
        with mock.patch.object(qt5_bleak.QMessageBox, "critical") as critical, mock.patch.object(
            qt5_bleak.QMessageBox, "information"
        ) as information:
            asyncio.run(qt5_bleak.window.connect_semg_click.__wrapped__(target))

        critical.assert_called_once()
        information.assert_not_called()
        self.assertEqual(identity_calls, [])
        self.assertEqual(target.device_key, DeviceKey("dev-" + "0" * 32))

    def test_late_failed_first_connect_cannot_enable_second_or_commit_identity(self):
        discovered = qt5_bleak.bleak_ble.DiscoveredDevice(
            "candidate-1", 1, "bracelet", "00:00:00:00:00:01", -40, True,
            object(),
        )
        snapshot = qt5_bleak.bleak_ble.ScanSnapshot((discovered,), (discovered,))

        class Profile:
            is_connected = False

            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()
                self.calls = 0

            async def connect(self, device):
                self.calls += 1
                self.started.set()
                await self.release.wait()
                raise OSError("late failure")

        profile = Profile()
        identity_calls = []
        target = SimpleNamespace(
            _controls_enabled=True,
            WG=profile,
            _scan_snapshot=snapshot,
            _scan_epoch=1,
            _completed_scan_epoch=1,
            _scan_in_progress=False,
            _connect_epoch=0,
            _connect_in_progress=False,
            _ble_ui_tasks=set(),
            device_combo=SelectionCombo(),
            logging_runtime=SimpleNamespace(register_sensitive_token=lambda token: None),
            identity_store=SimpleNamespace(
                device_key=lambda address: identity_calls.append(address)
            ),
            device_key=DeviceKey("dev-" + "0" * 32),
            _log=lambda *args, **kwargs: None,
        )

        async def exercise():
            first = asyncio.create_task(qt5_bleak.window.connect_semg_click.__wrapped__(target))
            await profile.started.wait()
            with mock.patch.object(qt5_bleak.QMessageBox, "warning") as warning:
                await qt5_bleak.window.connect_semg_click.__wrapped__(target)
            warning.assert_called_once()
            profile.release.set()
            with mock.patch.object(qt5_bleak.QMessageBox, "critical") as critical:
                await first
            critical.assert_called_once()

        asyncio.run(exercise())
        self.assertEqual(profile.calls, 1)
        self.assertEqual(identity_calls, [])
        self.assertEqual(target.device_key, DeviceKey("dev-" + "0" * 32))

    def test_overlapping_scans_publish_only_latest_success(self):
        older = [1, "older", "00:00:00:00:00:01", -30]
        newer = [2, "newer", "00:00:00:00:00:02", -70]

        class Profile:
            def __init__(self):
                self.started = [asyncio.Event(), asyncio.Event()]
                self.release = [asyncio.Event(), asyncio.Event()]
                self.calls = 0

            async def scan(self, timeout):
                call = self.calls
                self.calls += 1
                self.started[call].set()
                await self.release[call].wait()
                return ([older], [older]) if call == 0 else ([newer], [newer])

        profile = Profile()
        target = SimpleNamespace(
            _controls_enabled=True,
            WG=profile,
            WG_searched_devices=(None, []),
            _matched_device_entry=None,
            _matched_device_entries=(),
            _scan_epoch=0,
            _completed_scan_epoch=None,
            _scan_in_progress=False,
            device_combo=SelectionCombo(),
            devices_list=[],
            _log=lambda *args, **kwargs: None,
        )

        async def exercise():
            first_task = asyncio.create_task(qt5_bleak.window.search_click.__wrapped__(target))
            await profile.started[0].wait()
            second_task = asyncio.create_task(qt5_bleak.window.search_click.__wrapped__(target))
            await profile.started[1].wait()
            profile.release[1].set()
            await second_task
            profile.release[0].set()
            await first_task

        asyncio.run(exercise())
        self.assertEqual(target._scan_epoch, 2)
        self.assertEqual(target._completed_scan_epoch, 2)
        self.assertEqual(
            [device.address for device in target._scan_snapshot.matches], [newer[2]]
        )
        self.assertEqual(target.device_combo.items, ["2"])
        self.assertTrue(any("newer" in line for line in target.devices_list))
        self.assertFalse(any("older" in line for line in target.devices_list))

    def test_overlapping_scans_latest_failure_or_no_match_clears_older_success(self):
        older = [1, "older", "00:00:00:00:00:01", -30]
        for latest_outcome in (RuntimeError("scan failed"), ([], [])):
            with self.subTest(latest_outcome=type(latest_outcome).__name__):
                class Profile:
                    def __init__(self):
                        self.started = [asyncio.Event(), asyncio.Event()]
                        self.release = [asyncio.Event(), asyncio.Event()]
                        self.calls = 0

                    async def scan(self, timeout):
                        call = self.calls
                        self.calls += 1
                        self.started[call].set()
                        await self.release[call].wait()
                        if call == 0:
                            return [older], [older]
                        if isinstance(latest_outcome, Exception):
                            raise latest_outcome
                        return latest_outcome

                profile = Profile()
                target = SimpleNamespace(
                    _controls_enabled=True,
                    WG=profile,
                    WG_searched_devices=(None, []),
                    _matched_device_entry=None,
                    _matched_device_entries=(),
                    _scan_epoch=0,
                    _completed_scan_epoch=None,
                    _scan_in_progress=False,
                    device_combo=SelectionCombo(),
                    devices_list=[],
                    _log=lambda *args, **kwargs: None,
                )

                async def exercise():
                    first_task = asyncio.create_task(
                        qt5_bleak.window.search_click.__wrapped__(target)
                    )
                    await profile.started[0].wait()
                    second_task = asyncio.create_task(
                        qt5_bleak.window.search_click.__wrapped__(target)
                    )
                    await profile.started[1].wait()
                    profile.release[1].set()
                    with mock.patch.object(qt5_bleak.QMessageBox, "critical"):
                        await second_task
                    profile.release[0].set()
                    await first_task

                asyncio.run(exercise())
                self.assertTrue(
                    target._scan_snapshot is None or not target._scan_snapshot.matches
                )
                self.assertEqual(target.device_combo.items, [])
                self.assertFalse(any("older" in line for line in target.devices_list))

    def test_older_failed_scan_cannot_replace_newer_success(self):
        newer = [2, "newer", "00:00:00:00:00:02", -60]

        class Profile:
            def __init__(self):
                self.started = [asyncio.Event(), asyncio.Event()]
                self.release = [asyncio.Event(), asyncio.Event()]
                self.calls = 0

            async def scan(self, timeout):
                call = self.calls
                self.calls += 1
                self.started[call].set()
                await self.release[call].wait()
                if call == 0:
                    raise RuntimeError("older failed")
                return [newer], [newer]

        profile = Profile()
        target = SimpleNamespace(
            _controls_enabled=True,
            WG=profile,
            WG_searched_devices=(None, []),
            _matched_device_entry=None,
            _matched_device_entries=(),
            _scan_epoch=0,
            _completed_scan_epoch=None,
            _scan_in_progress=False,
            device_combo=SelectionCombo(),
            devices_list=[],
            _log=lambda *args, **kwargs: None,
        )

        async def exercise():
            first_task = asyncio.create_task(qt5_bleak.window.search_click.__wrapped__(target))
            await profile.started[0].wait()
            second_task = asyncio.create_task(qt5_bleak.window.search_click.__wrapped__(target))
            await profile.started[1].wait()
            profile.release[1].set()
            await second_task
            profile.release[0].set()
            await first_task

        asyncio.run(exercise())
        self.assertEqual(
            [device.address for device in target._scan_snapshot.matches], [newer[2]]
        )
        self.assertEqual(target.device_combo.items, ["2"])

    def test_connect_is_blocked_while_newer_scan_is_running(self):
        old = [1, "old", "00:00:00:00:00:01", -40]
        new = [2, "new", "00:00:00:00:00:02", -50]

        class Profile:
            is_connected = False

            def __init__(self):
                self.calls = 0
                self.second_started = asyncio.Event()
                self.release_second = asyncio.Event()
                self.connect_calls = []

            async def scan(self, timeout):
                self.calls += 1
                if self.calls == 1:
                    return [old], [old]
                self.second_started.set()
                await self.release_second.wait()
                return [new], [new]

            async def connect(self, address):
                self.connect_calls.append(address)

        profile = Profile()
        target = SimpleNamespace(
            _controls_enabled=True,
            WG=profile,
            WG_searched_devices=(None, []),
            _matched_device_entry=None,
            _matched_device_entries=(),
            _scan_epoch=0,
            _completed_scan_epoch=None,
            _scan_in_progress=False,
            device_combo=SelectionCombo(),
            devices_list=[],
            logging_runtime=SimpleNamespace(register_sensitive_token=lambda token: None),
            identity_store=SimpleNamespace(
                device_key=lambda address: DeviceKey("dev-" + "6" * 32)
            ),
            device_key=DeviceKey("dev-" + "0" * 32),
            _log=lambda *args, **kwargs: None,
        )

        async def exercise():
            await qt5_bleak.window.search_click.__wrapped__(target)
            second_task = asyncio.create_task(qt5_bleak.window.search_click.__wrapped__(target))
            await profile.second_started.wait()
            with mock.patch.object(qt5_bleak.QMessageBox, "warning") as warning:
                await qt5_bleak.window.connect_semg_click.__wrapped__(target)
            warning.assert_called_once()
            self.assertEqual(profile.connect_calls, [])
            profile.release_second.set()
            await second_task

        asyncio.run(exercise())
        self.assertEqual(profile.connect_calls, [])

    def test_cancelled_latest_scan_clears_target_and_in_progress_state(self):
        old = [1, "old", "00:00:00:00:00:01", -40]

        class Profile:
            def __init__(self):
                self.calls = 0
                self.started = asyncio.Event()

            async def scan(self, timeout):
                self.calls += 1
                if self.calls == 1:
                    return [old], [old]
                self.started.set()
                await asyncio.Event().wait()

        profile = Profile()
        target = SimpleNamespace(
            _controls_enabled=True,
            WG=profile,
            WG_searched_devices=(None, []),
            _matched_device_entry=None,
            _matched_device_entries=(),
            _scan_epoch=0,
            _completed_scan_epoch=None,
            _scan_in_progress=False,
            device_combo=SelectionCombo(),
            devices_list=[],
            _log=lambda *args, **kwargs: None,
        )

        async def exercise():
            await qt5_bleak.window.search_click.__wrapped__(target)
            scan_task = asyncio.create_task(qt5_bleak.window.search_click.__wrapped__(target))
            await profile.started.wait()
            scan_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await scan_task

        asyncio.run(exercise())
        self.assertFalse(target._scan_in_progress)
        self.assertIsNone(target._completed_scan_epoch)
        self.assertIsNone(target._scan_snapshot)
        self.assertEqual(target.device_combo.items, [])
        self.assertEqual(target.devices_list, ["搜索已取消"])

    def test_user_name_is_registered_but_not_used_as_subject_key_or_path(self):
        name = "Sensitive Person"

        class Runtime:
            def __init__(self):
                self.tokens = []

            def register_sensitive_token(self, token):
                self.tokens.append(token)

        runtime = Runtime()
        target = SimpleNamespace(
            logging_runtime=runtime,
            subject_key=None,
            temp_info={},
            statusBar=lambda: SimpleNamespace(showMessage=lambda *args: None),
        )
        dialog = SimpleNamespace(
            setWindowModality=lambda mode: None,
            exec_=lambda: qt5_bleak.QtWidgets.QDialog.Accepted,
            getuserInfo=lambda: {
                "subject_identifier": name,
                "side": HandSide.LEFT,
                "action_label": "rest",
                "action_phase": "hold",
                "experiment_id": "discrete_hand_v1",
            },
        )
        target.recording_state = qt5_bleak.RecordingState.IDLE
        target.is_recording = False
        target.subject_identity_store = SimpleNamespace(
            derive_subject_id=lambda value: "sub-" + "a" * 32
        )
        target.context_status_label = SimpleNamespace(setText=lambda value: None)
        target._log = lambda *args, **kwargs: None
        with mock.patch.object(qt5_bleak, "user_ui_dialog", return_value=dialog):
            qt5_bleak.window.user_info_click(target)
        self.assertEqual(runtime.tokens, [name])
        self.assertNotIn(name, target.subject_key)
        self.assertTrue(target.subject_key.startswith("sub-"))
        self.assertIs(target.temp_info["side"], HandSide.LEFT)

    def test_default_context_hook_preserves_non_training_recording(self):
        subject_id = "sub-" + "a" * 32
        context = qt5_bleak.window._build_recording_context(
            SimpleNamespace(),
            subject_id,
            {
                "side": HandSide.LEFT,
                "action_label": "rest",
                "action_phase": "hold",
                "experiment_id": "discrete_hand_v1",
            },
        )
        self.assertEqual(context.subject_id, subject_id)
        self.assertIsNone(context.training_provenance)

    def test_user_info_hooks_are_overrideable_without_changing_default_entry(self):
        subject_id = "sub-" + "b" * 32
        user_info = {
            "subject_identifier": "operator-subject",
            "side": HandSide.RIGHT,
            "action_label": "fist",
            "action_phase": "hold",
            "experiment_id": "training_batch_1",
        }
        dialog = SimpleNamespace(
            setWindowModality=lambda mode: None,
            exec_=lambda: qt5_bleak.QtWidgets.QDialog.Accepted,
            getuserInfo=lambda: user_info,
        )
        calls = []

        class HookTarget(SimpleNamespace):
            def _create_user_info_dialog(self):
                calls.append("dialog")
                return dialog

            def _build_recording_context(self, derived_subject_id, supplied_info):
                calls.append(("build", derived_subject_id, supplied_info))
                return RecordingContext(
                    subject_id=derived_subject_id,
                    hand_side=supplied_info["side"],
                    action_label=supplied_info["action_label"],
                    action_phase=supplied_info["action_phase"],
                    experiment_id=supplied_info["experiment_id"],
                    training_provenance="canonical_session",
                )

            def _after_recording_context_registered(self, context, supplied_info):
                calls.append(("registered", context, supplied_info))

        target = HookTarget(
            recording_state=qt5_bleak.RecordingState.IDLE,
            is_recording=False,
            _active_recording_context=None,
            logging_runtime=SimpleNamespace(register_sensitive_token=lambda value: None),
            subject_identity_store=SimpleNamespace(
                derive_subject_id=lambda value: subject_id
            ),
            context_status_label=SimpleNamespace(setText=lambda value: None),
            statusBar=lambda: SimpleNamespace(showMessage=lambda *args: None),
            _log=lambda *args, **kwargs: None,
        )
        qt5_bleak.window.user_info_click(target)
        self.assertEqual(calls[0], "dialog")
        self.assertEqual(calls[1], ("build", subject_id, user_info))
        self.assertEqual(calls[2], ("registered", target._pending_recording_context, user_info))
        self.assertEqual(
            target._pending_recording_context.training_provenance,
            "canonical_session",
        )

    def test_user_dialog_binds_hand_side_enum_as_item_data(self):
        dialog = qt5_bleak.user_ui_dialog()
        try:
            self.assertEqual(dialog.side.count(), 3)
            self.assertIs(dialog.side.itemData(0), HandSide.UNKNOWN)
            self.assertIs(dialog.side.itemData(1), HandSide.LEFT)
            self.assertIs(dialog.side.itemData(2), HandSide.RIGHT)
            self.assertNotIn("\ufffd", "".join(dialog.side.itemText(i) for i in range(3)))
            dialog.user_name.setText("test-subject")
            dialog.side.setCurrentIndex(2)
            dialog.action.setCurrentIndex(1)
            dialog.save_click()
            self.assertIs(dialog.getuserInfo()["side"], HandSide.RIGHT)
        finally:
            dialog.close()

    def test_user_dialog_requires_subject_side_and_action_and_returns_canonical_values(self):
        dialog = qt5_bleak.user_ui_dialog()
        try:
            with mock.patch.object(qt5_bleak.QMessageBox, "warning") as warning:
                dialog.save_click()
                self.assertEqual(dialog.result(), 0)
                self.assertTrue(warning.called)
            dialog.user_name.setText("operator-subject-7")
            dialog.side.setCurrentIndex(1)
            dialog.action.setCurrentIndex(2)
            dialog.experiment_id.setText("../unsafe")
            with mock.patch.object(qt5_bleak.QMessageBox, "warning") as warning:
                dialog.save_click()
                self.assertEqual(dialog.result(), 0)
                warning.assert_called_once()
            dialog.experiment_id.setText("discrete_hand_v1")
            dialog.save_click()
            result = dialog.getuserInfo()
            self.assertEqual(result["action_label"], "fist")
            self.assertEqual(result["action_phase"], "hold")
            self.assertEqual(result["experiment_id"], "discrete_hand_v1")
            self.assertIs(result["side"], HandSide.LEFT)
        finally:
            dialog.close()

    def test_subject_input_reuses_stable_pseudonym_and_plaintext_is_not_retained(self):
        plaintext = "Private Subject 42"
        with tempfile.TemporaryDirectory() as directory:
            target = SimpleNamespace(
                recording_state=qt5_bleak.RecordingState.IDLE,
                is_recording=False,
                logging_runtime=SimpleNamespace(register_sensitive_token=lambda token: None),
                subject_identity_store=qt5_bleak.SubjectIdentityStore(
                    directory, dataset_domain=qt5_bleak.SUBJECT_DATASET_DOMAIN
                ),
                context_status_label=SimpleNamespace(setText=lambda value: None),
                statusBar=lambda: SimpleNamespace(showMessage=lambda *args: None),
                _log=lambda *args, **kwargs: None,
            )

            def make_dialog():
                return SimpleNamespace(
                    setWindowModality=lambda mode: None,
                    exec_=lambda: qt5_bleak.QtWidgets.QDialog.Accepted,
                    getuserInfo=lambda: {
                        "subject_identifier": plaintext,
                        "side": HandSide.RIGHT,
                        "action_label": "open_hand",
                        "action_phase": "hold",
                        "experiment_id": "discrete_hand_v1",
                    },
                )

            with mock.patch.object(qt5_bleak, "user_ui_dialog", side_effect=lambda *args: make_dialog()):
                qt5_bleak.window.user_info_click(target)
                first = target._pending_recording_context
                qt5_bleak.window.user_info_click(target)
                second = target._pending_recording_context
            self.assertEqual(first.subject_id, second.subject_id)
            self.assertNotIn(plaintext, repr(target.temp_info))
            self.assertNotIn(plaintext, repr(first))
            self.assertNotIn(plaintext, (Path(directory) / "subject_identity.key").read_bytes().decode("latin1"))

    def test_recording_context_is_latched_into_recorder_and_pipeline(self):
        context = recording_context(side=HandSide.LEFT, action="open_hand")
        profile = FakeProfile()
        pipeline = FakePipeline()
        pipeline.queue_depth = 0
        pipeline.dropped_count = 0
        button_states = []
        target = SimpleNamespace(
            _controls_enabled=True,
            is_recording=False,
            recording_state=qt5_bleak.RecordingState.IDLE,
            _recording_lock=asyncio.Lock(),
            current_session_id=None,
            _pending_recording_context=context,
            WG=profile,
            pipeline=pipeline,
            config=SimpleNamespace(
                data=SimpleNamespace(data_path="unused", channels=8),
                to_acquisition_metadata=lambda: load_config(qt5_bleak.CONFIG_PATH).to_acquisition_metadata(),
            ),
            device_key=DeviceKey("dev-" + "2" * 32),
            _log=lambda *args, **kwargs: None,
            recording_status=SimpleNamespace(setText=lambda value: None),
            quality_status_label=SimpleNamespace(setText=lambda value: None),
            user_info=SimpleNamespace(setEnabled=button_states.append),
        )
        recorder = SimpleNamespace(
            session_id="session",
            session_dir=Path(tempfile.gettempdir()) / "fake-session",
            close=lambda **kwargs: None,
        )
        with mock.patch.object(qt5_bleak, "DataRecorder", return_value=recorder) as factory:
            asyncio.run(qt5_bleak.window._start_recording(target))
        self.assertIs(target._active_recording_context, context)
        self.assertIs(pipeline.recording_context, context)
        self.assertEqual(button_states, [False])
        kwargs = factory.call_args.kwargs
        self.assertIs(kwargs["recording_context"], context)
        self.assertNotIn("metadata_extra", kwargs)
        self.assertEqual(kwargs["subject_id"], context.subject_id)
        self.assertNotIn("Private", repr(kwargs))

    def test_default_recorder_hook_preserves_constructor_arguments(self):
        context = recording_context(side=HandSide.RIGHT, action="fist")
        acquisition = load_config(qt5_bleak.CONFIG_PATH).to_acquisition_metadata()
        target = SimpleNamespace(
            config=SimpleNamespace(
                data=SimpleNamespace(data_path="data-root", channels=8),
                to_acquisition_metadata=lambda: acquisition,
            ),
            device_key=DeviceKey("dev-" + "3" * 32),
        )
        recorder = object()
        with mock.patch.object(qt5_bleak, "DataRecorder", return_value=recorder) as factory:
            result = qt5_bleak.window._create_data_recorder(target, context)
        self.assertIs(result, recorder)
        factory.assert_called_once_with(
            "data-root",
            subject_id=context.subject_id,
            device_id=target.device_key,
            acquisition=acquisition,
            channels=8,
            side=HandSide.RIGHT,
            recording_context=context,
        )

    def test_user_info_is_locked_while_recording(self):
        target = SimpleNamespace(
            recording_state=qt5_bleak.RecordingState.RECORDING,
            is_recording=True,
            _active_recording_context=recording_context(),
        )
        with mock.patch.object(qt5_bleak, "user_ui_dialog") as dialog, mock.patch.object(
            qt5_bleak.QMessageBox, "warning"
        ) as warning:
            qt5_bleak.window.user_info_click(target)
        dialog.assert_not_called()
        warning.assert_called_once()

    def test_quality_report_runs_after_close_and_distinguishes_pass_warning_failure(self):
        async def exercise(status, usable):
            label_values = []
            logs = []
            target = SimpleNamespace(
                _controls_enabled=True,
                quality_status_label=SimpleNamespace(setText=label_values.append),
                _log=lambda *args, **kwargs: logs.append((args, kwargs)),
            )
            with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                qt5_bleak, "analyze_session", return_value={
                    "overall_status": status,
                    "training_usable": usable,
                }
            ):
                await qt5_bleak.window._analyze_closed_session(target, Path(directory))
                report_path = Path(directory) / "quality_report.json"
                self.assertTrue(report_path.is_file())
                self.assertIn(status, report_path.read_text(encoding="utf-8"))
            return label_values[-1], logs

        passed, pass_logs = asyncio.run(exercise("pass", True))
        warned, _ = asyncio.run(exercise("warning", False))
        failed, _ = asyncio.run(exercise("fail", False))
        self.assertIn("通过（可训练）", passed)
        self.assertIn("警告（不可训练）", warned)
        self.assertIn("失败（不可训练）", failed)
        self.assertEqual(pass_logs[-1][1]["error_count"], 0)

    def test_quality_hooks_are_overrideable_and_hook_failure_remains_fail_closed(self):
        label_values = []
        after_calls = []
        logs = []
        report = {"overall_status": "pass", "training_usable": True}
        target = SimpleNamespace(
            _controls_enabled=True,
            quality_status_label=SimpleNamespace(setText=label_values.append),
            _log=lambda *args, **kwargs: logs.append((args, kwargs)),
            _format_quality_report_message=lambda supplied_report, supplied_path: (
                f"custom:{supplied_report['overall_status']}:{supplied_path.name}"
            ),
            _after_quality_report=lambda supplied_report, supplied_path: after_calls.append(
                (supplied_report, supplied_path)
            ),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            qt5_bleak, "analyze_session", return_value=report
        ):
            session_dir = Path(directory)
            asyncio.run(qt5_bleak.window._analyze_closed_session(target, session_dir))
            report_path = session_dir / "quality_report.json"
            self.assertEqual(label_values[-1], f"custom:pass:{report_path.name}")
            self.assertEqual(after_calls, [(report, report_path)])

        target._after_quality_report = lambda *args: (_ for _ in ()).throw(
            RuntimeError("hook failed")
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            qt5_bleak, "analyze_session", return_value=report
        ):
            session_dir = Path(directory)
            asyncio.run(qt5_bleak.window._analyze_closed_session(target, session_dir))
            self.assertIn("执行失败（不可训练）", label_values[-1])
            self.assertEqual(logs[-1][0][1], "session quality analysis failed")
            self.assertEqual(logs[-1][1]["error_count"], 1)

    def test_quality_exception_is_fail_closed_and_exit_clears_context(self):
        label_values = []
        target = SimpleNamespace(
            _controls_enabled=True,
            quality_status_label=SimpleNamespace(setText=label_values.append),
            _log=lambda *args, **kwargs: None,
        )
        with mock.patch.object(
            qt5_bleak.window,
            "_analyze_and_write_quality_report",
            side_effect=OSError("disk error"),
        ):
            asyncio.run(
                qt5_bleak.window._analyze_closed_session(
                    target, Path(tempfile.gettempdir()) / "opaque-session"
                )
            )
        self.assertIn("执行失败（不可训练）", label_values[-1])

        shutdown_target = SimpleNamespace(
            _controls_enabled=True,
            _pending_recording_context=recording_context(),
            _active_recording_context=recording_context(),
            _active_session_dir=Path("opaque"),
            subject_key="sub-" + "1" * 32,
            temp_info={"action_label": "fist"},
            _shutdown_task=object(),
            _ble_ui_tasks=set(),
            _scan_epoch=0,
            _connect_epoch=0,
        )
        returned = qt5_bleak.window.request_shutdown(shutdown_target)
        self.assertIs(returned, shutdown_target._shutdown_task)
        self.assertIsNone(shutdown_target._pending_recording_context)
        self.assertIsNotNone(shutdown_target._active_recording_context)
        self.assertIsNone(shutdown_target.subject_key)

    def test_quality_report_preserves_existing_target_without_reanalysis(self):
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory)
            report_path = session_dir / "quality_report.json"
            existing = {"overall_status": "warning", "training_usable": False}
            original = json.dumps(existing).encode("utf-8")
            report_path.write_bytes(original)
            with mock.patch.object(qt5_bleak, "analyze_session") as analyze:
                report, returned_path = qt5_bleak.window._analyze_and_write_quality_report(
                    session_dir
                )
            analyze.assert_not_called()
            self.assertEqual(report, existing)
            self.assertEqual(returned_path, report_path)
            self.assertEqual(report_path.read_bytes(), original)

    def test_quality_report_concurrent_publish_keeps_one_complete_report(self):
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory)
            barrier = threading.Barrier(2)
            values = iter((1, 2))

            def analyze(_session_dir):
                value = next(values)
                barrier.wait(timeout=2)
                return {
                    "overall_status": "pass",
                    "training_usable": True,
                    "writer": value,
                }

            with mock.patch.object(qt5_bleak, "analyze_session", side_effect=analyze):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(
                            qt5_bleak.window._analyze_and_write_quality_report,
                            session_dir,
                        )
                        for _ in range(2)
                    ]
                    results = [future.result(timeout=3)[0] for future in futures]
            persisted = json.loads(
                (session_dir / "quality_report.json").read_text(encoding="utf-8")
            )
            self.assertIn(persisted["writer"], (1, 2))
            self.assertEqual(results, [persisted, persisted])
            self.assertEqual(list(session_dir.glob(".quality_report.*.tmp")), [])

    def test_quality_report_write_publish_and_cleanup_failures_are_isolated(self):
        report = {"overall_status": "fail", "training_usable": False}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            qt5_bleak, "analyze_session", return_value=report
        ):
            session_dir = Path(directory)
            original_open = Path.open

            def fail_temp_open(path, *args, **kwargs):
                if path.name.startswith(".quality_report."):
                    raise OSError("write denied")
                return original_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", fail_temp_open):
                with self.assertRaisesRegex(OSError, "write denied"):
                    qt5_bleak.window._analyze_and_write_quality_report(session_dir)
            self.assertFalse((session_dir / "quality_report.json").exists())

            existing = {"overall_status": "warning", "training_usable": False}

            def publish_conflict(source, target):
                Path(target).write_text(json.dumps(existing), encoding="utf-8")
                raise FileExistsError(target)

            with mock.patch.object(qt5_bleak.os, "link", side_effect=publish_conflict):
                returned, _ = qt5_bleak.window._analyze_and_write_quality_report(
                    session_dir
                )
            self.assertEqual(returned, existing)
            self.assertEqual(list(session_dir.glob(".quality_report.*.tmp")), [])

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            qt5_bleak, "analyze_session", return_value=report
        ), mock.patch.object(Path, "unlink", side_effect=OSError("cleanup denied")):
            session_dir = Path(directory)
            with self.assertRaisesRegex(OSError, "cleanup denied"):
                qt5_bleak.window._analyze_and_write_quality_report(session_dir)
            self.assertTrue((session_dir / "quality_report.json").is_file())

    def test_subject_identity_location_is_stable_local_app_data(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"LOCALAPPDATA": directory}
        ):
            location = qt5_bleak._subject_identity_directory()
        self.assertEqual(
            location,
            Path(directory).resolve() / "EMGBracelet" / "subject_identity",
        )
        self.assertNotEqual(location.parent, qt5_bleak.BASE_DIR)

    def test_ui_sources_and_generated_classes_use_the_same_object_names(self):
        main_ui = (qt5_bleak.BASE_DIR / "Qt5_MainWindow.ui").read_text(encoding="utf-8")
        dialog_ui = (qt5_bleak.BASE_DIR / "UI_patient_info.ui").read_text(encoding="utf-8")
        for name in ("user_info", "context_status_label", "quality_status_label"):
            self.assertIn(f'name="{name}"', main_ui)
        for name in ("user_name", "side", "action", "experiment_id"):
            self.assertIn(f'name="{name}"', dialog_ui)
        self.assertNotIn("�", main_ui + dialog_ui)
        self.assertIn("仅用于生成匿名编号，不保存明文", dialog_ui)
        self.assertIn("质量检查：尚未执行", main_ui)
        self.assertTrue(hasattr(qt5_bleak.Ui_MainWindow(), "setupUi"))

    def test_concurrent_start_and_stop_are_single_transactions(self):
        class Profile(FakeProfile):
            async def setNotify(self, flag):
                await asyncio.sleep(0)
                return await super().setNotify(flag)

        class Pipeline(FakePipeline):
            queue_depth = 0
            dropped_count = 0

            def __init__(self):
                super().__init__()
                self.attach_count = 0
                self.stop_count = 0

            def set_recorder(self, recorder, *, recording_context=None):
                self.attach_count += 1
                super().set_recorder(
                    recorder, recording_context=recording_context
                )

            def stop_recorder(self, **kwargs):
                self.stop_count += 1
                super().stop_recorder(**kwargs)

            def request_recorder_stop(self, **kwargs):
                self.stop_count += 1
                self.stop_token = object()
                return self.stop_token

            def finish_recorder_stop(self, token, *, timeout):
                self.assert_token = token
                self.recorder.close(complete=True)
                self.recorder = None
                return SimpleNamespace(tail_pending_count=0)

        profile = Profile()
        pipeline = Pipeline()
        recorder = SimpleNamespace(
            session_id="session",
            session_dir=Path(tempfile.gettempdir()) / "fake-session",
            close=lambda **kwargs: None,
        )
        recorder_arguments = []
        acquisition = load_config(qt5_bleak.CONFIG_PATH).to_acquisition_metadata()
        target = SimpleNamespace(
            _controls_enabled=True,
            is_recording=False,
            recording_state=qt5_bleak.RecordingState.IDLE,
            current_session_id=None,
            subject_key="sub-" + "1" * 32,
            temp_info={"side": HandSide.RIGHT},
            _pending_recording_context=recording_context(),
            WG=profile,
            pipeline=pipeline,
            config=SimpleNamespace(
                data=SimpleNamespace(data_path="unused", channels=8),
                to_acquisition_metadata=lambda: acquisition,
            ),
            device_key=DeviceKey("dev-" + "2" * 32),
            _log=lambda *args, **kwargs: None,
            recording_status=SimpleNamespace(setText=lambda value: None),
        )

        async def exercise():
            target._recording_lock = asyncio.Lock()
            def make_recorder(*args, **kwargs):
                recorder_arguments.append((args, kwargs))
                return recorder

            with mock.patch.object(qt5_bleak, "DataRecorder", side_effect=make_recorder), mock.patch.object(qt5_bleak.QMessageBox, "information"):
                await asyncio.gather(
                    qt5_bleak.window._start_recording(target),
                    qt5_bleak.window._start_recording(target),
                )
                await asyncio.gather(
                    qt5_bleak.window._stop_recording(target),
                    qt5_bleak.window._stop_recording(target),
                )

        asyncio.run(exercise())
        self.assertEqual(pipeline.attach_count, 1)
        self.assertEqual(pipeline.stop_count, 1)
        self.assertEqual(target.recording_state, qt5_bleak.RecordingState.IDLE)
        self.assertIs(recorder_arguments[0][1]["side"], HandSide.RIGHT)
        self.assertIs(
            recorder_arguments[0][1]["acquisition"].notification_packet_protocol,
            acquisition.notification_packet_protocol,
        )

    def test_recording_buttons_have_one_explicit_route_each(self):
        class Profile:
            is_connected = False
            is_notifying = False

            def add_state_listener(self, listener):
                pass

            def remove_state_listener(self, listener):
                pass

            def add_notification_listener(self, listener):
                pass

            def remove_notification_listener(self, listener):
                pass

        class Pipeline:
            def start(self):
                pass

            def notification_callback(self, notification):
                pass

        runtime = SimpleNamespace(
            get_logger=lambda **kwargs: SimpleNamespace(
                info=lambda *args, **kwargs: None,
                warning=lambda *args, **kwargs: None,
                error=lambda *args, **kwargs: None,
            ),
            register_sensitive_token=lambda token: None,
            shutdown=lambda: None,
        )
        form = qt5_bleak.window(
            SimpleNamespace(),
            config=load_config(qt5_bleak.CONFIG_PATH),
            logging_runtime=runtime,
            profile=Profile(),
            shared_writer=SimpleNamespace(),
            pipeline=Pipeline(),
            identity_store=SimpleNamespace(),
        )
        try:
            form._start_recording = mock.Mock(return_value="start-awaitable")
            form._stop_recording = mock.Mock(return_value="stop-awaitable")
            with mock.patch.object(qt5_bleak.asyncio, "ensure_future") as schedule:
                form.start_recording.click()
                form.recording_state = qt5_bleak.RecordingState.RECORDING
                form.is_recording = True
                form.stop_recording.click()
                self.app.processEvents()

            form._start_recording.assert_called_once_with()
            form._stop_recording.assert_called_once_with()
            self.assertEqual(schedule.call_count, 2)
            self.assertEqual(
                [call.args[0] for call in schedule.call_args_list],
                ["start-awaitable", "stop-awaitable"],
            )
            self.assertEqual(form.recording_status.text(), "正在停止…")
        finally:
            form._allow_close = True
            form.close()

    def test_stop_closes_device_notify_and_recorder_once_and_is_idempotent(self):
        class Pipeline:
            recorder_cleanup_pending = False

            def __init__(self):
                self.stop_count = 0
                self.closed = False

            def request_recorder_stop(self, **kwargs):
                self.stop_count += 1
                self.token = object()
                return self.token

            def finish_recorder_stop(self, token, *, timeout):
                self.assert_token = token
                self.closed = True
                return SimpleNamespace(tail_pending_count=0)

        profile = FakeProfile()
        profile.is_notifying = True
        pipeline = Pipeline()
        target = SimpleNamespace(
            _controls_enabled=True,
            _recording_lock=asyncio.Lock(),
            is_recording=True,
            recording_state=qt5_bleak.RecordingState.RECORDING,
            current_session_id="session",
            _active_recording_context=recording_context(),
            _active_session_dir=Path("opaque-session"),
            user_info=SimpleNamespace(setEnabled=lambda value: None),
            WG=profile,
            pipeline=pipeline,
            recording_status=SimpleNamespace(setText=lambda value: None),
            device_key=DeviceKey("dev-" + "7" * 32),
            _log=lambda *args, **kwargs: None,
        )

        async def exercise():
            await qt5_bleak.window._stop_recording(target)
            await qt5_bleak.window._stop_recording(target)

        def assert_closed_before_quality(target_arg, session_dir):
            self.assertTrue(pipeline.closed)
            self.assertEqual(session_dir, Path("opaque-session"))

        with mock.patch.object(qt5_bleak.QMessageBox, "information") as completed, mock.patch.object(
            qt5_bleak.window, "_schedule_session_quality", side_effect=assert_closed_before_quality
        ) as schedule_quality:
            asyncio.run(exercise())

        self.assertEqual(profile.calls, [("command", 0, 0), ("notify", 0)])
        self.assertEqual(pipeline.stop_count, 1)
        self.assertEqual(target.recording_state, qt5_bleak.RecordingState.IDLE)
        self.assertFalse(target.is_recording)
        self.assertIsNone(target.current_session_id)
        completed.assert_called_once()
        schedule_quality.assert_called_once()

    def test_stop_retries_only_pending_device_command(self):
        class Pipeline:
            recorder_cleanup_pending = False

            def __init__(self):
                self.stop_count = 0

            def request_recorder_stop(self, **kwargs):
                self.stop_count += 1
                self.token = object()
                return self.token

            def finish_recorder_stop(self, token, *, timeout):
                self.assert_token = token
                return SimpleNamespace(tail_pending_count=0)

        class Profile(FakeProfile):
            def __init__(self):
                super().__init__()
                self.stop_attempts = 0
                self.is_notifying = True

            async def setDataType(self, flag, wristband, *, expected_generation=None):
                self.calls.append(("command", flag, wristband))
                self.stop_attempts += 1
                if self.stop_attempts == 1:
                    raise RuntimeError("device stop temporarily failed")

        profile = Profile()
        pipeline = Pipeline()
        target = SimpleNamespace(
            _controls_enabled=True,
            _recording_lock=asyncio.Lock(),
            is_recording=True,
            recording_stop_pending={"generation": 1, "stage": "device_stop"},
            recording_state=qt5_bleak.RecordingState.RECORDING,
            current_session_id="session",
            WG=profile,
            pipeline=pipeline,
            recording_status=SimpleNamespace(setText=lambda value: None),
            device_key=DeviceKey("dev-" + "6" * 32),
            _log=lambda *args, **kwargs: None,
        )

        async def exercise():
            await qt5_bleak.window._stop_recording(target)
            self.assertEqual(target.recording_stop_pending["stage"], "device_stop")
            self.assertEqual(target.recording_state, qt5_bleak.RecordingState.STOPPING)
            self.assertEqual(profile.calls, [("command", 0, 0), ("notify", 0)])
            self.assertEqual(pipeline.stop_count, 1)

            await qt5_bleak.window._stop_recording(target)
            self.assertIsNone(target.recording_stop_pending)
            self.assertEqual(target.recording_state, qt5_bleak.RecordingState.IDLE)
            self.assertEqual(
                profile.calls,
                [("command", 0, 0), ("notify", 0), ("command", 0, 0)],
            )
            self.assertEqual(pipeline.stop_count, 1)

            await qt5_bleak.window._stop_recording(target)

        with mock.patch.object(qt5_bleak.QMessageBox, "critical") as failed, mock.patch.object(
            qt5_bleak.QMessageBox, "information"
        ) as completed:
            asyncio.run(exercise())

        failed.assert_called_once()
        completed.assert_called_once()
        self.assertEqual(profile.stop_attempts, 2)
        self.assertEqual(pipeline.stop_count, 1)

    def test_stop_device_timeout_is_bounded_and_retains_device_stage(self):
        class Profile(FakeProfile):
            def __init__(self):
                super().__init__()
                self.is_notifying = True

            async def setDataType(self, flag, wristband, *, expected_generation=None):
                self.calls.append(("command", flag, wristband))
                await asyncio.Event().wait()

        class Pipeline:
            recorder_cleanup_pending = False

            def __init__(self):
                self.token = object()
                self.finish_count = 0

            def request_recorder_stop(self, **kwargs):
                return self.token

            def finish_recorder_stop(self, token, *, timeout):
                self.finish_count += 1
                return SimpleNamespace(tail_pending_count=0)

        profile = Profile()
        pipeline = Pipeline()
        target = SimpleNamespace(
            _controls_enabled=True,
            _recording_lock=asyncio.Lock(),
            _accepted_connection_generation=1,
            _ble_ui_tasks=set(),
            is_recording=True,
            recording_state=qt5_bleak.RecordingState.RECORDING,
            recording_stop_pending={"generation": 1, "stage": "device_stop"},
            current_session_id="session",
            WG=profile,
            pipeline=pipeline,
            recording_status=SimpleNamespace(setText=lambda value: None),
            device_key=DeviceKey("dev-" + "5" * 32),
            _log=lambda *args, **kwargs: None,
        )

        with mock.patch.object(
            qt5_bleak, "RECORDING_STOP_BLE_TIMEOUT_SECONDS", 0.01
        ), mock.patch.object(qt5_bleak.QMessageBox, "critical"):
            started = time.monotonic()
            asyncio.run(qt5_bleak.window._stop_recording(target))
            elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.5)
        self.assertEqual(target.recording_stop_pending["stage"], "device_stop")
        self.assertEqual(profile.calls, [("command", 0, 0), ("notify", 0)])
        self.assertEqual(pipeline.finish_count, 1)

    def test_stop_notify_timeout_is_bounded_and_retains_notify_stage(self):
        class Profile(FakeProfile):
            def __init__(self):
                super().__init__()
                self.is_notifying = True

            async def setNotify(self, flag, *, expected_generation=None):
                self.calls.append(("notify", flag))
                await asyncio.Event().wait()

        profile = Profile()
        target = SimpleNamespace(
            _controls_enabled=True,
            _recording_lock=asyncio.Lock(),
            _accepted_connection_generation=1,
            _ble_ui_tasks=set(),
            is_recording=False,
            recording_state=qt5_bleak.RecordingState.STOPPING,
            recording_stop_pending={"generation": 1, "stage": "notify_stop"},
            current_session_id=None,
            WG=profile,
            pipeline=SimpleNamespace(recorder_cleanup_pending=False),
            recording_status=SimpleNamespace(setText=lambda value: None),
            device_key=DeviceKey("dev-" + "4" * 32),
            _log=lambda *args, **kwargs: None,
        )

        with mock.patch.object(
            qt5_bleak, "RECORDING_STOP_BLE_TIMEOUT_SECONDS", 0.01
        ), mock.patch.object(qt5_bleak.QMessageBox, "critical"):
            started = time.monotonic()
            asyncio.run(qt5_bleak.window._stop_recording(target))
            elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.5)
        self.assertEqual(target.recording_stop_pending["stage"], "notify_stop")
        self.assertEqual(profile.calls, [("notify", 0)])

    def test_stop_generation_change_during_device_wait_cannot_touch_new_connection(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        new_pending = {"generation": 2, "stage": "device_stop"}

        class Profile(FakeProfile):
            def __init__(self):
                super().__init__()
                self.is_notifying = True

            async def setDataType(self, flag, wristband, *, expected_generation=None):
                self.calls.append(("command", flag, wristband, expected_generation))
                entered.set()
                await release.wait()

        class Pipeline:
            recorder_cleanup_pending = False

            def __init__(self):
                self.request_count = 0

            def request_recorder_stop(self, **kwargs):
                self.request_count += 1
                return object()

        profile = Profile()
        pipeline = Pipeline()
        target = SimpleNamespace(
            _controls_enabled=True,
            _recording_lock=asyncio.Lock(),
            _accepted_connection_generation=1,
            _ble_ui_tasks=set(),
            is_recording=True,
            recording_state=qt5_bleak.RecordingState.RECORDING,
            recording_stop_pending={"generation": 1, "stage": "device_stop"},
            current_session_id="old-session",
            WG=profile,
            pipeline=pipeline,
            recording_status=SimpleNamespace(setText=lambda value: None),
        )

        async def exercise():
            task = asyncio.create_task(qt5_bleak.window._stop_recording(target))
            await entered.wait()
            profile.client_generation = 2
            target._accepted_connection_generation = 2
            target.recording_stop_pending = new_pending
            release.set()
            await task

        asyncio.run(exercise())
        self.assertIs(target.recording_stop_pending, new_pending)
        self.assertEqual(profile.calls, [("command", 0, 0, 1)])
        self.assertEqual(pipeline.request_count, 0)

    def test_stop_generation_change_between_ble_stages_skips_new_notify(self):
        new_pending = {"generation": 2, "stage": "device_stop"}
        target = None

        class Profile(FakeProfile):
            def __init__(self):
                super().__init__()
                self.is_notifying = True

            async def setDataType(self, flag, wristband, *, expected_generation=None):
                self.calls.append(("command", flag, wristband, expected_generation))
                self.client_generation = 2
                target._accepted_connection_generation = 2
                target.recording_stop_pending = new_pending

        profile = Profile()
        pipeline = SimpleNamespace(recorder_cleanup_pending=False)
        target = SimpleNamespace(
            _controls_enabled=True,
            _recording_lock=asyncio.Lock(),
            _accepted_connection_generation=1,
            _ble_ui_tasks=set(),
            is_recording=True,
            recording_state=qt5_bleak.RecordingState.RECORDING,
            recording_stop_pending={"generation": 1, "stage": "device_stop"},
            current_session_id="old-session",
            WG=profile,
            pipeline=pipeline,
            recording_status=SimpleNamespace(setText=lambda value: None),
        )

        asyncio.run(qt5_bleak.window._stop_recording(target))
        self.assertIs(target.recording_stop_pending, new_pending)
        self.assertEqual(profile.calls, [("command", 0, 0, 1)])

    def test_recorder_stop_runs_off_gui_thread_and_reuses_inflight_token(self):
        release = threading.Event()

        class Pipeline:
            recorder_cleanup_pending = False

            def __init__(self):
                self.token = object()
                self.request_count = 0
                self.finish_threads = []

            def request_recorder_stop(self, **kwargs):
                self.request_count += 1
                return self.token

            def finish_recorder_stop(self, token, *, timeout):
                self.finish_threads.append(threading.get_ident())
                release.wait(1.0)
                return SimpleNamespace(tail_pending_count=0)

        profile = FakeProfile()
        profile.is_notifying = False
        pipeline = Pipeline()
        target = SimpleNamespace(
            _controls_enabled=True,
            _recording_lock=asyncio.Lock(),
            is_recording=True,
            recording_state=qt5_bleak.RecordingState.RECORDING,
            current_session_id="session",
            WG=profile,
            pipeline=pipeline,
            recording_status=SimpleNamespace(setText=lambda value: None),
            device_key=DeviceKey("dev-" + "3" * 32),
            _log=lambda *args, **kwargs: None,
        )
        gui_thread = threading.get_ident()

        async def exercise():
            with mock.patch.object(
                qt5_bleak, "RECORDING_STOP_WORK_WAIT_SECONDS", 0.01
            ), mock.patch.object(qt5_bleak.QMessageBox, "critical"):
                await qt5_bleak.window._stop_recording(target)
            self.assertIs(target._recorder_stop_token, pipeline.token)
            release.set()
            await asyncio.sleep(0.03)
            with mock.patch.object(qt5_bleak.QMessageBox, "information"):
                await qt5_bleak.window._stop_recording(target)

        asyncio.run(exercise())
        self.assertEqual(pipeline.request_count, 1)
        self.assertEqual(len(pipeline.finish_threads), 1)
        self.assertNotEqual(pipeline.finish_threads[0], gui_thread)
        self.assertIsNone(target._recorder_stop_token)
        self.assertIsNone(target.recording_stop_pending)
        self.assertEqual(target.recording_state, qt5_bleak.RecordingState.IDLE)

    def test_stop_after_recorder_fault_still_stops_active_notifications(self):
        class Pipeline:
            recorder_cleanup_pending = False

            def stop_recorder(self, **kwargs):
                raise AssertionError("fault cleanup already closed the recorder")

        profile = FakeProfile()
        profile.is_notifying = True
        target = SimpleNamespace(
            _controls_enabled=True,
            _recording_lock=asyncio.Lock(),
            is_recording=False,
            recording_state=qt5_bleak.RecordingState.IDLE,
            current_session_id=None,
            WG=profile,
            pipeline=Pipeline(),
            recording_status=SimpleNamespace(setText=lambda value: None),
            device_key=DeviceKey("dev-" + "8" * 32),
            _log=lambda *args, **kwargs: None,
        )

        with mock.patch.object(qt5_bleak.QMessageBox, "information"):
            asyncio.run(qt5_bleak.window._stop_recording(target))

        self.assertEqual(profile.calls, [("command", 0, 0), ("notify", 0)])
        self.assertFalse(profile.is_notifying)
        self.assertEqual(target.recording_state, qt5_bleak.RecordingState.IDLE)

    def test_stream_watchdog_tracks_notify_start_stop_and_start_rollback(self):
        class Pipeline:
            recorder_cleanup_pending = False
            queue_depth = 0
            dropped_count = 0

            def __init__(self):
                self.calls = []

            def arm_stream_watchdog(self, generation):
                self.calls.append(("arm", generation))

            def disarm_stream_watchdog(self, generation):
                self.calls.append(("disarm", generation))

        def make_target(profile, pipeline):
            return SimpleNamespace(
                _controls_enabled=True,
                _recording_lock=asyncio.Lock(),
                is_recording=False,
                recording_state=qt5_bleak.RecordingState.IDLE,
                recording_stop_pending=None,
                current_session_id=None,
                WG=profile,
                pipeline=pipeline,
                recording_status=SimpleNamespace(setText=lambda value: None),
                device_key=DeviceKey("dev-" + "1" * 32),
                _log=lambda *args, **kwargs: None,
            )

        profile = FakeProfile()
        pipeline = Pipeline()
        target = make_target(profile, pipeline)

        async def start_then_stop():
            self.assertTrue(await qt5_bleak.window._ensure_transmitting(target))
            with mock.patch.object(qt5_bleak.QMessageBox, "information"):
                await qt5_bleak.window._stop_recording(target)

        asyncio.run(start_then_stop())
        self.assertEqual(pipeline.calls, [("arm", 1), ("disarm", 1)])

        failed_profile = FakeProfile(command_error=RuntimeError("start rejected"))
        failed_pipeline = Pipeline()
        failed_target = make_target(failed_profile, failed_pipeline)
        with self.assertRaisesRegex(RuntimeError, "start rejected"):
            asyncio.run(qt5_bleak.window._ensure_transmitting(failed_target))
        self.assertEqual(
            failed_pipeline.calls,
            [("arm", 1), ("disarm", 1)],
        )

    def test_start_rejects_disconnect_and_aba_without_stopping_new_connection(self):
        for outcome in ("notify_disconnect", "notify_aba", "command_aba"):
            with self.subTest(outcome=outcome):
                class Profile(FakeProfile):
                    client_generation = 10

                    async def setNotify(self, flag):
                        changed = await super().setNotify(flag)
                        if flag and outcome.startswith("notify_"):
                            if outcome == "notify_disconnect":
                                self.is_connected = False
                                self.client_generation += 1
                            else:
                                self.client_generation += 2
                                self.is_connected = True
                        return changed

                    async def setDataType(self, flag, wristband):
                        await super().setDataType(flag, wristband)
                        if flag and outcome == "command_aba":
                            self.client_generation += 2
                            self.is_connected = True

                profile = Profile()
                pipeline = FakePipeline()
                closed = []
                recorder = SimpleNamespace(
                    session_id="session",
                    session_dir=Path(tempfile.gettempdir()) / "fake-session",
                    close=lambda **kwargs: closed.append(kwargs),
                )
                target = SimpleNamespace(
                    _controls_enabled=True,
                    is_recording=False,
                    recording_state=qt5_bleak.RecordingState.IDLE,
                    _recording_lock=asyncio.Lock(),
                    current_session_id=None,
                    subject_key="sub-" + "1" * 32,
                    _pending_recording_context=recording_context(),
                    WG=profile,
                    pipeline=pipeline,
                    config=SimpleNamespace(
                        data=SimpleNamespace(data_path="unused", channels=8),
                        to_acquisition_metadata=lambda: None,
                    ),
                    device_key=DeviceKey("dev-" + "2" * 32),
                    _log=lambda *args, **kwargs: None,
                    recording_status=SimpleNamespace(setText=lambda value: None),
                )
                with mock.patch.object(qt5_bleak, "DataRecorder", return_value=recorder), mock.patch.object(qt5_bleak.QMessageBox, "critical"), mock.patch.object(qt5_bleak.window, "_schedule_session_quality") as schedule_quality:
                    asyncio.run(qt5_bleak.window._start_recording(target))
                expected = [("notify", 1)]
                if outcome == "command_aba":
                    expected.append(("command", 1, 0))
                self.assertEqual(profile.calls, expected)
                self.assertIsNone(pipeline.recorder)
                self.assertEqual(closed, [{"complete": False, "error": "start_failed"}])
                schedule_quality.assert_called_once_with(target, recorder.session_dir)
                self.assertFalse(target.is_recording)
                self.assertEqual(target.recording_state, qt5_bleak.RecordingState.IDLE)

    def test_start_header_and_audit_close_failure_retains_recorder_for_stop_retry(self):
        audits = []

        class FailingAudit:
            def __init__(self, path, **kwargs):
                self.close_attempts = 0
                audits.append(self)

            def write_policy(self, policy):
                raise RuntimeError("header failed")

            def write_protocol(self, protocol):
                pass

            def close(self):
                self.close_attempts += 1
                if self.close_attempts == 1:
                    raise OSError("audit close busy")

        class Recorder:
            session_id = "failed-session"

            def __init__(self, session_dir):
                self.session_dir = session_dir
                self.resource_state = "OPEN"
                self.close_calls = []

            def close(self, **kwargs):
                self.close_calls.append(kwargs)
                self.resource_state = "CLOSED"

        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory) / "failed-session")
            recorder.session_dir.mkdir()
            pipeline = qt5_bleak.AcquisitionPipeline(
                SimpleNamespace(generation=1),
                raw_audit_factory=FailingAudit,
                raw_audit_policy=qt5_bleak.RawAuditPolicy(enabled=True),
            )
            pipeline.start()
            profile = FakeProfile()
            target = SimpleNamespace(
                _controls_enabled=True,
                is_recording=False,
                recording_state=qt5_bleak.RecordingState.IDLE,
                _recording_lock=asyncio.Lock(),
                _shutdown_blocking_work={},
                _recorder_stop_token=None,
                _recorder_stop_work=None,
                _unattached_recorder=None,
                _unattached_cleanup_intent=None,
                _unattached_cleanup_work=None,
                current_session_id=None,
                _pending_recording_context=recording_context(),
                _active_recording_context=None,
                _active_session_dir=None,
                WG=profile,
                pipeline=pipeline,
                config=SimpleNamespace(
                    data=SimpleNamespace(data_path="unused", channels=8),
                    to_acquisition_metadata=lambda: None,
                ),
                device_key=DeviceKey("dev-" + "2" * 32),
                _log=lambda *args, **kwargs: None,
                recording_status=SimpleNamespace(setText=lambda value: None),
            )
            with mock.patch.object(
                qt5_bleak, "DataRecorder", return_value=recorder
            ), mock.patch.object(qt5_bleak.QMessageBox, "critical"), mock.patch.object(
                qt5_bleak.window, "_schedule_session_quality"
            ) as quality:
                asyncio.run(qt5_bleak.window._start_recording(target))
                self.assertIs(target._unattached_recorder, recorder)
                self.assertIsNotNone(target._active_recording_context)
                self.assertEqual(target._active_session_dir, recorder.session_dir)
                self.assertEqual(
                    target._unattached_cleanup_intent, (False, "start_failed")
                )
                self.assertEqual(
                    target.recording_state, qt5_bleak.RecordingState.STOPPING
                )
                quality.assert_not_called()
                with mock.patch.object(qt5_bleak.QMessageBox, "information"):
                    asyncio.run(qt5_bleak.window._stop_recording(target))
            self.assertEqual(audits[0].close_attempts, 2)
            self.assertEqual(
                recorder.close_calls, [{"complete": False, "error": "start_failed"}]
            )
            self.assertIsNone(target._unattached_recorder)
            self.assertIsNone(target._active_recording_context)
            self.assertEqual(target.recording_state, qt5_bleak.RecordingState.IDLE)
            quality.assert_called_once_with(target, recorder.session_dir)
            pipeline.close(drain=False)

    def test_disconnect_cleanup_does_not_block_gui_or_overwrite_new_generation(self):
        pipeline = qt5_bleak.AcquisitionPipeline(SimpleNamespace(generation=7))
        pipeline.start()
        real_mark_disconnected = pipeline.mark_disconnected
        real_resume = pipeline.resume
        pipeline.events = []

        def delayed_mark_disconnected(reason, *, connection_generation=None):
            time.sleep(0.25)
            real_mark_disconnected(
                reason, connection_generation=connection_generation
            )
            pipeline.events.append(("old_cleanup", connection_generation))

        def tracked_resume():
            real_resume()
            pipeline.events.append(("resume", None))

        pipeline.mark_disconnected = delayed_mark_disconnected
        pipeline.resume = tracked_resume

        status = SimpleNamespace(value=None)
        status.setText = lambda value: setattr(status, "value", value)
        target = SimpleNamespace(
            _controls_enabled=True,
            _accepted_connection_generation=9,
            _recording_start_token=object(),
            _recording_lock=asyncio.Lock(),
            _pipeline_cleanup_tasks=set(),
            _pipeline_resume_task=None,
            _pipeline_resume_generation=None,
            _unattached_recorder=None,
            is_recording=True,
            recording_state=qt5_bleak.RecordingState.RECORDING,
            recording_stop_pending=None,
            current_session_id="old-session",
            _active_recording_context=recording_context(),
            _active_session_dir=Path("old-session"),
            recording_status=status,
            pipeline=pipeline,
            WG=SimpleNamespace(is_connected=True),
            statusBar=lambda: SimpleNamespace(showMessage=lambda *args: None),
            device_key=DeviceKey("dev-" + "9" * 32),
            _log=lambda *args, **kwargs: None,
        )

        async def exercise():
            started = time.monotonic()
            qt5_bleak.window._on_ble_state(
                target,
                qt5_bleak.bleak_ble.BleStateEvent(
                    "disconnected", False, False, 9, "unexpected"
                ),
            )
            elapsed = time.monotonic() - started
            target._accepted_connection_generation = 12
            status.setText("new generation status")
            qt5_bleak.window._on_ble_state(
                target,
                qt5_bleak.bleak_ble.BleStateEvent(
                    "connected", True, False, 12, None
                ),
            )
            qt5_bleak.window._on_ble_state(
                target,
                qt5_bleak.bleak_ble.BleStateEvent(
                    "disconnected", False, False, 9, "duplicate"
                ),
            )
            await qt5_bleak.window._wait_pipeline_cleanup_tasks(target)
            return elapsed

        with mock.patch.object(qt5_bleak.window, "_schedule_session_quality") as quality:
            elapsed = asyncio.run(exercise())
        self.assertLess(elapsed, 0.05)
        self.assertEqual(target._accepted_connection_generation, 12)
        self.assertEqual(status.value, "new generation status")
        self.assertEqual(target.recording_state, qt5_bleak.RecordingState.IDLE)
        self.assertTrue(target.pipeline._accepting)
        self.assertEqual(
            target.pipeline.events, [("old_cleanup", 9), ("resume", None)]
        )
        quality.assert_called_once_with(target, Path("old-session"))
        pipeline.close(drain=False)

    def test_disconnect_callback_without_running_loop_retains_cleanup_intent(self):
        calls = []
        target = SimpleNamespace(
            _controls_enabled=True,
            _accepted_connection_generation=4,
            _recording_start_token=object(),
            is_recording=True,
            recording_state=qt5_bleak.RecordingState.RECORDING,
            recording_stop_pending=None,
            current_session_id="session",
            recording_status=SimpleNamespace(setText=lambda value: None),
            pipeline=SimpleNamespace(
                mark_disconnected=lambda reason: calls.append(reason)
            ),
            device_key=DeviceKey("dev-" + "4" * 32),
            _log=lambda *args, **kwargs: None,
            loop=None,
        )
        qt5_bleak.window._on_ble_state(
            target,
            qt5_bleak.bleak_ble.BleStateEvent(
                "disconnected", False, False, 4, "unexpected"
            ),
        )
        self.assertEqual(calls, [])
        self.assertEqual(target.recording_state, qt5_bleak.RecordingState.STOPPING)
        self.assertEqual(target.recording_stop_pending["stage"], "disconnect_cleanup")
        self.assertEqual(target.current_session_id, "session")

    def test_shutdown_gate_prevents_deferred_resume(self):
        resumes = []

        async def exercise():
            release = asyncio.Event()

            async def old_cleanup():
                await release.wait()

            old_task = asyncio.create_task(old_cleanup())
            target = SimpleNamespace(
                _controls_enabled=True,
                _accepted_connection_generation=22,
                _pipeline_cleanup_tasks={old_task},
                _pipeline_resume_task=None,
                _pipeline_resume_generation=None,
                pipeline=SimpleNamespace(resume=lambda: resumes.append(22)),
                WG=SimpleNamespace(is_connected=True),
                device_key=DeviceKey("dev-" + "2" * 32),
                statusBar=lambda: SimpleNamespace(showMessage=lambda *args: None),
                _log=lambda *args, **kwargs: None,
            )
            qt5_bleak.window._on_ble_state(
                target,
                qt5_bleak.bleak_ble.BleStateEvent(
                    "connected", True, False, 22, None
                ),
            )
            target._controls_enabled = False
            release.set()
            await qt5_bleak.window._wait_pipeline_cleanup_tasks(target)

        asyncio.run(exercise())
        self.assertEqual(resumes, [])

    def test_recording_fault_cleanup_pending_stays_stopping_and_stop_retries(self):
        class Pipeline:
            recorder_cleanup_pending = True

            def request_recorder_stop(self, **kwargs):
                self.token = object()
                return self.token

            def finish_recorder_stop(self, token, *, timeout):
                self.recorder_cleanup_pending = False
                return SimpleNamespace(tail_pending_count=0)

        pipeline = Pipeline()
        target = SimpleNamespace(
            _controls_enabled=True,
            _recording_lock=asyncio.Lock(),
            is_recording=True,
            recording_state=qt5_bleak.RecordingState.RECORDING,
            current_session_id="session",
            pipeline=pipeline,
            recording_status=SimpleNamespace(setText=lambda value: None),
            device_key=DeviceKey("dev-" + "4" * 32),
            _log=lambda *args, **kwargs: None,
        )
        root_error = OSError("disk full")
        close_error = RuntimeError("close busy")
        qt5_bleak.window._handle_pipeline_event(
            target,
            qt5_bleak.PipelineEvent(
                qt5_bleak.PipelineEventType.RECORDING_FAULT,
                "recorder_write",
                error=root_error,
                cleanup_pending=True,
                close_error=close_error,
            ),
        )
        self.assertEqual(target.recording_state, qt5_bleak.RecordingState.STOPPING)
        with mock.patch.object(qt5_bleak.QMessageBox, "information"):
            asyncio.run(qt5_bleak.window._stop_recording(target))
        self.assertFalse(pipeline.recorder_cleanup_pending)
        self.assertEqual(target.recording_state, qt5_bleak.RecordingState.IDLE)

    def test_start_rejects_pending_recorder_cleanup(self):
        target = SimpleNamespace(
            _controls_enabled=True,
            recording_state=qt5_bleak.RecordingState.IDLE,
            _recording_lock=asyncio.Lock(),
            pipeline=SimpleNamespace(recorder_cleanup_pending=True),
        )
        with mock.patch.object(qt5_bleak.QMessageBox, "warning") as warning, mock.patch.object(qt5_bleak, "DataRecorder") as recorder:
            asyncio.run(qt5_bleak.window._start_recording(target))
        warning.assert_called_once()
        recorder.assert_not_called()
        self.assertEqual(target.recording_state, qt5_bleak.RecordingState.STOPPING)

    def test_shutdown_hook_routes_about_to_quit_and_uncaught_exception(self):
        callbacks = []
        requests = []
        records = []
        signal = SimpleNamespace(connect=callbacks.append)
        app = SimpleNamespace(aboutToQuit=signal)
        form = SimpleNamespace(
            note_shutdown_request=requests.append,
            request_shutdown=requests.append,
        )
        logger = SimpleNamespace(error=lambda message, **kwargs: records.append((message, kwargs)))
        runtime = SimpleNamespace(get_logger=lambda **kwargs: logger)
        previous = sys.excepthook
        try:
            hook = qt5_bleak.install_shutdown_hooks(app, form, runtime)
            callbacks[0]()
            error = RuntimeError("boom")
            hook(RuntimeError, error, error.__traceback__)
        finally:
            sys.excepthook = previous
        self.assertEqual(requests, ["about_to_quit", "uncaught_exception"])
        self.assertEqual(records[0][1]["exc_info"][1], error)

    def test_close_event_uses_shutdown_gate_and_failed_finish_remains_retryable(self):
        calls = []
        event = SimpleNamespace(ignore=lambda: calls.append("ignore"), accept=lambda: calls.append("accept"))
        target = SimpleNamespace(
            _allow_close=False,
            _shutdown_complete=False,
            request_shutdown=lambda reason: calls.append(reason),
        )
        qt5_bleak.window.closeEvent(target, event)
        self.assertEqual(calls, ["ignore", "window_close"])

        class FailedTask:
            def result(self):
                raise qt5_bleak.ShutdownError([("consumer", TimeoutError("busy"))])

        finish_target = SimpleNamespace(_last_shutdown_error=None, _shutdown_task=object())
        with mock.patch.object(qt5_bleak.QMessageBox, "critical") as critical:
            qt5_bleak.window._finish_close(finish_target, FailedTask())
        self.assertIsInstance(finish_target._last_shutdown_error, qt5_bleak.ShutdownError)
        self.assertIsNone(finish_target._shutdown_task)
        critical.assert_called_once()

    def test_hmac_device_identity_and_logs_never_contain_mac(self):
        address = "AA:BB:CC:DD:EE:FF"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = qt5_bleak.DeviceIdentityStore(root / "identity" / "key", root / "data")
            runtime = configure_logging(root / "logs", queue_capacity=16)
            runtime.register_sensitive_token(address)
            runtime.get_logger(
                event=EventCode.BLE_CONNECT,
                device_id=store.device_key(address),
            ).info(f"connected {address}")
            log_path = runtime.log_file
            runtime.shutdown()
            content = log_path.read_text(encoding="utf-8")
            self.assertNotIn(address, content)
            self.assertIn("[REDACTED", content)

    def test_real_qasync_loop_with_patched_bleak_backend_has_hard_timeout(self):
        code = textwrap.dedent(
            """
            import asyncio, os, threading, time
            os.environ['QT_QPA_PLATFORM']='offscreen'
            from types import SimpleNamespace
            from unittest.mock import AsyncMock, MagicMock, patch
            from PyQt5.QtWidgets import QApplication
            from qasync import QEventLoop
            import bleak_ble
            app=QApplication([]); loop=QEventLoop(app); asyncio.set_event_loop(loop)
            profile=bleak_ble.BleakProfile('svc','svc','notify','cmd')
            service=SimpleNamespace(get_characteristic=lambda uuid: object())
            client=MagicMock(); client.is_connected=True; client.connect=AsyncMock()
            async def disconnect(): client.is_connected=False
            client.disconnect=AsyncMock(side_effect=disconnect); client.start_notify=AsyncMock(); client.stop_notify=AsyncMock(); client.write_gatt_char=AsyncMock()
            client.services.get_service.return_value=service
            class Scanner:
                async def discover(self, timeout=1):
                    return [SimpleNamespace(name='x',address='A',rssi=-1,metadata={'uuids':['svc']})]
            async def flow():
                await profile.scan(.01); await profile.connect('A'); profile.add_notification_listener(lambda notification: None)
                await profile.setNotify(1); await profile.setDataType(1,0); await profile.setNotify(0); await profile.disconnect()
            with patch.object(bleak_ble,'BleakScanner',Scanner), patch.object(bleak_ble,'BleakClient',return_value=client):
                with loop: loop.run_until_complete(asyncio.wait_for(flow(),2))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(qt5_bleak.BASE_DIR),
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_qasync_blocked_shared_close_reaches_hard_process_boundary(self):
        code = textwrap.dedent(
            """
            import asyncio, os, sys, threading, time
            os.environ['QT_QPA_PLATFORM']='offscreen'
            from types import SimpleNamespace
            from PyQt5 import QtCore
            from PyQt5.QtWidgets import QApplication
            from qasync import QEventLoop
            import qt5_bleak

            app=QApplication([]); loop=QEventLoop(app); asyncio.set_event_loop(loop)
            entered=threading.Event()

            class Profile:
                is_connected=False; is_notifying=False
                async def disconnect(self): pass
                def remove_notification_listener(self, listener): pass
                def remove_state_listener(self, listener): pass

            class Pipeline:
                is_closed=False
                notification_callback=staticmethod(lambda notification: None)
                def stop_accepting(self): pass
                def close(self, **kwargs): self.is_closed=True
                def stop_recorder(self, **kwargs): pass
                def publish_status(self, flags): pass

            class Shared:
                def close_resources(self):
                    entered.set(); time.sleep(10)
                def release_producer(self):
                    os._exit(99)

            class Form:
                def __init__(self):
                    self._quit_reason='test'; self._last_shutdown_error=None
                    self._shutdown_complete=False; self._shutdown_task=None
                    self._shutdown_lock=asyncio.Lock(); self._recording_lock=asyncio.Lock()
                    self._shutdown_stages=set(); self._shutdown_blocking_work={}
                    self._ble_ui_tasks=set(); self._controls_enabled=True
                    self.is_recording=False; self.recording_state=qt5_bleak.RecordingState.IDLE
                    self.current_session_id=None; self.WG=Profile(); self.pipeline=Pipeline()
                    self.shared_writer=Shared(); self.logging_runtime=SimpleNamespace(shutdown=lambda: None)
                    self._on_ble_state=lambda event: None; self._log=lambda *args, **kwargs: None
                async def shutdown(self, reason, *, deadline):
                    await qt5_bleak.window.shutdown(self, reason, deadline=deadline)

            form=Form(); loop.call_soon(loop.stop)
            def hard_terminate(code):
                assert entered.is_set()
                assert 'shared_resources_close' not in form._shutdown_stages
                assert 'shared_close' not in form._shutdown_stages
                assert qt5_bleak._active_blocking_shutdown_work(form)
                os._exit(23)
            with loop:
                qt5_bleak._run_application_loop(
                    loop, form, shutdown_attempts=1,
                    shutdown_timeout_seconds=.12,
                    hard_terminate=hard_terminate,
                )
            """
        )
        result = subprocess.run(
            [sys.executable, "-W", "error::ResourceWarning", "-c", code],
            cwd=str(qt5_bleak.BASE_DIR),
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertIn("forced_process_boundary", result.stderr)

    def test_application_loop_preserves_shutdown_cancellation_identity(self):
        loop = asyncio.new_event_loop()
        cancellation = asyncio.CancelledError("shutdown cancelled")
        events = []

        class Form:
            _quit_reason = "test"
            _last_shutdown_error = cancellation

            async def shutdown(self, reason):
                events.extend(["device_stop", "disconnect", "logging"])
                self._last_shutdown_error = cancellation
                self._shutdown_cancellation_attempt_token = self._shutdown_attempt_token
                raise cancellation

        try:
            loop.call_soon(loop.stop)
            with self.assertRaises(asyncio.CancelledError) as raised:
                qt5_bleak._run_application_loop(loop, Form())
        finally:
            loop.close()

        self.assertIs(raised.exception, cancellation)
        self.assertEqual(events, ["device_stop", "disconnect", "logging"])

    def test_cancelled_application_shutdown_with_stubborn_shutdown_task_hard_terminates(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        release = asyncio.Event()
        cancellation = asyncio.CancelledError("cancel with pending cleanup")
        termination = []

        class HardTerminate(BaseException):
            pass

        async def stubborn_task():
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue

        class Form:
            _quit_reason = "test"
            _last_shutdown_error = None
            _shutdown_pending_tasks = set()
            _ble_ui_tasks = set()
            _shutdown_blocking_work = {}
            WG = SimpleNamespace(_pending_disconnect_task=None)
            _shutdown_task = None
            spawned = None

            async def shutdown(self, reason, *, deadline):
                self.spawned = asyncio.create_task(stubborn_task())
                self._shutdown_pending_tasks.add(self.spawned)
                self._last_shutdown_error = cancellation
                self._shutdown_cancellation_attempt_token = self._shutdown_attempt_token
                raise cancellation

        form = Form()

        def hard_terminate(code):
            termination.append(code)
            raise HardTerminate()

        try:
            loop.call_soon(loop.stop)
            with self.assertRaises(HardTerminate):
                qt5_bleak._run_application_loop(
                    loop,
                    form,
                    shutdown_attempts=1,
                    shutdown_timeout_seconds=0.03,
                    hard_terminate=hard_terminate,
                )
            self.assertEqual(termination, [2])
            self.assertIs(form._last_shutdown_error, cancellation)
            self.assertEqual(
                form._last_shutdown_error.cleanup_failures[-1][0],
                "forced_process_boundary",
            )
        finally:
            release.set()
            loop.run_until_complete(asyncio.wait_for(form.spawned, 0.2))
            loop.close()
            asyncio.set_event_loop(None)

    def test_application_loop_clears_stale_cancellation_side_channel(self):
        loop = asyncio.new_event_loop()
        stale = asyncio.CancelledError("stale")

        class Form:
            _quit_reason = "test"
            _last_shutdown_error = stale

            async def shutdown(self, reason):
                raise asyncio.CancelledError("fresh")

        try:
            loop.call_soon(loop.stop)
            with self.assertRaises(asyncio.CancelledError) as raised:
                qt5_bleak._run_application_loop(loop, Form())
        finally:
            loop.close()

        self.assertIsNot(raised.exception, stale)

    def test_active_blocking_cleanup_forces_process_boundary(self):
        import threading

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        release = threading.Event()
        entered = threading.Event()
        termination = []

        class HardTerminate(BaseException):
            pass

        class Form:
            _quit_reason = "test"
            _last_shutdown_error = None
            _shutdown_pending_tasks = set()
            _ble_ui_tasks = set()
            _shutdown_blocking_work = {}
            _shutdown_task = None
            WG = SimpleNamespace(_pending_disconnect_task=None)

            async def shutdown(self, reason, *, deadline):
                def blocking_cleanup():
                    entered.set()
                    release.wait()

                await qt5_bleak.window._run_shutdown_blocking_work(
                    self, "logging_shutdown", blocking_cleanup
                )

        form = Form()

        def hard_terminate(code):
            termination.append(code)
            raise HardTerminate()

        try:
            loop.call_soon(loop.stop)
            with self.assertRaises(HardTerminate):
                qt5_bleak._run_application_loop(
                    loop,
                    form,
                    shutdown_attempts=1,
                    shutdown_timeout_seconds=0.03,
                    hard_terminate=hard_terminate,
                )
            self.assertTrue(entered.is_set())
            self.assertEqual(termination, [2])
            self.assertTrue(qt5_bleak._active_blocking_shutdown_work(form))
        finally:
            release.set()
            for work in form._shutdown_blocking_work.values():
                work.completed.wait(0.2)
            loop.run_until_complete(asyncio.sleep(0))
            loop.close()
            asyncio.set_event_loop(None)

    def test_reported_unterminated_cleanup_forces_boundary_after_inventory_race(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        termination = []

        class HardTerminate(BaseException):
            pass

        class Form:
            _quit_reason = "test"
            _last_shutdown_error = None
            _shutdown_task = None
            WG = SimpleNamespace(_pending_disconnect_task=None)

            async def shutdown(self, reason):
                raise qt5_bleak.ShutdownError(
                    [
                        (
                            "ble_disconnect",
                            TimeoutError(
                                "BLE client disconnect timed out and did not terminate"
                            ),
                        )
                    ]
                )

        form = Form()

        def hard_terminate(code):
            termination.append(code)
            raise HardTerminate()

        try:
            loop.call_soon(loop.stop)
            with self.assertRaises(HardTerminate):
                qt5_bleak._run_application_loop(
                    loop,
                    form,
                    shutdown_attempts=1,
                    shutdown_timeout_seconds=0.2,
                    hard_terminate=hard_terminate,
                )
            self.assertEqual(termination, [2])
            self.assertIn(
                "forced_process_boundary",
                [name for name, _ in form._last_shutdown_error.failures],
            )
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_cleanup_task_pending_forces_boundary_when_profile_inventory_is_clear(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        termination = []

        class HardTerminate(BaseException):
            pass

        async def pending_cleanup():
            await asyncio.Event().wait()

        cleanup_task = loop.create_task(pending_cleanup())
        timeout = TimeoutError("BLE client disconnect timed out and did not terminate")
        timeout.cleanup_pending = True
        timeout.cleanup_task = cleanup_task

        class Form:
            _quit_reason = "test"
            _last_shutdown_error = None
            _shutdown_task = None
            WG = SimpleNamespace(_pending_disconnect_task=None)

            async def shutdown(self, reason):
                raise qt5_bleak.ShutdownError([("ble_disconnect", timeout)])

        form = Form()

        def hard_terminate(code):
            termination.append(code)
            raise HardTerminate()

        try:
            self.assertIs(timeout.cleanup_task, cleanup_task)
            loop.call_soon(loop.stop)
            with self.assertRaises(HardTerminate):
                qt5_bleak._run_application_loop(
                    loop,
                    form,
                    shutdown_attempts=1,
                    shutdown_timeout_seconds=0.2,
                    hard_terminate=hard_terminate,
                )
            self.assertEqual(termination, [2])
            self.assertIn(
                "forced_process_boundary",
                [name for name, _ in form._last_shutdown_error.failures],
            )
        finally:
            cleanup_task.cancel()
            loop.run_until_complete(
                asyncio.gather(cleanup_task, return_exceptions=True)
            )
            loop.close()
            asyncio.set_event_loop(None)

    def test_cleanup_task_that_later_finishes_is_consumed_and_not_force_killed(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        gate = asyncio.Event()
        consumed = []
        termination = []

        async def finishing_cleanup():
            await gate.wait()
            raise OSError("late cleanup result")

        cleanup_task = loop.create_task(finishing_cleanup())
        timeout = TimeoutError("BLE client disconnect timed out and did not terminate")
        timeout.cleanup_pending = True
        timeout.cleanup_task = cleanup_task
        cleanup_task.add_done_callback(qt5_bleak.window._consume_ble_ui_task_result)
        cleanup_task.add_done_callback(lambda task: consumed.append(task))

        class Form:
            _quit_reason = "test"
            _last_shutdown_error = None
            _shutdown_task = None
            WG = SimpleNamespace(_pending_disconnect_task=None)

            async def shutdown(self, reason):
                raise qt5_bleak.ShutdownError([("ble_disconnect", timeout)])

        form = Form()
        try:
            self.assertIs(timeout.cleanup_task, cleanup_task)
            self.assertFalse(cleanup_task.done())
            gate.set()
            loop.run_until_complete(asyncio.sleep(0))
            self.assertTrue(cleanup_task.done())
            self.assertEqual(consumed, [cleanup_task])

            loop.call_soon(loop.stop)
            with self.assertRaises(qt5_bleak.ShutdownError) as raised:
                qt5_bleak._run_application_loop(
                    loop,
                    form,
                    shutdown_attempts=1,
                    shutdown_timeout_seconds=0.2,
                    hard_terminate=lambda code: termination.append(code),
                )
            self.assertEqual(termination, [])
            self.assertNotIn(
                "forced_process_boundary",
                [name for name, _ in raised.exception.failures],
            )
        finally:
            if not cleanup_task.done():
                cleanup_task.cancel()
                loop.run_until_complete(
                    asyncio.gather(cleanup_task, return_exceptions=True)
                )
            loop.close()
            asyncio.set_event_loop(None)

    def test_real_qasync_timeout_paths_drain_tasks_without_shutdown_warnings(self):
        code = textwrap.dedent(
            """
            import asyncio, os
            os.environ['QT_QPA_PLATFORM']='offscreen'
            from unittest.mock import AsyncMock, MagicMock, patch
            from PyQt5.QtWidgets import QApplication
            from qasync import QEventLoop
            import bleak_ble

            app=QApplication([]); loop=QEventLoop(app); asyncio.set_event_loop(loop)

            async def flow():
                locked=bleak_ble.BleakProfile('svc','svc','notify','cmd')
                locked._owner_loop=asyncio.get_running_loop()
                locked._operation_lock=asyncio.Lock()
                await locked._operation_lock.acquire()
                try:
                    await locked.disconnect()
                except TimeoutError as error:
                    assert 'operation lock' in str(error)
                else:
                    raise AssertionError('operation-lock timeout missing')
                locked._operation_lock.release()

                pending=bleak_ble.BleakProfile('svc','svc','notify','cmd')
                client=MagicMock(); client.is_connected=False
                async def disconnect_forever():
                    await asyncio.Event().wait()
                client.disconnect=AsyncMock(side_effect=disconnect_forever)
                pending._owner_loop=asyncio.get_running_loop()
                pending._operation_lock=asyncio.Lock()
                pending.client=client
                pending._pending_connection_generation=1
                pending._active_client_generation=1
                pending._active_attempt_epoch=1
                try:
                    await pending.disconnect()
                except TimeoutError as error:
                    assert 'disconnect timed out' in str(error)
                else:
                    raise AssertionError('backend timeout missing')
                assert pending._pending_disconnect_task is None
                assert pending.client is None
                await asyncio.sleep(0)

            with patch.object(bleak_ble,'CLIENT_DISCONNECT_TIMEOUT_SECONDS',.05):
                with loop: loop.run_until_complete(asyncio.wait_for(flow(),1))
            """
        )
        result = subprocess.run(
            [sys.executable, "-W", "error::ResourceWarning", "-c", code],
            cwd=str(qt5_bleak.BASE_DIR),
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Task was destroyed", result.stderr)
        self.assertNotIn("was never awaited", result.stderr)
        self.assertNotIn("ResourceWarning", result.stderr)

    def test_qasync_exit_harvests_backend_that_catches_cancel_and_keeps_waiting(self):
        code = textwrap.dedent(
            """
            import asyncio, os
            os.environ['QT_QPA_PLATFORM']='offscreen'
            from unittest.mock import AsyncMock, MagicMock, patch
            from PyQt5 import QtCore
            from PyQt5.QtWidgets import QApplication
            from qasync import QEventLoop
            import bleak_ble, qt5_bleak

            app=QApplication([]); loop=QEventLoop(app); asyncio.set_event_loop(loop)
            profile=bleak_ble.BleakProfile('svc','svc','notify','cmd')
            client=MagicMock(); client.is_connected=False
            cancellations=[]
            async def disconnect_after_cancellation():
                target=asyncio.get_running_loop().time()+.05
                while True:
                    remaining=target-asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        await asyncio.sleep(remaining)
                    except asyncio.CancelledError:
                        cancellations.append('caught')
                client.is_connected=False
            client.disconnect=AsyncMock(side_effect=disconnect_after_cancellation)
            profile.client=client
            profile._pending_connection_generation=1
            profile._active_client_generation=1
            profile._active_attempt_epoch=1

            class Form:
                _quit_reason='test'
                _last_shutdown_error=None
                WG=profile
                async def shutdown(self, reason):
                    try:
                        await profile.disconnect()
                    except TimeoutError as error:
                        raise qt5_bleak.ShutdownError([('ble_disconnect', error)])

            form=Form()
            QtCore.QTimer.singleShot(0, app.quit)
            caught=None
            with patch.object(bleak_ble,'CLIENT_DISCONNECT_TIMEOUT_SECONDS',.02):
                try:
                    with loop:
                        qt5_bleak._run_application_loop(
                            loop, form, shutdown_attempts=1,
                            shutdown_timeout_seconds=.2,
                        )
                except qt5_bleak.ShutdownError as error:
                    caught=error
            assert caught is not None
            assert cancellations
            assert profile._pending_disconnect_task is None
            assert not qt5_bleak._pending_application_tasks(Form())
            """
        )
        result = subprocess.run(
            [sys.executable, "-W", "error::ResourceWarning", "-c", code],
            cwd=str(qt5_bleak.BASE_DIR),
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Task was destroyed", result.stderr)
        self.assertNotIn("was never awaited", result.stderr)
        self.assertNotIn("ResourceWarning", result.stderr)

    def test_qasync_exit_marks_force_boundary_for_backend_refusing_cancellation(self):
        code = textwrap.dedent(
            """
            import asyncio, os, threading
            os.environ['QT_QPA_PLATFORM']='offscreen'
            from unittest.mock import AsyncMock, MagicMock, patch
            from PyQt5 import QtCore
            from PyQt5.QtWidgets import QApplication
            from qasync import QEventLoop
            import bleak_ble, qt5_bleak

            app=QApplication([]); loop=QEventLoop(app); asyncio.set_event_loop(loop)
            profile=bleak_ble.BleakProfile('svc','svc','notify','cmd')
            client=MagicMock(); client.is_connected=False
            cancel_handled=threading.Event()
            started=asyncio.Event()
            never=loop.create_future()
            async def refuse_cancellation():
                started.set()
                while True:
                    try:
                        await asyncio.shield(never)
                    except asyncio.CancelledError:
                        cancel_handled.set()
                        continue
            client.disconnect=AsyncMock(side_effect=refuse_cancellation)
            profile.client=client
            profile._pending_connection_generation=1
            profile._active_client_generation=1
            profile._active_attempt_epoch=1

            class Form:
                _quit_reason='test'
                _last_shutdown_error=None
                WG=profile
                backend_task=None
                async def shutdown(self, reason):
                    self.backend_task=asyncio.create_task(client.disconnect())
                    await started.wait()
                    raise qt5_bleak.ShutdownError([
                        ('ble_disconnect', TimeoutError('backend refused disconnect'))
                    ])

            form=Form()
            # Stop the asyncio/qasync loop without destroying QApplication;
            # destroying Qt first can erase QThread-backed pending-work
            # evidence before the application inventory is inspected.
            loop.call_soon(loop.stop)
            class HardTerminate(BaseException): pass
            termination=[]
            unreachable=[]
            def hard_terminate(code):
                assert cancel_handled.is_set()
                termination.append(code)
                raise HardTerminate()
            try:
                with loop:
                    qt5_bleak._run_application_loop(
                        loop, form, shutdown_attempts=1,
                        shutdown_timeout_seconds=.2,
                        hard_terminate=hard_terminate,
                    )
                    unreachable.append('returned')
            except HardTerminate:
                pass
            assert termination == [2]
            assert not unreachable
            assert any(
                stage == 'forced_process_boundary'
                for stage, _ in form._last_shutdown_error.failures
            )
            """
        )
        result = subprocess.run(
            [sys.executable, "-W", "error::ResourceWarning", "-c", code],
            cwd=str(qt5_bleak.BASE_DIR),
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Task was destroyed", result.stderr)
        self.assertNotIn("was never awaited", result.stderr)
        self.assertNotIn("ResourceWarning", result.stderr)

    def test_qasync_top_level_shutdown_refusing_cancel_reaches_hard_boundary(self):
        code = textwrap.dedent(
            """
            import asyncio, os, threading
            os.environ['QT_QPA_PLATFORM']='offscreen'
            from PyQt5 import QtCore
            from PyQt5.QtWidgets import QApplication
            from qasync import QEventLoop
            import qt5_bleak

            app=QApplication([]); loop=QEventLoop(app); asyncio.set_event_loop(loop)
            class HardTerminate(BaseException): pass
            events=[]
            cancel_handled=threading.Event()
            never=loop.create_future()
            class Form:
                _quit_reason='test'
                _last_shutdown_error=None
                _shutdown_task=None
                _shutdown_pending_tasks=set()
                _shutdown_blocking_work={}
                _ble_ui_tasks=set()
                attempts=0
                async def shutdown(self, reason, *, deadline):
                    self.attempts += 1
                    while True:
                        try:
                            await asyncio.shield(never)
                        except asyncio.CancelledError:
                            cancel_handled.set()
                            events.append('cancel-caught')
            form=Form()
            def hard_terminate(code):
                assert cancel_handled.is_set()
                events.append(('terminate', code))
                raise HardTerminate()
            QtCore.QTimer.singleShot(0, app.quit)
            try:
                with loop:
                    qt5_bleak._run_application_loop(
                        loop, form, shutdown_attempts=3,
                        shutdown_timeout_seconds=.2,
                        hard_terminate=hard_terminate,
                    )
                    events.append('unreachable')
            except HardTerminate:
                pass
            assert form.attempts == 1
            assert 'cancel-caught' in events
            assert ('terminate', 2) in events
            assert 'unreachable' not in events
            """
        )
        result = subprocess.run(
            [sys.executable, "-W", "error::ResourceWarning", "-c", code],
            cwd=str(qt5_bleak.BASE_DIR),
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Task was destroyed", result.stderr)
        self.assertNotIn("was never awaited", result.stderr)
        self.assertNotIn("ResourceWarning", result.stderr)

    def test_qasync_inventory_captures_unregistered_child_spawned_on_cancel(self):
        code = textwrap.dedent(
            """
            import asyncio, os
            os.environ['QT_QPA_PLATFORM']='offscreen'
            from PyQt5 import QtCore
            from PyQt5.QtWidgets import QApplication
            from qasync import QEventLoop
            import qt5_bleak

            app=QApplication([]); loop=QEventLoop(app); asyncio.set_event_loop(loop)
            class HardTerminate(BaseException): pass
            events=[]
            never=loop.create_future()
            async def stubborn_child():
                events.append('child-started')
                while True:
                    try:
                        await asyncio.shield(never)
                    except asyncio.CancelledError:
                        events.append('child-cancel-caught')
            class Form:
                _quit_reason='test'
                _last_shutdown_error=None
                _shutdown_task=None
                _shutdown_pending_tasks=set()
                _shutdown_blocking_work={}
                _ble_ui_tasks=set()
                attempts=0
                async def shutdown(self, reason, *, deadline):
                    self.attempts += 1
                    try:
                        await asyncio.shield(never)
                    except asyncio.CancelledError:
                        asyncio.create_task(stubborn_child())
                        await asyncio.sleep(0)
                        events.append('parent-cancel-handled')
                        raise
            form=Form()
            termination=[]
            def hard_terminate(code):
                termination.append(code)
                raise HardTerminate()
            QtCore.QTimer.singleShot(0, app.quit)
            try:
                with loop:
                    qt5_bleak._run_application_loop(
                        loop, form, shutdown_attempts=3,
                        shutdown_timeout_seconds=.2,
                        hard_terminate=hard_terminate,
                    )
                    events.append('unreachable')
            except HardTerminate:
                pass
            assert form.attempts == 1
            assert 'parent-cancel-handled' in events
            assert 'child-started' in events
            assert 'child-cancel-caught' in events
            assert termination == [2]
            assert 'unreachable' not in events
            assert any(
                stage == 'forced_process_boundary'
                for stage, _ in form._last_shutdown_error.failures
            )
            """
        )
        result = subprocess.run(
            [sys.executable, "-W", "error::ResourceWarning", "-c", code],
            cwd=str(qt5_bleak.BASE_DIR),
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Task was destroyed", result.stderr)
        self.assertNotIn("was never awaited", result.stderr)
        self.assertNotIn("ResourceWarning", result.stderr)

    def test_real_headless_app_quit_runs_shutdown_before_loop_close(self):
        code = textwrap.dedent(
            """
            import asyncio, os
            os.environ['QT_QPA_PLATFORM']='offscreen'
            from PyQt5 import QtCore, QtWidgets
            from qasync import QEventLoop
            import qt5_bleak

            app = QtWidgets.QApplication([])
            loop = QEventLoop(app)
            asyncio.set_event_loop(loop)
            events = []

            class Form:
                _quit_reason = None
                attempts = 0
                def note_shutdown_request(self, reason):
                    events.append(('request', reason))
                    self._quit_reason = reason
                async def shutdown(self, reason):
                    self.attempts += 1
                    events.append(('shutdown', reason, loop.is_closed()))
                    if self.attempts == 1:
                        raise qt5_bleak.ShutdownError([('recorder_stop', OSError('busy'))])

            form = Form()
            app.aboutToQuit.connect(lambda: form.note_shutdown_request('about_to_quit'))
            QtCore.QTimer.singleShot(0, app.quit)
            with loop:
                qt5_bleak._run_application_loop(loop, form)
            shutdown_events = [event for event in events if event[0] == 'shutdown']
            assert shutdown_events == [
                ('shutdown', 'about_to_quit', False),
                ('shutdown', 'about_to_quit', False),
            ], events
            assert events[0] == ('request', 'about_to_quit'), events
            assert all(
                event == ('request', 'about_to_quit')
                for event in events if event[0] == 'request'
            ), events
            assert loop.is_closed()
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(qt5_bleak.BASE_DIR),
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_headless_app_quit_bounds_permanent_shutdown_failures(self):
        code = textwrap.dedent(
            """
            import asyncio, os
            os.environ['QT_QPA_PLATFORM']='offscreen'
            from PyQt5 import QtCore, QtWidgets
            from qasync import QEventLoop
            import qt5_bleak

            app = QtWidgets.QApplication([])
            loop = QEventLoop(app)
            asyncio.set_event_loop(loop)

            class Form:
                _quit_reason = None
                _last_shutdown_error = None
                attempts = 0
                def note_shutdown_request(self, reason):
                    self._quit_reason = reason
                async def shutdown(self, reason):
                    self.attempts += 1
                    raise qt5_bleak.ShutdownError([
                        ('recorder_stop', OSError(f'busy-{self.attempts}')),
                    ])

            form = Form()
            app.aboutToQuit.connect(lambda: form.note_shutdown_request('about_to_quit'))
            QtCore.QTimer.singleShot(0, app.quit)
            caught = None
            try:
                with loop:
                    qt5_bleak._run_application_loop(
                        loop,
                        form,
                        shutdown_attempts=2,
                        shutdown_timeout_seconds=.5,
                    )
            except qt5_bleak.ShutdownError as error:
                caught = error
            assert caught is form._last_shutdown_error
            assert [stage for stage, error in caught.failures] == [
                'attempt_1.recorder_stop', 'attempt_2.recorder_stop'
            ], caught.failures
            assert form.attempts == 2
            assert loop.is_closed()
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(qt5_bleak.BASE_DIR),
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("EMG shutdown failed after bounded retries", result.stderr)


if __name__ == "__main__":
    unittest.main()
