import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import bleak_ble


SERVICE_UUID = "service-uuid"
NOTIFY_UUID = "notify-uuid"
COMMAND_UUID = "command-uuid"


class FakeService:
    def __init__(self, characteristics=None):
        self.characteristics = characteristics or {}

    def get_characteristic(self, uuid):
        return self.characteristics.get(uuid)


def make_profile():
    return bleak_ble.BleakProfile(
        SERVICE_UUID, SERVICE_UUID, NOTIFY_UUID, COMMAND_UUID
    )


def configured_client():
    notify_characteristic = SimpleNamespace(handle=1)
    command_characteristic = SimpleNamespace(handle=2)
    service = FakeService(
        {
            NOTIFY_UUID: notify_characteristic,
            COMMAND_UUID: command_characteristic,
        }
    )
    client = MagicMock()
    client.is_connected = True
    client.connect = AsyncMock()
    async def disconnect():
        client.is_connected = False

    client.disconnect = AsyncMock(side_effect=disconnect)
    client.start_notify = AsyncMock()
    client.stop_notify = AsyncMock()
    client.write_gatt_char = AsyncMock()
    client.services.get_service.return_value = service
    return client, notify_characteristic, command_characteristic


class BleakProfileTests(unittest.IsolatedAsyncioTestCase):
    def test_correct_characteristic_alias_keeps_legacy_name(self):
        self.assertEqual(
            bleak_ble.characteristic_uuid_2, bleak_ble.chararcteristic_uuid_2
        )
        self.assertEqual(
            bleak_ble.characteristic_uuid_3, bleak_ble.chararcteristic_uuid_3
        )

    async def test_scan_returns_stable_entries_and_ignores_missing_uuids(self):
        devices = [
            SimpleNamespace(name="unknown", address="A", rssi=-80, metadata=None),
            SimpleNamespace(name="empty", address="B", rssi=-70, metadata={}),
            SimpleNamespace(
                name="target",
                address="C",
                rssi=-60,
                metadata={"uuids": [SERVICE_UUID.upper()]},
            ),
        ]
        class Scanner:
            async def discover(self, timeout=5.0):
                return devices

        with patch.object(bleak_ble, "BleakScanner", Scanner):
            snapshot = await make_profile().scan(0.01)

        self.assertIsInstance(snapshot, bleak_ble.ScanSnapshot)
        self.assertEqual([device.address for device in snapshot.matches], ["C"])
        self.assertEqual(
            [device.address for device in snapshot.devices], ["A", "B", "C"]
        )
        self.assertIs(snapshot.matches[0].native_device, devices[2])
        with self.assertRaises(AttributeError):
            snapshot.devices = ()

    async def test_bleak_3_scan_uses_advertisement_uuids_and_rssi(self):
        class Scanner3:
            return_adv_received = None

            async def discover(self, timeout=5.0, return_adv=False):
                Scanner3.return_adv_received = return_adv
                device = SimpleNamespace(name="target", address="C")
                advertisement = SimpleNamespace(
                    service_uuids=[SERVICE_UUID], rssi=-42
                )
                return {"C": (device, advertisement)}

        with patch.object(bleak_ble, "BleakScanner", Scanner3):
            snapshot = await make_profile().scan(0.01)

        self.assertTrue(Scanner3.return_adv_received)
        self.assertEqual([device.address for device in snapshot.matches], ["C"])
        self.assertEqual(snapshot.devices, snapshot.matches)

    async def test_bleak_011_scan_uses_device_metadata_and_rssi(self):
        class Scanner011:
            async def discover(self, timeout=5.0):
                return [
                    SimpleNamespace(
                        name="target",
                        address="L",
                        rssi=-55,
                        metadata={"uuids": [SERVICE_UUID]},
                    )
                ]

        with patch.object(bleak_ble, "BleakScanner", Scanner011):
            snapshot = await make_profile().scan(0.01)

        self.assertEqual([device.address for device in snapshot.matches], ["L"])
        self.assertEqual(snapshot.devices, snapshot.matches)

    async def test_scan_returns_every_device_advertising_target_service(self):
        devices = [
            SimpleNamespace(
                name="first",
                address="A",
                rssi=-60,
                metadata={"uuids": [SERVICE_UUID]},
            ),
            SimpleNamespace(
                name="other",
                address="B",
                rssi=-40,
                metadata={"uuids": ["other-service"]},
            ),
            SimpleNamespace(
                name="second",
                address="C",
                rssi=-80,
                metadata={"uuids": [SERVICE_UUID.upper()]},
            ),
        ]

        class Scanner:
            async def discover(self, timeout=5.0):
                return devices

        with patch.object(bleak_ble, "BleakScanner", Scanner):
            snapshot = await make_profile().scan(0.01)

        self.assertEqual([device.address for device in snapshot.matches], ["A", "C"])
        self.assertEqual(len(snapshot.devices), 3)

    async def test_cancelled_connect_disconnects_partial_client_and_resets_attempt(self):
        profile = make_profile()
        connect_started = asyncio.Event()
        release_connect = asyncio.Event()
        client, _, _ = configured_client()

        async def connect():
            connect_started.set()
            await release_connect.wait()

        client.connect = AsyncMock(side_effect=connect)
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            task = asyncio.create_task(profile.connect("device"))
            await connect_started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        client.disconnect.assert_awaited_once()
        self.assertIsNone(profile.client)
        self.assertIsNone(profile._pending_connection_generation)
        self.assertIsNone(profile._active_client_generation)
        self.assertIsNone(profile._active_attempt_epoch)
        self.assertFalse(profile.is_connected)

    async def test_cancelled_connect_preserves_cancel_when_cleanup_fails(self):
        profile = make_profile()
        connect_started = asyncio.Event()
        client, _, _ = configured_client()

        async def connect():
            connect_started.set()
            await asyncio.Event().wait()

        client.connect = AsyncMock(side_effect=connect)
        client.disconnect = AsyncMock(side_effect=OSError("cleanup failed"))
        with self.assertLogs(bleak_ble.logger, level="ERROR") as logs:
            with patch.object(bleak_ble, "BleakClient", return_value=client):
                task = asyncio.create_task(profile.connect("device"))
                await connect_started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        self.assertTrue(any("cancelled connection failed" in row for row in logs.output))
        self.assertIsNone(profile.client)
        self.assertIsNone(profile._pending_connection_generation)
        self.assertIsNone(profile._active_client_generation)
        self.assertIsNone(profile._active_attempt_epoch)

    async def test_repeated_cancellation_cannot_interrupt_partial_client_cleanup(self):
        profile = make_profile()
        connect_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        client, _, _ = configured_client()

        async def connect():
            connect_started.set()
            await asyncio.Event().wait()

        async def disconnect():
            cleanup_started.set()
            await release_cleanup.wait()
            client.is_connected = False

        client.connect = AsyncMock(side_effect=connect)
        client.disconnect = AsyncMock(side_effect=disconnect)
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            task = asyncio.create_task(profile.connect("device"))
            await connect_started.wait()
            task.cancel()
            await cleanup_started.wait()
            task.cancel()
            release_cleanup.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        client.disconnect.assert_awaited_once()
        self.assertIsNone(profile.client)
        self.assertIsNone(profile._pending_connection_generation)
        self.assertIsNone(profile._active_client_generation)
        self.assertIsNone(profile._active_attempt_epoch)

    async def test_cancelled_connect_cleanup_timeout_preserves_original_cancel(self):
        profile = make_profile()
        connect_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        client, _, _ = configured_client()

        async def connect_forever():
            connect_started.set()
            await asyncio.Event().wait()

        async def disconnect_forever():
            cleanup_started.set()
            await asyncio.Event().wait()

        client.connect = AsyncMock(side_effect=connect_forever)
        client.disconnect = AsyncMock(side_effect=disconnect_forever)
        with self.assertLogs(bleak_ble.logger, level="ERROR") as logs:
            with patch.object(bleak_ble, "CLIENT_DISCONNECT_TIMEOUT_SECONDS", 0.05):
                with patch.object(bleak_ble, "BleakClient", return_value=client):
                    task = asyncio.create_task(profile.connect("device"))
                    await connect_started.wait()
                    task.cancel()
                    await cleanup_started.wait()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, timeout=0.2)

        self.assertTrue(any("TimeoutError" in row for row in logs.output))
        self.assertIsNone(profile.client)
        self.assertIsNone(profile._pending_connection_generation)
        self.assertIsNone(profile._active_client_generation)
        self.assertIsNone(profile._active_attempt_epoch)

    async def test_disconnect_cleans_nonconnected_pending_client(self):
        profile = make_profile()
        client, _, _ = configured_client()
        client.is_connected = False
        profile._owner_loop = asyncio.get_running_loop()
        profile._operation_lock = asyncio.Lock()
        profile.client = client
        profile._pending_connection_generation = 1
        profile._active_client_generation = 1
        profile._active_attempt_epoch = 1

        self.assertTrue(await profile.disconnect())

        client.disconnect.assert_awaited_once()
        self.assertIsNone(profile.client)
        self.assertIsNone(profile._pending_connection_generation)
        self.assertIsNone(profile._active_client_generation)
        self.assertIsNone(profile._active_attempt_epoch)

    async def test_pending_disconnect_timeout_is_bounded_and_state_is_cleared(self):
        profile = make_profile()
        client, _, _ = configured_client()
        client.is_connected = False

        async def disconnect_forever():
            await asyncio.Event().wait()

        client.disconnect = AsyncMock(side_effect=disconnect_forever)
        profile._owner_loop = asyncio.get_running_loop()
        profile._operation_lock = asyncio.Lock()
        profile.client = client
        profile._pending_connection_generation = 1
        profile._active_client_generation = 1
        profile._active_attempt_epoch = 1

        with patch.object(bleak_ble, "CLIENT_DISCONNECT_TIMEOUT_SECONDS", 0.05):
            with self.assertRaisesRegex(TimeoutError, "disconnect timed out"):
                await asyncio.wait_for(profile.disconnect(), timeout=0.2)

        self.assertIsNone(profile.client)
        self.assertIsNone(profile._pending_connection_generation)
        self.assertIsNone(profile._active_client_generation)
        self.assertIsNone(profile._active_attempt_epoch)

    async def test_uncooperative_disconnect_is_reused_and_blocks_reconnect(self):
        profile = make_profile()
        client, _, _ = configured_client()
        client.is_connected = False
        release = asyncio.Event()
        disconnect_calls = 0

        async def uncooperative_disconnect():
            nonlocal disconnect_calls
            disconnect_calls += 1
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue

        client.disconnect = AsyncMock(side_effect=uncooperative_disconnect)
        profile._owner_loop = asyncio.get_running_loop()
        profile._operation_lock = asyncio.Lock()
        profile.client = client
        profile._pending_connection_generation = 1
        profile._active_client_generation = 1
        profile._active_attempt_epoch = 1

        with patch.object(bleak_ble, "CLIENT_DISCONNECT_TIMEOUT_SECONDS", 0.03):
            with self.assertRaisesRegex(TimeoutError, "did not terminate"):
                await profile.disconnect()
            first_task = profile._pending_disconnect_task
            self.assertIsNotNone(first_task)
            with self.assertRaisesRegex(TimeoutError, "did not terminate"):
                await profile.disconnect()
            self.assertIs(profile._pending_disconnect_task, first_task)
            self.assertEqual(disconnect_calls, 1)
            with patch.object(bleak_ble, "BleakClient") as factory:
                with self.assertRaisesRegex(RuntimeError, "cleanup is still pending"):
                    await profile.connect("new-device")
            factory.assert_not_called()

        release.set()
        await asyncio.wait_for(first_task, timeout=0.2)
        await asyncio.sleep(0)
        self.assertIsNone(profile._pending_disconnect_task)
        self.assertIsNone(profile._pending_disconnect_client)

    async def test_disconnect_deadline_includes_operation_lock_wait(self):
        profile = make_profile()
        client, _, _ = configured_client()
        profile.client = client
        profile.connected = 1
        profile._owner_loop = asyncio.get_running_loop()
        profile._operation_lock = asyncio.Lock()
        await profile._operation_lock.acquire()

        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "operation lock"):
            await asyncio.wait_for(
                profile.disconnect(deadline=time.monotonic() + 0.02), timeout=0.2
            )
        self.assertLess(time.monotonic() - started, 0.15)

        client.disconnect.assert_not_awaited()
        self.assertIs(profile.client, client)
        profile._operation_lock.release()

    async def test_cancelled_disconnect_finishes_backend_cleanup_then_reraises(self):
        profile = make_profile()
        client, _, _ = configured_client()
        disconnect_started = asyncio.Event()

        async def disconnect():
            disconnect_started.set()
            await asyncio.sleep(0.01)
            client.is_connected = False

        client.disconnect = AsyncMock(side_effect=disconnect)
        profile.client = client
        profile.connected = 1
        profile._active_client_generation = 1
        profile._active_attempt_epoch = 1

        with patch.object(bleak_ble, "CLIENT_DISCONNECT_TIMEOUT_SECONDS", 0.1):
            task = asyncio.create_task(profile.disconnect())
            await disconnect_started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=0.2)

        self.assertFalse(profile.is_connected)
        self.assertIsNone(profile.client)

    async def test_cancelled_disconnect_attaches_cleanup_error_without_rewriting_cancel(self):
        profile = make_profile()
        client, _, _ = configured_client()
        disconnect_started = asyncio.Event()
        observed_cancellation = []

        async def disconnect():
            disconnect_started.set()
            await asyncio.sleep(0.01)
            raise OSError("backend cleanup failed")

        async def invoke_disconnect():
            try:
                await profile.disconnect()
            except asyncio.CancelledError as exc:
                observed_cancellation.append(exc)
                raise

        client.disconnect = AsyncMock(side_effect=disconnect)
        profile.client = client
        profile.connected = 1
        profile._active_client_generation = 1
        profile._active_attempt_epoch = 1

        with patch.object(bleak_ble, "CLIENT_DISCONNECT_TIMEOUT_SECONDS", 0.1):
            task = asyncio.create_task(invoke_disconnect())
            await disconnect_started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=0.2)

        self.assertEqual(len(observed_cancellation), 1)
        cleanup_failures = observed_cancellation[0].cleanup_failures
        self.assertEqual(cleanup_failures[0][0], "client_disconnect")
        self.assertIsInstance(cleanup_failures[0][1], OSError)

    async def test_connect_success_resolves_state_and_is_idempotent(self):
        profile = make_profile()
        client, notify_characteristic, command_characteristic = configured_client()
        signal = SimpleNamespace(emit=MagicMock())
        with patch.object(bleak_ble, "BleakClient", return_value=client) as factory:
            with self.assertWarns(DeprecationWarning):
                self.assertTrue(await profile.connect("device", signal))
            await asyncio.sleep(0)
            self.assertFalse(await profile.connect("device", signal))

        factory.assert_called_once()
        client.connect.assert_awaited_once()
        self.assertTrue(profile.is_connected)
        self.assertFalse(profile.is_notifying)
        self.assertIs(profile.notifyCharacteristic, notify_characteristic)
        self.assertIs(profile.cmdCharacteristic, command_characteristic)
        signal.emit.assert_called_once_with("connected successfully")

    async def test_legacy_ui_callback_failure_does_not_rollback_connection(self):
        profile = make_profile()
        client, _, _ = configured_client()
        signal = SimpleNamespace(emit=MagicMock(side_effect=RuntimeError("UI closed")))
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            with self.assertWarns(DeprecationWarning):
                self.assertTrue(await profile.connect("device", signal))
            await asyncio.sleep(0)

        self.assertTrue(profile.is_connected)
        self.assertIs(profile.client, client)
        client.disconnect.assert_not_awaited()

    async def test_unexpected_disconnect_publishes_state_event(self):
        profile = make_profile()
        client, _, _ = configured_client()
        callback = None
        events = []
        listener_threads = []

        def factory(device, disconnected_callback):
            nonlocal callback
            callback = disconnected_callback
            return client

        def listener(event):
            events.append(event)
            listener_threads.append(threading.get_ident())

        profile.add_state_listener(listener)
        with patch.object(bleak_ble, "BleakClient", side_effect=factory):
            await profile.connect("device")
        await asyncio.sleep(0)
        events.clear()
        listener_threads.clear()

        client.is_connected = False
        callback_thread = await asyncio.to_thread(
            lambda: (threading.get_ident(), callback(client))[0]
        )
        await asyncio.sleep(0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "disconnected")
        self.assertEqual(events[0].reason, "unexpected")
        self.assertFalse(events[0].connected)
        self.assertFalse(events[0].notifying)
        self.assertNotEqual(callback_thread, threading.get_ident())
        self.assertEqual(listener_threads, [threading.get_ident()])

    async def test_listener_runs_outside_lock_and_failure_does_not_pollute_state(self):
        profile = make_profile()
        client, _, _ = configured_client()
        received = []

        def reentrant_listener(event):
            self.assertFalse(profile.lock._is_owned())
            received.append(event)
            profile.remove_state_listener(reentrant_listener)
            profile.add_state_listener(reentrant_listener)

        def failing_listener(event):
            self.assertFalse(profile.lock._is_owned())
            raise RuntimeError("listener failed")

        profile.add_state_listener(reentrant_listener)
        profile.add_state_listener(failing_listener)
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")
        await asyncio.sleep(0)

        self.assertEqual([event.event_type for event in received], ["connected"])
        self.assertTrue(profile.is_connected)
        self.assertIs(profile.client, client)

    async def test_connected_event_and_legacy_callback_are_suppressed_by_disconnect_race(self):
        profile = make_profile()
        client, _, _ = configured_client()
        callback = None
        events = []
        signal = SimpleNamespace(emit=MagicMock())

        def factory(device, disconnected_callback):
            nonlocal callback
            callback = disconnected_callback
            return client

        profile.add_state_listener(events.append)
        with patch.object(bleak_ble, "BleakClient", side_effect=factory):
            with self.assertWarns(DeprecationWarning):
                self.assertTrue(await profile.connect("device", signal))
            client.is_connected = False
            callback(client)
        await asyncio.sleep(0)

        self.assertEqual(
            [event.event_type for event in events], ["connected", "disconnected"]
        )
        self.assertTrue(events[0].connected)
        self.assertFalse(events[1].connected)
        signal.emit.assert_not_called()

    async def test_state_event_order_snapshots_thread_and_async_listener_reentry(self):
        profile = make_profile()
        client, _, _ = configured_client()
        profile.set_notification_handler(MagicMock())
        events = []
        listener_threads = []

        def listener(event):
            events.append(event)
            listener_threads.append(threading.get_ident())
            if event.event_type == "connected":
                asyncio.create_task(profile.setNotify(True))

        profile.add_state_listener(listener)
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await profile.disconnect()
        await asyncio.sleep(0)

        self.assertEqual(
            [event.event_type for event in events],
            ["connected", "notify_changed", "disconnected"],
        )
        self.assertEqual(
            [(event.connected, event.notifying) for event in events],
            [(True, False), (True, True), (False, False)],
        )
        self.assertTrue(all(ident == threading.get_ident() for ident in listener_threads))
        self.assertEqual(events[0].generation, events[1].generation)
        self.assertEqual(events[1].generation, events[2].generation)

    def test_stopped_owner_loop_drops_event_without_backend_thread_delivery(self):
        class StoppedLoop:
            def is_running(self):
                return False

        profile = make_profile()
        client = SimpleNamespace(is_connected=False)
        profile._owner_loop = StoppedLoop()
        profile.client = client
        profile.connected = 1
        profile._client_generation = 4
        profile._active_client_generation = 4
        profile._active_attempt_epoch = 4
        listener = MagicMock()
        profile.add_state_listener(listener)

        profile._handle_disconnect(client, 4, 4)

        listener.assert_not_called()
        self.assertEqual(list(profile._pending_state_events), [])

    def test_owner_loop_close_race_drops_event_without_direct_delivery(self):
        class ClosingLoop:
            def is_running(self):
                return True

            def call_soon_threadsafe(self, *args):
                raise RuntimeError("loop closed")

        profile = make_profile()
        client = SimpleNamespace(is_connected=False)
        profile._owner_loop = ClosingLoop()
        profile.client = client
        profile.connected = 1
        profile._client_generation = 7
        profile._active_client_generation = 7
        profile._active_attempt_epoch = 7
        listener = MagicMock()
        profile.add_state_listener(listener)

        profile._handle_disconnect(client, 7, 7)

        listener.assert_not_called()
        self.assertFalse(profile._event_delivery_scheduled)
        self.assertEqual(list(profile._pending_state_events), [])

    def test_stopped_loop_clears_previously_scheduled_events_before_restart(self):
        class PausedLoop:
            def __init__(self):
                self.running = True
                self.callbacks = []

            def is_running(self):
                return self.running

            def call_soon_threadsafe(self, callback, *args):
                self.callbacks.append((callback, args))

        profile = make_profile()
        loop = PausedLoop()
        client = SimpleNamespace(is_connected=True)
        listener = MagicMock()
        profile._owner_loop = loop
        profile.client = client
        profile.connected = 1
        profile._client_generation = 9
        profile._active_client_generation = 9
        profile._active_attempt_epoch = 9
        profile.add_state_listener(listener)

        with profile.lock:
            profile._enqueue_state_event_locked("connected", generation=9)
        self.assertTrue(profile._event_delivery_scheduled)
        self.assertEqual(len(profile._pending_state_events), 1)
        self.assertEqual(len(loop.callbacks), 1)

        loop.running = False
        client.is_connected = False
        profile._handle_disconnect(client, 9, 9)
        self.assertFalse(profile._event_delivery_scheduled)
        self.assertEqual(list(profile._pending_state_events), [])

        loop.running = True
        callback, args = loop.callbacks.pop(0)
        callback(*args)
        listener.assert_not_called()
        self.assertEqual(list(profile._pending_state_events), [])

    async def test_bleak_011_dual_state_object_is_evaluated_as_boolean(self):
        class DeprecatedIsConnectedReturn:
            def __init__(self, value):
                self.value = value

            def __bool__(self):
                return self.value

            def __await__(self):
                async def result():
                    return self.value

                return result().__await__()

        profile = make_profile()
        client, _, _ = configured_client()
        client.is_connected = DeprecatedIsConnectedReturn(True)
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            self.assertTrue(await profile.connect("device"))
            self.assertFalse(await profile.connect("device"))

        client.connect.assert_awaited_once()
        self.assertTrue(profile.is_connected)

    async def test_connect_failure_resets_state_and_disconnects_partial_connection(self):
        profile = make_profile()
        client, _, _ = configured_client()
        client.services.get_service.return_value = None
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "notification service not found"):
                await profile.connect("device")

        client.disconnect.assert_awaited_once()
        self.assertFalse(profile.is_connected)
        self.assertFalse(profile.is_notifying)
        self.assertIsNone(profile.notifyCharacteristic)
        self.assertIsNone(profile.cmdCharacteristic)

    async def test_client_connect_exception_leaves_clean_state(self):
        profile = make_profile()
        client, _, _ = configured_client()
        client.connect.side_effect = OSError("adapter unavailable")
        client.is_connected = False
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            with self.assertRaisesRegex(OSError, "adapter unavailable"):
                await profile.connect("device")

        client.disconnect.assert_awaited_once()
        self.assertFalse(profile.is_connected)
        self.assertFalse(profile.is_notifying)

    async def test_failed_connect_does_not_consume_success_generation(self):
        profile = make_profile()
        failed_client, _, _ = configured_client()
        failed_client.connect.side_effect = OSError("failed")
        good_client, _, _ = configured_client()
        with patch.object(
            bleak_ble, "BleakClient", side_effect=[failed_client, good_client]
        ):
            with self.assertRaises(OSError):
                await profile.connect("failed")
            self.assertEqual(profile.client_generation, 0)
            await profile.connect("good")

        self.assertEqual(profile.client_generation, 1)

    async def test_failed_attempt_callbacks_cannot_tear_down_same_client_success(self):
        profile = make_profile()
        client, _, _ = configured_client()
        disconnect_callbacks = []
        received = []

        def factory(device, disconnected_callback):
            disconnect_callbacks.append(disconnected_callback)
            return client

        profile.add_notification_listener(received.append)
        client.connect.side_effect = OSError("first attempt failed")
        with patch.object(bleak_ble, "BleakClient", side_effect=factory):
            with self.assertRaises(OSError):
                await profile.connect("failed")
            failed_notify_callback = profile._make_notification_callback(
                client, 1, profile._attempt_epoch
            )

            client.connect.side_effect = None
            client.is_connected = True
            await profile.connect("success")
            successful_attempt = profile._active_attempt_epoch

        disconnect_callbacks[0](client)
        failed_notify_callback("stale", b"old")
        await asyncio.sleep(0)

        self.assertEqual(profile.client_generation, 1)
        self.assertEqual(profile._attempt_epoch, 2)
        self.assertEqual(successful_attempt, 2)
        self.assertTrue(profile.is_connected)
        self.assertIs(profile.client, client)
        self.assertEqual(received, [])
        self.assertEqual(profile.dropped_notification_count, 1)

    async def test_notify_handler_must_be_callable(self):
        profile = make_profile()
        with self.assertRaises(TypeError):
            profile.set_notification_handler("not callable")

        client, _, _ = configured_client()
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")
        with self.assertRaisesRegex(RuntimeError, "handler"):
            await profile.setNotify(True)

    async def test_notify_start_is_idempotent_and_stop_uses_characteristic_uuid(self):
        profile = make_profile()
        handler = MagicMock()
        profile.set_notification_handler(handler)
        client, _, _ = configured_client()
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")

        self.assertTrue(await profile.setNotify(True))
        self.assertFalse(await profile.setNotify(True))
        client.start_notify.assert_awaited_once()
        self.assertEqual(client.start_notify.await_args.args[0], NOTIFY_UUID)
        self.assertTrue(callable(client.start_notify.await_args.args[1]))
        self.assertTrue(profile.is_notifying)

        self.assertTrue(await profile.setNotify(False))
        self.assertFalse(await profile.setNotify(False))
        client.stop_notify.assert_awaited_once_with(NOTIFY_UUID)
        self.assertFalse(profile.is_notifying)

    async def test_notification_envelope_contains_connection_generation(self):
        profile = make_profile()
        client, _, _ = configured_client()
        envelopes = []
        profile.add_notification_listener(envelopes.append)
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")
        generation = profile.client_generation
        await profile.setNotify(True)
        backend_callback = client.start_notify.await_args.args[1]

        backend_callback("sender", bytearray(b"\x01\x02"))
        await asyncio.sleep(0)

        self.assertEqual(len(envelopes), 1)
        envelope = envelopes[0]
        self.assertEqual(envelope.sender, "sender")
        self.assertEqual(envelope.payload, b"\x01\x02")
        self.assertEqual(envelope.connection_generation, generation)
        self.assertGreater(envelope.host_wall_timestamp_ns, 0)
        self.assertGreater(envelope.host_monotonic_ns, 0)

    async def test_legacy_notification_failure_cannot_block_envelope_path(self):
        profile = make_profile()
        client, _, _ = configured_client()
        envelopes = []
        profile.add_notification_listener(envelopes.append)
        with self.assertWarns(DeprecationWarning):
            profile.set_notification_handler(MagicMock(side_effect=RuntimeError("legacy")))
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")
        await profile.setNotify(True)

        client.start_notify.await_args.args[1]("sender", b"payload")
        await asyncio.sleep(0)

        self.assertEqual(len(envelopes), 1)
        self.assertEqual(envelopes[0].payload, b"payload")

    async def test_stale_notification_is_dropped_after_reconnect(self):
        profile = make_profile()
        first_client, _, _ = configured_client()
        second_client, _, _ = configured_client()
        envelopes = []
        profile.add_notification_listener(envelopes.append)
        with patch.object(
            bleak_ble, "BleakClient", side_effect=[first_client, second_client]
        ):
            await profile.connect("first")
            first_generation = profile.client_generation
            await profile.setNotify(True)
            old_callback = first_client.start_notify.await_args.args[1]
            await profile.disconnect()
            await profile.connect("second")
            second_generation = profile.client_generation

        old_callback("old", b"old")
        await asyncio.sleep(0)

        self.assertEqual(second_generation, first_generation + 1)
        self.assertEqual(envelopes, [])
        self.assertEqual(profile.dropped_notification_count, 1)

    async def test_fast_aba_reconnect_rejects_all_old_notification_callbacks(self):
        profile = make_profile()
        clients = [configured_client()[0] for _ in range(3)]
        callbacks = []
        generations = []
        received = []
        profile.add_notification_listener(received.append)
        with patch.object(bleak_ble, "BleakClient", side_effect=clients):
            for index, client in enumerate(clients):
                await profile.connect(f"device-{index}")
                generations.append(profile.client_generation)
                await profile.setNotify(True)
                callbacks.append(client.start_notify.await_args.args[1])
                if index < len(clients) - 1:
                    await profile.disconnect()

        callbacks[0]("old-1", b"1")
        callbacks[1]("old-2", b"2")
        callbacks[2]("current", b"3")
        await asyncio.sleep(0)

        self.assertEqual(generations, [1, 2, 3])
        self.assertEqual(profile.dropped_notification_count, 2)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].connection_generation, 3)

    async def test_aba_reusing_same_client_identity_still_rejects_old_generation(self):
        profile = make_profile()
        client, _, _ = configured_client()
        received = []
        profile.add_notification_listener(received.append)
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("first")
            await profile.setNotify(True)
            old_callback = client.start_notify.await_args.args[1]
            await profile.disconnect()
            client.is_connected = True
            await profile.connect("second")
            await profile.setNotify(True)
            current_callback = client.start_notify.await_args.args[1]

        old_callback("old", b"1")
        current_callback("current", b"2")
        await asyncio.sleep(0)

        self.assertEqual(profile.client_generation, 2)
        self.assertEqual(profile.dropped_notification_count, 1)
        self.assertEqual([item.payload for item in received], [b"2"])

    async def test_command_backend_disconnect_without_callback_emits_once(self):
        profile = make_profile()
        client, _, _ = configured_client()
        events = []
        profile.add_state_listener(events.append)
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")
        await asyncio.sleep(0)
        events.clear()

        async def disconnect_during_write(*args):
            client.is_connected = False

        client.write_gatt_char.side_effect = disconnect_during_write
        with self.assertRaises(ConnectionError):
            await profile.setDataType(1, 0)
        await asyncio.sleep(0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].reason, "command_backend_disconnected")
        self.assertFalse(profile.is_connected)

    async def test_generation_bound_operations_reject_stale_transaction_before_io(self):
        profile = make_profile()
        client, _, _ = configured_client()
        profile.add_notification_listener(MagicMock())
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")

        stale_generation = profile.client_generation - 1
        with self.assertRaises(ConnectionError):
            await profile.setDataType(0, 0, expected_generation=stale_generation)
        with self.assertRaises(ConnectionError):
            await profile.setNotify(True, expected_generation=stale_generation)

        client.write_gatt_char.assert_not_awaited()
        client.start_notify.assert_not_awaited()

    async def test_notify_backend_disconnect_without_callback_emits_once(self):
        profile = make_profile()
        client, _, _ = configured_client()
        events = []
        profile.add_state_listener(events.append)
        profile.add_notification_listener(MagicMock())
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")
        await asyncio.sleep(0)
        events.clear()

        async def disconnect_during_notify(*args):
            client.is_connected = False

        client.start_notify.side_effect = disconnect_during_notify
        with self.assertRaises(ConnectionError):
            await profile.setNotify(True)
        await asyncio.sleep(0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].reason, "notify_backend_disconnected")
        self.assertFalse(profile.is_connected)

    async def test_disconnect_callback_resets_connection_and_notify_state(self):
        profile = make_profile()
        client, _, _ = configured_client()
        captured_callback = None

        def client_factory(device, disconnected_callback):
            nonlocal captured_callback
            captured_callback = disconnected_callback
            return client

        with patch.object(bleak_ble, "BleakClient", side_effect=client_factory):
            await profile.connect("device")
        profile.notify_enabled = True
        captured_callback(client)

        self.assertFalse(profile.is_connected)
        self.assertFalse(profile.is_notifying)
        self.assertIsNone(profile.notifyCharacteristic)
        self.assertIsNone(profile.cmdCharacteristic)

    async def test_disconnect_is_idempotent(self):
        profile = make_profile()
        self.assertFalse(await profile.disconnect())

        client, _, _ = configured_client()
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")
        self.assertTrue(await profile.disconnect())
        client.disconnect.assert_awaited_once()
        self.assertFalse(await profile.disconnect())
        client.disconnect.assert_awaited_once()
        self.assertFalse(profile.is_connected)

    async def test_requested_disconnect_sync_callback_emits_exactly_once(self):
        profile = make_profile()
        client, _, _ = configured_client()
        callback = None
        events = []

        def factory(device, disconnected_callback):
            nonlocal callback
            callback = disconnected_callback
            return client

        async def synchronous_callback_disconnect():
            client.is_connected = False
            callback(client)

        client.disconnect.side_effect = synchronous_callback_disconnect
        profile.add_state_listener(events.append)
        with patch.object(bleak_ble, "BleakClient", side_effect=factory):
            await profile.connect("device")
        await asyncio.sleep(0)
        events.clear()
        generation = profile.client_generation

        self.assertTrue(await profile.disconnect())
        await asyncio.sleep(0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "disconnected")
        self.assertEqual(events[0].reason, "requested")
        self.assertEqual(events[0].generation, generation)
        self.assertEqual((events[0].connected, events[0].notifying), (False, False))

    async def test_requested_disconnect_without_callback_and_late_callback_emit_once(self):
        profile = make_profile()
        client, _, _ = configured_client()
        callback = None
        events = []

        def factory(device, disconnected_callback):
            nonlocal callback
            callback = disconnected_callback
            return client

        profile.add_state_listener(events.append)
        with patch.object(bleak_ble, "BleakClient", side_effect=factory):
            await profile.connect("device")
        await asyncio.sleep(0)
        events.clear()
        generation = profile.client_generation

        self.assertTrue(await profile.disconnect())
        callback(client)
        await asyncio.sleep(0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].reason, "requested")
        self.assertEqual(events[0].generation, generation)
        self.assertFalse(events[0].connected)

    async def test_partial_connect_with_false_backend_state_is_cleaned_up(self):
        profile = make_profile()
        client, _, _ = configured_client()
        client.is_connected = False
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            with self.assertRaisesRegex(ConnectionError, "connected state"):
                await profile.connect("device")

        client.disconnect.assert_awaited_once()
        self.assertIsNone(profile.client)
        self.assertFalse(profile.is_connected)

    async def test_disconnect_failure_preserves_retryable_state(self):
        profile = make_profile()
        client, _, _ = configured_client()
        attempts = 0

        async def flaky_disconnect():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("temporary failure")
            client.is_connected = False

        client.disconnect.side_effect = flaky_disconnect
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")

        with self.assertRaisesRegex(OSError, "temporary failure"):
            await profile.disconnect()
        self.assertTrue(profile.is_connected)
        self.assertIs(profile.client, client)
        self.assertTrue(await profile.disconnect())
        self.assertFalse(profile.is_connected)
        self.assertEqual(attempts, 2)

    async def test_concurrent_connect_is_serialized_and_idempotent(self):
        profile = make_profile()
        client, _, _ = configured_client()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def delayed_connect():
            entered.set()
            await release.wait()

        client.connect.side_effect = delayed_connect
        with patch.object(bleak_ble, "BleakClient", return_value=client) as factory:
            first = asyncio.create_task(profile.connect("device"))
            await entered.wait()
            second = asyncio.create_task(profile.connect("device"))
            await asyncio.sleep(0)
            release.set()
            self.assertEqual(await asyncio.gather(first, second), [True, False])

        factory.assert_called_once()
        client.connect.assert_awaited_once()

    async def test_connect_then_concurrent_disconnect_is_serialized(self):
        profile = make_profile()
        client, _, _ = configured_client()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def delayed_connect():
            entered.set()
            await release.wait()

        client.connect.side_effect = delayed_connect
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            connect_task = asyncio.create_task(profile.connect("device"))
            await entered.wait()
            disconnect_task = asyncio.create_task(profile.disconnect())
            release.set()
            self.assertTrue(await connect_task)
            self.assertTrue(await disconnect_task)

        self.assertFalse(profile.is_connected)

    async def test_disconnect_callback_during_connect_prevents_stale_commit(self):
        profile = make_profile()
        client, _, _ = configured_client()
        callback = None

        def factory(device, disconnected_callback):
            nonlocal callback
            callback = disconnected_callback
            return client

        async def connect_then_disconnect():
            client.is_connected = False
            callback(client)

        client.connect.side_effect = connect_then_disconnect
        with patch.object(bleak_ble, "BleakClient", side_effect=factory):
            with self.assertRaisesRegex(ConnectionError, "stale"):
                await profile.connect("device")

        self.assertFalse(profile.is_connected)
        self.assertIsNone(profile.client)

    async def test_notify_then_concurrent_disconnect_is_serialized(self):
        profile = make_profile()
        client, _, _ = configured_client()
        profile.set_notification_handler(MagicMock())
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")
        entered = asyncio.Event()
        release = asyncio.Event()

        async def delayed_notify(*args):
            entered.set()
            await release.wait()

        client.start_notify.side_effect = delayed_notify
        notify_task = asyncio.create_task(profile.setNotify(True))
        await entered.wait()
        disconnect_task = asyncio.create_task(profile.disconnect())
        release.set()
        self.assertTrue(await notify_task)
        self.assertTrue(await disconnect_task)
        self.assertFalse(profile.is_notifying)
        self.assertFalse(profile.is_connected)

    async def test_disconnect_callback_during_notify_prevents_stale_commit(self):
        profile = make_profile()
        client, _, _ = configured_client()
        profile.set_notification_handler(MagicMock())
        callback = None

        def factory(device, disconnected_callback):
            nonlocal callback
            callback = disconnected_callback
            return client

        with patch.object(bleak_ble, "BleakClient", side_effect=factory):
            await profile.connect("device")

        async def notify_then_disconnect(*args):
            client.is_connected = False
            callback(client)

        client.start_notify.side_effect = notify_then_disconnect
        with self.assertRaisesRegex(ConnectionError, "disconnected"):
            await profile.setNotify(True)
        self.assertFalse(profile.is_notifying)
        self.assertFalse(profile.is_connected)
        self.assertIsNone(profile.client)

    async def test_notify_errors_preserve_last_confirmed_state(self):
        profile = make_profile()
        client, _, _ = configured_client()
        profile.set_notification_handler(MagicMock())
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")

        client.start_notify.side_effect = OSError("start failed")
        with self.assertRaises(OSError):
            await profile.setNotify(True)
        self.assertFalse(profile.is_notifying)

        client.start_notify.side_effect = None
        await profile.setNotify(True)
        client.stop_notify.side_effect = OSError("stop failed")
        with self.assertRaises(OSError):
            await profile.setNotify(False)
        self.assertTrue(profile.is_notifying)
        self.assertTrue(profile.is_connected)

    async def test_old_client_callback_does_not_change_new_connection_or_log(self):
        profile = make_profile()
        first_client, _, _ = configured_client()
        second_client, _, _ = configured_client()
        callbacks = []

        def factory(device, disconnected_callback):
            callbacks.append(disconnected_callback)
            return [first_client, second_client][len(callbacks) - 1]

        with patch.object(bleak_ble, "BleakClient", side_effect=factory):
            await profile.connect("first")
            await profile.disconnect()
            await profile.connect("second")

        generation = profile.client_generation
        with patch.object(bleak_ble.logger, "info") as info:
            callbacks[0](first_client)
        info.assert_not_called()
        self.assertIs(profile.client, second_client)
        self.assertTrue(profile.is_connected)
        self.assertEqual(profile.client_generation, generation)

    async def test_cross_loop_operation_is_rejected(self):
        profile = make_profile()
        client, _, _ = configured_client()
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")

        def use_other_loop():
            try:
                asyncio.run(profile.disconnect())
            except Exception as exc:
                return exc
            return None

        error = await asyncio.to_thread(use_other_loop)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("owner event loop", str(error))
        self.assertTrue(profile.is_connected)

    async def test_backend_state_drift_resets_public_state(self):
        profile = make_profile()
        client, _, _ = configured_client()
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")
        client.is_connected = False

        with self.assertRaisesRegex(RuntimeError, "not connected"):
            await profile.setDataType(1, 0)
        self.assertFalse(profile.is_connected)
        self.assertIsNone(profile.client)

    async def test_notify_flag_is_strictly_validated(self):
        profile = make_profile()
        for invalid in (2, -1, "1", 1.0, None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    await profile.setNotify(invalid)

    async def test_connection_log_uses_hashed_device_reference(self):
        profile = make_profile()
        client, _, _ = configured_client()
        raw_address = "AA:BB:CC:DD:EE:FF"
        with patch.object(bleak_ble, "BleakClient", return_value=client), patch.object(
            bleak_ble.logger, "info"
        ) as info:
            await profile.connect(raw_address)

        log_arguments = repr(info.call_args_list)
        self.assertNotIn(raw_address, log_arguments)
        self.assertIn("device_ref", log_arguments)

    async def test_set_data_type_preserves_commands_and_uses_explicit_stop_byte(self):
        profile = make_profile()
        client, _, command_characteristic = configured_client()
        with patch.object(bleak_ble, "BleakClient", return_value=client):
            await profile.connect("device")

        await profile.setDataType(1, 0)
        await profile.setDataType(0, 0)
        await profile.setDataType(0, 1)
        await profile.setDataType(1, 1)

        payloads = [call.args[1] for call in client.write_gatt_char.await_args_list]
        self.assertEqual(
            payloads,
            [
                b"\x01",
                b"\x00",
                b"\x04",
                b"?\xe8\x03\xff\x00\x80\x08",
                b"O\x80\x00\x00\x00",
            ],
        )
        self.assertTrue(
            all(
                call.args[0] is command_characteristic
                for call in client.write_gatt_char.await_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
