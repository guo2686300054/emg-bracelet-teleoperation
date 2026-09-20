import asyncio
import collections
import hashlib
import inspect
import logging
import queue
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.service import BleakGATTService


logger = logging.getLogger(__name__)

CLIENT_DISCONNECT_TIMEOUT_SECONDS = 2.0
CLIENT_DISCONNECT_DRAIN_MAX_SECONDS = 0.1
NativeDevice = Union[BLEDevice, str]

service_uuid_2 = "19b10000-e8f2-537e-4f6c-d104768a1214"
characteristic_uuid_2 = "19b10001-e8f2-537e-4f6c-d104768a1214"
# Deprecated misspelling kept for existing imports.
chararcteristic_uuid_2 = characteristic_uuid_2
cmd_uuid_2 = "19b10002-e8f2-537e-4f6c-d104768a1214"

service_uuid_3 = "19b10003-e8f2-537e-4f6c-d104768a1214"
characteristic_uuid_3 = "19b10004-e8f2-537e-4f6c-d104768a1214"
# Deprecated misspelling kept for existing imports.
chararcteristic_uuid_3 = characteristic_uuid_3
cmd_uuid_3 = "19b10005-e8f2-537e-4f6c-d104768a1214"


ConnectState = dict(Connected=0)


@dataclass(frozen=True)
class BleStateEvent:
    event_type: str
    connected: bool
    notifying: bool
    generation: int
    reason: str = ""


@dataclass(frozen=True)
class BleNotification:
    sender: object
    payload: bytes
    connection_generation: int
    host_wall_timestamp_ns: Optional[int] = None
    host_monotonic_ns: Optional[int] = None


@dataclass(frozen=True)
class DiscoveredDevice:
    candidate_id: str
    index: int
    name: Optional[str]
    address: Optional[str]
    rssi: Optional[int]
    matches_service: bool
    # Production scans retain BLEDevice. ``str`` is the isolated legacy
    # address boundary used by old callers/tests and supported by BleakClient.
    native_device: NativeDevice = field(compare=False, repr=False)


@dataclass(frozen=True)
class ScanSnapshot:
    devices: Tuple[DiscoveredDevice, ...]
    matches: Tuple[DiscoveredDevice, ...]

    def __post_init__(self):
        device_ids = tuple(device.candidate_id for device in self.devices)
        if len(device_ids) != len(set(device_ids)):
            raise ValueError("scan candidate identifiers must be unique")
        known_devices = {device.candidate_id: device for device in self.devices}
        if any(
            known_devices.get(match.candidate_id) is not match for match in self.matches
        ):
            raise ValueError("scan matches must belong to devices")
        if any(not match.matches_service for match in self.matches):
            raise ValueError("scan match must advertise the target service")


class BleakProfile:
    """Small stateful wrapper around Bleak for the EMG device."""

    def __init__(
        self,
        scan_service_uuid,
        notification_service_uuid,
        notification_characteristic_uuid,
        cmd_uuid,
    ):
        self.scan_service_uuid = scan_service_uuid
        self.notification_service_uuid = notification_service_uuid
        self.notification_characteristic_uuid = notification_characteristic_uuid
        self.cmd_uuid = cmd_uuid
        self.client = None
        self.cmdCharacteristic = None
        self.notifyCharacteristic = None
        self.timer = None
        self.cmdMap = {}
        self.mtu = None
        self.cmdForTimeout = -1
        self.incompleteCmdRespPacket = []
        self.lastIncompleteCmdRespPacketId = 0
        self.incompleteNotifPacket = []
        self.lastIncompleteNotifPacketId = 0
        self.onData = None
        self.lock = threading.RLock()
        self.send_queue = queue.Queue(maxsize=20)
        self.receive_time = 0
        self.start_time = 0
        self.notification_handler: Optional[Callable] = None
        self._notification_listeners = set()
        self.dropped_notification_count = 0
        # Keep the legacy integer state for qt5_bleak.py compatibility.
        self.connected = 0
        self.notify_enabled = False
        self._owner_loop = None
        self._operation_lock = None
        self._client_generation = 0
        self._pending_connection_generation = None
        self._active_client_generation = None
        self._attempt_epoch = 0
        self._active_attempt_epoch = None
        self._state_listeners = set()
        self._disconnect_requested_generation = None
        self._disconnect_requested_attempt_epoch = None
        self._pending_state_events = collections.deque()
        self._event_delivery_scheduled = False
        self._pending_disconnect_task = None
        self._pending_disconnect_client = None

    @property
    def is_connected(self):
        """Return the public connection state as a boolean."""
        return bool(self.connected)

    @property
    def is_notifying(self):
        """Return whether this profile has successfully started notifications."""
        return self.notify_enabled

    @property
    def owner_loop(self):
        """Event loop that owns all asynchronous operations for this profile."""
        return self._owner_loop

    @property
    def client_generation(self):
        """Monotonic identity used to reject stale client completions/callbacks."""
        return self._client_generation

    def _get_operation_lock(self):
        loop = asyncio.get_running_loop()
        if self._owner_loop is None:
            self._owner_loop = loop
            self._operation_lock = asyncio.Lock()
        elif self._owner_loop is not loop:
            raise RuntimeError("BleakProfile operations must use the owner event loop")
        return self._operation_lock

    @staticmethod
    def _device_reference(device):
        raw_identifier = getattr(device, "address", None) or str(device)
        digest = hashlib.sha256(raw_identifier.encode("utf-8", errors="replace")).hexdigest()
        return digest[:12]

    def add_state_listener(self, listener):
        if not callable(listener):
            raise TypeError("state listener must be callable")
        with self.lock:
            self._state_listeners.add(listener)

    def remove_state_listener(self, listener):
        with self.lock:
            self._state_listeners.discard(listener)

    def _drain_state_events(self):
        while True:
            with self.lock:
                if not self._pending_state_events:
                    self._event_delivery_scheduled = False
                    return
                event = self._pending_state_events.popleft()
                listeners = tuple(self._state_listeners)
            for listener in listeners:
                try:
                    listener(event)
                except Exception as exc:
                    logger.error("BLE state listener failed: %s", type(exc).__name__)

    def _enqueue_state_event_locked(self, event_type, reason="", generation=None):
        """Capture and enqueue state while the caller owns ``self.lock``."""
        event = BleStateEvent(
            event_type=event_type,
            connected=bool(self.connected),
            notifying=bool(self.notify_enabled),
            generation=self._client_generation if generation is None else generation,
            reason=reason,
        )
        self._pending_state_events.append(event)
        owner_loop = self._owner_loop
        if owner_loop is None or not owner_loop.is_running():
            self._pending_state_events.clear()
            self._event_delivery_scheduled = False
            logger.warning("BLE state event dropped: owner loop is not running")
            return
        if self._event_delivery_scheduled:
            return

        self._event_delivery_scheduled = True
        try:
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if current_loop is owner_loop:
                owner_loop.call_soon(self._drain_state_events)
            else:
                owner_loop.call_soon_threadsafe(self._drain_state_events)
        except RuntimeError:
            # The loop can close between is_running() and scheduling.
            self._event_delivery_scheduled = False
            self._pending_state_events.clear()
            logger.warning("BLE state event dropped: owner loop closed during scheduling")

    def _schedule_legacy_connection_callback(self, callback, generation, attempt_epoch):
        owner_loop = self._owner_loop
        if owner_loop is None or not owner_loop.is_running():
            logger.warning("Legacy connection callback dropped: owner loop is not running")
            return
        try:
            owner_loop.call_soon(
                self._deliver_legacy_connection_callback,
                callback,
                generation,
                attempt_epoch,
            )
        except RuntimeError:
            logger.warning("Legacy connection callback dropped: owner loop is closed")

    def _deliver_legacy_connection_callback(self, callback, generation, attempt_epoch):
        with self.lock:
            should_emit = (
                self._active_client_generation == generation
                and self._active_attempt_epoch == attempt_epoch
                and bool(self.connected)
                and self._client_is_connected()
            )
        if not should_emit:
            logger.info("Legacy connection callback skipped: connection is no longer active")
            return
        emit = getattr(callback, "emit", None)
        if not callable(emit):
            logger.error("Legacy connection callback ignored: emit is not callable")
            return
        try:
            emit("connected successfully")
        except Exception as exc:
            logger.error("Legacy connection callback failed: %s", type(exc).__name__)

    def _drop_notification(self, reason, generation):
        with self.lock:
            self.dropped_notification_count += 1
            dropped_count = self.dropped_notification_count
        logger.warning(
            "BLE notification dropped",
            extra={
                "reason": reason,
                "connection_generation": generation,
                "dropped_count": dropped_count,
            },
        )

    def _deliver_notification(
        self,
        client,
        generation,
        attempt_epoch,
        sender,
        payload,
        host_wall_timestamp_ns,
        host_monotonic_ns,
    ):
        with self.lock:
            if not self._is_current_client(
                client, generation, attempt_epoch
            ) or not self.connected:
                stale = True
                listeners = ()
                legacy_handler = None
            else:
                stale = False
                listeners = tuple(self._notification_listeners)
                legacy_handler = self.notification_handler
        if stale:
            self._drop_notification("stale_generation", generation)
            return

        envelope = BleNotification(
            sender,
            bytes(payload),
            generation,
            host_wall_timestamp_ns,
            host_monotonic_ns,
        )
        for listener in listeners:
            try:
                listener(envelope)
            except Exception as exc:
                logger.error("BLE notification listener failed: %s", type(exc).__name__)
        if legacy_handler is not None:
            try:
                legacy_handler(sender, payload)
            except Exception as exc:
                logger.error("Legacy notification handler failed: %s", type(exc).__name__)

    def _make_notification_callback(self, client, generation, attempt_epoch):
        def notification_callback(sender, payload):
            # Capture both clocks at the Bleak backend boundary, before event-loop
            # dispatch, queueing, parsing or UI work can add variable latency.
            host_monotonic_ns = time.monotonic_ns()
            host_wall_timestamp_ns = time.time_ns()
            with self.lock:
                current = self._is_current_client(
                    client, generation, attempt_epoch
                ) and bool(self.connected)
                owner_loop = self._owner_loop
            if not current:
                self._drop_notification("stale_generation", generation)
                return
            if owner_loop is None or not owner_loop.is_running():
                self._drop_notification("owner_loop_stopped", generation)
                return
            try:
                try:
                    current_loop = asyncio.get_running_loop()
                except RuntimeError:
                    current_loop = None
                if current_loop is owner_loop:
                    owner_loop.call_soon(
                        self._deliver_notification,
                        client,
                        generation,
                        attempt_epoch,
                        sender,
                        bytes(payload),
                        host_wall_timestamp_ns,
                        host_monotonic_ns,
                    )
                else:
                    owner_loop.call_soon_threadsafe(
                        self._deliver_notification,
                        client,
                        generation,
                        attempt_epoch,
                        sender,
                        bytes(payload),
                        host_wall_timestamp_ns,
                        host_monotonic_ns,
                    )
            except RuntimeError:
                self._drop_notification("owner_loop_closed", generation)

        return notification_callback

    async def scan(self, timeout: float):
        """Return one immutable snapshot retaining each native BLE device.

        Devices without advertisement metadata or UUIDs remain in ``devices`` and
        cannot accidentally terminate the scan.
        """
        operation_lock = self._get_operation_lock()
        async with operation_lock:
            logger.info("BLE scan started", extra={"timeout": timeout})
            try:
                scanner = BleakScanner()
                discover_parameters = inspect.signature(scanner.discover).parameters
                if "return_adv" in discover_parameters:
                    scan_result = await scanner.discover(timeout=timeout, return_adv=True)
                else:
                    scan_result = await scanner.discover(timeout=timeout)
            except Exception as exc:
                logger.error("BLE scan failed: %s", type(exc).__name__)
                raise

            all_devices = []
            expected_uuid = str(self.scan_service_uuid).lower()
            if isinstance(scan_result, dict):
                discovered = list(scan_result.values())
            else:
                discovered = [(device, None) for device in (scan_result or [])]
            for index, item in enumerate(discovered, start=1):
                device, advertisement = item
                if advertisement is not None:
                    uuids = getattr(advertisement, "service_uuids", None) or []
                else:
                    metadata = getattr(device, "metadata", None) or {}
                    uuids = metadata.get("uuids") or []
                advertised_uuids = {str(uuid).lower() for uuid in uuids if uuid}
                entry = DiscoveredDevice(
                    candidate_id=f"candidate-{index}",
                    index=index,
                    name=getattr(device, "name", None),
                    address=getattr(device, "address", None),
                    rssi=(
                        getattr(advertisement, "rssi", None)
                        if advertisement is not None
                        else getattr(device, "rssi", None)
                    ),
                    matches_service=expected_uuid in advertised_uuids,
                    native_device=device,
                )
                all_devices.append(entry)
            matching_devices = tuple(
                entry for entry in all_devices if entry.matches_service
            )
            snapshot = ScanSnapshot(tuple(all_devices), matching_devices)

            logger.info(
                "BLE scan completed",
                extra={
                    "device_count": len(all_devices),
                    "matched": bool(matching_devices),
                    "matched_count": len(matching_devices),
                },
            )
            return snapshot

    def set_notification_handler(self, fn):
        if fn is not None and not callable(fn):
            raise TypeError("notification handler must be callable or None")
        warnings.warn(
            "set_notification_handler() is deprecated; use add_notification_listener()",
            DeprecationWarning,
            stacklevel=2,
        )
        self.notification_handler = fn

    def add_notification_listener(self, listener):
        if not callable(listener):
            raise TypeError("notification listener must be callable")
        with self.lock:
            self._notification_listeners.add(listener)

    def remove_notification_listener(self, listener):
        with self.lock:
            self._notification_listeners.discard(listener)

    def getCharacteristic(self, service: BleakGATTService, uuid):
        if service is None:
            return None
        return service.get_characteristic(uuid)

    def _client_is_connected(self, client=None):
        client = client or self.client
        return client is not None and bool(getattr(client, "is_connected", False))

    def _is_current_client(self, client, generation, attempt_epoch):
        return (
            self.client is client
            and self._active_client_generation == generation
            and self._active_attempt_epoch == attempt_epoch
        )

    def _reset_connection_state(
        self,
        client=None,
        generation=None,
        attempt_epoch=None,
        invalidate=False,
    ):
        with self.lock:
            if client is not None and self.client is not client:
                return False
            if generation is not None and self._active_client_generation != generation:
                return False
            if attempt_epoch is not None and self._active_attempt_epoch != attempt_epoch:
                return False
            self.connected = 0
            self.notify_enabled = False
            self.notifyCharacteristic = None
            self.cmdCharacteristic = None
            self.client = None
            self._active_client_generation = None
            self._active_attempt_epoch = None
            if self._pending_connection_generation == generation:
                self._pending_connection_generation = None
            if self._disconnect_requested_generation == self._client_generation:
                self._disconnect_requested_generation = None
                self._disconnect_requested_attempt_epoch = None
            return True

    def _transition_disconnected(
        self, client, generation, attempt_epoch, reason
    ):
        with self.lock:
            if not self._is_current_client(client, generation, attempt_epoch):
                return False
            was_connected = bool(self.connected)
            accepted = self._reset_connection_state(
                client, generation, attempt_epoch, invalidate=True
            )
            if accepted and was_connected:
                self._enqueue_state_event_locked(
                    "disconnected",
                    reason,
                    generation=generation,
                )
            return accepted

    def _handle_disconnect(self, client, generation, attempt_epoch):
        with self.lock:
            requested = (
                self._disconnect_requested_generation == generation
                and self._disconnect_requested_attempt_epoch == attempt_epoch
            )
        accepted = self._transition_disconnected(
            client,
            generation,
            attempt_epoch,
            "requested" if requested else "unexpected",
        )
        if accepted:
            logger.info("BLE device disconnected")

    async def _disconnect_client_resisting_cancellation(self, client, deadline=None):
        if deadline is None:
            deadline = time.monotonic() + CLIENT_DISCONNECT_TIMEOUT_SECONDS
        cleanup_task = self._pending_disconnect_task
        if cleanup_task is not None:
            if cleanup_task.done():
                self._finish_pending_disconnect_task(cleanup_task)
                cleanup_task = None
            elif self._pending_disconnect_client is not client:
                return (
                    RuntimeError("another BLE client disconnect is still pending"),
                    None,
                    False,
                )
        if cleanup_task is None:
            cleanup_task = asyncio.create_task(client.disconnect())
            self._pending_disconnect_task = cleanup_task
            self._pending_disconnect_client = client
        cancellation = None
        available = max(0.0, deadline - time.monotonic())
        drain_budget = min(CLIENT_DISCONNECT_DRAIN_MAX_SECONDS, available / 2.0)
        cancel_at = deadline - drain_budget
        while not cleanup_task.done() and time.monotonic() < cancel_at:
            try:
                await asyncio.wait(
                    {cleanup_task}, timeout=min(cancel_at - time.monotonic(), 0.05)
                )
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
                continue

        timed_out = not cleanup_task.done()
        if timed_out:
            cleanup_task.cancel()
        while not cleanup_task.done() and time.monotonic() < deadline:
            try:
                await asyncio.wait(
                    {cleanup_task}, timeout=min(deadline - time.monotonic(), 0.01)
                )
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
                cleanup_task.cancel()

        if not cleanup_task.done():
            cleanup_task.add_done_callback(self._finish_pending_disconnect_task)
            timeout = TimeoutError(
                "BLE client disconnect timed out and did not terminate"
            )
            timeout.cleanup_pending = True
            timeout.cleanup_task = cleanup_task
            return (
                timeout,
                cancellation,
                False,
            )
        if self._pending_disconnect_task is cleanup_task:
            self._pending_disconnect_task = None
            self._pending_disconnect_client = None
        if timed_out:
            self._consume_task_result(cleanup_task)
            return TimeoutError("BLE client disconnect timed out"), cancellation, True
        try:
            cleanup_task.result()
        except asyncio.CancelledError as exc:
            return None, cancellation or exc, True
        except BaseException as cleanup_exc:
            return cleanup_exc, cancellation, True
        return None, cancellation, True

    @staticmethod
    def _consume_task_result(task):
        try:
            task.result()
        except BaseException:
            pass

    def _finish_pending_disconnect_task(self, task):
        if self._pending_disconnect_task is task:
            self._pending_disconnect_task = None
            self._pending_disconnect_client = None
        self._consume_task_result(task)

    def _clear_failed_connection_attempt(self, client, generation, attempt_epoch):
        if client is not None:
            self._reset_connection_state(
                client, generation, attempt_epoch, invalidate=True
            )
        with self.lock:
            if self._pending_connection_generation == generation:
                self._pending_connection_generation = None
            if self._active_client_generation == generation:
                self._active_client_generation = None
            if self._active_attempt_epoch == attempt_epoch:
                self._active_attempt_epoch = None
            if client is not None and self.client is client:
                self.client = None

    async def connect(self, device: NativeDevice, fun=None):
        """Connect and resolve required characteristics.

        Returns ``True`` for a new connection and ``False`` when already
        connected. Any failed connection leaves all public states reset.
        """
        operation_lock = self._get_operation_lock()
        connection_established = False
        async with operation_lock:
            if device is None:
                raise ValueError("device is required")
            pending_disconnect = self._pending_disconnect_task
            if pending_disconnect is not None:
                if pending_disconnect.done():
                    self._finish_pending_disconnect_task(pending_disconnect)
                else:
                    raise RuntimeError(
                        "previous BLE disconnect cleanup is still pending"
                    )
            if self.is_connected and self._client_is_connected():
                logger.info("BLE connect skipped: already connected")
                return False
            if self.client is not None:
                self._reset_connection_state(invalidate=True)

            self._attempt_epoch += 1
            attempt_epoch = self._attempt_epoch
            generation = self._client_generation + 1
            self._pending_connection_generation = generation
            device_ref = self._device_reference(device)
            logger.info("BLE connection started", extra={"device_ref": device_ref})
            client = None
            try:
                client = BleakClient(
                    device,
                    disconnected_callback=lambda disconnected_client: self._handle_disconnect(
                        disconnected_client, generation, attempt_epoch
                    ),
                )
                self.client = client
                self._active_client_generation = generation
                self._active_attempt_epoch = attempt_epoch
                await client.connect()
                if not self._is_current_client(client, generation, attempt_epoch):
                    raise ConnectionError("BLE client became stale during connection")
                if not self._client_is_connected(client):
                    raise ConnectionError("Bleak client did not enter the connected state")

                services = getattr(client, "services", None)
                notify_service = (
                    services.get_service(self.notification_service_uuid)
                    if services is not None
                    else None
                )
                if notify_service is None:
                    raise RuntimeError(
                        f"notification service not found: {self.notification_service_uuid}"
                    )

                notify_characteristic = self.getCharacteristic(
                    notify_service, self.notification_characteristic_uuid
                )
                command_characteristic = self.getCharacteristic(notify_service, self.cmd_uuid)
                if notify_characteristic is None:
                    raise RuntimeError(
                        "notification characteristic not found: "
                        f"{self.notification_characteristic_uuid}"
                    )
                if command_characteristic is None:
                    raise RuntimeError(f"command characteristic not found: {self.cmd_uuid}")
                with self.lock:
                    if not self._is_current_client(client, generation, attempt_epoch):
                        raise ConnectionError("BLE client became stale during service discovery")
                    if not self._client_is_connected(client):
                        raise ConnectionError("BLE client disconnected during service discovery")
                    self.notifyCharacteristic = notify_characteristic
                    self.cmdCharacteristic = command_characteristic
                    self._client_generation = generation
                    self._pending_connection_generation = None
                    self.connected = 1
                    self.notify_enabled = False
                    self._enqueue_state_event_locked("connected", generation=generation)
                logger.info("BLE connection established", extra={"device_ref": device_ref})
                connection_established = True
            except asyncio.CancelledError as exc:
                cleanup_result = (
                    await self._disconnect_client_resisting_cancellation(client)
                    if client is not None
                    else (None, None, True)
                )
                cleanup_error, _, _ = cleanup_result
                if cleanup_error is not None:
                    logger.error(
                        "BLE cleanup after cancelled connection failed: %s",
                        type(cleanup_error).__name__,
                    )
                    try:
                        exc.cleanup_failures = (("client_disconnect", cleanup_error),)
                    except Exception:
                        pass
                    if hasattr(exc, "add_note"):
                        exc.add_note(
                            "cancel cleanup client_disconnect failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                raise
            except Exception as exc:
                logger.error("BLE connection failed: %s", type(exc).__name__)
                cleanup_result = (
                    await self._disconnect_client_resisting_cancellation(client)
                    if client is not None
                    else (None, None, True)
                )
                cleanup_exc, _, _ = cleanup_result
                if cleanup_exc is not None:
                    logger.error(
                        "BLE cleanup after connection failure failed: %s",
                        type(cleanup_exc).__name__,
                    )
                    try:
                        exc.cleanup_failures = (("client_disconnect", cleanup_exc),)
                    except Exception:
                        pass
                    if hasattr(exc, "add_note"):
                        exc.add_note(
                            "cleanup client_disconnect failed: "
                            f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                        )
                raise
            finally:
                if not connection_established:
                    self._clear_failed_connection_attempt(
                        client, generation, attempt_epoch
                    )

        if fun is not None:
            warnings.warn(
                "connect(..., fun) is deprecated; use add_state_listener()",
                DeprecationWarning,
                stacklevel=2,
            )
            self._schedule_legacy_connection_callback(fun, generation, attempt_epoch)
        return connection_established

    def _require_connected_client(self):
        if not self.is_connected or not self._client_is_connected():
            client = self.client
            generation = self._client_generation
            attempt_epoch = self._active_attempt_epoch
            if client is not None:
                self._transition_disconnected(
                    client, generation, attempt_epoch, "backend_state_lost"
                )
            raise RuntimeError("BLE device is not connected")
        return self.client

    async def setDataType(self, flag, wristband=0, *, expected_generation=None):
        operation_lock = self._get_operation_lock()
        if type(flag) not in (bool, int) or flag not in (0, 1):
            raise ValueError("flag must be 0 or 1")
        if type(wristband) not in (bool, int) or wristband not in (0, 1):
            raise ValueError("flag and wristband must be 0 or 1")
        async with operation_lock:
            if expected_generation is not None and (
                self._client_generation != expected_generation
                or self._active_client_generation != expected_generation
            ):
                raise ConnectionError("BLE connection generation changed before command")
            client = self._require_connected_client()
            generation = self._client_generation
            attempt_epoch = self._active_attempt_epoch
            if self.cmdCharacteristic is None:
                raise RuntimeError("command characteristic is unavailable")

            logger.info(
                "BLE command requested", extra={"flag": flag, "wristband": wristband}
            )
            try:
                if flag == 1 and wristband == 0:
                    await client.write_gatt_char(self.cmdCharacteristic, b"\x01")
                elif flag == 0 and wristband == 1:
                    await client.write_gatt_char(self.cmdCharacteristic, b"\x04")
                elif flag == 1 and wristband == 1:
                    await client.write_gatt_char(
                        self.cmdCharacteristic, b"?\xe8\x03\xff\x00\x80\x08"
                    )
                    await client.write_gatt_char(
                        self.cmdCharacteristic, b"O\x80\x00\x00\x00"
                    )
                else:
                    await client.write_gatt_char(self.cmdCharacteristic, b"\x00")
                if not self._is_current_client(client, generation, attempt_epoch):
                    raise ConnectionError("BLE client disconnected during command")
                if not self._client_is_connected(client):
                    self._transition_disconnected(
                        client,
                        generation,
                        attempt_epoch,
                        "command_backend_disconnected",
                    )
                    raise ConnectionError("BLE backend disconnected during command")
            except Exception as exc:
                if self._is_current_client(
                    client, generation, attempt_epoch
                ) and not self._client_is_connected(client):
                    self._transition_disconnected(
                        client,
                        generation,
                        attempt_epoch,
                        "command_backend_disconnected",
                    )
                logger.error("BLE command failed: %s", type(exc).__name__)
                raise

    async def setNotify(self, flag, *, expected_generation=None):
        if type(flag) not in (bool, int) or flag not in (0, 1):
            raise ValueError("notify flag must be 0 or 1")
        enable = bool(flag)
        operation_lock = self._get_operation_lock()
        async with operation_lock:
            if expected_generation is not None and (
                self._client_generation != expected_generation
                or self._active_client_generation != expected_generation
            ):
                raise ConnectionError("BLE connection generation changed before notify operation")
            if enable == self.notify_enabled:
                logger.info(
                    "BLE notify request skipped: state unchanged",
                    extra={"enabled": enable},
                )
                return False

            client = self._require_connected_client()
            generation = self._client_generation
            attempt_epoch = self._active_attempt_epoch
            with self.lock:
                has_handler = bool(self._notification_listeners) or callable(
                    self.notification_handler
                )
            if enable and not has_handler:
                raise RuntimeError("notification handler is not configured")
            try:
                if enable:
                    await client.start_notify(
                        self.notification_characteristic_uuid,
                        self._make_notification_callback(
                            client, generation, attempt_epoch
                        ),
                    )
                else:
                    await client.stop_notify(self.notification_characteristic_uuid)
                with self.lock:
                    if not self._is_current_client(
                        client, generation, attempt_epoch
                    ):
                        raise ConnectionError("BLE client disconnected during notify operation")
                    if not self._client_is_connected(client):
                        self._transition_disconnected(
                            client,
                            generation,
                            attempt_epoch,
                            "notify_backend_disconnected",
                        )
                        raise ConnectionError("BLE client disconnected during notify operation")
                    self.notify_enabled = enable
                    self._enqueue_state_event_locked(
                        "notify_changed",
                        "started" if enable else "stopped",
                        generation=generation,
                    )
            except Exception as exc:
                if self._is_current_client(
                    client, generation, attempt_epoch
                ) and not self._client_is_connected(client):
                    self._transition_disconnected(
                        client,
                        generation,
                        attempt_epoch,
                        "notify_backend_disconnected",
                    )
                logger.error("BLE notify operation failed: %s", type(exc).__name__)
                raise

            logger.info("BLE notify state changed", extra={"enabled": enable})
        return True

    async def disconnect(self, *, deadline=None):
        """Disconnect safely; repeated calls are successful no-ops."""
        operation_lock = self._get_operation_lock()
        if deadline is None:
            deadline = time.monotonic() + CLIENT_DISCONNECT_TIMEOUT_SECONDS
        cancellation = None
        acquired = False
        lock_task = asyncio.create_task(operation_lock.acquire())
        available = max(0.0, deadline - time.monotonic())
        drain_budget = min(CLIENT_DISCONNECT_DRAIN_MAX_SECONDS, available / 2.0)
        cancel_at = deadline - drain_budget
        while not lock_task.done() and time.monotonic() < cancel_at:
            try:
                await asyncio.wait(
                    {lock_task}, timeout=min(cancel_at - time.monotonic(), 0.05)
                )
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
        if not lock_task.done():
            lock_task.cancel()
            while not lock_task.done() and time.monotonic() < deadline:
                try:
                    await asyncio.wait(
                        {lock_task}, timeout=min(deadline - time.monotonic(), 0.01)
                    )
                except asyncio.CancelledError as exc:
                    if cancellation is None:
                        cancellation = exc
                    lock_task.cancel()
        if lock_task.done():
            try:
                acquired = bool(lock_task.result())
            except asyncio.CancelledError:
                acquired = False
        if not acquired:
            error = TimeoutError("BLE disconnect timed out waiting for operation lock")
            if cancellation is not None:
                self._annotate_cancellation(cancellation, (("operation_lock", error),))
                raise cancellation
            raise error
        try:
            client = self.client
            generation = self._active_client_generation
            attempt_epoch = self._active_attempt_epoch
            if client is None:
                logger.info("BLE disconnect skipped: already disconnected")
                if cancellation is not None:
                    raise cancellation
                return False
            if not self.is_connected or not self._client_is_connected(client):
                cleanup_error, cleanup_cancel, terminated = (
                    await self._disconnect_client_resisting_cancellation(client, deadline)
                )
                cancellation = cancellation or cleanup_cancel
                if terminated:
                    self._clear_failed_connection_attempt(client, generation, attempt_epoch)
                if cleanup_error is not None:
                    logger.error(
                        "BLE pending client disconnect failed: %s",
                        type(cleanup_error).__name__,
                    )
                    if cancellation is not None:
                        self._annotate_cancellation(
                            cancellation, (("client_disconnect", cleanup_error),)
                        )
                        raise cancellation
                    raise cleanup_error
                if cancellation is not None:
                    raise cancellation
                logger.info("BLE pending client disconnected")
                return True

            logger.info("BLE disconnect started")
            self._disconnect_requested_generation = generation
            self._disconnect_requested_attempt_epoch = attempt_epoch
            cleanup_error, cleanup_cancel, terminated = (
                await self._disconnect_client_resisting_cancellation(client, deadline)
            )
            cancellation = cancellation or cleanup_cancel
            if cleanup_error is not None:
                exc = cleanup_error
                with self.lock:
                    if terminated and self._is_current_client(
                        client, generation, attempt_epoch
                    ):
                        if self._client_is_connected(client):
                            self.connected = 1
                            self._disconnect_requested_generation = None
                            self._disconnect_requested_attempt_epoch = None
                        else:
                            self._transition_disconnected(
                                client, generation, attempt_epoch, "requested"
                            )
                logger.error("BLE disconnect failed: %s", type(exc).__name__)
                if cancellation is not None:
                    self._annotate_cancellation(
                        cancellation, (("client_disconnect", exc),)
                    )
                    raise cancellation
                raise exc

            with self.lock:
                if self._is_current_client(client, generation, attempt_epoch):
                    if self._client_is_connected(client):
                        self.connected = 1
                        self._disconnect_requested_generation = None
                        self._disconnect_requested_attempt_epoch = None
                        raise ConnectionError("BLE backend remained connected after disconnect")
                    self._transition_disconnected(
                        client, generation, attempt_epoch, "requested"
                    )
            logger.info("BLE disconnect completed")
            if cancellation is not None:
                raise cancellation
            return True
        finally:
            operation_lock.release()

    @staticmethod
    def _annotate_cancellation(cancellation, failures):
        try:
            cancellation.cleanup_failures = tuple(failures)
        except Exception:
            pass
        if hasattr(cancellation, "add_note"):
            for stage, error in failures:
                cancellation.add_note(
                    f"cleanup {stage} failed: {type(error).__name__}: {error}"
                )

    def setReceiveTime(self, time):
        self.receive_time = time

    def setStartTime(self, time):
        self.start_time = time
