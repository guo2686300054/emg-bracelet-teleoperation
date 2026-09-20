"""Bounded, structured application logging with deterministic cleanup."""

from __future__ import annotations

import json
import logging
import logging.handlers
import queue
import re
import sys
import threading
import time
import traceback as traceback_module
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Union
from emg_protocol import DeviceKey

PathLike = Union[str, Path]
_REGISTRY_LOCK = threading.RLock()
_ACTIVE_RUNTIMES: dict[str, "LoggingRuntime"] = {}
LOGGER_NAMESPACE = "emg_app"
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+")
_MAC = re.compile(r"(?i)(?:[0-9a-f]{2}:){5}[0-9a-f]{2}")
_FALLBACK_SENTINEL = object()


def component_logger_name(component: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", component):
        raise ValueError("invalid component")
    return f"{LOGGER_NAMESPACE}.{component}"


def _redact(value: str) -> str:
    return _MAC.sub("[REDACTED_MAC]", _EMAIL.sub("[REDACTED_EMAIL]", value))


class EventCode(str,Enum):
    SYSTEM_SHUTDOWN="SYSTEM.SHUTDOWN"
    BLE_SEARCH="BLE.SEARCH"
    BLE_CONNECT="BLE.CONNECT"
    BLE_NOTIFY="BLE.NOTIFY"
    BLE_DISCONNECT="BLE.DISCONNECT"
    DATA_SAVE="DATA.SAVE"
    SHARED_WRITE="SHARED.WRITE"
    APP_EXCEPTION="APP.EXCEPTION"
    APP_BLOCK="APP.BLOCK"
    APP_OK="APP.OK"
    LOG_WRITE="LOG.WRITE"


_COUNTER_CONTEXT={"device_count","queue_depth","packet_index","error_count","dropped_count"}


@dataclass(frozen=True)
class EventField:
    value_type: type
    maximum_length: Optional[int] = None


@dataclass(frozen=True)
class EventSchema:
    required: Mapping[str, EventField]
    optional: Mapping[str, EventField]


_DEVICE_FIELD = EventField(DeviceKey)
_SESSION_FIELD = EventField(str, 128)
_COUNT_FIELD = EventField(int)
_EVENT_SCHEMAS = {
    EventCode.SYSTEM_SHUTDOWN: EventSchema({}, {}),
    EventCode.BLE_SEARCH: EventSchema({}, {
        "device_count": _COUNT_FIELD, "error_count": _COUNT_FIELD,
        "dropped_count": _COUNT_FIELD,
    }),
    EventCode.BLE_CONNECT: EventSchema({"device_id": _DEVICE_FIELD}, {
        "error_count": _COUNT_FIELD,
    }),
    EventCode.BLE_NOTIFY: EventSchema({
        "device_id": _DEVICE_FIELD, "packet_index": _COUNT_FIELD,
    }, {
        "error_count": _COUNT_FIELD, "dropped_count": _COUNT_FIELD,
        "queue_depth": _COUNT_FIELD,
    }),
    EventCode.BLE_DISCONNECT: EventSchema({"device_id": _DEVICE_FIELD}, {
        "error_count": _COUNT_FIELD,
    }),
    EventCode.DATA_SAVE: EventSchema({
        "device_id": _DEVICE_FIELD, "session_id": _SESSION_FIELD,
        "packet_index": _COUNT_FIELD,
    }, {
        "error_count": _COUNT_FIELD, "dropped_count": _COUNT_FIELD,
        "queue_depth": _COUNT_FIELD,
    }),
    EventCode.SHARED_WRITE: EventSchema({
        "device_id": _DEVICE_FIELD, "packet_index": _COUNT_FIELD,
    }, {
        "queue_depth": _COUNT_FIELD, "error_count": _COUNT_FIELD,
        "dropped_count": _COUNT_FIELD,
    }),
    EventCode.APP_EXCEPTION: EventSchema({}, {
        "device_id": _DEVICE_FIELD, "session_id": _SESSION_FIELD,
        "error_count": _COUNT_FIELD,
    }),
    EventCode.APP_BLOCK: EventSchema({}, {}),
    EventCode.APP_OK: EventSchema({}, {}),
    EventCode.LOG_WRITE: EventSchema({}, {
        "queue_depth": _COUNT_FIELD, "error_count": _COUNT_FIELD,
    }),
}


class SensitiveTokenRegistry:
    def __init__(self,capacity:int=128)->None:
        self._capacity=capacity;self._tokens:set[str]=set();self._lock=threading.Lock()
    def register(self,token:str)->None:
        if not isinstance(token,str) or not 2<=len(token)<=256 or any(ord(char)<32 for char in token): raise ValueError("sensitive token must be 2..256 printable characters")
        with self._lock:
            if token not in self._tokens and len(self._tokens)>=self._capacity: raise ValueError("sensitive token registry is full")
            self._tokens.add(token)
    def redact(self,value:str)->str:
        result=_redact(value)
        with self._lock:tokens=sorted(self._tokens,key=len,reverse=True)
        for token in tokens:result=result.replace(token,"[REDACTED_TOKEN]")
        return result


class LoggingShutdownError(RuntimeError):
    """Reports every resource that could not be released in one shutdown attempt."""

    def __init__(self, failures: list[tuple[str, BaseException]]) -> None:
        self.failures = tuple(failures)
        details = "; ".join(
            f"{stage}: {type(exc).__name__}: {exc}" for stage, exc in failures
        )
        super().__init__(f"logging shutdown incomplete: {details}")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "event": getattr(record, "event", None),
            "session_id": getattr(record, "session_id", None),
            "device_id": getattr(record, "device_id", None),
        }
        if hasattr(record, "redacted_exception"):
            payload["exception"] = record.redacted_exception
        elif record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": _redact(str(record.exc_info[1])),
                "traceback": _redact(
                    "".join(traceback_module.format_exception(*record.exc_info))
                ),
            }
        if hasattr(record, "health"):
            payload["health"] = record.health
        for key in _COUNTER_CONTEXT:
            if hasattr(record,key):payload[key]=getattr(record,key)
        return json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )


class EventLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        merged = dict(self.extra)
        supplied = kwargs.pop("extra", {})
        merged.update(supplied)
        event = merged.get("event")
        if not isinstance(event,EventCode): raise ValueError("event must be EventCode")
        context_keys=set(merged)-{"event"}
        schema = _EVENT_SCHEMAS[event]
        required = set(schema.required)
        allowed = required | set(schema.optional)
        if missing := required - context_keys:
            raise ValueError(f"missing required event context: {sorted(missing)}")
        if context_keys - allowed:
            raise ValueError("context key is not allowed for event")
        fields = dict(schema.optional)
        fields.update(schema.required)
        for key in context_keys:
            value = merged[key]
            field = fields[key]
            if field.value_type is int:
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"{key} must be a non-negative integer")
            elif not isinstance(value, field.value_type):
                raise ValueError(f"{key} has invalid type")
            if field.maximum_length is not None and len(value) > field.maximum_length:
                raise ValueError(f"{key} is too long")
        if "device_id" in merged:
            merged["device_id"]=str(merged["device_id"])
        if "session_id" in merged:
            value=merged["session_id"]
            if not isinstance(value,str) or not value or len(value)>128 or any(ord(char)<32 for char in value): raise ValueError("invalid session_id")
        merged["event"]=event.value
        json.dumps(merged, allow_nan=False)
        kwargs["extra"] = merged
        return msg, kwargs


class LocalQueueHandler(logging.handlers.QueueHandler):
    """Keep structured exception data because this queue never crosses processes."""

    def __init__(self, log_queue: queue.Queue, health: dict[str, Any],tokens:SensitiveTokenRegistry) -> None:
        super().__init__(log_queue)
        self.health = health
        self.tokens=tokens

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        prepared = copy(record)
        prepared.msg = self.tokens.redact(record.getMessage())
        prepared.args = None
        for key, value in list(prepared.__dict__.items()):
            if isinstance(value, str):
                setattr(prepared, key, self.tokens.redact(value))
        if prepared.exc_info:
            prepared.redacted_exception = {
                "type": prepared.exc_info[0].__name__,
                "message": self.tokens.redact(str(prepared.exc_info[1])),
                "traceback": self.tokens.redact(
                    "".join(traceback_module.format_exception(*prepared.exc_info))
                ),
            }
            prepared.exc_info = None
        return prepared

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
            with self.health["lock"]:
                self.health["max_depth"] = max(
                    self.health["max_depth"], self.queue.qsize()
                )
        except queue.Full:
            with self.health["lock"]:
                self.health["dropped"] += 1


class FallbackDispatcher:
    """Deliver sink failures through one bounded daemon worker."""

    def __init__(
        self,
        callback: Any,
        health: dict[str, Any],
        *,
        capacity: int,
        rate_limit_seconds: float = 0.05,
        tokens:Optional[SensitiveTokenRegistry]=None,
    ) -> None:
        self._callback = callback
        self._tokens=tokens or SensitiveTokenRegistry()
        self._health = health
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, capacity))
        self._rate_limit_seconds = rate_limit_seconds
        self._last_key: Optional[tuple[str, str, str]] = None
        self._last_submit = 0.0
        self._lock = threading.Lock()
        self._closed = False
        self._stopped = callback is None
        self._thread: Optional[threading.Thread] = None
        if callback is not None:
            self._thread = threading.Thread(
                target=self._run,
                name="emg-log-fallback",
                daemon=True,
            )
            self._thread.start()

    @property
    def stopped(self) -> bool:
        return self._stopped

    def submit(self, event: Mapping[str, str]) -> None:
        if self._callback is None:
            return
        key = (event["stage"], event["type"], event["detail"])
        now = time.monotonic()
        with self._lock:
            if self._closed:
                with self._health["lock"]:
                    self._health["fallback_dropped"] += 1
                return
            if key == self._last_key and now - self._last_submit < self._rate_limit_seconds:
                with self._health["lock"]:
                    self._health["fallback_rate_limited"] += 1
                return
            self._last_key = key
            self._last_submit = now
            try:
                self._queue.put_nowait(MappingProxyType(deepcopy(dict(event))))
                with self._health["lock"]:
                    self._health["fallback_max_depth"] = max(
                        self._health["fallback_max_depth"], self._queue.qsize()
                    )
            except queue.Full:
                with self._health["lock"]:
                    self._health["fallback_dropped"] += 1

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _FALLBACK_SENTINEL:
                    return
                try:
                    self._callback(item)
                    with self._health["lock"]:
                        self._health["fallback_delivered"] += 1
                except BaseException as exc:
                    with self._health["lock"]:
                        self._health["fallback_callback_errors"] += 1
                        self._health["last_fallback_error"] = {
                            "type": type(exc).__name__,
                            "detail": self._tokens.redact(str(exc))[:512],
                        }
            finally:
                self._queue.task_done()

    def shutdown(self, timeout: float = 0.5) -> None:
        if self._stopped:
            return
        with self._lock:
            self._closed = True
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._stopped = True
                return
            try:
                self._queue.put_nowait(_FALLBACK_SENTINEL)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    with self._health["lock"]:
                        self._health["fallback_dropped"] += 1
                    self._queue.put_nowait(_FALLBACK_SENTINEL)
                except queue.Empty as exc:
                    raise TimeoutError("fallback queue could not accept shutdown") from exc
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError("fallback worker did not stop before deadline")
        self._thread = None
        self._stopped = True


class HealthRotatingHandler(logging.handlers.RotatingFileHandler):
    def __init__(
        self, *args: Any, health: dict[str, Any], fallback: FallbackDispatcher,tokens:SensitiveTokenRegistry, **kwargs: Any
    ) -> None:
        self.health = health
        self.fallback = fallback
        self.tokens=tokens
        super().__init__(*args, **kwargs)

    def handleError(self, record: logging.LogRecord) -> None:
        exc_type, exc_value, _ = sys.exc_info()
        event = {
            "stage": self.tokens.redact(
                str(getattr(record, "sink_stage", "emit_or_rollover"))
            )[:64],
            "type": exc_type.__name__ if exc_type is not None else "UnknownSinkError",
            "detail": self.tokens.redact(str(exc_value))[:512]
            if exc_value is not None
            else "logging sink failed without an active exception",
        }
        with self.health["lock"]:
            self.health["sink_errors"] += 1
            self.health["last_sink_error"] = deepcopy(event)
        self.fallback.submit(event)


class SafeQueueListener(logging.handlers.QueueListener):
    def enqueue_sentinel(self) -> None:
        self.queue.put(self._sentinel, timeout=0.1)

    def _discard_pending(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return
            else:
                self.queue.task_done()

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        if not thread.is_alive():
            self._discard_pending()
            self._thread = None
            return
        self.enqueue_sentinel()
        thread.join(0.5)
        if thread.is_alive():
            raise TimeoutError("listener did not stop before deadline")
        self._thread = None


class LoggingRuntime:
    def __init__(
        self,
        logger: logging.Logger,
        queue_handler: LocalQueueHandler,
        listener: SafeQueueListener,
        file_handler: logging.Handler,
        fallback: FallbackDispatcher,
        log_file: Path,
        old_level: int,
        old_propagate: bool,
        tokens:SensitiveTokenRegistry,
    ) -> None:
        self.logger = logger
        self.log_file = log_file
        self._queue_handler = queue_handler
        self._listener = listener
        self._file_handler = file_handler
        self._fallback = fallback
        self._handler_removed = False
        self._listener_stopped = False
        self._queue_handler_closed = False
        self._file_handler_closed = False
        self._fallback_stopped = fallback.stopped
        self._lock = threading.Lock()
        self._old_level = old_level
        self._old_propagate = old_propagate
        self._tokens=tokens
        self.health = queue_handler.health
        self._shutdown_summary_sent = False

    @property
    def closed(self) -> bool:
        return all(
            (
                self._handler_removed,
                self._listener_stopped,
                self._queue_handler_closed,
                self._file_handler_closed,
                self._fallback_stopped,
            )
        )

    def health_snapshot(self) -> dict[str, Any]:
        with self.health["lock"]:
            return deepcopy({k: v for k, v in self.health.items() if k != "lock"})

    def get_logger(
        self,
        *,
        session_id: Optional[str] = None,
        device_id: Optional[DeviceKey] = None,
        event: Optional[EventCode] = None,
        **context:Any,
    ) -> EventLoggerAdapter:
        values={"event":event}
        if session_id is not None:values["session_id"]=session_id
        if device_id is not None:values["device_id"]=device_id
        values.update(context)
        return EventLoggerAdapter(
            self.logger,
            values,
        )

    def register_sensitive_token(self,token:str)->None:
        self._tokens.register(token)

    def shutdown(self) -> None:
        failures: list[tuple[str, BaseException]] = []
        with self._lock:
            if self.closed:
                return
            if not self._shutdown_summary_sent and not self._listener_stopped:
                record = self.logger.makeRecord(
                    self.logger.name,
                    logging.INFO,
                    "",
                    0,
                    "logging shutdown",
                    (),
                    None,
                    extra={"event": EventCode.SYSTEM_SHUTDOWN.value, "health": self.health_snapshot()},
                )
                try:
                    self._queue_handler.queue.put(
                        self._queue_handler.prepare(record), timeout=0.1
                    )
                    with self.health["lock"]:
                        self.health["max_depth"] = max(
                            self.health["max_depth"], self._queue_handler.queue.qsize()
                        )
                    self._shutdown_summary_sent = True
                except queue.Full:
                    with self.health["lock"]:
                        self.health["summary_lost"] = True
                    self._shutdown_summary_sent = True
            if not self._handler_removed:
                try:
                    self.logger.removeHandler(self._queue_handler)
                    self._handler_removed = True
                except BaseException as exc:
                    failures.append(("remove_queue_handler", exc))
            if not self._listener_stopped:
                try:
                    self._listener.stop()
                    self._listener_stopped = True
                except BaseException as exc:
                    failures.append(("stop_listener", exc))
            if not self._queue_handler_closed:
                try:
                    self._queue_handler.close()
                    self._queue_handler_closed = True
                except BaseException as exc:
                    failures.append(("close_queue_handler", exc))
            if self._listener_stopped and not self._file_handler_closed:
                try:
                    self._file_handler.close()
                    self._file_handler_closed = True
                except BaseException as exc:
                    failures.append(("close_file_handler", exc))
            if not self._fallback_stopped:
                try:
                    self._fallback.shutdown()
                    self._fallback_stopped = True
                except BaseException as exc:
                    failures.append(("stop_fallback", exc))
        if self.closed:
            self.logger.setLevel(self._old_level)
            self.logger.propagate = self._old_propagate
            with _REGISTRY_LOCK:
                if _ACTIVE_RUNTIMES.get(self.logger.name) is self:
                    del _ACTIVE_RUNTIMES[self.logger.name]
        if failures:
            raise LoggingShutdownError(failures)

    def __enter__(self) -> "LoggingRuntime":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            self.shutdown()
        except Exception as error:
            if exc_value is None:
                raise
            if hasattr(exc_value, "add_note"):
                exc_value.add_note(f"logging shutdown failed: {error}")


def _attach_cleanup_failures(
    primary: BaseException, failures: list[tuple[str, BaseException]]
) -> None:
    if not failures:
        return
    setattr(primary, "cleanup_failures", tuple(failures))
    detail = "; ".join(
        f"{stage}: {type(exc).__name__}: {_redact(str(exc))}"
        for stage, exc in failures
    )
    if hasattr(primary, "add_note"):
        primary.add_note(f"logging initialization cleanup failures: {detail}")


def configure_logging(
    log_path: PathLike,
    *,
    level: str = "INFO",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    queue_capacity: int = 2048,
    logger_name: str = "emg_app",
    file_name: str = "app.log",
    fallback_callback: Any = None,
) -> LoggingRuntime:
    normalized_level = level.strip().upper()
    level_value = logging._nameToLevel.get(normalized_level)
    if not isinstance(level_value, int):
        raise ValueError(f"unsupported logging level {level!r}")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")
    if backup_count < 0:
        raise ValueError("backup_count must not be negative")
    if (
        not isinstance(queue_capacity, int)
        or isinstance(queue_capacity, bool)
        or queue_capacity <= 0
    ):
        raise ValueError("queue_capacity must be positive integer")
    if not file_name or Path(file_name).name != file_name:
        raise ValueError("file_name must be a simple file name")
    if fallback_callback is not None and not callable(fallback_callback):
        raise ValueError("fallback_callback must be callable")

    directory = Path(log_path).expanduser().resolve()
    logger = logging.getLogger(logger_name)
    with _REGISTRY_LOCK:
        previous = _ACTIVE_RUNTIMES.get(logger_name)
        if previous is not None:
            previous.shutdown()
        old_level, old_propagate = logger.level, logger.propagate
        old_handlers = list(logger.handlers)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler: Optional[HealthRotatingHandler] = None
        queue_handler: Optional[LocalQueueHandler] = None
        listener: Optional[SafeQueueListener] = None
        fallback: Optional[FallbackDispatcher] = None
        health = {
            "dropped": 0,
            "max_depth": 0,
            "sink_errors": 0,
            "last_sink_error": None,
            "summary_lost": False,
            "fallback_dropped": 0,
            "fallback_rate_limited": 0,
            "fallback_max_depth": 0,
            "fallback_delivered": 0,
            "fallback_callback_errors": 0,
            "last_fallback_error": None,
            "lock": threading.Lock(),
        }
        tokens=SensitiveTokenRegistry()
        try:
            log_file = directory / file_name
            fallback = FallbackDispatcher(
                fallback_callback,
                health,
                capacity=min(queue_capacity, 64),
                tokens=tokens,
            )
            file_handler = HealthRotatingHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
                health=health,
                fallback=fallback,
                tokens=tokens,
            )
            file_handler.setFormatter(JsonFormatter())
            log_queue: queue.Queue[logging.LogRecord] = queue.Queue(
                maxsize=queue_capacity
            )
            queue_handler = LocalQueueHandler(log_queue, health,tokens)
            listener = SafeQueueListener(
                log_queue, file_handler, respect_handler_level=True
            )
            logger.setLevel(level_value)
            logger.propagate = False
            logger.addHandler(queue_handler)
            listener.start()
        except BaseException as primary:
            cleanup_failures: list[tuple[str, BaseException]] = []

            def cleanup(stage: str, action: Any) -> None:
                try:
                    action()
                except BaseException as exc:
                    cleanup_failures.append((stage, exc))

            if queue_handler is not None:
                cleanup("remove_queue_handler", lambda: logger.removeHandler(queue_handler))
            logger.handlers[:] = old_handlers
            if listener is not None:
                cleanup("stop_listener", listener.stop)
            if queue_handler is not None:
                cleanup("close_queue_handler", queue_handler.close)
            if file_handler is not None:
                cleanup("close_file_handler", file_handler.close)
            if fallback is not None:
                cleanup("stop_fallback", fallback.shutdown)
            logger.setLevel(old_level)
            logger.propagate = old_propagate
            _attach_cleanup_failures(primary, cleanup_failures)
            raise
        runtime = LoggingRuntime(
            logger,
            queue_handler,
            listener,
            file_handler,
            fallback,
            log_file,
            old_level,
            old_propagate,
            tokens,
        )
        _ACTIVE_RUNTIMES[logger_name] = runtime
        return runtime


def event_logger(
    logger: logging.Logger,
    context: Optional[Mapping[str,Any]] = None,
) -> EventLoggerAdapter:
    values = dict(context or {})
    return EventLoggerAdapter(logger,values)
