import json
import logging
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import app_logging
from app_logging import EventCode,LoggingShutdownError, component_logger_name, configure_logging
from emg_protocol import DeviceKey

DEV=DeviceKey("dev-0123456789abcdef0123456789abcdef")


class LoggingTests(unittest.TestCase):
    def tearDown(self):
        for runtime in list(app_logging._ACTIVE_RUNTIMES.values()):
            try:
                runtime.shutdown()
            except Exception:
                pass

    def test_bounded_drop_redaction_health_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            name = component_logger_name("bleak")
            base = logging.getLogger(name)
            old = (base.level, base.propagate)
            runtime = configure_logging(directory, logger_name=name, queue_capacity=1)
            log = runtime.get_logger(
                event=EventCode.BLE_CONNECT, device_id=DEV
            )
            log.info("mail a@b.com")
            with mock.patch.object(
                runtime._queue_handler.queue, "put_nowait", side_effect=queue.Full
            ):
                log.info("dropped")
            runtime.shutdown()
            lines = []
            for path in Path(directory).glob("app.log*"):
                lines.extend(path.read_text(encoding="utf8").splitlines())
            parsed = [json.loads(line) for line in lines]
            self.assertTrue(lines)
            self.assertNotIn("a@b.com", "".join(lines))
            self.assertIn(str(DEV), "".join(lines))
            self.assertGreaterEqual(runtime.health["dropped"], 1)
            self.assertTrue(any(row["event"] == "SYSTEM.SHUTDOWN" for row in parsed))
            self.assertEqual((base.level, base.propagate), old)

    def test_listener_start_failure_fully_rolls_back_logger(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = logging.getLogger("test.start.rollback")
            existing = logging.NullHandler()
            logger.handlers[:] = [existing]
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
            primary = RuntimeError("listener start failed")
            with mock.patch.object(
                app_logging.SafeQueueListener, "start", side_effect=primary
            ):
                with self.assertRaises(RuntimeError) as caught:
                    configure_logging(directory, logger_name=logger.name)
            self.assertIs(caught.exception, primary)
            self.assertEqual(logger.handlers, [existing])
            self.assertEqual(logger.level, logging.NOTSET)
            self.assertTrue(logger.propagate)
            self.assertNotIn(logger.name, app_logging._ACTIVE_RUNTIMES)

    def test_listener_start_failure_aggregates_all_cleanup_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = logging.getLogger("test.start.cleanup")
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
            primary = RuntimeError("primary")
            original_queue_close = app_logging.LocalQueueHandler.close
            original_file_close = app_logging.HealthRotatingHandler.close

            def queue_close(handler):
                original_queue_close(handler)
                raise OSError("queue close")

            def file_close(handler):
                original_file_close(handler)
                raise PermissionError("file close")

            with mock.patch.object(
                app_logging.SafeQueueListener, "start", side_effect=primary
            ), mock.patch.object(
                app_logging.SafeQueueListener, "stop", side_effect=TimeoutError("stop")
            ), mock.patch.object(
                app_logging.LocalQueueHandler, "close", queue_close
            ), mock.patch.object(
                app_logging.HealthRotatingHandler, "close", file_close
            ):
                with self.assertRaises(RuntimeError) as caught:
                    configure_logging(directory, logger_name=logger.name)
            self.assertIs(caught.exception, primary)
            stages = [stage for stage, _ in caught.exception.cleanup_failures]
            self.assertEqual(
                stages, ["stop_listener", "close_queue_handler", "close_file_handler"]
            )
            self.assertEqual(logger.handlers, [])
            self.assertEqual(logger.level, logging.NOTSET)
            self.assertTrue(logger.propagate)

    def test_fallback_uses_one_daemon_and_preserves_redacted_error(self):
        received = []
        ready = threading.Event()

        def callback(event):
            received.append(event)
            ready.set()

        with tempfile.TemporaryDirectory() as directory:
            runtime = configure_logging(
                directory,
                logger_name="test.fallback.detail",
                fallback_callback=callback,
            )
            before = [t for t in threading.enumerate() if t.name == "emg-log-fallback"]
            record = logging.LogRecord("x", 20, "", 1, "x", (), None)
            record.sink_stage = "rollover"
            try:
                raise OSError("disk failed for secret@example.com")
            except OSError:
                runtime._file_handler.handleError(record)
            self.assertTrue(ready.wait(1))
            after = [t for t in threading.enumerate() if t.name == "emg-log-fallback"]
            self.assertEqual(len(after), len(before))
            self.assertEqual(received[0]["stage"], "rollover")
            self.assertEqual(received[0]["type"], "OSError")
            self.assertIn("[REDACTED_EMAIL]", received[0]["detail"])
            self.assertNotIn("secret@example.com", str(runtime.health_snapshot()))
            runtime.shutdown()

    def test_health_snapshot_and_fallback_event_are_isolated(self):
        received = []
        immutable = []
        ready = threading.Event()

        def callback(event):
            received.append(event)
            try:
                event["detail"] = "corrupt"
            except TypeError:
                immutable.append(True)
            ready.set()

        with tempfile.TemporaryDirectory() as directory:
            runtime = configure_logging(
                directory,
                logger_name="test.health.isolation",
                fallback_callback=callback,
            )
            record = logging.LogRecord("x", 20, "", 1, "x", (), None)
            try:
                raise OSError("private@example.com")
            except OSError:
                runtime._file_handler.handleError(record)
            self.assertTrue(ready.wait(1))
            self.assertEqual(immutable, [True])
            snapshot = runtime.health_snapshot()
            snapshot["last_sink_error"]["detail"] = "changed"
            snapshot["sink_errors"] = 999
            current = runtime.health_snapshot()
            self.assertEqual(current["sink_errors"], 1)
            self.assertIn("[REDACTED_EMAIL]", current["last_sink_error"]["detail"])
            self.assertIsNot(received[0], current["last_sink_error"])
            runtime.shutdown()

    def test_permanent_fallback_failure_is_bounded_and_rate_limited(self):
        def failing_callback(_event):
            raise RuntimeError("callback secret@example.com")

        with tempfile.TemporaryDirectory() as directory:
            runtime = configure_logging(
                directory,
                logger_name="test.fallback.permanent",
                fallback_callback=failing_callback,
                queue_capacity=2,
            )
            record = logging.LogRecord("x", 20, "", 1, "x", (), None)
            for _ in range(100):
                try:
                    raise OSError("same sink error")
                except OSError:
                    runtime._file_handler.handleError(record)
            deadline = time.monotonic() + 1
            while runtime.health_snapshot()["fallback_callback_errors"] == 0:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            threads = [
                t for t in threading.enumerate() if t.name == "emg-log-fallback"
            ]
            self.assertEqual(len(threads), 1)
            health = runtime.health_snapshot()
            self.assertEqual(health["sink_errors"], 100)
            self.assertGreater(health["fallback_rate_limited"], 0)
            self.assertIn("[REDACTED_EMAIL]", health["last_fallback_error"]["detail"])
            runtime.shutdown()

    def test_permanent_file_sink_failure_reports_real_exception(self):
        received = []
        with tempfile.TemporaryDirectory() as directory:
            runtime = configure_logging(
                directory,
                logger_name="test.sink.permanent",
                fallback_callback=received.append,
                max_bytes=1,
            )
            with mock.patch.object(
                runtime._file_handler,
                "doRollover",
                side_effect=PermissionError("disk secret@example.com"),
            ):
                runtime.get_logger(event=EventCode.LOG_WRITE).info("force rollover")
                deadline = time.monotonic() + 1
                while not received:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
            self.assertEqual(received[0]["type"], "PermissionError")
            self.assertIn("[REDACTED_EMAIL]", received[0]["detail"])
            self.assertEqual(runtime.health_snapshot()["sink_errors"], 1)
            runtime.shutdown()

    def test_fallback_queue_capacity_drops_without_extra_threads(self):
        release = threading.Event()
        entered = threading.Event()

        def slow_callback(_event):
            entered.set()
            release.wait(1)

        with tempfile.TemporaryDirectory() as directory:
            runtime = configure_logging(
                directory,
                logger_name="test.fallback.capacity",
                fallback_callback=slow_callback,
                queue_capacity=1,
            )
            record = logging.LogRecord("x", 20, "", 1, "x", (), None)
            try:
                raise OSError("first")
            except OSError:
                runtime._file_handler.handleError(record)
            self.assertTrue(entered.wait(1))
            for index in range(20):
                try:
                    raise OSError(f"unique-{index}")
                except OSError:
                    runtime._file_handler.handleError(record)
            self.assertGreater(runtime.health_snapshot()["fallback_dropped"], 0)
            self.assertEqual(
                len([t for t in threading.enumerate() if t.name == "emg-log-fallback"]),
                1,
            )
            release.set()
            runtime.shutdown()

    def test_permanently_blocked_fallback_shutdown_is_bounded_and_retryable(self):
        entered = threading.Event()
        release = threading.Event()

        def blocked_callback(_event):
            entered.set()
            release.wait(2)

        with tempfile.TemporaryDirectory() as directory:
            runtime = configure_logging(
                directory,
                logger_name="test.fallback.blocked",
                fallback_callback=blocked_callback,
            )
            record = logging.LogRecord("x", 20, "", 1, "x", (), None)
            try:
                raise OSError("blocked")
            except OSError:
                runtime._file_handler.handleError(record)
            self.assertTrue(entered.wait(1))
            start = time.monotonic()
            with self.assertRaises(LoggingShutdownError) as caught:
                runtime.shutdown()
            self.assertLess(time.monotonic() - start, 0.8)
            self.assertEqual(caught.exception.failures[0][0], "stop_fallback")
            worker = runtime._fallback._thread
            self.assertIsNotNone(worker)
            self.assertTrue(worker.daemon)
            self.assertTrue(worker.is_alive())
            release.set()
            worker.join(1)
            self.assertFalse(worker.is_alive())
            runtime.shutdown()
            self.assertTrue(runtime.closed)

    def test_shutdown_retry_uses_public_failure_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = configure_logging(
                directory, logger_name="test.shutdown.retry"
            )
            original_stop = runtime._listener.stop
            calls = iter([OSError("first stop failure"), None])

            def flaky_stop():
                outcome = next(calls)
                if outcome is not None:
                    raise outcome
                return original_stop()

            with mock.patch.object(runtime._listener, "stop", side_effect=flaky_stop):
                with self.assertRaises(LoggingShutdownError) as caught:
                    runtime.shutdown()
                self.assertEqual(caught.exception.failures[0][0], "stop_listener")
                runtime.shutdown()
            self.assertTrue(runtime.closed)

    def test_dead_listener_with_full_queue_still_closes_bounded(self):
        blocker = threading.Event()
        entered = threading.Event()

        class BlockingHandler(logging.Handler):
            def emit(self, record):
                entered.set()
                blocker.wait(1)

        with tempfile.TemporaryDirectory() as directory:
            runtime = configure_logging(
                directory, logger_name="test.listener.dead", queue_capacity=1
            )
            blocking = BlockingHandler()
            runtime._listener.handlers = (blocking,)
            runtime.get_logger(event=EventCode.APP_BLOCK).info("one")
            self.assertTrue(entered.wait(1))
            runtime.get_logger(event=EventCode.APP_BLOCK).info("two")
            start = time.monotonic()
            with self.assertRaises(LoggingShutdownError):
                runtime.shutdown()
            self.assertLess(time.monotonic() - start, 0.8)
            blocker.set()
            runtime.shutdown()
            self.assertTrue(runtime.closed)
            blocking.close()

    def test_reconfigure_old_shutdown_failure_has_no_new_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = configure_logging(
                Path(directory) / "old", logger_name="test.replace"
            )
            target = Path(directory) / "new" / "nested"
            with mock.patch.object(
                runtime._listener, "stop", side_effect=OSError("stop")
            ):
                with self.assertRaises(LoggingShutdownError):
                    configure_logging(target, logger_name="test.replace")
            self.assertFalse(target.exists())
            runtime.shutdown()

    def test_controlled_event_and_json_context(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = configure_logging(directory, logger_name="test.control")
            with self.assertRaises(ValueError):
                runtime.get_logger(event="free text").info("x")
            with self.assertRaises((ValueError, TypeError)):
                runtime.get_logger(event=EventCode.APP_OK).info(
                    "x", extra={"bad": float("nan")}
                )
            runtime.shutdown()

    def test_event_registry_enforces_per_event_context_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=configure_logging(directory,logger_name="test.event.schema")
            runtime.get_logger(event=EventCode.BLE_SEARCH,device_count=2,error_count=0).info("search")
            runtime.get_logger(event=EventCode.BLE_NOTIFY,device_id=DEV,packet_index=3,error_count=0).info("notify")
            with self.assertRaises(ValueError):runtime.get_logger(event=EventCode.APP_OK,queue_depth=1).info("bad")
            with self.assertRaises(ValueError):runtime.get_logger(event=EventCode.BLE_CONNECT,device_id=str(DEV)).info("raw id")
            with self.assertRaises(ValueError):runtime.get_logger(event=EventCode.BLE_SEARCH,device_count=True).info("bool")
            with self.assertRaises(ValueError):runtime.get_logger(event="BLE.SEARCH",device_count=1).info("string event")
            runtime.shutdown()
            rows=[]
            for path in Path(directory).glob("app.log*"):rows.extend(json.loads(line) for line in path.read_text(encoding="utf8").splitlines())
            search=next(row for row in rows if row["event"]==EventCode.BLE_SEARCH.value);notify=next(row for row in rows if row["event"]==EventCode.BLE_NOTIFY.value)
            self.assertEqual(search["device_count"],2);self.assertEqual(notify["packet_index"],3)

    def test_dynamic_sensitive_token_is_redacted_before_enqueue(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=configure_logging(directory,logger_name="test.token")
            runtime.register_sensitive_token("张三")
            try:raise RuntimeError("张三 private failure")
            except RuntimeError:runtime.get_logger(event=EventCode.APP_EXCEPTION,error_count=1).exception("patient 张三 failed")
            runtime.shutdown();content="".join(path.read_text(encoding="utf8") for path in Path(directory).glob("app.log*"))
            self.assertNotIn("张三",content);self.assertIn("[REDACTED_TOKEN]",content)

    def test_device_key_is_opaque(self):
        with self.assertRaises(ValueError):DeviceKey("AA:BB:CC:DD:EE:FF")
        with self.assertRaises(ValueError):DeviceKey("patient-alice")

    def test_context_manager_preserves_business_error(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = configure_logging(directory, logger_name="test.business")
            with mock.patch.object(
                runtime._listener, "stop", side_effect=OSError("stop")
            ):
                with self.assertRaises(KeyError):
                    with runtime:
                        raise KeyError("business")
            runtime.shutdown()

    def test_invalid_constructor_args(self):
        with tempfile.TemporaryDirectory() as directory:
            for kwargs in (
                {"queue_capacity": 0},
                {"max_bytes": 0},
                {"backup_count": -1},
                {"fallback_callback": object()},
            ):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    configure_logging(
                        directory, logger_name="test.invalid", **kwargs
                    )

    def test_key_event_schemas_require_identity_and_index(self):
        cases = (
            (EventCode.BLE_CONNECT, {}),
            (EventCode.BLE_NOTIFY, {"device_id": DEV}),
            (EventCode.DATA_SAVE, {"device_id": DEV, "session_id": "session-a"}),
            (EventCode.SHARED_WRITE, {"device_id": DEV}),
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = configure_logging(directory, logger_name="test.required.context")
            for event, context in cases:
                with self.subTest(event=event), self.assertRaises(ValueError):
                    runtime.get_logger(event=event, **context).info("missing")
            runtime.shutdown()

    def test_event_schema_enforces_field_types_and_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = configure_logging(directory, logger_name="test.typed.context")
            invalid = (
                {"event": EventCode.BLE_NOTIFY, "device_id": DEV, "packet_index": True},
                {"event": EventCode.DATA_SAVE, "device_id": DEV, "session_id": "x" * 129, "packet_index": 1},
                {"event": EventCode.DATA_SAVE, "device_id": DEV, "session_id": "bad\nvalue", "packet_index": 1},
            )
            for context in invalid:
                with self.subTest(context=context), self.assertRaises(ValueError):
                    runtime.get_logger(**context).info("invalid")
            runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
