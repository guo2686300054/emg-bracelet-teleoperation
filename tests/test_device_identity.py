import concurrent.futures
import hashlib
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import device_identity as identity_module
from device_identity import (
    DeviceIdentityStore,
    IdentityIntegrityError,
    IdentityLockCleanupError,
    IdentityPermissionError,
    _INTEGRITY_CONTEXT,
)
from emg_protocol import DeviceKey


ADDRESS = "AA:BB:CC:DD:EE:FF"


class DeviceIdentityTests(unittest.TestCase):
    def test_stable_opaque_key_and_compatibility_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DeviceIdentityStore(root / "identity")
            first = store.get_device_key(ADDRESS)
            second = DeviceIdentityStore(root / "identity").device_key(ADDRESS.lower())
            self.assertIsInstance(first, DeviceKey)
            self.assertEqual(first, second)
            self.assertRegex(str(first), r"^dev-[0-9a-f]{32}$")
            self.assertNotIn("aa:bb", str(first))
            self.assertNotIn(ADDRESS.encode(), store.key_path.read_bytes())

    def test_different_stores_and_identifiers_produce_different_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one = DeviceIdentityStore(root / "one")
            two = DeviceIdentityStore(root / "two")
            self.assertNotEqual(one.device_key(ADDRESS), two.device_key(ADDRESS))
            self.assertNotEqual(one.device_key(ADDRESS), one.device_key("11:22:33:44:55:66"))

    def test_empty_identifier_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DeviceIdentityStore(Path(directory) / "identity")
            with self.assertRaises(ValueError):
                store.device_key("  ")

    def test_key_cannot_be_inside_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                DeviceIdentityStore(root / "data" / "identity.key", root / "data")

    def test_truncated_main_recovers_from_verified_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = DeviceIdentityStore(root / "identity")
            expected = original.device_key(ADDRESS)
            original.key_path.write_bytes(original.key_path.read_bytes()[:10])
            if os.name != "nt":
                original.key_path.chmod(0o600)
            recovered = DeviceIdentityStore(root / "identity")
            self.assertEqual(expected, recovered.device_key(ADDRESS))
            self.assertEqual(
                expected,
                DeviceIdentityStore(root / "identity").device_key(ADDRESS),
            )

    def test_equal_length_corruption_recovers_from_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DeviceIdentityStore(Path(directory) / "identity")
            expected = store.device_key(ADDRESS)
            damaged = bytearray(store.key_path.read_bytes())
            damaged[len(damaged) // 2] ^= 0x80
            store.key_path.write_bytes(damaged)
            if os.name != "nt":
                store.key_path.chmod(0o600)
            self.assertEqual(expected, DeviceIdentityStore(Path(directory) / "identity").device_key(ADDRESS))

    def test_main_and_backup_corruption_fails_closed_without_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DeviceIdentityStore(root / "identity")
            for path in (store.key_path, store.backup_path):
                data = bytearray(path.read_bytes())
                data[-1] ^= 1
                path.write_bytes(data)
                if os.name != "nt":
                    path.chmod(0o600)
            main_before = store.key_path.read_bytes()
            backup_before = store.backup_path.read_bytes()
            with self.assertRaises(IdentityIntegrityError):
                DeviceIdentityStore(root / "identity")
            self.assertEqual(main_before, store.key_path.read_bytes())
            self.assertEqual(backup_before, store.backup_path.read_bytes())

    def test_bad_backup_is_repaired_from_valid_main(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DeviceIdentityStore(root / "identity")
            expected = store.device_key(ADDRESS)
            store.backup_path.write_bytes(b"broken")
            if os.name != "nt":
                store.backup_path.chmod(0o600)
            repaired = DeviceIdentityStore(root / "identity")
            self.assertEqual(expected, repaired.device_key(ADDRESS))
            repaired.key_path.unlink()
            self.assertEqual(
                expected,
                DeviceIdentityStore(root / "identity").device_key(ADDRESS),
            )

    def test_thread_concurrent_creation_uses_one_key(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = Path(directory) / "identity"
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                values = list(executor.map(lambda _: str(DeviceIdentityStore(identity).device_key(ADDRESS)), range(36)))
            self.assertEqual(1, len(set(values)))

    def test_process_concurrent_creation_uses_one_key(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = Path(directory) / "identity"
            script = (
                "from device_identity import DeviceIdentityStore; import sys; "
                "print(DeviceIdentityStore(sys.argv[1]).device_key(sys.argv[2]))"
            )
            commands = [[sys.executable, "-c", script, str(identity), ADDRESS] for _ in range(8)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                outputs = list(
                    executor.map(
                        lambda command: subprocess.check_output(
                            command, text=True, timeout=5
                        ).strip(),
                        commands,
                    )
                )
            self.assertEqual(1, len(set(outputs)))

    def test_killed_lock_owner_is_recovered_without_stale_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = Path(directory) / "identity"
            identity.mkdir()
            lock_path = identity / "device_identity.key.lock"
            ready_path = identity / "owner.ready"
            script = (
                "import sys,time; from pathlib import Path; "
                "from device_identity import _exclusive_creation_lock; "
                "\nwith _exclusive_creation_lock(Path(sys.argv[1])):\n"
                " Path(sys.argv[2]).write_text('READY')\n time.sleep(60)"
            )
            child = subprocess.Popen(
                [sys.executable, "-c", script, str(lock_path), str(ready_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                ready_deadline = time.monotonic() + 5.0
                while not ready_path.exists() and time.monotonic() < ready_deadline:
                    if child.poll() is not None:
                        stderr = child.stderr.read() if child.stderr is not None else ""
                        self.fail(f"lock owner exited before readiness: {stderr}")
                    time.sleep(0.01)
                self.assertTrue(ready_path.exists(), "lock owner readiness timed out")
                child.kill()
                child.wait(timeout=5)
                started = time.monotonic()
                probe = (
                    "from device_identity import DeviceIdentityStore; import sys; "
                    "print(DeviceIdentityStore(sys.argv[1]).device_key(sys.argv[2]))"
                )
                output = subprocess.check_output(
                    [sys.executable, "-c", probe, str(identity), ADDRESS],
                    text=True,
                    timeout=3,
                ).strip()
                self.assertLess(time.monotonic() - started, 2.0)
                self.assertRegex(output, r"^dev-[0-9a-f]{32}$")
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
                if child.stderr is not None:
                    child.stderr.close()

    def test_permission_denial_is_immediate_and_preserves_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = Path(directory) / "identity"
            identity.mkdir()
            denied = PermissionError(13, "simulated ACL denial")
            started = time.monotonic()
            with mock.patch("device_identity._acquire_kernel_lock", side_effect=denied):
                with self.assertRaises(PermissionError) as raised:
                    DeviceIdentityStore(identity)
            self.assertIs(denied, raised.exception)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual({}, identity_module._LOCK_REGISTRY)

    def test_kernel_lock_cleanup_aggregates_release_and_close_failures(self):
        release_error = OSError("release failed")
        close_error = OSError("close failed")

        def fail_release(_handle):
            raise release_error

        def fail_close(_handle):
            raise close_error

        lock = identity_module._KernelLock(object(), fail_release, fail_close)
        with self.assertRaises(IdentityLockCleanupError) as raised:
            lock.release_and_close()
        self.assertEqual(
            (("release", release_error), ("close", close_error)),
            raised.exception.failures,
        )

    def test_cleanup_failures_do_not_leak_registry_entry(self):
        release_error = OSError("release failed")
        close_error = OSError("close failed")

        def fail_release(_handle):
            raise release_error

        def fail_close(_handle):
            raise close_error

        kernel_lock = identity_module._KernelLock(object(), fail_release, fail_close)
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "identity.lock"
            with mock.patch("device_identity._acquire_kernel_lock", return_value=kernel_lock):
                with self.assertRaises(IdentityLockCleanupError) as raised:
                    with identity_module._exclusive_creation_lock(lock_path):
                        pass
        self.assertEqual(
            (("release", release_error), ("close", close_error)),
            raised.exception.failures,
        )
        self.assertEqual({}, identity_module._LOCK_REGISTRY)

    def test_business_and_cleanup_failures_are_aggregated_in_order(self):
        business_error = RuntimeError("business failed")
        release_error = OSError("release failed")

        def fail_release(_handle):
            raise release_error

        kernel_lock = identity_module._KernelLock(
            object(),
            fail_release,
            lambda _handle: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "identity.lock"
            with mock.patch("device_identity._acquire_kernel_lock", return_value=kernel_lock):
                with self.assertRaises(IdentityLockCleanupError) as raised:
                    with identity_module._exclusive_creation_lock(lock_path):
                        raise business_error
        self.assertEqual(
            (("body", business_error), ("release", release_error)),
            raised.exception.failures,
        )
        self.assertIs(business_error, raised.exception.__cause__)
        self.assertEqual({}, identity_module._LOCK_REGISTRY)

    @unittest.skipIf(os.name == "nt", "POSIX-only permission invariant")
    def test_posix_mode_and_owner_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DeviceIdentityStore(Path(directory) / "identity")
            for path in (store.key_path, store.backup_path):
                metadata = path.stat()
                self.assertEqual(os.geteuid(), metadata.st_uid)
                self.assertEqual(0o600, stat.S_IMODE(metadata.st_mode))
            store.key_path.chmod(0o640)
            with self.assertRaises(IdentityPermissionError):
                DeviceIdentityStore(Path(directory) / "identity")

    @unittest.skipIf(os.name == "nt", "POSIX-only permission invariant")
    def test_posix_unsafe_backup_fails_closed_even_when_main_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = Path(directory) / "identity"
            store = DeviceIdentityStore(identity)
            store.backup_path.chmod(0o644)
            with self.assertRaises(IdentityPermissionError):
                DeviceIdentityStore(identity)

    @unittest.skipUnless(os.name == "nt", "Windows DPAPI-only assertion")
    def test_windows_file_does_not_contain_plaintext_secret_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DeviceIdentityStore(Path(directory) / "identity")
            data = store.key_path.read_bytes()
            self.assertGreater(len(data), 32 + 16)
            self.assertEqual(b"EMGIDKEY", data[:8])

    @unittest.skipUnless(os.name == "nt", "Windows DPAPI-only assertion")
    def test_windows_dpapi_integrity_rejects_rechecksummed_ciphertext(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = Path(directory) / "identity"
            store = DeviceIdentityStore(identity)
            for path in (store.key_path, store.backup_path):
                damaged = bytearray(path.read_bytes())
                damaged[-33] ^= 0x01
                damaged[-32:] = hashlib.sha256(
                    _INTEGRITY_CONTEXT + damaged[:-32]
                ).digest()
                path.write_bytes(damaged)
            with self.assertRaises(IdentityIntegrityError):
                DeviceIdentityStore(identity)


if __name__ == "__main__":
    unittest.main()
