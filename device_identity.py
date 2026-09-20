"""Secure, stable, host-local device identity derivation.

The persisted secret is protected with Windows DPAPI when available.  On
POSIX it is stored in an owner-only file and unsafe ownership or permissions
are rejected.  Both platforms use a versioned envelope and an atomic backup
so ordinary corruption never silently changes device identities.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import hashlib
import hmac
import os
import secrets
import stat
import struct
import threading
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Union

from emg_protocol import DeviceKey


PathLike = Union[str, os.PathLike[str]]

_MAGIC = b"EMGIDKEY"
_VERSION = 1
_SCHEME_DPAPI = 1
_SCHEME_POSIX_0600 = 2
_HEADER = struct.Struct("<8sBBHI")
_DIGEST_SIZE = hashlib.sha256().digest_size
_SECRET_SIZE = 32
_MAX_PAYLOAD_SIZE = 16 * 1024
_INTEGRITY_CONTEXT = b"emg-device-identity-envelope-v1\0"
_DERIVATION_CONTEXT = b"emg-device-v1\0"
_DPAPI_ENTROPY = b"emg-acquisition-device-identity-v1"


class DeviceIdentityError(RuntimeError):
    """Base error for persistent identity failures."""


class IdentityIntegrityError(DeviceIdentityError):
    """The primary and backup identity material cannot be trusted."""


class IdentityPermissionError(DeviceIdentityError):
    """POSIX identity material has unsafe ownership or permissions."""


class IdentityLockCleanupError(DeviceIdentityError):
    """One or more kernel-lock cleanup operations failed."""

    def __init__(self, failures):
        self.failures = tuple(failures)
        super().__init__(
            "; ".join(f"{stage}: {type(error).__name__}: {error}" for stage, error in self.failures)
        )


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _blob(data: bytes):
    buffer = ctypes.create_string_buffer(data)
    value = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    return value, buffer


def _dpapi_transform(data: bytes, *, protect: bool) -> bytes:
    if os.name != "nt":
        raise DeviceIdentityError("Windows DPAPI is unavailable on this platform")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    source, source_buffer = _blob(data)
    entropy, entropy_buffer = _blob(_DPAPI_ENTROPY)
    output = _DataBlob()
    flags = 0x1  # CRYPTPROTECT_UI_FORBIDDEN
    if protect:
        function = crypt32.CryptProtectData
        function.argtypes = (
            ctypes.POINTER(_DataBlob), ctypes.c_wchar_p, ctypes.POINTER(_DataBlob),
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(_DataBlob),
        )
        description = ctypes.c_wchar_p("EMG device identity")
        args = (ctypes.byref(source), description, ctypes.byref(entropy), None, None, flags, ctypes.byref(output))
    else:
        function = crypt32.CryptUnprotectData
        function.argtypes = (
            ctypes.POINTER(_DataBlob), ctypes.c_void_p, ctypes.POINTER(_DataBlob),
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(_DataBlob),
        )
        args = (ctypes.byref(source), None, ctypes.byref(entropy), None, None, flags, ctypes.byref(output))
    function.restype = ctypes.c_int
    if not function(*args):
        error = ctypes.get_last_error()
        raise IdentityIntegrityError(f"DPAPI operation failed (winerror={error})")
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        kernel32.LocalFree.argtypes = (ctypes.c_void_p,)
        kernel32.LocalFree.restype = ctypes.c_void_p
        kernel32.LocalFree(output.pbData)


def _protect_secret(secret: bytes) -> tuple[int, bytes]:
    if os.name == "nt":
        return _SCHEME_DPAPI, _dpapi_transform(secret, protect=True)
    return _SCHEME_POSIX_0600, secret


def _unprotect_secret(scheme: int, payload: bytes) -> bytes:
    expected = _SCHEME_DPAPI if os.name == "nt" else _SCHEME_POSIX_0600
    if scheme != expected:
        raise IdentityIntegrityError("identity protection scheme does not match this platform")
    return _dpapi_transform(payload, protect=False) if os.name == "nt" else payload


def _encode(secret: bytes) -> bytes:
    if len(secret) != _SECRET_SIZE:
        raise ValueError("identity secret must be exactly 32 bytes")
    scheme, payload = _protect_secret(secret)
    header = _HEADER.pack(_MAGIC, _VERSION, scheme, 0, len(payload))
    digest = hashlib.sha256(_INTEGRITY_CONTEXT + header + payload).digest()
    return header + payload + digest


def _decode(encoded: bytes) -> bytes:
    if len(encoded) < _HEADER.size + _DIGEST_SIZE:
        raise IdentityIntegrityError("identity file is truncated")
    magic, version, scheme, reserved, payload_length = _HEADER.unpack_from(encoded)
    if magic != _MAGIC or version != _VERSION or reserved != 0:
        raise IdentityIntegrityError("identity file header is invalid or unsupported")
    if payload_length <= 0 or payload_length > _MAX_PAYLOAD_SIZE:
        raise IdentityIntegrityError("identity payload length is invalid")
    expected_length = _HEADER.size + payload_length + _DIGEST_SIZE
    if len(encoded) != expected_length:
        raise IdentityIntegrityError("identity file length does not match its header")
    payload = encoded[_HEADER.size : _HEADER.size + payload_length]
    stored_digest = encoded[-_DIGEST_SIZE:]
    expected_digest = hashlib.sha256(_INTEGRITY_CONTEXT + encoded[:-_DIGEST_SIZE]).digest()
    if not hmac.compare_digest(stored_digest, expected_digest):
        raise IdentityIntegrityError("identity file integrity check failed")
    secret = _unprotect_secret(scheme, payload)
    if len(secret) != _SECRET_SIZE:
        raise IdentityIntegrityError("unprotected identity secret has invalid length")
    return secret


def _validate_identity_metadata(metadata: os.stat_result, path: Path) -> None:
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise IdentityPermissionError(f"identity path is not a regular file: {path}")
    if os.name == "nt":
        return
    if metadata.st_uid != os.geteuid():
        raise IdentityPermissionError(f"identity file is not owned by the current user: {path}")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise IdentityPermissionError(f"identity file permissions must be 0600: {path}")


def _validate_identity_file(path: Path) -> None:
    _validate_identity_metadata(path.lstat(), path)


def _read_verified(path: Path) -> bytes:
    try:
        if os.name == "nt":
            _validate_identity_file(path)
            data = path.read_bytes()
        else:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                _validate_identity_metadata(os.fstat(descriptor), path)
                with os.fdopen(descriptor, "rb", closefd=False) as stream:
                    data = stream.read(_HEADER.size + _MAX_PAYLOAD_SIZE + _DIGEST_SIZE + 1)
            finally:
                os.close(descriptor)
    except OSError as error:
        raise DeviceIdentityError(f"cannot read identity file {path}: {error}") from error
    return _decode(data)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(6)}.tmp"
    )
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name != "nt":
            os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        _validate_identity_file(path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class _RegistryEntry:
    def __init__(self):
        self.lock = threading.Lock()
        self.users = 0


_LOCK_REGISTRY: Dict[str, _RegistryEntry] = {}
_LOCK_REGISTRY_GUARD = threading.Lock()


def _register_local_lock(identity: str) -> _RegistryEntry:
    with _LOCK_REGISTRY_GUARD:
        entry = _LOCK_REGISTRY.get(identity)
        if entry is None:
            entry = _RegistryEntry()
            _LOCK_REGISTRY[identity] = entry
        entry.users += 1
        return entry


def _unregister_local_lock(identity: str, entry: _RegistryEntry) -> None:
    with _LOCK_REGISTRY_GUARD:
        current = _LOCK_REGISTRY.get(identity)
        if current is not entry or entry.users <= 0:
            raise RuntimeError("identity lock registry is inconsistent")
        entry.users -= 1
        if entry.users == 0:
            del _LOCK_REGISTRY[identity]


class _KernelLock:
    def __init__(self, handle, release, close):
        self.handle = handle
        self._release = release
        self._close = close

    def release_and_close(self) -> None:
        failures = []
        try:
            self._release(self.handle)
        except BaseException as error:
            failures.append(("release", error))
        try:
            self._close(self.handle)
        except BaseException as error:
            failures.append(("close", error))
        if failures:
            raise IdentityLockCleanupError(failures)


def _acquire_windows_mutex(identity: str, timeout_seconds: float) -> _KernelLock:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_mutex = kernel32.CreateMutexW
    create_mutex.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
    create_mutex.restype = ctypes.c_void_p
    digest = hashlib.sha256(identity.encode("utf-8", errors="strict")).hexdigest()
    handle = create_mutex(None, False, "Local\\EMGDeviceIdentity-" + digest)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())

    wait = kernel32.WaitForSingleObject
    wait.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    wait.restype = ctypes.c_uint32
    timeout_ms = min(0xFFFFFFFE, max(0, int(timeout_seconds * 1000)))
    result = wait(handle, timeout_ms)
    if result not in (0x00000000, 0x00000080):  # WAIT_OBJECT_0, WAIT_ABANDONED
        wait_error_code = ctypes.get_last_error() if result == 0xFFFFFFFF else 0
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (ctypes.c_void_p,)
        close_handle.restype = ctypes.c_int
        close_error = None
        if not close_handle(handle):
            close_error = ctypes.WinError(ctypes.get_last_error())
        if result == 0x00000102:  # WAIT_TIMEOUT
            error = DeviceIdentityError("timed out waiting for identity mutex")
        elif result == 0xFFFFFFFF:  # WAIT_FAILED
            error = ctypes.WinError(wait_error_code)
        else:
            error = DeviceIdentityError(f"unexpected mutex wait result: {result:#x}")
        if close_error is not None:
            raise IdentityLockCleanupError((("wait", error), ("close", close_error)))
        raise error

    def release(value) -> None:
        function = kernel32.ReleaseMutex
        function.argtypes = (ctypes.c_void_p,)
        function.restype = ctypes.c_int
        if not function(value):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(value) -> None:
        function = kernel32.CloseHandle
        function.argtypes = (ctypes.c_void_p,)
        function.restype = ctypes.c_int
        if not function(value):
            raise ctypes.WinError(ctypes.get_last_error())

    return _KernelLock(handle, release, close)


def _acquire_posix_flock(path: Path, timeout_seconds: float) -> _KernelLock:
    import fcntl

    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        _validate_identity_metadata(os.fstat(descriptor), path)
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise DeviceIdentityError(f"timed out waiting for identity lock: {path}")
                time.sleep(0.01)
    except BaseException:
        os.close(descriptor)
        raise

    return _KernelLock(
        descriptor,
        lambda value: fcntl.flock(value, fcntl.LOCK_UN),
        os.close,
    )


def _acquire_kernel_lock(path: Path, identity: str, timeout_seconds: float) -> _KernelLock:
    if os.name == "nt":
        return _acquire_windows_mutex(identity, timeout_seconds)
    return _acquire_posix_flock(path, timeout_seconds)


def _canonical_lock_identity(path: Path) -> str:
    identity = os.path.normcase(os.path.abspath(str(path)))
    if os.name == "nt" and identity.startswith("\\\\?\\"):
        if identity.startswith("\\\\?\\unc\\"):
            identity = "\\\\" + identity[8:]
        else:
            identity = identity[4:]
    return identity


@contextlib.contextmanager
def _exclusive_creation_lock(path: Path, timeout_seconds: float = 10.0) -> Iterator[None]:
    identity = _canonical_lock_identity(path)
    deadline = time.monotonic() + timeout_seconds
    entry = _register_local_lock(identity)
    local_acquired = False
    kernel_lock = None
    primary_error = None
    try:
        remaining = max(0.0, deadline - time.monotonic())
        local_acquired = entry.lock.acquire(timeout=remaining)
        if not local_acquired:
            raise DeviceIdentityError(f"timed out waiting for local identity lock: {path}")
        remaining = max(0.0, deadline - time.monotonic())
        kernel_lock = _acquire_kernel_lock(path, identity, remaining)
        yield
    except BaseException as error:
        primary_error = error
        raise
    finally:
        failures = []
        if kernel_lock is not None:
            try:
                kernel_lock.release_and_close()
            except IdentityLockCleanupError as error:
                failures.extend(error.failures)
            except BaseException as error:
                failures.append(("kernel_cleanup", error))
        if local_acquired:
            try:
                entry.lock.release()
            except BaseException as error:
                failures.append(("local_release", error))
        try:
            _unregister_local_lock(identity, entry)
        except BaseException as error:
            failures.append(("registry_cleanup", error))
        if failures:
            if primary_error is not None:
                failures.insert(0, ("body", primary_error))
            raise IdentityLockCleanupError(failures) from primary_error


class DeviceIdentityStore:
    """Persist one protected host key and derive opaque :class:`DeviceKey` values.

    ``identity_path`` may be an explicit ``*.key`` file (compatibility mode) or
    a directory, in which case ``device_identity.key`` is used.  ``data_root``
    is optional but, when supplied, prevents putting the host secret inside the
    recorded-data tree.
    """

    def __init__(self, identity_path: PathLike, data_root: Optional[PathLike] = None):
        requested = Path(identity_path).expanduser()
        if requested.suffix.lower() == ".key":
            key_path = requested.resolve()
        else:
            key_path = (requested / "device_identity.key").resolve()
        if data_root is not None:
            resolved_data_root = Path(data_root).expanduser().resolve()
            if key_path == resolved_data_root or resolved_data_root in key_path.parents:
                raise ValueError("device identity key must be stored outside the data root")
        key_path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path = key_path
        self.backup_path = key_path.with_name(key_path.name + ".bak")
        self._secret = self._load_or_create()

    def _load_or_create(self) -> bytes:
        lock_path = self.key_path.with_name(self.key_path.name + ".lock")
        with _exclusive_creation_lock(lock_path):
            main_exists = self.key_path.exists() or self.key_path.is_symlink()
            backup_exists = self.backup_path.exists() or self.backup_path.is_symlink()
            main_secret, main_error = self._try_read(self.key_path) if main_exists else (None, None)
            backup_secret, backup_error = self._try_read(self.backup_path) if backup_exists else (None, None)

            if main_secret is not None:
                encoded = _encode(main_secret)
                if backup_secret != main_secret:
                    _atomic_write(self.backup_path, encoded)
                return main_secret
            if backup_secret is not None:
                _atomic_write(self.key_path, _encode(backup_secret))
                return backup_secret
            if main_exists or backup_exists:
                details = "; ".join(
                    str(error) for error in (main_error, backup_error) if error is not None
                )
                raise IdentityIntegrityError(
                    "existing identity material is unusable; refusing to generate a new key"
                    + (f": {details}" if details else "")
                )

            secret = secrets.token_bytes(_SECRET_SIZE)
            encoded = _encode(secret)
            _atomic_write(self.backup_path, encoded)
            _atomic_write(self.key_path, encoded)
            return secret

    @staticmethod
    def _try_read(path: Path):
        try:
            return _read_verified(path), None
        except IdentityPermissionError:
            raise
        except (DeviceIdentityError, OSError) as error:
            return None, error

    def get_device_key(self, identifier: str) -> DeviceKey:
        normalized = identifier.strip().casefold().encode("utf-8", errors="strict")
        if not normalized:
            raise ValueError("device identifier is empty")
        digest = hmac.digest(self._secret, _DERIVATION_CONTEXT + normalized, "sha256").hex()[:32]
        return DeviceKey("dev-" + digest)

    def device_key(self, identifier: str) -> DeviceKey:
        """Compatibility alias for the desktop application's original API."""
        return self.get_device_key(identifier)

__all__ = [
    "DeviceIdentityError",
    "DeviceIdentityStore",
    "IdentityIntegrityError",
    "IdentityLockCleanupError",
    "IdentityPermissionError",
]
