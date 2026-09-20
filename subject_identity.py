"""Stable local pseudonyms for recording subjects without persisting plaintext identity."""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import os
import re
import stat
import subprocess
import threading
import time
import unicodedata
from ctypes import wintypes
from pathlib import Path
from typing import BinaryIO, Union
from uuid import uuid4

PathLike = Union[str, os.PathLike[str]]
_SUBJECT_ID = re.compile(r"sub-[0-9a-f]{32}\Z", re.ASCII)
_DATASET_DOMAIN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z", re.ASCII)
_KEY_MAGIC = b"EMGSUBJ\x02"
_KEY_BYTES = 32
_KEY_FILE = "subject_identity.key"
_KEY_ID_FILE = "subject_identity.key.id"
_STATE_FILE = "subject_identity.state"
_LOCK_FILE = "subject_identity.init.lock"
_SUBJECT_DOMAIN = b"emg.subject-identity.v2\x00"
_MAX_INPUT_CHARS = 256
_BANNED_UNICODE_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})


def validate_subject_id(value: object) -> str:
    if not isinstance(value, str) or not _SUBJECT_ID.fullmatch(value):
        raise ValueError("subject_id must be an opaque sub-<32 lowercase hex> key")
    return value


def _validate_dataset_domain(value: object) -> str:
    if not isinstance(value, str) or not _DATASET_DOMAIN.fullmatch(value):
        raise ValueError("dataset_domain must be a stable lowercase safe identifier")
    return value


def normalize_subject_input(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("subject input must be text")
    if len(value) > _MAX_INPUT_CHARS:
        raise ValueError(f"subject input must not exceed {_MAX_INPUT_CHARS} characters")
    if any(unicodedata.category(char) in _BANNED_UNICODE_CATEGORIES for char in value):
        raise ValueError("subject input contains forbidden control or format characters")
    normalized = unicodedata.normalize("NFKC", value)
    if len(normalized) > _MAX_INPUT_CHARS:
        raise ValueError(f"normalized subject input must not exceed {_MAX_INPUT_CHARS} characters")
    if any(unicodedata.category(char) in _BANNED_UNICODE_CATEGORIES for char in normalized):
        raise ValueError("normalized subject input contains forbidden Unicode characters")
    normalized = " ".join(normalized.split()).casefold()
    if not normalized:
        raise ValueError("subject input must not be empty")
    if len(normalized) > _MAX_INPUT_CHARS:
        raise ValueError(f"normalized subject input must not exceed {_MAX_INPUT_CHARS} characters")
    return normalized


class _InitializationLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream: BinaryIO | None = None

    def __enter__(self) -> "_InitializationLock":
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt
            deadline = time.monotonic() + 10.0
            while True:
                try:
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("timed out acquiring subject identity initialization lock") from exc
                    time.sleep(0.01)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
        if self.stream.seek(0, os.SEEK_END) == 0:
            self.stream.write(b"0")
            self.stream.flush()
            os.fsync(self.stream.fileno())
        self.stream.seek(0)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert self.stream is not None
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        self.stream = None

    def was_initialized(self) -> bool:
        assert self.stream is not None
        self.stream.seek(0)
        marker = self.stream.read(2)
        if marker not in {b"0", b"1", b"01"}:
            raise ValueError("subject identity initialization lock state is corrupt")
        return marker in {b"1", b"01"}

    def mark_initialized(self) -> None:
        assert self.stream is not None
        self.stream.seek(0)
        self.stream.truncate(0)
        self.stream.write(b"1")
        self.stream.truncate()
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.seek(0)


class SubjectIdentityStore:
    def __init__(self, storage_dir: PathLike, *, dataset_domain: str) -> None:
        self.storage_dir = Path(storage_dir).expanduser().resolve()
        self.dataset_domain = _validate_dataset_domain(dataset_domain)
        self.key_path = self.storage_dir / _KEY_FILE
        self.key_id_path = self.storage_dir / _KEY_ID_FILE
        self.state_path = self.storage_dir / _STATE_FILE
        self.lock_path = self.storage_dir / _LOCK_FILE
        self._lock = threading.Lock()

    def derive_subject_id(self, subject_input: object) -> str:
        normalized = normalize_subject_input(subject_input)
        key = self._load_or_create_key()
        message = _SUBJECT_DOMAIN + self.dataset_domain.encode("ascii") + b"\x00" + normalized.encode("utf-8")
        return "sub-" + hmac.new(key, message, hashlib.sha256).hexdigest()[:32]

    def _load_or_create_key(self) -> bytes:
        with self._lock:
            self.storage_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._secure_path(self.storage_dir, directory=True)
            with _InitializationLock(self.lock_path) as initialization_lock:
                self._secure_path(self.lock_path, directory=False)
                key_exists = self.key_path.exists() or self.key_path.is_symlink()
                identity_exists = self.key_id_path.exists() or self.key_id_path.is_symlink()
                state_exists = self.state_path.exists() or self.state_path.is_symlink()
                if key_exists != identity_exists:
                    raise FileNotFoundError("subject identity key or key-id is missing; refusing rotation")
                if not key_exists:
                    if state_exists or initialization_lock.was_initialized():
                        raise FileNotFoundError(
                            "subject identity was previously initialized but key material is "
                            "missing; refusing rotation"
                        )
                    self._create_identity()
                key = self._read_key()
                self._validate_key_identity(key)
                if self.state_path.exists() or self.state_path.is_symlink():
                    self._validate_initialization_state(key)
                else:
                    if initialization_lock.was_initialized():
                        raise FileNotFoundError(
                            "subject identity initialization state is missing; refusing repair"
                        )
                    self._create_initialization_state(key)
                initialization_lock.mark_initialized()
                return key

    def _create_identity(self) -> None:
        key = os.urandom(_KEY_BYTES)
        key_id = hashlib.sha256(_KEY_MAGIC + key).hexdigest()[:32]
        metadata = json.dumps({"schema": 1, "key_id": key_id, "dataset_domain": self.dataset_domain}, sort_keys=True, separators=(",", ":")).encode("ascii")
        self._atomic_create(self.key_path, _KEY_MAGIC + key)
        self._atomic_create(self.key_id_path, metadata)
        self._create_initialization_state(key)
        self._sync_directory()

    def _create_initialization_state(self, key: bytes) -> None:
        key_id = hashlib.sha256(_KEY_MAGIC + key).hexdigest()[:32]
        state = json.dumps(
            {
                "schema": 1,
                "initialized": True,
                "key_id": key_id,
                "dataset_domain": self.dataset_domain,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        self._atomic_create(self.state_path, state)
        self._sync_directory()

    def _atomic_create(self, destination: Path, payload: bytes) -> None:
        temporary = self.storage_dir / f".{destination.name}.{uuid4().hex}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            self._secure_path(temporary, directory=False)
            os.link(temporary, destination)
            self._secure_path(destination, directory=False)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _read_key(self) -> bytes:
        payload = self._read_secure_regular_file(self.key_path, len(_KEY_MAGIC) + _KEY_BYTES + 1)
        if len(payload) != len(_KEY_MAGIC) + _KEY_BYTES or not payload.startswith(_KEY_MAGIC):
            raise ValueError("subject identity key is corrupt or uses an unsupported version")
        return payload[len(_KEY_MAGIC):]

    def _validate_key_identity(self, key: bytes) -> None:
        raw = self._read_secure_regular_file(self.key_id_path, 1025)
        try:
            metadata = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("subject identity key-id is corrupt") from exc
        if not isinstance(metadata, dict) or set(metadata) != {"schema", "key_id", "dataset_domain"} or metadata["schema"] != 1:
            raise ValueError("subject identity key-id has an invalid schema")
        actual_id = hashlib.sha256(_KEY_MAGIC + key).hexdigest()[:32]
        if not isinstance(metadata["key_id"], str) or not hmac.compare_digest(metadata["key_id"], actual_id):
            raise ValueError("subject identity key-id does not match the key")
        if metadata["dataset_domain"] != self.dataset_domain:
            raise ValueError("dataset_domain differs from the persisted identity domain")

    def _validate_initialization_state(self, key: bytes) -> None:
        raw = self._read_secure_regular_file(self.state_path, 1025)
        try:
            state = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("subject identity initialization state is corrupt") from exc
        expected_fields = {"schema", "initialized", "key_id", "dataset_domain"}
        if (
            not isinstance(state, dict)
            or set(state) != expected_fields
            or state["schema"] != 1
            or state["initialized"] is not True
        ):
            raise ValueError("subject identity initialization state has an invalid schema")
        actual_id = hashlib.sha256(_KEY_MAGIC + key).hexdigest()[:32]
        if not isinstance(state["key_id"], str) or not hmac.compare_digest(
            state["key_id"], actual_id
        ):
            raise ValueError("subject identity initialization state does not match the key")
        if state["dataset_domain"] != self.dataset_domain:
            raise ValueError("dataset_domain differs from the persisted initialization state")

    def _read_secure_regular_file(self, path: Path, limit: int) -> bytes:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise PermissionError("subject identity files must be regular non-symlink files")
        self._secure_path(path, directory=False, mutate=False)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            return stream.read(limit)

    def _secure_path(self, path: Path, *, directory: bool, mutate: bool = True) -> None:
        if os.name == "nt":
            if mutate:
                _set_windows_private_acl(path)
            _validate_windows_private_acl(path)
            return
        if mutate:
            os.chmod(path, 0o700 if directory else 0o600)
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise PermissionError("subject identity path permissions are too broad")

    def _sync_directory(self) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(self.storage_dir, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _windows_current_user_sid() -> str:
    completed = subprocess.run(["whoami.exe", "/user", "/fo", "csv", "/nh"], check=True, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    match = re.search(rb"S-\d+(?:-\d+)+", completed.stdout)
    if match is None:
        raise PermissionError("unable to determine current Windows user SID")
    return match.group(0).decode("ascii")


def _set_windows_private_acl(path: Path) -> None:
    user_sid = _windows_current_user_sid()
    completed = subprocess.run(["icacls.exe", str(path), "/inheritance:r", "/grant:r", f"*{user_sid}:(F)", "*S-1-5-18:(F)", "*S-1-5-32-544:(F)"], check=False, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if completed.returncode != 0:
        raise PermissionError("failed to set private Windows ACL")
    permitted = {user_sid, "SY", "BA", "S-1-5-18", "S-1-5-32-544"}
    aliases = {"WD": "S-1-1-0", "BU": "S-1-5-32-545", "AU": "S-1-5-11"}
    unwanted = set()
    for raw_ace in re.findall(r"\(([^()]*)\)", _windows_sddl(path)):
        fields = raw_ace.split(";")
        if len(fields) >= 6 and fields[0] == "A" and fields[5] not in permitted:
            unwanted.add(aliases.get(fields[5], fields[5]))
    for sid in unwanted:
        removed = subprocess.run(["icacls.exe", str(path), "/remove:g", f"*{sid}"], check=False, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if removed.returncode != 0:
            raise PermissionError("failed to remove an unexpected Windows ACL principal")


def _windows_sddl(path: Path) -> str:
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
        ctypes.POINTER(wintypes.LPWSTR), ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    security_descriptor = ctypes.c_void_p()
    result = advapi32.GetNamedSecurityInfoW(str(path), 1, 0x00000004, None, None, None, None, ctypes.byref(security_descriptor))
    if result:
        raise PermissionError(f"cannot read Windows ACL ({result})")
    text_pointer = wintypes.LPWSTR()
    text_length = wintypes.DWORD()
    try:
        if not advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW(security_descriptor, 1, 0x00000004, ctypes.byref(text_pointer), ctypes.byref(text_length)):
            raise PermissionError(f"cannot convert Windows ACL ({ctypes.get_last_error()})")
        return text_pointer.value
    finally:
        if text_pointer:
            kernel32.LocalFree(ctypes.cast(text_pointer, ctypes.c_void_p))
        if security_descriptor:
            kernel32.LocalFree(security_descriptor)


def _validate_windows_private_acl(path: Path) -> None:
    user_sid = _windows_current_user_sid()
    required = {user_sid, "SY", "BA"}
    permitted = required | {"S-1-5-18", "S-1-5-32-544"}
    dacl = _windows_sddl(path).split("D:", 1)[-1]
    allowed_seen: set[str] = set()
    for raw_ace in re.findall(r"\(([^()]*)\)", dacl):
        fields = raw_ace.split(";")
        if len(fields) < 6:
            raise PermissionError("Windows ACL contains an unparseable ACE")
        ace_type, ace_flags, permissions, sid = fields[0], fields[1], fields[2], fields[5]
        if ace_type != "A" or "ID" in ace_flags or sid not in permitted or "FA" not in permissions:
            raise PermissionError("Windows ACL grants unexpected or inherited access")
        allowed_seen.add(sid)
    canonical_seen = {"SY" if sid == "S-1-5-18" else "BA" if sid == "S-1-5-32-544" else sid for sid in allowed_seen}
    if not required.issubset(canonical_seen):
        raise PermissionError("Windows ACL is missing a required protected principal")
