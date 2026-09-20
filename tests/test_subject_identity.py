import json
import os
import re
import stat
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import subject_identity
from subject_identity import SubjectIdentityStore, normalize_subject_input

DOMAIN = "dexterous_hand_v1"


class SubjectIdentityStoreTests(unittest.TestCase):
    @staticmethod
    def _identity_file_bytes(store):
        return {
            path.name: path.read_bytes()
            for path in (
                store.lock_path,
                store.key_path,
                store.key_id_path,
                store.state_path,
            )
            if path.is_file()
        }

    def test_stable_across_instances_and_distinct_for_different_input(self):
        with tempfile.TemporaryDirectory() as directory:
            first = SubjectIdentityStore(directory, dataset_domain=DOMAIN).derive_subject_id("S-001")
            restarted = SubjectIdentityStore(directory, dataset_domain=DOMAIN).derive_subject_id("S-001")
            other = SubjectIdentityStore(directory, dataset_domain=DOMAIN).derive_subject_id("S-002")
            self.assertEqual(first, restarted)
            self.assertNotEqual(first, other)
            self.assertRegex(first, r"\Asub-[0-9a-f]{32}\Z")

    def test_first_initialization_persists_completed_non_sensitive_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SubjectIdentityStore(directory, dataset_domain=DOMAIN)
            store.derive_subject_id("P-001")

            state = json.loads(store.state_path.read_text(encoding="ascii"))
            self.assertEqual(
                set(state), {"schema", "initialized", "key_id", "dataset_domain"}
            )
            self.assertEqual(state["schema"], 1)
            self.assertIs(state["initialized"], True)
            self.assertEqual(state["dataset_domain"], DOMAIN)
            self.assertRegex(state["key_id"], r"\A[0-9a-f]{32}\Z")
            self.assertEqual(store.lock_path.read_bytes(), b"1")

    def test_unicode_normalization_case_and_whitespace_are_canonical(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SubjectIdentityStore(directory, dataset_domain=DOMAIN)
            variants = ("  Ａlice\u3000Smith  ", "alice smith", "ALICE   SMITH")
            self.assertEqual(len({store.derive_subject_id(item) for item in variants}), 1)
            self.assertEqual(normalize_subject_input("  编号  ００１ "), "编号 001")

    def test_plaintext_is_never_written_to_identity_store(self):
        with tempfile.TemporaryDirectory() as directory:
            secret_input = "Patient Alice 1980-01-01"
            store = SubjectIdentityStore(directory, dataset_domain=DOMAIN)
            subject_id = store.derive_subject_id(secret_input)
            disk_bytes = b"".join(
                path.read_bytes() for path in Path(directory).iterdir() if path.is_file()
            )
            self.assertNotIn(secret_input.encode("utf-8"), disk_bytes)
            self.assertNotIn(subject_id.encode("ascii"), disk_bytes)

    def test_concurrent_first_use_converges_on_one_key(self):
        with tempfile.TemporaryDirectory() as directory:
            stores = [SubjectIdentityStore(directory, dataset_domain=DOMAIN) for _ in range(24)]
            with ThreadPoolExecutor(max_workers=12) as pool:
                results = list(pool.map(lambda item: item.derive_subject_id("P-007"), stores))
            self.assertEqual(len(set(results)), 1)
            files = [path.name for path in Path(directory).iterdir() if path.is_file()]
            self.assertEqual(
                set(files),
                {
                    "subject_identity.init.lock",
                    "subject_identity.key",
                    "subject_identity.key.id",
                    "subject_identity.state",
                },
            )

    def test_corrupt_or_symlink_key_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SubjectIdentityStore(directory, dataset_domain=DOMAIN)
            store.derive_subject_id("P-001")
            key = store.key_path
            key.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                SubjectIdentityStore(directory, dataset_domain=DOMAIN).derive_subject_id("P-001")
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            store = SubjectIdentityStore(directory, dataset_domain=DOMAIN)
            store.derive_subject_id("P-001")
            target = Path(outside) / "key"
            target.write_bytes(b"x" * 40)
            key = store.key_path
            key.unlink()
            try:
                key.symlink_to(target)
            except OSError:
                self.skipTest("symlink creation is unavailable")
            with self.assertRaises(PermissionError):
                SubjectIdentityStore(directory, dataset_domain=DOMAIN).derive_subject_id("P-001")

    @unittest.skipIf(os.name == "nt", "POSIX mode bits are not authoritative on Windows")
    def test_broad_key_permissions_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SubjectIdentityStore(directory, dataset_domain=DOMAIN)
            store.derive_subject_id("P-001")
            os.chmod(store.key_path, 0o644)
            with self.assertRaisesRegex(PermissionError, "permissions"):
                SubjectIdentityStore(directory, dataset_domain=DOMAIN).derive_subject_id("P-001")
            self.assertEqual(stat.S_IMODE(store.storage_dir.stat().st_mode), 0o700)

    def test_invalid_input_is_rejected_without_creating_key(self):
        invalid = (None, "", "   ", "line\nbreak", "a" * 257)
        for value in invalid:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    SubjectIdentityStore(directory, dataset_domain=DOMAIN).derive_subject_id(value)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_key_or_key_id_loss_refuses_silent_rotation(self):
        for missing_name in ("subject_identity.key", "subject_identity.key.id"):
            with self.subTest(missing=missing_name), tempfile.TemporaryDirectory() as directory:
                store = SubjectIdentityStore(directory, dataset_domain=DOMAIN)
                original = store.derive_subject_id("P-001")
                (Path(directory) / missing_name).unlink()
                with self.assertRaisesRegex(FileNotFoundError, "refusing rotation"):
                    SubjectIdentityStore(directory, dataset_domain=DOMAIN).derive_subject_id("P-001")
                self.assertRegex(original, r"\Asub-[0-9a-f]{32}\Z")

    def test_both_key_files_lost_after_initialization_refuses_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SubjectIdentityStore(directory, dataset_domain=DOMAIN)
            original = store.derive_subject_id("P-001")
            store.key_path.unlink()
            store.key_id_path.unlink()

            with self.assertRaisesRegex(FileNotFoundError, "previously initialized"):
                SubjectIdentityStore(directory, dataset_domain=DOMAIN).derive_subject_id("P-001")

            self.assertTrue(store.lock_path.is_file())
            self.assertTrue(store.state_path.is_file())
            self.assertRegex(original, r"\Asub-[0-9a-f]{32}\Z")

    def test_lock_marker_alone_refuses_reinitialization(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SubjectIdentityStore(directory, dataset_domain=DOMAIN)
            store.derive_subject_id("P-001")
            store.key_path.unlink()
            store.key_id_path.unlink()
            store.state_path.unlink()

            with self.assertRaisesRegex(FileNotFoundError, "previously initialized"):
                SubjectIdentityStore(directory, dataset_domain=DOMAIN).derive_subject_id("P-001")

    def test_state_loss_after_completed_initialization_never_repairs_or_rewrites(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SubjectIdentityStore(directory, dataset_domain=DOMAIN)
            store.derive_subject_id("P-001")
            store.state_path.unlink()
            expected = self._identity_file_bytes(store)

            for _ in range(2):
                with self.assertRaisesRegex(FileNotFoundError, "state is missing"):
                    SubjectIdentityStore(directory, dataset_domain=DOMAIN).derive_subject_id(
                        "P-001"
                    )
                self.assertEqual(self._identity_file_bytes(store), expected)
                self.assertFalse(store.state_path.exists())

    def test_marker_zero_allows_completion_of_interrupted_first_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SubjectIdentityStore(directory, dataset_domain=DOMAIN)
            original = store.derive_subject_id("P-001")
            store.state_path.unlink()
            store.lock_path.write_bytes(b"0")

            recovered = SubjectIdentityStore(
                directory, dataset_domain=DOMAIN
            ).derive_subject_id("P-001")

            self.assertEqual(recovered, original)
            self.assertTrue(store.state_path.is_file())
            self.assertEqual(store.lock_path.read_bytes(), b"1")

    def test_corrupt_persisted_identity_state_fails_without_rewriting_any_file(self):
        cases = {
            "non-json": lambda store: store.state_path.write_bytes(b"not-json"),
            "wrong-schema": lambda store: store.state_path.write_text(
                json.dumps(
                    {
                        "schema": 2,
                        "initialized": True,
                        "key_id": json.loads(store.key_id_path.read_text(encoding="ascii"))["key_id"],
                        "dataset_domain": DOMAIN,
                    }
                ),
                encoding="ascii",
            ),
            "wrong-key-id": lambda store: store.state_path.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "initialized": True,
                        "key_id": "f" * 32,
                        "dataset_domain": DOMAIN,
                    }
                ),
                encoding="ascii",
            ),
            "wrong-domain": lambda store: store.state_path.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "initialized": True,
                        "key_id": json.loads(store.key_id_path.read_text(encoding="ascii"))["key_id"],
                        "dataset_domain": "other_study",
                    }
                ),
                encoding="ascii",
            ),
            "corrupt-key-id-file": lambda store: store.key_id_path.write_bytes(b"not-json"),
        }
        for name, corrupt in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                store = SubjectIdentityStore(directory, dataset_domain=DOMAIN)
                store.derive_subject_id("P-001")
                corrupt(store)
                expected = self._identity_file_bytes(store)

                for _ in range(2):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        SubjectIdentityStore(directory, dataset_domain=DOMAIN).derive_subject_id(
                            "P-001"
                        )
                    self.assertEqual(self._identity_file_bytes(store), expected)

    def test_persisted_domain_cannot_change_and_domain_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            SubjectIdentityStore(directory, dataset_domain=DOMAIN).derive_subject_id("P-001")
            with self.assertRaisesRegex(ValueError, "dataset_domain differs"):
                SubjectIdentityStore(directory, dataset_domain="other_study").derive_subject_id("P-001")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TypeError):
                SubjectIdentityStore(directory)
            for invalid in ("", "Project", "../x", "with space", "实验"):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    SubjectIdentityStore(directory, dataset_domain=invalid)

    def test_format_bidi_zero_width_and_surrogate_characters_are_rejected(self):
        for value in ("a\u200bb", "a\u202eb", "a\ufeffb", "a\ud800b", "a\x00b"):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                normalize_subject_input(value)

    @unittest.skipUnless(os.name == "nt", "Windows ACL verification")
    def test_windows_acl_is_really_private_and_acl_failure_is_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SubjectIdentityStore(Path(directory) / "identity", dataset_domain=DOMAIN)
            store.derive_subject_id("P-001")
            for path in (
                store.storage_dir,
                store.lock_path,
                store.key_path,
                store.key_id_path,
                store.state_path,
            ):
                subject_identity._validate_windows_private_acl(path)
            with mock.patch.object(subject_identity, "_validate_windows_private_acl", side_effect=PermissionError("acl")):
                with self.assertRaises(PermissionError):
                    SubjectIdentityStore(Path(directory) / "blocked", dataset_domain=DOMAIN).derive_subject_id("P-001")


if __name__ == "__main__":
    unittest.main()
