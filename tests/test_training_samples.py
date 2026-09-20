import json
import dataclasses
import gc
import tempfile
import weakref
from pathlib import Path

import numpy as np
import pytest
import training_samples as training_samples_module
import training_dataset as training_dataset_module

from test_training_dataset import TrainingDatasetTests as _DatasetFixture
from training_dataset import prepare_training_dataset
from training_samples import (
    TrainingCorpus,
    TrainingSamplesError,
    iter_split_windows,
    load_training_manifest,
)

_DatasetFixture.__test__ = False


def _prepared_corpus(root: Path):
    fixture = _DatasetFixture()
    sessions = [
        fixture.make_session(
            root,
            number,
            f"session-{number}-{label}",
            [label] * 40,
        )
        for number in range(1, 4)
        for label in ("rest", "fist", "open_hand")
    ]
    manifest_path = root / "manifest.json"
    manifest = prepare_training_dataset(
        sessions,
        manifest_path,
        window_ms=200,
        step_ms=50,
        seed=7,
    )
    return manifest_path, manifest


def test_loads_exact_manifest_windows_and_channel_order():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, manifest = _prepared_corpus(root)
        corpus = load_training_manifest(manifest_path, root)

        for split_name in ("train", "validation", "test"):
            examples = list(iter_split_windows(corpus, split_name))
            declared = manifest["splits"][split_name]["windows"]
            assert [item.window_id for item in examples] == [item["window_id"] for item in declared]
            assert [item.label for item in examples] == [item["label"] for item in declared]
            assert all(item.samples.shape == (40, 8) for item in examples)
            assert all(item.samples.dtype == np.float64 for item in examples)
            assert all(not item.samples.flags.writeable for item in examples)


def test_manifest_source_hash_and_traversal_fail_closed():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, _ = _prepared_corpus(root)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["inputs"][0]["source_files"]["samples.csv"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(TrainingSamplesError, match="snapshot"):
            load_training_manifest(manifest_path, root)

        manifest_path.unlink()
        manifest_path, _ = _prepared_corpus(root / "second")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["inputs"][0]["session_locator"] = "../outside"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(TrainingSamplesError, match="subject/session"):
            load_training_manifest(manifest_path, root / "second")


def test_subject_and_window_overlap_are_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, _ = _prepared_corpus(root)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["splits"]["validation"]["subject_ids"][0] = payload["splits"]["train"]["subject_ids"][0]
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(TrainingSamplesError, match="overlap|identity"):
            load_training_manifest(manifest_path, root)


def test_corpus_is_deeply_frozen_and_test_provenance_is_loader_derived():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, _ = _prepared_corpus(root)
        corpus = load_training_manifest(manifest_path, root)

        assert corpus.provenance == "synthetic_test"
        with pytest.raises(TypeError):
            corpus.manifest["splits"]["train"]["windows"][0]["label"] = "fist"
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            corpus.manifest_path = root / "other.json"
        with pytest.raises(TypeError, match="load_training_manifest"):
            TrainingCorpus(
                manifest_path=manifest_path,
                session_root=root,
                manifest_sha256="0" * 64,
                manifest={},
                sessions={},
                provenance="canonical_session",
                _token=object(),
            )

        forged = dataclasses.replace(corpus, provenance="canonical_session")
        assert forged._token is corpus._token
        with pytest.raises(TrainingSamplesError, match="load_training_manifest"):
            list(iter_split_windows(forged, "train"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda window: window.update(window_id="0" * 64),
        lambda window: window.update(start_row=window["start_row"] + 1),
        lambda window: window.update(label="fist" if window["label"] != "fist" else "rest"),
    ],
)
def test_stale_manifest_window_rewrites_cannot_be_iterated(mutation):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, _ = _prepared_corpus(root)
        corpus = load_training_manifest(manifest_path, root)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutation(payload["splits"]["train"]["windows"][0])
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(TrainingSamplesError):
            list(iter_split_windows(corpus, "train"))


def test_duplicate_json_key_and_wrong_root_fail_closed():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, _ = _prepared_corpus(root)
        duplicate = manifest_path.read_text(encoding="utf-8").replace(
            '"schema": "emg.training.dataset_manifest",',
            '"schema": "emg.training.dataset_manifest", "schema": "emg.training.dataset_manifest",',
            1,
        )
        manifest_path.write_text(duplicate, encoding="utf-8")
        with pytest.raises(TrainingSamplesError, match="cannot parse training manifest"):
            load_training_manifest(manifest_path, root)

        manifest_path, _ = _prepared_corpus(root / "other")
        wrong_root = root / "wrong-root"
        wrong_root.mkdir()
        with pytest.raises(TrainingSamplesError):
            load_training_manifest(manifest_path, wrong_root)


def test_symlink_or_reparse_manifest_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, _ = _prepared_corpus(root)
        link = root / "manifest-link.json"
        try:
            link.symlink_to(manifest_path)
        except OSError:
            pytest.skip("symlink creation is unavailable")
        with pytest.raises(TrainingSamplesError, match="symlink|reparse"):
            load_training_manifest(link, root)


def test_duplicate_locator_and_duplicate_session_id_are_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, _ = _prepared_corpus(root)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["inputs"].append(dict(payload["inputs"][0]))
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(TrainingSamplesError, match="locator.*duplicated"):
            load_training_manifest(manifest_path, root)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _DatasetFixture()
        sessions = [
            fixture.make_session(root, subject, f"shared-{label}", [label] * 40)
            for subject in range(1, 4)
            for label in ("rest", "fist", "open_hand")
        ]
        manifest_path = root / "manifest.json"
        prepare_training_dataset(
            sessions, manifest_path, window_ms=200, step_ms=50, seed=7
        )
        with pytest.raises(TrainingSamplesError, match="duplicate session_id"):
            load_training_manifest(manifest_path, root)


def test_source_mutation_during_window_read_fails_closed(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, _ = _prepared_corpus(root)
        corpus = load_training_manifest(manifest_path, root)
        real_bounded_bytes = training_samples_module._bounded_bytes
        mutated = False

        def mutate_after_snapshot(path, limit):
            nonlocal mutated
            result = real_bounded_bytes(path, limit)
            if path.name == "samples.csv" and not mutated:
                mutated = True
                with path.open("ab") as stream:
                    stream.write(b"\n")
            return result

        monkeypatch.setattr(training_samples_module, "_bounded_bytes", mutate_after_snapshot)
        with pytest.raises(
            TrainingSamplesError, match="changed while streaming|resource limit"
        ):
            list(iter_split_windows(corpus, "train"))


def test_csv_snapshot_working_set_boundary_is_exact(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, _ = _prepared_corpus(root)
        corpus = load_training_manifest(manifest_path, root)
        windows = corpus.manifest["splits"]["train"]["windows"]
        locators = {
            f"{window['subject_id']}/{window['session_id']}" for window in windows
        }
        window_bytes = (
            corpus.window_contract["window_size_samples"]
            * corpus.data_contract["channel_count"]
            * np.dtype(np.float64).itemsize
        )
        exact_peak = max(
            corpus.sessions[locator].samples_size_bytes
            + corpus.sessions[locator].row_count
            * corpus.sessions[locator].channel_count
            * np.dtype(np.float64).itemsize
            + window_bytes
            for locator in locators
        )
        monkeypatch.setattr(
            training_samples_module, "MAX_TRAINING_WORKING_SET_BYTES", exact_peak
        )
        for example in iter_split_windows(corpus, "train"):
            assert example.samples.shape == (40, 8)
        monkeypatch.setattr(
            training_samples_module, "MAX_TRAINING_WORKING_SET_BYTES", exact_peak - 1
        )
        with pytest.raises(TrainingSamplesError, match="working-set budget"):
            for _ in iter_split_windows(corpus, "train"):
                pass


def test_session_loading_checks_finiteness_per_value_without_matrix_bool_mask(
    monkeypatch,
):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, _ = _prepared_corpus(root)
        corpus = load_training_manifest(manifest_path, root)
        window = corpus.manifest["splits"]["train"]["windows"][0]
        session = corpus.sessions[f"{window['subject_id']}/{window['session_id']}"]

        def forbid_matrix_isfinite(*args, **kwargs):
            raise AssertionError("matrix-sized isfinite mask must not be allocated")

        monkeypatch.setattr(training_samples_module.np, "isfinite", forbid_matrix_isfinite)
        rows = training_samples_module._load_session_values(
            session,
            reserved_bytes=0,
            window_bytes=(
                corpus.window_contract["window_size_samples"]
                * session.channel_count
                * np.dtype(np.float64).itemsize
            ),
        )
        assert rows.shape == (session.row_count, session.channel_count)


def test_session_switch_releases_previous_near_budget_matrix(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, _ = _prepared_corpus(root)
        corpus = load_training_manifest(manifest_path, root)
        windows = corpus.manifest["splits"]["train"]["windows"]
        locators = {
            f"{window['subject_id']}/{window['session_id']}" for window in windows
        }
        window_bytes = (
            corpus.window_contract["window_size_samples"]
            * corpus.data_contract["channel_count"]
            * np.dtype(np.float64).itemsize
        )
        exact_peak = max(
            corpus.sessions[locator].samples_size_bytes
            + corpus.sessions[locator].row_count
            * corpus.sessions[locator].channel_count
            * np.dtype(np.float64).itemsize
            + window_bytes
            for locator in locators
        )
        monkeypatch.setattr(
            training_samples_module, "MAX_TRAINING_WORKING_SET_BYTES", exact_peak
        )
        real_load = training_samples_module._load_session_values
        previous: weakref.ReferenceType[np.ndarray] | None = None
        releases_checked = 0

        def tracked_load(session, *, reserved_bytes, window_bytes):
            nonlocal previous, releases_checked
            if previous is not None:
                assert previous() is None
                releases_checked += 1
            result = real_load(
                session,
                reserved_bytes=reserved_bytes,
                window_bytes=window_bytes,
            )
            previous = weakref.ref(result)
            return result

        monkeypatch.setattr(
            training_samples_module, "_load_session_values", tracked_load
        )
        observed_sessions = set()
        for example in iter_split_windows(corpus, "train"):
            observed_sessions.add(example.session_id)
        assert len(observed_sessions) >= 2
        assert releases_checked == len(observed_sessions) - 1


def test_public_iterator_caller_keeps_previous_window_alive_during_next():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _DatasetFixture()
        sessions = [
            fixture.make_session(
                root, number, f"session-{number}-{label}", [label] * 80
            )
            for number in range(1, 4)
            for label in ("rest", "fist", "open_hand")
        ]
        manifest_path = root / "manifest.json"
        prepare_training_dataset(
            sessions, manifest_path, window_ms=200, step_ms=50, seed=7
        )
        corpus = load_training_manifest(manifest_path, root)
        iterator = iter_split_windows(corpus, "train")
        previous = next(iterator)
        previous_samples = weakref.ref(previous.samples)
        current = next(iterator)
        assert previous_samples() is previous.samples
        assert current.samples is not previous.samples
        del previous
        gc.collect()
        assert previous_samples() is None


def test_double_window_public_peak_rejects_budget_that_single_window_would_fit(
    monkeypatch,
):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _DatasetFixture()
        sessions = [
            fixture.make_session(
                root,
                number,
                f"wide-{number}-{label}",
                [label] * 40,
                channels=8,
            )
            for number in range(1, 4)
            for label in ("rest", "fist", "open_hand")
        ]
        manifest_path = root / "manifest.json"
        prepare_training_dataset(
            sessions, manifest_path, window_ms=200, step_ms=50, seed=7
        )
        corpus = load_training_manifest(manifest_path, root)
        windows = corpus.manifest["splits"]["train"]["windows"]
        locators = {
            f"{window['subject_id']}/{window['session_id']}" for window in windows
        }
        window_bytes = 40 * 8 * np.dtype(np.float64).itemsize
        required_peak = max(
            training_samples_module._session_iteration_peak_bytes(
                corpus.sessions[locator],
                reserved_bytes=0,
                window_bytes=window_bytes,
            )
            for locator in locators
        )
        budget = required_peak - 1
        monkeypatch.setattr(
            training_samples_module, "MAX_TRAINING_WORKING_SET_BYTES", budget
        )
        with pytest.raises(TrainingSamplesError, match="split iteration.*budget"):
            next(iter_split_windows(corpus, "train"))


def test_post_admission_csv_growth_reads_only_admitted_size_plus_one(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, _ = _prepared_corpus(root)
        corpus = load_training_manifest(manifest_path, root)
        window = corpus.manifest["splits"]["train"]["windows"][0]
        locator = f"{window['subject_id']}/{window['session_id']}"
        session = corpus.sessions[locator]
        sample_path = session.session_dir / "samples.csv"
        with sample_path.open("ab") as stream:
            stream.write(b"x" * (1024 * 1024))

        real_open = Path.open
        read_requests: list[int] = []

        class TrackingReader:
            def __init__(self, stream):
                self._stream = stream

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                self._stream.close()

            def read(self, size=-1):
                read_requests.append(size)
                return self._stream.read(size)

        def tracking_open(path, mode="r", *args, **kwargs):
            stream = real_open(path, mode, *args, **kwargs)
            if Path(path) == sample_path and mode == "rb":
                return TrackingReader(stream)
            return stream

        monkeypatch.setattr(Path, "open", tracking_open)
        window_bytes = (
            corpus.window_contract["window_size_samples"]
            * session.channel_count
            * np.dtype(np.float64).itemsize
        )
        with pytest.raises(
            TrainingSamplesError,
            match=rf"exceeds the {session.samples_size_bytes}-byte resource limit",
        ):
            training_samples_module._load_session_values(
                session, reserved_bytes=0, window_bytes=window_bytes
            )
        assert read_requests == [session.samples_size_bytes + 1]


def test_metadata_mutation_during_admission_fails_closed(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, _ = _prepared_corpus(root)
        real_read = training_dataset_module._read_bounded_bytes
        mutated = False

        def mutate_after_metadata_read(path, limit):
            nonlocal mutated
            result = real_read(path, limit)
            if path.name == "metadata.json" and not mutated:
                mutated = True
                with path.open("ab") as stream:
                    stream.write(b" ")
            return result

        monkeypatch.setattr(
            training_dataset_module, "_read_bounded_bytes", mutate_after_metadata_read
        )
        with pytest.raises(TrainingSamplesError, match="metadata.json changed"):
            load_training_manifest(manifest_path, root)


def test_session_matrix_memory_budget_fails_before_allocation(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, _ = _prepared_corpus(root)
        corpus = load_training_manifest(manifest_path, root)
        monkeypatch.setattr(training_samples_module, "MAX_SESSION_MATRIX_BYTES", 1)
        with pytest.raises(TrainingSamplesError, match="memory budget"):
            list(iter_split_windows(corpus, "train"))


@pytest.mark.parametrize("mode", ["empty", "missing_class"])
def test_empty_or_missing_class_split_is_rejected(mode):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, _ = _prepared_corpus(root)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        windows = payload["splits"]["validation"]["windows"]
        payload["splits"]["validation"]["windows"] = (
            [] if mode == "empty" else [window for window in windows if window["label"] != "rest"]
        )
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(TrainingSamplesError, match="empty|coverage"):
            load_training_manifest(manifest_path, root)


def test_noncanonical_window_order_and_locator_reentry_are_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _DatasetFixture()
        sessions = [
            fixture.make_session(
                root, number, f"session-{number}-{label}", [label] * 80
            )
            for number in range(1, 4)
            for label in ("rest", "fist", "open_hand")
        ]
        manifest_path = root / "manifest.json"
        prepare_training_dataset(
            sessions, manifest_path, window_ms=200, step_ms=50, seed=7
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        windows = payload["splits"]["train"]["windows"]
        windows[0], windows[1] = windows[1], windows[0]
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(TrainingSamplesError, match="canonical.*order"):
            load_training_manifest(manifest_path, root)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _DatasetFixture()
        sessions = [
            fixture.make_session(
                root, number, f"session-{number}-{label}", [label] * 80
            )
            for number in range(1, 4)
            for label in ("rest", "fist", "open_hand")
        ]
        manifest_path = root / "manifest.json"
        prepare_training_dataset(
            sessions, manifest_path, window_ms=200, step_ms=50, seed=7
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        windows = payload["splits"]["train"]["windows"]
        first_locator = (windows[0]["subject_id"], windows[0]["session_id"])
        boundary = next(
            index
            for index, window in enumerate(windows)
            if (window["subject_id"], window["session_id"]) != first_locator
        )
        reentered = windows.pop(1)
        windows.insert(boundary + 1, reentered)
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(TrainingSamplesError, match="re-enter"):
            load_training_manifest(manifest_path, root)
