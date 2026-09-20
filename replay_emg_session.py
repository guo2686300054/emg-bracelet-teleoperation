"""Fail-closed offline replay for strictly validated legacy EMG transforms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from emg_model_bundle import EmgLdaBundle
from emg_protocol import AcquisitionMetadata, HandSide, NotificationPacketProtocol, RateDescriptor, SignalChain
from legacy_dataset import CHANNEL_COLUMNS, LegacyDatasetError, validate_legacy_transform
from realtime_emg_inference import DecisionPolicy, RealtimeDecisionFilter, RealtimeEmgClassifier
from shared_memory_v2 import (
    FLAG_DISCONNECTED, FLAG_REPLAY, FLAG_SYNTHETIC_TIME, SAMPLE_FLOAT32,
    SharedMemoryWriter,
)

REPLAY_PROVENANCE = "legacy_experimental_replay"
TIMESTAMP_SOURCE = "synthetic_assumed_rate_not_device_time"
REPLAY_CODE_VERSION = "2.0"
MIN_ASSUMED_RATE_HZ = 10.0
MAX_ASSUMED_RATE_HZ = 5_000.0


class ReplayInputError(ValueError):
    """The replay request violates a provenance or runtime contract."""


@dataclass(frozen=True)
class ReplayReport:
    status: str
    source_csv: str
    source_csv_sha256: str
    manifest_path: str
    manifest_sha256: str
    shared_file: str
    provenance: str
    timestamp_source: str
    replay_code_version: str
    assumed_sample_rate_hz: float
    source_rows: int
    frames_written: int
    predictions_emitted: int
    output_labels: dict[str, int]
    bundle_id: str | None
    decision_policy: dict[str, object] | None
    generation: int | None
    final_sequence: int | None
    connection_generation: int
    elapsed_seconds: float
    effective_write_rate_hz: float
    end_marker_written: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def training_usable(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["training_usable"] = False
        return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class _FrozenFile:
    path: Path
    size: int
    sha256: str


def _freeze_file(path: Path) -> _FrozenFile:
    resolved = path.expanduser().resolve()
    info = os.lstat(resolved)
    if not stat.S_ISREG(info.st_mode):
        raise ReplayInputError(f"validation closure member is not a regular file: {resolved}")
    digest = _sha256(resolved)
    after = os.lstat(resolved)
    if info.st_size != after.st_size or info.st_mtime_ns != after.st_mtime_ns:
        raise ReplayInputError(f"validation closure changed while hashing: {resolved}")
    return _FrozenFile(resolved, info.st_size, digest)


def _same_frozen(left: _FrozenFile, right: _FrozenFile) -> bool:
    return left.path == right.path and left.size == right.size and left.sha256 == right.sha256


def _verify_frozen_files(files: tuple[_FrozenFile, ...]) -> list[str]:
    errors: list[str] = []
    for frozen in files:
        try:
            current = _freeze_file(frozen.path)
        except Exception as exc:
            errors.append(
                f"source_changed_during_replay: {frozen.path}: {type(exc).__name__}: {exc}"
            )
        else:
            if not _same_frozen(frozen, current):
                errors.append(f"source_changed_during_replay: {frozen.path}")
    return errors


def _validated_rows(csv_path: Path):
    manifest_path = csv_path.with_name("transform_manifest.json")
    try:
        csv_before = _freeze_file(csv_path)
        manifest_before = _freeze_file(manifest_path)
    except (OSError, ReplayInputError) as exc:
        raise ReplayInputError(f"legacy transform validation failed: {exc}") from exc
    try:
        validated = validate_legacy_transform(manifest_path)
    except (LegacyDatasetError, OSError, ValueError) as exc:
        raise ReplayInputError(f"legacy transform validation failed: {exc}") from exc
    expected_csv = manifest_path.parent / str(validated.manifest["output"]["file"])
    if csv_path != expected_csv.resolve():
        raise ReplayInputError("manifest does not identify the requested derived CSV")
    try:
        csv_after = _freeze_file(csv_path)
        manifest_after = _freeze_file(manifest_path)
        source_frozen = _freeze_file(validated.source_path)
        annotation_frozen = _freeze_file(validated.annotation_path)
    except (OSError, ReplayInputError) as exc:
        raise ReplayInputError(f"legacy transform validation closure failed: {exc}") from exc
    if not _same_frozen(csv_before, csv_after) or not _same_frozen(
        manifest_before, manifest_after
    ):
        raise ReplayInputError("legacy transform changed during strict validation")
    if (
        source_frozen.sha256 != validated.source_sha256
        or source_frozen.size != validated.source_size_bytes
        or annotation_frozen.sha256 != validated.annotation_sha256
        or annotation_frozen.size != validated.annotation_size_bytes
    ):
        raise ReplayInputError("legacy transform validation closure identity mismatch")
    closure = (csv_after, manifest_after, source_frozen, annotation_frozen)
    return validated, manifest_path, closure


def _validate_rate(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not MIN_ASSUMED_RATE_HZ <= float(value) <= MAX_ASSUMED_RATE_HZ
    ):
        raise ReplayInputError(
            f"assumed_sample_rate_hz must be between {MIN_ASSUMED_RATE_HZ:g} "
            f"and {MAX_ASSUMED_RATE_HZ:g} Hz"
        )
    return float(value)


def _error_text(prefix: str, exc: BaseException) -> str:
    return f"{prefix}{type(exc).__name__}: {exc}"


def replay_session(
    source_csv: str | Path,
    shared_file: str | Path,
    *,
    assumed_sample_rate_hz: float,
    decision_filter: RealtimeDecisionFilter | None = None,
    realtime: bool = False,
    cancel_event: threading.Event | None = None,
    connection_generation: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
    writer_factory: Callable[..., SharedMemoryWriter] = SharedMemoryWriter,
) -> ReplayReport:
    """Replay legacy samples and always attempt a disconnected terminal frame."""

    rate_hz = _validate_rate(assumed_sample_rate_hz)
    if isinstance(connection_generation, bool) or not isinstance(connection_generation, int) or connection_generation <= 0:
        raise ReplayInputError("connection_generation must be a positive integer")
    if decision_filter is not None and not isinstance(decision_filter, RealtimeDecisionFilter):
        raise TypeError("decision_filter must be RealtimeDecisionFilter or None")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")

    csv_path = Path(source_csv).expanduser().resolve()
    output_path = Path(shared_file).expanduser().resolve()
    if csv_path == output_path:
        raise ReplayInputError("shared replay file must not overwrite source CSV")
    validated, manifest_path, frozen_files = _validated_rows(csv_path)
    csv_frozen, manifest_frozen = frozen_files[:2]
    rows = validated.rows
    if not rows:
        raise ReplayInputError("derived CSV contains no samples")
    if decision_filter is not None:
        classifier = decision_filter.classifier
        if classifier.channel_count != len(CHANNEL_COLUMNS):
            raise ReplayInputError("classifier channel count does not match derived CSV")
        if classifier.acquisition.sample_rate.value_hz != rate_hz:
            raise ReplayInputError("assumed sample rate must exactly match classifier acquisition rate")
        decision_filter.reset()

    period_ns = round(1_000_000_000 / rate_hz)
    cancelled = cancel_event or threading.Event()
    labels: Counter[str] = Counter()
    frames_written = predictions = 0
    generation = final_sequence = None
    end_marker = False
    errors: list[str] = []
    start = time.perf_counter()
    writer: SharedMemoryWriter | None = None
    last_payload = bytes(4 * len(CHANNEL_COLUMNS))
    status = "completed"
    replay_flags = FLAG_REPLAY | FLAG_SYNTHETIC_TIME
    try:
        writer = writer_factory(output_path)
        generation = writer.generation
        for index, row in enumerate(rows):
            if cancelled.is_set():
                status = "cancelled"
                break
            if realtime and index:
                delay = start + index / rate_hz - time.perf_counter()
                if delay > 0 and cancelled.wait(delay):
                    status = "cancelled"
                    break
            sample = tuple(float(row[column]) for column in CHANNEL_COLUMNS)
            last_payload = struct.pack("<8f", *sample)
            frame = writer.write_frame(
                last_payload,
                host_receive_index=index,
                connection_generation=connection_generation,
                flags=replay_flags,
                channel_count=8,
                sample_count=1,
                sample_format=SAMPLE_FLOAT32,
            )
            frames_written += 1
            final_sequence = frame.sequence
            if decision_filter is not None:
                result = decision_filter.push(sample, (index + 1) * period_ns)
                if result is not None:
                    predictions += 1
                    labels[result.output_label] += 1
            if progress_callback is not None:
                progress_callback(frames_written, len(rows))
    except Exception as exc:
        status = "failed"
        errors.append(_error_text("", exc))
    finally:
        if writer is not None:
            try:
                frame = writer.write_frame(
                    last_payload,
                    host_receive_index=frames_written,
                    connection_generation=connection_generation,
                    flags=replay_flags | FLAG_DISCONNECTED,
                    channel_count=8,
                    sample_count=1,
                    sample_format=SAMPLE_FLOAT32,
                )
                final_sequence = frame.sequence
                end_marker = True
            except Exception as exc:
                status = "failed"
                errors.append(_error_text("terminal frame failed: ", exc))
            try:
                writer.close()
            except Exception as exc:
                status = "failed"
                errors.append(_error_text("writer cleanup failed: ", exc))
        if decision_filter is not None:
            try:
                decision_filter.disconnect()
            except Exception as exc:
                status = "failed"
                errors.append(_error_text("decision cleanup failed: ", exc))

    closure_errors = _verify_frozen_files(frozen_files)
    if closure_errors:
        status = "failed"
        errors.extend(closure_errors)

    elapsed = max(0.0, time.perf_counter() - start)
    policy = None if decision_filter is None else asdict(decision_filter.policy)
    return ReplayReport(
        status=status,
        source_csv=str(csv_path), source_csv_sha256=csv_frozen.sha256,
        manifest_path=str(manifest_path), manifest_sha256=manifest_frozen.sha256,
        shared_file=str(output_path), provenance=REPLAY_PROVENANCE,
        timestamp_source=TIMESTAMP_SOURCE, replay_code_version=REPLAY_CODE_VERSION,
        assumed_sample_rate_hz=rate_hz, source_rows=len(rows),
        frames_written=frames_written, predictions_emitted=predictions,
        output_labels=dict(sorted(labels.items())),
        bundle_id=None if decision_filter is None else decision_filter.classifier.bundle.bundle_id,
        decision_policy=policy, generation=generation, final_sequence=final_sequence,
        connection_generation=connection_generation, elapsed_seconds=elapsed,
        effective_write_rate_hz=frames_written / elapsed if elapsed > 0 else 0.0,
        end_marker_written=end_marker,
        warnings=(
            "synthetic timeline uses an assumed rate; it is not device timestamp evidence",
            "legacy replay is never eligible for supervised training or live control",
            "decision policy requires canonical-data calibration",
        ),
        errors=tuple(errors),
    )


def decision_filter_from_bundle(
    bundle_path: str | Path,
    *,
    hand_side: HandSide,
    policy: DecisionPolicy | None = None,
) -> RealtimeDecisionFilter:
    bundle = EmgLdaBundle.load(bundle_path)
    acquisition = AcquisitionMetadata(
        sample_rate=RateDescriptor.from_mapping(bundle.sample_rate),
        signal_chain=SignalChain.from_mapping(bundle.signal_chain),
        notification_packet_protocol=NotificationPacketProtocol.from_mapping(bundle.notification_protocol),
    )
    classifier = RealtimeEmgClassifier(bundle, acquisition, hand_side, evaluation_mode=True)
    return RealtimeDecisionFilter(classifier, policy)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("shared_file", type=Path)
    parser.add_argument("--assumed-rate", type=float, required=True)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--hand-side", choices=("left", "right"), default="left")
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--low-confidence-policy", choices=("unknown", "hold"), default="unknown")
    parser.add_argument("--smoothing-mode", choices=("consecutive", "majority"), default="consecutive")
    parser.add_argument("--smoothing-windows", type=int, default=3)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        decision_filter = None
        if args.bundle is not None:
            policy = DecisionPolicy(
                confidence_threshold=args.confidence_threshold,
                low_confidence_policy=args.low_confidence_policy,
                smoothing_mode=args.smoothing_mode,
                smoothing_windows=args.smoothing_windows,
            )
            decision_filter = decision_filter_from_bundle(
                args.bundle, hand_side=HandSide(args.hand_side), policy=policy
            )
        report = replay_session(
            args.source_csv, args.shared_file,
            assumed_sample_rate_hz=args.assumed_rate,
            decision_filter=decision_filter, realtime=args.realtime,
        )
        payload = report.to_dict()
    except Exception as exc:
        payload = {
            "status": "failed", "training_usable": False,
            "provenance": REPLAY_PROVENANCE, "timestamp_source": TIMESTAMP_SOURCE,
            "replay_code_version": REPLAY_CODE_VERSION,
            "errors": [_error_text("", exc)],
        }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["status"] in {"completed", "cancelled"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
