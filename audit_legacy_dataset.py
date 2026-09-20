"""Read-only quality audit for annotated legacy EMG measurement batches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import legacy_dataset

AUDIT_SCHEMA = "emg_legacy_dataset_audit"
AUDIT_VERSION = "1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_summary(run_lengths: Sequence[int], factor: int) -> Mapping[str, Any]:
    repeated = [length for length in run_lengths if length > 1]
    return {
        "total_runs": len(run_lengths),
        "repeated_runs": len(repeated),
        "adjacent_equal_pairs": sum(length - 1 for length in repeated),
        "rows_in_repeated_runs": sum(repeated),
        "max_run_length": max(run_lengths, default=0),
        "run_length_histogram": {
            str(length): count for length, count in sorted(Counter(run_lengths).items())
        },
        "declared_legacy_write_factor": factor,
        "repeated_runs_aligned_to_declared_factor": sum(
            1 for length in repeated if length % factor == 0
        ),
        "interpretation": (
            "The declared factor documents a known legacy host write pattern and is used only "
            "for experimental run recovery. Equal adjacent rows may also be legitimate signal "
            "values, so these counts do not prove device packet duplication."
        ),
    }


def _audit_csv(source: Path, factor: int) -> Mapping[str, Any]:
    row_count = 0
    width_counts: Counter[int] = Counter()
    missing_values = 0
    non_numeric_values = 0
    out_of_range_values = 0
    saturation_low = [0] * len(legacy_dataset.CHANNEL_COLUMNS)
    saturation_high = [0] * len(legacy_dataset.CHANNEL_COLUMNS)
    minimums: list[int | None] = [None] * len(legacy_dataset.CHANNEL_COLUMNS)
    maximums: list[int | None] = [None] * len(legacy_dataset.CHANNEL_COLUMNS)
    run_lengths: list[int] = []
    previous: tuple[str, ...] | None = None
    current_run = 0

    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = tuple(next(reader, ()))
        for row in reader:
            row_count += 1
            width_counts[len(row)] += 1
            normalized = tuple(value.strip() for value in row)
            if previous is None or normalized == previous:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
            previous = normalized
            for index in range(len(legacy_dataset.CHANNEL_COLUMNS)):
                if index >= len(row) or not row[index].strip():
                    missing_values += 1
                    continue
                try:
                    number = int(row[index].strip())
                except ValueError:
                    non_numeric_values += 1
                    continue
                if not 0 <= number <= 255:
                    out_of_range_values += 1
                    continue
                current_minimum = minimums[index]
                current_maximum = maximums[index]
                minimums[index] = number if current_minimum is None else min(current_minimum, number)
                maximums[index] = number if current_maximum is None else max(current_maximum, number)
                saturation_low[index] += number == 0
                saturation_high[index] += number == 255
            if len(row) > len(legacy_dataset.CHANNEL_COLUMNS):
                non_numeric_values += len(row) - len(legacy_dataset.CHANNEL_COLUMNS)
    if current_run:
        run_lengths.append(current_run)

    channel_stats = {}
    for index, name in enumerate(legacy_dataset.CHANNEL_COLUMNS):
        channel_stats[name] = {
            "minimum": minimums[index],
            "maximum": maximums[index],
            "saturation_low_0_count": saturation_low[index],
            "saturation_high_255_count": saturation_high[index],
        }
    return {
        "header": list(header),
        "header_valid": header == legacy_dataset.CHANNEL_COLUMNS,
        "declared_channel_count": len(legacy_dataset.CHANNEL_COLUMNS),
        "observed_row_width_counts": {
            str(width): count for width, count in sorted(width_counts.items())
        },
        "observed_row_count": row_count,
        "missing_values": missing_values,
        "non_numeric_values": non_numeric_values,
        "out_of_uint8_range_values": out_of_range_values,
        "channel_statistics": channel_stats,
        "adjacent_identical_row_runs": _run_summary(run_lengths, factor),
    }


def audit_measurement_manifest(manifest_path: str | Path) -> Mapping[str, Any]:
    """Validate batch provenance, then audit its source CSVs without modifying them."""
    manifest = Path(manifest_path).expanduser().resolve()
    annotations = legacy_dataset._validated_measurement_annotations(manifest)
    source_manifest = legacy_dataset._load_json(manifest)
    source_entries = {
        str(entry["session_id"]): entry for entry in source_manifest["sessions"]
    }
    sessions = []
    total_rows = 0
    label_rows: Counter[str] = Counter()
    excluded_sessions = 0
    experimental_candidates: list[str] = []

    for annotation_path in annotations:
        annotation = legacy_dataset._load_json(annotation_path)
        session_id = str(annotation["session_id"])
        entry = source_entries[session_id]
        source = annotation_path.parent / "client_data.csv"
        factor = int(annotation["legacy_write_factor"])
        csv_audit = _audit_csv(source, factor)
        total_rows += int(csv_audit["observed_row_count"])
        session_labels: Counter[str] = Counter()
        for segment in annotation["segments"]:
            count = int(segment["end_row_exclusive"]) - int(segment["start_row"])
            session_labels[str(segment["action_label"])] += count
            label_rows[str(segment["action_label"])] += count

        exclusion_reasons = [
            "legacy_experimental provenance is not canonical_session",
            "device packet identity is unavailable; device duplicates cannot be determined",
            "sampling rate and device timestamps are unconfirmed",
        ]
        status = str(annotation["status"])
        width_counts = csv_audit["observed_row_width_counts"]
        structural_quality_valid = (
            csv_audit["header_valid"]
            and width_counts == {str(len(legacy_dataset.CHANNEL_COLUMNS)): csv_audit["observed_row_count"]}
            and csv_audit["observed_row_count"] == annotation["source_row_count"]
            and not any(int(csv_audit[field]) for field in (
                "missing_values", "non_numeric_values", "out_of_uint8_range_values"
            ))
        )
        experimental_candidate = status == "completed" and structural_quality_valid
        if experimental_candidate:
            experimental_candidates.append(session_id)
        if status != "completed":
            excluded_sessions += 1
            exclusion_reasons.append(f"session status is {status}")
        if entry.get("excluded_reason"):
            exclusion_reasons.append(str(entry["excluded_reason"]))
        if not csv_audit["header_valid"]:
            exclusion_reasons.append("CSV channel header is invalid")
        if csv_audit["observed_row_count"] != annotation["source_row_count"]:
            exclusion_reasons.append("observed row count disagrees with annotation")
        if width_counts != {str(len(legacy_dataset.CHANNEL_COLUMNS)): csv_audit["observed_row_count"]}:
            exclusion_reasons.append("one or more CSV rows do not contain exactly 8 channels")
        if any(
            int(csv_audit[field])
            for field in ("missing_values", "non_numeric_values", "out_of_uint8_range_values")
        ):
            exclusion_reasons.append("CSV contains missing, non-numeric, or out-of-range values")

        sessions.append({
            "session_id": session_id,
            "status": status,
            "reported_action": entry.get("reported_action"),
            "source_csv": str(source),
            "source_sha256": _sha256(source),
            "declared_row_count": annotation["source_row_count"],
            "raw_host_write_label_row_distribution": dict(sorted(session_labels.items())),
            "training_usable": False,
            "experimental_analysis_candidate": experimental_candidate,
            "exclusion_reasons": exclusion_reasons,
            "csv_audit": csv_audit,
        })

    return {
        "schema": AUDIT_SCHEMA,
        "version": AUDIT_VERSION,
        "source_measurement_manifest": {
            "path": str(manifest),
            "sha256": _sha256(manifest),
        },
        "training_provenance": dict(legacy_dataset.LEGACY_TRAINING_PROVENANCE),
        "training_usable": False,
        "device_packet_duplicate_detection": {
            "status": "indeterminate",
            "reason": (
                "No device packet sequence or packet identity was recorded. The known "
                "legacy_write_factor describes host-side repeated writes only."
            ),
        },
        "summary": {
            "session_count": len(sessions),
            "excluded_or_non_target_session_count": excluded_sessions,
            "observed_row_count": total_rows,
            "raw_host_write_label_row_distribution": dict(sorted(label_rows.items())),
            "canonical_training_candidate_session_ids": [],
            "experimental_analysis_candidate_session_ids": experimental_candidates,
        },
        "sessions": sessions,
    }


def write_audit_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only audit of a legacy EMG measurement manifest.")
    parser.add_argument("measurement_manifest")
    parser.add_argument("--output", required=True, help="new JSON report path; existing files are never overwritten")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_measurement_manifest(args.measurement_manifest)
        destination = write_audit_report(report, args.output)
    except (legacy_dataset.LegacyDatasetError, OSError, UnicodeError, csv.Error) as exc:
        print(f"legacy audit error: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote legacy audit: {destination} "
        f"({report['summary']['session_count']} sessions, "
        f"{report['summary']['observed_row_count']} rows; training_usable=false)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
