"""CLI for strict, non-training conversion of legacy EMG measurement batches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from legacy_dataset import (
    LEGACY_TRAINING_PROVENANCE,
    LegacyDatasetError,
    import_legacy_measurement_manifest,
    load_experimental_analysis_groups,
)


def convert_measurement_manifest(
    measurement_manifest: str | Path, output_root: str | Path
) -> Mapping[str, Any]:
    """Run the strict converter and immediately validate the published batch."""
    destination = Path(output_root).expanduser().resolve()
    results = import_legacy_measurement_manifest(measurement_manifest, destination)
    dataset_manifest = destination / "dataset_manifest.json"
    groups = load_experimental_analysis_groups(dataset_manifest)
    payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    if payload.get("training_usable") is not False:
        raise LegacyDatasetError("converted legacy batch must remain training_usable=false")
    if payload.get("training_provenance") != LEGACY_TRAINING_PROVENANCE:
        raise LegacyDatasetError("converted legacy batch must declare legacy_experimental provenance")
    return {
        "output_root": str(destination),
        "dataset_manifest": str(dataset_manifest),
        "session_count": len(results),
        "input_rows": sum(item.input_rows for item in results),
        "output_rows": sum(item.output_rows for item in results),
        "validated_analysis_group_count": len(groups),
        "training_usable": False,
        "training_provenance": dict(LEGACY_TRAINING_PROVENANCE),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a legacy measurement manifest for diagnostic analysis only."
    )
    parser.add_argument("measurement_manifest")
    parser.add_argument("--output", required=True, help="new output directory; never overwritten")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = convert_measurement_manifest(args.measurement_manifest, args.output)
    except (LegacyDatasetError, OSError) as exc:
        print(f"legacy conversion error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
