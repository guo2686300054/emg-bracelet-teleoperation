import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from convert_legacy_dataset import convert_measurement_manifest
from legacy_dataset import CHANNEL_COLUMNS


def _batch(root: Path) -> Path:
    entries = []
    for factor in range(1, 8):
        session = root / f"source-{factor}"
        session.mkdir()
        source = session / "client_data.csv"
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(CHANNEL_COLUMNS)
            writer.writerows([[factor] * 8] * factor)
        session_id = f"session-s{factor}"
        annotation = {
            "schema": "emg_session_annotations", "version": "1.1",
            "source_file": "client_data.csv", "source_directory_name": session.name,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_size_bytes": source.stat().st_size, "source_row_count": factor,
            "session_id": session_id, "status": "completed", "training_usable_raw": False,
            "split_group_id": "subject-1", "full_session_group_id": f"subject-1:{session_id}",
            "legacy_write_factor": factor,
            "segments": [{"start_row": 0, "end_row_exclusive": factor,
                          "action_label": "rest", "action_phase": "hold", "confidence": "reported",
                          "training_usable_raw": False, "eligible_for_future_training_review": True,
                          "basis": "test"}],
        }
        (session / "annotations.json").write_text(json.dumps(annotation), encoding="utf-8")
        entries.append({"relative_directory": session.name, "annotation_file": "annotations.json",
                        "session_id": session_id, "status": "completed", "training_usable_raw": False,
                        "source_sha256": annotation["source_sha256"],
                        "source_size_bytes": annotation["source_size_bytes"],
                        "source_row_count": factor, "legacy_write_factor": factor,
                        "full_session_group_id": f"subject-1:{session_id}"})
    manifest = root / "measurement.json"
    manifest.write_text(json.dumps({"schema": "emg_measurement_manifest", "version": "1.1",
                                    "session_count": 7, "split_group_id": "subject-1",
                                    "raw_training_policy": {"training_usable_raw": False},
                                    "sessions": entries}), encoding="utf-8")
    return manifest


class ConvertLegacyDatasetTests(unittest.TestCase):
    def test_batch_conversion_is_validated_and_explicitly_non_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "converted"
            summary = convert_measurement_manifest(_batch(root), output)
            dataset = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
            child = json.loads((output / "session-s1" / "transform_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["session_count"], 7)
            self.assertFalse(summary["training_usable"])
            self.assertEqual(dataset["training_provenance"]["kind"], "legacy_experimental")
            self.assertEqual(dataset["version"], "1.1")
            self.assertFalse(dataset["training_usable"])
            self.assertEqual(child["training_provenance"]["kind"], "legacy_experimental")
            self.assertEqual(child["version"], "1.2")
            self.assertFalse(child["training_usable"])

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _batch(root)
            output = root / "converted"
            convert_measurement_manifest(manifest, output)
            with self.assertRaises(FileExistsError):
                convert_measurement_manifest(manifest, output)


if __name__ == "__main__":
    unittest.main()
