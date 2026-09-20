import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from audit_legacy_dataset import audit_measurement_manifest, main, write_audit_report
from legacy_dataset import CHANNEL_COLUMNS


def _measurement(root: Path, rows):
    session = root / "source-1"
    session.mkdir()
    source = session / "client_data.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(CHANNEL_COLUMNS)
        writer.writerows(rows)
    annotation = {
        "schema": "emg_session_annotations", "version": "1.1",
        "source_file": "client_data.csv", "source_directory_name": session.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_size_bytes": source.stat().st_size, "source_row_count": len(rows),
        "session_id": "session-s1", "status": "completed", "training_usable_raw": False,
        "split_group_id": "subject-1", "full_session_group_id": "subject-1:session-s1",
        "legacy_write_factor": 2,
        "segments": [{"start_row": 0, "end_row_exclusive": len(rows),
                      "action_label": "rest", "action_phase": "hold", "confidence": "reported",
                      "training_usable_raw": False, "eligible_for_future_training_review": True,
                      "basis": "test"}],
    }
    (session / "annotations.json").write_text(json.dumps(annotation), encoding="utf-8")
    manifest = {
        "schema": "emg_measurement_manifest", "version": "1.1", "session_count": 1,
        "split_group_id": "subject-1", "raw_training_policy": {"training_usable_raw": False},
        "sessions": [{"relative_directory": session.name, "annotation_file": "annotations.json",
                      "session_id": "session-s1", "status": "completed",
                      "reported_action": "rest", "training_usable_raw": False,
                      "source_sha256": annotation["source_sha256"],
                      "source_size_bytes": annotation["source_size_bytes"],
                      "source_row_count": len(rows), "legacy_write_factor": 2,
                      "full_session_group_id": "subject-1:session-s1"}],
    }
    path = root / "measurement.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class AuditLegacyDatasetTests(unittest.TestCase):
    def test_reports_quality_distribution_and_packet_uncertainty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [[0, 1, 2, 3, 4, 5, 6, 255], [0, 1, 2, 3, 4, 5, 6, 255],
                    ["", "bad", 256, 3, 4, 5, 6, 7]]
            report = audit_measurement_manifest(_measurement(root, rows))
            session = report["sessions"][0]
            self.assertFalse(report["training_usable"])
            self.assertEqual(report["training_provenance"]["kind"], "legacy_experimental")
            self.assertEqual(report["device_packet_duplicate_detection"]["status"], "indeterminate")
            self.assertEqual(report["summary"]["raw_host_write_label_row_distribution"], {"rest": 3})
            self.assertEqual(report["summary"]["canonical_training_candidate_session_ids"], [])
            self.assertEqual(report["summary"]["experimental_analysis_candidate_session_ids"], [])
            self.assertFalse(session["experimental_analysis_candidate"])
            self.assertEqual(session["csv_audit"]["missing_values"], 1)
            self.assertEqual(session["csv_audit"]["non_numeric_values"], 1)
            self.assertEqual(session["csv_audit"]["out_of_uint8_range_values"], 1)
            runs = session["csv_audit"]["adjacent_identical_row_runs"]
            self.assertEqual(runs["adjacent_equal_pairs"], 1)
            self.assertEqual(runs["declared_legacy_write_factor"], 2)
            self.assertIn("do not prove device packet duplication", runs["interpretation"])

    def test_report_is_json_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _measurement(root, [[1] * 8])
            output = root / "audit.json"
            self.assertEqual(main([str(manifest), "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["schema"],
                             "emg_legacy_dataset_audit")
            with self.assertRaises(FileExistsError):
                write_audit_report(audit_measurement_manifest(manifest), output)


if __name__ == "__main__":
    unittest.main()
