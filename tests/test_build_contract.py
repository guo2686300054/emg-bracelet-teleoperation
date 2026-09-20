import configparser
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_NAME = "ble_protocol_evidence_summary.json"
BUILD_INFO_NAME = "BUILD_INFO.json"


def validate_protocol_contract(directory: Path) -> dict:
    config_path = directory / "config.ini"
    summary_path = directory / SUMMARY_NAME
    if not config_path.is_file() or not summary_path.is_file():
        raise ValueError("deployment must contain config.ini and protocol evidence summary")
    config = configparser.ConfigParser(interpolation=None)
    config.read(config_path, encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = config["Protocol"]
    expected_ref = f'{summary["evidence_id"]}:{summary["source_sha256"]}'
    checks = {
        "evidence_ref": protocol["evidence_ref"] == expected_ref,
        "wire_packet_size": int(protocol["wire_packet_size"])
        == summary["wire_packet_size"],
        "logical_packet_size": int(protocol["logical_packet_size"])
        == summary["logical_packet_size"],
        "padding_rule": protocol["padding_rule"] == "zero_suffix",
        "zero_suffix_bytes": summary["zero_suffix_bytes"]
        == summary["wire_packet_size"] - summary["logical_packet_size"]
        == 12,
        "zero_suffix_matches": summary["zero_suffix_match_count"]
        == summary["notification_count"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("protocol deployment contract mismatch: " + ", ".join(failed))
    return summary


def _assigned_integer(source_path: Path, name: str) -> int:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                value = ast.literal_eval(node.value)
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
    raise ValueError(f"missing integer assignment {name} in {source_path}")


def validate_build_info(deployment: Path) -> dict:
    source_info = json.loads((ROOT / BUILD_INFO_NAME).read_text(encoding="utf-8"))
    deployed_info_path = deployment / BUILD_INFO_NAME
    deployed_info = json.loads(deployed_info_path.read_text(encoding="utf-8"))
    if deployed_info != source_info:
        raise ValueError("source and deployed BUILD_INFO.json differ")

    artifact = deployment / deployed_info["artifact"]
    artifact_bytes = artifact.read_bytes()
    config = configparser.ConfigParser(interpolation=None)
    config.read(deployment / "config.ini", encoding="utf-8")
    shared_source = deployment / "shared_memory_v2.py"
    shared_version = (
        f'{_assigned_integer(shared_source, "VERSION_MAJOR")}.'
        f'{_assigned_integer(shared_source, "VERSION_MINOR")}'
    )
    protocol = config["Protocol"]
    build_protocol = deployed_info["ble_notification_protocol"]
    review = deployed_info["review"]
    hardware = deployed_info["hardware_validation"]
    hardware_json = json.dumps(hardware, ensure_ascii=False).casefold()
    checks = {
        "release_status": deployed_info["release_status"] == "release_approved",
        "code_review": review["code_review"] == "APPROVE",
        "architecture_review": review["architecture"] == "CLEAR",
        "artifact_name": deployed_info["artifact"] == "qt5_bleak_v2.exe",
        "artifact_size": len(artifact_bytes) == deployed_info["artifact_size_bytes"],
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest()
        == deployed_info["artifact_sha256"],
        "app_config_version": config["App"]["config_version"]
        == deployed_info["app_config_version"],
        "data_schema_version": deployed_info["data_schema_version"] == "1.7",
        "shared_memory_protocol_version": shared_version
        == deployed_info["shared_memory_protocol_version"],
        "protocol_mode": protocol["mode"] == build_protocol["mode"],
        "protocol_wire": int(protocol["wire_packet_size"])
        == build_protocol["wire_packet_size"],
        "protocol_logical": int(protocol["logical_packet_size"])
        == build_protocol["logical_packet_size"],
        "protocol_padding": protocol["padding_rule"]
        == build_protocol["padding_rule"],
        "protocol_evidence": protocol["evidence_ref"]
        == build_protocol["evidence_ref"],
        "stop_generation_feature": "stop_generation_aba_guard"
        in deployed_info["included_features"],
        "recorder_watermark_feature": "recorder_watermarks_start_exclusive_end_inclusive"
        in deployed_info["included_features"],
        "session_sample_index_feature": "session_sample_index_zero_based_contiguous"
        in deployed_info["included_features"],
        "monotonic_nondecreasing_feature": "host_monotonic_nondecreasing"
        in deployed_info["included_features"],
        "monotonic_tie_count_feature": "host_monotonic_tie_count"
        in deployed_info["included_features"],
        "schema_1_6_migration_feature": "schema_1_6_migration"
        in deployed_info["included_features"],
        "stream_expected_watchdog_feature": "stream_expected_watchdog"
        in deployed_info["included_features"],
        "watchdog_generation_feature": "watchdog_generation_aba_guard"
        in deployed_info["included_features"],
        "watchdog_baseline_feature": "watchdog_arm_baseline_monotonic"
        in deployed_info["included_features"],
        "hardware_rows": hardware["csv_rows"] == 3478,
        "hardware_tie_count": hardware["host_monotonic_tie_count"] == 412,
        "hardware_tie_match": hardware["tie_count_matches_metadata"] is True,
        "hardware_drop_free": hardware["queue_drop_session"] == 0
        and hardware["tail_pending_count"] == 0
        and hardware["tail_loss_count"] == 0,
        "hardware_session_complete": hardware["session_complete"] is True,
        "hardware_shutdown_complete": hardware["shutdown_complete"] is True,
        "hardware_privacy": "subject_id" not in hardware_json
        and "device_id" not in hardware_json
        and "wxid" not in hardware_json,
        "pytest_result": deployed_info["verification"]["pytest"]
        == "409 passed, 3 skipped, 177 subtests passed",
        "deployment_result": deployed_info["verification"]["deployment_contract"]
        == "4 passed, 2 subtests passed",
        "smoke_result": deployed_info["verification"]["windows_gui_smoke"]
        == "exit_code=0, no residual process, temporary deployment cleaned",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("BUILD_INFO contract mismatch: " + ", ".join(failed))
    return deployed_info


class BuildContractTests(unittest.TestCase):
    def test_protocol_config_matches_evidence_summary(self):
        validate_protocol_contract(ROOT)

    def test_spec_bundles_protocol_evidence(self):
        spec = (ROOT / "qt5_bleak.spec").read_text(encoding="utf-8")
        self.assertIn("('ble_protocol_evidence_summary.json', '.')", spec)

    def test_requested_deployment_contains_parseable_matching_evidence(self):
        raw_deploy_dir = os.environ.get("EMG_DEPLOY_DIR")
        if not raw_deploy_dir:
            self.skipTest("set EMG_DEPLOY_DIR for post-build deployment verification")
        deployed_dir = Path(raw_deploy_dir).resolve()
        required_files = (
            "qt5_bleak_v2.exe",
            "config.ini",
            SUMMARY_NAME,
            "test.py",
            "shared_memory_v2.py",
            BUILD_INFO_NAME,
        )
        missing = [name for name in required_files if not (deployed_dir / name).is_file()]
        self.assertFalse(missing, f"deployment missing files: {missing}")
        deployed_summary = validate_protocol_contract(deployed_dir)
        self.assertEqual(
            deployed_summary,
            json.loads((ROOT / SUMMARY_NAME).read_text(encoding="utf-8")),
        )
        forbidden_runtime_paths = (
            ".emg_identity",
            "logs",
            "logs/app.log",
            "emg_shared_data_v2.bin",
            "raw_packets.jsonl",
        )
        residual = [
            name for name in forbidden_runtime_paths if (deployed_dir / name).exists()
        ]
        self.assertFalse(residual, f"release contains runtime residue: {residual}")
        validate_build_info(deployed_dir)

    def test_old_or_tampered_deployment_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw_temp:
            deployment = Path(raw_temp)
            shutil.copyfile(ROOT / SUMMARY_NAME, deployment / SUMMARY_NAME)
            source_config = (ROOT / "config.ini").read_text(encoding="utf-8")
            cases = {
                "old_logical16": source_config.replace(
                    "wire_packet_size = 28", "wire_packet_size = 16"
                ),
                "tampered_evidence": source_config.replace(
                    "evidence_ref = emg-live-probe-20260907:",
                    "evidence_ref = tampered-evidence:",
                ),
            }
            for name, config_text in cases.items():
                with self.subTest(name=name):
                    (deployment / "config.ini").write_text(
                        config_text, encoding="utf-8"
                    )
                    with self.assertRaises(ValueError):
                        validate_protocol_contract(deployment)


if __name__ == "__main__":
    unittest.main()
