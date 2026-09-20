import dataclasses
import unittest

from emg_protocol import DeviceKey, HandSide, RateDescriptor
from training_capture_support import (
    TRAINING_ACTIONS,
    TrainingCaptureSettings,
    format_live_quality,
    format_stop_report,
)


SUBJECT = "sub-0123456789abcdef0123456789abcdef"
DEVICE = DeviceKey("dev-0123456789abcdef0123456789abcdef")


class TrainingCaptureSettingsTests(unittest.TestCase):
    def settings(self, **changes):
        values = {
            "subject_id": SUBJECT,
            "session_id": "batch01-rest-001",
            "device_id": DEVICE,
            "hand_side": HandSide.LEFT,
            "action_label": "rest",
            "experiment_batch": "batch01",
            "countdown_seconds": 3,
            "duration_seconds": 30,
        }
        values.update(changes)
        return TrainingCaptureSettings(**values)

    def test_fixed_labels_hold_phase_and_canonical_provenance(self):
        self.assertEqual(TRAINING_ACTIONS, ("rest", "fist", "open_hand"))
        for action in TRAINING_ACTIONS:
            with self.subTest(action=action):
                settings = self.settings(action_label=action)
                context = settings.to_recording_context()
                self.assertEqual(settings.action_phase, "hold")
                self.assertEqual(context.action_label, action)
                self.assertEqual(context.action_phase, "hold")
                self.assertEqual(context.training_provenance, "canonical_session")
                self.assertEqual(
                    context.to_training_provenance_metadata(),
                    {
                        "schema": "emg.training.provenance",
                        "version": "1.0",
                        "kind": "canonical_session",
                    },
                )

    def test_settings_are_immutable_and_device_id_is_typed(self):
        settings = self.settings()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            settings.device_id = DeviceKey("dev-ffffffffffffffffffffffffffffffff")
        with self.assertRaisesRegex(ValueError, "connected DeviceKey"):
            self.settings(device_id=str(DEVICE))

    def test_rejects_unsafe_identifiers_side_label_and_timing(self):
        invalid = (
            {"subject_id": "Alice"},
            {"session_id": "../escape"},
            {"session_id": "COM1"},
            {"hand_side": HandSide.UNKNOWN},
            {"action_label": "custom"},
            {"experiment_batch": "Batch 1"},
            {"countdown_seconds": -1},
            {"countdown_seconds": 61},
            {"countdown_seconds": True},
            {"duration_seconds": 0},
            {"duration_seconds": 3601},
            {"duration_seconds": 1.5},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.settings(**changes)

    def test_session_id_uses_recorder_canonicalization(self):
        self.assertEqual(self.settings(session_id="HasUpper").session_id, "hasupper")


class TrainingCaptureFormattingTests(unittest.TestCase):
    def recorder(self, **changes):
        values = {
            "sequence_detection_available": False,
            "duplicate_count": 0,
            "out_of_order_count": 0,
            "gap_count": 0,
            "connection_generation": 2,
            "recorded_rows": 120,
        }
        values.update(changes)
        return values

    def test_live_quality_never_promotes_host_rate_to_confirmed_rate(self):
        text = format_live_quality(
            pipeline_snapshot={
                "dropped_count": 3,
                "queue_depth": 1,
                "freshness_seconds": 0.125,
            },
            recorder_snapshot=self.recorder(),
            sample_rate=RateDescriptor(),
            host_observed_rate_hz=49.8,
        )
        self.assertIn("设备序列检测=不可检测", text)
        self.assertIn("确认采样率=不可用", text)
        self.assertIn("主机观测速率=49.80 Hz", text)
        self.assertIn("不等于确认采样率", text)

    def test_live_quality_shows_sequence_counts_and_rate_evidence(self):
        text = format_live_quality(
            pipeline_snapshot={
                "dropped_count": 0,
                "queue_depth": 0,
                "freshness_seconds": None,
            },
            recorder_snapshot=self.recorder(
                sequence_detection_available=True,
                duplicate_count=2,
                out_of_order_count=1,
                gap_count=4,
            ),
            sample_rate=RateDescriptor(200, "firmware", "fw-v2", True),
        )
        self.assertIn("设备序列重复=2, 设备序列乱序=1, 设备序列缺口=4", text)
        self.assertIn("上位机队列丢弃=0", text)
        self.assertIn("连接代次=2", text)
        self.assertIn("确认采样率=200 Hz (firmware, fw-v2)", text)

    def test_live_quality_rejects_missing_or_invalid_evidence(self):
        with self.assertRaises(ValueError):
            format_live_quality(
                pipeline_snapshot={"dropped_count": 0},
                recorder_snapshot=self.recorder(),
                sample_rate=RateDescriptor(),
            )
        with self.assertRaises(ValueError):
            format_live_quality(
                pipeline_snapshot={
                    "dropped_count": 0,
                    "queue_depth": 0,
                    "freshness_seconds": -1,
                },
                recorder_snapshot=self.recorder(),
                sample_rate=RateDescriptor(),
            )

    def test_stop_report_lists_every_fail_and_warning(self):
        text = format_stop_report(
            {
                "training_usable": False,
                "checks": {
                    "metadata": {"status": "pass", "message": "ok"},
                    "packet_loss": {"status": "fail", "message": "loss above limit"},
                    "rate": {"status": "warning", "message": "rate unconfirmed"},
                },
            }
        )
        self.assertTrue(text.startswith("训练资格：不可训练"))
        self.assertIn("[FAIL] packet_loss: loss above limit", text)
        self.assertIn("[WARNING] rate: rate unconfirmed", text)
        self.assertNotIn("metadata: ok", text)

    def test_stop_report_passes_only_clean_usable_report_and_fails_closed(self):
        clean = format_stop_report(
            {
                "training_usable": True,
                "checks": {"all": {"status": "pass", "message": "ok"}},
            }
        )
        self.assertIn("训练资格：可训练", clean)
        inconsistent = format_stop_report(
            {
                "training_usable": True,
                "checks": {"rate": {"status": "warning", "message": "unknown"}},
            }
        )
        self.assertIn("训练资格：不可训练", inconsistent)
        unspecified = format_stop_report({"training_usable": False, "checks": {}})
        self.assertIn("未提供具体", unspecified)


if __name__ == "__main__":
    unittest.main()
