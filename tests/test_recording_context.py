import dataclasses
import unittest

from emg_protocol import HandSide
from recording_context import RecordingContext, validate_experiment_id


SUBJECT = "sub-0123456789abcdef0123456789abcdef"


class RecordingContextTests(unittest.TestCase):
    def make(self, **changes):
        values = {
            "subject_id": SUBJECT,
            "action_label": "fist",
            "action_phase": "hold",
            "experiment_id": "discrete_hand_v1",
            "hand_side": HandSide.LEFT,
        }
        values.update(changes)
        return RecordingContext(**values)

    def test_context_is_immutable_and_serializes_only_action_metadata(self):
        context = self.make()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            context.action_label = "rest"
        self.assertEqual(
            context.to_metadata_extra(),
            {
                "action_label": "fist",
                "action_phase": "hold",
                "experiment_id": "discrete_hand_v1",
            },
        )
        self.assertNotIn("name", repr(context.to_metadata_extra()).casefold())

    def test_explicit_training_provenance_is_versioned_but_optional_for_recording(self):
        self.assertIsNone(self.make().to_training_provenance_metadata())
        context = self.make(training_provenance="synthetic_test")
        self.assertEqual(
            context.to_training_provenance_metadata(),
            {
                "schema": "emg.training.provenance",
                "version": "1.0",
                "kind": "synthetic_test",
            },
        )
        with self.assertRaises(ValueError):
            self.make(training_provenance="unknown")

    def test_rejects_noncanonical_action_and_phase(self):
        for changes in (
            {"action_label": "pinch"},
            {"action_label": "FIST"},
            {"action_phase": "transition"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.make(**changes)

    def test_rejects_invalid_subject_and_unknown_or_untyped_hand(self):
        for changes in (
            {"subject_id": "Alice"},
            {"hand_side": HandSide.UNKNOWN},
            {"hand_side": "left"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.make(**changes)

    def test_experiment_identifier_is_path_safe_and_bounded(self):
        for value in (
            "Experiment",
            "../escape",
            "with space",
            "",
            ".",
            "con",
            "com1.txt",
            "nul.json",
            "a" * 65,
            "实验",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_experiment_id(value)
        self.assertEqual(validate_experiment_id("hand-v1.2_trial"), "hand-v1.2_trial")

    def test_constructor_has_no_name_or_freeform_identity_field(self):
        with self.assertRaises(TypeError):
            RecordingContext(
                subject_id=SUBJECT,
                action_label="rest",
                action_phase="hold",
                experiment_id="study_v1",
                hand_side=HandSide.LEFT,
                patient_name="Alice",
            )


if __name__ == "__main__":
    unittest.main()
