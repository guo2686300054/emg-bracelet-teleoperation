"""Validated, privacy-safe context for a single labelled recording session."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from emg_protocol import HandSide
from subject_identity import validate_subject_id
from training_contract import make_training_provenance, validate_training_annotation


_EXPERIMENT_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z", re.ASCII)
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul", "clock$",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def validate_experiment_id(value: object) -> str:
    if not isinstance(value, str) or not _EXPERIMENT_ID.fullmatch(value):
        raise ValueError(
            "experiment_id must be a lowercase safe identifier (1-64 ASCII characters)"
        )
    if value in {".", ".."} or value.endswith((".", " ")):
        raise ValueError("experiment_id is reserved")
    if value.split(".", 1)[0] in _WINDOWS_RESERVED:
        raise ValueError("experiment_id is a reserved Windows device name")
    return value


@dataclass(frozen=True, slots=True)
class RecordingContext:
    """Immutable non-identifying metadata for one action and one hand."""

    subject_id: str
    action_label: str
    action_phase: str
    experiment_id: str
    hand_side: HandSide
    training_provenance: str | None = None

    def __post_init__(self) -> None:
        validate_subject_id(self.subject_id)
        validate_training_annotation(self.action_label, self.action_phase)
        validate_experiment_id(self.experiment_id)
        if not isinstance(self.hand_side, HandSide) or self.hand_side is HandSide.UNKNOWN:
            raise ValueError("hand_side must be HandSide.LEFT or HandSide.RIGHT")
        if self.training_provenance is not None:
            make_training_provenance(self.training_provenance)

    def to_metadata_extra(self) -> dict[str, Any]:
        """Return only the non-identifying action fields accepted by DataRecorder."""
        return {
            "action_label": self.action_label,
            "action_phase": self.action_phase,
            "experiment_id": self.experiment_id,
        }

    def to_training_provenance_metadata(self) -> dict[str, str] | None:
        if self.training_provenance is None:
            return None
        return make_training_provenance(self.training_provenance)
