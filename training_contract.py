"""Canonical, dependency-free validation rules shared by training tools."""

from __future__ import annotations


CANONICAL_LABELS = ("rest", "fist", "open_hand")
EMG_CHANNEL_COUNT = 8
ALLOWED_ACTION_LABELS = frozenset(CANONICAL_LABELS)
ALLOWED_ACTION_PHASES = frozenset({"hold"})
TRUSTED_SAMPLE_RATE_SOURCE_KINDS = frozenset(
    {"protocol", "firmware", "device-spec", "hardware-spec"}
)

TRAINING_PROVENANCE_SCHEMA = "emg.training.provenance"
TRAINING_PROVENANCE_VERSION = "1.0"
ALLOWED_TRAINING_PROVENANCE = frozenset(
    {"canonical_session", "synthetic_test", "external_benchmark"}
)


def validate_emg_channel_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != EMG_CHANNEL_COUNT:
        raise ValueError(f"formal EMG training requires exactly {EMG_CHANNEL_COUNT} channels")
    return value


def validate_training_annotation(action_label: object, action_phase: object) -> tuple[str, str]:
    if not isinstance(action_label, str) or action_label not in ALLOWED_ACTION_LABELS:
        raise ValueError("action_label is not in the canonical training label set")
    if not isinstance(action_phase, str) or action_phase not in ALLOWED_ACTION_PHASES:
        raise ValueError("action_phase is not in the canonical training phase set")
    return action_label, action_phase


def validate_sample_rate_source_kind(source_kind: object) -> str:
    if not isinstance(source_kind, str):
        raise ValueError("sample rate source kind is not in the trusted source allowlist")
    normalized = source_kind.strip().casefold().replace("_", "-")
    if normalized not in TRUSTED_SAMPLE_RATE_SOURCE_KINDS:
        raise ValueError("sample rate source is not in the trusted source allowlist")
    return normalized


def make_training_provenance(kind: object) -> dict[str, str]:
    if not isinstance(kind, str) or kind not in ALLOWED_TRAINING_PROVENANCE:
        raise ValueError("training provenance kind is unsupported")
    return {
        "schema": TRAINING_PROVENANCE_SCHEMA,
        "version": TRAINING_PROVENANCE_VERSION,
        "kind": kind,
    }


def validate_training_provenance(value: object) -> str:
    if not isinstance(value, dict) or set(value) != {"schema", "version", "kind"}:
        raise ValueError("training provenance must be an exact versioned object")
    if value["schema"] != TRAINING_PROVENANCE_SCHEMA:
        raise ValueError("training provenance schema is unsupported")
    if value["version"] != TRAINING_PROVENANCE_VERSION:
        raise ValueError("training provenance version is unsupported")
    kind = value["kind"]
    if not isinstance(kind, str) or kind not in ALLOWED_TRAINING_PROVENANCE:
        raise ValueError("training provenance kind is unsupported")
    return kind
