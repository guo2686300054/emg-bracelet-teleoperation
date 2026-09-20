import pytest

from training_contract import (
    ALLOWED_ACTION_LABELS,
    CANONICAL_LABELS,
    EMG_CHANNEL_COUNT,
    make_training_provenance,
    validate_training_provenance,
    validate_sample_rate_source_kind,
    validate_training_annotation,
    validate_emg_channel_count,
)


def test_canonical_labels_have_one_stable_ordered_source():
    assert CANONICAL_LABELS == ("rest", "fist", "open_hand")
    assert ALLOWED_ACTION_LABELS == frozenset(CANONICAL_LABELS)


@pytest.mark.parametrize("channels,accepted", [(1, False), (7, False), (8, True), (9, False)])
def test_formal_emg_channel_count_is_exactly_eight(channels, accepted):
    assert EMG_CHANNEL_COUNT == 8
    if accepted:
        assert validate_emg_channel_count(channels) == 8
    else:
        with pytest.raises(ValueError, match="exactly 8"):
            validate_emg_channel_count(channels)


@pytest.mark.parametrize(
    "kind", ["canonical_session", "synthetic_test", "external_benchmark"]
)
def test_training_provenance_is_explicit_versioned_and_strict(kind):
    payload = make_training_provenance(kind)
    assert payload == {
        "schema": "emg.training.provenance",
        "version": "1.0",
        "kind": kind,
    }
    assert validate_training_provenance(payload) == kind


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"schema": "emg.training.provenance", "version": "2.0", "kind": "synthetic_test"},
        {"schema": "emg.training.provenance", "version": "1.0", "kind": "unknown"},
    ],
)
def test_missing_or_unknown_training_provenance_is_rejected(payload):
    with pytest.raises(ValueError, match="provenance"):
        validate_training_provenance(payload)


@pytest.mark.parametrize("label", ["rest", "fist", "open_hand"])
def test_canonical_training_labels_are_accepted(label):
    assert validate_training_annotation(label, "hold") == (label, "hold")


@pytest.mark.parametrize(
    "label",
    ["unknown", "custom", "", " fist ", "open", "pinch", "wrist_flex", None],
)
def test_noncanonical_training_labels_are_rejected(label):
    with pytest.raises(ValueError, match="action_label"):
        validate_training_annotation(label, "hold")


@pytest.mark.parametrize("phase", ["transition", "release", "", " hold ", None])
def test_noncanonical_training_phases_are_rejected(phase):
    with pytest.raises(ValueError, match="action_phase"):
        validate_training_annotation("fist", phase)


@pytest.mark.parametrize("source", ["protocol", "firmware", "device-spec", "hardware-spec"])
def test_trusted_sample_rate_sources_are_accepted(source):
    assert validate_sample_rate_source_kind(source) == source


@pytest.mark.parametrize("source", ["host_observed", "host-derived", "unknown", "", None])
def test_untrusted_sample_rate_sources_are_rejected(source):
    with pytest.raises(ValueError, match="trusted"):
        validate_sample_rate_source_kind(source)
