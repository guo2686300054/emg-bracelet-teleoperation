import numpy as np
import pytest

from emg_features import (
    FEATURE_NAMES,
    FEATURE_SCRATCH_WINDOW_MULTIPLIER,
    FeatureSpec,
    extract_time_domain_features,
    feature_extraction_scratch_bytes,
)


def test_exact_features_are_channel_major_and_input_is_unchanged():
    window = np.array(
        [[1.0, -1.0], [-1.0, -1.0], [2.0, 1.0], [-2.0, 1.0]],
        dtype=np.float32,
    )
    original = window.copy()

    actual = extract_time_domain_features(window, FeatureSpec(channel_count=2))

    expected = np.array(
        [1.5, np.sqrt(2.5), 9.0, 3.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.0]
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(window, original)
    assert FEATURE_NAMES == ("MAV", "RMS", "WL", "ZC", "SSC")


def test_feature_scratch_contract_is_conservative_and_dimension_checked():
    window_bytes = 40 * 8 * np.dtype(np.float64).itemsize
    output_bytes = 8 * len(FEATURE_NAMES) * np.dtype(np.float64).itemsize
    assert FEATURE_SCRATCH_WINDOW_MULTIPLIER == 5
    assert feature_extraction_scratch_bytes(40, 8) == 5 * window_bytes + output_bytes
    with pytest.raises(ValueError, match="dimensions"):
        feature_extraction_scratch_bytes(2, 8)


def test_thresholds_are_strict_and_zero_slopes_products_do_not_count():
    window = np.array([[1.0], [-1.0], [0.0], [2.0], [0.0]])
    # Deltas are -2, +1, +2, -2. ZC delta equality at 2 does not count.
    # At the final peak, both adjacent slopes equal the SSC threshold, so it
    # also does not count; zero sample products are not crossings.
    result = extract_time_domain_features(
        window,
        FeatureSpec(channel_count=1, zc_threshold=2.0, ssc_threshold=2.0),
    )
    assert result[3] == 0.0
    assert result[4] == 0.0


@pytest.mark.parametrize(
    "window, expected",
    [
        (np.zeros((3, 1)), [0.0, 0.0, 0.0, 0.0, 0.0]),
        (np.full((3, 1), -2.0), [2.0, 2.0, 0.0, 0.0, 0.0]),
        (np.array([[-1.0], [1.0], [-1.0]]), [1.0, 1.0, 4.0, 2.0, 1.0]),
    ],
)
def test_shortest_and_degenerate_windows_are_deterministic(window, expected):
    actual = extract_time_domain_features(window, FeatureSpec(channel_count=1))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "window",
    [
        np.array([1.0, 2.0, 3.0]),
        np.zeros((2, 1)),
        np.zeros((3, 0)),
        np.array([[True], [False], [True]]),
        np.array([[1], [2], [3]], dtype=object),
        np.array([[1.0], [np.nan], [2.0]]),
        np.array([[1.0], [np.inf], [2.0]]),
    ],
)
def test_invalid_windows_are_rejected(window):
    with pytest.raises(ValueError):
        extract_time_domain_features(window, FeatureSpec(channel_count=1))


def test_non_array_and_wrong_channel_count_are_rejected():
    with pytest.raises(ValueError):
        extract_time_domain_features([[1.0], [2.0], [3.0]], FeatureSpec(channel_count=1))
    with pytest.raises(ValueError):
        extract_time_domain_features(np.zeros((3, 2)), FeatureSpec(channel_count=1))


@pytest.mark.parametrize("field", ["zc_threshold", "ssc_threshold"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, True, "0"])
def test_invalid_thresholds_are_rejected(field, value):
    with pytest.raises(ValueError):
        FeatureSpec(**{field: value})


def test_feature_spec_rejects_unknown_version_or_order():
    with pytest.raises(ValueError):
        FeatureSpec(version="time_domain_v2")
    with pytest.raises(ValueError):
        FeatureSpec(names=("RMS", "MAV", "WL", "ZC", "SSC"))
