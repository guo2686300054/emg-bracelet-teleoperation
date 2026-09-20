"""Pure, versioned time-domain EMG feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


FEATURE_VERSION = "time_domain_v1"
FEATURE_NAMES = ("MAV", "RMS", "WL", "ZC", "SSC")
# Conservative upper bound for simultaneous vectorized temporaries created by
# diff/abs/subtract/comparison chains, excluding the caller-owned input window.
FEATURE_SCRATCH_WINDOW_MULTIPLIER = 5


def feature_extraction_scratch_bytes(window_samples: int, channel_count: int) -> int:
    if (
        not isinstance(window_samples, int)
        or isinstance(window_samples, bool)
        or window_samples < 3
        or not isinstance(channel_count, int)
        or isinstance(channel_count, bool)
        or channel_count < 1
    ):
        raise ValueError("feature scratch dimensions are invalid")
    window_bytes = window_samples * channel_count * np.dtype(np.float64).itemsize
    output_bytes = channel_count * len(FEATURE_NAMES) * np.dtype(np.float64).itemsize
    return FEATURE_SCRATCH_WINDOW_MULTIPLIER * window_bytes + output_bytes


@dataclass(frozen=True)
class FeatureSpec:
    """The complete contract for the ``time_domain_v1`` transform.

    Thresholds use the same units as the input samples.  ``channel_count`` may
    be omitted for standalone experimentation, but bundles always set it.
    """

    version: str = FEATURE_VERSION
    names: tuple[str, ...] = FEATURE_NAMES
    zc_threshold: float = 0.0
    ssc_threshold: float = 0.0
    channel_count: int | None = None

    def __post_init__(self) -> None:
        if self.version != FEATURE_VERSION:
            raise ValueError(f"unsupported feature version: {self.version!r}")
        try:
            names = tuple(self.names)
        except TypeError as exc:
            raise ValueError(f"feature names must be exactly {FEATURE_NAMES!r}") from exc
        if names != FEATURE_NAMES:
            raise ValueError(f"feature names must be exactly {FEATURE_NAMES!r}")
        for field_name in ("zc_threshold", "ssc_threshold"):
            value = getattr(self, field_name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"{field_name} must be a finite non-negative number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{field_name} must be a finite non-negative number")
        if self.channel_count is not None and (
            isinstance(self.channel_count, (bool, np.bool_))
            or not isinstance(self.channel_count, (int, np.integer))
            or int(self.channel_count) <= 0
        ):
            raise ValueError("channel_count must be a positive integer or None")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "zc_threshold", float(self.zc_threshold))
        object.__setattr__(self, "ssc_threshold", float(self.ssc_threshold))
        if self.channel_count is not None:
            object.__setattr__(self, "channel_count", int(self.channel_count))

    @property
    def features_per_channel(self) -> int:
        return len(FEATURE_NAMES)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "names": list(self.names),
            "zc_threshold": float(self.zc_threshold),
            "ssc_threshold": float(self.ssc_threshold),
            "channel_count": self.channel_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureSpec":
        if not isinstance(value, Mapping):
            raise ValueError("feature_spec must be an object")
        expected = {
            "version",
            "names",
            "zc_threshold",
            "ssc_threshold",
            "channel_count",
        }
        if set(value) != expected:
            raise ValueError("feature_spec has missing or unknown fields")
        names = value["names"]
        if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
            raise ValueError("feature_spec names must be a string array")
        return cls(
            version=value["version"],
            names=tuple(names),
            zc_threshold=value["zc_threshold"],
            ssc_threshold=value["ssc_threshold"],
            channel_count=value["channel_count"],
        )


def _validated_window(window: object, spec: FeatureSpec) -> np.ndarray:
    if not isinstance(spec, FeatureSpec):
        raise TypeError("spec must be a FeatureSpec")
    if not isinstance(window, np.ndarray):
        raise ValueError("window must be a two-dimensional numeric NumPy array")
    if window.ndim != 2:
        raise ValueError("window must have shape (samples, channels)")
    samples, channels = window.shape
    if samples < 3:
        raise ValueError("window must contain at least three samples")
    if channels < 1:
        raise ValueError("window must contain at least one channel")
    if spec.channel_count is not None and channels != spec.channel_count:
        raise ValueError(
            f"window has {channels} channels; expected {spec.channel_count}"
        )
    if window.dtype.kind not in "iuf":
        raise ValueError("window must have a real numeric dtype (bool/object are rejected)")
    values = np.asarray(window, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("window contains NaN or infinity")
    return values


def extract_time_domain_features(window: object, spec: FeatureSpec) -> np.ndarray:
    """Return channel-major ``[MAV, RMS, WL, ZC, SSC]`` float64 features."""

    values = _validated_window(window, spec)
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.diff(values, axis=0)
        mav = np.mean(np.abs(values), axis=0, dtype=np.float64)
        rms = np.sqrt(np.mean(np.square(values), axis=0, dtype=np.float64))
        waveform_length = np.sum(np.abs(difference), axis=0, dtype=np.float64)

    left = values[:-1]
    right = values[1:]
    zero_crossings = np.sum(
        (((left < 0.0) & (right > 0.0)) | ((left > 0.0) & (right < 0.0)))
        & (np.abs(right - left) > float(spec.zc_threshold)),
        axis=0,
        dtype=np.int64,
    )

    previous_slope = difference[:-1]
    next_slope = difference[1:]
    slope_sign_changes = np.sum(
        (
            ((previous_slope < 0.0) & (next_slope > 0.0))
            | ((previous_slope > 0.0) & (next_slope < 0.0))
        )
        & (np.abs(previous_slope) > float(spec.ssc_threshold))
        & (np.abs(next_slope) > float(spec.ssc_threshold)),
        axis=0,
        dtype=np.int64,
    )

    per_channel = np.column_stack(
        (mav, rms, waveform_length, zero_crossings, slope_sign_changes)
    )
    result = np.asarray(per_channel.reshape(-1), dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("feature extraction produced a non-finite value")
    return result


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "FeatureSpec",
    "feature_extraction_scratch_bytes",
    "extract_time_domain_features",
]
