"""
Slice parsing utilities for parameter sweep generation.

This module provides functions for parsing slice notation strings
into lists of values for parameter sweeps.
"""

from typing import List

import numpy as np


def parse_slice(slice_str: str, precision: int = 3) -> List[List[float]]:
    """Parse slice string into list of float ranges.
    Example:
        >>> parse_slice("0.5:1.5:0.1,2:3:0.2", precision=2)
        [[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], [2.0, 2.2, 2.4, 2.6, 2.8]]
    """
    result = []
    slice_segments = slice_str.split(",")

    for slice_segment in slice_segments:
        slice_parts = slice_segment.split(":")
        assert len(slice_parts) == 3, (
            f"Invalid slice format: {slice_segment}. Expected format: start:end:step"
        )

        try:
            start, end, step = map(float, slice_parts)
            segment = np.arange(start, end, step)
            rounded_seg = [round(val, precision) for val in segment]
            result.append(rounded_seg)
        except ValueError as e:
            raise ValueError(f"Invalid slice values in {slice_segment}: {e}")

    return result


def concatenate_slices(segment_values: List[List[float]]) -> List[float]:
    """Flatten a list of lists into a single list.
    Example:
        >>> concatenate_slices([[1.0, 1.1], [2.0, 2.1]])
        [1.0, 1.1, 2.0, 2.1]
    """
    return [item for sublist in segment_values for item in sublist]
