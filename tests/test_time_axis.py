"""TimeAxis must agree with the canonical bin utilities (get_total_steps,
time_to_step, step_to_time) for any data_config — and the bundle's config
must win over the global cfg."""

import numpy as np
import pytest

from astra.evaluation.utils import get_total_steps, step_to_time, time_to_step
from astra.inference.responses import TimeAxis

# A deliberately non-default grid: 6 x 10min + 4 x 30min = 10 steps.
# (The 'end' interval is open-ended: the grid window is the largest NAMED
# interval, so 'end' contributes no bins here.)
SMALL_CONFIG = {
    "bin_intervals": {"1h": "10min", "3h": "30min", "end": "1h"},
    "bin_freq_include": ["10min", "30min", "1h"],
}


def _default_data_config():
    from astra.utils import get_cfg
    cfg = get_cfg()
    return {
        "bin_intervals": cfg["bin_intervals"],
        "bin_freq_include": cfg["bin_freq_include"],
    }


class TestTimeAxisSmallConfig:
    def test_length_matches_get_total_steps(self):
        axis = TimeAxis.from_data_config(SMALL_CONFIG)
        assert len(axis) == get_total_steps(data_config=SMALL_CONFIG)

    def test_steps_contiguous_and_hours_monotonic(self):
        axis = TimeAxis.from_data_config(SMALL_CONFIG)
        assert axis.steps == list(range(len(axis)))
        assert all(e > s for s, e in zip(axis.hours_start, axis.hours_end))
        assert all(b >= a for a, b in zip(axis.hours_end, axis.hours_end[1:]))

    def test_hours_end_matches_step_to_time(self):
        axis = TimeAxis.from_data_config(SMALL_CONFIG)
        for step in range(len(axis)):
            minutes = step_to_time(step, data_config=SMALL_CONFIG)
            assert axis.hours_end[step] == pytest.approx(minutes / 60.0)

    def test_round_trip_with_time_to_step(self):
        axis = TimeAxis.from_data_config(SMALL_CONFIG)
        for step in range(len(axis)):
            # A time strictly inside the bin must map back to the same step.
            mid_min = (axis.hours_start[step] + axis.hours_end[step]) / 2 * 60.0
            assert time_to_step(mid_min, "min", data_config=SMALL_CONFIG) == step


class TestTimeAxisDefaultConfig:
    def test_matches_global_config_steps(self):
        dc = _default_data_config()
        axis = TimeAxis.from_data_config(dc)
        assert len(axis) == get_total_steps(data_config=dc)

    def test_bundle_config_wins_over_global(self):
        """A bundle with a different grid must produce a different axis —
        proving nothing falls back to the global cfg internally."""
        default_axis = TimeAxis.from_data_config(_default_data_config())
        small_axis = TimeAxis.from_data_config(SMALL_CONFIG)
        assert len(small_axis) != len(default_axis)

    def test_serializes(self):
        import json
        axis = TimeAxis.from_data_config(SMALL_CONFIG)
        d = axis.to_dict()
        json.dumps(d)
        assert set(d) == {"steps", "hours_start", "hours_end", "bin_freq"}
