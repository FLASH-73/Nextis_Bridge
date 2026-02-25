"""Tests for the leader-side position read filter (hybrid median-EMA).

Tests the algorithm from DamiaoMotorsBus._filter_position() without requiring
hardware dependencies (CAN bus, serial). Uses a minimal replica of the filter
interface to test the algorithm in isolation.
"""
import collections
import statistics

import pytest

# Constants matching damiao.py module-level values
LEADER_READ_FILTER_WINDOW = 5
LEADER_READ_FILTER_ALPHA = 0.92


class FilterUnderTest:
    """Minimal replica of DamiaoMotorsBus filter state for unit testing."""

    def __init__(self):
        self._read_filter_buffers: dict[str, collections.deque] = {}
        self._read_filter_prev: dict[str, float] = {}

    def _filter_position(self, name: str, raw: float) -> float:
        if name not in self._read_filter_buffers:
            self._read_filter_buffers[name] = collections.deque(maxlen=LEADER_READ_FILTER_WINDOW)
            self._read_filter_prev[name] = raw

        buf = self._read_filter_buffers[name]
        buf.append(raw)

        if len(buf) < 3:
            self._read_filter_prev[name] = raw
            return raw

        median_val = statistics.median(buf)
        prev = self._read_filter_prev[name]
        filtered = LEADER_READ_FILTER_ALPHA * median_val + (1.0 - LEADER_READ_FILTER_ALPHA) * prev
        self._read_filter_prev[name] = filtered
        return filtered

    def reset_read_filters(self) -> None:
        self._read_filter_buffers.clear()
        self._read_filter_prev.clear()


@pytest.fixture
def filt():
    return FilterUnderTest()


class TestStaircaseElimination:
    """Feed a quantized staircase and verify smooth transition."""

    def test_step_is_spread_over_multiple_frames(self, filt):
        """A 0.005 rad step after 10 constant frames should not appear as
        a single hard jump in the output."""
        motor = "base"
        step_size = 0.005

        # Prime with 10 frames of constant position
        for _ in range(10):
            filt._filter_position(motor, 0.0)

        # Now feed the step
        outputs = []
        for _ in range(5):
            outputs.append(filt._filter_position(motor, step_size))

        # The first output after the step must be less than the full step
        # (filter spreads the transition)
        assert outputs[0] < step_size, (
            f"First output {outputs[0]} should be less than step {step_size}"
        )
        # After a few frames, output should approach the step value
        assert outputs[-1] > step_size * 0.8, (
            f"Output after 5 frames {outputs[-1]} should be >80% of step {step_size}"
        )


class TestLatency:
    """Feed a large step and verify 95% response within 4 samples."""

    def test_reaches_95_percent_within_4_samples(self, filt):
        motor = "link1"
        # Prime buffer with 5 frames at 0.0
        for _ in range(5):
            filt._filter_position(motor, 0.0)

        # Feed 1.0 rad step
        outputs = []
        for _ in range(4):
            outputs.append(filt._filter_position(motor, 1.0))

        assert outputs[-1] >= 0.95, (
            f"Output after 4 step samples: {outputs[-1]}, expected >= 0.95"
        )


class TestMonotonicity:
    """Feed a strictly increasing ramp and verify no reversals."""

    def test_ramp_output_is_non_decreasing(self, filt):
        motor = "link2"
        increment = 0.001
        prev_output = -float("inf")

        for i in range(50):
            raw = increment * i
            out = filt._filter_position(motor, raw)
            assert out >= prev_output, (
                f"Non-monotonic at sample {i}: {out} < {prev_output}"
            )
            prev_output = out


class TestPassthroughAtStartup:
    """With < 3 samples, raw value should be returned unchanged."""

    def test_first_sample_passthrough(self, filt):
        assert filt._filter_position("link3", 1.234) == 1.234

    def test_second_sample_passthrough(self, filt):
        filt._filter_position("link3", 1.0)
        assert filt._filter_position("link3", 2.0) == 2.0

    def test_third_sample_is_filtered(self, filt):
        filt._filter_position("link4", 1.0)
        filt._filter_position("link4", 2.0)
        result = filt._filter_position("link4", 3.0)
        # With 3 samples [1, 2, 3], median=2.0, prev=3.0 (last raw passthrough)
        # filtered = 0.92 * 2.0 + 0.08 * 3.0 = 2.08
        # So result should NOT equal the raw value 3.0
        assert result != 3.0


class TestReset:
    """Verify reset_read_filters() clears state properly."""

    def test_buffers_cleared(self, filt):
        # Build up filter state
        for i in range(10):
            filt._filter_position("base", float(i))
            filt._filter_position("link1", float(i) * 2)

        assert len(filt._read_filter_buffers) == 2
        assert len(filt._read_filter_prev) == 2

        filt.reset_read_filters()

        assert len(filt._read_filter_buffers) == 0
        assert len(filt._read_filter_prev) == 0

    def test_filter_works_after_reset(self, filt):
        # Prime and reset
        for i in range(10):
            filt._filter_position("base", 1.0)
        filt.reset_read_filters()

        # First sample after reset should be passthrough
        assert filt._filter_position("base", 5.0) == 5.0
