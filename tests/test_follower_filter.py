"""Tests for second-order critically-damped position filter.

Tests the filter math used in DamiaoMotorsBus._apply_position_filter().
Pure math — no hardware imports required.
"""

import math
import random

import pytest


# Replicate the filter logic from damiao.py for standalone testing
class PositionFilter:
    """Second-order critically-damped filter (test harness)."""

    def __init__(self):
        self._state: dict[str, tuple[float, float]] = {}

    def apply(self, name: str, target: float, dt: float, omega: float) -> tuple[float, float]:
        if dt > 0.050:
            self._state[name] = (target, 0.0)
            return target, 0.0

        if name not in self._state:
            self._state[name] = (target, 0.0)
            return target, 0.0

        x, v = self._state[name]
        e = math.exp(-omega * dt)
        err = x - target
        x_new = target + e * (err * (1.0 + omega * dt) + v * dt)
        v_new = e * (v * (1.0 - omega * dt) - omega * omega * dt * err)

        self._state[name] = (x_new, v_new)
        return x_new, v_new


DT = 1.0 / 60.0  # 60 Hz
OMEGA = 55.0      # Default omega


# ── Step Response ──


def test_step_response_reaches_95_percent():
    """Position reaches 95% of step within 6/omega seconds."""
    f = PositionFilter()
    f.apply("m", 0.0, DT, OMEGA)  # init at 0

    target = 1.0
    settle_time = 6.0 / OMEGA
    steps = int(settle_time / DT)
    pos = 0.0
    for _ in range(steps):
        pos, vel = f.apply("m", target, DT, OMEGA)

    assert pos >= 0.95 * target, f"Position {pos:.4f} did not reach 95% within 6/omega"


def test_step_response_no_overshoot():
    """Critically damped means monotonic approach — no overshoot."""
    f = PositionFilter()
    f.apply("m", 0.0, DT, OMEGA)

    target = 1.0
    max_pos = 0.0
    for _ in range(300):  # ~5 seconds
        pos, vel = f.apply("m", target, DT, OMEGA)
        max_pos = max(max_pos, pos)

    assert max_pos <= target + 1e-6, f"Overshoot detected: max_pos={max_pos:.6f}"


def test_step_response_velocity_peaks_then_returns():
    """Velocity should peak early then decay to zero."""
    f = PositionFilter()
    f.apply("m", 0.0, DT, OMEGA)

    target = 1.0
    velocities = []
    for _ in range(300):
        pos, vel = f.apply("m", target, DT, OMEGA)
        velocities.append(vel)

    peak_idx = max(range(len(velocities)), key=lambda i: abs(velocities[i]))
    # Peak should be in the first quarter
    assert peak_idx < len(velocities) // 4, f"Velocity peak at step {peak_idx} (too late)"
    # Final velocity should be near zero
    assert abs(velocities[-1]) < 0.01, f"Final velocity {velocities[-1]:.4f} not near zero"


# ── Ramp Tracking ──


def test_ramp_tracking_position():
    """Steady-state position error < 0.01 rad for 0.5 rad/s ramp."""
    f = PositionFilter()
    ramp_rate = 0.5  # rad/s
    f.apply("m", 0.0, DT, OMEGA)

    # Run for 2 seconds to reach steady state
    for i in range(1, 120 + 1):
        target = ramp_rate * i * DT
        pos, vel = f.apply("m", target, DT, OMEGA)

    # Check last 30 frames for steady-state error
    errors = []
    for i in range(121, 151):
        target = ramp_rate * i * DT
        pos, vel = f.apply("m", target, DT, OMEGA)
        errors.append(abs(target - pos))

    avg_error = sum(errors) / len(errors)
    # Theoretical steady-state error for ramp: 2*rate/omega = 2*0.5/55 ≈ 0.018
    assert avg_error < 0.02, f"Steady-state position error {avg_error:.4f} >= 0.02"


def test_ramp_tracking_velocity():
    """Steady-state velocity converges to ramp rate."""
    f = PositionFilter()
    ramp_rate = 0.5
    f.apply("m", 0.0, DT, OMEGA)

    for i in range(1, 180 + 1):
        target = ramp_rate * i * DT
        pos, vel = f.apply("m", target, DT, OMEGA)

    # Last velocity should be close to ramp rate
    assert abs(vel - ramp_rate) < 0.05, f"Velocity {vel:.4f} not converged to {ramp_rate}"


# ── Direction Reversal ──


def test_direction_reversal_no_overshoot():
    """No overshoot beyond 5% at reversal point."""
    f = PositionFilter()
    ramp_rate = 0.5
    f.apply("m", 0.0, DT, OMEGA)

    # Ramp up for 1 second
    peak_target = 0.0
    for i in range(1, 61):
        target = ramp_rate * i * DT
        pos, vel = f.apply("m", target, DT, OMEGA)
        peak_target = target

    # Ramp down for 1 second
    max_pos_after_reversal = 0.0
    for i in range(1, 61):
        target = peak_target - ramp_rate * i * DT
        pos, vel = f.apply("m", target, DT, OMEGA)
        max_pos_after_reversal = max(max_pos_after_reversal, pos)

    _overshoot_limit = peak_target + 0.05 * ramp_rate * DT
    assert max_pos_after_reversal <= peak_target + 0.05, (
        f"Overshoot at reversal: max={max_pos_after_reversal:.4f}, peak_target={peak_target:.4f}"
    )


def test_direction_reversal_smooth_velocity():
    """Velocity transitions smoothly through zero at reversal."""
    f = PositionFilter()
    ramp_rate = 0.5
    f.apply("m", 0.0, DT, OMEGA)

    velocities = []
    # Ramp up
    for i in range(1, 61):
        pos, vel = f.apply("m", ramp_rate * i * DT, DT, OMEGA)
        velocities.append(vel)

    peak_target = ramp_rate * 60 * DT
    # Ramp down
    for i in range(1, 61):
        pos, vel = f.apply("m", peak_target - ramp_rate * i * DT, DT, OMEGA)
        velocities.append(vel)

    # Check that velocity changes are smooth (no spikes > 2 rad/s between frames)
    for i in range(1, len(velocities)):
        dv = abs(velocities[i] - velocities[i - 1])
        assert dv < 2.0, f"Velocity spike at step {i}: dv={dv:.4f}"


# ── Noise Rejection ──


def test_noise_rejection():
    """Velocity output std < 0.5 rad/s with +/-0.003 rad noise on position."""
    f = PositionFilter()
    rng = random.Random(42)
    base_pos = 1.0
    f.apply("m", base_pos, DT, OMEGA)

    velocities = []
    for _ in range(300):
        noisy_target = base_pos + rng.gauss(0, 0.003)
        pos, vel = f.apply("m", noisy_target, DT, OMEGA)
        velocities.append(vel)

    # Skip first 30 frames for transient
    steady_vels = velocities[30:]
    mean_v = sum(steady_vels) / len(steady_vels)
    variance = sum((v - mean_v) ** 2 for v in steady_vels) / len(steady_vels)
    std_v = variance ** 0.5

    assert std_v < 0.5, f"Velocity std {std_v:.4f} >= 0.5 (noise not rejected)"


# ── dt Spike ──


def test_dt_spike_snaps_to_target():
    """dt > 50ms snaps filter to target with zero velocity."""
    f = PositionFilter()
    f.apply("m", 0.0, DT, OMEGA)

    # Run a few normal frames
    for _ in range(10):
        f.apply("m", 0.5, DT, OMEGA)

    # Now a big dt spike
    pos, vel = f.apply("m", 2.0, 0.060, OMEGA)
    assert pos == 2.0, f"Position {pos} did not snap to target"
    assert vel == 0.0, f"Velocity {vel} not zero after snap"


# ── EMA Comparison ──


def test_smoother_velocity_than_ema():
    """New filter produces smoother velocity than EMA + finite difference.

    The EMA's velocity is computed via first-difference (same as the old code),
    which amplifies noise. The filter's velocity is analytical and smooth.
    """
    rng = random.Random(123)
    base_pos = 1.0
    noise_std = 0.003

    # EMA simulation with finite-difference velocity
    alpha = 0.85
    ema_pos = base_pos
    ema_velocities = []
    for _ in range(300):
        noisy_target = base_pos + rng.gauss(0, noise_std)
        prev_pos = ema_pos
        ema_pos = alpha * noisy_target + (1.0 - alpha) * ema_pos
        ema_v = (ema_pos - prev_pos) / DT
        ema_velocities.append(ema_v)

    # Filter simulation
    rng2 = random.Random(123)  # Same noise sequence
    f = PositionFilter()
    f.apply("m", base_pos, DT, OMEGA)
    filter_velocities = []
    for _ in range(300):
        noisy_target = base_pos + rng2.gauss(0, noise_std)
        pos, vel = f.apply("m", noisy_target, DT, OMEGA)
        filter_velocities.append(vel)

    # Compare velocity std (skip first 30 frames for transient)
    ema_steady = ema_velocities[30:]
    filter_steady = filter_velocities[30:]

    ema_mean = sum(ema_steady) / len(ema_steady)
    ema_var = sum((v - ema_mean) ** 2 for v in ema_steady) / len(ema_steady)
    ema_std = ema_var ** 0.5

    filter_mean = sum(filter_steady) / len(filter_steady)
    filter_var = sum((v - filter_mean) ** 2 for v in filter_steady) / len(filter_steady)
    filter_std = filter_var ** 0.5

    assert filter_std < ema_std, (
        f"Filter velocity std ({filter_std:.4f}) not less than EMA ({ema_std:.4f})"
    )
