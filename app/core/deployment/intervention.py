"""Intervention detector using leader arm position deltas.

Replaces both the legacy InterventionEngine (which incorrectly used observation
velocity vectors) and the HIL loop's inline _get_leader_velocity with a clean
standalone class that uses the correct approach: tracking position changes on
the leader arm between calls.
"""

import logging
import time
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class InterventionDetector:
    """Detect human intervention by tracking leader arm position deltas.

    Filters by policy-relevant arms so a left-arm-only policy only triggers
    on left leader velocity.

    Usage::

        detector = InterventionDetector(policy_arms=["left"], loop_hz=30)
        is_intervening, velocity = detector.check(leader)
    """

    def __init__(
        self,
        policy_arms: Optional[List[str]] = None,
        move_threshold: float = 0.05,
        idle_timeout: float = 2.0,
        loop_hz: float = 30.0,
    ):
        self.policy_arms = policy_arms or ["left", "right"]
        self.move_threshold = move_threshold
        self.idle_timeout = idle_timeout
        self.loop_hz = loop_hz

        self._last_positions: Optional[dict] = None
        self._last_move_time: float = 0.0

    def check(self, leader) -> Tuple[bool, float]:
        """Check leader arm for human intervention.

        Args:
            leader: Leader arm instance with ``get_action()`` method.

        Returns:
            (is_intervening, velocity) — ``is_intervening`` is True when the
            leader velocity exceeds the move threshold.
        """
        if leader is None:
            return False, 0.0

        try:
            current_pos = leader.get_action()
            if not current_pos:
                return False, 0.0

            # First call — initialize tracking
            if self._last_positions is None:
                self._last_positions = dict(current_pos)
                return False, 0.0

            # Compute max position delta across policy-relevant motors
            max_delta = 0.0
            for key, val in current_pos.items():
                if not self._is_relevant(key):
                    continue
                if key in self._last_positions:
                    delta = abs(float(val) - float(self._last_positions[key]))
                    max_delta = max(max_delta, delta)

            self._last_positions = dict(current_pos)

            # Scale by loop rate to get velocity estimate
            velocity = max_delta * self.loop_hz

            if velocity > self.move_threshold:
                self._last_move_time = time.monotonic()
                return True, velocity

            return False, velocity

        except Exception as e:
            msg = str(e)
            if "has no calibration registered" not in msg and "Failed to sync read" not in msg:
                logger.debug("Error reading leader: %s", e)
            return False, 0.0

    def is_idle(self) -> bool:
        """True if no human movement detected for longer than idle_timeout."""
        if self._last_move_time == 0.0:
            return True
        return (time.monotonic() - self._last_move_time) > self.idle_timeout

    def reset(self) -> None:
        """Clear position tracking and timing state."""
        self._last_positions = None
        self._last_move_time = 0.0

    def _is_relevant(self, key: str) -> bool:
        """Check if a motor key belongs to a policy-relevant arm."""
        if key.startswith("left_"):
            return "left" in self.policy_arms
        if key.startswith("right_"):
            return "right" in self.policy_arms
        # Non-prefixed keys (e.g. 'gripper') are always relevant
        return True
