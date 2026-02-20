"""Independent safety monitoring thread.

The teleop control loop checks Damiao torque limits every 6th frame (~10Hz).
But during HIL and RL modes, those loops skip safety entirely. This watchdog
fills that gap — it runs continuously at ~10Hz whenever any follower arm is
connected, regardless of which control mode is active.

This is a DEFENSE-IN-DEPTH measure. It does NOT replace the per-frame checks
in the teleop loop (those catch issues faster). It catches issues that the
mode-specific loops miss.
"""

import logging
import threading
import time

from app.core.hardware.types import ArmRole, MotorType

logger = logging.getLogger(__name__)


class SafetyWatchdog:
    """Independent 10Hz safety monitor for all connected follower arms."""

    CHECK_HZ = 10
    VIOLATION_LIMIT = 5  # Higher than teleop's 3 — watchdog is slower, needs more debounce

    def __init__(self, arm_registry, safety_layer):
        self._registry = arm_registry
        self._safety = safety_layer
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.is_running = False

        # Own violation tracking — separate from SafetyLayer's counts
        # to avoid race conditions with the teleop loop.
        self._violation_counts: dict[str, int] = {}

    def start(self):
        """Start the watchdog thread. Idempotent."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._violation_counts.clear()
        self.is_running = True
        self._thread = threading.Thread(
            target=self._watchdog_loop,
            name="safety-watchdog",
            daemon=True,
        )
        self._thread.start()
        logger.info("Safety watchdog started (interval: %dms)", int(1000 / self.CHECK_HZ))

    def stop(self):
        """Stop the watchdog thread."""
        if not self.is_running:
            return
        self.is_running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Safety watchdog stopped")

    def _watchdog_loop(self):
        interval = 1.0 / self.CHECK_HZ
        while not self._stop_event.is_set():
            try:
                self._check_all_arms()
            except Exception as e:
                # Safety monitor crash = UNSAFE — E-STOP everything
                logger.error("Safety watchdog exception — triggering E-STOP all: %s", e)
                self._emergency_stop_all()
                break
            self._stop_event.wait(timeout=interval)

    def _check_all_arms(self):
        """Check every connected follower arm for safety violations."""
        # Snapshot arm_instances to avoid issues with dict changing during iteration
        try:
            instances = list(self._registry.arm_instances.items())
        except Exception:
            return

        for arm_id, instance in instances:
            arm_def = self._registry.arms.get(arm_id)
            if not arm_def:
                continue

            # Only monitor followers — leaders are input devices
            if arm_def.role == ArmRole.LEADER:
                continue

            is_safe = True

            if arm_def.motor_type == MotorType.DAMIAO:
                is_safe = self._check_damiao(arm_id, instance)
            elif arm_def.motor_type == MotorType.STS3215:
                is_safe = self._check_feetech(arm_id, instance)
            # Dynamixel leaders: skip (caught by role check above)

            if is_safe:
                self._violation_counts[arm_id] = 0
            else:
                count = self._violation_counts.get(arm_id, 0) + 1
                self._violation_counts[arm_id] = count
                if count >= self.VIOLATION_LIMIT:
                    logger.error(
                        "SAFETY WATCHDOG: %d consecutive violations on %s — E-STOP",
                        count, arm_id,
                    )
                    self._emergency_stop_arm(arm_id, instance)
                    self._violation_counts[arm_id] = 0

    def _check_damiao(self, arm_id: str, instance) -> bool:
        """Check Damiao arm torque limits. Returns True if safe."""
        try:
            if not hasattr(instance, 'get_torques') or not hasattr(instance, 'get_torque_limits'):
                return True

            torques = instance.get_torques()
            limits = instance.get_torque_limits()

            for motor_name, torque in torques.items():
                limit = limits.get(motor_name, 10.0)
                if abs(torque) > limit:
                    logger.warning(
                        "Watchdog: Damiao %s/%s torque %.2fNm > %.1fNm",
                        arm_id, motor_name, torque, limit,
                    )
                    return False
            return True
        except Exception as e:
            logger.warning("Watchdog: Damiao check failed for %s: %s", arm_id, e)
            return False  # Fail-closed

    def _check_feetech(self, arm_id: str, instance) -> bool:
        """Check Feetech arm load limits. Returns True if safe.

        Feetech STS3215 Present_Load register: 0-1000 = 0-100%.
        Sustained load > 50% (LOAD_THRESHOLD=500) indicates stall/collision.
        """
        try:
            buses = []
            if hasattr(instance, 'bus'):
                buses.append(instance.bus)
            if hasattr(instance, 'left_arm'):
                buses.append(instance.left_arm.bus)
            if hasattr(instance, 'right_arm'):
                buses.append(instance.right_arm.bus)

            for bus in buses:
                # Skip Damiao buses
                try:
                    from lerobot.motors.damiao.damiao import DamiaoMotorsBus
                    if isinstance(bus, DamiaoMotorsBus):
                        continue
                except ImportError:
                    pass

                for motor_name in bus.motors:
                    try:
                        load_val = bus.read("Present_Load", motor_name, normalize=False)
                        magnitude = load_val % 1024
                        if magnitude > self._safety.LOAD_THRESHOLD:
                            logger.warning(
                                "Watchdog: Feetech %s/%s load %d > %d",
                                arm_id, motor_name, magnitude, self._safety.LOAD_THRESHOLD,
                            )
                            return False
                    except Exception:
                        continue  # Single motor read failure — not a safety issue
            return True
        except Exception as e:
            logger.warning("Watchdog: Feetech check failed for %s: %s", arm_id, e)
            return False  # Fail-closed

    def _emergency_stop_arm(self, arm_id: str, instance):
        """E-STOP a specific arm via SafetyLayer."""
        arm_def = self._registry.arms.get(arm_id)
        motor_type = arm_def.motor_type.value if arm_def else None
        try:
            self._safety.emergency_stop(instance, motor_type=motor_type)
        except Exception as e:
            logger.error("Watchdog: E-STOP failed for %s: %s — forcing disconnect", arm_id, e)
            try:
                instance.disconnect()
            except Exception:
                pass

    def _emergency_stop_all(self):
        """E-STOP all connected arms — used when the watchdog itself crashes."""
        try:
            instances = list(self._registry.arm_instances.items())
        except Exception:
            return
        for arm_id, instance in instances:
            self._emergency_stop_arm(arm_id, instance)
        self.is_running = False
