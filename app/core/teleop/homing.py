import logging
import time

logger = logging.getLogger(__name__)


def check_calibration(svc):
    """Verify leader and follower calibration status."""
    # Check Leader Calibration
    if svc.leader:
        if not getattr(svc.leader, "is_calibrated", False):
            logger.error(f"Leader {svc.leader} is NOT calibrated.")
            return False

    # Check Follower Calibration
    if svc.robot:
        if not getattr(svc.robot, "is_calibrated", False):
            logger.error(f"Robot {svc.robot} is NOT calibrated.")
            return False

    return True


def get_home_position(svc, robot) -> dict | None:
    """Get saved home position from arm registry config for the active follower."""
    if not svc.arm_registry:
        return None
    for arm_id, instance in svc.arm_registry.arm_instances.items():
        if instance is robot:
            arm_def = svc.arm_registry.arms.get(arm_id)
            if arm_def and arm_def.config.get("home_position"):
                return arm_def.config["home_position"]
    return None


def disable_follower_motors(robot):
    """Immediately disable all Damiao follower motors."""
    try:
        from lerobot.motors.damiao.damiao import DamiaoMotorsBus
        bus = getattr(robot, 'bus', None)
        if bus and isinstance(bus, DamiaoMotorsBus):
            logger.info("Disabling Damiao follower torque...")
            for motor in bus._motors.values():
                bus._control.disable(motor)
    except Exception as e:
        logger.warning(f"Failed to disable Damiao follower torque: {e}")


def homing_loop(svc, robot, home_pos, duration=10.0, homing_vel=0.05):
    """Move robot to home position over duration, then disable motors.

    Uses the existing MIT rate limiter in sync_write() for smooth movement.
    At homing_vel=0.15: J8009P ~1 rad/s, J4340P ~0.8 rad/s.
    """
    try:
        from lerobot.motors.damiao.damiao import DamiaoMotorsBus
        bus = getattr(robot, 'bus', None)
        if not bus or not isinstance(bus, DamiaoMotorsBus):
            return

        old_vel = bus.velocity_limit
        bus.velocity_limit = homing_vel
        print(f"[Teleop] Homing started (vel={homing_vel}, duration={duration}s)", flush=True)

        start = time.time()
        while time.time() - start < duration:
            if getattr(svc, '_homing_cancel', False):
                print("[Teleop] Homing cancelled", flush=True)
                break
            bus.sync_write("Goal_Position", home_pos)
            time.sleep(1.0 / 30)  # 30Hz

        bus.velocity_limit = old_vel
    except Exception as e:
        print(f"[Teleop] Homing error: {e}", flush=True)
    finally:
        # Always disable motors when done
        try:
            from lerobot.motors.damiao.damiao import DamiaoMotorsBus
            bus = getattr(robot, 'bus', None)
            if bus and isinstance(bus, DamiaoMotorsBus):
                for motor in bus._motors.values():
                    bus._control.disable(motor)
                print("[Teleop] Homing complete — motors disabled", flush=True)
        except Exception:
            pass
