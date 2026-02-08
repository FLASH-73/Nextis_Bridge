#!/usr/bin/env python3
"""Calibrate dual-encoder motors (J8009P-2EC) by moving to MIT encoder zero.

The J8009P-2EC motors have two independent encoders. The set_zero_position (0xFE)
command only resets one encoder's reference. If the MIT encoder has accumulated
offset from previous incidents (violent spinning, etc.), this script will slowly
rotate the motor until the MIT encoder reads near 0, then zero both.

The script uses configure() to detect the TRUE MIT offset (the second-pass
measurement), then sends DIRECT MIT commands to move the motor — bypassing the
joint limit clamping in sync_write.

IMPORTANT:
- The motor may need to rotate up to 2 full turns (~12.5 rad)
- Ensure the joint can rotate freely (detach arm if needed)
- The script moves SLOWLY (~0.3 rad/s) and can be stopped with Ctrl+C

Usage:
    python calibrate_encoder.py link1           # Calibrate link1
    python calibrate_encoder.py link1 --dry-run # Show encoder state only
"""
import sys
import time

sys.path.insert(0, "lerobot/src")

from lerobot.motors.damiao.damiao import DamiaoMotorsBusConfig, DamiaoMotorsBus
from lerobot.motors.damiao.tables import DEFAULT_DAMIAO_MOTORS, DAMIAO_MOTOR_SPECS

RATE_HZ = 60
MOVE_SPEED = 0.3  # rad/s — slow and safe
MAX_OFFSET = 13.0  # rad — refuse to calibrate if offset exceeds this


def read_all_encoders(bus, name):
    """Read all encoder values for a motor."""
    motor = bus._motors[name]

    # 0xCC position (zeroed reference)
    bus._control.refresh_motor_status(motor)
    time.sleep(0.02)
    p_cc = motor.getPosition()

    # RID 80: motor encoder (raw)
    p_m = bus._control.read_motor_param(motor, 80)

    # RID 81: output encoder (raw)
    xout = bus._control.read_motor_param(motor, 81)

    # RID 54: internal calibration offset
    m_off = bus._control.read_motor_param(motor, 54)

    return {
        "p_cc": p_cc,
        "p_m": p_m,
        "xout": xout,
        "m_off": m_off,
    }


def print_encoders(enc, label=""):
    """Pretty-print encoder values."""
    if label:
        print(f"\n  {label}")
    print(f"  {'0xCC (zeroed ref)':>20s}: {enc['p_cc']:+.4f} rad")
    fmt = lambda k, v: f"  {k:>20s}: {v}" if v is not None else f"  {k:>20s}: TIMEOUT"
    print(fmt("RID80 p_m (motor)", enc["p_m"]))
    print(fmt("RID81 xout (output)", enc["xout"]))
    print(fmt("RID54 m_off (cal)", enc["m_off"]))


def main():
    if len(sys.argv) < 2:
        print("Usage: python calibrate_encoder.py <motor_name> [--dry-run]")
        print("       python calibrate_encoder.py link1")
        print("       python calibrate_encoder.py link1 --dry-run")
        print()
        print("Available motors:", ", ".join(DEFAULT_DAMIAO_MOTORS.keys()))
        sys.exit(1)

    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    if dry_run:
        args.remove("--dry-run")

    target_name = args[0]
    if target_name not in DEFAULT_DAMIAO_MOTORS:
        print(f"ERROR: Unknown motor '{target_name}'. Available: {', '.join(DEFAULT_DAMIAO_MOTORS.keys())}")
        sys.exit(1)

    motor_info = DEFAULT_DAMIAO_MOTORS[target_name]
    motor_type = motor_info["motor_type"]
    specs = DAMIAO_MOTOR_SPECS[motor_type]
    gear_ratio = specs["gear_ratio"]

    # Use only the target motor (safer — won't accidentally enable others)
    motors = {target_name: motor_info}

    config = DamiaoMotorsBusConfig(
        port="can0",
        motors=motors,
        velocity_limit=1.0,
        use_mit_mode=True,
        mit_kp=15.0,
        mit_kd=1.5,
    )

    bus = DamiaoMotorsBus(config)

    try:
        print("=" * 70)
        print(f"Encoder Calibration — Motor: {target_name} ({motor_type})")
        print(f"Gear ratio: {gear_ratio}:1")
        print("=" * 70)

        # ─── Phase 1: Configure to detect MIT offset ───
        print("\n--- Phase 1: Detect MIT Encoder Offset ---")
        print("Connecting and configuring motor in MIT mode...")
        bus.connect()
        bus.configure()

        # The configure() second-pass measurement detects the MIT offset
        mit_offset = bus._mit_offsets.get(target_name, 0.0)

        # Check if motor was quarantined
        if target_name in bus._quarantined_motors:
            print(f"\nERROR: Motor '{target_name}' was QUARANTINED during configure().")
            print("Re-zero the motor first: python rezero_motor.py " + target_name)
            return

        # Also read RID encoders for diagnostics
        enc_before = read_all_encoders(bus, target_name)
        print_encoders(enc_before, "Encoder state:")

        print(f"\n  MIT encoder offset: {mit_offset:+.4f} rad")
        if abs(mit_offset) > 0.1:
            print(f"  The MIT encoder is {abs(mit_offset):.1f} rad from 0xCC encoder")
            print(f"  Need to rotate ~{abs(mit_offset) / (2 * 3.14159):.1f} output turns to align")
            print(f"  At {MOVE_SPEED} rad/s, this will take ~{abs(mit_offset) / MOVE_SPEED:.0f}s")

        # ─── Phase 2: Decide action ───
        if abs(mit_offset) < 0.5:
            print(f"\n  Offset is small ({mit_offset:+.4f} rad). Encoders already aligned!")
            if dry_run:
                print("  --dry-run: No changes made.")
            else:
                print("  Re-zeroing at current position...")
                motor = bus._motors[target_name]
                bus._control.disable(motor)
                time.sleep(0.05)
                bus._control.set_zero_position(motor)
                time.sleep(0.5)
                bus._control.disable(motor)
                time.sleep(0.1)
                enc_after = read_all_encoders(bus, target_name)
                print_encoders(enc_after, "After re-zero:")
            return

        if abs(mit_offset) > MAX_OFFSET:
            print(f"\n  WARNING: Offset ({mit_offset:+.2f} rad) is near the ±12.5 limit!")

        if dry_run:
            print("\n  --dry-run mode. Exiting without making changes.")
            return

        # ─── Phase 3: Confirm with user ───
        # Move direction: to bring MIT encoder to 0, we move +offset in physical space.
        # MIT_pos = user_pos + offset. For MIT_pos → 0: user_pos = -offset.
        move_direction = "positive" if mit_offset < 0 else "negative"
        print(f"\n{'!' * 70}")
        print(f"  This will ROTATE the motor ~{abs(mit_offset):.1f} rad ({move_direction} direction)")
        print(f"  Ensure the joint can rotate freely!")
        print(f"{'!' * 70}")
        print()
        response = input("Type 'GO' to start calibration (Ctrl+C to abort): ")
        if response != "GO":
            print("Aborted.")
            return

        # ─── Phase 4: Move motor using DIRECT MIT commands ───
        # We bypass sync_write to avoid joint limit clamping.
        # Ramp p_des in MIT-encoder-space from current MIT pos toward 0.
        print(f"\n--- Phase 4: Calibration Move ---")

        motor = bus._motors[target_name]
        mcfg = bus._motor_configs[target_name]
        kp = config.mit_kp
        kd = config.mit_kd

        # Read current user-space position via sync_read (applies offset)
        positions = bus.sync_read("Present_Position")
        user_pos = positions[target_name]
        mit_start = user_pos + mit_offset  # Current MIT encoder position
        mit_target = 0.0                    # Want MIT encoder at 0

        total_move = mit_target - mit_start
        duration = abs(total_move) / MOVE_SPEED
        frames = max(int(RATE_HZ * duration), 1)

        print(f"  MIT start:  {mit_start:+.4f} rad")
        print(f"  MIT target: {mit_target:+.4f} rad")
        print(f"  Movement:   {total_move:+.2f} rad over {duration:.1f}s")
        print(f"  Direct MIT commands (kp={kp}, kd={kd})")
        print(f"  Press Ctrl+C to STOP at any time.\n")

        try:
            for i in range(frames + 1):
                t = min(i / frames, 1.0)
                p_des = mit_start + t * total_move

                # Direct MIT command — bypasses joint limit clamping
                bus._control.control_MIT(
                    motor, p_des, 0.0,
                    kp, kd, 0.0,
                    mcfg.p_max, mcfg.v_max, mcfg.t_max,
                )
                time.sleep(1 / RATE_HZ)

                # Print progress every 30 frames (~0.5s)
                if i % 30 == 0 or i == frames:
                    actual_mit = motor.getPosition()
                    err = abs(actual_mit - p_des)
                    pct = t * 100
                    print(f"  [{pct:5.1f}%] p_des={p_des:+.3f}  actual={actual_mit:+.3f}  err={err:.4f}")

            # Hold at target for 1 second
            print("\n  Holding at MIT=0 for 1s...")
            for _ in range(RATE_HZ):
                bus._control.control_MIT(
                    motor, mit_target, 0.0,
                    kp, kd, 0.0,
                    mcfg.p_max, mcfg.v_max, mcfg.t_max,
                )
                time.sleep(1 / RATE_HZ)

        except KeyboardInterrupt:
            print("\n\n  STOPPED by user. Holding current position...")
            current = motor.getPosition()
            for _ in range(30):
                bus._control.control_MIT(
                    motor, current, 0.0,
                    kp, kd, 0.0,
                    mcfg.p_max, mcfg.v_max, mcfg.t_max,
                )
                time.sleep(1 / RATE_HZ)

        # ─── Phase 5: Disable, verify, and zero ───
        print(f"\n--- Phase 5: Zero Encoders ---")
        print("  Disabling motor...")
        bus._control.disable(motor)
        time.sleep(0.2)

        print("  Reading encoder state at new position...")
        enc_mid = read_all_encoders(bus, target_name)
        print_encoders(enc_mid, "At new position (before zeroing):")

        # MIT probe to verify encoder position
        bus._control.enable(motor)
        time.sleep(0.05)
        bus._control.control_MIT(
            motor, 0.0, 0.0,
            0.0, 0.0, 0.0,  # kp=0, kd=0: zero-torque probe
            mcfg.p_max, mcfg.v_max, mcfg.t_max,
        )
        time.sleep(0.05)
        final_mit = motor.getPosition()
        bus._control.disable(motor)
        time.sleep(0.1)

        print(f"\n  MIT encoder at new position: {final_mit:+.4f} rad")
        if abs(final_mit) < 1.0:
            print(f"  MIT encoder is near zero — ready to zero!")
        else:
            print(f"  WARNING: MIT encoder still at {final_mit:+.2f}, not near zero.")

        response = input("\n  Type 'ZERO' to set zero position here (anything else to skip): ")
        if response == "ZERO":
            bus._control.set_zero_position(motor)
            time.sleep(0.5)
            bus._control.disable(motor)
            time.sleep(0.1)

            print("\n  Zeroed! Verifying...")
            enc_after = read_all_encoders(bus, target_name)
            print_encoders(enc_after, "After zeroing:")

            # MIT probe to check alignment
            bus._control.enable(motor)
            time.sleep(0.05)
            bus._control.control_MIT(
                motor, 0.0, 0.0,
                0.0, 0.0, 0.0,
                mcfg.p_max, mcfg.v_max, mcfg.t_max,
            )
            time.sleep(0.05)
            mit_check = motor.getPosition()
            bus._control.disable(motor)
            time.sleep(0.1)

            gap = abs(mit_check - enc_after["p_cc"])
            print(f"\n  MIT encoder: {mit_check:+.4f} rad")
            print(f"  0xCC encoder: {enc_after['p_cc']:+.4f} rad")
            print(f"  Gap: {gap:.4f} rad")
            if gap < 0.5:
                print(f"\n  SUCCESS! Encoders aligned!")
            else:
                print(f"\n  WARNING: Gap still {gap:.2f} rad. May need another attempt.")
        else:
            print("  Skipped zeroing.")

        print(f"\nCalibration complete.")
        print(f"Run 'python test_mit_mode_position.py all' to verify.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        print("\nDisabling motor and disconnecting...")
        bus.disconnect(disable_torque=True)
        print("Done.")


if __name__ == "__main__":
    main()
