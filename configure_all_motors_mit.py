#!/usr/bin/env python3
"""Diagnose all 7 Damiao motors and configure them for MIT mode.

Reads current parameters from all motors, compares motor 1 vs motor 2,
switches any non-MIT motors to MIT mode, and saves to flash.

Usage:
    python configure_all_motors_mit.py
"""
import sys
import time

sys.path.insert(0, "lerobot/src")

from lerobot.motors.damiao.damiao import DamiaoMotorsBusConfig, DamiaoMotorsBus
from lerobot.motors.damiao.tables import DEFAULT_DAMIAO_MOTORS

# All 7 motors
ALL_MOTORS = DEFAULT_DAMIAO_MOTORS.copy()

# RIDs to read for diagnostics
PARAM_RIDS = {
    "CTRL_MODE": 10,
    "Damp":      11,
    "ACC":        4,
    "DEC":        5,
    "KP_ASR":    25,
    "KI_ASR":    26,
    "KP_APR":    27,
    "KI_APR":    28,
}

MODE_NAMES = {1: "MIT", 2: "POS_VEL", 3: "VEL", 4: "Torque_Pos"}


def read_all_params(bus, motor_name):
    """Read all diagnostic parameters from a single motor."""
    motor = bus._motors[motor_name]
    params = {}
    for param_name, rid in PARAM_RIDS.items():
        val = bus._control.read_motor_param(motor, rid)
        params[param_name] = val
    return params


def print_params(name, params):
    """Print parameters for a motor in a readable format."""
    mode = params.get("CTRL_MODE")
    mode_str = MODE_NAMES.get(int(mode) if mode is not None else -1, f"UNKNOWN({mode})")
    print(f"  {name:8s}  mode={mode_str:10s}  "
          f"Damp={params.get('Damp', '?'):>6}  "
          f"ACC={params.get('ACC', '?'):>6}  DEC={params.get('DEC', '?'):>6}  "
          f"KP_ASR={params.get('KP_ASR', '?'):>6}  KI_ASR={params.get('KI_ASR', '?'):>6}  "
          f"KP_APR={params.get('KP_APR', '?'):>6}  KI_APR={params.get('KI_APR', '?'):>6}")


def main():
    config = DamiaoMotorsBusConfig(
        port="can0",
        motors=ALL_MOTORS,
        velocity_limit=0.1,
        use_mit_mode=True,
    )

    bus = DamiaoMotorsBus(config)

    try:
        print("=" * 90)
        print("Damiao Motor Diagnostic — All 7 Motors")
        print("=" * 90)

        # Connect (no configure — just raw connection)
        print("\nConnecting to CAN bus...")
        bus.connect()

        # ─── Phase 1: Read current state of all motors ───
        print("\n--- Phase 1: Current Motor Parameters ---\n")
        all_params = {}
        for name in ALL_MOTORS:
            motor = bus._motors[name]
            # Refresh to verify motor responds
            bus._control.refresh_motor_status(motor)
            time.sleep(0.01)
            params = read_all_params(bus, name)
            all_params[name] = params
            print_params(name, params)

        # ─── Phase 2: Compare motor 1 (base) vs motor 2 (link1) ───
        print("\n--- Phase 2: Motor 1 (base) vs Motor 2 (link1) Comparison ---\n")
        m1 = all_params.get("base", {})
        m2 = all_params.get("link1", {})
        print(f"  {'Parameter':12s}  {'Motor 1 (base)':>15s}  {'Motor 2 (link1)':>15s}  {'Match?':>6s}")
        print(f"  {'-'*12}  {'-'*15}  {'-'*15}  {'-'*6}")
        for param_name in PARAM_RIDS:
            v1 = m1.get(param_name, "N/A")
            v2 = m2.get(param_name, "N/A")
            v1_str = f"{v1:.4f}" if isinstance(v1, float) else str(v1)
            v2_str = f"{v2:.4f}" if isinstance(v2, float) else str(v2)
            match = "YES" if v1 == v2 else "NO"
            print(f"  {param_name:12s}  {v1_str:>15s}  {v2_str:>15s}  {match:>6s}")

        # ─── Phase 3: Switch non-MIT motors to MIT mode + save to flash ───
        print("\n--- Phase 3: Apply MIT Mode to All Motors + Save to Flash ---\n")

        motors_changed = []
        motors_already_mit = []

        for name in ALL_MOTORS:
            motor = bus._motors[name]
            current_mode = all_params[name].get("CTRL_MODE")

            if current_mode is not None and int(current_mode) == 1:
                motors_already_mit.append(name)
                print(f"  {name:8s}: Already in MIT mode — saving to flash to ensure persistence...")
                # Save anyway to ensure it's in flash, not just RAM
                bus._control.save_motor_param(motor)
                print(f"  {name:8s}: Flash save complete.")
            else:
                mode_str = MODE_NAMES.get(int(current_mode) if current_mode is not None else -1, f"UNKNOWN({current_mode})")
                print(f"  {name:8s}: Currently in {mode_str} mode — switching to MIT...")

                # Disable motor first (required for mode change)
                bus._control.disable(motor)
                time.sleep(0.05)

                # Switch to MIT mode (Control_Type.MIT = 1)
                bus._control.switchControlMode(motor, 1)
                time.sleep(0.05)

                # Save to flash (1 second wait for reliable write)
                bus._control.save_motor_param(motor)
                print(f"  {name:8s}: Switched to MIT + saved to flash.")
                motors_changed.append(name)

        print(f"\n  Already MIT: {len(motors_already_mit)} motors {motors_already_mit}")
        print(f"  Switched:    {len(motors_changed)} motors {motors_changed}")

        # ─── Phase 4: Verify all motors are now in MIT mode ───
        print("\n--- Phase 4: Verification — Read Back CTRL_MODE ---\n")

        all_ok = True
        for name in ALL_MOTORS:
            motor = bus._motors[name]
            mode = bus._control.read_motor_param(motor, 10)  # RID 10 = CTRL_MODE
            mode_str = MODE_NAMES.get(int(mode) if mode is not None else -1, f"UNKNOWN({mode})")
            status = "OK" if mode is not None and int(mode) == 1 else "FAIL"
            if status == "FAIL":
                all_ok = False
            print(f"  {name:8s}: CTRL_MODE={mode} ({mode_str}) — {status}")

        # ─── Summary ───
        print(f"\n{'=' * 90}")
        print("SUMMARY")
        print(f"{'=' * 90}")
        print(f"  Motors already in MIT:   {len(motors_already_mit)}")
        print(f"  Motors switched to MIT:  {len(motors_changed)}")
        print(f"  Verification:            {'ALL PASS' if all_ok else 'SOME FAILED'}")
        print(f"\n  Configuration saved to flash — will persist across power cycles.")
        if all_ok:
            print(f"  All 7 motors ready for MIT mode teleoperation.")
        else:
            print(f"  WARNING: Some motors failed verification! Check connections.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        print("\nDisabling all motors and disconnecting...")
        bus.disconnect(disable_torque=True)
        print("Done.")


if __name__ == "__main__":
    main()
