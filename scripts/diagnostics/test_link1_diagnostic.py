#!/usr/bin/env python3
"""Diagnostic test to isolate link1 dual-encoder offset issue.

The -12.2 rad "offset" only appears with 7 motors configured, not 1.
This script tests three configurations to find why:

Mode A (single):  Only link1, 5-second settle before offset measurement
Mode B (pair):    base + link1 (2 J8009P motors)
Mode C (all):     All 7 motors, but only test link1 movement

Usage:
    python test_link1_diagnostic.py single
    python test_link1_diagnostic.py pair
    python test_link1_diagnostic.py all
"""
import sys
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "lerobot" / "src"))

from lerobot.motors.damiao.damiao import DamiaoMotorsBusConfig, DamiaoMotorsBus
from lerobot.motors.damiao.tables import DAMIAO_MOTOR_SPECS

RATE_HZ = 60
HOLD_SECONDS = 2
MOVE_SECONDS = 2
MOVE_OFFSET = 0.1

MOTOR_SETS = {
    "single": {
        "link1": {"motor_type": "J8009P", "can_id": 0x02, "master_id": 0x12},
    },
    "pair": {
        "base":  {"motor_type": "J8009P", "can_id": 0x01, "master_id": 0x11},
        "link1": {"motor_type": "J8009P", "can_id": 0x02, "master_id": 0x12},
    },
    "all": {
        "base":    {"motor_type": "J8009P", "can_id": 0x01, "master_id": 0x11},
        "link1":   {"motor_type": "J8009P", "can_id": 0x02, "master_id": 0x12},
        "link2":   {"motor_type": "J4340P", "can_id": 0x03, "master_id": 0x13},
        "link3":   {"motor_type": "J4340P", "can_id": 0x04, "master_id": 0x14},
        "link4":   {"motor_type": "J4310",  "can_id": 0x05, "master_id": 0x15},
        "link5":   {"motor_type": "J4310",  "can_id": 0x06, "master_id": 0x16},
        "gripper": {"motor_type": "J4310",  "can_id": 0x07, "master_id": 0x17},
    },
}


def read_all_encoders(bus, name):
    """Read all encoder values for a motor."""
    motor = bus._motors[name]
    bus._control.refresh_motor_status(motor)
    time.sleep(0.02)
    p_cc = motor.getPosition()
    p_m = bus._control.read_motor_param(motor, 80)
    xout = bus._control.read_motor_param(motor, 81)
    return p_cc, p_m, xout


def mit_probe(bus, name):
    """Send zero-torque MIT probe and read position."""
    motor = bus._motors[name]
    mcfg = bus._motor_configs[name]
    before_ts = motor.last_seen
    bus._control.control_MIT(
        motor, 0.0, 0.0,
        0.0, 0.0, 0.0,
        mcfg.p_max, mcfg.v_max, mcfg.t_max,
    )
    deadline = time.time() + 0.100
    while motor.last_seen <= before_ts and time.time() < deadline:
        time.sleep(0.001)
    return motor.getPosition()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MOTOR_SETS:
        print("Usage: python test_link1_diagnostic.py <mode>")
        print("  single  — link1 only, 5s settle")
        print("  pair    — base + link1 (2 motors)")
        print("  all     — all 7 motors, test link1 only")
        sys.exit(1)

    mode = sys.argv[1]
    motors = MOTOR_SETS[mode]

    config = DamiaoMotorsBusConfig(
        port="can0",
        motors=motors,
        velocity_limit=0.1,
        use_mit_mode=True,
        mit_kp=15.0,
        mit_kd=1.5,
    )

    bus = DamiaoMotorsBus(config)

    try:
        print("=" * 70)
        print(f"Link1 Diagnostic — Mode: {mode} ({len(motors)} motor(s))")
        print("=" * 70)

        print("\nConnecting and configuring...")
        bus.connect()
        bus.configure()

        # Check quarantine
        if "link1" in bus._quarantined_motors:
            print("\nERROR: link1 was QUARANTINED. Re-zero first.")
            return

        mit_offset_configure = bus._mit_offsets.get("link1", 0.0)
        print(f"\n  configure() offset: {mit_offset_configure:+.4f} rad")

        # ─── Phase 1: Immediate encoder reads (right after configure) ───
        print(f"\n--- Phase 1: Immediate Reads (0s after configure) ---")
        p_cc, p_m, xout = read_all_encoders(bus, "link1")
        p_mit = mit_probe(bus, "link1")
        print(f"  0xCC:      {p_cc:+.4f}")
        print(f"  RID80 p_m: {p_m}")
        print(f"  RID81 xout:{xout}")
        print(f"  MIT probe: {p_mit:+.4f}")
        print(f"  Offset:    {p_mit - p_cc:+.4f}")

        # ─── Phase 2: 1-second settle ───
        print(f"\n--- Phase 2: After 1s of MIT probes ---")
        motor = bus._motors["link1"]
        mcfg = bus._motor_configs["link1"]
        for _ in range(60):
            bus._control.control_MIT(
                motor, 0.0, 0.0, 0.0, 0.0, 0.0,
                mcfg.p_max, mcfg.v_max, mcfg.t_max,
            )
            time.sleep(1 / RATE_HZ)
        p_cc2, p_m2, xout2 = read_all_encoders(bus, "link1")
        p_mit2 = mit_probe(bus, "link1")
        print(f"  0xCC:      {p_cc2:+.4f}")
        print(f"  RID80 p_m: {p_m2}")
        print(f"  RID81 xout:{xout2}")
        print(f"  MIT probe: {p_mit2:+.4f}")
        print(f"  Offset:    {p_mit2 - p_cc2:+.4f}")

        # ─── Phase 3: 3-second settle ───
        print(f"\n--- Phase 3: After 3s more of MIT probes ---")
        for _ in range(180):
            bus._control.control_MIT(
                motor, 0.0, 0.0, 0.0, 0.0, 0.0,
                mcfg.p_max, mcfg.v_max, mcfg.t_max,
            )
            time.sleep(1 / RATE_HZ)
        p_cc3, p_m3, xout3 = read_all_encoders(bus, "link1")
        p_mit3 = mit_probe(bus, "link1")
        print(f"  0xCC:      {p_cc3:+.4f}")
        print(f"  RID80 p_m: {p_m3}")
        print(f"  RID81 xout:{xout3}")
        print(f"  MIT probe: {p_mit3:+.4f}")
        print(f"  Offset:    {p_mit3 - p_cc3:+.4f}")

        # ─── Phase 4: 5-second settle (total ~9s from enable) ───
        print(f"\n--- Phase 4: After 5s more of MIT probes ---")
        for _ in range(300):
            bus._control.control_MIT(
                motor, 0.0, 0.0, 0.0, 0.0, 0.0,
                mcfg.p_max, mcfg.v_max, mcfg.t_max,
            )
            time.sleep(1 / RATE_HZ)
        p_cc4, p_m4, xout4 = read_all_encoders(bus, "link1")
        p_mit4 = mit_probe(bus, "link1")
        print(f"  0xCC:      {p_cc4:+.4f}")
        print(f"  RID80 p_m: {p_m4}")
        print(f"  RID81 xout:{xout4}")
        print(f"  MIT probe: {p_mit4:+.4f}")
        print(f"  Offset:    {p_mit4 - p_cc4:+.4f}")

        # Use the latest offset for movement test
        final_offset = p_mit4 - p_cc4
        if abs(final_offset) > 0.1:
            bus._mit_offsets["link1"] = final_offset
            print(f"\n  OFFSET DETECTED: {final_offset:+.4f} — applying to _mit_offsets")
        else:
            print(f"\n  No significant offset. Encoders appear aligned.")

        # ─── Phase 5: Movement test on link1 ───
        print(f"\n--- Phase 5: Movement Test (link1 only) ---")
        positions = bus.sync_read("Present_Position")
        if "link1" not in positions:
            print("  ERROR: link1 not responding")
            return

        start_pos = positions["link1"]
        print(f"  Start: {start_pos:+.4f} (user-space)")
        print(f"  MIT offset: {bus._mit_offsets.get('link1', 0.0):+.4f}")

        # Hold
        print(f"  Holding 2s...")
        for _ in range(RATE_HZ * HOLD_SECONDS):
            bus.sync_write("Goal_Position", {"link1": start_pos})
            time.sleep(1 / RATE_HZ)
        pos = bus.sync_read("Present_Position")
        hold_drift = abs(pos.get("link1", start_pos) - start_pos)

        # Move +0.1
        target = start_pos + MOVE_OFFSET
        print(f"  Moving to {target:+.4f}...")
        for _ in range(RATE_HZ * MOVE_SECONDS):
            bus.sync_write("Goal_Position", {"link1": target})
            time.sleep(1 / RATE_HZ)
        pos = bus.sync_read("Present_Position")
        move_error = abs(pos.get("link1", start_pos) - target)

        # Return
        print(f"  Returning to {start_pos:+.4f}...")
        for _ in range(RATE_HZ * MOVE_SECONDS):
            bus.sync_write("Goal_Position", {"link1": start_pos})
            time.sleep(1 / RATE_HZ)
        pos = bus.sync_read("Present_Position")
        return_error = abs(pos.get("link1", start_pos) - start_pos)

        # ─── Summary ───
        print(f"\n{'=' * 70}")
        print(f"DIAGNOSTIC SUMMARY — Mode: {mode} ({len(motors)} motor(s))")
        print(f"{'=' * 70}")
        print(f"  configure() offset:  {mit_offset_configure:+.4f}")
        print(f"  Final offset (9s):   {final_offset:+.4f}")
        print(f"  Hold drift:          {hold_drift:.4f}  {'PASS' if hold_drift < 0.02 else 'WARN'}")
        print(f"  Move error:          {move_error:.4f}  {'PASS' if move_error < 0.1 else 'WARN'}")
        print(f"  Return error:        {return_error:.4f}  {'PASS' if return_error < 0.1 else 'WARN'}")

        ok = hold_drift < 0.02 and move_error < 0.1 and return_error < 0.1
        print(f"\n  Overall: {'PASS' if ok else 'WARN'}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        print("\nDisabling all motors and disconnecting...")
        bus.disconnect(disable_torque=True)
        print("Done.")


if __name__ == "__main__":
    main()
