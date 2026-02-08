#!/usr/bin/env python3
"""Test MIT mode position control on link1 only.

Configures link1 in MIT mode with explicit offset detection retry
(single-motor timing fix: motor needs ~1s to report MIT encoder values).

Each phase: hold position for 2s, move +0.1 rad, return.

Usage:
    python test_link1.py
"""
import sys
import time

sys.path.insert(0, "lerobot/src")

from lerobot.motors.damiao.damiao import DamiaoMotorsBusConfig, DamiaoMotorsBus
from lerobot.motors.damiao.tables import DAMIAO_MOTOR_SPECS

RATE_HZ = 60
HOLD_SECONDS = 2
MOVE_SECONDS = 2
MOVE_OFFSET = 0.1  # radians

LINK1_MOTOR = {
    "link1": {"motor_type": "J8009P", "can_id": 0x02, "master_id": 0x12},
}


def main():
    config = DamiaoMotorsBusConfig(
        port="can0",
        motors=LINK1_MOTOR,
        velocity_limit=0.1,
        use_mit_mode=True,
        mit_kp=15.0,
        mit_kd=1.5,
    )

    bus = DamiaoMotorsBus(config)

    try:
        print("=" * 70)
        print("Link1 Isolated Test — MIT Mode (kp=15.0, kd=1.5)")
        print("=" * 70)

        print("\nConnecting to CAN bus...")
        bus.connect()

        print("Configuring link1 in MIT mode...")
        bus.configure()

        # ─── Single-motor offset detection retry ───
        # With 1 motor, configure()'s second-pass MIT probe runs too quickly
        # (~50ms after enable). The motor needs ~1s to start reporting MIT
        # encoder values. With 7 motors this happens naturally; with 1 we wait.
        mit_offset = bus._mit_offsets.get("link1", 0.0)
        print(f"\n  Initial MIT offset from configure(): {mit_offset:+.4f} rad")

        if abs(mit_offset) < 0.1 and "link1" in bus._enabled_motors:
            motor = bus._motors["link1"]
            mcfg = bus._motor_configs["link1"]

            # Read 0xCC position (from configure's first pass)
            bus._control.refresh_motor_status(motor)
            time.sleep(0.02)
            p_cc = motor.getPosition()

            print(f"  0xCC position: {p_cc:+.4f} rad")
            print("  Waiting 1s for MIT encoder to settle...")

            # Send 60 zero-torque MIT probes (1 second at 60Hz)
            for _ in range(60):
                bus._control.control_MIT(
                    motor, p_cc, 0.0,
                    0.0, 0.0, 0.0,
                    mcfg.p_max, mcfg.v_max, mcfg.t_max,
                )
                time.sleep(1 / RATE_HZ)

            p_mit = motor.getPosition()
            offset = p_mit - p_cc
            print(f"  After 1s settle: MIT probe = {p_mit:+.4f}, 0xCC = {p_cc:+.4f}")
            print(f"  Measured offset: {offset:+.4f} rad")

            if abs(offset) > 0.1:
                mit_offset = offset
                bus._mit_offsets["link1"] = mit_offset
                print(f"  DUAL-ENCODER OFFSET DETECTED: {mit_offset:+.4f} rad")
            else:
                print(f"  No significant offset. Encoders appear aligned.")

        print(f"\n  Final MIT offset: {mit_offset:+.4f} rad")

        # ─── Read starting position ───
        positions = bus.sync_read("Present_Position")
        if "link1" not in positions:
            print("\nERROR: link1 not responding to sync_read")
            return

        start_pos = positions["link1"]
        print(f"  Starting user-space position: {start_pos:+.4f} rad")
        print(f"  Estimated MIT position: {start_pos + mit_offset:+.4f} rad")

        # ─── Phase 1: Hold at current position ───
        print(f"\n--- Phase 1: Hold for {HOLD_SECONDS}s ---")
        frames = int(RATE_HZ * HOLD_SECONDS)
        for i in range(frames):
            bus.sync_write("Goal_Position", {"link1": start_pos})
            time.sleep(1 / RATE_HZ)
            if i == 0:
                print("  Motor should be holding firm (no vibration).")

        positions = bus.sync_read("Present_Position")
        hold_pos = positions.get("link1", start_pos)
        hold_drift = abs(hold_pos - start_pos)
        print(f"  Hold drift: {hold_drift:.4f} rad")

        # ─── Phase 2: Move +0.1 rad ───
        target = start_pos + MOVE_OFFSET
        print(f"\n--- Phase 2: Move +{MOVE_OFFSET} rad (target: {target:+.4f}) ---")
        frames = int(RATE_HZ * MOVE_SECONDS)
        for i in range(frames):
            bus.sync_write("Goal_Position", {"link1": target})
            time.sleep(1 / RATE_HZ)

        positions = bus.sync_read("Present_Position")
        move_pos = positions.get("link1", start_pos)
        move_error = abs(move_pos - target)
        print(f"  Actual: {move_pos:+.4f}, Target: {target:+.4f}, Error: {move_error:.4f} rad")

        # ─── Phase 3: Return to start ───
        print(f"\n--- Phase 3: Return to start ---")
        frames = int(RATE_HZ * MOVE_SECONDS)
        for i in range(frames):
            bus.sync_write("Goal_Position", {"link1": start_pos})
            time.sleep(1 / RATE_HZ)

        positions = bus.sync_read("Present_Position")
        return_pos = positions.get("link1", start_pos)
        return_error = abs(return_pos - start_pos)
        print(f"  Actual: {return_pos:+.4f}, Start: {start_pos:+.4f}, Error: {return_error:.4f} rad")

        # ─── Summary ───
        print(f"\n{'=' * 70}")
        print(f"LINK1 TEST SUMMARY")
        print(f"{'=' * 70}")
        print(f"  MIT offset:    {mit_offset:+.4f} rad")
        print(f"  Hold drift:    {hold_drift:.4f} rad  {'PASS' if hold_drift < 0.02 else 'WARN'}")
        print(f"  Move error:    {move_error:.4f} rad  {'PASS' if move_error < 0.1 else 'WARN'}")
        print(f"  Return error:  {return_error:.4f} rad  {'PASS' if return_error < 0.1 else 'WARN'}")

        ok = hold_drift < 0.02 and move_error < 0.1 and return_error < 0.1
        print(f"\n  Overall: {'PASS' if ok else 'WARN — check errors above'}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        print("\nDisabling motor and disconnecting...")
        bus.disconnect(disable_torque=True)
        print("Done.")


if __name__ == "__main__":
    main()
