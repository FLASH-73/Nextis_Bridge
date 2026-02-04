"""
Motor Recovery Service

Provides high-level interface for recovering motors from error states
(overload, overheat, etc.) before connection attempts.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MotorStatus:
    """Status of a single motor after recovery attempt."""
    motor_id: int
    motor_name: str
    port: str
    reachable: bool
    error_type: Optional[str]
    recovered: bool
    recommendation: str


class MotorRecoveryService:
    """
    High-level service for recovering motors from error states.

    Provides user-friendly interface for:
    1. Pre-connection diagnostics
    2. Recovery attempts
    3. Clear status reporting
    """

    def __init__(self):
        self.last_recovery_results: dict[str, list[MotorStatus]] = {}

    def attempt_recovery_on_bus(self, bus, port_name: str) -> list[MotorStatus]:
        """
        Attempt recovery on a FeetechMotorsBus.

        Args:
            bus: FeetechMotorsBus instance (port should be open)
            port_name: Human-readable port identifier (e.g., "left_arm")

        Returns:
            List of MotorStatus for each motor on the bus
        """
        results = []

        # Check if port is open, try to open if not
        if not bus.port_handler.is_open:
            try:
                bus.port_handler.openPort()
                bus.set_baudrate(bus.default_baudrate)
                logger.info(f"Opened port {bus.port} for recovery")
            except Exception as e:
                logger.error(f"Cannot open port {bus.port}: {e}")
                # Return all motors as unreachable
                for motor_name, motor in bus.motors.items():
                    results.append(MotorStatus(
                        motor_id=motor.id,
                        motor_name=motor_name,
                        port=port_name,
                        reachable=False,
                        error_type="Port not accessible",
                        recovered=False,
                        recommendation=f"Check USB connection on {bus.port}"
                    ))
                return results

        # Call the low-level recovery method
        try:
            recovery_report = bus.attempt_error_recovery()
        except Exception as e:
            logger.error(f"Recovery failed on {port_name}: {e}")
            for motor_name, motor in bus.motors.items():
                results.append(MotorStatus(
                    motor_id=motor.id,
                    motor_name=motor_name,
                    port=port_name,
                    reachable=False,
                    error_type=f"Recovery failed: {e}",
                    recovered=False,
                    recommendation="Power cycle may be required"
                ))
            return results

        # Process results for each motor
        for motor_name, motor in bus.motors.items():
            report = recovery_report.get(motor.id, {})
            reachable = report.get('reachable', False)
            error_type = report.get('error_type')
            recovered = report.get('recovered', False)

            # Generate recommendation based on error type
            if not reachable:
                rec = "Motor not responding. Check wiring and power."
            elif not recovered and error_type:
                if "Overload" in error_type:
                    rec = "Manually move joint to release tension, then retry."
                elif "Overheating" in error_type:
                    rec = "Wait 2-3 minutes for cooling."
                elif "Voltage" in error_type:
                    rec = "Check power supply voltage (should be 7.4V)."
                elif "Angle Limit" in error_type:
                    rec = "Joint outside calibrated range. Check calibration."
                else:
                    rec = f"Error: {error_type}. Power cycle may be required."
            else:
                rec = "OK"

            results.append(MotorStatus(
                motor_id=motor.id,
                motor_name=motor_name,
                port=port_name,
                reachable=reachable,
                error_type=error_type if not recovered else None,
                recovered=recovered,
                recommendation=rec
            ))

        self.last_recovery_results[port_name] = results
        return results

    def format_status_report(self, results: list[MotorStatus]) -> str:
        """Format recovery results as human-readable report."""
        lines = [
            "",
            "=" * 60,
            "Motor Status Report",
            "=" * 60
        ]

        for s in results:
            if s.recovered:
                icon = " OK "
            elif s.reachable:
                icon = "WARN"
            else:
                icon = "FAIL"

            lines.append(f"[{icon}] {s.motor_name} (ID {s.motor_id}) on {s.port}")
            if s.error_type:
                lines.append(f"       Error: {s.error_type}")
            if s.recommendation != "OK":
                lines.append(f"       Action: {s.recommendation}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def all_recovered(self, results: list[MotorStatus]) -> bool:
        """Check if all motors recovered successfully."""
        return all(
            s.recovered or (s.reachable and s.error_type is None)
            for s in results
        )

    def any_reachable(self, results: list[MotorStatus]) -> bool:
        """Check if at least one motor is reachable."""
        return any(s.reachable for s in results)

    def get_failed_motors(self, results: list[MotorStatus]) -> list[MotorStatus]:
        """Get list of motors that failed recovery."""
        return [s for s in results if not s.recovered and not s.reachable]

    def get_warning_motors(self, results: list[MotorStatus]) -> list[MotorStatus]:
        """Get list of motors with warnings (reachable but have errors)."""
        return [s for s in results if s.reachable and not s.recovered]
