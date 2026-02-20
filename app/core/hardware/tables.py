"""App-level motor specification constants.

NOTE: The authoritative motor specs live in lerobot/motors/damiao/tables.py
(DAMIAO_MOTOR_SPECS). These values are for reference only — runtime enforcement
uses robot.get_torque_limits() which reads from DAMIAO_MOTOR_SPECS directly.
"""

# Damiao motor torque limits (Nm) - 10% of max for initial testing safety
# Reference only — not used by SafetyLayer check methods at runtime.
DAMIAO_TORQUE_LIMITS = {
    "J8009P": 3.5,   # 10% of 35Nm max
    "J4340P": 2.7,   # 10% of 27Nm max
    "J4310": 1.25,   # 10% of 12.5Nm max
}
