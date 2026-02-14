"""App-level motor specification constants.

NOTE: The authoritative motor specs live in lerobot/motors/damiao/tables.py
(DAMIAO_MOTOR_SPECS). These are simplified reference values for the safety layer.
"""

# Damiao motor torque limits (Nm) - 10% of max for initial testing safety
DAMIAO_TORQUE_LIMITS = {
    "J8009P": 3.5,   # 10% of 35Nm max (conservative for testing)
    "J4340P": 0.8,   # 10% of 8Nm max
    "J4310": 0.4,    # 10% of 4Nm max
}
