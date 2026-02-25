"""Hardware abstraction sub-package."""

from app.core.hardware.arm_registry import ArmRegistryService
from app.core.hardware.connection import create_arm_instance
from app.core.hardware.gravity_comp import GravityCompensationService
from app.core.hardware.leader_assist import LeaderAssistService
from app.core.hardware.motor_recovery import MotorRecoveryService, MotorStatus
from app.core.hardware.safety import SafetyLayer
from app.core.hardware.tables import DAMIAO_TORQUE_LIMITS
from app.core.hardware.types import (
    ArmDefinition,
    ArmRole,
    ConnectionStatus,
    MotorType,
    Pairing,
)

__all__ = [
    "MotorType", "ArmRole", "ConnectionStatus", "ArmDefinition", "Pairing",
    "ArmRegistryService", "SafetyLayer", "MotorRecoveryService", "MotorStatus",
    "GravityCompensationService", "LeaderAssistService",
    "create_arm_instance", "DAMIAO_TORQUE_LIMITS",
]
