"""Hardware type definitions for the arm registry system."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional


class MotorType(str, Enum):
    """Supported motor types with their connection protocols"""
    STS3215 = "sts3215"           # Feetech STS3215 - UART TTL
    DAMIAO = "damiao"             # Damiao J-series - CAN-to-serial
    DYNAMIXEL_XL330 = "dynamixel_xl330"  # Dynamixel XL330 - Waveshare USB
    DYNAMIXEL_XL430 = "dynamixel_xl430"  # Dynamixel XL430


class ArmRole(str, Enum):
    """Arm role in teleoperation"""
    LEADER = "leader"
    FOLLOWER = "follower"


class ConnectionStatus(str, Enum):
    """Connection state of an arm"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class ArmDefinition:
    """Definition of a single arm in the registry"""
    id: str
    name: str
    role: ArmRole
    motor_type: MotorType
    port: str
    enabled: bool = True
    structural_design: Optional[str] = None  # e.g., "damiao_7dof", "umbra_7dof"
    config: Dict = field(default_factory=dict)
    calibrated: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "motor_type": self.motor_type.value,
            "port": self.port,
            "enabled": self.enabled,
            "structural_design": self.structural_design,
            "config": self.config,
            "calibrated": self.calibrated,
        }


@dataclass
class Pairing:
    """Leader-follower pairing definition"""
    leader_id: str
    follower_id: str
    name: str

    def to_dict(self) -> Dict:
        return {
            "leader_id": self.leader_id,
            "follower_id": self.follower_id,
            "name": self.name,
        }
