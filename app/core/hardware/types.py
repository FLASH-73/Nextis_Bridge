"""Hardware type definitions for the arm registry system."""

from dataclasses import dataclass, field
from enum import Enum
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


# ---------------------------------------------------------------------------
# Tool support types (single-actuator devices controlled by triggers)
# ---------------------------------------------------------------------------

class ToolType(str, Enum):
    """Type of tool attached to the system"""
    SCREWDRIVER = "screwdriver"
    GRIPPER = "gripper"
    PUMP = "pump"
    CUSTOM = "custom"


class TriggerType(str, Enum):
    """How a tool is activated"""
    GPIO_SWITCH = "gpio_switch"
    LEADER_BUTTON = "leader_button"
    SOFTWARE = "software"
    CUSTOM = "custom"


@dataclass
class ToolDefinition:
    """Definition of a single-actuator tool"""
    id: str
    name: str
    motor_type: MotorType
    port: str
    motor_id: int
    tool_type: ToolType
    enabled: bool = True
    config: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "motor_type": self.motor_type.value,
            "port": self.port,
            "motor_id": self.motor_id,
            "tool_type": self.tool_type.value,
            "enabled": self.enabled,
            "config": self.config,
        }


@dataclass
class TriggerDefinition:
    """Definition of a trigger input (GPIO switch, button, etc.)"""
    id: str
    name: str
    trigger_type: TriggerType
    port: str
    pin: int
    active_low: bool = True
    enabled: bool = True
    config: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "trigger_type": self.trigger_type.value,
            "port": self.port,
            "pin": self.pin,
            "active_low": self.active_low,
            "enabled": self.enabled,
            "config": self.config,
        }


@dataclass
class ToolPairing:
    """Trigger-to-tool pairing definition"""
    trigger_id: str
    tool_id: str
    name: str
    action: str = "toggle"
    config: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "trigger_id": self.trigger_id,
            "tool_id": self.tool_id,
            "name": self.name,
            "action": self.action,
            "config": self.config,
        }
