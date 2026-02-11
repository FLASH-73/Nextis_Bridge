"""
Arm Registry Service - Central management for multi-arm robotics systems.

This service provides:
- Named arm registration with custom identifiers
- Explicit leader-follower pairing (not just left/right)
- Connection status tracking
- Legacy config migration from robot/teleop format
- Port scanning for device discovery
"""

import os
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import serial.tools.list_ports

logger = logging.getLogger(__name__)


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


class ArmRegistryService:
    """
    Central service for arm management.

    Handles:
    - Loading arm definitions from config
    - Managing connections to individual arms
    - Storing leader-follower pairings
    - Legacy config migration
    """

    def __init__(self, config_path: str = "app/config/settings.yaml"):
        self.config_path = Path(config_path)
        self.arms: Dict[str, ArmDefinition] = {}
        self.pairings: List[Pairing] = []
        self.arm_instances: Dict[str, Any] = {}  # Robot/Teleoperator instances
        self.arm_status: Dict[str, ConnectionStatus] = {}
        self._lock = threading.Lock()
        self._config_data: Dict = {}

        self._load_config()

    def _load_config(self):
        """Load configuration from settings.yaml"""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return

        with open(self.config_path, 'r') as f:
            self._config_data = yaml.safe_load(f) or {}

        # Check for new-style arms config
        if "arms" in self._config_data:
            self._load_new_format()
        else:
            # Migrate from legacy format
            self._migrate_legacy_config()

    def _load_new_format(self):
        """Load arms from new-style configuration"""
        arms_config = self._config_data.get("arms", {})

        for arm_id, arm_cfg in arms_config.items():
            try:
                arm = ArmDefinition(
                    id=arm_id,
                    name=arm_cfg.get("name", arm_id),
                    role=ArmRole(arm_cfg.get("role", "follower")),
                    motor_type=MotorType(arm_cfg.get("motor_type", "sts3215")),
                    port=arm_cfg.get("port", ""),
                    enabled=arm_cfg.get("enabled", True),
                    structural_design=arm_cfg.get("structural_design"),
                    config=arm_cfg.get("config", {}),
                    calibrated=arm_cfg.get("calibrated", False),
                )
                self.arms[arm_id] = arm
                self.arm_status[arm_id] = ConnectionStatus.DISCONNECTED
                logger.info(f"Loaded arm: {arm.name} ({arm_id}) - {arm.motor_type.value}")
            except Exception as e:
                logger.error(f"Failed to load arm {arm_id}: {e}")

        # Load pairings
        pairings_config = self._config_data.get("pairings", [])
        for pairing_cfg in pairings_config:
            try:
                pairing = Pairing(
                    leader_id=pairing_cfg["leader"],
                    follower_id=pairing_cfg["follower"],
                    name=pairing_cfg.get("name", f"{pairing_cfg['leader']} -> {pairing_cfg['follower']}"),
                )
                self.pairings.append(pairing)
                logger.info(f"Loaded pairing: {pairing.name}")
            except Exception as e:
                logger.error(f"Failed to load pairing: {e}")

    def _migrate_legacy_config(self):
        """
        Convert old robot/teleop config to new arms format.

        Legacy format:
            robot:
                type: bi_umbra_follower
                left_arm_port: /dev/ttyUSB3
                right_arm_port: /dev/ttyUSB2
            teleop:
                type: bi_umbra_leader
                left_arm_port: /dev/ttyUSB0
                right_arm_port: /dev/ttyUSB1
        """
        robot_cfg = self._config_data.get("robot", {})
        teleop_cfg = self._config_data.get("teleop", {})

        # Migrate follower arms
        robot_type = robot_cfg.get("type", "")

        if robot_type == "bi_umbra_follower":
            # Create left and right follower arms
            if robot_cfg.get("left_arm_port"):
                self.arms["left_follower"] = ArmDefinition(
                    id="left_follower",
                    name="Left Follower",
                    role=ArmRole.FOLLOWER,
                    motor_type=MotorType.STS3215,
                    port=robot_cfg["left_arm_port"],
                    enabled=True,
                    structural_design="umbra_7dof",
                )
                self.arm_status["left_follower"] = ConnectionStatus.DISCONNECTED

            if robot_cfg.get("right_arm_port"):
                self.arms["right_follower"] = ArmDefinition(
                    id="right_follower",
                    name="Right Follower",
                    role=ArmRole.FOLLOWER,
                    motor_type=MotorType.STS3215,
                    port=robot_cfg["right_arm_port"],
                    enabled=True,
                    structural_design="umbra_7dof",
                )
                self.arm_status["right_follower"] = ConnectionStatus.DISCONNECTED

        elif robot_type == "damiao_follower":
            self.arms["damiao_follower"] = ArmDefinition(
                id="damiao_follower",
                name="Damiao Follower",
                role=ArmRole.FOLLOWER,
                motor_type=MotorType.DAMIAO,
                port=robot_cfg.get("port", ""),
                enabled=True,
                structural_design="damiao_7dof",
                config=robot_cfg.get("config", {}),
            )
            self.arm_status["damiao_follower"] = ConnectionStatus.DISCONNECTED

        # Migrate leader arms
        teleop_type = teleop_cfg.get("type", "")

        if teleop_type == "bi_umbra_leader":
            if teleop_cfg.get("left_arm_port"):
                self.arms["left_leader"] = ArmDefinition(
                    id="left_leader",
                    name="Left Leader",
                    role=ArmRole.LEADER,
                    motor_type=MotorType.STS3215,  # Could be XL330 depending on setup
                    port=teleop_cfg["left_arm_port"],
                    enabled=True,
                    structural_design="umbra_7dof",
                )
                self.arm_status["left_leader"] = ConnectionStatus.DISCONNECTED

            if teleop_cfg.get("right_arm_port"):
                self.arms["right_leader"] = ArmDefinition(
                    id="right_leader",
                    name="Right Leader",
                    role=ArmRole.LEADER,
                    motor_type=MotorType.STS3215,
                    port=teleop_cfg["right_arm_port"],
                    enabled=True,
                    structural_design="umbra_7dof",
                )
                self.arm_status["right_leader"] = ConnectionStatus.DISCONNECTED

        # Create default side-based pairings
        if "left_leader" in self.arms and "left_follower" in self.arms:
            self.pairings.append(Pairing(
                leader_id="left_leader",
                follower_id="left_follower",
                name="Left Pair",
            ))
        if "right_leader" in self.arms and "right_follower" in self.arms:
            self.pairings.append(Pairing(
                leader_id="right_leader",
                follower_id="right_follower",
                name="Right Pair",
            ))

        logger.info(f"Migrated legacy config: {len(self.arms)} arms, {len(self.pairings)} pairings")

    def get_all_arms(self) -> List[Dict]:
        """Return all arms with their current status"""
        result = []
        for arm_id, arm in self.arms.items():
            arm_dict = arm.to_dict()
            arm_dict["status"] = self.arm_status.get(arm_id, ConnectionStatus.DISCONNECTED).value
            result.append(arm_dict)
        return result

    def get_arm(self, arm_id: str) -> Optional[Dict]:
        """Get a specific arm by ID"""
        if arm_id not in self.arms:
            return None
        arm = self.arms[arm_id]
        arm_dict = arm.to_dict()
        arm_dict["status"] = self.arm_status.get(arm_id, ConnectionStatus.DISCONNECTED).value
        return arm_dict

    def get_leaders(self) -> List[Dict]:
        """Return all leader arms"""
        return [a for a in self.get_all_arms() if a["role"] == "leader"]

    def get_followers(self) -> List[Dict]:
        """Return all follower arms"""
        return [a for a in self.get_all_arms() if a["role"] == "follower"]

    def get_pairings(self) -> List[Dict]:
        """Return all leader-follower pairings"""
        return [p.to_dict() for p in self.pairings]

    def get_active_pairings(self, active_arm_ids: Optional[List[str]] = None) -> List[Dict]:
        """Return pairings where both arms are in the active selection"""
        if active_arm_ids is None:
            return self.get_pairings()

        result = []
        for pairing in self.pairings:
            if pairing.leader_id in active_arm_ids and pairing.follower_id in active_arm_ids:
                result.append(pairing.to_dict())
        return result

    def create_pairing(self, leader_id: str, follower_id: str, name: Optional[str] = None) -> Dict:
        """
        Create a new leader-follower pairing.
        Returns success status and optional compatibility warning.
        """
        # Validate arms exist
        if leader_id not in self.arms:
            return {"success": False, "error": f"Leader arm '{leader_id}' not found"}
        if follower_id not in self.arms:
            return {"success": False, "error": f"Follower arm '{follower_id}' not found"}

        leader = self.arms[leader_id]
        follower = self.arms[follower_id]

        # Validate roles
        if leader.role != ArmRole.LEADER:
            return {"success": False, "error": f"'{leader_id}' is not a leader arm"}
        if follower.role != ArmRole.FOLLOWER:
            return {"success": False, "error": f"'{follower_id}' is not a follower arm"}

        # Check for existing pairing
        for p in self.pairings:
            if p.leader_id == leader_id and p.follower_id == follower_id:
                return {"success": False, "error": "Pairing already exists"}

        # Check structural compatibility (soft warning only)
        warning = None
        if leader.structural_design and follower.structural_design:
            if leader.structural_design != follower.structural_design:
                warning = f"Structural mismatch: {leader.name} ({leader.structural_design}) may not match {follower.name} ({follower.structural_design})"

        # Create pairing
        pairing_name = name or f"{leader.name} -> {follower.name}"
        pairing = Pairing(leader_id=leader_id, follower_id=follower_id, name=pairing_name)
        self.pairings.append(pairing)

        # Save to config
        self._save_config()

        return {"success": True, "warning": warning, "pairing": pairing.to_dict()}

    def remove_pairing(self, leader_id: str, follower_id: str) -> Dict:
        """Remove an existing pairing"""
        for i, p in enumerate(self.pairings):
            if p.leader_id == leader_id and p.follower_id == follower_id:
                self.pairings.pop(i)
                self._save_config()
                return {"success": True}
        return {"success": False, "error": "Pairing not found"}

    def add_arm(self, arm_data: Dict) -> Dict:
        """Add a new arm to the registry"""
        arm_id = arm_data.get("id")
        if not arm_id:
            return {"success": False, "error": "Arm ID is required"}
        if arm_id in self.arms:
            return {"success": False, "error": f"Arm '{arm_id}' already exists"}

        try:
            arm = ArmDefinition(
                id=arm_id,
                name=arm_data.get("name", arm_id),
                role=ArmRole(arm_data.get("role", "follower")),
                motor_type=MotorType(arm_data.get("motor_type", "sts3215")),
                port=arm_data.get("port", ""),
                enabled=arm_data.get("enabled", True),
                structural_design=arm_data.get("structural_design"),
                config=arm_data.get("config", {}),
            )
            self.arms[arm_id] = arm
            self.arm_status[arm_id] = ConnectionStatus.DISCONNECTED
            self._save_config()
            return {"success": True, "arm": arm.to_dict()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_arm(self, arm_id: str, **kwargs) -> Dict:
        """Update arm properties"""
        if arm_id not in self.arms:
            return {"success": False, "error": f"Arm '{arm_id}' not found"}

        arm = self.arms[arm_id]

        # Update allowed fields
        if "name" in kwargs:
            arm.name = kwargs["name"]
        if "port" in kwargs:
            arm.port = kwargs["port"]
        if "enabled" in kwargs:
            arm.enabled = kwargs["enabled"]
        if "structural_design" in kwargs:
            arm.structural_design = kwargs["structural_design"]
        if "config" in kwargs:
            arm.config.update(kwargs["config"])

        self._save_config()
        return {"success": True, "arm": arm.to_dict()}

    def remove_arm(self, arm_id: str) -> Dict:
        """Remove an arm from the registry"""
        if arm_id not in self.arms:
            return {"success": False, "error": f"Arm '{arm_id}' not found"}

        # Disconnect if connected
        if self.arm_status.get(arm_id) == ConnectionStatus.CONNECTED:
            self.disconnect_arm(arm_id)

        # Remove from pairings
        self.pairings = [p for p in self.pairings if p.leader_id != arm_id and p.follower_id != arm_id]

        # Remove arm
        del self.arms[arm_id]
        del self.arm_status[arm_id]
        if arm_id in self.arm_instances:
            del self.arm_instances[arm_id]

        self._save_config()
        return {"success": True}

    def connect_arm(self, arm_id: str) -> Dict:
        """
        Connect a specific arm.
        Creates the appropriate robot/teleoperator instance based on motor type.
        """
        if arm_id not in self.arms:
            return {"success": False, "error": f"Arm '{arm_id}' not found"}

        arm = self.arms[arm_id]

        if not arm.enabled:
            return {"success": False, "error": f"Arm '{arm_id}' is disabled"}

        self.arm_status[arm_id] = ConnectionStatus.CONNECTING

        try:
            instance = self._create_arm_instance(arm)
            if instance:
                self.arm_instances[arm_id] = instance
                self.arm_status[arm_id] = ConnectionStatus.CONNECTED
                logger.info(f"Connected arm: {arm.name} ({arm_id})")
                return {"success": True, "status": "connected"}
            else:
                self.arm_status[arm_id] = ConnectionStatus.ERROR
                return {"success": False, "error": "Failed to create arm instance"}
        except Exception as e:
            self.arm_status[arm_id] = ConnectionStatus.ERROR
            logger.error(f"Failed to connect arm {arm_id}: {e}")
            return {"success": False, "error": str(e)}

    def disconnect_arm(self, arm_id: str) -> Dict:
        """Disconnect a specific arm"""
        if arm_id not in self.arms:
            return {"success": False, "error": f"Arm '{arm_id}' not found"}

        if arm_id in self.arm_instances:
            try:
                instance = self.arm_instances[arm_id]
                if hasattr(instance, 'disconnect'):
                    instance.disconnect()
                del self.arm_instances[arm_id]
            except Exception as e:
                logger.error(f"Error disconnecting arm {arm_id}: {e}")

        self.arm_status[arm_id] = ConnectionStatus.DISCONNECTED
        logger.info(f"Disconnected arm: {arm_id}")
        return {"success": True, "status": "disconnected"}

    def _create_arm_instance(self, arm: ArmDefinition) -> Any:
        """
        Create the appropriate robot/teleoperator instance based on motor type.
        This is a factory method that returns the correct hardware interface.
        """
        # Import here to avoid circular imports
        if arm.motor_type == MotorType.STS3215:
            if arm.role == ArmRole.FOLLOWER:
                from lerobot.robots.umbra_follower import UmbraFollowerRobot
                from lerobot.robots.umbra_follower.config_umbra_follower import UmbraFollowerConfig
                config = UmbraFollowerConfig(
                    id=arm.id,
                    port=arm.port,
                    cameras={},  # No cameras for individual arms
                )
                robot = UmbraFollowerRobot(config)
                robot.connect(calibrate=False)
                return robot
            else:
                # Leader arm - use LeaderArm class
                from lerobot.teleoperators.umbra_leader import UmbraLeader
                from lerobot.teleoperators.umbra_leader.config_umbra_leader import UmbraLeaderConfig
                config = UmbraLeaderConfig(
                    id=arm.id,
                    port=arm.port,
                )
                leader = UmbraLeader(config)
                leader.connect(calibrate=False)
                return leader

        elif arm.motor_type == MotorType.DAMIAO:
            if arm.role == ArmRole.FOLLOWER:
                from lerobot.robots.damiao_follower import DamiaoFollowerRobot
                from lerobot.robots.damiao_follower.config_damiao_follower import DamiaoFollowerConfig
                config = DamiaoFollowerConfig(
                    id=arm.id,
                    port=arm.port,
                    velocity_limit=arm.config.get("velocity_limit", 0.3),
                    cameras={},
                )
                robot = DamiaoFollowerRobot(config)
                robot.connect()
                return robot
            else:
                # Damiao leader not yet implemented
                logger.warning(f"Damiao leader arms not yet supported")
                return None

        elif arm.motor_type in [MotorType.DYNAMIXEL_XL330, MotorType.DYNAMIXEL_XL430]:
            if arm.role == ArmRole.LEADER:
                # Dynamixel XL330 leader arm (Waveshare USB-C bus)
                from lerobot.teleoperators.dynamixel_leader import DynamixelLeader
                from lerobot.teleoperators.dynamixel_leader.config_dynamixel_leader import DynamixelLeaderConfig
                config = DynamixelLeaderConfig(
                    id=arm.id,
                    port=arm.port,
                    structural_design=arm.structural_design or "",
                )
                leader = DynamixelLeader(config)
                leader.connect(calibrate=False)
                return leader
            else:
                logger.warning(f"Dynamixel follower arms not typical use case")
                return None

        logger.warning(f"Unknown motor type: {arm.motor_type}")
        return None

    def scan_ports(self) -> List[Dict]:
        """
        Scan for available serial ports and CAN interfaces.
        Returns list of ports with basic info.
        Filters out virtual ports (ttyS*) that have no real hardware.
        """
        ports = []
        for port in serial.tools.list_ports.comports():
            # Skip virtual serial ports with no hardware (ttyS* with no VID/PID)
            if port.vid is None and port.pid is None:
                # Skip ports with no hardware info (virtual/legacy ports)
                continue

            port_info = {
                "device": port.device,
                "name": port.name,
                "description": port.description,
                "hwid": port.hwid,
                "manufacturer": port.manufacturer,
                "product": port.product,
                "serial_number": port.serial_number,
                "vid": port.vid,
                "pid": port.pid,
                "in_use": self._is_port_in_use(port.device),
            }
            ports.append(port_info)

        # Detect CAN interfaces (for Damiao motors via SocketCAN)
        import os
        try:
            for iface in sorted(os.listdir('/sys/class/net')):
                if iface.startswith('can'):
                    state = 'unknown'
                    try:
                        with open(f'/sys/class/net/{iface}/operstate') as f:
                            state = f.read().strip()
                    except Exception:
                        pass

                    # Only show CAN interfaces that are UP
                    if state != 'up':
                        continue

                    ports.append({
                        "device": iface,
                        "name": iface,
                        "description": f"CAN Bus Interface ({state})",
                        "hwid": "socketcan",
                        "manufacturer": "SocketCAN",
                        "product": "CAN Interface",
                        "serial_number": None,
                        "vid": None,
                        "pid": None,
                        "in_use": self._is_port_in_use(iface),
                    })
        except OSError:
            pass

        # Sort by device name for consistent ordering
        ports.sort(key=lambda p: p["device"])
        return ports

    def _is_port_in_use(self, port: str) -> bool:
        """Check if a port is already assigned to an arm"""
        for arm in self.arms.values():
            if arm.port == port:
                return True
        return False

    def scan_motors(self, port: str, motor_type: str) -> Dict:
        """
        Scan a port for connected motors and return their IDs and model info.
        Used for motor ID configuration (connect one motor at a time).

        Uses broadcast_ping which is efficient and scans all IDs at once.
        Also scans multiple baud rates since motors can have different settings.

        Args:
            port: Serial port path (e.g., /dev/ttyACM0)
            motor_type: Type of motor ("dynamixel_xl330", "dynamixel_xl430", "sts3215")

        Returns:
            Dictionary with "found_ids" list, "motor_info" dict, and any errors
        """
        # Model number to model string mapping for Dynamixel motors
        DYNAMIXEL_MODEL_MAP = {
            1190: "xl330-m077",  # XL330 lower torque (0.077 Nm)
            1200: "xl330-m288",  # XL330 higher torque (0.288 Nm)
            1060: "xl430-w250",  # XL430
            1020: "xm430-w350",
            1120: "xm540-w270",
            1070: "xc430-w150",
        }

        try:
            # CAN interfaces only work with Damiao motors
            if port.startswith('can') and motor_type != 'damiao':
                return {"success": False, "error": f"CAN interfaces only work with Damiao motors, not {motor_type}"}

            if motor_type in ["dynamixel_xl330", "dynamixel_xl430"]:
                from lerobot.motors.dynamixel import DynamixelMotorsBus

                logger.info(f"Scanning for Dynamixel motors on {port}...")

                # Custom scan to get model numbers (scan_port only returns IDs)
                bus = DynamixelMotorsBus(port=port, motors={})
                bus._connect(handshake=False)

                found_motors = {}  # {id: {model_number, model_name, baudrate}}
                baudrate_ids = {}

                HARDWARE_ERROR_STATUS_ADDR = 70

                # Scan XL330-relevant baud rates only (57600 = factory default, 1M = common config)
                for baudrate in [57600, 1_000_000]:
                    bus.set_baudrate(baudrate)
                    id_model = bus.broadcast_ping(num_retry=2)
                    if id_model:
                        # Only show IDs 0-20 (typical arm range)
                        id_model = {k: v for k, v in id_model.items() if k <= 20}
                    if id_model:
                        baudrate_ids[baudrate] = list(id_model.keys())
                        for motor_id, model_number in id_model.items():
                            if motor_id not in found_motors:
                                model_name = DYNAMIXEL_MODEL_MAP.get(model_number, f"unknown-{model_number}")

                                # Check for hardware errors
                                error_status = 0
                                has_error = False
                                error_names = []
                                try:
                                    error_status, _, _ = bus.packet_handler.read1ByteTxRx(
                                        bus.port_handler, motor_id, HARDWARE_ERROR_STATUS_ADDR
                                    )
                                    if error_status != 0:
                                        has_error = True
                                        if error_status & 0x01: error_names.append("Input Voltage")
                                        if error_status & 0x04: error_names.append("Overheating")
                                        if error_status & 0x08: error_names.append("Motor Encoder")
                                        if error_status & 0x10: error_names.append("Electrical Shock")
                                        if error_status & 0x20: error_names.append("Overload")
                                except Exception:
                                    pass

                                found_motors[motor_id] = {
                                    "model_number": model_number,
                                    "model_name": model_name,
                                    "baudrate": baudrate,
                                    "has_error": has_error,
                                    "error_status": error_status,
                                    "error_names": error_names
                                }

                                if has_error:
                                    logger.warning(f"Found motor: ID={motor_id}, model={model_name} ({model_number}) at {baudrate} baud - HAS HARDWARE ERROR: {', '.join(error_names)}")
                                else:
                                    logger.info(f"Found motor: ID={motor_id}, model={model_name} ({model_number}) at {baudrate} baud")

                bus.port_handler.closePort()

                found_ids = sorted(found_motors.keys())

                return {
                    "success": True,
                    "found_ids": found_ids,
                    "motor_info": found_motors,
                    "baudrate_info": baudrate_ids
                }

            elif motor_type == "sts3215":
                from lerobot.motors.feetech import FeetechMotorsBus

                logger.info(f"Scanning for Feetech motors on {port}...")
                baudrate_ids = FeetechMotorsBus.scan_port(port)

                all_found_ids = []
                for baudrate, ids in baudrate_ids.items():
                    all_found_ids.extend(ids)
                    logger.info(f"Found motors at {baudrate} baud: {ids}")

                found_ids = list(dict.fromkeys(all_found_ids))

                return {
                    "success": True,
                    "found_ids": found_ids,
                    "baudrate_info": baudrate_ids
                }

            elif motor_type == "damiao":
                logger.info(f"Scanning for Damiao motors on {port}...")

                if not port.startswith('can'):
                    return {"success": False, "error": f"Damiao motors require a CAN interface (e.g. can0), got: {port}"}

                try:
                    import can
                except ImportError:
                    return {"success": False, "error": "python-can package not installed. Run: pip install python-can"}

                import struct
                import time

                bus = can.interface.Bus(channel=port, bustype='socketcan', bitrate=1000000)

                found_motors = {}
                try:
                    for motor_id in range(1, 8):
                        # Send enable command to probe if motor exists
                        enable_data = bytes([0xFF] * 7 + [0xFC])
                        msg = can.Message(arbitration_id=motor_id, data=enable_data, is_extended_id=False)
                        try:
                            bus.send(msg)
                        except can.CanError:
                            continue

                        # Wait for response
                        deadline = time.time() + 0.3
                        responded = False
                        while time.time() < deadline:
                            resp = bus.recv(timeout=0.1)
                            if resp and resp.data and len(resp.data) >= 1:
                                esc_id = resp.data[0] & 0x0F
                                if esc_id == motor_id:
                                    responded = True
                                    break

                        if responded:
                            # Immediately disable the motor we just enabled
                            disable_data = bytes([0xFF] * 7 + [0xFD])
                            disable_msg = can.Message(arbitration_id=motor_id, data=disable_data, is_extended_id=False)
                            try:
                                bus.send(disable_msg)
                            except can.CanError:
                                pass

                            found_motors[motor_id] = {
                                "model_name": "damiao",
                                "has_error": False,
                                "error_status": 0,
                                "error_names": [],
                            }
                            logger.info(f"Found Damiao motor at CAN ID {motor_id}")

                finally:
                    bus.shutdown()

                found_ids = sorted(found_motors.keys())
                logger.info(f"Damiao scan complete: found {len(found_ids)} motors: {found_ids}")

                return {
                    "success": True,
                    "found_ids": found_ids,
                    "motor_info": found_motors,
                }

            else:
                return {"success": False, "error": f"Unsupported motor type: {motor_type}"}

        except Exception as e:
            logger.error(f"Error scanning motors on {port}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    def set_motor_id(self, port: str, motor_type: str, current_id: int, new_id: int) -> Dict:
        """
        Change a motor's ID from current_id to new_id.
        IMPORTANT: Only ONE motor should be connected when running this!

        Auto-detects the motor model (e.g., xl330-m077 vs xl330-m288) before
        setting the ID to handle different motor variants.

        Args:
            port: Serial port path
            motor_type: Type of motor
            current_id: Current motor ID (or 1 for factory default)
            new_id: New ID to set (1-253)

        Returns:
            Dictionary with success status and any errors
        """
        # Model number to model string mapping for Dynamixel motors
        DYNAMIXEL_MODEL_MAP = {
            1190: "xl330-m077",  # XL330 lower torque (0.077 Nm)
            1200: "xl330-m288",  # XL330 higher torque (0.288 Nm)
            1060: "xl430-w250",  # XL430
            1020: "xm430-w350",
            1120: "xm540-w270",
            1070: "xc430-w150",
        }

        try:
            if new_id < 1 or new_id > 253:
                return {"success": False, "error": "Motor ID must be between 1 and 253"}

            if motor_type in ["dynamixel_xl330", "dynamixel_xl430"]:
                import time
                from lerobot.motors.dynamixel import DynamixelMotorsBus
                from lerobot.motors import Motor, MotorNormMode

                # Step 1: Auto-detect the motor model by scanning
                logger.info(f"Auto-detecting motor model on {port}...")
                scan_bus = DynamixelMotorsBus(port=port, motors={})
                scan_bus._connect(handshake=False)

                detected_model = None
                detected_model_number = None
                detected_baudrate = 57600

                # Try common baud rates (57600 is factory default for XL330)
                for baudrate in [57600, 1_000_000, 115200]:
                    scan_bus.set_baudrate(baudrate)
                    id_model = scan_bus.broadcast_ping()
                    if id_model and current_id in id_model:
                        detected_model_number = id_model[current_id]
                        detected_model = DYNAMIXEL_MODEL_MAP.get(detected_model_number)
                        detected_baudrate = baudrate
                        logger.info(f"Detected motor: ID={current_id}, model_number={detected_model_number}, model={detected_model} at {baudrate} baud")
                        break

                if not detected_model:
                    scan_bus.port_handler.closePort()
                    # Fallback based on motor_type if detection failed
                    detected_model = "xl330-m288" if motor_type == "dynamixel_xl330" else "xl430-w250"
                    logger.warning(f"Could not detect motor model, falling back to {detected_model}")
                else:
                    # Step 2: Check for hardware errors and clear them if needed
                    HARDWARE_ERROR_STATUS_ADDR = 70
                    scan_bus.set_baudrate(detected_baudrate)

                    # Read hardware error status
                    error_status, comm, _ = scan_bus.packet_handler.read1ByteTxRx(
                        scan_bus.port_handler, current_id, HARDWARE_ERROR_STATUS_ADDR
                    )

                    if error_status != 0:
                        # Decode error bits
                        error_names = []
                        if error_status & 0x01: error_names.append("Input Voltage")
                        if error_status & 0x04: error_names.append("Overheating")
                        if error_status & 0x08: error_names.append("Motor Encoder")
                        if error_status & 0x10: error_names.append("Electrical Shock")
                        if error_status & 0x20: error_names.append("Overload")

                        logger.warning(f"Motor has hardware error (status=0x{error_status:02X}): {', '.join(error_names)}")
                        logger.info(f"Rebooting motor to clear error...")

                        # Reboot motor to clear error flags
                        scan_bus.packet_handler.reboot(scan_bus.port_handler, current_id)
                        time.sleep(1.0)  # Wait for motor to reboot

                        # Verify error is cleared
                        error_status_after, _, _ = scan_bus.packet_handler.read1ByteTxRx(
                            scan_bus.port_handler, current_id, HARDWARE_ERROR_STATUS_ADDR
                        )
                        if error_status_after != 0:
                            logger.warning(f"Hardware error persists after reboot (status=0x{error_status_after:02X})")
                        else:
                            logger.info("Hardware error cleared successfully")

                    scan_bus.port_handler.closePort()

                # Step 3: Create bus with detected model and set the ID
                bus = DynamixelMotorsBus(
                    port=port,
                    motors={"target_motor": Motor(new_id, detected_model, MotorNormMode.DEGREES)}
                )

                logger.info(f"Setting up {detected_model} motor to ID {new_id}...")
                bus.setup_motor("target_motor", initial_id=current_id)

                bus.port_handler.closePort()
                logger.info(f"Changed {detected_model} motor ID from {current_id} to {new_id}")
                return {
                    "success": True,
                    "new_id": new_id,
                    "detected_model": detected_model,
                    "model_number": detected_model_number
                }

            elif motor_type == "sts3215":
                from lerobot.motors.feetech import FeetechMotorsBus
                from lerobot.motors import Motor, MotorNormMode

                # Create bus with motor defined at TARGET ID
                bus = FeetechMotorsBus(
                    port=port,
                    motors={"target_motor": Motor(new_id, "sts3215", MotorNormMode.DEGREES)}
                )

                logger.info(f"Setting up STS3215 motor to ID {new_id}...")
                bus.setup_motor("target_motor", initial_id=current_id)

                bus.port_handler.closePort()
                logger.info(f"Changed STS3215 motor ID from {current_id} to {new_id}")
                return {"success": True, "new_id": new_id}

            else:
                return {"success": False, "error": f"Unsupported motor type: {motor_type}"}

        except Exception as e:
            logger.error(f"Error setting motor ID: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    def recover_motor(self, port: str, motor_type: str) -> Dict:
        """
        Attempt to recover a Dynamixel motor that is unresponsive or in error state.

        Recovery steps (in order):
        1. Scan all baud rates with broadcast_ping
        2. If found: read error status, reboot to clear errors
        3. If not found: try reboot at ID=1 (factory default) on all baud rates
        4. If still not found: try factory reset at ID=1 on all baud rates
        5. Final scan to verify recovery

        Returns a detailed log of each step and the result.
        """
        import time

        DYNAMIXEL_MODEL_MAP = {
            1190: "xl330-m077",
            1200: "xl330-m288",
            1060: "xl430-w250",
            1020: "xm430-w350",
            1120: "xm540-w270",
            1070: "xc430-w150",
        }

        log = []
        recovered = False
        found_motor = None

        if motor_type not in ["dynamixel_xl330", "dynamixel_xl430"]:
            return {"success": False, "error": "Recovery is only supported for Dynamixel motors", "log": []}

        try:
            from lerobot.motors.dynamixel import DynamixelMotorsBus

            bus = DynamixelMotorsBus(port=port, motors={})
            bus._connect(handshake=False)

            HARDWARE_ERROR_STATUS_ADDR = 70
            SCAN_BAUDRATES = [57600, 1_000_000, 115200, 9600, 2_000_000, 3_000_000, 4_000_000]

            # --- Step 1: Scan all baud rates with broadcast_ping ---
            log.append({"step": 1, "action": "Scanning all baud rates with broadcast ping...", "status": "running"})
            logger.info("Recovery Step 1: Scanning all baud rates...")

            for baudrate in SCAN_BAUDRATES:
                bus.set_baudrate(baudrate)
                id_model = bus.broadcast_ping()
                if id_model:
                    for motor_id, model_number in id_model.items():
                        model_name = DYNAMIXEL_MODEL_MAP.get(model_number, f"unknown-{model_number}")
                        found_motor = {
                            "id": motor_id,
                            "model_number": model_number,
                            "model_name": model_name,
                            "baudrate": baudrate,
                        }
                        log[-1]["status"] = "success"
                        log[-1]["detail"] = f"Found motor ID={motor_id}, model={model_name} at {baudrate} baud"
                        logger.info(f"Recovery: Found motor ID={motor_id}, model={model_name} at {baudrate} baud")
                        break
                    if found_motor:
                        break

            if not found_motor:
                log[-1]["status"] = "failed"
                log[-1]["detail"] = "No motors found on any baud rate"

            # --- Step 2: If found, check error and reboot ---
            if found_motor:
                log.append({"step": 2, "action": "Checking hardware error status...", "status": "running"})
                bus.set_baudrate(found_motor["baudrate"])

                try:
                    error_status, _, _ = bus.packet_handler.read1ByteTxRx(
                        bus.port_handler, found_motor["id"], HARDWARE_ERROR_STATUS_ADDR
                    )
                    if error_status != 0:
                        error_names = []
                        if error_status & 0x01: error_names.append("Input Voltage")
                        if error_status & 0x04: error_names.append("Overheating")
                        if error_status & 0x08: error_names.append("Motor Encoder")
                        if error_status & 0x10: error_names.append("Electrical Shock")
                        if error_status & 0x20: error_names.append("Overload")

                        log[-1]["detail"] = f"Hardware error detected: {', '.join(error_names)} (0x{error_status:02X})"
                        log.append({"step": 2, "action": "Rebooting motor to clear error...", "status": "running"})

                        bus.packet_handler.reboot(bus.port_handler, found_motor["id"])
                        time.sleep(1.5)

                        # Verify reboot
                        bus.set_baudrate(found_motor["baudrate"])
                        id_model = bus.broadcast_ping()
                        if id_model and found_motor["id"] in id_model:
                            error_after, _, _ = bus.packet_handler.read1ByteTxRx(
                                bus.port_handler, found_motor["id"], HARDWARE_ERROR_STATUS_ADDR
                            )
                            if error_after == 0:
                                log[-1]["status"] = "success"
                                log[-1]["detail"] = "Reboot successful - hardware error cleared"
                                recovered = True
                            else:
                                log[-1]["status"] = "warning"
                                log[-1]["detail"] = f"Reboot completed but error persists (0x{error_after:02X})"
                                recovered = True  # Motor is at least responding
                        else:
                            log[-1]["status"] = "failed"
                            log[-1]["detail"] = "Motor not responding after reboot"
                    else:
                        log[-1]["status"] = "success"
                        log[-1]["detail"] = "No hardware errors detected - motor is healthy"
                        recovered = True
                except Exception as e:
                    log[-1]["status"] = "failed"
                    log[-1]["detail"] = f"Error reading status: {str(e)}"

            # --- Step 3: If not found, try reboot at ID=1 on all baud rates ---
            if not found_motor:
                log.append({"step": 3, "action": "Trying reboot at factory default ID=1...", "status": "running"})
                logger.info("Recovery Step 3: Trying reboot at ID=1...")

                rebooted = False
                for baudrate in SCAN_BAUDRATES:
                    bus.set_baudrate(baudrate)
                    result, error = bus.packet_handler.reboot(bus.port_handler, 1)
                    if bus._is_comm_success(result):
                        log[-1]["status"] = "success"
                        log[-1]["detail"] = f"Reboot command sent at {baudrate} baud - waiting for motor..."
                        rebooted = True
                        time.sleep(1.5)
                        break

                if not rebooted:
                    log[-1]["status"] = "failed"
                    log[-1]["detail"] = "Reboot command failed on all baud rates"

                # Check if motor came back
                if rebooted:
                    log.append({"step": 3, "action": "Scanning for motor after reboot...", "status": "running"})
                    for baudrate in SCAN_BAUDRATES:
                        bus.set_baudrate(baudrate)
                        id_model = bus.broadcast_ping()
                        if id_model:
                            for motor_id, model_number in id_model.items():
                                model_name = DYNAMIXEL_MODEL_MAP.get(model_number, f"unknown-{model_number}")
                                found_motor = {
                                    "id": motor_id,
                                    "model_number": model_number,
                                    "model_name": model_name,
                                    "baudrate": baudrate,
                                }
                                break
                            if found_motor:
                                log[-1]["status"] = "success"
                                log[-1]["detail"] = f"Motor recovered! ID={found_motor['id']}, model={found_motor['model_name']} at {baudrate} baud"
                                recovered = True
                                break

                    if not found_motor:
                        log[-1]["status"] = "failed"
                        log[-1]["detail"] = "Motor still not responding after reboot"

            # --- Step 4: If still not found, try factory reset at ID=1 ---
            if not found_motor:
                log.append({"step": 4, "action": "Trying factory reset (keeps ID and baud rate)...", "status": "running"})
                logger.info("Recovery Step 4: Trying factory reset at ID=1...")

                reset_sent = False
                for baudrate in SCAN_BAUDRATES:
                    bus.set_baudrate(baudrate)
                    # Option 0x02 = reset all except ID and baudrate
                    result, error = bus.packet_handler.factoryReset(bus.port_handler, 1, 0x02)
                    if bus._is_comm_success(result):
                        log[-1]["status"] = "success"
                        log[-1]["detail"] = f"Factory reset sent at {baudrate} baud - waiting..."
                        reset_sent = True
                        time.sleep(2.0)
                        break

                if not reset_sent:
                    # Try more aggressive: reset all except ID only
                    log[-1]["status"] = "warning"
                    log[-1]["detail"] = "Conservative reset failed, trying full reset (keeps only ID)..."
                    log.append({"step": 4, "action": "Trying full factory reset (keeps only ID)...", "status": "running"})

                    for baudrate in SCAN_BAUDRATES:
                        bus.set_baudrate(baudrate)
                        result, error = bus.packet_handler.factoryReset(bus.port_handler, 1, 0x01)
                        if bus._is_comm_success(result):
                            log[-1]["status"] = "success"
                            log[-1]["detail"] = f"Full reset sent at {baudrate} baud - baud rate may have changed to 57600"
                            reset_sent = True
                            time.sleep(2.0)
                            break

                if not reset_sent:
                    log[-1]["status"] = "failed"
                    log[-1]["detail"] = "Factory reset failed on all baud rates"

                # Check if motor came back after factory reset
                if reset_sent:
                    log.append({"step": 4, "action": "Scanning for motor after factory reset...", "status": "running"})
                    for baudrate in SCAN_BAUDRATES:
                        bus.set_baudrate(baudrate)
                        id_model = bus.broadcast_ping()
                        if id_model:
                            for motor_id, model_number in id_model.items():
                                model_name = DYNAMIXEL_MODEL_MAP.get(model_number, f"unknown-{model_number}")
                                found_motor = {
                                    "id": motor_id,
                                    "model_number": model_number,
                                    "model_name": model_name,
                                    "baudrate": baudrate,
                                }
                                break
                            if found_motor:
                                log[-1]["status"] = "success"
                                log[-1]["detail"] = f"Motor recovered! ID={found_motor['id']}, model={found_motor['model_name']} at {baudrate} baud"
                                recovered = True
                                break

                    if not found_motor:
                        log[-1]["status"] = "failed"
                        log[-1]["detail"] = "Motor still not responding after factory reset"

            # --- Step 5: Final verification scan ---
            if recovered and found_motor:
                log.append({"step": 5, "action": "Final verification...", "status": "running"})
                bus.set_baudrate(found_motor["baudrate"])
                id_model = bus.broadcast_ping()
                if id_model and found_motor["id"] in id_model:
                    log[-1]["status"] = "success"
                    log[-1]["detail"] = f"Motor verified: ID={found_motor['id']}, model={found_motor['model_name']}"
                else:
                    log[-1]["status"] = "warning"
                    log[-1]["detail"] = "Final verification ping failed - motor may need power cycle"
                    recovered = False

            bus.port_handler.closePort()

            # Summary
            if recovered:
                return {
                    "success": True,
                    "recovered": True,
                    "motor": found_motor,
                    "log": log,
                    "message": f"Motor recovered successfully! ID={found_motor['id']}, model={found_motor['model_name']}"
                }
            else:
                return {
                    "success": True,
                    "recovered": False,
                    "motor": found_motor,
                    "log": log,
                    "message": "Recovery failed. Motor may need Dynamixel Wizard 2.0 firmware recovery or may be physically damaged."
                }

        except Exception as e:
            logger.error(f"Error during motor recovery: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "recovered": False,
                "log": log,
                "error": str(e),
                "message": f"Recovery error: {str(e)}"
            }

    def get_compatible_followers(self, leader_id: str) -> List[Dict]:
        """
        Get followers that are structurally compatible with a leader.
        Returns all followers but flags compatibility.
        """
        if leader_id not in self.arms:
            return []

        leader = self.arms[leader_id]
        followers = self.get_followers()

        for follower in followers:
            follower["compatible"] = True
            follower["warning"] = None
            if leader.structural_design and follower.get("structural_design"):
                if leader.structural_design != follower["structural_design"]:
                    follower["compatible"] = False
                    follower["warning"] = f"Structural design mismatch: {leader.structural_design} vs {follower['structural_design']}"

        return followers

    def _save_config(self):
        """Save current configuration to settings.yaml"""
        # Build arms section
        arms_config = {}
        for arm_id, arm in self.arms.items():
            arms_config[arm_id] = {
                "name": arm.name,
                "role": arm.role.value,
                "motor_type": arm.motor_type.value,
                "port": arm.port,
                "enabled": arm.enabled,
            }
            if arm.structural_design:
                arms_config[arm_id]["structural_design"] = arm.structural_design
            if arm.config:
                arms_config[arm_id]["config"] = arm.config

        # Build pairings section
        pairings_config = []
        for pairing in self.pairings:
            pairings_config.append({
                "leader": pairing.leader_id,
                "follower": pairing.follower_id,
                "name": pairing.name,
            })

        # Update config data
        self._config_data["arms"] = arms_config
        self._config_data["pairings"] = pairings_config

        # Write to file
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self._config_data, f, default_flow_style=False, sort_keys=False)
            logger.info("Saved arm configuration to settings.yaml")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def set_arm_calibrated(self, arm_id: str, calibrated: bool):
        """Update calibration status for an arm"""
        if arm_id in self.arms:
            self.arms[arm_id].calibrated = calibrated

    def get_arm_instance(self, arm_id: str) -> Optional[Any]:
        """Get the connected robot/teleoperator instance for an arm"""
        return self.arm_instances.get(arm_id)

    def get_status_summary(self) -> Dict:
        """Get a summary of all arm statuses"""
        total = len(self.arms)
        connected = sum(1 for s in self.arm_status.values() if s == ConnectionStatus.CONNECTED)
        leaders = len([a for a in self.arms.values() if a.role == ArmRole.LEADER])
        followers = len([a for a in self.arms.values() if a.role == ArmRole.FOLLOWER])

        return {
            "total_arms": total,
            "connected": connected,
            "disconnected": total - connected,
            "leaders": leaders,
            "followers": followers,
            "pairings": len(self.pairings),
        }
