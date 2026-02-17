import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import serial
import yaml

from app.core.hardware.types import (
    ConnectionStatus,
    MotorType,
    ToolDefinition,
    ToolPairing,
    ToolType,
    TriggerDefinition,
    TriggerType,
)

logger = logging.getLogger(__name__)


class ToolRegistryService:
    """Manages single-actuator tools, GPIO triggers, and trigger-to-tool pairings."""

    def __init__(self, config_path: str = "app/config/settings.yaml"):
        self.config_path = Path(config_path)
        self.tools: Dict[str, ToolDefinition] = {}
        self.triggers: Dict[str, TriggerDefinition] = {}
        self.tool_pairings: List[ToolPairing] = []

        self.tool_instances: Dict[str, Any] = {}
        self._tool_status: Dict[str, ConnectionStatus] = {}
        self._tool_active: Dict[str, bool] = {}

        self._lock = threading.Lock()
        self._config_data: Dict = {}

        self._load_config()

    # ── Config I/O ──────────────────────────────────────────────────────

    def _load_config(self):
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return

        with open(self.config_path, "r") as f:
            self._config_data = yaml.safe_load(f) or {}

        for tool_id, cfg in self._config_data.get("tools", {}).items():
            try:
                tool = ToolDefinition(
                    id=tool_id,
                    name=cfg.get("name", tool_id),
                    motor_type=MotorType(cfg.get("motor_type", "sts3215")),
                    port=cfg.get("port", ""),
                    motor_id=int(cfg.get("motor_id", 1)),
                    tool_type=ToolType(cfg.get("tool_type", "custom")),
                    enabled=cfg.get("enabled", True),
                    config=cfg.get("config", {}),
                )
                self.tools[tool_id] = tool
                self._tool_status[tool_id] = ConnectionStatus.DISCONNECTED
                logger.info(f"Loaded tool: {tool.name} ({tool_id})")
            except Exception as e:
                logger.error(f"Failed to load tool {tool_id}: {e}")

        for trig_id, cfg in self._config_data.get("triggers", {}).items():
            try:
                trigger = TriggerDefinition(
                    id=trig_id,
                    name=cfg.get("name", trig_id),
                    trigger_type=TriggerType(cfg.get("trigger_type", "gpio_switch")),
                    port=cfg.get("port", ""),
                    pin=int(cfg.get("pin", 0)),
                    active_low=cfg.get("active_low", True),
                    enabled=cfg.get("enabled", True),
                    config=cfg.get("config", {}),
                )
                self.triggers[trig_id] = trigger
                logger.info(f"Loaded trigger: {trigger.name} ({trig_id})")
            except Exception as e:
                logger.error(f"Failed to load trigger {trig_id}: {e}")

        for pcfg in self._config_data.get("tool_pairings", []):
            try:
                pairing = ToolPairing(
                    trigger_id=pcfg["trigger_id"],
                    tool_id=pcfg["tool_id"],
                    name=pcfg.get("name", ""),
                    action=pcfg.get("action", "toggle"),
                    config=pcfg.get("config", {}),
                )
                self.tool_pairings.append(pairing)
            except Exception as e:
                logger.error(f"Failed to load tool pairing: {e}")

    def save_config(self):
        tools_cfg: Dict[str, Dict] = {}
        for tool_id, tool in self.tools.items():
            entry: Dict[str, Any] = {
                "name": tool.name,
                "motor_type": tool.motor_type.value,
                "port": tool.port,
                "motor_id": tool.motor_id,
                "tool_type": tool.tool_type.value,
                "enabled": tool.enabled,
            }
            if tool.config:
                entry["config"] = tool.config
            tools_cfg[tool_id] = entry

        triggers_cfg: Dict[str, Dict] = {}
        for trig_id, trig in self.triggers.items():
            entry = {
                "name": trig.name,
                "trigger_type": trig.trigger_type.value,
                "port": trig.port,
                "pin": trig.pin,
                "active_low": trig.active_low,
                "enabled": trig.enabled,
            }
            if trig.config:
                entry["config"] = trig.config
            triggers_cfg[trig_id] = entry

        pairings_cfg: List[Dict] = []
        for p in self.tool_pairings:
            pairings_cfg.append({
                "trigger_id": p.trigger_id,
                "tool_id": p.tool_id,
                "name": p.name,
                "action": p.action,
                "config": p.config,
            })

        self._config_data["tools"] = tools_cfg
        self._config_data["triggers"] = triggers_cfg
        self._config_data["tool_pairings"] = pairings_cfg

        try:
            with open(self.config_path, "w") as f:
                yaml.dump(self._config_data, f, default_flow_style=False, sort_keys=False)
            logger.info("Saved tool configuration to settings.yaml")
        except Exception as e:
            logger.error(f"Failed to save tool config: {e}")

    # ── Tool CRUD ───────────────────────────────────────────────────────

    def get_all_tools(self) -> List[Dict]:
        result = []
        for tool_id, tool in self.tools.items():
            d = tool.to_dict()
            d["status"] = self._tool_status.get(tool_id, ConnectionStatus.DISCONNECTED).value
            d["active"] = self._tool_active.get(tool_id, False)
            result.append(d)
        return result

    def add_tool(self, data: Dict) -> Dict:
        with self._lock:
            tool_id = data.get("id")
            if not tool_id:
                return {"success": False, "error": "Tool ID is required"}
            if tool_id in self.tools:
                return {"success": False, "error": f"Tool '{tool_id}' already exists"}

            try:
                tool = ToolDefinition(
                    id=tool_id,
                    name=data.get("name", tool_id),
                    motor_type=MotorType(data.get("motor_type", "sts3215")),
                    port=data.get("port", ""),
                    motor_id=int(data.get("motor_id", 1)),
                    tool_type=ToolType(data.get("tool_type", "custom")),
                    enabled=data.get("enabled", True),
                    config=data.get("config", {}),
                )
                self.tools[tool_id] = tool
                self._tool_status[tool_id] = ConnectionStatus.DISCONNECTED
                self.save_config()
                return {"success": True, "tool": tool.to_dict()}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def update_tool(self, tool_id: str, **kwargs) -> Dict:
        with self._lock:
            if tool_id not in self.tools:
                return {"success": False, "error": f"Tool '{tool_id}' not found"}

            tool = self.tools[tool_id]
            try:
                for key, value in kwargs.items():
                    if key == "name":
                        tool.name = value
                    elif key == "port":
                        tool.port = value
                    elif key == "motor_id":
                        tool.motor_id = int(value)
                    elif key == "tool_type":
                        tool.tool_type = ToolType(value)
                    elif key == "motor_type":
                        tool.motor_type = MotorType(value)
                    elif key == "enabled":
                        tool.enabled = bool(value)
                    elif key == "config":
                        tool.config = value
                self.save_config()
                return {"success": True, "tool": tool.to_dict()}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def remove_tool(self, tool_id: str) -> Dict:
        with self._lock:
            if tool_id not in self.tools:
                return {"success": False, "error": f"Tool '{tool_id}' not found"}

            if self._tool_status.get(tool_id) == ConnectionStatus.CONNECTED:
                self.disconnect_tool(tool_id)

            self.tool_pairings = [
                p for p in self.tool_pairings if p.tool_id != tool_id
            ]
            del self.tools[tool_id]
            del self._tool_status[tool_id]
            self.tool_instances.pop(tool_id, None)
            self._tool_active.pop(tool_id, None)

            self.save_config()
            return {"success": True}

    # ── Tool Connection ─────────────────────────────────────────────────

    def connect_tool(self, tool_id: str) -> Dict:
        with self._lock:
            if tool_id not in self.tools:
                return {"success": False, "error": f"Tool '{tool_id}' not found"}

            tool = self.tools[tool_id]
            if not tool.enabled:
                return {"success": False, "error": f"Tool '{tool_id}' is disabled"}

            self._tool_status[tool_id] = ConnectionStatus.CONNECTING

            try:
                bus = self._create_tool_bus(tool)
                bus.connect()

                if tool.tool_type in (ToolType.SCREWDRIVER, ToolType.PUMP):
                    self._set_velocity_mode(bus, tool.motor_type)

                self.tool_instances[tool_id] = bus
                self._tool_status[tool_id] = ConnectionStatus.CONNECTED
                logger.info(f"Connected tool: {tool.name} ({tool_id})")
                return {"success": True, "status": "connected"}
            except Exception as e:
                self._tool_status[tool_id] = ConnectionStatus.ERROR
                logger.error(f"Failed to connect tool {tool_id}: {e}")
                return {"success": False, "error": str(e)}

    def disconnect_tool(self, tool_id: str) -> Dict:
        if tool_id not in self.tools:
            return {"success": False, "error": f"Tool '{tool_id}' not found"}

        if self._tool_active.get(tool_id):
            self.deactivate_tool(tool_id)

        if tool_id in self.tool_instances:
            try:
                bus = self.tool_instances[tool_id]
                tool = self.tools[tool_id]
                if tool.tool_type in (ToolType.SCREWDRIVER, ToolType.PUMP):
                    self._set_position_mode(bus, tool.motor_type)
                bus.disconnect()
                del self.tool_instances[tool_id]
            except Exception as e:
                logger.error(f"Error disconnecting tool {tool_id}: {e}")

        self._tool_status[tool_id] = ConnectionStatus.DISCONNECTED
        logger.info(f"Disconnected tool: {tool_id}")
        return {"success": True, "status": "disconnected"}

    def _create_tool_bus(self, tool: ToolDefinition) -> Any:
        from lerobot.motors import Motor, MotorNormMode

        if tool.motor_type == MotorType.STS3215:
            from lerobot.motors.feetech import FeetechMotorsBus

            return FeetechMotorsBus(
                port=tool.port,
                motors={"tool": Motor(tool.motor_id, "sts3215", MotorNormMode.DEGREES)},
            )
        elif tool.motor_type in (MotorType.DYNAMIXEL_XL330, MotorType.DYNAMIXEL_XL430):
            from lerobot.motors.dynamixel import DynamixelMotorsBus

            model = (
                "xl330-m288"
                if tool.motor_type == MotorType.DYNAMIXEL_XL330
                else "xl430-w250"
            )
            return DynamixelMotorsBus(
                port=tool.port,
                motors={"tool": Motor(tool.motor_id, model, MotorNormMode.DEGREES)},
            )
        else:
            raise ValueError(f"Unsupported motor type for tools: {tool.motor_type.value}")

    def _set_velocity_mode(self, bus: Any, motor_type: MotorType):
        if motor_type == MotorType.STS3215:
            from lerobot.motors.feetech import OperatingMode
        else:
            from lerobot.motors.dynamixel import OperatingMode

        bus.set_operating_mode(OperatingMode.VELOCITY, ["tool"])
        bus.enable_torque(["tool"])

    def _set_position_mode(self, bus: Any, motor_type: MotorType):
        if motor_type == MotorType.STS3215:
            from lerobot.motors.feetech import OperatingMode
        else:
            from lerobot.motors.dynamixel import OperatingMode

        try:
            bus.disable_torque(["tool"])
            bus.set_operating_mode(OperatingMode.POSITION, ["tool"])
        except Exception as e:
            logger.warning(f"Could not reset to position mode: {e}")

    # ── Tool Activation ─────────────────────────────────────────────────

    def activate_tool(
        self, tool_id: str, speed: int = 500, direction: int = 1
    ) -> Dict:
        with self._lock:
            if tool_id not in self.tools:
                return {"success": False, "error": f"Tool '{tool_id}' not found"}
            if self._tool_status.get(tool_id) != ConnectionStatus.CONNECTED:
                return {"success": False, "error": f"Tool '{tool_id}' is not connected"}
            if tool_id not in self.tool_instances:
                return {"success": False, "error": f"No bus instance for tool '{tool_id}'"}

            try:
                bus = self.tool_instances[tool_id]
                velocity = speed * direction
                bus.write("Goal_Velocity", "tool", velocity, normalize=False)
                self._tool_active[tool_id] = True
                logger.info(f"Activated tool {tool_id}: speed={speed}, direction={direction}")
                return {
                    "success": True,
                    "tool_id": tool_id,
                    "speed": speed,
                    "direction": direction,
                }
            except Exception as e:
                logger.error(f"Failed to activate tool {tool_id}: {e}")
                return {"success": False, "error": str(e)}

    def deactivate_tool(self, tool_id: str) -> Dict:
        if tool_id not in self.tools:
            return {"success": False, "error": f"Tool '{tool_id}' not found"}
        if tool_id not in self.tool_instances:
            self._tool_active[tool_id] = False
            return {"success": True}

        try:
            bus = self.tool_instances[tool_id]
            bus.write("Goal_Velocity", "tool", 0, normalize=False)
            self._tool_active[tool_id] = False
            logger.info(f"Deactivated tool {tool_id}")
            return {"success": True}
        except Exception as e:
            logger.error(f"Failed to deactivate tool {tool_id}: {e}")
            return {"success": False, "error": str(e)}

    # ── Trigger CRUD ────────────────────────────────────────────────────

    def get_all_triggers(self) -> List[Dict]:
        return [t.to_dict() for t in self.triggers.values()]

    def add_trigger(self, data: Dict) -> Dict:
        with self._lock:
            trigger_id = data.get("id")
            if not trigger_id:
                return {"success": False, "error": "Trigger ID is required"}
            if trigger_id in self.triggers:
                return {"success": False, "error": f"Trigger '{trigger_id}' already exists"}

            try:
                trigger = TriggerDefinition(
                    id=trigger_id,
                    name=data.get("name", trigger_id),
                    trigger_type=TriggerType(data.get("trigger_type", "gpio_switch")),
                    port=data.get("port", ""),
                    pin=int(data.get("pin", 0)),
                    active_low=data.get("active_low", True),
                    enabled=data.get("enabled", True),
                    config=data.get("config", {}),
                )
                self.triggers[trigger_id] = trigger
                self.save_config()
                return {"success": True, "trigger": trigger.to_dict()}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def remove_trigger(self, trigger_id: str) -> Dict:
        with self._lock:
            if trigger_id not in self.triggers:
                return {"success": False, "error": f"Trigger '{trigger_id}' not found"}

            self.tool_pairings = [
                p for p in self.tool_pairings if p.trigger_id != trigger_id
            ]
            del self.triggers[trigger_id]
            self.save_config()
            return {"success": True}

    # ── Tool Pairing CRUD ───────────────────────────────────────────────

    def get_all_tool_pairings(self) -> List[Dict]:
        return [p.to_dict() for p in self.tool_pairings]

    def create_tool_pairing(
        self,
        trigger_id: str,
        tool_id: str,
        name: Optional[str] = None,
        action: str = "toggle",
        config: Optional[Dict] = None,
    ) -> Dict:
        with self._lock:
            if trigger_id not in self.triggers:
                return {"success": False, "error": f"Trigger '{trigger_id}' not found"}
            if tool_id not in self.tools:
                return {"success": False, "error": f"Tool '{tool_id}' not found"}

            for p in self.tool_pairings:
                if p.trigger_id == trigger_id and p.tool_id == tool_id:
                    return {"success": False, "error": "This pairing already exists"}

            pairing_name = name or f"{self.triggers[trigger_id].name} -> {self.tools[tool_id].name}"
            pairing = ToolPairing(
                trigger_id=trigger_id,
                tool_id=tool_id,
                name=pairing_name,
                action=action,
                config=config or {},
            )
            self.tool_pairings.append(pairing)
            self.save_config()
            return {"success": True, "pairing": pairing.to_dict()}

    def remove_tool_pairing(self, trigger_id: str, tool_id: str) -> Dict:
        with self._lock:
            before = len(self.tool_pairings)
            self.tool_pairings = [
                p
                for p in self.tool_pairings
                if not (p.trigger_id == trigger_id and p.tool_id == tool_id)
            ]
            if len(self.tool_pairings) == before:
                return {"success": False, "error": "Pairing not found"}

            self.save_config()
            return {"success": True}

    # ── Scanning ────────────────────────────────────────────────────────

    def scan_tool_motors(self, port: str, motor_type: str) -> Dict:
        try:
            mt = MotorType(motor_type)
        except ValueError:
            return {"success": False, "error": f"Unknown motor type: {motor_type}"}

        if mt == MotorType.DAMIAO:
            return {"success": False, "error": "Damiao motors are not supported as tools"}

        try:
            if mt == MotorType.STS3215:
                from lerobot.motors.feetech import FeetechMotorsBus

                logger.info(f"Scanning for Feetech tool motors on {port}...")
                baudrate_ids = FeetechMotorsBus.scan_port(port)
                found_ids: List[int] = []
                for ids in baudrate_ids.values():
                    found_ids.extend(ids)
                found_ids = list(dict.fromkeys(found_ids))
                return {
                    "success": True,
                    "found_ids": found_ids,
                    "baudrate_info": {str(k): v for k, v in baudrate_ids.items()},
                }

            else:
                from lerobot.motors.dynamixel import DynamixelMotorsBus

                logger.info(f"Scanning for Dynamixel tool motors on {port}...")
                bus = DynamixelMotorsBus(port=port, motors={})
                bus._connect(handshake=False)
                found_ids = []
                baudrate_info: Dict[str, List[int]] = {}
                for baudrate in [57600, 1_000_000]:
                    bus.set_baudrate(baudrate)
                    id_model = bus.broadcast_ping(num_retry=2)
                    if id_model:
                        ids = sorted(id_model.keys())
                        found_ids.extend(ids)
                        baudrate_info[str(baudrate)] = ids
                bus.port_handler.closePort()
                found_ids = sorted(set(found_ids))
                return {
                    "success": True,
                    "found_ids": found_ids,
                    "baudrate_info": baudrate_info,
                }

        except Exception as e:
            logger.error(f"Tool motor scan failed on {port}: {e}")
            return {"success": False, "error": str(e)}

    def identify_trigger_device(self, port: str) -> Dict:
        try:
            ser = serial.Serial(port, baudrate=115200, timeout=1.0)
            ser.write(b"NEXTIS_TRIGGER?\n")
            response = ser.readline().decode("utf-8", errors="ignore").strip()
            ser.close()

            if response.startswith("NEXTIS_TRIGGER:"):
                version = response.split(":")[1] if ":" in response else "unknown"
                return {"success": True, "is_trigger": True, "version": version, "port": port}
            else:
                return {"success": True, "is_trigger": False, "response": response}
        except Exception as e:
            return {"success": False, "error": str(e)}
