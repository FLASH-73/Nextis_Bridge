"""Background service that polls RP2040 trigger devices over serial
and fires tool activations when GPIO pin states change."""

import logging
import threading
import time
from collections import defaultdict
from typing import Dict, List

import serial

from app.core.hardware.tool_registry import ToolRegistryService
from app.core.hardware.types import TriggerDefinition

logger = logging.getLogger(__name__)


class TriggerListenerService:
    """Listens for GPIO trigger state changes over serial and dispatches
    tool activations via ToolRegistryService."""

    def __init__(self, tool_registry: ToolRegistryService):
        self._registry = tool_registry

        # trigger_id -> is_pressed (after active_low inversion)
        self._trigger_states: Dict[str, bool] = {}
        # tool_id -> is_active (tracks toggle flip state)
        self._tool_toggle_states: Dict[str, bool] = {}
        self._lock = threading.Lock()

        # Per-port thread management
        self._port_threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._serial_ports: Dict[str, serial.Serial] = {}

    # ── Public API ───────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """True if any port reader thread is alive."""
        return any(t.is_alive() for t in self._port_threads.values())

    def start(self) -> dict:
        """Start a reader thread for each unique trigger port.

        Returns a result dict with ``success``, optional ``error``/``warning``,
        ``started_threads`` count and ``ports`` list.
        """
        all_triggers = self._registry.triggers
        if not all_triggers:
            logger.info("No triggers registered — listener idle")
            return {
                "success": False,
                "error": "No triggers registered. Add a trigger device first.",
            }

        # Group enabled triggers by port
        port_triggers: Dict[str, List[TriggerDefinition]] = defaultdict(list)
        for trigger in all_triggers.values():
            if trigger.enabled:
                port_triggers[trigger.port].append(trigger)

        if not port_triggers:
            logger.info("All %d trigger(s) disabled — listener idle", len(all_triggers))
            return {
                "success": False,
                "error": f"All {len(all_triggers)} trigger(s) are disabled. Enable at least one trigger.",
            }

        started = 0
        ports_started: List[str] = []
        for port, triggers in port_triggers.items():
            if port in self._port_threads and self._port_threads[port].is_alive():
                continue  # Already running

            stop_event = threading.Event()
            self._stop_events[port] = stop_event

            thread = threading.Thread(
                target=self._port_reader_thread,
                args=(port, triggers, stop_event),
                daemon=True,
                name=f"TriggerPort-{port.split('/')[-1]}",
            )
            self._port_threads[port] = thread
            thread.start()
            trigger_names = ", ".join(t.name for t in triggers)
            logger.info(f"Started trigger listener on {port}: [{trigger_names}]")
            started += 1
            ports_started.append(port)

        result: dict = {
            "success": True,
            "message": "Trigger listener started",
            "started_threads": started,
            "ports": ports_started,
        }

        if not self._registry.tool_pairings:
            result["warning"] = (
                "Listener started but no tool pairings configured. "
                "Triggers will be detected but no tools will activate."
            )

        return result

    def stop(self):
        """Signal all port threads to stop, join them, close serial ports."""
        for event in self._stop_events.values():
            event.set()

        for port, thread in self._port_threads.items():
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning(f"Trigger thread for {port} did not stop in time")

        for port, ser in self._serial_ports.items():
            try:
                ser.close()
            except Exception:
                pass

        self._port_threads.clear()
        self._stop_events.clear()
        self._serial_ports.clear()
        logger.info("All trigger listener threads stopped")

    def get_trigger_states(self) -> Dict[str, bool]:
        """Return a snapshot of current trigger pressed states."""
        with self._lock:
            return dict(self._trigger_states)

    def get_tool_states(self) -> Dict[str, bool]:
        """Return a snapshot of current tool toggle states."""
        with self._lock:
            return dict(self._tool_toggle_states)

    # ── Port Reader Thread ───────────────────────────────────────────

    def _port_reader_thread(
        self,
        port: str,
        triggers: List[TriggerDefinition],
        stop_event: threading.Event,
    ):
        """Main loop: connect to serial port, identify device, read pin changes."""
        # Build pin -> trigger lookup for this port
        pin_to_triggers: Dict[int, List[TriggerDefinition]] = defaultdict(list)
        for t in triggers:
            pin_to_triggers[t.pin].append(t)

        while not stop_event.is_set():
            ser = None
            try:
                ser = serial.Serial(port, baudrate=115200, timeout=0.05)
                self._serial_ports[port] = ser

                # Identify device
                if not self._identify_device(ser, port, stop_event):
                    continue  # Retry handled inside _identify_device

                # Request initial state
                self._request_initial_state(ser, pin_to_triggers)

                # Read loop
                while not stop_event.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue  # Timeout, no data

                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("PIN:"):
                        continue

                    self._process_pin_message(line, pin_to_triggers)

            except serial.SerialException as e:
                logger.error(f"Serial error on {port}: {e}")
                self._close_serial(port, ser)
                # Wait before retry
                if not stop_event.wait(2.0):
                    logger.info(f"Retrying trigger connection on {port}...")
            except Exception as e:
                logger.error(f"Unexpected error in trigger listener for {port}: {e}")
                self._close_serial(port, ser)
                if not stop_event.wait(2.0):
                    continue

        self._close_serial(port, ser)
        logger.info(f"Trigger listener thread exiting for {port}")

    # ── Helpers ───────────────────────────────────────────────────────

    def _identify_device(
        self,
        ser: serial.Serial,
        port: str,
        stop_event: threading.Event,
    ) -> bool:
        """Send INFO command and verify this is a trigger device.
        Returns True on success, False if should retry (after delay)."""
        try:
            ser.reset_input_buffer()
            ser.write(b"INFO\n")
            # Use longer timeout for identification response
            old_timeout = ser.timeout
            ser.timeout = 1.0
            response = ser.readline().decode("utf-8", errors="ignore").strip()
            ser.timeout = old_timeout

            if response.startswith("NEXTIS_TRIGGER:"):
                version = response.split(":")[1] if ":" in response else "unknown"
                logger.info(f"Trigger device identified on {port} (version={version})")
                return True
            else:
                logger.warning(
                    f"Device on {port} is not a trigger device (got: {response!r})"
                )
                self._close_serial(port, ser)
                stop_event.wait(2.0)
                return False
        except Exception as e:
            logger.error(f"Failed to identify device on {port}: {e}")
            self._close_serial(port, ser)
            stop_event.wait(2.0)
            return False

    def _request_initial_state(
        self,
        ser: serial.Serial,
        pin_to_triggers: Dict[int, List[TriggerDefinition]],
    ):
        """Send STATE command to get initial pin values."""
        try:
            ser.write(b"STATE\n")
            # Read responses until we get a non-PIN line or timeout
            old_timeout = ser.timeout
            ser.timeout = 0.5
            for _ in range(32):  # Cap at 32 pins
                raw = ser.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="ignore").strip()
                if line.startswith("PIN:"):
                    self._process_pin_message(line, pin_to_triggers)
                elif line == "END":
                    break
            ser.timeout = old_timeout
        except Exception as e:
            logger.warning(f"Failed to read initial trigger state: {e}")

    def _process_pin_message(
        self,
        line: str,
        pin_to_triggers: Dict[int, List[TriggerDefinition]],
    ):
        """Parse 'PIN:<pin>:<0|1>' and dispatch trigger changes."""
        parts = line.split(":")
        if len(parts) != 3:
            return

        try:
            pin = int(parts[1])
            raw_state = int(parts[2])
        except ValueError:
            return

        triggers = pin_to_triggers.get(pin)
        if not triggers:
            return

        for trigger in triggers:
            # Apply active_low inversion
            pressed = (raw_state == 0) if trigger.active_low else (raw_state == 1)

            with self._lock:
                prev = self._trigger_states.get(trigger.id)
                if prev == pressed:
                    continue  # No change
                self._trigger_states[trigger.id] = pressed

            logger.info(
                f"Trigger {trigger.name} ({trigger.id}) pin {pin}: "
                f"{'pressed' if pressed else 'released'}"
            )

            # Find pairings for this trigger
            for pairing in self._registry.tool_pairings:
                if pairing.trigger_id == trigger.id:
                    self._handle_trigger_action(trigger.id, pressed, pairing)

    def _handle_trigger_action(self, trigger_id: str, pressed: bool, pairing):
        """Execute the pairing action (toggle/hold/pulse)."""
        tool_id = pairing.tool_id
        action = pairing.action

        try:
            if action == "toggle":
                # Only act on rising edge (press)
                if not pressed:
                    return
                with self._lock:
                    currently_active = self._tool_toggle_states.get(tool_id, False)
                    self._tool_toggle_states[tool_id] = not currently_active
                    new_state = not currently_active

                if new_state:
                    result = self._registry.activate_tool(tool_id)
                else:
                    result = self._registry.deactivate_tool(tool_id)
                logger.info(
                    f"Toggle {tool_id}: {'ON' if new_state else 'OFF'} "
                    f"(success={result.get('success')})"
                )

            elif action == "hold":
                if pressed:
                    result = self._registry.activate_tool(tool_id)
                else:
                    result = self._registry.deactivate_tool(tool_id)
                logger.info(
                    f"Hold {tool_id}: {'activate' if pressed else 'deactivate'} "
                    f"(success={result.get('success')})"
                )

            elif action == "pulse":
                # Only act on rising edge
                if not pressed:
                    return
                duration_ms = pairing.config.get("duration_ms", 500)
                self._registry.activate_tool(tool_id)
                logger.info(f"Pulse {tool_id}: activated for {duration_ms}ms")

                timer = threading.Timer(
                    duration_ms / 1000.0,
                    self._pulse_deactivate,
                    args=(tool_id,),
                )
                timer.daemon = True
                timer.start()

            else:
                logger.warning(f"Unknown action type '{action}' for pairing {pairing.name}")

        except Exception as e:
            logger.warning(f"Error handling trigger action for {tool_id}: {e}")

    def _pulse_deactivate(self, tool_id: str):
        """Timer callback to deactivate a tool after a pulse."""
        try:
            self._registry.deactivate_tool(tool_id)
            logger.info(f"Pulse {tool_id}: deactivated")
        except Exception as e:
            logger.warning(f"Failed to deactivate tool {tool_id} after pulse: {e}")

    def _close_serial(self, port: str, ser):
        """Safely close a serial port."""
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        self._serial_ports.pop(port, None)
