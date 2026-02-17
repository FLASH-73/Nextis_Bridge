"""Stateless helpers for reading tool/trigger state into observation dicts.

Pure functions — no threading, no side effects. Just reads from the
ToolRegistryService and TriggerListenerService and returns flat dicts
compatible with LeRobot dataset format (all float values).
"""

import logging

logger = logging.getLogger(__name__)


def get_tool_observations(tool_registry, trigger_listener) -> dict:
    """Build a flat dict of tool/trigger state for injection into observation frames.

    Returns keys like:
        "tool.screwdriver_1.active": 1.0 or 0.0
        "tool.screwdriver_1.speed": 200.0 or 0.0
        "trigger.leader_switch.pressed": 1.0 or 0.0

    All values are floats for compatibility with LeRobot dataset format.
    Returns empty dict if services are None or not running.
    """
    result = {}

    # Tool active/speed state
    if tool_registry is not None:
        for tool_id, tool in tool_registry.tools.items():
            if not tool.enabled:
                continue

            is_active = tool_registry._tool_active.get(tool_id, False)
            result[f"tool.{tool_id}.active"] = 1.0 if is_active else 0.0

            if is_active:
                speed = float(tool.config.get("speed", 500))
            else:
                speed = 0.0
            result[f"tool.{tool_id}.speed"] = speed

    # Trigger pressed state
    if trigger_listener is not None:
        try:
            trigger_states = trigger_listener.get_trigger_states()
        except Exception:
            trigger_states = {}

        if tool_registry is not None:
            for trigger_id, trigger in tool_registry.triggers.items():
                if not trigger.enabled:
                    continue
                pressed = trigger_states.get(trigger_id, False)
                result[f"trigger.{trigger_id}.pressed"] = 1.0 if pressed else 0.0

    return result


def get_tool_action_features(tool_registry) -> dict:
    """Build feature spec dict for dataset creation.

    Returns {key: 1} for each tool observation key (scalar features).
    Used when creating/extending a LeRobotDataset to register tool columns.
    """
    if tool_registry is None:
        return {}

    features = {}

    for tool_id, tool in tool_registry.tools.items():
        if not tool.enabled:
            continue
        features[f"tool.{tool_id}.active"] = 1
        features[f"tool.{tool_id}.speed"] = 1

    for trigger_id, trigger in tool_registry.triggers.items():
        if not trigger.enabled:
            continue
        features[f"trigger.{trigger_id}.pressed"] = 1

    return features
