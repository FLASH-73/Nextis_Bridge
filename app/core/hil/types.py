"""HIL type definitions: mode enum and session state."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class HILMode(str, Enum):
    """Current control mode during HIL session."""
    IDLE = "idle"          # Session not active or episode not started
    AUTONOMOUS = "autonomous"  # Policy is executing
    HUMAN = "human"        # Human has taken control
    PAUSED = "paused"      # Intervention ended, waiting for user decision


@dataclass
class HILSessionState:
    """Tracks current HIL session state."""
    active: bool = False
    mode: HILMode = HILMode.IDLE
    policy_id: str = ""
    intervention_dataset: str = ""
    task: str = ""

    # Policy configuration (detected from checkpoint)
    policy_cameras: list = field(default_factory=list)  # e.g., ["camera_1", "camera_2"]
    policy_arms: list = field(default_factory=list)     # e.g., ["left"] or ["left", "right"]
    policy_type: str = ""                               # e.g., "diffusion"

    # Episode tracking
    episode_active: bool = False
    episode_count: int = 0

    # Intervention tracking
    intervention_count: int = 0  # Total interventions this session
    current_episode_interventions: int = 0  # Interventions in current episode

    # Frame tracking
    autonomous_frames: int = 0
    human_frames: int = 0

    # Safety settings
    movement_scale: float = 1.0  # 0.1 to 1.0 - limits autonomous movement range
