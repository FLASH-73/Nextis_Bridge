"""Data contracts for multi-step deployment pipelines.

Defines the types needed to sequence multiple policies with automatic
transition conditions. All types are pure data — no logic, no methods.
Every dataclass is JSON-serialisable via ``dataclasses.asdict()``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TransitionTrigger(str, Enum):
    """What causes a pipeline step to advance."""
    FRAME_COUNT = "frame_count"
    TIMEOUT = "timeout"
    GRIPPER_CLOSED = "gripper_closed"
    GRIPPER_OPENED = "gripper_opened"
    TORQUE_SPIKE = "torque_spike"
    POSITION_REACHED = "position_reached"
    MANUAL = "manual"


class PipelineState(str, Enum):
    """Lifecycle state of a pipeline execution."""
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    TRANSITIONING = "transitioning"
    COMPLETED = "completed"
    ESTOP = "estop"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TransitionCondition:
    """When and how a pipeline step should advance to the next."""
    trigger: TransitionTrigger = TransitionTrigger.MANUAL
    threshold_value: float = 0.0
    threshold_position: Dict[str, float] = field(default_factory=dict)
    timeout_seconds: float = 0.0
    debounce_frames: int = 8


@dataclass
class BridgeConfig:
    """Configuration for the bridge move between pipeline steps.

    When enabled, the arm smoothly moves to the next step's empirical
    start pose (extracted from its training dataset) before the policy starts.
    """
    enabled: bool = True
    speed_scale: float = 0.3
    settle_frames: int = 15
    source: str = "auto"


@dataclass
class PipelineStep:
    """A single step in a multi-step deployment pipeline."""
    policy_id: str = ""
    name: str = ""
    transition: Optional[TransitionCondition] = None
    warmup_frames: int = 12
    speed_scale: float = 1.0
    temporal_ensemble_coeff: Optional[float] = None
    bridge: Optional['BridgeConfig'] = None


@dataclass
class PipelineConfig:
    """Full configuration for a multi-step pipeline."""
    name: str = ""
    steps: List[PipelineStep] = field(default_factory=list)
    active_arms: List[str] = field(default_factory=list)
    loop_hz: int = 30
    safety_overrides: Optional[dict] = None


@dataclass
class TransitionProgress:
    """Live progress toward the current step's transition condition.

    Provides everything the frontend needs for a progress bar.
    """
    current_value: float = 0.0
    threshold_value: float = 0.0
    label: str = ""
    debounce_current: int = 0
    debounce_required: int = 0


@dataclass
class PipelineStatus:
    """Full live status of a pipeline execution."""
    state: PipelineState = PipelineState.IDLE
    current_step_index: int = 0
    current_step_name: str = ""
    total_steps: int = 0
    step_frame_count: int = 0
    total_frame_count: int = 0
    step_elapsed_seconds: float = 0.0
    total_elapsed_seconds: float = 0.0
    transition_progress: Optional[TransitionProgress] = None
    error_message: str = ""


@dataclass
class AlignmentWarning:
    """Warning when consecutive pipeline steps have misaligned action/obs distributions."""
    step_from: str
    step_to: str
    joint_name: str
    delta_rad: float
    message: str
