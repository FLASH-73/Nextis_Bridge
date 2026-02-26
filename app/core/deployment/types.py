"""Deployment types: enums, dataclasses, and state machine for policy deployment."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DeploymentMode(str, Enum):
    """Type of deployment session."""
    INFERENCE = "inference"    # Pure policy execution (orchestrator)
    HIL = "hil"               # Human-in-the-loop DAgger
    HIL_SERL = "hil_serl"     # RL with human interventions (SERL-style)


class ActionSource(str, Enum):
    """Source of the current action being processed."""
    POLICY = "policy"
    HUMAN = "human"
    HOLD = "hold"


# Valid state transitions.  ESTOP and ERROR are terminal — require reset().
VALID_TRANSITIONS: Dict[str, set] = {
    "idle": {"starting"},
    "starting": {"running", "error", "estop"},
    "running": {"human_active", "paused", "stopping", "estop", "error"},
    "human_active": {"running", "paused", "stopping", "estop", "error"},
    "paused": {"running", "stopping", "estop", "error"},
    "stopping": {"idle", "error", "estop"},
    "estop": set(),
    "error": set(),
}


class RuntimeState(str, Enum):
    """Deployment runtime state with defined transitions.

    IDLE → STARTING → RUNNING ↔ HUMAN_ACTIVE ↔ PAUSED → STOPPING → IDLE
    ESTOP and ERROR are reachable from any non-terminal state.
    """
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    HUMAN_ACTIVE = "human_active"
    PAUSED = "paused"
    STOPPING = "stopping"
    ESTOP = "estop"
    ERROR = "error"

    @classmethod
    def can_transition(cls, from_state: "RuntimeState", to_state: "RuntimeState") -> bool:
        """Check whether a transition between two states is valid."""
        return to_state.value in VALID_TRANSITIONS.get(from_state.value, set())


# ---------------------------------------------------------------------------
# Velocity limit defaults (rad/s) — conservative for deployment safety
# ---------------------------------------------------------------------------

DEFAULT_VELOCITY_LIMITS: Dict[str, float] = {
    "J8009P": 1.5,    # 35Nm motor (datasheet: 6.6 rad/s)
    "J4340P": 2.5,    # Medium motor
    "J4310": 3.5,     # Wrist motor (raw: 20.9, rate-limited to 10.0 in MIT)
    "STS3215": 4.0,   # Feetech servo
}
FALLBACK_VELOCITY_LIMIT: float = 2.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SafetyConfig:
    """Configuration for the safety pipeline.

    Populated from calibration profiles and motor specs at deploy time.
    """
    # Per-motor joint limits: motor_name -> (min, max) in native units
    joint_limits: Dict[str, tuple] = field(default_factory=dict)
    # Per-motor model mapping: motor_name -> model string (e.g. "J8009P")
    motor_models: Dict[str, str] = field(default_factory=dict)
    # Max allowed acceleration (rad/s^2)
    max_acceleration: float = 15.0
    # EMA smoothing factor: 1.0 = no filtering, lower = more smoothing
    smoothing_alpha: float = 0.3
    # Frames between torque checks (~10Hz at 30Hz loop with interval=3)
    torque_check_interval: int = 3
    # Global speed multiplier for all velocity limits [0.1, 1.0]
    speed_scale: float = 1.0

    def effective_max_velocity(self, motor_name: str) -> float:
        """Look up the velocity limit for a motor by its model and apply speed_scale."""
        model = self.motor_models.get(motor_name, "")
        base_limit = DEFAULT_VELOCITY_LIMITS.get(model, FALLBACK_VELOCITY_LIMIT)
        return base_limit * max(0.1, min(1.0, self.speed_scale))


@dataclass
class DeploymentConfig:
    """Full configuration for starting a deployment session."""
    mode: DeploymentMode = DeploymentMode.INFERENCE
    policy_id: str = ""
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    # Legacy compat — callers should set safety.speed_scale = movement_scale
    movement_scale: float = 1.0
    # HIL options
    intervention_dataset: Optional[str] = None
    task: Optional[str] = None
    # RL options
    reward_source: Optional[str] = None
    reward_model: Optional[str] = None
    max_episodes: Optional[int] = None


@dataclass
class DeploymentStatus:
    """Live deployment status for frontend monitoring."""
    state: RuntimeState = RuntimeState.IDLE
    mode: DeploymentMode = DeploymentMode.INFERENCE
    frame_count: int = 0
    episode_count: int = 0
    current_episode_frames: int = 0
    safety: Dict = field(default_factory=dict)
    rl_metrics: Optional[Dict] = None
    policy_config: Optional[Dict] = None
