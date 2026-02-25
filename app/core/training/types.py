"""Types for training service."""
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import DATASETS_DIR, LEROBOT_SRC, POLICIES_DIR, PROJECT_ROOT

# Re-export under old names for backward compat within this sub-package
_PROJECT_ROOT = PROJECT_ROOT
_DEFAULT_DATASETS_PATH = DATASETS_DIR
_DEFAULT_OUTPUTS_PATH = POLICIES_DIR


class JobStatus(str, Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PolicyType(str, Enum):
    SMOLVLA = "smolvla"
    DIFFUSION = "diffusion"
    ACT = "act"
    PI05 = "pi05"


@dataclass
class TrainingProgress:
    step: int = 0
    total_steps: int = 0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    eta_seconds: Optional[int] = None
    epoch: int = 0
    loss_history: List[List] = field(default_factory=list)  # [[step, loss], ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingJob:
    id: str
    status: JobStatus
    policy_type: PolicyType
    dataset_repo_id: str
    config: Dict[str, Any]
    created_at: datetime
    progress: TrainingProgress = field(default_factory=TrainingProgress)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    output_dir: Optional[Path] = None
    pid: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.value,
            "policy_type": self.policy_type.value,
            "dataset_repo_id": self.dataset_repo_id,
            "config": self.config,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress.to_dict(),
            "error": self.error,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "pid": self.pid,
        }


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
    features: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PolicyInfo:
    """Represents a trained or training policy."""
    id: str
    name: str
    policy_type: str
    status: str  # "completed", "training", "failed"
    steps: int
    total_steps: int
    dataset_repo_id: str
    created_at: str
    final_loss: Optional[float]
    checkpoint_path: str
    loss_history: List[List]  # [[step, loss], ...]
    output_dir: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PolicyConfig:
    """Extracted policy configuration from checkpoint - describes what the policy was trained on."""
    cameras: List[str]  # Camera names used, e.g., ["camera_1", "camera_2"]
    arms: List[str]     # Arms used, e.g., ["left"] or ["left", "right"]
    state_dim: int      # Dimension of observation.state, e.g., 7 for single arm
    action_dim: int     # Dimension of action output, e.g., 7 for single arm
    policy_type: str    # Policy type, e.g., "diffusion"

    def to_dict(self) -> dict:
        return asdict(self)
