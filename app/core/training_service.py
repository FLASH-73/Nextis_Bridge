"""
Training Service for SmolVLA and other policy training.
Manages training jobs, validates datasets, and executes training in background subprocesses.
"""

import os
import sys
import json
import logging
import threading
import subprocess
import signal
import re
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

# Compute project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASETS_PATH = _PROJECT_ROOT / "datasets"
_DEFAULT_OUTPUTS_PATH = _PROJECT_ROOT / "training" / "outputs"


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


# Training presets for SmolVLA
SMOLVLA_PRESETS = {
    "quick": {
        "steps": 10_000,
        "batch_size": 8,
        "save_freq": 5_000,
        "eval_freq": 5_000,
        "description": "Quick test run (~30 min on GPU)"
    },
    "standard": {
        "steps": 100_000,
        "batch_size": 8,
        "save_freq": 20_000,
        "eval_freq": 20_000,
        "description": "Standard training (~5 hours on GPU)"
    },
    "full": {
        "steps": 200_000,
        "batch_size": 16,
        "save_freq": 25_000,
        "eval_freq": 25_000,
        "description": "Full training for best quality (~10 hours on GPU)"
    }
}

# Training presets for Diffusion Policy
DIFFUSION_PRESETS = {
    "quick": {
        "steps": 10_000,
        "batch_size": 32,
        "save_freq": 5_000,
        "eval_freq": 5_000,
        "description": "Quick test (~15 min on GPU)"
    },
    "standard": {
        "steps": 50_000,
        "batch_size": 64,
        "save_freq": 10_000,
        "eval_freq": 10_000,
        "description": "Standard training (~1 hour)"
    },
    "full": {
        "steps": 100_000,
        "batch_size": 64,
        "save_freq": 20_000,
        "eval_freq": 20_000,
        "description": "Full training (~2 hours)"
    }
}

# Training presets for Pi0.5 Policy (with LoRA: ~22GB VRAM, without: ~40GB+)
# Optimized for small datasets (20-50 episodes) to prevent overfitting
PI05_PRESETS = {
    "quick": {
        "steps": 2_000,
        "batch_size": 8,
        "save_freq": 500,
        "eval_freq": 500,
        "description": "Quick test (~10 min, good for debugging)"
    },
    "standard": {
        "steps": 5_000,
        "batch_size": 8,
        "save_freq": 1_000,
        "eval_freq": 1_000,
        "description": "Standard finetuning (best for 20-50 episodes)"
    },
    "full": {
        "steps": 15_000,
        "batch_size": 8,
        "save_freq": 2_500,
        "eval_freq": 2_500,
        "description": "Extended training (50+ episodes, risk of overfitting with less)"
    }
}

# Default training config for SmolVLA
SMOLVLA_DEFAULTS = {
    "pretrained_model": "lerobot/smolvla_base",
    "freeze_vision_encoder": True,
    "train_expert_only": True,
    "train_state_proj": True,
    "learning_rate": 1e-4,
    "warmup_steps": 1000,
    "num_workers": 4,
}

# Default training config for Diffusion Policy
DIFFUSION_DEFAULTS = {
    "n_obs_steps": 2,
    "horizon": 16,
    "n_action_steps": 8,
    "noise_scheduler_type": "DDPM",
    "num_train_timesteps": 100,
    "vision_backbone": "resnet18",
    "learning_rate": 1e-4,
    "warmup_steps": 500,
    "num_workers": 4,
    "resize_shape": [480, 640],  # Default resize for multi-camera support (height, width)
}

# Default training config for Pi0.5 Policy
PI05_DEFAULTS = {
    "pretrained_path": "lerobot/pi05_base",  # or "lerobot/pi05_libero"
    "gradient_checkpointing": True,  # CRITICAL: reduces VRAM significantly
    "dtype": "bfloat16",  # CRITICAL: half precision
    "compile_model": False,  # Disabled - compilation warmup uses extra memory
    "chunk_size": 50,
    "n_action_steps": 50,
    "learning_rate": 2.5e-5,
    "warmup_steps": 1_000,
    "num_workers": 4,
    "use_quantile_normalization": True,  # If False, use MEAN_STD fallback
    # LoRA settings (for memory-efficient fine-tuning)
    "lora_rank": 8,  # 0 = disabled, 8 = recommended for small datasets
    "lora_alpha": 16,  # Scaling factor (typically 2x rank)
    "lora_dropout": 0.1,
}

# Training presets for ACT (Action Chunking with Transformers)
# Lightweight model (~8GB VRAM), designed for bimanual fine manipulation
ACT_PRESETS = {
    "quick": {
        "steps": 50_000,
        "batch_size": 8,
        "save_freq": 10_000,
        "eval_freq": 10_000,
        "description": "Quick test (~20 min on GPU)"
    },
    "standard": {
        "steps": 100_000,
        "batch_size": 8,
        "save_freq": 25_000,
        "eval_freq": 25_000,
        "description": "Standard training (best for 20-50 episodes)"
    },
    "full": {
        "steps": 200_000,
        "batch_size": 8,
        "save_freq": 50_000,
        "eval_freq": 50_000,
        "description": "Extended training for maximum quality"
    }
}

# Default training config for ACT
ACT_DEFAULTS = {
    "chunk_size": 100,
    "n_action_steps": 100,
    "n_obs_steps": 1,
    "use_vae": True,
    "latent_dim": 32,
    "kl_weight": 10.0,
    "vision_backbone": "resnet18",
    "dim_model": 512,
    "learning_rate": 1e-5,
    "warmup_steps": 500,
    "num_workers": 4,
}


class TrainingService:
    """Manages training jobs for robot learning policies."""

    def __init__(
        self,
        datasets_path: Path = None,
        outputs_path: Path = None
    ):
        self.datasets_path = Path(datasets_path) if datasets_path else _DEFAULT_DATASETS_PATH
        self.outputs_path = Path(outputs_path) if outputs_path else _DEFAULT_OUTPUTS_PATH

        # Create outputs directory
        self.outputs_path.mkdir(parents=True, exist_ok=True)

        # Job storage
        self.jobs: Dict[str, TrainingJob] = {}
        self.active_job: Optional[TrainingJob] = None
        self.job_lock = threading.Lock()

        # Log storage per job
        self.job_logs: Dict[str, deque] = {}

        # Active subprocess
        self._process: Optional[subprocess.Popen] = None
        self._training_thread: Optional[threading.Thread] = None

        logger.info(f"TrainingService initialized. Datasets: {self.datasets_path}, Outputs: {self.outputs_path}")

    def validate_dataset(self, repo_id: str, policy_type: str) -> ValidationResult:
        """
        Validates that a dataset is compatible with a given policy type.
        Returns validation result with errors, warnings, and detected features.
        """
        errors = []
        warnings = []
        features = {}

        # Security check
        if ".." in repo_id:
            return ValidationResult(
                valid=False,
                errors=["Invalid repo_id: path traversal not allowed"],
                warnings=[],
                features={}
            )

        dataset_path = self.datasets_path / repo_id

        # Check dataset exists
        if not dataset_path.exists():
            return ValidationResult(
                valid=False,
                errors=[f"Dataset not found at {dataset_path}"],
                warnings=[],
                features={}
            )

        # Check info.json exists
        info_path = dataset_path / "meta" / "info.json"
        if not info_path.exists():
            return ValidationResult(
                valid=False,
                errors=["Dataset missing meta/info.json - not a valid LeRobot dataset"],
                warnings=[],
                features={}
            )

        # Load dataset info
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[f"Failed to read info.json: {e}"],
                warnings=[],
                features={}
            )

        features = info.get("features", {})
        total_episodes = info.get("total_episodes", 0)
        total_frames = info.get("total_frames", 0)

        # Check basic dataset validity
        if total_episodes == 0:
            errors.append("Dataset has no episodes")

        if total_frames == 0:
            errors.append("Dataset has no frames")

        # Policy-specific validation
        if policy_type == PolicyType.SMOLVLA.value or policy_type == "smolvla":
            self._validate_for_smolvla(features, info, errors, warnings, dataset_path)
        elif policy_type == PolicyType.DIFFUSION.value or policy_type == "diffusion":
            self._validate_for_diffusion(features, info, errors, warnings)
        elif policy_type == PolicyType.ACT.value or policy_type == "act":
            self._validate_for_act(features, info, errors, warnings)
        elif policy_type == PolicyType.PI05.value or policy_type == "pi05":
            self._validate_for_pi05(features, info, errors, warnings, dataset_path)
        else:
            warnings.append(f"Unknown policy type '{policy_type}', skipping policy-specific validation")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            features={
                "detected": list(features.keys()),
                "total_episodes": total_episodes,
                "total_frames": total_frames,
                "fps": info.get("fps"),
                "robot_type": info.get("robot_type"),
            }
        )

    def _validate_for_smolvla(
        self,
        features: dict,
        info: dict,
        errors: list,
        warnings: list,
        dataset_path: Path
    ):
        """SmolVLA-specific validation."""
        # Required: at least one camera image
        image_keys = [k for k in features.keys() if k.startswith("observation.images")]
        if not image_keys:
            errors.append("SmolVLA requires at least one camera (observation.images.*)")
        else:
            # Check image dimensions
            for key in image_keys:
                feat = features[key]
                shape = feat.get("shape", [])
                if len(shape) >= 2:
                    # Shape could be (C, H, W) or (H, W, C)
                    h, w = shape[-2], shape[-1]
                    if isinstance(feat.get("dtype"), str) and "image" in feat.get("dtype", ""):
                        # Video/image feature
                        pass
                    if h < 480 or w < 480:
                        warnings.append(f"{key} resolution may be low for SmolVLA (recommended: 512x512+)")

        # Required: observation.state
        if "observation.state" not in features:
            errors.append("SmolVLA requires observation.state (robot joint states)")
        else:
            state_shape = features["observation.state"].get("shape", [])
            if state_shape and state_shape[0] > 32:
                warnings.append(f"State dimension ({state_shape[0]}) exceeds SmolVLA max (32), will be truncated")

        # Required: action
        if "action" not in features:
            errors.append("SmolVLA requires action feature")
        else:
            action_shape = features["action"].get("shape", [])
            if action_shape and action_shape[0] > 32:
                warnings.append(f"Action dimension ({action_shape[0]}) exceeds SmolVLA max (32), will be truncated")

        # Optional but recommended: task descriptions
        # Check if task is in features (per-frame) or in episode metadata
        has_task = "task" in features
        if not has_task:
            # Also check episode metadata as fallback
            try:
                import pandas as pd
                episodes_dir = dataset_path / "meta" / "episodes"
                if episodes_dir.exists():
                    df = pd.read_parquet(episodes_dir)
                    if "task" in df.columns or "task_index" in df.columns:
                        has_task = True
            except Exception:
                pass

        if not has_task:
            warnings.append("Dataset missing 'task' descriptions - will use default task prompt")

    def _validate_for_diffusion(self, features: dict, info: dict, errors: list, warnings: list):
        """Diffusion policy validation."""
        # Requires images and actions
        image_keys = [k for k in features.keys() if k.startswith("observation.images") and not k.endswith("_depth")]
        if not image_keys:
            errors.append("Diffusion policy requires at least one camera (observation.images.*)")

        if "action" not in features:
            errors.append("Diffusion policy requires action feature")

        if "observation.state" not in features:
            warnings.append("observation.state recommended for diffusion policy")

        # Inform about auto-resize when cameras have different shapes
        if len(image_keys) > 1:
            shapes = {}
            for k in image_keys:
                shape = tuple(features[k].get("shape", []))
                if shape:
                    shapes[k] = shape

            unique_shapes = set(shapes.values())
            if len(unique_shapes) > 1:
                shape_info = ", ".join([f"{k}: {shapes[k]}" for k in sorted(shapes.keys())])
                warnings.append(
                    f"Cameras have different shapes ({shape_info}). "
                    f"All images will be resized to 480x640 during training."
                )

    def _validate_for_act(self, features: dict, info: dict, errors: list, warnings: list):
        """ACT policy validation."""
        # Similar to diffusion
        image_keys = [k for k in features.keys() if k.startswith("observation.images")]
        if not image_keys:
            errors.append("ACT policy requires at least one camera (observation.images.*)")

        if "action" not in features:
            errors.append("ACT policy requires action feature")

        if "observation.state" not in features:
            errors.append("ACT policy requires observation.state")

    def _validate_for_pi05(
        self,
        features: dict,
        info: dict,
        errors: list,
        warnings: list,
        dataset_path: Path
    ):
        """Pi0.5 policy validation."""
        # Requires images (at least one camera)
        image_keys = [k for k in features.keys() if k.startswith("observation.images") and not k.endswith("_depth")]
        if not image_keys:
            errors.append("Pi0.5 requires at least one camera (observation.images.*)")

        # Requires action
        if "action" not in features:
            errors.append("Pi0.5 requires action feature")
        else:
            action_shape = features["action"].get("shape", [])
            if action_shape and action_shape[0] > 32:
                warnings.append(f"Action dimension ({action_shape[0]}) exceeds Pi0.5 max (32), will be padded/truncated")

        # Requires observation.state
        if "observation.state" not in features:
            warnings.append("Pi0.5 works best with observation.state")
        else:
            state_shape = features["observation.state"].get("shape", [])
            if state_shape and state_shape[0] > 32:
                warnings.append(f"State dimension ({state_shape[0]}) exceeds Pi0.5 max (32), will be padded/truncated")

        # Check for quantile stats in stats.json (required for default normalization)
        stats_path = dataset_path / "meta" / "stats.json"
        has_quantiles = False
        if stats_path.exists():
            try:
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                # Check if action and state have q01/q99
                has_action_quantiles = (
                    "action" in stats and
                    "q01" in stats.get("action", {}) and
                    "q99" in stats.get("action", {})
                )
                has_state_quantiles = True  # Optional
                if "observation.state" in stats:
                    has_state_quantiles = "q01" in stats.get("observation.state", {})
                has_quantiles = has_action_quantiles and has_state_quantiles
            except Exception:
                pass

        if not has_quantiles:
            warnings.append(
                "Dataset missing quantile statistics (q01/q99). "
                "Either compute quantiles first, or use MEAN_STD normalization fallback."
            )

    def _get_dataset_features(self, dataset_repo_id: str) -> dict:
        """Extract feature information from dataset metadata."""
        dataset_path = self.datasets_path / dataset_repo_id / "meta" / "info.json"
        with open(dataset_path) as f:
            info = json.load(f)
        return info.get("features", {})

    def _build_smolvla_rename_map(self, features: dict) -> dict:
        """Build rename_map to translate dataset features to SmolVLA expected format.

        SmolVLA expects: observation.images.camera1, camera2, camera3
        Datasets may have: camera_1, left_cam, wrist, etc.
        """
        rename_map = {}

        # Find all image features (exclude depth cameras - SmolVLA doesn't use them by default)
        image_features = sorted([
            k for k in features.keys()
            if k.startswith("observation.images.")
            and not k.endswith("_depth")
        ])

        # Map to camera1, camera2, camera3, etc.
        for idx, feature_name in enumerate(image_features, start=1):
            expected_name = f"observation.images.camera{idx}"
            if feature_name != expected_name:
                rename_map[feature_name] = expected_name

        return rename_map

    def _build_diffusion_rename_map(self, features: dict) -> dict:
        """Build rename_map for diffusion policy - rename ALL cameras to standard format.

        Since we use resize_shape, all cameras can be used regardless of resolution.
        Images will be resized to a common shape during training.
        """
        rename_map = {}

        # Find all image features (exclude depth cameras)
        image_cameras = []
        for k in sorted(features.keys()):
            if k.startswith("observation.images.") and not k.endswith("_depth"):
                image_cameras.append(k)

        if not image_cameras:
            return rename_map

        # Map ALL cameras to camera1, camera2, etc. format
        for idx, feature_name in enumerate(image_cameras, start=1):
            expected_name = f"observation.images.camera{idx}"
            if feature_name != expected_name:
                rename_map[feature_name] = expected_name

        return rename_map

    def create_job(
        self,
        dataset_repo_id: str,
        policy_type: str,
        config: dict
    ) -> TrainingJob:
        """Creates a new training job (does not start it)."""
        job_id = str(uuid.uuid4())[:8]

        # Merge with defaults
        if policy_type == "smolvla":
            merged_config = {**SMOLVLA_DEFAULTS, **config}
        elif policy_type == "diffusion":
            merged_config = {**DIFFUSION_DEFAULTS, **config}
        elif policy_type == "pi05":
            merged_config = {**PI05_DEFAULTS, **config}
        elif policy_type == "act":
            merged_config = {**ACT_DEFAULTS, **config}
        else:
            merged_config = config

        # Create unique output directory path with timestamp to avoid conflicts
        # NOTE: Don't create the directory - LeRobot expects a non-existent path and will create it
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy_name = merged_config.get("policy_name", "").strip()
        if policy_name:
            # Sanitize: replace spaces with underscores, remove special chars
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', policy_name.replace(' ', '_'))
            output_dir = self.outputs_path / f"{safe_name}_{job_id}_{timestamp}"
        else:
            output_dir = self.outputs_path / f"{policy_type}_{job_id}_{timestamp}"

        # If directory somehow exists, remove it first (LeRobot requires fresh directory)
        if output_dir.exists():
            shutil.rmtree(output_dir)

        # Ensure parent directory exists (but NOT the output_dir itself)
        self.outputs_path.mkdir(parents=True, exist_ok=True)

        job = TrainingJob(
            id=job_id,
            status=JobStatus.PENDING,
            policy_type=PolicyType(policy_type),
            dataset_repo_id=dataset_repo_id,
            config=merged_config,
            created_at=datetime.now(),
            output_dir=output_dir,
            progress=TrainingProgress(total_steps=merged_config.get("steps", 100000))
        )

        with self.job_lock:
            self.jobs[job_id] = job
            self.job_logs[job_id] = deque(maxlen=1000)

        logger.info(f"Created training job {job_id} for {policy_type} on {dataset_repo_id}")
        return job

    def start_job(self, job_id: str) -> bool:
        """Starts a pending training job in a background thread."""
        with self.job_lock:
            if job_id not in self.jobs:
                raise ValueError(f"Job {job_id} not found")

            job = self.jobs[job_id]

            if job.status != JobStatus.PENDING:
                raise ValueError(f"Job {job_id} is not pending (status: {job.status})")

            if self.active_job is not None:
                raise ValueError(f"Another job is already running: {self.active_job.id}")

            job.status = JobStatus.VALIDATING
            job.started_at = datetime.now()
            self.active_job = job

        # Start training in background thread
        self._training_thread = threading.Thread(
            target=self._run_training,
            args=(job,),
            daemon=True
        )
        self._training_thread.start()

        return True

    def cancel_job(self, job_id: str) -> bool:
        """Cancels a running training job."""
        with self.job_lock:
            if job_id not in self.jobs:
                raise ValueError(f"Job {job_id} not found")

            job = self.jobs[job_id]

            if job.status not in [JobStatus.VALIDATING, JobStatus.TRAINING]:
                raise ValueError(f"Job {job_id} cannot be cancelled (status: {job.status})")

            # Terminate subprocess
            if self._process and self._process.poll() is None:
                logger.info(f"Terminating training process for job {job_id}")
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            self.active_job = None

        logger.info(f"Cancelled training job {job_id}")
        return True

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Returns a job by ID."""
        return self.jobs.get(job_id)

    def get_job_status(self, job_id: str) -> dict:
        """Returns the current status and progress of a job."""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        return job.to_dict()

    def get_job_logs(self, job_id: str, offset: int = 0, limit: int = 100) -> dict:
        """Returns logs for a job."""
        if job_id not in self.job_logs:
            raise ValueError(f"Job {job_id} not found")

        logs = list(self.job_logs[job_id])
        total = len(logs)

        # Apply offset and limit
        logs = logs[offset:offset + limit]

        return {
            "logs": logs,
            "total": total,
            "has_more": offset + limit < total
        }

    def list_jobs(self) -> List[dict]:
        """Returns all jobs."""
        return [job.to_dict() for job in self.jobs.values()]

    def get_presets(self, policy_type: str = "smolvla") -> dict:
        """Returns available training presets for a policy type."""
        if policy_type == "smolvla":
            return SMOLVLA_PRESETS
        elif policy_type == "diffusion":
            return DIFFUSION_PRESETS
        elif policy_type == "pi05":
            return PI05_PRESETS
        elif policy_type == "act":
            return ACT_PRESETS
        return {}

    def detect_hardware(self) -> dict:
        """Detect available training hardware (CUDA, MPS, CPU)."""
        try:
            import torch
        except ImportError:
            return {
                "devices": [{"id": "cpu", "type": "cpu", "name": "CPU", "memory_gb": None, "recommended": True}],
                "default": "cpu"
            }

        devices = []
        cuda_warning = None

        # Check CUDA (NVIDIA GPU) - with functional test
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory // (1024**3)

                    # Functional test: actually try to use the GPU
                    # This catches compute capability mismatches (e.g., RTX 5090 with old PyTorch)
                    test_tensor = torch.zeros(1, device=f"cuda:{i}")
                    del test_tensor

                    devices.append({
                        "id": f"cuda:{i}",
                        "type": "cuda",
                        "name": name,
                        "memory_gb": memory_gb,
                        "recommended": True
                    })
                except RuntimeError as e:
                    # GPU detected but not usable (compute capability mismatch)
                    error_msg = str(e)
                    try:
                        name = torch.cuda.get_device_name(i)
                    except:
                        name = f"GPU {i}"
                    if "capability" in error_msg.lower() or "sm_" in error_msg.lower() or "no kernel image" in error_msg.lower():
                        cuda_warning = f"{name} detected but not compatible with this PyTorch version. Install PyTorch nightly with CUDA 12.8+ for Blackwell support."
                        logger.warning(cuda_warning)
                    else:
                        logger.warning(f"CUDA device {i} ({name}) not usable: {e}")
                except Exception as e:
                    logger.warning(f"Could not use CUDA device {i}: {e}")

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                # Functional test for MPS too
                test_tensor = torch.zeros(1, device="mps")
                del test_tensor
                devices.append({
                    "id": "mps",
                    "type": "mps",
                    "name": "Apple Silicon GPU",
                    "memory_gb": None,
                    "recommended": True
                })
            except Exception as e:
                logger.warning(f"MPS not usable: {e}")

        # Always include CPU as fallback
        devices.append({
            "id": "cpu",
            "type": "cpu",
            "name": "CPU",
            "memory_gb": None,
            "recommended": len(devices) == 0  # Only recommended if no GPU available
        })

        default_device = devices[0]["id"] if devices else "cpu"

        result = {
            "devices": devices,
            "default": default_device
        }

        # Include warning if GPU was detected but not usable
        if cuda_warning:
            result["warning"] = cuda_warning

        return result

    def _run_training(self, job: TrainingJob):
        """Executes training in a subprocess. Runs in background thread."""
        try:
            # First validate dataset
            validation = self.validate_dataset(job.dataset_repo_id, job.policy_type.value)
            if not validation.valid:
                job.status = JobStatus.FAILED
                job.error = f"Dataset validation failed: {'; '.join(validation.errors)}"
                job.completed_at = datetime.now()
                with self.job_lock:
                    self.active_job = None
                return

            # Update status
            job.status = JobStatus.TRAINING
            self._add_log(job.id, f"Starting {job.policy_type.value} training on {job.dataset_repo_id}")

            # Build command
            cmd = self._build_training_command(job)
            self._add_log(job.id, f"Command: {' '.join(cmd)}")

            # Set up environment
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            # Add lerobot to path
            lerobot_src = _PROJECT_ROOT / "lerobot" / "src"
            if "PYTHONPATH" in env:
                env["PYTHONPATH"] = f"{lerobot_src}:{env['PYTHONPATH']}"
            else:
                env["PYTHONPATH"] = str(lerobot_src)

            # Handle device selection via environment variables
            device = job.config.get("device", "auto")
            if device == "cpu":
                # Force CPU by hiding CUDA devices
                env["CUDA_VISIBLE_DEVICES"] = ""
                self._add_log(job.id, "Device: CPU (forced via CUDA_VISIBLE_DEVICES='')")
            elif device.startswith("cuda:"):
                # Specific GPU - extract index
                gpu_id = device.split(":")[1]
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                self._add_log(job.id, f"Device: {device} (CUDA_VISIBLE_DEVICES={gpu_id})")
            elif device == "mps":
                # Apple Silicon - handled automatically by PyTorch/Accelerate
                self._add_log(job.id, "Device: MPS (Apple Silicon)")
            else:
                # Auto-detect - let Accelerate handle it
                self._add_log(job.id, "Device: auto (using best available)")

            # Start subprocess with separate stderr capture
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Capture stderr separately
                text=True,
                bufsize=1,
                env=env,
                cwd=str(_PROJECT_ROOT)
            )

            job.pid = self._process.pid
            self._add_log(job.id, f"Training process started with PID {job.pid}")

            # Use threads to read both stdout and stderr without blocking
            stderr_lines = []

            def read_stderr():
                """Read stderr in background thread."""
                for line in self._process.stderr:
                    line = line.strip()
                    if line:
                        # Parse progress from stderr (LeRobot logs training progress here)
                        self._parse_training_output(job, line)
                        if not self._should_filter_log_line(line):
                            stderr_lines.append(line)
                            self._add_log(job.id, f"[STDERR] {line}")

            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

            # Stream stdout and parse progress
            for line in self._process.stdout:
                line = line.strip()
                if line:
                    # Always parse for progress, even if we filter the log display
                    self._parse_training_output(job, line)
                    # Only add to visible logs if not filtered
                    if not self._should_filter_log_line(line):
                        self._add_log(job.id, line)

            # Wait for stderr thread to finish
            stderr_thread.join(timeout=5.0)

            # Wait for completion
            return_code = self._process.wait()

            # Log any remaining stderr if there were errors
            if return_code != 0 and stderr_lines:
                self._add_log(job.id, f"--- Process exited with errors (code {return_code}) ---")

            if return_code == 0:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                self._save_policy_metadata(job)
                self._add_log(job.id, "Training completed successfully!")
            elif job.status != JobStatus.CANCELLED:
                job.status = JobStatus.FAILED
                job.error = f"Training process exited with code {return_code}"
                job.completed_at = datetime.now()
                self._save_policy_metadata(job)  # Save metadata even on failure
                self._add_log(job.id, f"Training failed with exit code {return_code}")

        except Exception as e:
            logger.exception(f"Training job {job.id} failed with exception")
            job.status = JobStatus.FAILED
            job.error = str(e)
            self._add_log(job.id, f"Error: {e}")

        finally:
            job.completed_at = datetime.now()
            with self.job_lock:
                self.active_job = None
                self._process = None

    def _build_training_command(self, job: TrainingJob) -> List[str]:
        """Builds the command to run LeRobot training."""
        config = job.config

        # Base command using lerobot train script (local submodule uses lerobot_ prefix)
        cmd = [
            sys.executable, "-m", "lerobot.scripts.lerobot_train",
        ]

        # Dataset configuration
        # Note: LeRobot expects root to be the full dataset path, not just the parent directory
        dataset_full_path = self.datasets_path / job.dataset_repo_id
        cmd.extend([
            f"--dataset.repo_id={job.dataset_repo_id}",
            f"--dataset.root={dataset_full_path}",
            "--dataset.video_backend=pyav",  # Use pyav (more stable than torchcodec)
        ])

        # Policy configuration
        if job.policy_type == PolicyType.SMOLVLA:
            # Use pretrained model path or type
            pretrained = config.get("pretrained_model", "lerobot/smolvla_base")
            cmd.append(f"--policy.path={pretrained}")

            # SmolVLA-specific settings
            if config.get("freeze_vision_encoder", True):
                cmd.append("--policy.freeze_vision_encoder=true")
            if config.get("train_expert_only", True):
                cmd.append("--policy.train_expert_only=true")
            if config.get("train_state_proj", True):
                cmd.append("--policy.train_state_proj=true")
        elif job.policy_type == PolicyType.DIFFUSION:
            # Diffusion policy - set policy type
            cmd.append("--policy.type=diffusion")

            # Diffusion-specific parameters
            if config.get("n_obs_steps"):
                cmd.append(f"--policy.n_obs_steps={config['n_obs_steps']}")
            if config.get("horizon"):
                cmd.append(f"--policy.horizon={config['horizon']}")
            if config.get("n_action_steps"):
                cmd.append(f"--policy.n_action_steps={config['n_action_steps']}")
            if config.get("noise_scheduler_type"):
                cmd.append(f"--policy.noise_scheduler_type={config['noise_scheduler_type']}")
            if config.get("num_train_timesteps"):
                cmd.append(f"--policy.num_train_timesteps={config['num_train_timesteps']}")
            if config.get("vision_backbone"):
                cmd.append(f"--policy.vision_backbone={config['vision_backbone']}")
        elif job.policy_type == PolicyType.PI05:
            # Pi0.5 policy - set policy type
            cmd.append("--policy.type=pi05")

            # Pretrained model path
            pretrained = config.get("pretrained_path", "lerobot/pi05_base")
            cmd.append(f"--policy.pretrained_path={pretrained}")

            # Compile model (disabled by default to save memory during warmup)
            if config.get("compile_model", False):
                cmd.append("--policy.compile_model=true")
            else:
                cmd.append("--policy.compile_model=false")

            # Gradient checkpointing (recommended for memory)
            if config.get("gradient_checkpointing", True):
                cmd.append("--policy.gradient_checkpointing=true")

            # Data type (bfloat16 recommended)
            dtype = config.get("dtype", "bfloat16")
            cmd.append(f"--policy.dtype={dtype}")

            # LoRA settings (for memory-efficient fine-tuning)
            lora_rank = config.get("lora_rank", 8)
            if lora_rank > 0:
                cmd.append(f"--policy.lora_rank={lora_rank}")
                cmd.append(f"--policy.lora_alpha={config.get('lora_alpha', 16)}")
                cmd.append(f"--policy.lora_dropout={config.get('lora_dropout', 0.1)}")

            # Chunk size and action steps
            if config.get("chunk_size"):
                cmd.append(f"--policy.chunk_size={config['chunk_size']}")
            if config.get("n_action_steps"):
                cmd.append(f"--policy.n_action_steps={config['n_action_steps']}")

            # Handle normalization - use MEAN_STD fallback if quantiles not available
            if not config.get("use_quantile_normalization", True):
                normalization_map = '{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'
                cmd.append(f"--policy.normalization_mapping={normalization_map}")
        elif job.policy_type == PolicyType.ACT:
            # ACT (Action Chunking with Transformers)
            cmd.append("--policy.type=act")

            # Action chunking parameters
            if config.get("chunk_size"):
                cmd.append(f"--policy.chunk_size={config['chunk_size']}")
            if config.get("n_action_steps"):
                cmd.append(f"--policy.n_action_steps={config['n_action_steps']}")
            if config.get("n_obs_steps"):
                cmd.append(f"--policy.n_obs_steps={config['n_obs_steps']}")

            # VAE settings
            if config.get("use_vae") is not None:
                cmd.append(f"--policy.use_vae={'true' if config.get('use_vae', True) else 'false'}")
            if config.get("latent_dim"):
                cmd.append(f"--policy.latent_dim={config['latent_dim']}")
            if config.get("kl_weight") is not None:
                cmd.append(f"--policy.kl_weight={config['kl_weight']}")

            # Architecture
            if config.get("vision_backbone"):
                cmd.append(f"--policy.vision_backbone={config['vision_backbone']}")
            if config.get("dim_model"):
                cmd.append(f"--policy.dim_model={config['dim_model']}")
        else:
            # Generic policy type
            cmd.append(f"--policy.type={job.policy_type.value}")

        # Training parameters
        cmd.extend([
            f"--steps={config.get('steps', 100000)}",
            f"--batch_size={config.get('batch_size', 8)}",
            f"--num_workers={config.get('num_workers', 4)}",
            f"--save_freq={config.get('save_freq', 20000)}",
            f"--eval_freq={config.get('eval_freq', 20000)}",
            f"--log_freq=200",
            f"--output_dir={job.output_dir}",
        ])

        # Learning rate
        if "learning_rate" in config:
            cmd.append(f"--policy.optimizer_lr={config['learning_rate']}")

        # Warmup steps
        if "warmup_steps" in config:
            cmd.append(f"--policy.scheduler_warmup_steps={config['warmup_steps']}")

        # Disable push to hub by default (user can push manually later)
        cmd.append("--policy.push_to_hub=false")

        # Build dynamic rename_map for feature compatibility (SmolVLA needs camera1, camera2 format)
        if job.policy_type == PolicyType.SMOLVLA:
            try:
                features = self._get_dataset_features(job.dataset_repo_id)
                rename_map = self._build_smolvla_rename_map(features)
                if rename_map:
                    cmd.append(f"--rename_map={json.dumps(rename_map)}")
                    logger.info(f"[Job {job.id}] Feature rename map: {rename_map}")
            except Exception as e:
                logger.warning(f"[Job {job.id}] Could not build rename_map: {e}")

        # Diffusion also needs rename_map to filter cameras with matching shapes
        elif job.policy_type == PolicyType.DIFFUSION:
            try:
                features = self._get_dataset_features(job.dataset_repo_id)

                # Get target resize shape (use config or default)
                resize_shape = config.get("resize_shape", [480, 640])
                cmd.append(f"--policy.resize_shape=[{resize_shape[0]},{resize_shape[1]}]")

                # Build rename_map for ALL cameras
                rename_map = self._build_diffusion_rename_map(features)
                if rename_map:
                    cmd.append(f"--rename_map={json.dumps(rename_map)}")

                logger.info(f"[Job {job.id}] Diffusion using resize_shape={resize_shape}, rename_map={rename_map}")
            except Exception as e:
                logger.warning(f"[Job {job.id}] Could not configure diffusion multi-camera: {e}")

        # Note: Pi0.5 does NOT need rename_map - it uses the dataset's original feature names
        # (unlike SmolVLA which requires camera1, camera2, ... format)

        return cmd

    def _parse_training_output(self, job: TrainingJob, line: str):
        """Parses training output to extract progress info.

        LeRobot outputs in format: step:1K smpl:10K ep:13 epch:13.15 loss:0.053 grdn:0.974 lr:1.0e-04
        """

        def parse_number_with_suffix(value: str) -> int:
            """Parse numbers with K/M suffix (e.g., '1K' -> 1000, '10M' -> 10000000)."""
            value = value.strip().upper()
            if value.endswith('K'):
                return int(float(value[:-1]) * 1000)
            elif value.endswith('M'):
                return int(float(value[:-1]) * 1000000)
            else:
                return int(float(value))

        # Pattern: "step:1K" or "step: 1000" (no space or with space, with K/M suffix)
        step_match = re.search(r"(?:step)[:\s]*([0-9.]+[KkMm]?)", line, re.IGNORECASE)
        if step_match:
            try:
                job.progress.step = parse_number_with_suffix(step_match.group(1))
            except (ValueError, IndexError):
                pass

        # Pattern: "loss:0.053" or "loss: 0.123" (no space or with space)
        loss_match = re.search(r"(?:loss)[:\s]*([\d.]+)", line, re.IGNORECASE)
        if loss_match:
            try:
                loss = float(loss_match.group(1))
                job.progress.loss = loss
                # Track loss history for graphing (keep last 500 points)
                if job.progress.step > 0:
                    job.progress.loss_history.append([job.progress.step, loss])
                    if len(job.progress.loss_history) > 500:
                        job.progress.loss_history = job.progress.loss_history[-500:]
            except ValueError:
                pass

        # Pattern: "lr:1.0e-04" or "lr: 1e-4" or "learning_rate: 0.0001"
        lr_match = re.search(r"(?:lr|learning_rate)[:\s]*([\d.e+-]+)", line, re.IGNORECASE)
        if lr_match:
            try:
                job.progress.learning_rate = float(lr_match.group(1))
            except ValueError:
                pass

        # Pattern: "ep:13" or "epch:13.15" or "epoch: 1" (handle both abbreviated and full)
        epoch_match = re.search(r"(?:epch|epoch|ep)[:\s]*([\d.]+)", line, re.IGNORECASE)
        if epoch_match:
            try:
                job.progress.epoch = int(float(epoch_match.group(1)))
            except ValueError:
                pass

        # Calculate ETA based on progress
        if job.progress.step > 0 and job.started_at:
            elapsed = (datetime.now() - job.started_at).total_seconds()
            steps_remaining = job.progress.total_steps - job.progress.step
            if job.progress.step > 0:
                time_per_step = elapsed / job.progress.step
                job.progress.eta_seconds = int(steps_remaining * time_per_step)

    def _should_filter_log_line(self, line: str) -> bool:
        """Returns True if the log line should be filtered (not shown to user).

        Filters out common warning spam that clutters the training log.
        """
        filter_patterns = [
            # Tokenizers parallelism warnings (full message text)
            "The current process just got forked",
            "Disabling parallelism",
            "TOKENIZERS_PARALLELISM",
            "avoid deadlocks",
            "To disable this warning, you can either",
            "Avoid using `tokenizers` before the fork",
            "explicitly set TOKENIZERS_PARALLELISM",

            # Torchvision deprecation warnings (full message text)
            "video decoding and encoding capabilities of torchvision are deprecated",
            "will be removed in version 0.24",
            "We recommend that you migrate to TorchCodec",
            "torchcodec",
            "torchvision.transforms.functional_tensor",
            "torchvision.transforms.v2.functional",
            "UserWarning: The torchvision.datapoints",
            "is deprecated since 0.15",

            # warnings.warn() source line
            "warnings.warn(",

            # Common harmless warnings
            "UserWarning: TypedStorage is deprecated",
            "please use torch.Tensor.untyped_storage",

            # HuggingFace warnings
            "Some weights of the model checkpoint",
            "were not used when initializing",

            # tqdm/progress bar partial lines
            "\r ",
            "\x1b[",  # ANSI escape codes
        ]
        line_lower = line.lower()
        return any(pattern.lower() in line_lower for pattern in filter_patterns)

    def _add_log(self, job_id: str, message: str):
        """Adds a log message to a job's log buffer."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        if job_id in self.job_logs:
            self.job_logs[job_id].append(log_entry)

        logger.info(f"[Job {job_id}] {message}")

    def _save_policy_metadata(self, job: TrainingJob):
        """Save policy metadata and loss history after training completes."""
        if not job.output_dir or not job.output_dir.exists():
            return

        # Save loss history
        loss_history_path = job.output_dir / "loss_history.json"
        try:
            with open(loss_history_path, "w") as f:
                json.dump(job.progress.loss_history, f)
        except Exception as e:
            logger.warning(f"Failed to save loss history: {e}")

        # Save policy metadata
        metadata_path = job.output_dir / "policy_metadata.json"
        metadata = {
            "name": job.config.get("policy_name", job.id),
            "policy_type": job.policy_type.value,
            "dataset_repo_id": job.dataset_repo_id,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "total_steps": job.progress.total_steps,
            "final_step": job.progress.step,
            "final_loss": job.progress.loss,
            "config": job.config,
        }
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save policy metadata: {e}")

    def has_quantile_stats(self, repo_id: str) -> dict:
        """Check if dataset has quantile statistics for Pi0.5 training.

        Returns:
            dict with keys: has_quantiles (bool), missing_features (list), message (str)
        """
        dataset_path = self.datasets_path / repo_id
        stats_path = dataset_path / "meta" / "stats.json"

        result = {
            "has_quantiles": False,
            "missing_features": [],
            "message": ""
        }

        if not dataset_path.exists():
            result["message"] = f"Dataset not found: {repo_id}"
            return result

        if not stats_path.exists():
            result["message"] = "No stats.json found in dataset metadata"
            return result

        try:
            with open(stats_path, "r") as f:
                stats = json.load(f)

            required_quantile_keys = ["q01", "q99"]
            features_to_check = ["action", "observation.state"]

            for feature in features_to_check:
                if feature in stats:
                    feature_stats = stats[feature]
                    if not all(k in feature_stats for k in required_quantile_keys):
                        result["missing_features"].append(feature)

            result["has_quantiles"] = len(result["missing_features"]) == 0
            if result["has_quantiles"]:
                result["message"] = "Dataset has quantile statistics"
            else:
                result["message"] = f"Missing quantile stats for: {', '.join(result['missing_features'])}"

        except Exception as e:
            result["message"] = f"Error reading stats: {e}"

        return result

    def compute_quantile_stats(self, repo_id: str) -> dict:
        """Compute quantile statistics for a dataset using LeRobot's augment script.

        This runs synchronously and may take a while for large datasets.
        Returns status dict with success/error message.
        """
        dataset_path = self.datasets_path / repo_id
        if not dataset_path.exists():
            return {"status": "error", "message": f"Dataset not found: {repo_id}"}

        # Build command to run the augment script
        lerobot_src = _PROJECT_ROOT / "lerobot" / "src"
        script_path = lerobot_src / "lerobot" / "datasets" / "v30" / "augment_dataset_quantile_stats.py"

        if not script_path.exists():
            return {"status": "error", "message": f"Quantile augment script not found at {script_path}"}

        cmd = [
            sys.executable,
            str(script_path),
            f"--repo-id={repo_id}",
            f"--root={dataset_path}",
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(lerobot_src)

        logger.info(f"Computing quantile stats for {repo_id}: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=str(_PROJECT_ROOT),
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"Quantile stats computed successfully for {repo_id}")
                return {
                    "status": "success",
                    "message": "Quantile statistics computed successfully"
                }
            else:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                logger.error(f"Quantile computation failed for {repo_id}: {error_msg}")
                return {
                    "status": "error",
                    "message": f"Script failed: {error_msg}"
                }
        except subprocess.TimeoutExpired:
            logger.error(f"Quantile computation timed out for {repo_id}")
            return {
                "status": "error",
                "message": "Quantile computation timed out (>10 minutes)"
            }
        except Exception as e:
            logger.error(f"Quantile computation error for {repo_id}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def list_policies(self) -> List[PolicyInfo]:
        """Scan training/outputs/ directory and return all trained policies."""
        policies = []

        if not self.outputs_path.exists():
            return policies

        for output_dir in self.outputs_path.iterdir():
            if not output_dir.is_dir():
                continue

            policy = self._parse_policy_directory(output_dir)
            if policy:
                policies.append(policy)

        # Also include currently training job
        if self.active_job and self.active_job.output_dir:
            # Check if not already in list
            active_id = self.active_job.output_dir.name
            if not any(p.id == active_id for p in policies):
                policies.append(PolicyInfo(
                    id=active_id,
                    name=self.active_job.config.get("policy_name", active_id),
                    policy_type=self.active_job.policy_type.value,
                    status="training",
                    steps=self.active_job.progress.step,
                    total_steps=self.active_job.progress.total_steps,
                    dataset_repo_id=self.active_job.dataset_repo_id,
                    created_at=self.active_job.created_at.isoformat(),
                    final_loss=self.active_job.progress.loss,
                    checkpoint_path="",
                    loss_history=self.active_job.progress.loss_history,
                    output_dir=str(self.active_job.output_dir),
                ))

        # Sort by created_at descending (newest first)
        policies.sort(key=lambda p: p.created_at, reverse=True)
        return policies

    def _parse_policy_directory(self, output_dir: Path) -> Optional[PolicyInfo]:
        """Parse a training output directory to extract policy info."""
        dir_name = output_dir.name

        # Try to load metadata file first
        metadata_path = output_dir / "policy_metadata.json"
        loss_history_path = output_dir / "loss_history.json"

        metadata = {}
        loss_history = []

        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except Exception:
                pass

        if loss_history_path.exists():
            try:
                with open(loss_history_path, "r") as f:
                    loss_history = json.load(f)
            except Exception:
                pass

        # Parse directory name: {policy_type}_{job_id}_{timestamp} or {name}_{job_id}_{timestamp}
        parts = dir_name.rsplit("_", 2)
        if len(parts) >= 2:
            # Could be name_jobid_timestamp or policytype_jobid_timestamp
            policy_type = metadata.get("policy_type", "unknown")
            if not policy_type or policy_type == "unknown":
                # Try to infer from directory name
                first_part = parts[0].lower()
                if first_part in ["smolvla", "diffusion", "act", "pi05"]:
                    policy_type = first_part
                else:
                    policy_type = "smolvla"  # Default assumption
        else:
            policy_type = metadata.get("policy_type", "smolvla")

        # Check for checkpoints
        checkpoint_path = ""
        checkpoints_dir = output_dir / "checkpoints"
        last_checkpoint = checkpoints_dir / "last" / "pretrained_model"

        if last_checkpoint.exists():
            checkpoint_path = str(last_checkpoint)
            status = "completed"
        elif checkpoints_dir.exists() and any(checkpoints_dir.iterdir()):
            # Has some checkpoints but not "last" - might be interrupted
            status = "failed"
            # Find latest checkpoint
            for ckpt in sorted(checkpoints_dir.iterdir(), reverse=True):
                if ckpt.is_dir() and (ckpt / "pretrained_model").exists():
                    checkpoint_path = str(ckpt / "pretrained_model")
                    break
        else:
            # No checkpoints - might be empty/failed early
            status = "failed"

        # Get steps from train_config.json in checkpoint if available
        steps = metadata.get("final_step", 0)
        total_steps = metadata.get("total_steps", 0)
        final_loss = metadata.get("final_loss")

        if checkpoint_path and not steps:
            # Try to read from checkpoint
            train_config_path = Path(checkpoint_path) / "train_config.json"
            if train_config_path.exists():
                try:
                    with open(train_config_path, "r") as f:
                        train_config = json.load(f)
                        total_steps = train_config.get("steps", 0)
                except Exception:
                    pass

            # Try to get step from training_step.json in training_state
            training_state_dir = Path(checkpoint_path).parent / "training_state"
            step_file = training_state_dir / "training_step.json"
            if step_file.exists():
                try:
                    with open(step_file, "r") as f:
                        step_data = json.load(f)
                        steps = step_data.get("step", 0)
                except Exception:
                    pass

        # Get created_at from metadata or directory timestamp
        created_at = metadata.get("created_at")
        if not created_at:
            # Parse from directory name: ..._YYYYMMDD_HHMMSS
            try:
                timestamp_str = parts[-1] if len(parts) >= 3 else ""
                date_str = parts[-2] if len(parts) >= 3 else ""
                if timestamp_str and date_str:
                    dt = datetime.strptime(f"{date_str}_{timestamp_str}", "%Y%m%d_%H%M%S")
                    created_at = dt.isoformat()
            except Exception:
                pass

        if not created_at:
            # Fallback to directory modification time
            created_at = datetime.fromtimestamp(output_dir.stat().st_mtime).isoformat()

        # Get display name
        name = metadata.get("name", dir_name)

        return PolicyInfo(
            id=dir_name,
            name=name,
            policy_type=policy_type,
            status=status,
            steps=steps,
            total_steps=total_steps,
            dataset_repo_id=metadata.get("dataset_repo_id", ""),
            created_at=created_at,
            final_loss=final_loss,
            checkpoint_path=checkpoint_path,
            loss_history=loss_history,
            output_dir=str(output_dir),
        )

    def get_policy(self, policy_id: str) -> Optional[PolicyInfo]:
        """Get a specific policy by ID."""
        # Check if it's the active training job
        if self.active_job and self.active_job.output_dir and self.active_job.output_dir.name == policy_id:
            return PolicyInfo(
                id=policy_id,
                name=self.active_job.config.get("policy_name", policy_id),
                policy_type=self.active_job.policy_type.value,
                status="training",
                steps=self.active_job.progress.step,
                total_steps=self.active_job.progress.total_steps,
                dataset_repo_id=self.active_job.dataset_repo_id,
                created_at=self.active_job.created_at.isoformat(),
                final_loss=self.active_job.progress.loss,
                checkpoint_path="",
                loss_history=self.active_job.progress.loss_history,
                output_dir=str(self.active_job.output_dir),
            )

        # Look in outputs directory
        output_dir = self.outputs_path / policy_id
        if output_dir.exists():
            return self._parse_policy_directory(output_dir)

        return None

    def get_policy_config(self, policy_id: str) -> Optional[PolicyConfig]:
        """Parse the policy's config.json to extract input/output features.

        Returns:
            PolicyConfig with cameras, arms, and dimensions extracted,
            or None if policy or config not found.
        """
        policy = self.get_policy(policy_id)
        if not policy or not policy.checkpoint_path:
            return None

        config_path = Path(policy.checkpoint_path) / "config.json"
        if not config_path.exists():
            logger.warning(f"[TrainingService] No config.json found at {config_path}")
            return None

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"[TrainingService] Failed to read config.json: {e}")
            return None

        # Extract cameras from input_features
        cameras = []
        input_features = config.get("input_features", {})
        for key in input_features:
            if key.startswith("observation.images."):
                camera_name = key.replace("observation.images.", "")
                cameras.append(camera_name)

        # Extract state dimension
        state_shape = input_features.get("observation.state", {}).get("shape", [14])
        state_dim = state_shape[0] if state_shape else 14

        # Extract action dimension
        output_features = config.get("output_features", {})
        action_shape = output_features.get("action", {}).get("shape", [14])
        action_dim = action_shape[0] if action_shape else 14

        # Infer which arms were used
        arms = self._infer_arms_from_policy(policy, state_dim)

        policy_type = config.get("type", "unknown")

        logger.info(f"[TrainingService] Policy config: cameras={cameras}, arms={arms}, state_dim={state_dim}")

        return PolicyConfig(
            cameras=cameras,
            arms=arms,
            state_dim=state_dim,
            action_dim=action_dim,
            policy_type=policy_type
        )

    def _infer_arms_from_policy(self, policy: PolicyInfo, state_dim: int) -> List[str]:
        """Infer which arms were used by checking dataset info.json or dimension heuristics."""
        # First try to get from dataset metadata
        if policy.dataset_repo_id:
            info_path = self.datasets_path / policy.dataset_repo_id / "meta" / "info.json"
            if info_path.exists():
                try:
                    with open(info_path, "r") as f:
                        info = json.load(f)
                    state_names = info.get("features", {}).get("observation.state", {}).get("names", [])
                    arms = set()
                    for name in state_names:
                        if name.startswith("left_"):
                            arms.add("left")
                        elif name.startswith("right_"):
                            arms.add("right")
                    if arms:
                        return list(arms)
                except Exception as e:
                    logger.warning(f"[TrainingService] Failed to read dataset info.json: {e}")

        # Fallback: dimension-based heuristic
        # 7 DOF = single arm (default to left), 14 DOF = both arms
        if state_dim <= 7:
            return ["left"]
        else:
            return ["left", "right"]

    def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy and its output directory."""
        # Don't delete if it's currently training
        if self.active_job and self.active_job.output_dir and self.active_job.output_dir.name == policy_id:
            raise ValueError("Cannot delete a policy that is currently training")

        output_dir = self.outputs_path / policy_id
        if not output_dir.exists():
            raise ValueError(f"Policy {policy_id} not found")

        # Security check
        if ".." in policy_id:
            raise ValueError("Invalid policy_id")

        try:
            shutil.rmtree(output_dir)
            logger.info(f"Deleted policy {policy_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete policy {policy_id}: {e}")
            raise

    def rename_policy(self, policy_id: str, new_name: str) -> bool:
        """Rename a policy by updating its metadata."""
        output_dir = self.outputs_path / policy_id
        if not output_dir.exists():
            raise ValueError(f"Policy {policy_id} not found")

        metadata_path = output_dir / "policy_metadata.json"

        # Load existing metadata or create new
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except Exception:
                pass

        metadata["name"] = new_name

        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Renamed policy {policy_id} to '{new_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to rename policy {policy_id}: {e}")
            raise

    def resume_training(self, policy_id: str, additional_steps: int) -> TrainingJob:
        """Resume training from a checkpoint with additional steps."""
        policy = self.get_policy(policy_id)
        if not policy:
            raise ValueError(f"Policy {policy_id} not found")

        if policy.status == "training":
            raise ValueError("Policy is already training")

        if not policy.checkpoint_path:
            raise ValueError("No checkpoint found to resume from")

        # Load original config from metadata
        metadata_path = Path(policy.output_dir) / "policy_metadata.json"
        config = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    config = metadata.get("config", {})
            except Exception:
                pass

        # Update steps for continuation
        config["steps"] = policy.steps + additional_steps
        config["policy_name"] = policy.name
        config["resume_from_checkpoint"] = policy.checkpoint_path

        # Create new job
        job = self.create_job(
            dataset_repo_id=policy.dataset_repo_id,
            policy_type=policy.policy_type,
            config=config
        )

        # Start the job
        self.start_job(job.id)

        return job
