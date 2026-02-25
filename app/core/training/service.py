"""
Training Service for SmolVLA and other policy training.
Manages training jobs, validates datasets, and executes training in background subprocesses.
"""

import json
import logging
import re
import shutil
import subprocess
import threading
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .commands import CommandMixin
from .policies import PolicyMixin
from .presets import (
    ACT_DEFAULTS,
    ACT_PRESETS,
    DIFFUSION_DEFAULTS,
    DIFFUSION_PRESETS,
    PI05_DEFAULTS,
    PI05_PRESETS,
    SMOLVLA_DEFAULTS,
    SMOLVLA_PRESETS,
)
from .types import (
    _DEFAULT_DATASETS_PATH,
    _DEFAULT_OUTPUTS_PATH,
    JobStatus,
    PolicyInfo,
    PolicyType,
    TrainingJob,
    TrainingProgress,
)
from .validators import ValidatorMixin

logger = logging.getLogger(__name__)


class TrainingService(ValidatorMixin, CommandMixin, PolicyMixin):
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
                    except Exception:
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
