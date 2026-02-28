import json
import logging
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from .types import _PROJECT_ROOT, JobStatus, PolicyType, TrainingJob

logger = logging.getLogger(__name__)


class CommandMixin:
    """Mixin providing training command execution methods for TrainingService."""

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

    def _build_training_command(self, job: TrainingJob) -> list:
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
            f"--dataset.video_backend={config.get('video_backend', 'pyav')}",
        ])

        # Optional episode filtering (e.g. "0:76" or "0,1,2,5,10" or "[0,1,2]")
        if config.get("dataset_episodes"):
            episodes = config["dataset_episodes"]
            if isinstance(episodes, str):
                episodes = episodes.strip()
                if ":" in episodes:
                    # Range syntax: "0:76" → [0,1,2,...,75]
                    parts = episodes.split(":")
                    start, end = int(parts[0]), int(parts[1])
                    episode_list = list(range(start, end))
                    cmd.append(f"--dataset.episodes=[{','.join(str(i) for i in episode_list)}]")
                elif episodes.startswith("["):
                    # Already a list literal: "[0,1,2,3]"
                    cmd.append(f"--dataset.episodes={episodes}")
                else:
                    # Comma-separated: "0,1,2,3" → "[0,1,2,3]"
                    cmd.append(f"--dataset.episodes=[{episodes}]")
            elif isinstance(episodes, list):
                cmd.append(f"--dataset.episodes=[{','.join(str(i) for i in episodes)}]")

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

            # Temporal ensembling
            if config.get("temporal_ensemble_coeff") is not None:
                cmd.append(
                    f"--policy.temporal_ensemble_coeff={config['temporal_ensemble_coeff']}"
                )
        else:
            # Generic policy type
            cmd.append(f"--policy.type={job.policy_type.value}")

        # Output directory (allow override from config)
        output_dir = config.get("output_dir_custom") or str(job.output_dir)

        # Training parameters
        cmd.extend([
            f"--steps={config.get('steps', 100000)}",
            f"--batch_size={config.get('batch_size', 8)}",
            f"--num_workers={config.get('num_workers', 4)}",
            f"--save_freq={config.get('save_freq', 20000)}",
            f"--eval_freq={config.get('eval_freq', 20000)}",
            f"--log_freq={config.get('log_freq', 200)}",
            f"--output_dir={output_dir}",
        ])

        # Optional image_size / resize_size (applies to all policies with vision)
        if config.get("image_size"):
            cmd.append(f"--policy.image_size={config['image_size']}")
        if config.get("resize_size"):
            rs = config["resize_size"]
            if isinstance(rs, str) and rs.strip():
                # Accept "512,512" or "512" format
                cmd.append(f"--policy.resize_size=[{rs.strip()}]")
            elif isinstance(rs, list):
                cmd.append(f"--policy.resize_size=[{','.join(str(x) for x in rs)}]")

        # Optional evaluation settings
        if config.get("eval_n_episodes"):
            cmd.append(f"--eval.n_episodes={config['eval_n_episodes']}")

        # Learning rate
        if "learning_rate" in config:
            cmd.append(f"--policy.optimizer_lr={config['learning_rate']}")

        # Warmup steps (ACT has no scheduler)
        if "warmup_steps" in config and job.policy_type != PolicyType.ACT:
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
