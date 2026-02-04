"""
Human-in-the-Loop (HIL) Service for DAgger-style learning.

Enables running trained policies with seamless human intervention:
- Human can take control at any moment by moving leader arms
- All data (autonomous + human) is recorded for retraining
- Supports episode-wise inference with manual episode boundaries

Integrates:
- TeleopService: Recording infrastructure (sessions, episodes, frames)
- TaskOrchestrator: Policy deployment and robot control
- TrainingService: Retraining on intervention data
"""

import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


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


class HILService:
    """
    Manages Human-in-the-Loop deployment sessions.

    Workflow:
    1. start_session() - Deploy policy, start recording session
    2. start_episode() - Begin recording an episode
    3. [HIL loop runs: detects human takeover, records frames]
    4. stop_episode() - Save episode data
    5. trigger_retrain() - Optional: retrain on collected data
    6. stop_session() - Finalize recording
    """

    def __init__(self, teleop_service, orchestrator, training_service, robot_lock=None):
        """
        Initialize HIL service.

        Args:
            teleop_service: TeleopService instance for recording
            orchestrator: TaskOrchestrator instance with deployed_policy
            training_service: TrainingService for retraining
            robot_lock: Optional threading lock for robot access
        """
        self.teleop = teleop_service
        self.orchestrator = orchestrator
        self.training = training_service
        self.robot_lock = robot_lock

        self.state = HILSessionState()
        self._hil_thread = None
        self._stop_event = threading.Event()

        # Intervention detection thresholds
        # Match InterventionEngine defaults
        self.MOVE_THRESHOLD = 0.05  # Velocity threshold to detect human input
        self.IDLE_TIMEOUT = 2.0     # Seconds of idle before returning to autonomous
        self._last_human_move_time = 0

        # Policy inference rate
        self.INFERENCE_HZ = 30

    def start_session(self, policy_id: str, intervention_dataset: str, task: str, movement_scale: float = 1.0) -> Dict[str, Any]:
        """
        Start a HIL deployment session.

        1. Deploy the selected policy
        2. Start recording session for intervention data
        3. Begin HIL control loop

        Args:
            policy_id: ID of policy to deploy (from training_service.list_policies)
            intervention_dataset: Dataset name for saving intervention data
            task: Task description for recorded episodes
            movement_scale: Safety limiter (0.1 to 1.0) - scales autonomous movement

        Returns:
            Status dict with session info
        """
        # Clamp movement_scale to valid range
        movement_scale = max(0.1, min(1.0, movement_scale))
        if self.state.active:
            raise Exception("HIL session already active")

        logger.info(f"[HIL] Starting session: policy={policy_id}, dataset={intervention_dataset}")

        # 1. Deploy policy
        policy = self.training.get_policy(policy_id)
        if not policy:
            raise Exception(f"Policy '{policy_id}' not found")
        if not policy.checkpoint_path:
            raise Exception(f"Policy '{policy_id}' has no checkpoint")

        # Get policy configuration (cameras and arms it was trained on)
        policy_config = self.training.get_policy_config(policy_id)
        if policy_config:
            logger.info(f"[HIL] Policy config detected: cameras={policy_config.cameras}, arms={policy_config.arms}")
        else:
            logger.warning(f"[HIL] No policy config found, using defaults (all cameras/arms)")

        try:
            self.orchestrator.deploy_policy(policy.checkpoint_path)
            logger.info(f"[HIL] Policy deployed from {policy.checkpoint_path}")
        except Exception as e:
            logger.error(f"[HIL] Failed to deploy policy: {e}")
            raise Exception(f"Failed to deploy policy: {e}")

        # 2. Start recording session
        try:
            self.teleop.start_recording_session(
                repo_id=intervention_dataset,
                task=task,
                fps=30
            )
            logger.info(f"[HIL] Recording session started: {intervention_dataset}")
        except Exception as e:
            # Clean up deployed policy on failure
            self.orchestrator.deployed_policy = None
            self.orchestrator.deployed_policy_path = None
            logger.error(f"[HIL] Failed to start recording session: {e}")
            raise Exception(f"Failed to start recording session: {e}")

        # 3. Initialize state with policy configuration
        self.state = HILSessionState(
            active=True,
            mode=HILMode.IDLE,  # Will become AUTONOMOUS when episode starts
            policy_id=policy_id,
            intervention_dataset=intervention_dataset,
            task=task,
            # Store policy configuration for frontend filtering
            policy_cameras=policy_config.cameras if policy_config else [],
            policy_arms=policy_config.arms if policy_config else [],
            policy_type=policy_config.policy_type if policy_config else "",
            episode_count=self.teleop.episode_count,
            # Safety settings
            movement_scale=movement_scale
        )

        # 4. Start HIL control loop
        self._stop_event.clear()
        self._hil_thread = threading.Thread(target=self._hil_loop, daemon=True)
        self._hil_thread.start()

        logger.info(f"[HIL] Session started successfully (movement_scale={movement_scale:.1f})")
        return {
            "status": "started",
            "policy_id": policy_id,
            "dataset": intervention_dataset,
            "task": task,
            "movement_scale": movement_scale
        }

    def stop_session(self) -> Dict[str, Any]:
        """
        Stop HIL session and finalize recording.

        Returns:
            Summary of session statistics
        """
        if not self.state.active:
            return {"status": "not_active"}

        logger.info("[HIL] Stopping session...")

        # 1. Stop HIL loop
        self._stop_event.set()
        if self._hil_thread and self._hil_thread.is_alive():
            self._hil_thread.join(timeout=2.0)

        # 2. Stop any active episode
        if self.state.episode_active:
            try:
                self.stop_episode()
            except Exception as e:
                logger.warning(f"[HIL] Error stopping episode: {e}")

        # 3. Finalize recording session
        try:
            self.teleop.stop_recording_session()
        except Exception as e:
            logger.warning(f"[HIL] Error stopping recording session: {e}")

        # 4. Clear deployed policy
        self.orchestrator.deployed_policy = None
        self.orchestrator.deployed_policy_path = None

        result = {
            "status": "stopped",
            "total_episodes": self.state.episode_count,
            "total_interventions": self.state.intervention_count,
            "autonomous_frames": self.state.autonomous_frames,
            "human_frames": self.state.human_frames
        }

        logger.info(f"[HIL] Session stopped: {result}")

        self.state = HILSessionState()
        return result

    def start_episode(self) -> Dict[str, Any]:
        """
        Start a new HIL episode.

        Begins recording and sets mode to AUTONOMOUS.

        Returns:
            Status dict with episode number
        """
        if not self.state.active:
            raise Exception("No active HIL session")
        if self.state.episode_active:
            return {"status": "already_recording", "episode": self.state.episode_count + 1}

        logger.info("[HIL] Starting episode...")

        # Pre-populate motor cache so recording capture has data on first frame
        # This prevents the race condition where recording starts before HIL loop runs
        robot = self.orchestrator.robot
        if robot and hasattr(robot, 'get_observation'):
            try:
                obs = robot.get_observation()
                motor_data = {k: v for k, v in obs.items() if '.pos' in k}
                if motor_data and hasattr(self.teleop, '_action_lock'):
                    with self.teleop._action_lock:
                        self.teleop._latest_leader_action = motor_data.copy()
                    print(f"[HIL] Pre-populated motor cache with {len(motor_data)} keys")
            except Exception as e:
                print(f"[HIL] WARNING: Could not pre-populate motor cache: {e}")

        self.teleop.start_episode()
        self.state.episode_active = True
        self.state.current_episode_interventions = 0
        self.state.mode = HILMode.AUTONOMOUS
        self._last_human_move_time = 0  # Reset for new episode

        logger.info(f"[HIL] Episode started: #{self.state.episode_count + 1}")
        return {"status": "started", "episode": self.state.episode_count + 1}

    def stop_episode(self) -> Dict[str, Any]:
        """
        Stop current episode and save.

        Returns:
            Status dict with episode info and intervention count
        """
        if not self.state.episode_active:
            return {"status": "not_recording"}

        logger.info("[HIL] Stopping episode...")

        self.teleop.stop_episode()
        self.state.episode_active = False
        self.state.mode = HILMode.IDLE
        self.state.episode_count = self.teleop.episode_count

        result = {
            "status": "saved",
            "episode": self.state.episode_count,
            "interventions_in_episode": self.state.current_episode_interventions
        }

        logger.info(f"[HIL] Episode saved: {result}")
        return result

    def resume_autonomous(self) -> Dict[str, Any]:
        """
        Explicitly resume autonomous mode after intervention pause.

        Called when user clicks "Resume Autonomous" button after intervention
        completes and system is in PAUSED state.

        Returns:
            Status dict with new mode
        """
        if self.state.mode == HILMode.PAUSED:
            print("[HIL] User resumed autonomous mode")
            self.state.mode = HILMode.AUTONOMOUS
            return {"status": "resumed", "mode": "autonomous"}
        elif self.state.mode == HILMode.AUTONOMOUS:
            return {"status": "already_autonomous", "mode": "autonomous"}
        else:
            return {"status": "error", "message": f"Cannot resume from {self.state.mode.value}", "mode": self.state.mode.value}

    def get_status(self) -> Dict[str, Any]:
        """
        Get current HIL session status.

        Returns:
            Full status dict for frontend display including policy configuration
        """
        return {
            "active": self.state.active,
            "mode": self.state.mode.value,
            "policy_id": self.state.policy_id,
            "intervention_dataset": self.state.intervention_dataset,
            "task": self.state.task,
            "episode_active": self.state.episode_active,
            "episode_count": self.state.episode_count,
            "intervention_count": self.state.intervention_count,
            "current_episode_interventions": self.state.current_episode_interventions,
            "autonomous_frames": self.state.autonomous_frames,
            "human_frames": self.state.human_frames,
            # Policy configuration for frontend camera/arm filtering
            "policy_config": {
                "cameras": self.state.policy_cameras,
                "arms": self.state.policy_arms,
                "type": self.state.policy_type
            },
            # Safety settings
            "movement_scale": self.state.movement_scale
        }

    def trigger_retrain(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Trigger retraining on the intervention dataset.

        Fine-tunes from the original policy checkpoint (DAgger approach)
        rather than training from scratch.

        Args:
            config: Optional training config overrides

        Returns:
            Status dict with job_id
        """
        if not self.state.intervention_dataset:
            raise Exception("No intervention dataset configured")

        # Get original policy checkpoint for fine-tuning
        original_policy = self.training.get_policy(self.state.policy_id)
        if not original_policy:
            raise Exception(f"Original policy {self.state.policy_id} not found")

        if not original_policy.checkpoint_path:
            raise Exception(f"Original policy {self.state.policy_id} has no checkpoint")

        policy_type = original_policy.policy_type

        # Build training config with checkpoint for fine-tuning
        # Use shorter steps since we're fine-tuning, not training from scratch
        default_config = {
            "steps": 10000,  # Shorter for fine-tuning
            "batch_size": 8,
            "policy_name": f"hil_finetuned_{self.state.policy_id}",
            # KEY: Start from original policy weights (DAgger approach)
            "resume_from_checkpoint": original_policy.checkpoint_path
        }
        if config:
            default_config.update(config)

        logger.info(f"[HIL] Triggering fine-tune retrain: dataset={self.state.intervention_dataset}, "
                    f"type={policy_type}, checkpoint={original_policy.checkpoint_path}")

        job = self.training.create_job(
            dataset_repo_id=self.state.intervention_dataset,
            policy_type=policy_type,
            config=default_config
        )
        self.training.start_job(job.id)

        return {"status": "started", "job_id": job.id}

    def _hil_loop(self):
        """
        Main HIL control loop (30Hz).

        Handles:
        - Human intervention detection via leader arm velocity
        - Mode switching (autonomous <-> human)
        - Policy inference when in autonomous mode
        - Frame counting for statistics
        """
        print("[HIL] Control loop started")
        loop_period = 1.0 / self.INFERENCE_HZ

        while not self._stop_event.is_set():
            loop_start = time.time()

            # Only process when episode is active
            if not self.state.active or not self.state.episode_active:
                # Log waiting state once
                if not hasattr(self, '_logged_waiting'):
                    print(f"[HIL] Loop waiting: active={self.state.active}, episode_active={self.state.episode_active}")
                    self._logged_waiting = True
                time.sleep(0.01)
                continue

            # Reset waiting flag when we start processing
            if hasattr(self, '_logged_waiting'):
                print(f"[HIL] Loop now active: mode={self.state.mode}")
                delattr(self, '_logged_waiting')

            try:
                # 1. Check for human intervention
                human_velocity = self._get_leader_velocity()

                if human_velocity > self.MOVE_THRESHOLD:
                    # Human is intervening
                    if self.state.mode == HILMode.AUTONOMOUS:
                        print("[HIL] Human takeover detected!")
                        self.state.intervention_count += 1
                        self.state.current_episode_interventions += 1
                    self.state.mode = HILMode.HUMAN
                    self._last_human_move_time = time.time()
                    self.state.human_frames += 1

                elif self._last_human_move_time > 0 and (time.time() - self._last_human_move_time) > self.IDLE_TIMEOUT:
                    # Human idle - transition to PAUSED (not auto-resume)
                    if self.state.mode == HILMode.HUMAN:
                        print("[HIL] Intervention complete - PAUSED, waiting for user decision")
                        self.state.mode = HILMode.PAUSED

                # Also check if user grabs arm during PAUSED state
                elif self.state.mode == HILMode.PAUSED and human_velocity > self.MOVE_THRESHOLD:
                    # User grabbed arm again while paused
                    print("[HIL] Human resumed intervention from paused state")
                    self.state.mode = HILMode.HUMAN
                    self._last_human_move_time = time.time()

                # 2. Execute based on mode
                if self.state.mode == HILMode.AUTONOMOUS:
                    self._run_policy_inference()
                    self.state.autonomous_frames += 1
                elif self.state.mode == HILMode.HUMAN:
                    # Human intervention - read leader, send to follower, cache for recording
                    self._run_human_teleop()
                elif self.state.mode == HILMode.PAUSED:
                    # Robot holds position - don't run policy, don't move
                    # Just wait for user to click Resume or Stop Episode
                    pass

            except Exception as e:
                # Suppress frequent errors during normal operation
                if "has no calibration registered" not in str(e):
                    print(f"[HIL] Loop error: {e}")

            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = loop_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("[HIL] Control loop stopped")

    def _get_leader_velocity(self) -> float:
        """
        Get velocity from LEADER arm to detect human input.

        Reads from the leader arm (human control device), not the follower robot.
        Computes velocity by tracking position changes between calls.

        Only checks velocity from arms the policy was trained on.
        For a left-arm-only policy, only left arm velocity triggers intervention.

        Returns:
            Maximum velocity magnitude across policy-relevant arms (scaled by loop rate)
        """
        # Check leader arm exists
        if not hasattr(self.teleop, 'leader') or self.teleop.leader is None:
            return 0.0

        try:
            # Get current leader arm positions
            leader = self.teleop.leader
            current_pos = leader.get_action()  # Returns dict like {'left_base.pos': 0.5, ...}

            if not current_pos:
                return 0.0

            # Initialize position tracking on first call
            if not hasattr(self, '_last_leader_pos') or self._last_leader_pos is None:
                self._last_leader_pos = current_pos.copy()
                return 0.0

            # Determine which arms to check based on policy configuration
            policy_arms = self.state.policy_arms if self.state.policy_arms else ["left", "right"]

            # Compute max position delta across relevant motors
            max_delta = 0.0
            for key, val in current_pos.items():
                # Filter by policy arms
                is_relevant = False
                if "left" in policy_arms and key.startswith("left_"):
                    is_relevant = True
                if "right" in policy_arms and key.startswith("right_"):
                    is_relevant = True
                # Non-arm-specific keys (like 'gripper') - check if any arm matches
                if not key.startswith("left_") and not key.startswith("right_"):
                    is_relevant = True

                if is_relevant and key in self._last_leader_pos:
                    delta = abs(float(val) - float(self._last_leader_pos[key]))
                    max_delta = max(max_delta, delta)

            # Update position cache
            self._last_leader_pos = current_pos.copy()

            # Scale by loop rate to get velocity-like value
            # At 30Hz loop, multiply by 30 to convert position delta to velocity estimate
            velocity = max_delta * self.INFERENCE_HZ

            # Log occasionally for debugging
            if velocity > self.MOVE_THRESHOLD and not hasattr(self, '_logged_leader_vel'):
                print(f"[HIL] Leader velocity detected: {velocity:.3f} (threshold: {self.MOVE_THRESHOLD})")
                self._logged_leader_vel = True

            return velocity

        except Exception as e:
            msg = str(e)
            # Suppress known spam errors
            if "has no calibration registered" not in msg and "Failed to sync read" not in msg:
                logger.debug(f"[HIL] Error reading leader velocity: {e}")
            return 0.0

    def _get_training_state_names(self) -> list:
        """
        Get motor names from the policy's training dataset.

        The policy checkpoint includes train_config.json which references
        the original training dataset. We need those motor names (e.g., 7 for left arm)
        not the intervention dataset's names (e.g., 14 for both arms).

        Returns:
            List of motor names like ['left_base.pos', 'left_link3.pos', ...] or None
        """
        from pathlib import Path
        import json

        # Return cached result if available
        if hasattr(self, '_cached_training_state_names'):
            return self._cached_training_state_names

        # Get checkpoint path
        checkpoint_path = Path(self.orchestrator.deployed_policy_path)

        # Load train_config.json from checkpoint directory
        train_config_path = checkpoint_path / "train_config.json"
        if not train_config_path.exists():
            print(f"[HIL] WARNING: train_config.json not found at {train_config_path}")
            self._cached_training_state_names = None
            return None

        with open(train_config_path) as f:
            train_config = json.load(f)

        # Get training dataset path
        dataset_root = train_config.get("dataset", {}).get("root")
        if not dataset_root:
            print(f"[HIL] WARNING: dataset.root not found in train_config")
            self._cached_training_state_names = None
            return None

        # Load training dataset's info.json
        info_path = Path(dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            print(f"[HIL] WARNING: Training dataset info.json not found: {info_path}")
            self._cached_training_state_names = None
            return None

        with open(info_path) as f:
            info = json.load(f)

        # Get state feature names
        state_names = info.get("features", {}).get("observation.state", {}).get("names")
        if state_names:
            print(f"[HIL] Loaded {len(state_names)} state names from training dataset: {state_names}")

        self._cached_training_state_names = state_names
        return state_names

    def _load_normalization_stats(self) -> dict:
        """
        Load normalization statistics from policy checkpoint or training dataset.

        The checkpoint may include a safetensors file with min/max/mean/std values.
        For Pi0.5, the safetensors file is not saved, so we fallback to loading
        stats from the training dataset's meta/stats.json.

        Returns:
            Dict with normalization stats or None if not found
        """
        from pathlib import Path
        import safetensors.torch as st
        import json
        import torch

        # Return cached result if available
        if hasattr(self, '_cached_norm_stats'):
            return self._cached_norm_stats

        # Get checkpoint path
        checkpoint_path = Path(self.orchestrator.deployed_policy_path)
        stats_path = checkpoint_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"

        # Try loading from checkpoint safetensors first
        if stats_path.exists():
            try:
                stats = st.load_file(str(stats_path))
                print(f"[HIL] Loaded normalization stats from checkpoint ({len(stats)} keys)")
                print(f"[HIL DEBUG] Stats keys: {list(stats.keys())}")

                # Log key stats for debugging
                if 'observation.state.min' in stats:
                    print(f"[HIL]   observation.state.min: {stats['observation.state.min'].tolist()}")
                    print(f"[HIL]   observation.state.max: {stats['observation.state.max'].tolist()}")

                # DEBUG: Log action stats (critical for denormalization)
                if 'action.min' in stats:
                    print(f"[HIL DEBUG] action.min: {stats['action.min'].tolist()}")
                    print(f"[HIL DEBUG] action.max: {stats['action.max'].tolist()}")
                    action_range = stats['action.max'] - stats['action.min']
                    print(f"[HIL DEBUG] action range (max-min): {action_range.tolist()}")
                else:
                    print(f"[HIL DEBUG] WARNING: 'action.min' not found in stats!")

                self._cached_norm_stats = stats
                return stats

            except Exception as e:
                print(f"[HIL] WARNING: Failed to load normalization stats from safetensors: {e}")

        # FALLBACK: Load from training dataset's stats.json (needed for Pi0.5)
        print(f"[HIL] Checkpoint stats not found, trying training dataset...")

        # Find policy metadata to get training dataset
        metadata_path = checkpoint_path / "policy_metadata.json"
        if not metadata_path.exists():
            # Try parent directories (checkpoints/last/pretrained_model -> policy_dir)
            for parent_level in [1, 2, 3]:
                parent = checkpoint_path
                for _ in range(parent_level):
                    parent = parent.parent
                test_path = parent / "policy_metadata.json"
                if test_path.exists():
                    metadata_path = test_path
                    break

        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)

                dataset_repo_id = metadata.get("dataset_repo_id")
                use_quantile = metadata.get("config", {}).get("use_quantile_normalization", True)

                if dataset_repo_id:
                    # Find datasets path - go up from checkpoint to project root
                    # checkpoint_path: .../training/outputs/policy_dir/checkpoints/last/pretrained_model
                    # datasets_path: .../datasets
                    project_root = checkpoint_path
                    while project_root.name != "nextis_app" and project_root.parent != project_root:
                        project_root = project_root.parent

                    datasets_path = project_root / "datasets"
                    stats_json_path = datasets_path / dataset_repo_id / "meta" / "stats.json"

                    print(f"[HIL] Looking for stats at: {stats_json_path}")

                    if stats_json_path.exists():
                        with open(stats_json_path) as f:
                            dataset_stats = json.load(f)

                        if "action" in dataset_stats:
                            action_stats = dataset_stats["action"]

                            # IMPORTANT: Normalization type MUST match what was used during training!
                            # Pi0.5 uses QUANTILES normalization by default (use_quantile_normalization=True)
                            # Using wrong normalization range shifts all positions to wrong values
                            print(f"[HIL] Training used quantile normalization: {use_quantile}")

                            if use_quantile and "q01" in action_stats and "q99" in action_stats:
                                # Pi0.5 default: use QUANTILES (q01/q99) - must match training!
                                stats = {
                                    "action.min": torch.tensor(action_stats["q01"], dtype=torch.float32),
                                    "action.max": torch.tensor(action_stats["q99"], dtype=torch.float32),
                                }
                                print(f"[HIL] Loaded action normalization stats (QUANTILES q01/q99 - matching training)")
                                print(f"[HIL DEBUG] action.min (q01): {action_stats['q01']}")
                                print(f"[HIL DEBUG] action.max (q99): {action_stats['q99']}")
                            elif "min" in action_stats and "max" in action_stats:
                                # Fallback to min/max if training used MEAN_STD or quantiles unavailable
                                stats = {
                                    "action.min": torch.tensor(action_stats["min"], dtype=torch.float32),
                                    "action.max": torch.tensor(action_stats["max"], dtype=torch.float32),
                                }
                                print(f"[HIL] Loaded action normalization stats (min/max - {'fallback' if use_quantile else 'matching training'})")
                                print(f"[HIL DEBUG] action.min: {action_stats['min']}")
                                print(f"[HIL DEBUG] action.max: {action_stats['max']}")
                            else:
                                print(f"[HIL] WARNING: No valid action normalization stats found")
                                stats = {}

                            # CRITICAL: Also load state normalization stats!
                            # Pi0.5 normalizes the input state too - without this, the model receives
                            # raw motor positions instead of normalized values, causing garbage output
                            if "observation.state" in dataset_stats:
                                state_stats = dataset_stats["observation.state"]
                                if use_quantile and "q01" in state_stats and "q99" in state_stats:
                                    stats["observation.state.min"] = torch.tensor(state_stats["q01"], dtype=torch.float32)
                                    stats["observation.state.max"] = torch.tensor(state_stats["q99"], dtype=torch.float32)
                                    print(f"[HIL] Loaded state normalization stats (QUANTILES q01/q99)")
                                    print(f"[HIL DEBUG] state.min (q01): {state_stats['q01']}")
                                    print(f"[HIL DEBUG] state.max (q99): {state_stats['q99']}")
                                elif "min" in state_stats and "max" in state_stats:
                                    stats["observation.state.min"] = torch.tensor(state_stats["min"], dtype=torch.float32)
                                    stats["observation.state.max"] = torch.tensor(state_stats["max"], dtype=torch.float32)
                                    print(f"[HIL] Loaded state normalization stats (min/max)")
                                    print(f"[HIL DEBUG] state.min: {state_stats['min']}")
                                    print(f"[HIL DEBUG] state.max: {state_stats['max']}")
                                else:
                                    print(f"[HIL] WARNING: No valid state normalization stats found - state will NOT be normalized!")
                            else:
                                print(f"[HIL] WARNING: No observation.state in dataset stats - state will NOT be normalized!")

                            if stats:
                                self._cached_norm_stats = stats
                                return stats
                    else:
                        print(f"[HIL] WARNING: Training dataset stats not found: {stats_json_path}")
                else:
                    print(f"[HIL] WARNING: No dataset_repo_id in policy metadata")

            except Exception as e:
                print(f"[HIL] WARNING: Failed to load stats from training dataset: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[HIL] WARNING: Policy metadata not found at: {metadata_path}")

        print(f"[HIL] WARNING: No normalization stats found - actions will NOT be denormalized!")
        self._cached_norm_stats = None
        return None

    def _prepare_policy_observation(self, raw_obs: dict) -> dict:
        """
        Transform robot observation dict to policy-expected format.

        Robot returns: {'camera_1': np.array(H,W,C), 'left_base.pos': 0.5, ...}
        Policy expects: {'observation.images.camera_1': Tensor(C,H,W), 'observation.state': Tensor(N)}

        IMPORTANT: Applies normalization using stats from the policy checkpoint:
        - Images: (x - mean) / std using ImageNet stats
        - State: (x - min) / (max - min) scaled to [-1, 1]
        """
        import torch

        policy = self.orchestrator.deployed_policy
        device = policy.config.device
        policy_obs = {}

        # Load normalization stats from checkpoint
        norm_stats = self._load_normalization_stats()

        # 1. Transform and NORMALIZE camera images
        if hasattr(policy.config, 'image_features') and policy.config.image_features:
            for key in policy.config.image_features:
                # key is like 'observation.images.camera_2'
                cam_name = key.split('.')[-1]  # 'camera_2'

                if cam_name in raw_obs:
                    img = raw_obs[cam_name]  # numpy HWC uint8

                    # HWC → CHW, normalize to [0,1]
                    img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0

                    # Apply ImageNet-style normalization: (x - mean) / std
                    if norm_stats:
                        mean_key = f"{key}.mean"
                        std_key = f"{key}.std"
                        if mean_key in norm_stats and std_key in norm_stats:
                            mean = norm_stats[mean_key].view(3, 1, 1)  # Shape: (3, 1, 1)
                            std = norm_stats[std_key].view(3, 1, 1)
                            img_tensor = (img_tensor - mean) / (std + 1e-8)

                            if not hasattr(self, '_logged_img_norm'):
                                print(f"[HIL] Image {key} normalized: mean={mean.squeeze().tolist()}, std={std.squeeze().tolist()}")
                                self._logged_img_norm = True

                    img_tensor = img_tensor.unsqueeze(0)  # Add batch dim: (1, C, H, W)
                    policy_obs[key] = img_tensor.to(device)

        # 2. Build and NORMALIZE observation.state tensor
        # IMPORTANT: Use motor names from TRAINING dataset, not intervention dataset!
        if hasattr(policy.config, 'robot_state_feature') and policy.config.robot_state_feature:
            # Get motor names from training dataset (correct dimension)
            state_names = self._get_training_state_names()

            if state_names is None:
                # Fallback: use intervention dataset (may cause dimension mismatch!)
                print("[HIL] WARNING: Falling back to intervention dataset state names - may cause dimension mismatch!")
                if hasattr(self.teleop, 'dataset') and self.teleop.dataset is not None:
                    features = self.teleop.dataset.features
                    if 'action' in features and 'names' in features['action']:
                        state_names = features['action']['names']

            if state_names:
                state_values = [float(raw_obs.get(name, 0.0)) for name in state_names]
                state_tensor = torch.tensor(state_values, dtype=torch.float32)

                # Apply MIN_MAX normalization: (x - min) / (max - min) * 2 - 1 → [-1, 1]
                if norm_stats and 'observation.state.min' in norm_stats and 'observation.state.max' in norm_stats:
                    state_min = norm_stats['observation.state.min']
                    state_max = norm_stats['observation.state.max']

                    # Handle dead motors (min == max) - these didn't move during training
                    # Set normalized value to 0.0 (middle of [-1, 1] range) for these motors
                    state_range = state_max - state_min
                    dead_motors = state_range.abs() < 1e-6  # Motors with essentially no range

                    # Replace zero ranges with 1.0 to avoid division by zero
                    safe_range = torch.where(dead_motors, torch.ones_like(state_range), state_range)

                    # Normalize to [0, 1]
                    state_tensor = (state_tensor - state_min) / safe_range

                    # For dead motors, set normalized value to 0.5 (which becomes 0.0 after scaling to [-1, 1])
                    state_tensor = torch.where(dead_motors, torch.full_like(state_tensor, 0.5), state_tensor)

                    # Scale to [-1, 1]
                    state_tensor = state_tensor * 2.0 - 1.0

                    if not hasattr(self, '_logged_state_norm'):
                        dead_count = dead_motors.sum().item()
                        print(f"[HIL] State normalized using MIN_MAX: min={state_min.tolist()}, max={state_max.tolist()}")
                        if dead_count > 0:
                            dead_indices = torch.where(dead_motors)[0].tolist()
                            print(f"[HIL] WARNING: {dead_count} motors have min==max (didn't move in training), indices: {dead_indices}")
                            print(f"[HIL]   These motors normalized to 0.0 (neutral position)")
                        print(f"[HIL] State after norm: min={state_tensor.min():.3f}, max={state_tensor.max():.3f}, mean={state_tensor.mean():.3f}")
                        self._logged_state_norm = True

                    # Clamp to [-1, 1] to prevent out-of-range values from confusing the policy
                    # This can happen if robot position differs from training range
                    if state_tensor.min() < -1.0 or state_tensor.max() > 1.0:
                        if not hasattr(self, '_logged_clamp_warning'):
                            # Log which motors are out of range (once)
                            print(f"[HIL] WARNING: Some motors outside training range!")
                            for i, name in enumerate(state_names):
                                val = state_tensor[i].item()
                                if val < -1.0 or val > 1.0:
                                    raw_val = float(raw_obs.get(name, 0.0))
                                    print(f"[HIL]   Motor '{name}': normalized={val:.2f} (raw={raw_val:.1f}, expected=[{state_min[i].item():.1f}, {state_max[i].item():.1f}])")
                            print(f"[HIL]   Clamping all normalized values to [-1, 1]")
                            self._logged_clamp_warning = True
                        state_tensor = torch.clamp(state_tensor, -1.0, 1.0)

                state_tensor = state_tensor.unsqueeze(0)  # Add batch dim: (1, N)
                policy_obs['observation.state'] = state_tensor.to(device)

        # 3. For Pi0.5: Add language tokenization
        # Pi0.5 is a VLA model that requires tokenized task instruction
        if self.state.policy_type == "pi05":
            task = self.state.task or "Do the task"  # Fallback task

            # Get the normalized state for discretization
            if 'observation.state' in policy_obs:
                state_for_tokenization = policy_obs['observation.state'].squeeze(0).cpu().numpy()

                # Pad state to max_state_dim (32 for Pi0.5)
                max_state_dim = getattr(policy.config, 'max_state_dim', 32)
                if len(state_for_tokenization) < max_state_dim:
                    padded_state = np.zeros(max_state_dim, dtype=np.float32)
                    padded_state[:len(state_for_tokenization)] = state_for_tokenization
                    state_for_tokenization = padded_state

                # Discretize state into 256 bins (Pi0.5 protocol)
                # State should be in [-1, 1] range after normalization
                discretized = np.digitize(
                    state_for_tokenization,
                    bins=np.linspace(-1, 1, 256 + 1)[:-1]
                ) - 1
                discretized = np.clip(discretized, 0, 255)  # Ensure valid range

                # Create Pi0.5 prompt format
                state_str = " ".join(map(str, discretized))
                cleaned_task = task.strip().replace("_", " ").replace("\n", " ")
                prompt = f"Task: {cleaned_task}, State: {state_str};\nAction: "

                # Tokenize using PaLI-Gemma tokenizer (cached for efficiency)
                if not hasattr(self, '_pi05_tokenizer'):
                    from transformers import AutoTokenizer
                    self._pi05_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
                    self._pi05_tokenizer.padding_side = "right"
                    print(f"[HIL] Pi0.5 tokenizer loaded: google/paligemma-3b-pt-224")

                max_length = getattr(policy.config, 'tokenizer_max_length', 200)
                tokenized = self._pi05_tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt"
                )

                policy_obs['observation.language.tokens'] = tokenized['input_ids'].to(device)
                # IMPORTANT: Pi0.5 expects attention_mask as boolean, not Long
                policy_obs['observation.language.attention_mask'] = tokenized['attention_mask'].bool().to(device)

                if not hasattr(self, '_logged_pi05_tokenization'):
                    print(f"[HIL] Pi0.5 tokenization: prompt length={len(prompt)}, tokens shape={tokenized['input_ids'].shape}")
                    self._logged_pi05_tokenization = True

        # Log transformed observation once
        if not hasattr(self, '_logged_policy_obs'):
            print(f"[HIL] Transformed observation keys: {list(policy_obs.keys())}")
            for k, v in policy_obs.items():
                if hasattr(v, 'shape'):
                    print(f"[HIL]   {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
            self._logged_policy_obs = True

        return policy_obs

    def _run_policy_inference(self):
        """
        Execute one step of policy inference.

        Gets observation, runs policy, sends action to robot.
        Converts tensor action to dict format for recording compatibility.
        """
        if not hasattr(self.orchestrator, 'deployed_policy') or self.orchestrator.deployed_policy is None:
            if not hasattr(self, '_logged_no_policy'):
                print("[HIL] WARNING: No deployed policy - _run_policy_inference skipped")
                self._logged_no_policy = True
            return

        robot = self.orchestrator.robot if self.orchestrator else None
        if robot is None or not robot.is_connected:
            if not hasattr(self, '_logged_no_robot'):
                print("[HIL] WARNING: Robot not connected - _run_policy_inference skipped")
                self._logged_no_robot = True
            return

        try:
            # Get raw observation from robot
            if self.robot_lock:
                with self.robot_lock:
                    raw_obs = robot.get_observation()
            else:
                raw_obs = robot.get_observation()

            # Log raw observation keys once
            if not hasattr(self, '_logged_obs_keys'):
                print(f"[HIL] Raw observation keys: {list(raw_obs.keys())}")
                self._logged_obs_keys = True

            # Transform to policy-expected format
            policy_obs = self._prepare_policy_observation(raw_obs)

            # Run policy with transformed observation
            policy = self.orchestrator.deployed_policy
            action = policy.select_action(policy_obs)

            # Log action info once
            if not hasattr(self, '_logged_action_info'):
                action_shape = action.shape if hasattr(action, 'shape') else 'N/A'
                print(f"[HIL] Policy returned action: type={type(action).__name__}, shape={action_shape}")
                self._logged_action_info = True

            # DEBUG: Log raw action values to diagnose "home position" issue
            if hasattr(action, 'cpu'):
                action_np_debug = action.cpu().numpy()
            else:
                action_np_debug = np.array(action)
            # Flatten if multi-dimensional
            if action_np_debug.ndim > 1:
                action_np_debug = action_np_debug.flatten()[:14]  # First 14 values (both arms)
            print(f"[HIL DEBUG] Raw policy output (first 14): {action_np_debug[:14].tolist()}")
            print(f"[HIL DEBUG] Raw policy output stats: min={action_np_debug.min():.4f}, max={action_np_debug.max():.4f}, mean={action_np_debug.mean():.4f}, std={action_np_debug.std():.4f}")
            # Check if output is near-zero (indicates broken model)
            if np.abs(action_np_debug).max() < 0.01:
                print(f"[HIL DEBUG] WARNING: Policy output is near-zero! Model may not be loaded correctly.")

            # Convert tensor action to dict with named keys for robot and recording
            # The recording capture loop expects dict with keys like 'left_base.pos'
            # Pass raw_obs for movement scaling (safety limiter)
            action_dict = self._convert_action_to_dict(action, raw_obs)

            if not action_dict:
                if not hasattr(self, '_logged_empty_action'):
                    print("[HIL] WARNING: Action dict is empty - cache will NOT be populated!")
                    self._logged_empty_action = True
                return

            # Log successful conversion once
            if not hasattr(self, '_logged_action_dict'):
                print(f"[HIL] Converted action to dict with keys: {list(action_dict.keys())}")
                self._logged_action_dict = True

            # Send action to robot
            # Use partial action sending for single-arm policies (avoids StopIteration on empty arm)
            if self.robot_lock:
                with self.robot_lock:
                    self._send_partial_action(robot, action_dict)
            else:
                self._send_partial_action(robot, action_dict)

            # Cache action for recording capture loop
            # The recording capture loop in TeleopService reads from _latest_leader_action
            # During HIL mode, the teleop loop isn't running, so we need to populate this cache

            # Pad action_dict with other arm positions if recording dataset expects both arms
            # This handles the case where policy trained on left-only but dataset has both arms
            padded_action = self._pad_action_for_recording(action_dict, raw_obs)

            if hasattr(self.teleop, '_action_lock') and hasattr(self.teleop, '_latest_leader_action'):
                with self.teleop._action_lock:
                    self.teleop._latest_leader_action = padded_action.copy()
                # Log caching success once
                if not hasattr(self, '_logged_cache_success'):
                    print(f"[HIL] Successfully cached action with {len(padded_action)} keys for recording")
                    self._logged_cache_success = True
            else:
                if not hasattr(self, '_logged_no_cache'):
                    print("[HIL] WARNING: Cannot cache action - teleop missing _action_lock or _latest_leader_action")
                    self._logged_no_cache = True

        except Exception as e:
            # Log errors with full traceback
            import traceback
            print(f"[HIL] ERROR: Policy inference error: {e}")
            print(traceback.format_exc())

    def _send_partial_action(self, robot, action_dict: dict):
        """
        Send action to robot, handling single-arm policies.

        For bimanual robots (bi_umbra_follower), the standard send_action() tries
        to send to BOTH arms. If the policy only outputs LEFT arm actions,
        the RIGHT arm receives an empty dict causing StopIteration.

        This method only sends to arms that have actions in the dict.

        Args:
            robot: The robot instance (may be bimanual or single arm)
            action_dict: Action dict with keys like 'left_base.pos', 'right_base.pos', etc.
        """
        # Split actions by arm AND strip prefix (individual arms expect keys without arm prefix)
        # e.g., left_arm.send_action expects {'base.pos': 0.5}, not {'left_base.pos': 0.5}
        left_action = {k.removeprefix('left_'): v for k, v in action_dict.items() if k.startswith('left_')}
        right_action = {k.removeprefix('right_'): v for k, v in action_dict.items() if k.startswith('right_')}

        # Check if robot is bimanual
        is_bimanual = hasattr(robot, 'left_arm') and hasattr(robot, 'right_arm')

        if is_bimanual:
            # Send only to arms that have actions
            if left_action:
                try:
                    robot.left_arm.send_action(left_action)
                except Exception as e:
                    if not hasattr(self, '_logged_left_send_error'):
                        print(f"[HIL] WARNING: Failed to send left arm action: {e}")
                        self._logged_left_send_error = True

            if right_action:
                try:
                    robot.right_arm.send_action(right_action)
                except Exception as e:
                    if not hasattr(self, '_logged_right_send_error'):
                        print(f"[HIL] WARNING: Failed to send right arm action: {e}")
                        self._logged_right_send_error = True

            # Log which arms received actions (once)
            if not hasattr(self, '_logged_partial_send'):
                arms_sent = []
                if left_action:
                    arms_sent.append(f"left ({len(left_action)} keys)")
                if right_action:
                    arms_sent.append(f"right ({len(right_action)} keys)")
                print(f"[HIL] Sent partial action to: {', '.join(arms_sent) if arms_sent else 'no arms'}")
                self._logged_partial_send = True
        else:
            # Single arm robot - use standard send_action
            robot.send_action(action_dict)

    def _convert_action_to_dict(self, action, raw_obs=None) -> dict:
        """
        Convert policy action tensor to dict with named keys.

        The recording capture loop expects action dicts with keys like 'left_base.pos',
        but the policy outputs a normalized tensor in [-1, 1] range. This:
        1. Denormalizes using MIN_MAX stats from checkpoint
        2. Applies movement scaling (safety limiter) if configured
        3. Converts to dict using TRAINING dataset feature names
        """
        import torch
        import numpy as np

        # If already a dict, return as-is
        if isinstance(action, dict):
            if not hasattr(self, '_logged_action_already_dict'):
                print("[HIL] Action is already a dict, returning as-is")
                self._logged_action_already_dict = True
            return action

        # Get action names from TRAINING dataset (same dimension as policy)
        action_names = self._get_training_state_names()

        if action_names is None:
            # Fallback: try intervention dataset (may cause mismatch)
            print("[HIL] WARNING: Falling back to intervention dataset for action names")
            if not hasattr(self.teleop, 'dataset') or self.teleop.dataset is None:
                if not hasattr(self, '_logged_no_dataset'):
                    print("[HIL] WARNING: teleop.dataset is None! Cannot convert action tensor to dict")
                    self._logged_no_dataset = True
                return {}

            features = self.teleop.dataset.features

            if 'action' not in features:
                if not hasattr(self, '_logged_no_action'):
                    print(f"[HIL] WARNING: 'action' not in features! Available keys: {list(features.keys())}")
                    self._logged_no_action = True
                return {}

            action_feature = features['action']
            if 'names' not in action_feature:
                if not hasattr(self, '_logged_no_names'):
                    print(f"[HIL] WARNING: 'names' not in action feature! Keys: {list(action_feature.keys())}")
                    self._logged_no_names = True
                return {}

            action_names = action_feature['names']

        # Log action names once
        if not hasattr(self, '_logged_action_names'):
            print(f"[HIL] Action names for conversion: {action_names}")
            self._logged_action_names = True

        # Convert tensor to numpy
        # Handle diffusion policy's multi-step action output (n_action_steps, action_dim)
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = np.array(action)

        # Log raw action shape for debugging
        if not hasattr(self, '_logged_raw_action_shape'):
            print(f"[HIL] Raw action tensor shape: {action_np.shape}, ndim={action_np.ndim}")
            self._logged_raw_action_shape = True

        # Handle multi-step action output from diffusion policy
        # Shape could be (batch, n_action_steps, action_dim) or (n_action_steps, action_dim)
        if action_np.ndim == 3:
            # (batch, n_action_steps, action_dim) -> take first batch, first step
            action_np = action_np[0, 0]
            if not hasattr(self, '_logged_multistep'):
                print(f"[HIL] Extracted first action from 3D tensor: shape now {action_np.shape}")
                self._logged_multistep = True
        elif action_np.ndim == 2:
            # (n_action_steps, action_dim) -> take first step
            action_np = action_np[0]
            if not hasattr(self, '_logged_multistep'):
                print(f"[HIL] Extracted first action from {action_np.shape[0]}-step sequence")
                self._logged_multistep = True
        elif action_np.ndim == 1:
            # Already single action (action_dim,) - nothing to do
            pass
        else:
            # Squeeze any remaining dimensions
            action_np = action_np.squeeze()

        # Log raw normalized action values
        if not hasattr(self, '_logged_raw_action'):
            print(f"[HIL] Raw policy output (normalized): shape={action_np.shape}, min={action_np.min():.3f}, max={action_np.max():.3f}, mean={action_np.mean():.3f}")
            self._logged_raw_action = True

        # DEBUG: Periodic logging every 30 frames (1 second at 30Hz)
        if not hasattr(self, '_debug_frame_count'):
            self._debug_frame_count = 0
        self._debug_frame_count += 1
        if self._debug_frame_count % 30 == 0:
            print(f"[HIL DEBUG] Frame {self._debug_frame_count}: normalized action (first 7) = {action_np[:7].tolist()}")

        # DENORMALIZE: Convert from [-1, 1] back to motor position range
        # Formula: action_raw = (action_normalized + 1) / 2 * (max - min) + min
        norm_stats = self._load_normalization_stats()
        if norm_stats and 'action.min' in norm_stats and 'action.max' in norm_stats:
            action_min = norm_stats['action.min'].cpu().numpy()
            action_max = norm_stats['action.max'].cpu().numpy()

            # Handle dead motors (min == max) - these didn't move during training
            # For dead motors, use the mid-point value instead of denormalizing
            action_range = action_max - action_min
            dead_motors = np.abs(action_range) < 1e-6

            # Replace zero ranges with 1.0 to avoid producing NaN/inf
            safe_range = np.where(dead_motors, 1.0, action_range)

            # Mid-point for dead motors
            action_mid = (action_max + action_min) / 2.0

            # Denormalize: [-1, 1] → [0, 1] → [min, max]
            action_denorm = (action_np + 1.0) / 2.0  # [-1, 1] → [0, 1]
            action_denorm = action_denorm * safe_range + action_min  # [0, 1] → [min, max]

            # For dead motors, use mid-point value (they should stay where they are)
            action_denorm = np.where(dead_motors, action_mid, action_denorm)

            if not hasattr(self, '_logged_denorm_action'):
                dead_count = np.sum(dead_motors)
                print(f"[HIL] ===== DENORMALIZATION DETAILS =====")
                print(f"[HIL]   Normalized input (first 7): {action_np_pre_denorm[:7].tolist() if 'action_np_pre_denorm' in dir() else action_np[:7].tolist()}")
                print(f"[HIL]   action.min: {action_min.tolist()}")
                print(f"[HIL]   action.max: {action_max.tolist()}")
                print(f"[HIL]   action.range: {action_range.tolist()}")
                print(f"[HIL]   Denormalized output (first 7): {action_denorm[:7].tolist()}")
                print(f"[HIL]   Output stats: min={action_denorm.min():.1f}, max={action_denorm.max():.1f}, mean={action_denorm.mean():.1f}")
                if dead_count > 0:
                    dead_indices = np.where(dead_motors)[0].tolist()
                    print(f"[HIL] WARNING: {dead_count} actions have min==max (didn't move in training), indices: {dead_indices}")
                    print(f"[HIL]   These actions use mid-point values: {[action_mid[i] for i in dead_indices]}")
                print(f"[HIL] =====================================")
                self._logged_denorm_action = True

            action_np = action_denorm

            # DEBUG: Periodic logging of denormalized values
            if hasattr(self, '_debug_frame_count') and self._debug_frame_count % 30 == 0:
                print(f"[HIL DEBUG] Frame {self._debug_frame_count}: denormalized action (first 7) = {action_np[:7].tolist()}")

        # Apply movement scaling (safety limiter)
        # Scales the delta between current position and target action
        # At scale=0.5, robot only moves halfway to the policy's target
        movement_scale = self.state.movement_scale if self.state else 1.0
        if movement_scale < 1.0 and raw_obs is not None:
            # Build current state from individual motor keys (matching action_names order)
            # raw_obs contains keys like 'left_base.pos', NOT 'observation.state'
            current_state = None
            if action_names and all(name in raw_obs for name in action_names):
                try:
                    current_state = np.array([float(raw_obs[name]) for name in action_names])
                    if not hasattr(self, '_logged_movement_scale_state_built'):
                        print(f"[HIL] Built current_state from {len(action_names)} motor keys for movement scaling")
                        self._logged_movement_scale_state_built = True
                except Exception as e:
                    if not hasattr(self, '_logged_movement_scale_build_error'):
                        print(f"[HIL] WARNING: Could not build current_state for movement scaling: {e}")
                        self._logged_movement_scale_build_error = True

            if current_state is not None:
                # Only apply scaling if dimensions match
                if len(current_state) == len(action_np):
                    # Scale: new_action = current_pos + (target_action - current_pos) * scale
                    delta = action_np - current_state
                    action_np = current_state + delta * movement_scale

                    if not hasattr(self, '_logged_movement_scale'):
                        print(f"[HIL] Movement scaling applied: scale={movement_scale:.1f}, max_delta={np.abs(delta).max():.2f}")
                        print(f"[HIL]   Current state (first 7): {current_state[:7].tolist()}")
                        print(f"[HIL]   Target action (first 7): {(current_state + delta)[:7].tolist()}")
                        print(f"[HIL]   Scaled action (first 7): {action_np[:7].tolist()}")
                        self._logged_movement_scale = True
                else:
                    if not hasattr(self, '_logged_scale_dim_mismatch'):
                        print(f"[HIL] WARNING: Cannot apply movement scaling - dimension mismatch: "
                              f"action={len(action_np)}, current_state={len(current_state)}")
                        self._logged_scale_dim_mismatch = True
            else:
                if not hasattr(self, '_logged_no_current_state'):
                    print(f"[HIL] WARNING: Cannot apply movement scaling - could not get current robot state")
                    print(f"[HIL]   action_names: {action_names}")
                    print(f"[HIL]   raw_obs keys: {list(raw_obs.keys()) if raw_obs else 'None'}")
                    self._logged_no_current_state = True

        # Build dict with named keys
        action_len = len(action_np) if hasattr(action_np, '__len__') else None
        if action_len is not None and action_len == len(action_names):
            result = {name: float(action_np[i]) for i, name in enumerate(action_names)}
            return result
        else:
            if not hasattr(self, '_logged_shape_mismatch'):
                print(f"[HIL] WARNING: Action shape mismatch: action has {action_len} elements, but names has {len(action_names)} elements")
                print(f"[HIL] Action type: {type(action_np)}, shape: {action_np.shape if hasattr(action_np, 'shape') else 'N/A'}")
                self._logged_shape_mismatch = True
            return {}

    def _pad_action_for_recording(self, action_dict: dict, raw_obs: dict) -> dict:
        """
        Pad action dict with missing motor positions for recording.

        If the policy only outputs actions for one arm (e.g., left arm = 7 motors)
        but the intervention dataset expects both arms (14 motors), we pad the
        action dict with the current robot positions for the other arm.

        This ensures recording can build frames without KeyError.

        Args:
            action_dict: Action dict from policy (may only have left arm keys)
            raw_obs: Raw robot observation with all motor positions

        Returns:
            Padded action dict with all motors expected by recording dataset
        """
        # Check if recording dataset exists and has action features
        if not hasattr(self.teleop, 'dataset') or self.teleop.dataset is None:
            return action_dict

        try:
            features = self.teleop.dataset.features
            if 'action' not in features or 'names' not in features['action']:
                return action_dict

            expected_names = features['action']['names']

            # Check if we need to pad (action_dict has fewer keys than expected)
            if len(action_dict) >= len(expected_names):
                return action_dict

            # Create padded dict starting with policy's action
            padded = action_dict.copy()

            # Add missing motor positions from raw_obs (current robot state)
            added_count = 0
            for name in expected_names:
                if name not in padded:
                    # Get from raw observation (robot's current position)
                    if name in raw_obs:
                        padded[name] = float(raw_obs[name])
                        added_count += 1

            if added_count > 0 and not hasattr(self, '_logged_padding'):
                print(f"[HIL] Padded action dict: added {added_count} motor positions from robot state")
                print(f"[HIL]   Policy outputs: {list(action_dict.keys())}")
                print(f"[HIL]   Dataset expects: {expected_names}")
                self._logged_padding = True

            return padded

        except Exception as e:
            if not hasattr(self, '_logged_pad_error'):
                print(f"[HIL] WARNING: Error padding action dict: {e}")
                self._logged_pad_error = True
            return action_dict

    def _run_human_teleop(self):
        """
        Execute one step of human teleoperation during HUMAN mode.

        Reads leader arm positions, sends to follower, and caches for recording.
        This is needed because the main teleop loop doesn't run during HIL mode.
        """
        # Check if teleop service has leader arm
        if not hasattr(self.teleop, 'leader') or self.teleop.leader is None:
            return

        robot = self.teleop.robot if hasattr(self.teleop, 'robot') else None
        if robot is None or not robot.is_connected:
            return

        try:
            # Read leader arm positions
            leader_action = self.teleop.leader.get_action()
            if not leader_action:
                return

            # Send to follower robot
            if self.robot_lock:
                with self.robot_lock:
                    robot.send_action(leader_action)
            else:
                robot.send_action(leader_action)

            # Cache for recording capture loop
            if hasattr(self.teleop, '_action_lock') and hasattr(self.teleop, '_latest_leader_action'):
                with self.teleop._action_lock:
                    self.teleop._latest_leader_action = leader_action.copy() if hasattr(leader_action, 'copy') else dict(leader_action)

        except Exception as e:
            # Suppress frequent errors during normal operation
            pass
