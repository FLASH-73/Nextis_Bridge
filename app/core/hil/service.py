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

import threading
from typing import Dict, Any
import logging

from .types import HILMode, HILSessionState
from .loop import HILLoopMixin

logger = logging.getLogger(__name__)


class HILService(HILLoopMixin):
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
