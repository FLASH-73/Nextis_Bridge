"""Unified deployment runtime for all policy execution modes.

Replaces the three separate control loops (orchestrator._inference_loop,
hil/loop._hil_loop, rl/service._run_actor_loop) with a single 30Hz loop.
The mode only determines (a) where the action comes from and (b) what
happens with the data.  Safety is always applied.
"""

import logging
import threading
import time
from typing import Dict, List, Optional

from .intervention import InterventionDetector
from .observation_builder import ObservationBuilder
from .safety_pipeline import SafetyPipeline
from .types import (
    ActionSource,
    DeploymentConfig,
    DeploymentMode,
    DeploymentStatus,
    RuntimeState,
    SafetyConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_LOOP_HZ = 30


class DeploymentRuntime:
    """Unified runtime for INFERENCE, HIL, and HIL_SERL deployment modes.

    Usage::

        runtime = DeploymentRuntime(teleop, training, arm_registry, cameras, lock)
        runtime.start(config, active_arm_ids=["leader_left", "follower_left"])
        # ... later ...
        runtime.stop()
    """

    def __init__(
        self,
        teleop_service,
        training_service,
        arm_registry,
        camera_service,
        robot_lock: threading.Lock,
    ):
        self._teleop = teleop_service
        self._training = training_service
        self._arm_registry = arm_registry
        self._camera_service = camera_service
        self._robot_lock = robot_lock

        # Runtime state
        self._state = RuntimeState.IDLE
        self._state_lock = threading.Lock()
        self._config: Optional[DeploymentConfig] = None
        self._stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None

        # Per-session objects (set during start, cleared on stop)
        self._policy = None
        self._checkpoint_path = None
        self._policy_config = None
        self._obs_builder: Optional[ObservationBuilder] = None
        self._safety_pipeline: Optional[SafetyPipeline] = None
        self._intervention_detector: Optional[InterventionDetector] = None
        self._leader = None
        self._follower = None
        self._arm_defs: List = []

        # Counters
        self._frame_count = 0
        self._episode_count = 0
        self._current_episode_frames = 0
        self._autonomous_frames = 0
        self._human_frames = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        config: DeploymentConfig,
        active_arm_ids: List[str],
    ) -> None:
        """Start a deployment session.

        Args:
            config: Deployment configuration (mode, policy_id, safety, etc.).
            active_arm_ids: Follower arm IDs to control.

        Raises:
            RuntimeError: If not in IDLE state or startup fails.
        """
        if not self._transition(RuntimeState.STARTING):
            raise RuntimeError(
                f"Cannot start from state {self._state.value}"
            )

        try:
            self._config = config
            self._stop_event.clear()
            self._frame_count = 0
            self._episode_count = 0
            self._current_episode_frames = 0
            self._autonomous_frames = 0
            self._human_frames = 0

            # 1. Resolve arms
            self._resolve_arms(active_arm_ids)

            # 2. Load policy
            self._load_policy(config.policy_id)

            # 3. Populate safety config from arm definitions
            self._populate_safety_config(config.safety)

            # 4. Create safety pipeline
            safety_layer = getattr(self._teleop, "safety", None)
            self._safety_pipeline = SafetyPipeline(
                config.safety, safety_layer=safety_layer
            )

            # 5. Create observation builder
            self._obs_builder = ObservationBuilder(
                checkpoint_path=self._checkpoint_path,
                policy=self._policy,
                policy_type=getattr(self._policy_config, "policy_type", ""),
                task=config.task or "",
            )

            # 6. Create intervention detector (for HIL/SERL modes)
            policy_arms = (
                getattr(self._policy_config, "arms", None) or ["left", "right"]
            )
            self._intervention_detector = InterventionDetector(
                policy_arms=policy_arms,
                loop_hz=DEFAULT_LOOP_HZ,
            )

            # 7. Start recording for HIL/SERL modes
            if config.mode in (DeploymentMode.HIL, DeploymentMode.HIL_SERL):
                self._start_recording(config)

            # 8. Start control loop
            self._transition(RuntimeState.RUNNING)
            self._loop_thread = threading.Thread(
                target=self._control_loop,
                name="deployment-runtime",
                daemon=True,
            )
            self._loop_thread.start()
            logger.info(
                "Deployment started: mode=%s, policy=%s",
                config.mode.value,
                config.policy_id,
            )

        except Exception as e:
            logger.error("Deployment start failed: %s", e)
            self._transition(RuntimeState.ERROR)
            raise RuntimeError(f"Deployment start failed: {e}") from e

    def stop(self) -> None:
        """Stop the deployment session."""
        if self._state == RuntimeState.IDLE:
            return

        self._transition(RuntimeState.STOPPING)
        self._stop_event.set()

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)

        # Cleanup
        if self._safety_pipeline:
            self._safety_pipeline.reset()
        if self._intervention_detector:
            self._intervention_detector.reset()

        self._policy = None
        self._checkpoint_path = None
        self._policy_config = None
        self._obs_builder = None
        self._safety_pipeline = None
        self._intervention_detector = None
        self._leader = None
        self._follower = None
        self._arm_defs = []
        self._config = None
        self._loop_thread = None

        self._transition(RuntimeState.IDLE)
        logger.info("Deployment stopped")

    def pause(self) -> bool:
        """Pause the deployment (hold position)."""
        return self._transition(RuntimeState.PAUSED)

    def resume(self) -> bool:
        """Resume autonomous execution from PAUSED state."""
        return self._transition(RuntimeState.RUNNING)

    def reset(self) -> bool:
        """Reset from ESTOP or ERROR back to IDLE.

        Clears safety pipeline state.  Caller should verify physical safety
        before calling.
        """
        if self._state not in (RuntimeState.ESTOP, RuntimeState.ERROR):
            return False

        self._stop_event.set()
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)

        if self._safety_pipeline:
            self._safety_pipeline.clear_estop()
            self._safety_pipeline.reset()

        # Force transition (ESTOP/ERROR have no valid transitions in the table,
        # but reset is the explicit escape hatch)
        with self._state_lock:
            self._state = RuntimeState.IDLE
        logger.info("Deployment reset from %s to IDLE", self._state.value)
        return True

    def estop(self) -> bool:
        """Emergency stop — hold position immediately."""
        if self._safety_pipeline:
            obs = self._get_observation() or {}
            self._safety_pipeline.trigger_estop(obs)
        with self._state_lock:
            self._state = RuntimeState.ESTOP
        logger.critical("E-STOP triggered via API")
        return True

    def update_speed_scale(self, scale: float) -> None:
        """Update the safety pipeline speed scale."""
        if self._safety_pipeline:
            self._safety_pipeline.update_speed_scale(scale)

    def get_status(self) -> DeploymentStatus:
        """Return a snapshot of current deployment status."""
        safety_readings = {}
        if self._safety_pipeline:
            readings = self._safety_pipeline.get_readings()
            safety_readings = {
                "per_motor_velocity": readings.per_motor_velocity,
                "per_motor_torque": readings.per_motor_torque,
                "active_clamps": readings.active_clamps,
                "estop_active": readings.estop_active,
                "speed_scale": readings.speed_scale,
            }

        policy_config_dict = None
        if self._policy_config:
            policy_config_dict = {
                "cameras": getattr(self._policy_config, "cameras", []),
                "arms": getattr(self._policy_config, "arms", []),
                "policy_type": getattr(self._policy_config, "policy_type", ""),
            }

        return DeploymentStatus(
            state=self._state,
            mode=self._config.mode if self._config else DeploymentMode.INFERENCE,
            frame_count=self._frame_count,
            episode_count=self._episode_count,
            current_episode_frames=self._current_episode_frames,
            safety=safety_readings,
            policy_config=policy_config_dict,
        )

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        """Unified 30Hz control loop for all deployment modes."""
        loop_period = 1.0 / DEFAULT_LOOP_HZ
        dt = loop_period
        is_hil = self._config and self._config.mode in (
            DeploymentMode.HIL,
            DeploymentMode.HIL_SERL,
        )
        logger.info("Control loop started (%.0f Hz)", DEFAULT_LOOP_HZ)

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            try:
                # 1. Get observation from follower
                raw_obs = self._get_observation()
                if raw_obs is None:
                    time.sleep(0.01)
                    continue

                # 2. Intervention detection (HIL/SERL only)
                action_source = ActionSource.POLICY
                if is_hil and self._intervention_detector and self._leader:
                    is_intervening, velocity = self._intervention_detector.check(
                        self._leader
                    )
                    action_source = self._update_state_from_intervention(
                        is_intervening
                    )

                # 3. Determine action based on source
                if self._state == RuntimeState.PAUSED:
                    action_source = ActionSource.HOLD

                action = self._get_action(action_source, raw_obs)
                if action is None:
                    time.sleep(0.001)
                    continue

                # 4. ALWAYS apply safety pipeline
                observation_positions = {
                    k: v
                    for k, v in raw_obs.items()
                    if isinstance(v, (int, float))
                }
                filtered_action = self._safety_pipeline.process(
                    action, observation_positions, robot=self._follower, dt=dt
                )

                # 5. Send to robot
                self._send_action(filtered_action)

                # 6. Cache for recording
                self._cache_for_recording(filtered_action, raw_obs)

                # 7. Update counters
                self._frame_count += 1
                self._current_episode_frames += 1
                if action_source == ActionSource.HUMAN:
                    self._human_frames += 1
                elif action_source == ActionSource.POLICY:
                    self._autonomous_frames += 1

            except Exception as e:
                logger.error("Control loop error: %s", e)

            # Maintain loop rate
            elapsed = time.monotonic() - t0
            sleep_time = loop_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Control loop stopped")

    # ------------------------------------------------------------------
    # Action sources
    # ------------------------------------------------------------------

    def _get_action(
        self, source: ActionSource, raw_obs: dict
    ) -> Optional[Dict[str, float]]:
        """Get action dict based on the current action source."""
        if source == ActionSource.HOLD:
            return {
                k: v
                for k, v in raw_obs.items()
                if isinstance(v, (int, float))
            }

        if source == ActionSource.HUMAN:
            return self._get_human_action()

        # POLICY
        return self._get_policy_action(raw_obs)

    def _get_policy_action(self, raw_obs: dict) -> Optional[Dict[str, float]]:
        """Run policy inference and return denormalized action dict."""
        if self._policy is None or self._obs_builder is None:
            return None

        policy_obs = self._obs_builder.prepare_observation(raw_obs)
        action = self._policy.select_action(policy_obs)

        movement_scale = 1.0
        if self._config:
            movement_scale = self._config.movement_scale

        return self._obs_builder.convert_action_to_dict(
            action, raw_obs, movement_scale=movement_scale
        )

    def _get_human_action(self) -> Optional[Dict[str, float]]:
        """Read leader arm positions for human teleop."""
        if self._leader is None:
            return None
        try:
            action = self._leader.get_action()
            return dict(action) if action else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self) -> Optional[dict]:
        """Get observation from follower robot with lock."""
        if self._follower is None:
            return None
        if hasattr(self._follower, "is_connected") and not self._follower.is_connected:
            return None

        try:
            if self._robot_lock:
                with self._robot_lock:
                    return self._follower.get_observation()
            return self._follower.get_observation()
        except Exception as e:
            logger.debug("Observation error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Action sending
    # ------------------------------------------------------------------

    def _send_action(self, action_dict: Dict[str, float]) -> None:
        """Send action to robot, handling bimanual/single-arm cases."""
        if self._follower is None or not action_dict:
            return

        try:
            if self._robot_lock:
                with self._robot_lock:
                    self._send_partial_action(self._follower, action_dict)
            else:
                self._send_partial_action(self._follower, action_dict)
        except Exception as e:
            logger.debug("Send action error: %s", e)

    @staticmethod
    def _send_partial_action(robot, action_dict: dict) -> None:
        """Send action only to arms that have entries in the dict.

        For bimanual robots, splits by left_/right_ prefix and strips
        the prefix before sending to individual arms.
        """
        is_bimanual = hasattr(robot, "left_arm") and hasattr(robot, "right_arm")
        if not is_bimanual:
            robot.send_action(action_dict)
            return

        left_action = {
            k.removeprefix("left_"): v
            for k, v in action_dict.items()
            if k.startswith("left_")
        }
        right_action = {
            k.removeprefix("right_"): v
            for k, v in action_dict.items()
            if k.startswith("right_")
        }

        if left_action:
            try:
                robot.left_arm.send_action(left_action)
            except Exception as e:
                logger.debug("Left arm send error: %s", e)

        if right_action:
            try:
                robot.right_arm.send_action(right_action)
            except Exception as e:
                logger.debug("Right arm send error: %s", e)

    # ------------------------------------------------------------------
    # Recording cache
    # ------------------------------------------------------------------

    def _cache_for_recording(
        self, action_dict: Dict[str, float], raw_obs: dict
    ) -> None:
        """Cache action in teleop for the recording capture thread."""
        if not hasattr(self._teleop, "_action_lock") or not hasattr(
            self._teleop, "_latest_leader_action"
        ):
            return

        padded = self._pad_action_for_recording(action_dict, raw_obs)
        with self._teleop._action_lock:
            self._teleop._latest_leader_action = padded

    def _pad_action_for_recording(
        self, action_dict: dict, raw_obs: dict
    ) -> dict:
        """Pad action dict with missing motor positions for recording.

        If the policy only outputs one arm but the dataset expects both,
        fill in the other arm's current positions from raw_obs.
        """
        if not hasattr(self._teleop, "dataset") or self._teleop.dataset is None:
            return dict(action_dict)

        try:
            features = self._teleop.dataset.features
            if "action" not in features or "names" not in features["action"]:
                return dict(action_dict)

            expected_names = features["action"]["names"]
            if len(action_dict) >= len(expected_names):
                return dict(action_dict)

            padded = dict(action_dict)
            for name in expected_names:
                if name not in padded and name in raw_obs:
                    padded[name] = float(raw_obs[name])

            return padded

        except Exception:
            return dict(action_dict)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _transition(self, new_state: RuntimeState) -> bool:
        """Attempt a state transition.  Returns True if successful."""
        with self._state_lock:
            if RuntimeState.can_transition(self._state, new_state):
                old = self._state
                self._state = new_state
                logger.debug("State: %s → %s", old.value, new_state.value)
                return True

            logger.warning(
                "Invalid transition: %s → %s",
                self._state.value,
                new_state.value,
            )
            return False

    def _update_state_from_intervention(
        self, is_intervening: bool
    ) -> ActionSource:
        """Update runtime state based on intervention detection.

        Returns the ActionSource to use for this frame.
        """
        if is_intervening:
            if self._state == RuntimeState.RUNNING:
                self._transition(RuntimeState.HUMAN_ACTIVE)
            return ActionSource.HUMAN

        # Not intervening — check if human went idle
        if self._state == RuntimeState.HUMAN_ACTIVE:
            if self._intervention_detector and self._intervention_detector.is_idle():
                self._transition(RuntimeState.PAUSED)
                return ActionSource.HOLD

            # Still in human mode (within idle timeout)
            return ActionSource.HUMAN

        if self._state == RuntimeState.PAUSED:
            return ActionSource.HOLD

        return ActionSource.POLICY

    # ------------------------------------------------------------------
    # Startup helpers
    # ------------------------------------------------------------------

    def _resolve_arms(self, active_arm_ids: List[str]) -> None:
        """Resolve leader/follower from arm registry pairings.

        Follows the same pattern as teleop/service.py start().
        """
        if not self._arm_registry:
            raise RuntimeError("No arm registry available")

        pairings = self._arm_registry.get_active_pairings(active_arm_ids)
        if not pairings:
            raise RuntimeError(
                f"No pairings found for arms: {active_arm_ids}"
            )

        # Use first pairing (deployment controls one policy at a time)
        pairing = pairings[0]
        leader_id = pairing["leader_id"]
        follower_id = pairing["follower_id"]

        # Auto-connect if needed
        for arm_id in (leader_id, follower_id):
            if arm_id not in self._arm_registry.arm_instances:
                logger.info("Auto-connecting arm: %s", arm_id)
                self._arm_registry.connect_arm(arm_id)

        self._leader = self._arm_registry.arm_instances.get(leader_id)
        self._follower = self._arm_registry.arm_instances.get(follower_id)

        if self._follower is None:
            raise RuntimeError(f"Follower arm {follower_id} not available")

        # Collect arm definitions for safety config
        self._arm_defs = [
            self._arm_registry.arms[aid]
            for aid in (leader_id, follower_id)
            if aid in self._arm_registry.arms
        ]

        logger.info(
            "Arms resolved: leader=%s, follower=%s",
            leader_id,
            follower_id,
        )

    def _load_policy(self, policy_id: str) -> None:
        """Load policy from training service.

        Follows the same pattern as orchestrator.py deploy_policy().
        """
        if not self._training:
            raise RuntimeError("No training service available")

        policy_info = self._training.get_policy(policy_id)
        if policy_info is None:
            raise RuntimeError(f"Policy not found: {policy_id}")

        self._policy_config = self._training.get_policy_config(policy_id)

        checkpoint_path = getattr(policy_info, "checkpoint_path", None)
        if not checkpoint_path:
            raise RuntimeError(
                f"Policy {policy_id} has no checkpoint path"
            )

        self._checkpoint_path = checkpoint_path

        # Load the policy model
        try:
            import json
            from pathlib import Path

            cp = Path(checkpoint_path)
            config_path = cp / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"config.json not found in {cp}"
                )

            with open(config_path) as f:
                policy_cfg = json.load(f)

            policy_type = policy_cfg.get("policy_type", "")
            from lerobot.policies.factory import get_policy_class

            policy_cls = get_policy_class(policy_type)
            self._policy = policy_cls.from_pretrained(str(cp))
            logger.info(
                "Policy loaded: %s (%s)", policy_id, policy_type
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load policy {policy_id}: {e}"
            ) from e

    def _populate_safety_config(self, safety: SafetyConfig) -> None:
        """Fill joint_limits and motor_models from arm definitions."""
        for arm_def in self._arm_defs:
            motor_model = arm_def.motor_type.value.upper()

            # Map motor type names to deployment velocity limit keys
            model_map = {
                "DAMIAO": "J8009P",    # Default to largest Damiao
                "STS3215": "STS3215",
                "DYNAMIXEL_XL330": "STS3215",  # Similar class
                "DYNAMIXEL_XL430": "STS3215",
            }
            model_key = model_map.get(motor_model, motor_model)

            # Get motor names from arm config if available
            motor_names = arm_def.config.get("motor_names", [])
            motor_models_cfg = arm_def.config.get("motor_models", {})

            for name in motor_names:
                # Per-motor model if available (e.g. Damiao arms have
                # different motor types per joint)
                per_motor_model = motor_models_cfg.get(name, model_key)
                safety.motor_models[name] = per_motor_model

            # Joint limits from calibration ranges in config
            joint_limits = arm_def.config.get("joint_limits", {})
            for name, limits in joint_limits.items():
                if isinstance(limits, (list, tuple)) and len(limits) == 2:
                    safety.joint_limits[name] = tuple(limits)

    def _start_recording(self, config: DeploymentConfig) -> None:
        """Start recording session for HIL/SERL modes."""
        if not self._teleop:
            return

        try:
            if hasattr(self._teleop, "start_recording_session"):
                repo_id = config.intervention_dataset or f"deployment_{config.policy_id}"
                task = config.task or "deployment"
                self._teleop.start_recording_session(
                    repo_id=repo_id,
                    task=task,
                    fps=DEFAULT_LOOP_HZ,
                )
                logger.info("Recording session started: %s", repo_id)
        except Exception as e:
            logger.warning("Failed to start recording session: %s", e)
