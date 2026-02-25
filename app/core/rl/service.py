"""HIL-SERL Reinforcement Learning Training Service.

Orchestrates online RL training with human-in-the-loop interventions.
Uses SAC (Soft Actor-Critic) with dual replay buffers:
  - Online buffer: ALL transitions (autonomous + human)
  - Offline buffer: Human intervention transitions only

Supports three reward sources:
  - SARM (Stage-Aware Reward Modeling): Learned from demos, fast local inference (recommended)
  - GVL (Gemini API): Zero-shot dense rewards via Gemini vision
  - Trained classifier: Binary reward from vision classifier

Architecture:
  - Actor thread: In-process, controls robot at 30Hz
  - Learner process: Separate process on GPU for SAC updates
  - Communication via multiprocessing Queues
"""

import json
import logging
import multiprocessing as mp
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty

import numpy as np
import torch

from .types import _APP_ROOT, _DEFAULT_DATASETS_PATH, _DEFAULT_MODELS_PATH, RLConfig, RLTrainingState

logger = logging.getLogger(__name__)


class RLService:
    """HIL-SERL training service with actor-learner architecture."""

    def __init__(
        self,
        robot=None,
        leader=None,
        teleop_service=None,
        camera_service=None,
        calibration_service=None,
        reward_classifier_service=None,
        gvl_reward_service=None,
        sarm_reward_service=None,
        robot_lock=None,
    ):
        self.robot = robot
        self.leader = leader
        self.teleop = teleop_service
        self.cameras = camera_service
        self.calibration = calibration_service
        self.reward_classifier_svc = reward_classifier_service
        self.gvl_reward_svc = gvl_reward_service
        self.sarm_reward_svc = sarm_reward_service
        self.robot_lock = robot_lock

        self.state = RLTrainingState()
        self.config = RLConfig()

        # Threading
        self._actor_thread = None
        self._learner_process = None
        self._stop_event = threading.Event()

        # Communication queues
        self._transitions_queue = None  # Actor → Learner
        self._params_queue = None  # Learner → Actor
        self._metrics_queue = None  # Learner → Actor (loss metrics)

        # Policy
        self._policy = None
        self._policy_config = None

        # Environment
        self._env = None

        # Reward model
        self._reward_classifier = None
        self._reward_config = None
        self._sarm_model = None
        self._sarm_image_history = None  # deque for SARM frame history

        # Buffers (actor-side references)
        self._online_buffer = None

        # Intervention tracking
        self._last_leader_pos = None
        self._intervention_count = 0
        self._autonomous_count = 0
        self._episode_rewards_sum = 0.0

        # Checkpoints
        self._models_path = _DEFAULT_MODELS_PATH / "rl_policies"
        self._models_path.mkdir(parents=True, exist_ok=True)

    def start_training(self, config: dict) -> dict:
        """Start HIL-SERL training session.

        Args:
            config: Training configuration dict

        Returns:
            dict with status
        """
        if self.state.status in ("running", "initializing"):
            return {"status": "error", "message": "Training already in progress"}

        if self.robot is None:
            return {"status": "error", "message": "No robot connected"}

        if not config.get("dataset_repo_id"):
            return {"status": "error", "message": "A demonstration dataset is required for RL training"}

        self.config = RLConfig(**{k: v for k, v in config.items() if hasattr(RLConfig, k)})
        self.state = RLTrainingState(total_episodes=self.config.max_episodes)
        self.state.status = "initializing"

        # Start in background
        self._stop_event.clear()
        self._actor_thread = threading.Thread(
            target=self._training_main,
            daemon=True,
        )
        self._actor_thread.start()

        return {"status": "started"}

    def stop_training(self) -> dict:
        """Stop RL training and save current policy."""
        if self.state.status not in ("running", "initializing", "paused"):
            return {"status": "error", "message": "No training in progress"}

        self._stop_event.set()

        # Wait for actor thread to finish
        if self._actor_thread and self._actor_thread.is_alive():
            self._actor_thread.join(timeout=5.0)

        # Stop learner process
        if self._learner_process and self._learner_process.is_alive():
            self._learner_process.terminate()
            self._learner_process.join(timeout=3.0)

        # Save final checkpoint
        self._save_checkpoint("final")

        self.state.status = "completed"
        return {"status": "stopped", "message": "Training stopped and policy saved"}

    def pause_training(self) -> dict:
        """Pause RL training (actor stops, learner keeps running)."""
        if self.state.status != "running":
            return {"status": "error", "message": "Training not running"}
        self.state.status = "paused"
        return {"status": "paused"}

    def resume_training(self) -> dict:
        """Resume paused RL training."""
        if self.state.status != "paused":
            return {"status": "error", "message": "Training not paused"}
        self.state.status = "running"
        return {"status": "resumed"}

    def get_status(self) -> dict:
        """Get current RL training status and metrics."""
        # Update GVL metrics if available
        if self.gvl_reward_svc and self.config.reward_source == "gvl":
            gvl_status = self.gvl_reward_svc.get_status()
            self.state.gvl_queries = gvl_status.get("total_queries", 0)
            self.state.gvl_avg_latency_ms = gvl_status.get("avg_latency_ms", 0)

        return asdict(self.state)

    def update_settings(self, settings: dict) -> dict:
        """Update settings mid-training."""
        for key, value in settings.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"[RL] Updated {key} = {value}")

        # Update GVL config if relevant
        if self.gvl_reward_svc and self.config.reward_source == "gvl":
            gvl_keys = {"gvl_query_interval": "query_interval",
                         "gvl_success_threshold": "success_threshold",
                         "task_description": "task_description"}
            for rl_key, gvl_key in gvl_keys.items():
                if rl_key in settings:
                    self.gvl_reward_svc.update_config(**{gvl_key: settings[rl_key]})

        return {"status": "updated"}

    # ----------- Main training loop -----------

    def _training_main(self):
        """Main training orchestration (runs in actor thread)."""
        try:
            self._initialize_training()
            self.state.status = "running"
            self._run_actor_loop()
        except Exception as e:
            logger.exception(f"[RL] Training failed: {e}")
            self.state.status = "failed"
            self.state.error = str(e)
        finally:
            self._cleanup()

    def _initialize_training(self):
        """Initialize all training components."""
        logger.info("[RL] Initializing HIL-SERL training...")

        # 1. Create environment
        from app.core.rl.env import NextisRobotEnv
        self._env = NextisRobotEnv(
            robot=self.robot,
            cameras=self.cameras,
            reset_position=self.config.reset_position,
            fps=self.config.fps,
            max_episode_steps=self.config.max_steps_per_episode,
            reset_time_s=self.config.reset_time_s,
            image_size=self.config.image_size,
            robot_lock=self.robot_lock,
        )

        # 2. Initialize reward source
        self._init_reward_source()

        # 3. Create SAC policy
        self._init_policy()

        # 4. Create replay buffers
        self._init_buffers()

        # 5. Create communication queues
        self._transitions_queue = mp.Queue(maxsize=100)
        self._params_queue = mp.Queue(maxsize=5)
        self._metrics_queue = mp.Queue(maxsize=50)

        # 6. Start learner process
        self._start_learner()

        logger.info("[RL] Initialization complete")

    def _init_reward_source(self):
        """Initialize the reward source (SARM, GVL, or classifier)."""
        if self.config.reward_source == "sarm":
            # SARM: Stage-Aware Reward Modeling (recommended)
            if not self.config.sarm_model_name:
                raise ValueError("SARM model name required when using SARM reward source")

            if self.sarm_reward_svc is None:
                from app.core.rl.rewards.sarm import SARMRewardService
                self.sarm_reward_svc = SARMRewardService()

            # Load the trained SARM model
            self._sarm_model = self.sarm_reward_svc.load_sarm(self.config.sarm_model_name)
            if self._sarm_model is None:
                raise ValueError(f"Failed to load SARM model: {self.config.sarm_model_name}")

            # Initialize image history buffer for SARM (needs frame history)
            # SARM uses n_obs_steps=8 frames by default
            sarm_config = self.sarm_reward_svc.get_model_config(self.config.sarm_model_name)
            n_frames = sarm_config.get("n_obs_steps", 8) if sarm_config else 8
            self._sarm_image_history = deque(maxlen=n_frames)

            logger.info(f"[RL] Using SARM reward model: {self.config.sarm_model_name} (history={n_frames} frames)")

        elif self.config.reward_source == "gvl":
            if self.gvl_reward_svc is None:
                from app.core.rl.rewards.gvl import GVLConfig, GVLRewardService
                gvl_config = GVLConfig(
                    task_description=self.config.task_description,
                    query_interval=self.config.gvl_query_interval,
                    success_threshold=self.config.gvl_success_threshold,
                    image_size=self.config.image_size,
                )
                self.gvl_reward_svc = GVLRewardService(config=gvl_config)
            else:
                # Update config
                self.gvl_reward_svc.update_config(
                    task_description=self.config.task_description,
                    query_interval=self.config.gvl_query_interval,
                    success_threshold=self.config.gvl_success_threshold,
                )
            logger.info("[RL] Using GVL (Gemini) reward source")

        elif self.config.reward_source == "classifier":
            if not self.reward_classifier_svc:
                from app.core.rl.rewards.classifier import RewardClassifierService
                self.reward_classifier_svc = RewardClassifierService()

            classifier, config_dict = self.reward_classifier_svc.load_classifier(
                self.config.reward_classifier_name
            )
            self._reward_classifier = classifier
            self._reward_config = config_dict
            logger.info(f"[RL] Using trained classifier: {self.config.reward_classifier_name}")

    def _init_policy(self):
        """Initialize SAC policy."""
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.sac.configuration_sac import SACConfig
        from lerobot.policies.sac.modeling_sac import SACPolicy

        # Build input/output features from environment
        input_features = {
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(self._env.n_joints,),
            ),
        }
        for cam_key in self._env.camera_keys:
            input_features[f"observation.images.{cam_key}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.config.image_size),
            )

        output_features = {
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self._env.n_joints,),
            ),
        }

        self._policy_config = SACConfig(
            input_features=input_features,
            output_features=output_features,
            device="cuda" if torch.cuda.is_available() else "cpu",
            discount=self.config.discount,
            actor_lr=self.config.actor_lr,
            critic_lr=self.config.critic_lr,
            temperature_lr=self.config.temperature_lr,
            online_buffer_capacity=self.config.online_buffer_capacity,
            offline_buffer_capacity=self.config.offline_buffer_capacity,
            utd_ratio=self.config.utd_ratio,
        )

        self._policy = SACPolicy(self._policy_config)
        self._policy.to(self._policy_config.device)
        logger.info(f"[RL] SAC policy initialized on {self._policy_config.device}")

    def _init_buffers(self):
        """Initialize replay buffers."""
        from lerobot.rl.buffer import ReplayBuffer

        state_keys = ["observation.state"]
        for cam_key in self._env.camera_keys:
            state_keys.append(f"observation.images.{cam_key}")

        self._online_buffer = ReplayBuffer(
            capacity=self.config.online_buffer_capacity,
            device=self._policy_config.device,
            state_keys=state_keys,
            storage_device="cpu",
        )

        # Load offline buffer from demonstrations if provided
        if self.config.dataset_repo_id:
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                dataset_path = _DEFAULT_DATASETS_PATH / self.config.dataset_repo_id
                dataset = LeRobotDataset(
                    repo_id=self.config.dataset_repo_id,
                    root=dataset_path,
                )

                self._offline_buffer = ReplayBuffer.from_lerobot_dataset(
                    dataset,
                    device=self._policy_config.device,
                    state_keys=state_keys,
                    capacity=self.config.offline_buffer_capacity,
                )
                logger.info(f"[RL] Loaded {len(self._offline_buffer)} transitions into offline buffer")
            except Exception as e:
                logger.warning(f"[RL] Failed to load offline buffer: {e}")
                self._offline_buffer = ReplayBuffer(
                    capacity=self.config.offline_buffer_capacity,
                    device=self._policy_config.device,
                    state_keys=state_keys,
                    storage_device="cpu",
                )
        else:
            self._offline_buffer = ReplayBuffer(
                capacity=self.config.offline_buffer_capacity,
                device=self._policy_config.device,
                state_keys=state_keys,
                storage_device="cpu",
            )

    def _start_learner(self):
        """Start the learner process for SAC updates."""
        # For now, run learner in-process on a separate thread
        # (multiprocessing requires pickling which is complex with robot references)
        self._learner_thread = threading.Thread(
            target=self._learner_loop,
            daemon=True,
        )
        self._learner_thread.start()
        logger.info("[RL] Learner thread started")

    def _learner_loop(self):
        """Learner loop: trains SAC on buffered transitions."""
        step = 0
        while not self._stop_event.is_set():
            # Wait for enough data
            if hasattr(self._online_buffer, '_num_added') and self._online_buffer._num_added < self.config.warmup_steps:
                time.sleep(0.5)
                continue

            # Check if buffers have data
            online_size = getattr(self._online_buffer, '_num_added', 0)
            if online_size < self.config.batch_size:
                time.sleep(0.5)
                continue

            try:
                # Sample from online buffer
                online_batch = self._online_buffer.sample(self.config.batch_size)

                # Optionally mix with offline buffer
                offline_size = getattr(self._offline_buffer, '_num_added', 0)
                if offline_size >= self.config.batch_size:
                    offline_batch = self._offline_buffer.sample(self.config.batch_size // 2)
                    # Concatenate batches
                    from lerobot.rl.buffer import concatenate_batch_transitions
                    batch = concatenate_batch_transitions(online_batch, offline_batch)
                else:
                    batch = online_batch

                # Update policy
                for _ in range(self.config.utd_ratio):
                    losses = self._policy.update(batch)
                    step += 1

                # Update state metrics
                if losses:
                    self.state.loss_critic = losses.get("critic_loss", 0.0)
                    self.state.loss_actor = losses.get("actor_loss", 0.0)
                    self.state.loss_temperature = losses.get("temperature_loss", 0.0)
                    self.state.training_step = step

            except Exception as e:
                if "empty" not in str(e).lower():
                    logger.debug(f"[RL Learner] Update error: {e}")
                time.sleep(0.1)

            # Control learner update rate
            time.sleep(0.01)

    # ----------- Actor loop -----------

    def _run_actor_loop(self):
        """Main actor loop: runs episodes, collects transitions."""
        for episode in range(self.config.max_episodes):
            if self._stop_event.is_set():
                break

            # Wait if paused
            while self.state.status == "paused" and not self._stop_event.is_set():
                time.sleep(0.1)

            if self._stop_event.is_set():
                break

            # Run one episode
            self._run_episode(episode)

            # Save checkpoint periodically
            if (episode + 1) % self.config.save_interval_episodes == 0:
                self._save_checkpoint(f"episode_{episode + 1}")

        if not self._stop_event.is_set():
            self.state.status = "completed"
            self._save_checkpoint("final")

    def _run_episode(self, episode_idx: int):
        """Run a single RL episode."""
        self.state.episode = episode_idx + 1
        self._episode_rewards_sum = 0.0
        self._last_leader_pos = None

        # Reset environment
        obs, _ = self._env.reset()

        # Reset reward source state
        if self.config.reward_source == "sarm" and self._sarm_image_history is not None:
            self._sarm_image_history.clear()
        elif self.config.reward_source == "gvl" and self.gvl_reward_svc:
            self.gvl_reward_svc.reset()

        for step_i in range(self.config.max_steps_per_episode):
            if self._stop_event.is_set() or self.state.status == "paused":
                break

            self.state.episode_step = step_i + 1
            loop_start = time.time()

            # 1. Check for human intervention
            is_intervention, leader_action = self._detect_intervention()
            self.state.is_human_intervening = is_intervention

            # 2. Get action
            if is_intervention and leader_action is not None:
                action = leader_action
                self._intervention_count += 1
            else:
                action = self._get_policy_action(obs)
                self._autonomous_count += 1

            # 3. Apply movement scale for safety
            action = self._apply_movement_scale(action, obs)

            # 3b. Safety check (every 3rd step = ~10Hz at 30fps)
            if step_i % 3 == 0 and self.teleop and hasattr(self.teleop, 'safety'):
                robot = self._env.robot if self._env else None
                if robot and hasattr(robot, 'is_connected') and robot.is_connected:
                    if not self.teleop.safety.check_all_limits(robot):
                        logger.error("[RL] SAFETY: Limit exceeded — stopping training")
                        self._stop_event.set()
                        break

            # 4. Step environment
            next_obs, _, terminated, truncated, info = self._env.step(action)

            # 5. Get reward
            reward = self._compute_reward(next_obs)
            self._episode_rewards_sum += reward
            self.state.current_reward = reward

            # Check for success
            if self._is_success(reward):
                terminated = True

            # 6. Store transition in online buffer
            state_dict = self._obs_to_state_dict(obs)
            next_state_dict = self._obs_to_state_dict(next_obs)

            try:
                self._online_buffer.add(
                    state=state_dict,
                    action=torch.tensor(action, dtype=torch.float32),
                    reward=reward,
                    next_state=next_state_dict,
                    done=terminated,
                    truncated=truncated,
                    complementary_info={"is_intervention": float(is_intervention)},
                )
                self.state.online_buffer_size = getattr(self._online_buffer, '_num_added', 0)

                # Also add interventions to offline buffer
                if is_intervention:
                    self._offline_buffer.add(
                        state=state_dict,
                        action=torch.tensor(action, dtype=torch.float32),
                        reward=reward,
                        next_state=next_state_dict,
                        done=terminated,
                        truncated=truncated,
                    )
                    self.state.offline_buffer_size = getattr(self._offline_buffer, '_num_added', 0)
            except Exception as e:
                logger.debug(f"[RL] Buffer add error: {e}")

            obs = next_obs

            if terminated or truncated:
                break

            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = (1.0 / self.config.fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Episode metrics
        total_steps = self._intervention_count + self._autonomous_count
        if total_steps > 0:
            self.state.intervention_rate = self._intervention_count / total_steps
        self.state.total_interventions = self._intervention_count
        self.state.total_autonomous_steps = self._autonomous_count

        # Track episode reward
        self.state.episode_rewards.append(self._episode_rewards_sum)
        if self.state.episode_rewards:
            # Average of last 10 episodes
            recent = self.state.episode_rewards[-10:]
            self.state.avg_reward = sum(recent) / len(recent)

        logger.info(f"[RL] Episode {episode_idx + 1}: reward={self._episode_rewards_sum:.2f}, "
                    f"steps={self.state.episode_step}, "
                    f"interventions={self._intervention_count}")

    # ----------- Helper methods -----------

    def _detect_intervention(self) -> tuple:
        """Detect leader arm intervention via velocity threshold.

        Returns:
            (is_intervention: bool, leader_action: np.ndarray or None)
        """
        if self.leader is None and self.teleop is None:
            return False, None

        try:
            # Get leader arm positions
            leader = self.teleop.leader if self.teleop else self.leader
            if leader is None:
                return False, None

            if self.robot_lock:
                with self.robot_lock:
                    current_pos = leader.get_action()
            else:
                current_pos = leader.get_action()

            if current_pos is None:
                return False, None

            # Compute velocity (max position delta)
            if self._last_leader_pos is None:
                self._last_leader_pos = current_pos.copy() if isinstance(current_pos, dict) else current_pos
                return False, None

            if isinstance(current_pos, dict):
                max_delta = 0.0
                for key, val in current_pos.items():
                    prev = self._last_leader_pos.get(key, val)
                    delta = abs(val - prev)
                    max_delta = max(max_delta, delta)
                self._last_leader_pos = current_pos.copy()

                # If velocity exceeds threshold, human is intervening
                velocity = max_delta * self.config.fps
                if velocity > self.config.intervention_velocity_threshold:
                    # Convert leader positions to action array
                    action = self._leader_pos_to_action(current_pos)
                    return True, action
            else:
                # Array-based positions
                if isinstance(current_pos, np.ndarray):
                    delta = np.max(np.abs(current_pos - self._last_leader_pos))
                    self._last_leader_pos = current_pos.copy()
                    velocity = delta * self.config.fps
                    if velocity > self.config.intervention_velocity_threshold:
                        return True, current_pos
                else:
                    self._last_leader_pos = current_pos

            return False, None

        except Exception as e:
            logger.debug(f"[RL] Intervention detection error: {e}")
            return False, None

    def _leader_pos_to_action(self, leader_pos: dict) -> np.ndarray:
        """Convert leader position dict to normalized action array."""
        action = np.zeros(self._env.n_joints, dtype=np.float32)
        for i, name in enumerate(self._env.motor_names):
            # Try various key patterns
            for key_pattern in [name, f"{name}.pos", name.split("_", 1)[-1]]:
                if key_pattern in leader_pos:
                    action[i] = self._env._normalize_position(name, leader_pos[key_pattern])
                    break
        return action

    def _get_policy_action(self, obs: dict) -> np.ndarray:
        """Get action from SAC policy."""
        try:
            with torch.no_grad():
                # Build policy observation
                policy_obs = self._obs_to_policy_input(obs)
                action = self._policy.select_action(policy_obs)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                return action.flatten()
        except Exception as e:
            logger.debug(f"[RL] Policy inference error: {e}")
            return np.zeros(self._env.n_joints, dtype=np.float32)

    def _apply_movement_scale(self, action: np.ndarray, obs: dict) -> np.ndarray:
        """Apply movement scale safety limiter."""
        if self.config.movement_scale >= 1.0:
            return action

        current_pos = obs.get("agent_pos", np.zeros_like(action))
        delta = action - current_pos
        scaled_action = current_pos + delta * self.config.movement_scale
        return np.clip(scaled_action, -1.0, 1.0)

    def _compute_reward(self, obs: dict) -> float:
        """Compute reward from observation using configured source."""
        if self.config.reward_source == "sarm" and self._sarm_model is not None:
            # SARM: Stage-Aware Reward Modeling
            # Get current camera image and add to history
            images = []
            for cam_key in self._env.camera_keys:
                pixel_key = f"pixels/{cam_key}"
                if pixel_key in obs:
                    images.append(obs[pixel_key])

            if images:
                # Use first camera for SARM (primary view)
                self._sarm_image_history.append(images[0])

                # Get state (joint positions)
                state = obs.get("agent_pos", np.zeros(self._env.n_joints))

                # Predict reward using SARM
                return self.sarm_reward_svc.predict_reward(
                    self._sarm_model,
                    list(self._sarm_image_history),
                    state,
                    self.config.task_description,
                )
            return 0.0

        elif self.config.reward_source == "gvl" and self.gvl_reward_svc:
            # GVL: Zero-shot via Gemini API
            images = []
            for cam_key in self._env.camera_keys:
                pixel_key = f"pixels/{cam_key}"
                if pixel_key in obs:
                    images.append(obs[pixel_key])

            if images:
                return self.gvl_reward_svc.predict_reward(images)
            return 0.0

        elif self.config.reward_source == "classifier" and self._reward_classifier:
            # Trained classifier: Binary reward
            images = []
            for cam_key in self._env.camera_keys:
                pixel_key = f"pixels/{cam_key}"
                if pixel_key in obs:
                    img = obs[pixel_key]
                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img).permute(2, 0, 1).float()  # HWC -> CHW
                    images.append(img.unsqueeze(0).to(self._reward_classifier.device))

            if images:
                return self.reward_classifier_svc.predict_reward(
                    self._reward_classifier, images
                )
            return 0.0

        return 0.0

    def _is_success(self, reward: float) -> bool:
        """Check if current reward indicates task success."""
        if self.config.reward_source == "sarm":
            # SARM: Success when progress reaches threshold (e.g., 0.95)
            return reward >= 0.95
        elif self.config.reward_source == "gvl" and self.gvl_reward_svc:
            return self.gvl_reward_svc.is_success(reward)
        elif self.config.reward_source == "classifier":
            return reward >= 1.0
        return False

    def _obs_to_state_dict(self, obs: dict) -> dict:
        """Convert environment observation to buffer state dict."""
        state = {}

        # Joint positions
        if "agent_pos" in obs:
            state["observation.state"] = torch.tensor(obs["agent_pos"], dtype=torch.float32)

        # Camera images
        for cam_key in self._env.camera_keys:
            pixel_key = f"pixels/{cam_key}"
            if pixel_key in obs:
                img = obs[pixel_key]
                if isinstance(img, np.ndarray):
                    # HWC uint8 -> CHW float32
                    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                state[f"observation.images.{cam_key}"] = img

        return state

    def _obs_to_policy_input(self, obs: dict) -> dict:
        """Convert environment observation to policy input format."""
        device = self._policy_config.device
        policy_input = {}

        if "agent_pos" in obs:
            policy_input["observation.state"] = torch.tensor(
                obs["agent_pos"], dtype=torch.float32
            ).unsqueeze(0).to(device)

        for cam_key in self._env.camera_keys:
            pixel_key = f"pixels/{cam_key}"
            if pixel_key in obs:
                img = obs[pixel_key]
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                policy_input[f"observation.images.{cam_key}"] = img.unsqueeze(0).to(device)

        return policy_input

    def _save_checkpoint(self, name: str):
        """Save policy checkpoint."""
        try:
            save_dir = self._models_path / f"sac_{name}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save policy state dict
            torch.save(self._policy.state_dict(), save_dir / "policy.pt")

            # Save config
            with open(save_dir / "config.json", "w") as f:
                json.dump(asdict(self.config), f, indent=2, default=str)

            # Save training state
            state_dict = asdict(self.state)
            with open(save_dir / "training_state.json", "w") as f:
                json.dump(state_dict, f, indent=2, default=str)

            logger.info(f"[RL] Saved checkpoint: {save_dir}")
        except Exception as e:
            logger.warning(f"[RL] Failed to save checkpoint: {e}")

    def _cleanup(self):
        """Clean up resources after training."""
        if self._learner_process and self._learner_process.is_alive():
            self._learner_process.terminate()

        if self._env:
            self._env.close()

        # Clear queues
        for q in [self._transitions_queue, self._params_queue, self._metrics_queue]:
            if q:
                try:
                    while not q.empty():
                        q.get_nowait()
                except Exception:
                    pass

        logger.info("[RL] Training cleanup complete")
