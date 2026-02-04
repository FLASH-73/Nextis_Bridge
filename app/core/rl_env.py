"""Gym-compatible environment wrapper for Nextis robots.

Wraps the existing Nextis robot + camera infrastructure as a
gymnasium.Env for use with HIL-SERL reinforcement learning.
Uses joint-space control (direct motor positions) - no IK needed.
"""

import logging
import time

import gymnasium as gym
import numpy as np
import torch

logger = logging.getLogger(__name__)


class NextisRobotEnv(gym.Env):
    """Gym environment wrapping Nextis's existing robot for RL training.

    Observation space:
        - agent_pos: Joint positions of the robot (normalized to [-1, 1])
        - pixels: Dict of camera images {camera_key: (H, W, C) uint8}

    Action space:
        - Joint position targets (normalized to [-1, 1])

    Episode termination:
        - Reward classifier predicts success
        - Max steps reached (truncation)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        robot,
        cameras,
        reset_position: dict,
        motor_names: list = None,
        fps: int = 30,
        max_episode_steps: int = 300,
        reset_time_s: float = 3.0,
        image_size: tuple = (224, 224),
        robot_lock=None,
    ):
        """Initialize the environment.

        Args:
            robot: LeRobot robot instance (BiUmbraFollower or UmbraFollowerRobot)
            cameras: CameraService or dict of camera objects
            reset_position: Dict of motor_name -> position for reset homing
            motor_names: List of motor names to control (auto-detect if None)
            fps: Control frequency in Hz
            max_episode_steps: Maximum steps per episode
            reset_time_s: Time in seconds for smooth reset trajectory
            image_size: Target image size (H, W) for observations
            robot_lock: Optional threading lock for thread-safe robot access
        """
        super().__init__()

        self.robot = robot
        self.cameras = cameras
        self.reset_position = reset_position
        self.fps = fps
        self.max_episode_steps = max_episode_steps
        self.reset_time_s = reset_time_s
        self.image_size = image_size
        self.robot_lock = robot_lock

        # Auto-detect motor names if not provided
        if motor_names is None:
            self.motor_names = self._detect_motor_names()
        else:
            self.motor_names = motor_names

        self.n_joints = len(self.motor_names)
        self._step_count = 0

        # Detect camera keys
        self.camera_keys = self._detect_camera_keys()

        # Define spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        # Observation space: agent_pos + pixels
        obs_spaces = {
            "agent_pos": gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
            ),
        }
        for cam_key in self.camera_keys:
            obs_spaces[f"pixels/{cam_key}"] = gym.spaces.Box(
                low=0, high=255,
                shape=(image_size[0], image_size[1], 3),
                dtype=np.uint8,
            )

        self.observation_space = gym.spaces.Dict(obs_spaces)

        # Motor range for normalization (will be populated on first observation)
        self._motor_ranges = {}
        self._init_motor_ranges()

        logger.info(f"[NextisRobotEnv] Initialized: {self.n_joints} joints, "
                    f"{len(self.camera_keys)} cameras, {fps}Hz, "
                    f"max {max_episode_steps} steps/episode")

    def _detect_motor_names(self) -> list:
        """Auto-detect motor names from the robot."""
        from lerobot.robots.bi_umbra_follower.bi_umbra_follower import BiUmbraFollower

        if isinstance(self.robot, BiUmbraFollower):
            # Bimanual: get motors from both arms
            names = []
            for arm_name, arm in [("left", self.robot.left_arm), ("right", self.robot.right_arm)]:
                for motor_name in arm.bus.motors:
                    names.append(f"{arm_name}_{motor_name}")
            return names
        else:
            # Single arm
            return list(self.robot.bus.motors.keys())

    def _detect_camera_keys(self) -> list:
        """Detect available camera keys."""
        if hasattr(self.cameras, 'camera_map') and self.cameras.camera_map:
            return sorted(self.cameras.camera_map.keys())
        elif isinstance(self.cameras, dict):
            return sorted(self.cameras.keys())
        return []

    def _init_motor_ranges(self):
        """Initialize motor position ranges for normalization."""
        from lerobot.robots.bi_umbra_follower.bi_umbra_follower import BiUmbraFollower

        if isinstance(self.robot, BiUmbraFollower):
            for arm_name, arm in [("left", self.robot.left_arm), ("right", self.robot.right_arm)]:
                if hasattr(arm, 'calibration') and arm.calibration:
                    for motor_name, cal in arm.calibration.items():
                        key = f"{arm_name}_{motor_name}"
                        self._motor_ranges[key] = {
                            "min": getattr(cal, "range_min", 0),
                            "max": getattr(cal, "range_max", 4096),
                        }
                else:
                    for motor_name in arm.bus.motors:
                        key = f"{arm_name}_{motor_name}"
                        self._motor_ranges[key] = {"min": 0, "max": 4096}
        else:
            if hasattr(self.robot, 'calibration') and self.robot.calibration:
                for motor_name, cal in self.robot.calibration.items():
                    self._motor_ranges[motor_name] = {
                        "min": getattr(cal, "range_min", 0),
                        "max": getattr(cal, "range_max", 4096),
                    }
            else:
                for motor_name in self.robot.bus.motors:
                    self._motor_ranges[motor_name] = {"min": 0, "max": 4096}

    def _normalize_position(self, name: str, value: float) -> float:
        """Normalize motor position to [-1, 1]."""
        r = self._motor_ranges.get(name, {"min": 0, "max": 4096})
        r_min, r_max = r["min"], r["max"]
        if r_max == r_min:
            return 0.0
        return 2.0 * (value - r_min) / (r_max - r_min) - 1.0

    def _denormalize_position(self, name: str, value: float) -> float:
        """Denormalize from [-1, 1] to motor position."""
        r = self._motor_ranges.get(name, {"min": 0, "max": 4096})
        r_min, r_max = r["min"], r["max"]
        return (value + 1.0) / 2.0 * (r_max - r_min) + r_min

    def _get_observation(self) -> dict:
        """Read current observation from robot and cameras."""
        # Read joint positions
        if self.robot_lock:
            with self.robot_lock:
                raw_obs = self.robot.get_observation()
        else:
            raw_obs = self.robot.get_observation()

        # Normalize joint positions
        joint_positions = np.zeros(self.n_joints, dtype=np.float32)
        for i, name in enumerate(self.motor_names):
            # Handle both prefixed and unprefixed keys
            pos_key = f"{name}.pos" if f"{name}.pos" in raw_obs else name
            if pos_key in raw_obs:
                joint_positions[i] = self._normalize_position(name, raw_obs[pos_key])

        obs = {"agent_pos": joint_positions}

        # Read camera images
        for cam_key in self.camera_keys:
            try:
                if hasattr(self.cameras, 'camera_map'):
                    cam = self.cameras.camera_map.get(cam_key)
                    if cam and hasattr(cam, 'async_read'):
                        frame = cam.async_read(blocking=False)
                    else:
                        frame = None
                elif isinstance(self.cameras, dict):
                    cam = self.cameras.get(cam_key)
                    if cam and hasattr(cam, 'async_read'):
                        frame = cam.async_read(blocking=False)
                    else:
                        frame = None
                else:
                    frame = None

                if frame is not None:
                    # Ensure correct format (H, W, C) uint8
                    if isinstance(frame, torch.Tensor):
                        frame = frame.numpy()
                    if frame.ndim == 3 and frame.shape[0] == 3:
                        frame = np.transpose(frame, (1, 2, 0))  # CHW -> HWC

                    # Resize if needed
                    if frame.shape[:2] != self.image_size:
                        import cv2
                        frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))

                    obs[f"pixels/{cam_key}"] = frame.astype(np.uint8)
                else:
                    obs[f"pixels/{cam_key}"] = np.zeros(
                        (self.image_size[0], self.image_size[1], 3), dtype=np.uint8
                    )
            except Exception as e:
                logger.debug(f"[NextisRobotEnv] Camera {cam_key} read error: {e}")
                obs[f"pixels/{cam_key}"] = np.zeros(
                    (self.image_size[0], self.image_size[1], 3), dtype=np.uint8
                )

        return obs

    def reset(self, seed=None, options=None):
        """Reset environment: smooth homing to reset position.

        Performs a smooth trajectory from current position to reset_position
        over reset_time_s seconds.
        """
        super().reset(seed=seed)
        self._step_count = 0

        if self.reset_position:
            self._smooth_reset()

        # Wait for settling
        time.sleep(0.5)

        obs = self._get_observation()
        return obs, {}

    def _smooth_reset(self):
        """Smooth trajectory to reset position."""
        num_steps = max(int(self.reset_time_s * self.fps), 10)

        # Get current positions
        if self.robot_lock:
            with self.robot_lock:
                current_obs = self.robot.get_observation()
        else:
            current_obs = self.robot.get_observation()

        # Build current and target position arrays
        current_pos = {}
        target_pos = {}
        for name in self.motor_names:
            pos_key = f"{name}.pos" if f"{name}.pos" in current_obs else name
            if pos_key in current_obs:
                current_pos[name] = current_obs[pos_key]
            if name in self.reset_position:
                target_pos[name] = self.reset_position[name]

        # Interpolate smoothly
        for step_i in range(num_steps):
            t = (step_i + 1) / num_steps
            # Smooth interpolation (cosine easing)
            t_smooth = 0.5 * (1 - np.cos(np.pi * t))

            action_dict = {}
            for name in self.motor_names:
                if name in current_pos and name in target_pos:
                    pos = current_pos[name] + (target_pos[name] - current_pos[name]) * t_smooth
                    # Strip arm prefix if needed for motor bus
                    motor_key = name
                    action_dict[motor_key] = pos

            if action_dict:
                if self.robot_lock:
                    with self.robot_lock:
                        self.robot.send_action(action_dict)
                else:
                    self.robot.send_action(action_dict)

            time.sleep(1.0 / self.fps)

    def step(self, action: np.ndarray):
        """Execute action on the robot.

        Args:
            action: Joint position targets, normalized to [-1, 1], shape (n_joints,)

        Returns:
            observation, reward, terminated, truncated, info
        """
        self._step_count += 1

        # Denormalize and send action
        action_dict = {}
        for i, name in enumerate(self.motor_names):
            denorm_pos = self._denormalize_position(name, float(action[i]))
            action_dict[name] = denorm_pos

        if self.robot_lock:
            with self.robot_lock:
                self.robot.send_action(action_dict)
        else:
            self.robot.send_action(action_dict)

        # Wait for control period
        time.sleep(1.0 / self.fps)

        # Get observation
        obs = self._get_observation()

        # Reward is computed externally by the reward classifier
        reward = 0.0

        # Check termination
        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "step": self._step_count,
            "action_dict": action_dict,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Return current camera frame for visualization."""
        if self.camera_keys:
            obs = self._get_observation()
            first_cam = f"pixels/{self.camera_keys[0]}"
            if first_cam in obs:
                return obs[first_cam]
        return np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

    def close(self):
        """Clean up resources."""
        logger.info("[NextisRobotEnv] Environment closed")

    def get_motor_names(self) -> list:
        """Return list of motor names being controlled."""
        return self.motor_names.copy()

    def get_current_joint_positions(self) -> np.ndarray:
        """Get current joint positions (normalized)."""
        obs = self._get_observation()
        return obs["agent_pos"]
