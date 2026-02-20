"""HIL loop mixin: main control loop, leader velocity, policy inference, and action conversion."""

import time
import numpy as np
import logging

from .observation import HILObservationMixin
from .types import HILMode

logger = logging.getLogger(__name__)


class HILLoopMixin(HILObservationMixin):
    """Mixin providing the HIL control loop and inference methods."""

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

                # 3. Safety check (every 3rd frame = ~10Hz at 30Hz loop)
                self._safety_frame_count = getattr(self, '_safety_frame_count', 0) + 1
                if self._safety_frame_count % 3 == 0 and hasattr(self.teleop, 'safety'):
                    robot = self.teleop.robot if hasattr(self.teleop, 'robot') else None
                    if robot and hasattr(robot, 'is_connected') and robot.is_connected:
                        if not self.teleop.safety.check_all_limits(robot):
                            logger.error("[HIL] SAFETY: Limit exceeded — stopping")
                            print("[HIL] SAFETY: Limit exceeded — EMERGENCY STOP", flush=True)
                            self._stop_event.set()
                            break

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
