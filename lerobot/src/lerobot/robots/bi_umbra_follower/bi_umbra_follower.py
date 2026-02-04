#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.umbra_follower import UmbraFollowerRobot
from lerobot.robots.umbra_follower.config_umbra_follower import UmbraFollowerConfig

from ..robot import Robot
from .config_bi_umbra_follower import BiUmbraFollowerConfig

logger = logging.getLogger(__name__)


class BiUmbraFollower(Robot):
    """Robot class for BiUmbra Follower arm."""
    config_class = BiUmbraFollowerConfig
    name = "bi_umbra_follower"  

    def __init__(self, config: BiUmbraFollowerConfig):
        super().__init__(config)
        self.config = config
        
        left_arm_config = UmbraFollowerConfig(
            id="left_follower",  # Align with CalibrationService ID
            calibration_dir=config.calibration_dir,
            arm_side="left",
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
        )
        right_arm_config = UmbraFollowerConfig(
            id="right_follower",  # Align with CalibrationService ID
            calibration_dir=config.calibration_dir,
            arm_side="right",
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            cameras={},
        )
        
        self.left_arm = UmbraFollowerRobot(left_arm_config)
        self.right_arm = UmbraFollowerRobot(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)
        
    @property
    def _motors_ft(self) -> dict[str, type]:
        # FIX: Filter out '_follower' motors to match the single arm's send_action logic
        return {
            f"left_{motor}.pos": float 
            for motor in self.left_arm.bus.motors 
            if not motor.endswith("_follower")
        } | {
            f"right_{motor}.pos": float 
            for motor in self.right_arm.bus.motors 
            if not motor.endswith("_follower")
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        cam_ft = {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }
        # Add depth features for cameras with depth enabled
        for cam in self.cameras:
            if hasattr(self.config.cameras[cam], 'use_depth') and self.config.cameras[cam].use_depth:
                cam_ft[f"{cam}_depth"] = (self.config.cameras[cam].height, self.config.cameras[cam].width, 1)
        return {**self._motors_ft, **cam_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = False) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        for cam in self.cameras.values():
            cam.connect()
            # Delay between camera connections to reduce USB bandwidth congestion
            # This gives each camera time to stabilize before the next one starts
            time.sleep(0.5)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_observation(self, include_images: bool = True) -> dict[str, Any]:
        obs_dict = {}

        # Add "left_" prefix
        left_obs = self.left_arm.get_observation(include_images=include_images)
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # Add "right_" prefix
        right_obs = self.right_arm.get_observation(include_images=include_images)
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        # DEBUG
        # print(f"DEBUG: Left obs keys: {list(left_obs.keys())}")
        # print(f"DEBUG: Right obs keys: {list(right_obs.keys())}")
        # print(f"DEBUG: Cameras: {list(self.cameras.keys())}")

        if include_images:
            for cam_key, cam in self.cameras.items():
                # start = time.perf_counter()
                # USE ZOH (blocking=False) to ensure control loop runs at 60Hz even if cameras are 30Hz
                obs_dict[cam_key] = cam.async_read(blocking=False)
                # Capture depth if enabled for this camera
                if hasattr(cam.config, 'use_depth') and cam.config.use_depth:
                    try:
                        obs_dict[f"{cam_key}_depth"] = cam.async_read_depth(blocking=False)
                    except Exception as e:
                        logger.debug(f"Depth read failed for {cam_key}: {e}")
                # dt_ms = (time.perf_counter() - start) * 1e3
                # logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict
    
    def reload_inversions(self, new_inversions: dict | None = None):
        """Reloads inversions for both arms. Argument is ignored for BiUmbra as we load from disk per-arm."""
        # We ignore new_inversions dict here because it would be ambiguous which arm it applies to.
        # Instead, we force reload from disk for both.
        self.left_arm.reload_inversions()
        self.right_arm.reload_inversions()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        # Add prefixes back
        prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def disconnect(self):
        # Airtight Disconnect: Attempt to close everything even if one fails
        
        # Left Arm
        try:
            self.left_arm.disconnect()
        except Exception as e:
            logger.warning(f"Failed to disconnect left arm: {e}")
            
        # Right Arm
        try:
            self.right_arm.disconnect()
        except Exception as e:
            logger.warning(f"Failed to disconnect right arm: {e}")

        # Cameras
        for cam_key, cam in self.cameras.items():
            try:
                cam.disconnect()
            except Exception as e:
                logger.warning(f"Failed to disconnect camera {cam_key}: {e}")
                
        logger.info(f"{self} disconnected.")
