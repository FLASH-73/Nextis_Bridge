"""Hardware connection factory.

Creates the appropriate robot/teleoperator instance based on motor type and role.
"""

import logging
from typing import Any, Optional

from app.core.config import CALIBRATION_DIR
from app.core.hardware.types import ArmDefinition, MotorType, ArmRole

logger = logging.getLogger(__name__)


def create_arm_instance(arm: ArmDefinition) -> Optional[Any]:
    """
    Factory function: create the appropriate robot/teleoperator instance.

    Previously ArmRegistryService._create_arm_instance().
    Import-heavy by design (deferred imports avoid circular deps).
    """
    if arm.motor_type == MotorType.STS3215:
        if arm.role == ArmRole.FOLLOWER:
            from lerobot.robots.umbra_follower import UmbraFollowerRobot
            from lerobot.robots.umbra_follower.config_umbra_follower import UmbraFollowerConfig
            config = UmbraFollowerConfig(
                id=arm.id,
                port=arm.port,
                cameras={},  # No cameras for individual arms
                calibration_dir=CALIBRATION_DIR / arm.id,
            )
            robot = UmbraFollowerRobot(config)
            robot.connect(calibrate=False)
            return robot
        else:
            # Leader arm - use LeaderArm class
            from lerobot.teleoperators.umbra_leader import UmbraLeader
            from lerobot.teleoperators.umbra_leader.config_umbra_leader import UmbraLeaderConfig
            config = UmbraLeaderConfig(
                id=arm.id,
                port=arm.port,
                calibration_dir=CALIBRATION_DIR / arm.id,
            )
            leader = UmbraLeader(config)
            leader.connect(calibrate=False)
            return leader

    elif arm.motor_type == MotorType.DAMIAO:
        if arm.role == ArmRole.FOLLOWER:
            from lerobot.robots.damiao_follower import DamiaoFollowerRobot
            from lerobot.robots.damiao_follower.config_damiao_follower import DamiaoFollowerConfig
            config = DamiaoFollowerConfig(
                id=arm.id,
                port=arm.port,
                velocity_limit=arm.config.get("velocity_limit", 0.3),
                cameras={},
                calibration_dir=CALIBRATION_DIR / arm.id,
            )
            robot = DamiaoFollowerRobot(config)
            robot.connect()
            return robot
        else:
            # Damiao leader not yet implemented
            logger.warning(f"Damiao leader arms not yet supported")
            return None

    elif arm.motor_type in [MotorType.DYNAMIXEL_XL330, MotorType.DYNAMIXEL_XL430]:
        if arm.role == ArmRole.LEADER:
            # Dynamixel XL330 leader arm (Waveshare USB-C bus)
            from lerobot.teleoperators.dynamixel_leader import DynamixelLeader
            from lerobot.teleoperators.dynamixel_leader.config_dynamixel_leader import DynamixelLeaderConfig
            config = DynamixelLeaderConfig(
                id=arm.id,
                port=arm.port,
                structural_design=arm.structural_design or "",
                calibration_dir=CALIBRATION_DIR / arm.id,
            )
            leader = DynamixelLeader(config)
            leader.connect(calibrate=False)
            return leader
        else:
            logger.warning(f"Dynamixel follower arms not typical use case")
            return None

    logger.warning(f"Unknown motor type: {arm.motor_type}")
    return None
