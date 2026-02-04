from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation


@RobotConfig.register_subclass("umbra_follower")
@dataclass
class UmbraFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str = "/dev/ttyUSB1"
    arm_side: str = "left" 
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False
    camera_interface: str = "intelrealsense"

    if camera_interface == "intelrealsense":
        # Troubleshooting: If one of your IntelRealSense cameras freeze during
        # data recording due to bandwidth limit, you might need to plug the camera
        # on another USB hub or PCIe card.
        cameras: dict[str, CameraConfig] = field(
            default_factory=lambda: {
                "camera1": RealSenseCameraConfig(
                    serial_number_or_name= "336222072963",
                    fps=30,
                    width=640,
                    height=480,
                    color_mode=ColorMode.RGB,
                    use_depth=True,
                ),
                "camera2": RealSenseCameraConfig(
                    serial_number_or_name="218622275492",
                    fps=30,
                    width=848,
                    height=480,
                    color_mode=ColorMode.RGB,
                    use_depth=True, 
                ),
            }
        )
'''
@CameraConfig.register_subclass("intelrealsense")
@dataclass
class UmbraWristCameraConfig(RealSenseCameraConfig):
    serial_number_or_name: str = "218622275492",
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    color_mode: ColorMode = ColorMode.RGB,
    use_depth: bool = True,
    '''