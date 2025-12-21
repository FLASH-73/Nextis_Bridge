import sys
from pathlib import Path
import traceback

# Add paths
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "lerobot" / "src"))

from app.core.config import load_config
from lerobot.robots.utils import make_robot_from_config

def verify_init():
    print("Loading Config...")
    config_data = load_config()
    robot_cfg = config_data.get("robot", {})
    print(f"Robot Config: {robot_cfg}")
    
    try:
        print(f"Attempting to connect to robot: {robot_cfg.get('type')}...")
        
        # Import Config Classes dynamically
        from lerobot.robots.bi_umbra_follower.config_bi_umbra_follower import BiUmbraFollowerConfig
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.cameras.realsense import RealSenseCameraConfig
        
        # Construct Camera Configs
        cameras = {}
        for name, cam_data in robot_cfg.get("cameras", {}).items():
            print(f"Configuring camera: {name} ({cam_data['type']})")
            if cam_data["type"] == "opencv":
                cameras[name] = OpenCVCameraConfig(
                    fps=cam_data["fps"],
                    width=cam_data["width"],
                    height=cam_data["height"],
                    index_or_path=cam_data.get("index_or_path")
                )
            elif cam_data["type"] == "intelrealsense":
                cameras[name] = RealSenseCameraConfig(
                    fps=cam_data["fps"],
                    width=cam_data["width"],
                    height=cam_data["height"],
                    serial_number_or_name=cam_data.get("serial_number_or_name")
                )
        
        # Construct Robot Config
        if robot_cfg.get("type") == "bi_umbra_follower":
            r_config = BiUmbraFollowerConfig(
                left_arm_port=robot_cfg["left_arm_port"],
                right_arm_port=robot_cfg["right_arm_port"],
                cameras=cameras
            )
            
            print("Creating Robot...")
            robot = make_robot_from_config(r_config)
            print("Connecting to Robot...")
            robot.connect(calibrate=False)
            print("✅ Real Robot Connected Successfully!")
            
            print(f"Connected Arms: {robot.available_arms}")
            
        else:
            print(f"Unknown robot type: {robot_cfg.get('type')}")

    except Exception as e:
        print(f"❌ Failed to connect robot: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    verify_init()
