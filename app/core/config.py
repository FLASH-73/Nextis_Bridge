import yaml
from pathlib import Path

CONFIG_PATH = Path("app/config/settings.yaml")

def load_config():
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def save_config(config_data):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config_data, f)

def get_robot_config():
    config = load_config()
    return config.get("robot", {})

def get_teleop_config():
    config = load_config()
    return config.get("teleop", {})

def get_camera_config():
    config = load_config()
    robot_config = config.get("robot", {})
    return robot_config.get("cameras", {})
