import yaml
from pathlib import Path

# Project paths (derived from this file's location: app/core/config.py)
PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH     = PROJECT_ROOT / "app" / "config" / "settings.yaml"
CALIBRATION_DIR = PROJECT_ROOT / "calibration_profiles"
GRAVITY_DIR     = PROJECT_ROOT / "calibration_gravity"
DATASETS_DIR    = PROJECT_ROOT / "datasets"
POLICIES_DIR    = PROJECT_ROOT / "training" / "outputs"
MODELS_DIR      = PROJECT_ROOT / "models"
RAW_DATA_DIR    = PROJECT_ROOT / "data" / "raw"
LEROBOT_SRC     = PROJECT_ROOT / "lerobot" / "src"

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
