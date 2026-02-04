import time
import logging
from lerobot.robots import Robot
from lerobot.robots.umbra_follower.umbra_follower import UmbraFollowerRobot
from lerobot.robots.umbra_follower.config_umbra_follower import UmbraFollowerConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Config - adjust port/id as needed
config = UmbraFollowerConfig(
    port="/dev/ttyUSB0",
    id="umbra_follower_1",  # Your ID
    use_degrees=False,
    disable_torque_on_disconnect=True,
    cameras={},  # No cameras needed for test
)

robot = UmbraFollowerRobot(config)

try:
    robot.connect(calibrate=False)  # Skip calib if file exists
    ping_results = {motor: robot.bus.ping(robot.bus.motors[motor].id) for motor in robot.bus.motors}
    print("Ping results (True=responsive):", ping_results)
    # Print initial state
    obs = robot.get_observation()
    print("Initial positions:", obs)

    # Command gripper (single joint) to new position (e.g., from ~100 to 50)
    action = {"gripper.pos": 50.0}
    sent = robot.send_action(action)
    print("Sent action:", sent)
    
    time.sleep(2)  # Wait for movement
    
    # Read new state
    obs = robot.get_observation()
    print("New positions:", obs)
    
    # Command base to small change (e.g., +10)
    action = {"base.pos": obs["base.pos"] + 10.0}
    sent = robot.send_action(action)
    print("Sent base action:", sent)
    
    time.sleep(2)
    
    obs = robot.get_observation()
    print("Final positions:", obs)

except Exception as e:
    print(f"Error: {e}")
finally:
    robot.disconnect()