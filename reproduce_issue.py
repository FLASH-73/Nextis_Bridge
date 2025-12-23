
import sys
import pathlib
from unittest.mock import MagicMock, patch

# Add paths
sys.path.append('/home/roberto/nextis_app')
sys.path.append('/home/roberto/nextis_app/lerobot/src')

from app.core.calibration_service import CalibrationService
from lerobot.robots.bi_umbra_follower.bi_umbra_follower import BiUmbraFollower
from lerobot.motors import MotorCalibration


def test_logic():
    print("--- Starting Reproduction Test ---")
    
    # Mock Robot Structure
    # BiUmbraFollower -> left_arm (UmbraFollowerRobot) -> bus -> motors
    
    left_arm = MagicMock()
    left_arm.calibration = {}
    left_arm.bus.motors = {
        "base": MagicMock(id=1),
        "link1": MagicMock(id=2),
        "link1_follower": MagicMock(id=3),
        "link2": MagicMock(id=4),
        "link2_follower": MagicMock(id=5),
        "link3": MagicMock(id=6),
        "link4": MagicMock(id=7),
        "link5": MagicMock(id=8),
        "gripper": MagicMock(id=9)
    }
    
    # Mock read/write
    left_arm.bus.sync_read.return_value = {
        "base": 2048, "link1": 2048, "link1_follower": 2048,
        "link2": 2048, "link2_follower": 2048, "link3": 2048,
        "link4": 2048, "link5": 2048, "gripper": 2048
    }
    
    # Populate initial calibration with defaults (as load_calibration would, or empty)
    # If empty, get_arms should be False.
    
    # Create a dummy object and assign the class to pass isinstance checks
    class DummyBiUmbra:
        pass
    
    robot = DummyBiUmbra()
    robot.__class__ = BiUmbraFollower # Force class to match what CalibrationService expects
    
    robot.left_arm = left_arm
    robot.right_arm = MagicMock()
    
    # We need to manually set attributes that BiUmbraFollower might have if accessed
    # But CalibrationService only checks isinstance and access .left_arm/right_arm
    
    cs = CalibrationService(robot)
        
    # 1. Check Initial Status
    print("1. Checking Initial Status (expecting False)...")
    arms = cs.get_arms()
    left_follower = next(a for a in arms if a["id"] == "left_follower")
    print(f"   Left Follower Calibrated: {left_follower['calibrated']}")
    assert not left_follower['calibrated']

        
    # 2. Simulate Discovery
    print("\n2. Simulating start_discovery...")
    cs.start_discovery("left_follower")
    assert cs.is_discovering
    
    # Simulate moving to min/max
    print("   Simulating movements...")
    # Move to 1000
    left_arm.bus.sync_read.return_value = {k: 1000 for k in left_arm.bus.motors}
    cs.get_calibration_state("left_follower")
    
    # Move to 3000
    left_arm.bus.sync_read.return_value = {k: 3000 for k in left_arm.bus.motors}
    cs.get_calibration_state("left_follower")
    
    print(f"   Session Ranges: {cs.session_ranges['base']}")
    
    # 3. Stop Discovery
    print("\n3. Simulating stop_discovery...")
    
    # CalibrationService.stop_discovery updates arm.calibration
    # Ensure arm.calibration has entries to receive updates!
    # If arm.calibration is empty, stop_discovery loop `if motor in arm.calibration` FAILS.
    # THIS IS LIKELY THE BUG!
    # Start Discovery should probably Initialize arm.calibration if empty?
    # Or Populate it?
    
    # Mock pre-existing calibration entries (usually created by Homing or Load)
    if not left_arm.calibration:
        print("   (Simulating pre-population of calibration keys as expected after Homing)")
        for m_name, m_obj in left_arm.bus.motors.items():
            left_arm.calibration[m_name] = MotorCalibration(
                id=m_obj.id, drive_mode=0, homing_offset=0, range_min=0, range_max=0
            )
    
    cs.stop_discovery("left_follower")
    
    print("\n4. Checking Results after Stop...")
    # Check if 'base' is in calibration
    if "base" in left_arm.calibration:
        cal_base = left_arm.calibration["base"]
        print(f"   Base min: {cal_base.range_min} (Expected 1000)")
        print(f"   Base max: {cal_base.range_max} (Expected 3000)")
        
        if cal_base.range_min == 1000 and cal_base.range_max == 3000:
            print("   SUCCESS: Ranges updated.")
        else:
            print("   FAILURE: Ranges NOT updated.")
    else:
        print("   FAILURE: 'base' not found in calibration.")
        
    # 4. Check Status Again
    print("\n5. Checking Status (expecting True)...")
    arms = cs.get_arms()
    left_follower = next(a for a in arms if a["id"] == "left_follower")
    print(f"   Left Follower Calibrated: {left_follower['calibrated']}")
    
    if left_follower['calibrated']:
            print("   SUCCESS: Status is Calibrated.")
    else:
            print("   FAILURE: Status is Uncalibrated.")


if __name__ == "__main__":
    try:
        test_logic()
    except Exception as e:
        import traceback
        traceback.print_exc()
