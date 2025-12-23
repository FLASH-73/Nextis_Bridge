
import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append("/home/roberto/nextis_app")

from app.core.calibration_service import CalibrationService

class TestCalibrationServiceFix(unittest.TestCase):
    def test_list_files_no_crash(self):
        # Mock robot dependencies
        mock_robot = MagicMock()
        service = CalibrationService(mock_robot)
        
        # Simulate MISSING active_profiles (as if instance was old)
        if hasattr(service, "active_profiles"):
            del service.active_profiles
            
        print("Checking if active_profiles is missing:", not hasattr(service, "active_profiles"))
        
        # Call list_calibration_files - expected to handle the missing attr gracefully
        # We need _get_arm_context to return something valid-ish or handle return
        # But _get_arm_context relies on robot type checks.
        # Let's mock _get_arm_context to return None, [] to simulate "arm not found" 
        # (which is the safe path if robot is disconnected/mocked improperly)
        # OR better, mock specific robot type.
        
        service._get_arm_context = MagicMock(return_value=(MagicMock(), []))
        
        try:
            files = service.list_calibration_files("left_leader")
            print("Successfully listed files (mocked):", files)
        except AttributeError as e:
            self.fail(f"Raised AttributeError: {e}")
        except Exception as e:
            self.fail(f"Raised Exception: {e}")
            
    def test_malformed_id(self):
        mock_robot = MagicMock()
        service = CalibrationService(mock_robot)
        
        # Should not crash
        try:
            service.list_calibration_files("testarm") # No underscore
            print("Successfully handled malformed ID")
        except IndexError:
            self.fail("Raised IndexError on malformed ID")
            
if __name__ == "__main__":
    unittest.main()
