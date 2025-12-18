import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock

# Add lerobot/src to sys.path
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path / "lerobot" / "src"))

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.feetech.tables import MODEL_ENCODING_TABLE

class TestFeetechSigned(unittest.TestCase):
    def test_signed_encoding(self):
        # Create a mock bus
        # We don't need a real port, just need to access the _encode_sign and _decode_sign methods
        # But __init__ tries to open port.
        # We can subclass and override __init__ or just mock the class methods if they were static, but they are instance methods using self.model_encoding_table
        
        # Let's mock the __init__ to do nothing
        original_init = FeetechMotorsBus.__init__
        FeetechMotorsBus.__init__ = lambda self: None
        
        bus = FeetechMotorsBus()
        # Manually set the table
        bus.model_encoding_table = MODEL_ENCODING_TABLE
        # Mock motors dict to return a model that uses the table
        # "sts3215" uses STS_SMS_SERIES_ENCODINGS_TABLE
        mock_motor = MagicMock()
        mock_motor.model = "sts3215"
        bus.motors = {"motor1": mock_motor}
        bus.ids = [1]
        
        # Helper to mock _id_to_model
        bus._id_to_model = lambda id_: "sts3215"
        
        print("Testing Present_Position decoding...")
        # 65535 (0xFFFF) should be -1
        # 65536 is wrap around, but 16-bit is max 65535.
        # In 2's complement 16-bit:
        # 0 -> 0
        # 32767 -> 32767
        # 32768 -> -32768
        # 65535 -> -1
        
        # Test Decoding
        ids_values = {1: 65535}
        decoded = bus._decode_sign("Present_Position", ids_values)
        print(f"Decoded 65535 as: {decoded[1]}")
        self.assertEqual(decoded[1], -1)
        
        ids_values = {1: 0}
        decoded = bus._decode_sign("Present_Position", ids_values)
        self.assertEqual(decoded[1], 0)
        
        ids_values = {1: 32767}
        decoded = bus._decode_sign("Present_Position", ids_values)
        self.assertEqual(decoded[1], 32767)
        
        # Test Encoding
        print("Testing Goal_Position encoding...")
        ids_values = {1: -1}
        encoded = bus._encode_sign("Goal_Position", ids_values)
        print(f"Encoded -1 as: {encoded[1]}")
        self.assertEqual(encoded[1], 65535)
        
        ids_values = {1: -100}
        encoded = bus._encode_sign("Goal_Position", ids_values)
        # -100 = 0xFF9C = 65436
        print(f"Encoded -100 as: {encoded[1]}")
        self.assertEqual(encoded[1], 65436)

        print("✅ Signed Integer Verification Passed!")

if __name__ == "__main__":
    unittest.main()
