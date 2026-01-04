import time
try:
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera, RealSenseCameraConfig
    print("Imported LeRobot RealSense classes successfully.")
except ImportError as e:
    print(f"Failed to import LeRobot: {e}")
    exit(1)

def test_lerobot_class():
    serial = "218622275492" # From settings.yaml
    width = 848
    height = 480
    fps = 30
    
    print(f"Testing RealSenseCamera with: {width}x{height} @ {fps}fps, Serial: {serial}")
    
    conf = RealSenseCameraConfig(
        serial_number_or_name=serial,
        fps=fps,
        width=width,
        height=height
    )
    
    cam = RealSenseCamera(conf)
    print("Instantiated Camera object.")
    
    try:
        print("Connecting...")
        cam.connect()
        print("Connected!")
        
        for i in range(10):
            frame = cam.read()
            if frame is not None:
                # Expecting (H, W, C) or (C, H, W)? LeRobot usually does (C, H, W) pytorch style? 
                # Or numpy (H, W, C)?
                shape = frame.shape if hasattr(frame, 'shape') else "Unknown"
                print(f"Frame {i}: OK, Shape: {shape}")
            else:
                print(f"Frame {i}: None")
                
            time.sleep(0.1)
            
        cam.disconnect()
        print("Disconnected.")
        
    except Exception as e:
        print(f"LeRobot Camera Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lerobot_class()

