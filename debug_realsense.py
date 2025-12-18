import pyrealsense2 as rs

def list_devices():
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"Found {len(devices)} RealSense devices.")
    for dev in devices:
        print(f"  Device: {dev.get_info(rs.camera_info.name)}")
        print(f"  Serial: {dev.get_info(rs.camera_info.serial_number)}")
        print(f"  Firmware: {dev.get_info(rs.camera_info.firmware_version)}")

if __name__ == "__main__":
    try:
        list_devices()
    except Exception as e:
        print(f"Error: {e}")
