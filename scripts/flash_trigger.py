#!/usr/bin/env python3
"""Flash the RP2040 trigger firmware (main.py) to a Raspberry Pi Pico.

Usage:
    1. Hold the BOOTSEL button on the RP2040 and plug in USB (or press reset).
    2. The device appears as a USB mass storage drive (RPI-RP2).
    3. Run this script — it installs MicroPython if needed, then copies main.py.

    python scripts/flash_trigger.py

If MicroPython is already installed, you can skip BOOTSEL and just run:

    python scripts/flash_trigger.py --firmware-only
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
import urllib.request

MICROPYTHON_UF2_URL = (
    "https://micropython.org/resources/firmware/"
    "RPI_PICO-20241025-v1.24.0.uf2"
)
FIRMWARE_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "firmware", "rp2040_trigger", "main.py",
)


def find_bootsel_mount():
    """Find the RPI-RP2 mass storage mount point."""
    candidates = [
        "/media/*/RPI-RP2",
        f"/media/{os.environ.get('USER', '*')}/RPI-RP2",
        "/run/media/*/RPI-RP2",
        "/mnt/RPI-RP2",
    ]
    for pattern in candidates:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def find_pico_tty():
    """Find the /dev/ttyACM* port for a Raspberry Pi Pico."""
    import serial.tools.list_ports
    for p in serial.tools.list_ports.comports():
        if p.vid == 0x2E8A:  # Raspberry Pi vendor ID
            return p.device
    return None


def install_micropython():
    """Download MicroPython UF2 and copy to BOOTSEL drive."""
    mount = find_bootsel_mount()
    if not mount:
        print("\nRP2040 not found in BOOTSEL mode.")
        print("To enter BOOTSEL mode:")
        print("  1. Hold the BOOTSEL button on the RP2040 board")
        print("  2. While holding, plug in USB (or press the reset button)")
        print("  3. Release BOOTSEL — a drive named 'RPI-RP2' should appear")
        print("\nWaiting for RPI-RP2 drive...")
        for _ in range(60):
            time.sleep(1)
            mount = find_bootsel_mount()
            if mount:
                break
        if not mount:
            print("Timed out waiting for BOOTSEL mode. Aborting.")
            sys.exit(1)

    print(f"Found RPI-RP2 at {mount}")

    uf2_path = os.path.join(mount, "micropython.uf2")
    cache_path = os.path.expanduser("~/.cache/nextis/micropython_pico.uf2")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        print(f"Using cached MicroPython UF2: {cache_path}")
    else:
        print(f"Downloading MicroPython UF2...")
        urllib.request.urlretrieve(MICROPYTHON_UF2_URL, cache_path)
        print(f"Downloaded to {cache_path}")

    print(f"Copying MicroPython UF2 to {mount}...")
    shutil.copy2(cache_path, uf2_path)
    print("MicroPython UF2 copied. Device will reboot into MicroPython.")

    # Wait for the device to reboot and re-enumerate as CDC serial
    print("Waiting for device to reboot...")
    time.sleep(5)

    tty = None
    for _ in range(20):
        tty = find_pico_tty()
        if tty:
            break
        time.sleep(1)

    if not tty:
        print("WARNING: Could not find Pico serial port after reboot.")
        print("Check `ls /dev/ttyACM*` and pass --port manually.")
        return None

    print(f"Device is back on {tty}")
    return tty


def install_firmware(port):
    """Copy main.py to the RP2040 via mpremote."""
    if not os.path.exists(FIRMWARE_SRC):
        print(f"ERROR: Firmware not found at {FIRMWARE_SRC}")
        sys.exit(1)

    print(f"\nInstalling trigger firmware to {port}...")
    print(f"  Source: {FIRMWARE_SRC}")

    # Try mpremote first
    mpremote = shutil.which("mpremote")
    if not mpremote:
        # Check conda bin
        conda_bin = os.path.join(sys.prefix, "bin", "mpremote")
        if os.path.exists(conda_bin):
            mpremote = conda_bin

    if mpremote:
        print(f"  Using mpremote: {mpremote}")
        result = subprocess.run(
            [mpremote, "connect", port, "fs", "cp", FIRMWARE_SRC, ":main.py"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  mpremote error: {result.stderr}")
            print("  Falling back to raw REPL method...")
        else:
            print("  Firmware copied successfully.")
            # Soft-reboot
            subprocess.run(
                [mpremote, "connect", port, "reset"],
                capture_output=True, timeout=10,
            )
            print("  Device rebooted. Trigger firmware is now active.")
            return True

    # Fallback: raw REPL via pyserial
    print("  Using raw REPL fallback...")
    import serial

    with open(FIRMWARE_SRC, "r") as f:
        firmware_content = f.read()

    ser = serial.Serial(port, 115200, timeout=2.0)
    time.sleep(1.0)

    # Interrupt + enter raw REPL
    ser.write(b"\x03\x03")
    time.sleep(0.5)
    ser.write(b"\x01")  # Ctrl+A = raw REPL
    time.sleep(0.5)
    ser.read(ser.in_waiting or 1)  # drain

    # Write the file via raw REPL
    # Use base64 to avoid quoting issues
    import base64
    b64 = base64.b64encode(firmware_content.encode()).decode()

    cmd = (
        f"import base64\n"
        f"data = base64.b64decode('{b64}')\n"
        f"with open('main.py', 'wb') as f:\n"
        f"    f.write(data)\n"
        f"print('OK:' + str(len(data)))\n"
    )
    ser.write(cmd.encode())
    ser.write(b"\x04")  # Ctrl+D = execute
    time.sleep(3.0)

    resp = ser.read(4096).decode("utf-8", errors="ignore")
    if "OK:" in resp:
        size = resp.split("OK:")[1].split("\r")[0].split("\n")[0]
        print(f"  Firmware written ({size} bytes)")
    else:
        print(f"  WARNING: Could not confirm write. Response: {resp[:200]!r}")

    # Soft reboot
    ser.write(b"\x02")  # Ctrl+B = exit raw REPL
    time.sleep(0.2)
    ser.write(b"\x04")  # Ctrl+D = soft reboot
    time.sleep(0.5)
    ser.close()

    print("  Device rebooted. Trigger firmware should be active.")
    return True


def verify_firmware(port):
    """Send INFO command and check for NEXTIS_TRIGGER response."""
    import serial

    print(f"\nVerifying trigger firmware on {port}...")
    time.sleep(2.0)  # Wait for firmware to boot

    ser = serial.Serial(port, 115200, timeout=2.0)
    time.sleep(1.0)
    ser.reset_input_buffer()
    ser.write(b"INFO\n")
    time.sleep(0.5)
    resp = ser.readline().decode("utf-8", errors="ignore").strip()
    ser.close()

    if resp.startswith("NEXTIS_TRIGGER:"):
        version = resp.split(":")[1] if ":" in resp else "unknown"
        print(f"  Trigger firmware verified (version={version})")
        return True
    else:
        print(f"  WARNING: Unexpected response: {resp!r}")
        print("  The firmware may not have installed correctly.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Flash RP2040 trigger firmware")
    parser.add_argument("--port", help="Serial port (auto-detected if omitted)")
    parser.add_argument(
        "--firmware-only", action="store_true",
        help="Skip MicroPython install (device already has MicroPython)",
    )
    args = parser.parse_args()

    print("=== Nextis RP2040 Trigger Firmware Flasher ===\n")

    port = args.port

    if not args.firmware_only:
        port = install_micropython()
        if not port:
            sys.exit(1)
    else:
        if not port:
            port = find_pico_tty()
            if not port:
                print("ERROR: No Raspberry Pi Pico found. Specify --port manually.")
                sys.exit(1)
        print(f"Using existing MicroPython on {port}")

    install_firmware(port)
    verify_firmware(port)

    print("\nDone! You can now start the trigger listener in the Nextis UI.")


if __name__ == "__main__":
    main()
