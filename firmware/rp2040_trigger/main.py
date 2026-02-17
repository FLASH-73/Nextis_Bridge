# RP2040 Zero Trigger Firmware — MicroPython
#
# Wiring:
#   GPIO 0 ──┤ momentary switch ├── GND
#   (internal pull-up enabled; active-low)
#
# Protocol (USB CDC serial, 115200 baud):
#   Device → Host:
#     PIN:<gpio>:1\n   switch pressed
#     PIN:<gpio>:0\n   switch released
#   Host → Device:
#     PING\n   → PONG\n
#     STATE\n  → PIN:<gpio>:<0|1>\n  (one line per pin)
#     INFO\n   → NEXTIS_TRIGGER:v1\n
#
# To add more pins, append entries to MONITORED_PINS below.

import sys
import select
import time
from machine import Pin

# --- Configuration -----------------------------------------------------------
# Each entry: (gpio_number, pull_mode, active_level)
# active_level=0 means the pin reads 0 when the switch is pressed (active-low).
MONITORED_PINS = [
    (0, Pin.PULL_UP, 0),
]

DEBOUNCE_MS = 20
LED_FLASH_MS = 50
POLL_INTERVAL_MS = 10  # 100 Hz

LED_PIN = 16  # Waveshare RP2040 Zero onboard LED

# --- Setup --------------------------------------------------------------------
led = Pin(LED_PIN, Pin.OUT, value=0)
led_off_at = 0  # ticks_ms when LED should turn off (0 = already off)

pins = []  # list of (gpio, pin_obj, active_level, last_stable, last_change_ms)
for gpio, pull, active in MONITORED_PINS:
    p = Pin(gpio, Pin.IN, pull)
    logical = 1 if p.value() == active else 0
    pins.append([gpio, p, active, logical, time.ticks_ms()])

poller = select.poll()
poller.register(sys.stdin, select.POLLIN)
cmd_buf = ""

def send(msg):
    sys.stdout.write(msg)

# --- Main loop ----------------------------------------------------------------
while True:
    now = time.ticks_ms()

    # Turn off LED after flash duration
    if led_off_at and time.ticks_diff(now, led_off_at) >= 0:
        led.value(0)
        led_off_at = 0

    # Debounce and report pin changes
    for entry in pins:
        gpio, pin, active, last_stable, last_change = entry
        raw = pin.value()
        logical = 1 if raw == active else 0
        if logical != last_stable and time.ticks_diff(now, last_change) >= DEBOUNCE_MS:
            entry[3] = logical
            entry[4] = now
            send("PIN:{}:{}\n".format(gpio, logical))
            led.value(1)
            led_off_at = time.ticks_add(now, LED_FLASH_MS)

    # Process serial commands (non-blocking)
    while poller.poll(0):
        ch = sys.stdin.read(1)
        if ch is None:
            break
        if ch in ("\n", "\r"):
            cmd = cmd_buf.strip().upper()
            cmd_buf = ""
            if cmd == "PING":
                send("PONG\n")
            elif cmd == "STATE":
                for entry in pins:
                    send("PIN:{}:{}\n".format(entry[0], entry[3]))
            elif cmd == "INFO":
                send("NEXTIS_TRIGGER:v1\n")
        else:
            cmd_buf += ch

    time.sleep_ms(POLL_INTERVAL_MS)
