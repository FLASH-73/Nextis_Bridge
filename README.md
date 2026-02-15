# Nextis Bridge

Hardware control, teleoperation, and training platform for Nextis robotic arms.

## What It Does

Nextis Bridge is an operator workstation for robotic manipulation research. It manages the
full workflow from hardware bring-up to trained policy: connect and configure arms (mixed
motor types on CAN and serial buses), calibrate joints interactively, teleoperate with
force feedback at 60 Hz, record demonstration episodes with synchronized multi-camera
video, train manipulation policies from those demonstrations, then deploy policies and
improve them online with human-in-the-loop intervention capture. Everything is accessed
through a web dashboard — no terminal required for day-to-day operation.

## Quick Start

```bash
git clone https://github.com/FLASH-73/Nextis_Bridge.git && cd Nextis_Bridge

# Python environment
conda create -n nextis python=3.11 -y && conda activate nextis
pip install -r requirements.txt
pip install -e lerobot

# CAN bus setup (required for Damiao arms — run once per boot)
sudo ./setup_can.sh

# Start backend
python run_backend.py                      # FastAPI on http://localhost:8000

# Start frontend (separate terminal)
cd frontend && npm install && npm run dev  # Next.js on http://localhost:3000
```

Open `http://localhost:3000` to access the dashboard. Hardware is optional — the
dashboard, dataset tools, and training pipeline all work without connected arms.

## What You Can Do

- **Multi-arm management** — Damiao (CAN bus), Dynamixel (USB serial), and Feetech (USB serial) motors in the same session. Hot-plug connect/disconnect from the UI.
- **Joint calibration** — Interactive range discovery, homing procedures, motor inversion tracking, and persistent calibration profiles.
- **Dual-arm teleoperation at 60 Hz** — Leader-to-follower mapping with MIT impedance control. Each leader-follower pair gets isolated state (`PairingContext`) so mixed motor types don't interfere.
- **Force feedback** — Follower gripper torque is fed back to the leader via EMA filtering. Joint-level force feedback uses a virtual spring in current-position mode.
- **Episode recording** — Parallel capture thread records joint states + camera frames into LeRobot v3 format (HDF5 + MP4 video). Start/stop/discard episodes from the UI.
- **Dataset management** — Browse recorded datasets, review individual episodes with video playback, merge compatible datasets.
- **Policy training** — ACT, Diffusion Policy, SmolVLA, and PI0.5 with configurable presets (quick/standard/full). Dataset validation, GPU detection, real-time log streaming, and checkpoint management — all from the browser.
- **Human-in-the-loop fine-tuning** — During autonomous policy execution, velocity-based intervention detection captures when the human takes over. The corrective trajectory becomes training data for online policy improvement (HIL-SERL).
- **RL training** — SAC-based actor-learner architecture with dual replay buffers. Reward sources: SARM (learned from demos), GVL (Gemini vision), or trained classifiers.
- **Safety monitoring** — Per-motor torque limits with debounced violation detection, joint bounds, and emergency stop that broadcasts to all buses.
- **Camera management** — Auto-discovery of USB and RealSense cameras, independent connect/disconnect, MJPEG streaming at 60 FPS.
- **LLM task planning** — Chat interface for task decomposition via Gemini or local Qwen2.5.

## Architecture

```
app/
├── core/
│   ├── hardware/       # Arm registry, motor control (Damiao + Dynamixel + Feetech), safety
│   ├── teleop/         # 60 Hz control loop, force feedback, multi-pair mapping, recording
│   ├── calibration/    # Joint calibration, homing, range discovery, profile persistence
│   ├── cameras/        # Camera lifecycle, MJPEG streaming, device discovery
│   ├── dataset/        # Episode browsing, dataset merge (LeRobot v3 format)
│   ├── training/       # Job management, policy presets, GPU detection, log streaming
│   ├── hil/            # Human-in-the-loop (DAgger-style intervention capture)
│   ├── rl/             # SAC policy, Gym environment, reward classifiers (SARM, GVL)
│   ├── intervention.py # Velocity-based human override detection
│   └── orchestrator.py # Policy deployment and task routing
├── routes/             # 14 FastAPI route modules — one file per domain
├── state.py            # Service initialization singleton (no hardware needed at startup)
└── config/             # YAML-based arm definitions, pairings, camera config
frontend/               # Next.js + React + TypeScript + Tailwind CSS
scripts/                # Motor configuration, encoder calibration, diagnostics
lerobot/                # Vendored LeRobot fork (policy training + motor drivers)
```

All services initialize without hardware. The arm registry, calibration profiles, cameras,
datasets, and training pipeline are independent subsystems that can be used in isolation.

### API Surface

The backend exposes 14 route groups via FastAPI:

| Prefix | Purpose |
|--------|---------|
| `/status`, `/config`, `/system/*` | System status, restart, emergency stop |
| `/arms/*` | Arm CRUD, connect/disconnect, motor scanning, pairings |
| `/motors/*` | Motor diagnostics, ID configuration, recovery |
| `/calibration/*` | Joint calibration, homing, gravity compensation, profiles |
| `/teleop/*` | Start/stop teleoperation, force feedback tuning |
| `/cameras/*` | Camera connect/disconnect, MJPEG streaming, scanning |
| `/recording/*` | Session/episode lifecycle, recording options |
| `/datasets/*` | Dataset browsing, episode review, merge, video streaming |
| `/training/*` | Job management, validation, presets, hardware detection |
| `/policies/*` | Policy listing, deployment, deletion, resume training |
| `/hil/*` | HIL session management, intervention episodes, retraining |
| `/rl/*` | RL training, SARM/GVL reward services, classifiers |
| `/chat` | LLM task planning |
| `/debug/*` | Observation inspection |

## Hardware

| Component | Spec |
|-----------|------|
| Follower arms | 2x Damiao AIRA Zero — J8009P (35 Nm) + J4340P (8 Nm) + J4310 (4 Nm), 7-DOF, CAN bus |
| Leader arms | 2x Dynamixel XL330 — USB serial, force feedback via current-position mode |
| Grippers | Damiao J4310 with torque-modulated force feedback |
| Frame | CNC aluminum extrusion, 1200x800 mm workspace |
| Cameras | 2x USB (wrist + overhead), RealSense + OpenCV supported |
| Compute | Linux workstation, NVIDIA GPU for policy training |

Motor control uses MIT impedance mode exclusively — position-velocity mode causes violent
oscillation on Damiao motors. Per-motor-type gains (J8009P: kp=30/kd=1.5, J4310:
kp=15/kd=0.25) are tuned to stay within torque saturation limits.

## Current Status

| Capability | Status |
|------------|--------|
| Dual-arm teleoperation (60 Hz, MIT impedance) | Working |
| Force feedback (gripper torque + joint compliance) | Working |
| Multi-pair teleop (Damiao + Feetech simultaneous) | Working |
| Safety layer (torque limits, joint bounds, E-STOP) | Working |
| Joint calibration + homing profiles | Working |
| Camera management + MJPEG streaming | Working |
| Episode recording (HDF5 + multi-camera video) | Working |
| Dataset browsing + merge | Working |
| Policy training — ACT, Diffusion Policy, SmolVLA, PI0.5 | Working (via LeRobot) |
| Policy deployment + autonomous execution | Working |
| Human-in-the-loop fine-tuning (HIL-SERL) | In progress |
| RL training (SAC, Gym environment) | In progress |
| Reward classifiers (SARM, GVL) | In progress |
| LLM task planning | In progress |

## Relationship to AURA

Nextis Bridge handles hardware control, teleoperation, and training.
[AURA](https://github.com/FLASH-73/AURA) handles assembly planning, CAD parsing, and
task sequencing. They are separate systems that will integrate — Bridge provides the
low-level control and learning pipeline that AURA's assembly sequencer will invoke.

## Testing

```bash
python -m pytest tests/ -v --tb=short   # All tests run without hardware
ruff check app/ tests/                  # Lint
```

Tests mock all hardware interfaces via `sys.modules` patching in `tests/conftest.py`.
Fixtures provide a FastAPI `TestClient` with mocked `SystemState` for endpoint testing.
