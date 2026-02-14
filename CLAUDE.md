# Nextis App

Robotics teleoperation, data collection, and policy training platform for the
Aira Zero 7-DOF arm (Damiao J-series) and Umbra arms (Feetech/Dynamixel).

## Quick Start

```bash
# Activate environment
conda activate base  # uses /home/roberto/miniconda3/bin/python3

# Set up CAN bus (required for Damiao arm, run once per boot)
sudo ./setup_can.sh

# Start backend
python run_backend.py          # FastAPI on http://localhost:8000

# Start frontend (separate terminal)
cd frontend && npm run dev     # Vite dev server on http://localhost:5173
```

## Source Tree

```
nextis_app/
├── app/
│   ├── main.py                 # FastAPI app, router includes (~54 lines)
│   ├── state.py                # SystemState singleton — service initialization
│   ├── dependencies.py         # FastAPI dependency injection (get_state)
│   ├── config/
│   │   └── settings.yaml       # Arms, pairings, camera config
│   ├── routes/                 # API endpoints (one file per domain)
│   │   ├── system.py           # /status, /config, /system/restart
│   │   ├── arms.py             # Arm CRUD, connect/disconnect
│   │   ├── motors.py           # Motor diagnostics, torque control
│   │   ├── calibration.py      # Joint calibration, gravity cal, homing
│   │   ├── teleop.py           # Start/stop teleoperation
│   │   ├── cameras.py          # Camera connect/disconnect, streaming
│   │   ├── recording.py        # Session/episode management
│   │   ├── datasets.py         # Dataset browsing, merge operations
│   │   ├── training.py         # Training job management
│   │   ├── policies.py         # Policy listing, deployment
│   │   ├── hil.py              # Human-in-the-loop sessions
│   │   ├── rl.py               # RL training, reward classifiers
│   │   ├── chat.py             # LLM task planning
│   │   └── debug.py            # Debug endpoints
│   └── core/                   # Backend services
│       ├── config.py           # Path constants + YAML config I/O
│       ├── hardware/           # Motor control, arm registry, safety
│       │   ├── arm_registry.py # ArmRegistryService — multi-arm lifecycle
│       │   ├── safety.py       # SafetyLayer — torque limits
│       │   ├── motor_recovery.py
│       │   ├── leader_assist.py # Gravity compensation for leader arms
│       │   ├── gravity_comp.py
│       │   ├── connection.py   # create_arm_instance() factory
│       │   ├── types.py        # MotorType, ArmRole, ArmDefinition, Pairing
│       │   └── tables.py       # DAMIAO_TORQUE_LIMITS
│       ├── teleop/             # Teleoperation system
│       │   ├── service.py      # TeleoperationService — multi-pair control
│       │   ├── pairing.py      # PairingContext — isolated per-pair state
│       │   ├── control_loop.py # 60Hz leader→follower mapping
│       │   ├── recording.py    # Session/episode recording, video capture
│       │   ├── homing.py       # Homing procedures
│       │   └── observation.py  # Observation extraction
│       ├── cameras/            # Camera management
│       │   ├── service.py      # CameraService — connect/disconnect/stream
│       │   └── discovery.py    # Device enumeration
│       ├── calibration/        # Calibration system
│       │   ├── service.py      # CalibrationService — homing + profiles
│       │   ├── profiles.py     # Profile persistence
│       │   ├── discovery.py    # Range discovery
│       │   └── homing.py       # Homing procedures, inversion tracking
│       ├── dataset/            # Dataset browsing + merge
│       │   ├── service.py      # DatasetService — CRUD, episode data
│       │   └── merge.py        # MergeJobManager — background merge jobs
│       ├── training/           # Training infrastructure
│       │   ├── service.py      # TrainingService — job management
│       │   ├── commands.py     # Subprocess execution
│       │   ├── policies.py     # Policy loading, inference
│       │   ├── validators.py   # Dataset validation
│       │   ├── presets.py      # SmolVLA, Diffusion, PI0, ACT presets
│       │   └── types.py        # Training type definitions
│       ├── hil/                # Human-in-the-loop
│       │   ├── service.py      # HILService — DAgger-style training
│       │   ├── loop.py         # HIL control loop
│       │   ├── observation.py  # HIL observation extraction
│       │   └── types.py        # HILMode, HILSessionState
│       ├── rl/                 # Reinforcement learning
│       │   ├── service.py      # RLService — training orchestration
│       │   ├── env.py          # NextisRobotEnv — Gym environment
│       │   ├── types.py        # RLConfig, RLTrainingState
│       │   └── rewards/        # Reward classifiers
│       │       ├── classifier.py
│       │       ├── sarm.py     # Stage-Aware Reward Modeling
│       │       └── gvl.py      # Goal-Value Learning
│       ├── intervention.py     # InterventionEngine — human override detection
│       ├── orchestrator.py     # TaskOrchestrator — policy deployment
│       ├── planner.py          # Gemini / local LLM task planning
│       ├── recorder.py         # DataRecorder (legacy LeRobot writer)
│       └── shared_memory.py    # SharedMemoryRingBuffer (IPC utility)
├── frontend/                   # React + Vite + TypeScript
├── scripts/                    # CLI tools (run from project root)
│   ├── calibrate_encoder.py    # Dual-encoder calibration (J8009P-2EC)
│   ├── calibrate_grippers.py   # Interactive gripper calibration
│   ├── calibrate_gravity.py    # Gravity compensation data collection
│   ├── configure_all_motors_mit.py
│   ├── rezero_motor.py         # Reset encoder zero position
│   ├── force_reset_offsets.py
│   ├── nuclear_reset_calibration.py
│   └── diagnostics/
│       ├── gripper_diagnostic.py
│       ├── test_link1.py
│       ├── test_link1_diagnostic.py
│       ├── test_link2_recovery.py
│       └── test_mit_mode_position.py
├── tests/
│   ├── test_intervention.py
│   └── test_training.py
├── lerobot/                    # LeRobot fork (vendored)
├── calibration_profiles/       # Saved joint calibration files
├── calibration_gravity/        # Gravity compensation models
├── datasets/                   # Recorded datasets (LeRobot v3 format)
├── training/outputs/           # Trained policy checkpoints
├── models/                     # RL models
├── run_backend.py              # Backend entry point (uvicorn wrapper)
├── setup_can.sh                # CAN bus setup (txqueuelen, bitrate)
├── start.sh / stop.sh          # Process management
└── CLAUDE.md                   # This file
```

## Service Initialization (app/state.py)

1. Lightweight services: CameraService, DatasetService, TrainingService
2. Load config from `app/config/settings.yaml`
3. ArmRegistryService (reads config, no hardware)
4. DataRecorder
5. CalibrationService, TeleoperationService (robot=None)
6. TaskOrchestrator (mock robot)
7. RL reward services
8. HILService
9. Planner: lazy-loaded on first /chat request

## API Route Groups

| Prefix | File | Purpose |
|--------|------|---------|
| `/status`, `/config` | system.py | System status, configuration |
| `/arms/*` | arms.py | Arm CRUD, connect/disconnect |
| `/motors/*` | motors.py | Motor diagnostics, torque control |
| `/calibration/*` | calibration.py | Joint calibration, gravity cal |
| `/teleop/*` | teleop.py | Start/stop teleoperation |
| `/cameras/*` | cameras.py | Camera connect/disconnect, streaming |
| `/recording/*` | recording.py | Session/episode management |
| `/datasets/*` | datasets.py | Dataset browsing, merge operations |
| `/training/*` | training.py | Training job management |
| `/policies/*` | policies.py | Policy listing, deployment |
| `/hil/*` | hil.py | Human-in-the-loop sessions |
| `/rl/*` | rl.py | RL training, reward classifiers |
| `/chat` | chat.py | LLM task planning |

## Key Design Decisions

- **MIT mode only** for Damiao motors (POS_VEL causes vibration). Per-motor-type
  gains in `lerobot/motors/damiao/tables.py` MIT_GAINS.
- **PairingContext** for multi-pair teleop: each leader-follower pair gets isolated
  state to prevent mapping overwrites between Damiao and Feetech pairs.
- **CameraService** is independent of robot lifecycle: cameras connect/disconnect
  via dedicated API, not tied to arm connections.
- **Path constants** in `app/core/config.py`: all well-known directories (datasets,
  calibration, training outputs) defined once, imported everywhere.
- **Delegation pattern** for dataset merge: DatasetService delegates to MergeJobManager
  (separate threading model, shared base_path).
- **Backward-compat shims** in `app/core/`: flat-file imports like
  `from app.core.teleop_service import TeleoperationService` still work.

## Safety Rules

- **MIT mode only** for Damiao motors (POS_VEL causes violent vibration)
- **J4310**: kp=15, kd=0.25 (higher kd causes torque saturation oscillation)
- **J8009P/J4340P**: kp=30, kd=1.5
- sync_read probes MUST use kp=0, kd=0
- CAN txqueuelen >= 256 for 7 motors at 60Hz (`setup_can.sh`)
- NEVER use CURRENT mode (mode 0) on leader arm joints -- positive feedback loop

## Running Tests

```bash
# Run all tests (no hardware needed)
python -m pytest tests/ -v --tb=short

# Lint
ruff check app/ tests/
```

Tests run without hardware. All lerobot imports are mocked via `sys.modules` in
`tests/conftest.py`. Fixtures: `mock_robot`, `safety_layer`, `training_service`,
`app_client` (FastAPI TestClient with mocked SystemState).

## Python Environment

Use conda: `/home/roberto/miniconda3/bin/python3` (NOT `/usr/bin/python3`).
The `lerobot/src` path is added to `sys.path` by `run_backend.py` and individual scripts.
