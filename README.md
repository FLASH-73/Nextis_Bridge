# AURA — Autonomous Universal Robotic Assembly

Upload a CAD file, robot builds it. Teaching a new assembly takes hours, not months.

<!-- TODO: Add demo GIF or architecture diagram showing CAD → Plan → Teach → Learn → Run pipeline -->

## The Problem

Industrial robots require months of programming for every new product — fixture design,
motion planning, force tuning, error handling. Small-batch and custom manufacturing can't
justify that setup cost, so it stays manual.

Assembly is fundamentally harder than pick-and-place: it requires contact reasoning,
compliance control, and recovery from partial insertions and misalignments. Current
approaches either need end-to-end foundation models trained on millions of demonstrations,
or brittle scripted motion that breaks on the slightest variation. Neither scales to the
long tail of real manufacturing tasks.

## How It Works

1. **Parse** — Upload a STEP file. Extract parts, contact surfaces, and geometry.
   Export meshes for 3D visualization.
   `PythonOCC contact graph → GLB export → Three.js viewer` *(planned)*

2. **Plan** — Generate an assembly sequence from contact analysis and geometric
   heuristics. Human can reorder or override any step.
   `Topological sort on contact graph, size-based ordering, pick/place/insert primitives` *(planned)*

3. **Teach** — Teleoperate the hard steps with force feedback at 60 Hz. Record
   demonstrations as HDF5 episodes with synchronized multi-camera video.
   `Dynamixel leader → Damiao follower, MIT impedance control, 50 Hz recording`

4. **Learn** — Train per-step manipulation policies from demonstrations. Fine-tune
   on failure cases with human-in-the-loop RL.
   `ACT / Diffusion Policy / SmolVLA via LeRobot, HIL-SERL for online improvement`

5. **Run** — Execute the assembly autonomously. Retry failed steps with learned
   recovery behaviors. Escalate to human when stuck.
   `State machine sequencer, per-step policy router, step-level analytics` *(in development)*

## Quick Start

```bash
git clone https://github.com/FLASH-73/AURA.git && cd AURA

# Python environment
conda create -n aura python=3.11 -y && conda activate aura
pip install -r requirements.txt
pip install -e lerobot

# CAN bus setup (required for Damiao arms — run once per boot)
sudo ./setup_can.sh

# Start backend
python run_backend.py                      # FastAPI on http://localhost:8000

# Start frontend (separate terminal)
cd frontend && npm install && npm run dev  # Next.js on http://localhost:3000
```

Open `http://localhost:3000` to access the AURA dashboard. From there you can manage
arm connections, run joint calibration, start teleoperation with force feedback, record
demonstration episodes, browse datasets, and launch policy training jobs.

Hardware is optional — the dashboard, dataset tools, and training pipeline run without
connected arms.

## Architecture

```
app/
├── core/
│   ├── hardware/       # Arm registry, motor control (Damiao + Dynamixel + Feetech), safety layer
│   ├── teleop/         # 60 Hz control loop, force feedback, multi-pair leader-follower mapping
│   ├── calibration/    # Joint calibration, homing procedures, profile persistence
│   ├── cameras/        # Camera lifecycle, MJPEG streaming, device discovery
│   ├── dataset/        # Episode browsing, dataset merge (LeRobot v3 format)
│   ├── training/       # Job management, policy presets (ACT, Diffusion, SmolVLA, PI0)
│   ├── hil/            # Human-in-the-loop training (DAgger-style intervention capture)
│   ├── rl/             # Gym environment, reward classifiers (SARM, GVL)
│   ├── intervention.py # Human override detection during policy execution
│   └── orchestrator.py # Policy deployment and task routing
├── routes/             # FastAPI REST API — one file per domain (14 route modules)
├── state.py            # Service initialization singleton
└── config/             # YAML-based arm definitions, pairings, camera config
frontend/               # Next.js 16 + React 19 + Tailwind CSS + TypeScript
scripts/                # Motor configuration, calibration, diagnostics
lerobot/                # Vendored LeRobot fork (policy training + motor drivers)
```

Services initialize without hardware — the system degrades gracefully when arms aren't
connected. The arm registry, calibration profiles, and camera connections are all managed
independently so any subsystem can be tested in isolation.

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
| Episode recording (HDF5, multi-camera video) | Working |
| Policy training — ACT, Diffusion Policy, SmolVLA | Working (via LeRobot) |
| Dataset merge + browsing | Working |
| Human-in-the-loop fine-tuning | In progress |
| RL training (SERL-style, Gym environment) | In progress |
| Reward classifiers (SARM, GVL) | In progress |
| CAD parsing + assembly sequence generation | Planned |
| 3D assembly visualization | Planned |
| Autonomous assembly execution | Target milestone |

## Technical Approach

Most robotics systems treat manipulation tasks as monolithic — one policy, one task,
train until it works. AURA decomposes assembly into a graph of discrete steps, each
with its own control strategy. Easy steps (pick up a bolt, move to a fixture) get
scripted motion primitives. Hard steps (thread insertion, snap-fit engagement, compliant
alignment) get learned policies trained on small, focused demonstration datasets. This
means the system doesn't need a generalist foundation model. It needs 10-50
demonstrations per hard step and a way to improve from failures.

That improvement mechanism is human-in-the-loop SERL. When a learned step fails during
autonomous execution, a human takes over via teleoperation. The intervention — the
moment of takeover, the corrective trajectory, and the successful completion — becomes
training data. The policy for that step improves with every assembly run, converging
on robustness without requiring offline data collection campaigns. The intervention
engine tracks force signatures and position divergence to detect when the human
overrides, so the boundary between autonomous and human control is captured automatically.

Assembly is the right problem to solve because it's where the economics are most
compelling. Pick-and-place is largely solved. Locomotion is impressive but the market is
diffuse. Assembly — inserting pins, threading fasteners, routing cables, snapping
housings — requires contact reasoning, force control, and sequential planning that
current systems can't handle without extensive per-product engineering. The metric that
matters is setup time: how long it takes to go from a new CAD file to a running assembly
cell. The target is days, not months. That's what makes small-batch manufacturing
automation economically viable.

## Testing

```bash
python -m pytest tests/ -v --tb=short   # All tests run without hardware
ruff check app/ tests/                  # Lint
```

Tests mock all hardware interfaces via `sys.modules` patching in `tests/conftest.py`.
Fixtures provide a FastAPI `TestClient` with mocked `SystemState` for endpoint testing.

## Vision

AURA is the software platform. AIRA is the arm. Together, they're building toward robots
that can assemble anything — starting with industrial assembly because that's where the
economics work today.
