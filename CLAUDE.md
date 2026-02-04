# Nextis App - Project Documentation

## General instructions:

Dont delete any features and functionality if not entirely sure. Rather ask, before overriding some other features just because i have not told you about it. Try to keep in mind when solving problems and fixing things that the end goal is that many different users can opensource download my software and it also works allmost effortlessly on their computer and with their arms.

## Claude Code Plan Mode Instructions

When using plan mode:
- **Keep plans concise**: Only include current/pending tasks
- **Remove completed tasks**: Once a task is done, delete it from the plan file
- **No appending**: Do not endlessly append new sections - update existing content instead
- **Current state only**: The plan should reflect what remains to be done, not a history of everything completed
- **Before exiting plan mode**: Clean up the plan file to remove any completed items

## YC Demo Goal (Feb 9, 2026)

**Task**: Robot autonomously assembles part of itself

**Operations**:
- **Full sub-assembly**: Insert motor/component into housing, orient correctly, fasten (peg-in-hole)
- **Screw fastening**: Force-sensitive tightening of screws/fasteners
- **Part manipulation**: Pick, orient, and place components

**Demo Flow**:
1. Teleop with Dynamixel XL330 leader arms (low friction, smooth demonstrations)
2. Collect 30-50 high-quality, consistent demonstrations
3. Train SARM reward model on demonstration dataset
4. Run HIL-SERL online RL to achieve 100% success rate
5. Autonomous replay of the learned assembly task

**Hardware Setup (Target)**:
- 1x Damiao 7-DOF follower arm (J4340P/J8009P/J4310 motors) - high torque, precision
- 1x STS3215 7-DOF follower arm (existing)
- 2x Dynamixel XL330 leader arms - low friction teleoperation

**Policy Strategy**:
- Primary: SARM/HIL-SERL (Stage-Aware Reward Modeling + Human-in-the-Loop RL)
  - Best for longer-horizon assembly tasks with multiple sub-stages
  - Learns dense rewards from demonstrations, achieves 100% success via online RL refinement
- Secondary: ACT (Action Chunking with Transformers) - designed for bimanual fine manipulation
- Backup: Diffusion Policy - already proven in the system, robust to multimodal data

**Key Constraints**:
- 20-50 episodes: Keep demonstrations narrow and consistent (same positions, same approach)
- No recovery demos: Each demo should be a clean, successful execution
- Camera placement: Must clearly show insertion/fastening workspace

See `big_plan.md` for detailed timeline, milestones, and AI model research.

## Overview

Nextis is an advanced robotics control and learning platform for bimanual (dual-arm) robot systems. It combines real-time teleoperation, demonstration recording, policy training, and autonomous execution into a unified application.

**Core Goals:**
1. Enable intuitive teleoperation of dual-arm robots via leader arms
2. Record high-quality demonstration datasets for imitation learning
3. Train policies (ACT, Diffusion, SmolVLA, Pi0.5) on collected demonstrations
4. Execute learned policies with human intervention capability (HIL)
5. Provide comprehensive calibration and safety systems
6. Support multiple motor types (STS3215, Damiao, Dynamixel) for diverse arm configurations

## Technology Stack

### Backend
- **Framework**: FastAPI (Python)
- **Robot Framework**: LeRobot (HuggingFace) - custom fork with multi-motor support
- **Motor Control**: Feetech STS3215 (serial), Damiao J-series (CAN), Dynamixel XL330 (TTL)
- **Cameras**: Intel RealSense (depth + RGB) and OpenCV (USB cameras)
- **ML**: PyTorch for policy training (ACT, Diffusion, SmolVLA, Pi0.5 with LoRA)
- **LLM**: Google Gemini API for task planning

### Frontend
- **Framework**: Next.js 16 (React 19, TypeScript)
- **Styling**: Tailwind CSS 4
- **Charts**: Recharts
- **Icons**: Lucide React

### Hardware
- **Follower Arms**: Bi-Umbra (dual 7-DOF, STS3215) + Damiao arm (7-DOF, J4340P/J8009P/J4310)
- **Leader Arms**: Dynamixel XL330 (low friction, high-quality teleoperation)
- **Cameras**: Intel RealSense D400 series + USB webcams
- **GPU**: NVIDIA RTX 5090 (32GB) for training

### Motor Types
| Motor | Protocol | Use | Torque | Notes |
|-------|----------|-----|--------|-------|
| Feetech STS3215 | Serial (TTL) | Follower + Legacy Leader | Medium | Current primary, load monitoring |
| Damiao J4340P/J8009P/J4310 | CAN bus | Follower (high-torque) | High | Dual encoders, MIT mode, assembly-grade |
| Dynamixel XL330 | Serial (TTL) | Leader (teleop) | Low | Ultra-low friction, 12-bit encoders, GELLO-compatible |

## Project Structure

```
nextis_app/
├── app/                          # Backend (FastAPI)
│   ├── main.py                   # API routes & WebSocket handlers
│   ├── core/
│   │   ├── teleop_service.py     # Real-time teleoperation (60Hz)
│   │   ├── dataset_service.py    # Dataset discovery & loading
│   │   ├── calibration_service.py# Motor calibration & homing
│   │   ├── orchestrator.py       # Autonomous task execution
│   │   ├── leader_assist.py      # Gravity compensation & haptics
│   │   ├── safety_layer.py       # Motor load monitoring
│   │   ├── camera_service.py     # Camera management
│   │   ├── intervention_controller.py  # Human takeover detection
│   │   ├── recorder.py           # Dataset recording
│   │   └── planner.py            # LLM task decomposition
│   └── config/
│       └── settings.yaml         # Hardware configuration
│
├── frontend/                     # Frontend (Next.js)
│   ├── app/
│   │   └── page.tsx              # Main dashboard
│   └── components/
│       ├── TeleopModal.tsx       # Teleoperation UI
│       ├── CalibrationModal.tsx  # Calibration wizard
│       ├── RecordingModal.tsx    # Data collection UI
│       ├── DatasetViewerModal.tsx# Dataset browser
│       ├── CameraModal.tsx       # Camera configuration
│       └── GravityWizard.tsx     # Gravity calibration
│
├── lerobot/                      # LeRobot framework (submodule)
│   └── src/lerobot/
│       ├── robots/               # Robot implementations
│       ├── cameras/              # Camera drivers
│       ├── motors/               # Motor control
│       ├── datasets/             # Dataset utilities
│       └── policies/             # Policy implementations
│
├── datasets/                     # Recorded datasets (LeRobot format)
├── models/                       # Trained policy models
├── calibration_gravity/          # Calibration profiles (JSON)
└── start.sh                      # Launch script
```

## Key Services

### TeleoperationService (`teleop_service.py`)
Real-time control loop running at 60Hz:
- Maps leader arm positions to follower motors
- Applies velocity smoothing (EMA filter)
- Integrates leader assist (gravity compensation)
- Records frames at 30fps during demonstration collection
- Thread-safe action caching for recording

**Critical variables:**
- `_latest_leader_action`: Cached motor positions (used by recording)
- `recording_active`: Episode recording state
- `session_active`: Dataset session state

### Recording System
Two-thread architecture for smooth recording:
1. **Teleop loop** (60Hz): Controls robot, caches motor positions
2. **Recording capture loop** (30fps): Builds frames from cached data + cameras

**Important**: Recording uses cached teleop data to avoid slow hardware reads. Cameras use `async_read(blocking=False)` for Zero-Order Hold pattern.

### Dataset Format (LeRobot v3)
```
datasets/{repo_id}/
├── meta/
│   ├── info.json                 # fps, robot_type, total_episodes
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet  # Episode metadata
├── data/
│   └── chunk-000/
│       └── file-000.parquet      # Frame data (action, state)
└── videos/
    └── observation.images.{camera}/
        └── episode_000000.mp4    # Video files
```

### CalibrationService
- **Range Discovery**: Finds min/max for each joint
- **Gravity Compensation**: Learns linear gravity model via sampling
- **Homing**: Returns robot to calibrated zero position
- **Persistence**: Saves to `calibration_gravity/{arm_id}.json`

### Safety Layer
- Monitors motor loads (threshold: 500/1000)
- Triggers emergency stop on sustained overload
- Debounces violations (3 consecutive triggers)

## API Endpoints (Key Routes)

### Teleoperation
- `POST /teleop/start` - Start teleoperation
- `POST /teleop/stop` - Stop teleoperation
- `GET /teleop/data` - Joint positions & forces

### Recording
- `POST /recording/session/start` - Begin dataset session
- `POST /recording/session/stop` - End session (finalizes dataset)
- `POST /recording/episode/start` - Start recording episode
- `POST /recording/episode/stop` - Stop episode (saves to dataset)

### Datasets
- `GET /datasets` - List all datasets
- `GET /datasets/{repo_id}/episode/{index}` - Get episode data
- `DELETE /datasets/{repo_id}` - Delete dataset

### Cameras
- `GET /cameras/scan` - Discover cameras
- `GET /video_feed/{camera_key}` - MJPEG stream

## Development Guidelines

### Recording Flow
1. User clicks "Start Session" → `start_recording_session(repo_id, task)`
2. Creates/opens LeRobotDataset, starts video encoding
3. User clicks "Record" → `start_episode()`
4. Recording capture loop builds frames from:
   - Motor positions: `_latest_leader_action` (cached from teleop)
   - Camera images: `cam.async_read(blocking=False)`
5. Frame writer thread calls `dataset.add_frame(frame)`
6. User clicks "Stop" → `stop_episode()` → `dataset.save_episode()`
7. User clicks "Finish" → `stop_recording_session()` → `dataset.finalize()`

### Frame Structure
```python
frame = {
    "observation.state": np.array([...]),      # Motor positions
    "observation.images.camera_1": np.array([...]),  # RGB image
    "observation.images.camera_2": np.array([...]),
    "action": np.array([...]),                 # Target positions
    "task": "Pick up the cube"                 # Task description
}
```

### Common Issues & Solutions

**Issue**: Frames not being recorded
- Check `_latest_leader_action` is populated (teleop must be running)
- Verify cameras return frames via `async_read`
- Check for validation errors in frame writer logs

**Issue**: Low FPS during recording
- Recording loop must NOT call `robot.get_observation()` (slow hardware read)
- Use cached teleop data instead
- Cameras must use `async_read(blocking=False)`

**Issue**: Episode count not resetting
- `episode_count` must be set from `dataset.meta.total_episodes` on session start
- Frontend must use `episode_count` from API response

### Important Patterns

**Zero-Order Hold (ZOH)**: Cameras run in background threads, `async_read(blocking=False)` returns last cached frame instantly.

**Thread Safety**: Use `_action_lock` when accessing `_latest_leader_action` from recording thread.

**LeRobot Frame Building**:
```python
from lerobot.datasets.utils import build_dataset_frame
obs_frame = build_dataset_frame(dataset.features, obs_dict, prefix="observation")
action_frame = build_dataset_frame(dataset.features, action_dict, prefix="action")
```

## Configuration

### settings.yaml
```yaml
robot:
  type: bi_umbra_follower
  left_arm_port: /dev/ttyUSB2
  right_arm_port: /dev/ttyUSB1
  cameras:
    camera_1:
      type: intelrealsense
      serial_number_or_name: '218622275492'
      fps: 30
      use_depth: true

teleop:
  type: bi_umbra_leader
  left_arm_port: /dev/ttyUSB0
  right_arm_port: /dev/ttyUSB3
```

## Running the App

```bash
./start.sh
```

This launches:
- Backend: `uvicorn app.main:app --port 8000`
- Frontend: `npm run dev` (port 3000)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (Next.js)                       │
│  Dashboard │ Teleop │ Recording │ Calibration │ Datasets    │
└─────────────────────────┬───────────────────────────────────┘
                          │ REST + WebSocket
┌─────────────────────────▼───────────────────────────────────┐
│                     Backend (FastAPI)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ TeleopService│  │ Recording   │  │ CalibrationService  │ │
│  │ (60Hz loop) │  │ (30fps)     │  │ (gravity, homing)   │ │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────┘ │
│         │                │                                   │
│  ┌──────▼────────────────▼──────┐  ┌─────────────────────┐ │
│  │      LeRobot Framework        │  │    Safety Layer     │ │
│  │  Robots │ Cameras │ Datasets  │  │  (load monitoring)  │ │
│  └──────────────┬────────────────┘  └─────────────────────┘ │
└─────────────────┼───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐   ┌────▼────┐   ┌────▼────┐
│Leader │   │Follower │   │ Cameras │
│ Arms  │   │  Robot  │   │(RS+USB) │
└───────┘   └─────────┘   └─────────┘
```

## Policy Selection Guide

| Policy | Best For | Demos Needed | Inference Speed | VRAM |
|--------|----------|-------------|-----------------|------|
| **SARM/HIL-SERL** | Long-horizon assembly, multi-stage tasks | 20-50 + RL | Fast (real-time) | ~8GB |
| **ACT** | Fine bimanual manipulation, single-step | 10-50 | Fast (real-time) | ~8GB |
| **Diffusion** | General manipulation, multi-modal tasks | 20-100 | Medium | ~8GB |
| **SmolVLA** | Language-conditioned, multi-task | 50-200 | Medium | ~16GB |
| **Pi0.5** | Open-world VLA, generalization | 20-50 | Slow | ~22GB (LoRA) |

**For assembly tasks**: Use SARM/HIL-SERL as primary (learns dense stage-aware rewards, achieves 100% success via online RL). ACT for simpler single-step manipulation. Diffusion as backup.

## Assembly Demo Best Practices

**Data Collection**:
- Record 30-50 nearly-identical demonstrations (same object positions, same approach angles)
- Use consistent camera placement showing the workspace clearly
- Avoid recovery/correction demos - each should be a clean successful execution
- Leader arms should be low-friction (Dynamixel XL330) for smooth, natural motions

**Camera Setup for Assembly**:
- Camera 1: Overhead view of workspace (insertion visibility)
- Camera 2: Side/angled view (depth perception for insertion)
- Ensure lighting is consistent between training and inference

**Training Tips**:
- ACT: chunk_size=100 (2 seconds at 50Hz), 50-100k steps
- Diffusion: horizon=16, n_action_steps=8, 100k+ steps
- Always use quantile normalization for Pi0.5

## Glossary

- **LeRobot**: HuggingFace robotics framework for datasets and policies
- **Bi-Umbra**: Dual-arm robot configuration (left + right arms)
- **Leader/Follower**: Teleop control arms / actual robot arms
- **ZOH**: Zero-Order Hold - returns last cached value instantly
- **Episode**: Single demonstration recording (start to stop)
- **Session**: Collection of episodes for one dataset
- **SARM**: Stage-Aware Reward Modeling - learns dense rewards from demonstrations for multi-stage tasks
- **ACT**: Action Chunking with Transformers - predicts action sequences for fine manipulation
- **Diffusion Policy**: Neural network that learns actions via denoising
- **HIL-SERL**: Human-in-the-Loop Sample-Efficient RL - online RL with human interventions, achieves 100% success
- **Damiao**: High-torque brushless servo motors with CAN bus protocol
- **GELLO**: General Low-cost teleoperation framework using Dynamixel servos

