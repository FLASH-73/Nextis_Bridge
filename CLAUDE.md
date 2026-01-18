# Nextis App - Project Documentation

## Overview

Nextis is an advanced robotics control and learning platform for bimanual (dual-arm) robot systems. It combines real-time teleoperation, demonstration recording, policy training, and autonomous execution into a unified application.

**Core Goals:**
1. Enable intuitive teleoperation of dual-arm robots via leader arms
2. Record high-quality demonstration datasets for imitation learning
3. Train diffusion policies on collected demonstrations
4. Execute learned policies with human intervention capability
5. Provide comprehensive calibration and safety systems

## Technology Stack

### Backend
- **Framework**: FastAPI (Python)
- **Robot Framework**: LeRobot (HuggingFace) - custom fork with Bi-Umbra support
- **Motor Control**: Feetech STS3215 servos via serial communication
- **Cameras**: Intel RealSense (depth + RGB) and OpenCV (USB cameras)
- **ML**: PyTorch for diffusion policy training
- **LLM**: Google Gemini API for task planning

### Frontend
- **Framework**: Next.js 16 (React 19, TypeScript)
- **Styling**: Tailwind CSS 4
- **Charts**: Recharts
- **Icons**: Lucide React

### Hardware
- **Robot**: Bi-Umbra Follower (dual 7-DOF arms)
- **Teleop**: Bi-Umbra Leader (control arms)
- **Cameras**: Intel RealSense D400 series + USB webcams

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

## Glossary

- **LeRobot**: HuggingFace robotics framework for datasets and policies
- **Bi-Umbra**: Dual-arm robot configuration (left + right arms)
- **Leader/Follower**: Teleop control arms / actual robot arms
- **ZOH**: Zero-Order Hold - returns last cached value instantly
- **Episode**: Single demonstration recording (start to stop)
- **Session**: Collection of episodes for one dataset
- **Diffusion Policy**: Neural network that learns actions via denoising
