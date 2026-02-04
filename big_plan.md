# Nextis - Grand Vision & Roadmap

## Grand Vision

Nextis is building the **open-source operating system for robot manipulation**. Any researcher, hobbyist, or company should be able to:

1. Connect any robot arm (STS3215, Damiao, Dynamixel, or custom)
2. Teleoperate with intuitive leader arms and force feedback
3. Record demonstrations in minutes
4. Train state-of-the-art policies (ACT, Diffusion, Pi0.5, HIL-SERL)
5. Deploy autonomously with human-in-the-loop safety
6. Share datasets and policies with the community

The platform bridges the gap between cutting-edge research (LeRobot, HIL-SERL, GR00T) and practical robotics deployment.

---

## Current Achievements

### Working End-to-End Pipeline
- [x] **Teleoperation**: 60Hz control loop with gravity compensation, velocity smoothing
- [x] **Recording**: 30fps multi-camera capture, selective recording, LeRobot v3 format
- [x] **Training**: Diffusion Policy, SmolVLA, Pi0.5 (with LoRA fine-tuning)
- [x] **HIL Deployment**: DAgger-style retraining with human intervention detection
- [x] **Safety**: Motor load monitoring, debounced emergency stop
- [x] **Calibration**: Range discovery, gravity compensation, homing persistence
- [x] **Camera System**: Intel RealSense (RGB+depth) + USB webcam support
- [x] **Frontend UI**: Full dashboard for all operations (Next.js 16)
- [x] **Movement Scale**: Safety limiter for autonomous operation (0.1-1.0x)

### Policy Training Status
| Policy | Training | HIL Inference | Quality |
|--------|----------|---------------|---------|
| Diffusion | Working | Working | Good |
| SmolVLA | Working | Working | Good |
| Pi0.5 (LoRA) | Working (15 bugs fixed) | Working | Needs more data |
| ACT | LeRobot native (not wired in UI) | Not tested | - |

### Hardware Status (Jan 28, 2026)
- [x] 1x STS3215 7-DOF follower arm (operational)
- [x] 1x STS3215 leader arm (operational)
- [ ] 1x Damiao 7-DOF follower arm (J8009P/J4340P/J4310 motors, software integration in progress)
  - Base/Link1: J8009P (35Nm high torque)
  - Link2/Link3: J4340P (8Nm medium torque)
  - Link4/Link5/Gripper: J4310 (4Nm precision)
- [ ] 2x Dynamixel XL330 leader arms (motors arriving)
- [x] Intel RealSense D435 camera
- [x] USB webcam
- [x] NVIDIA RTX 5090 (32GB VRAM)

---

## YC Demo Goal (February 9, 2026)

### The Demo
**Robot autonomously assembles part of itself** - demonstrating:
1. **Sub-assembly**: Pick component, orient, insert into housing (peg-in-hole)
2. **Screw fastening**: Pick screw, align, tighten with appropriate force
3. **Full sequence**: Multiple operations chained together autonomously

### Why This Demo Matters
- Proves the platform works end-to-end: teleop -> record -> train -> deploy
- Assembly is the hardest manipulation task (requires precision + force awareness)
- "Robot builds itself" is a compelling narrative
- Demonstrates multi-motor support (Damiao high-torque for assembly)

### Success Criteria
- [ ] Robot completes at least 1 assembly operation autonomously (70%+ success rate)
- [ ] Smooth teleoperation with Dynamixel leaders (visibly better than STS3215)
- [ ] Training pipeline works out-of-the-box (< 1 hour from demos to deployment)
- [ ] Live demo runs reliably (3+ consecutive successes)

---

## Day-by-Day Timeline (Jan 28 - Feb 9)

| Day | Date | Focus | Deliverable | Risk |
|-----|------|-------|-------------|------|
| 1 | Jan 28 (Tue) | Planning & research | CLAUDE.md, big_plan.md, model research | Low |
| 2 | Jan 29 (Wed) | Build Dynamixel XL330 leader arms | Physical hardware assembled | Medium (parts arrival) |
| 3 | Jan 30 (Thu) | Build Damiao follower arm | Mechanical assembly complete | Medium (assembly complexity) |
| 4 | Jan 31 (Fri) | **SOFTWARE**: Damiao CAN driver + Dynamixel integration | Motor communication working | High (new protocol) |
| 5 | Feb 1 (Sat) | Calibration + teleop testing | Smooth teleoperation with all arms | Medium |
| 6 | Feb 2 (Sun) | Camera setup + workspace design | Recording-ready assembly station | Low |
| 7 | Feb 3 (Mon) | Record 30-50 assembly demos | Training dataset complete | Low |
| 8 | Feb 4 (Tue) | Train SARM reward model + start HIL-SERL | Trained models | Low |
| 9 | Feb 5 (Wed) | Test + iterate (re-record if needed) | Working policy (>50% success) | Medium |
| 10 | Feb 6 (Thu) | Refine demos + retrain | Improved policy (>70% success) | Medium |
| 11 | Feb 7 (Fri) | Demo rehearsal + backup plan | Demo-ready system | Low |
| 12 | Feb 8 (Sat) | Final rehearsal | Polished demo | Low |
| 13 | Feb 9 (Sun) | **YC DEMO** | Live demonstration | - |

### Critical Path
```
Damiao assembly (Day 3) -> CAN driver (Day 4) -> Calibration (Day 5) -> Recording (Day 7)
```
If Damiao arm isn't ready by Day 5, **fallback**: use STS3215 arms for the demo.

---

## AI Model Strategy

### Tier 1: Primary (Feb 9 Demo)

#### SARM + HIL-SERL (Stage-Aware Reward Modeling + Human-in-the-Loop RL)
**Why**: Achieves 100% success rates on assembly tasks through dense stage-aware rewards + online RL.
- **SARM**: Learns dense rewards from demonstrations, decomposes task into stages automatically
- **HIL-SERL**: Online RL refinement with human interventions (1-2.5 hours to converge)
- **Key results**: Timing belt assembly, motherboard assembly, IKEA shelf - all 100% success
- Best for longer-horizon assembly with multiple sub-stages (approach, insert, tighten)
- Fast local inference (no API calls needed once trained)

**Implementation in Nextis**:
- `app/core/sarm_reward_service.py` - SARM reward model training
- `app/core/rl_service.py` - HIL-SERL orchestration with SAC
- `frontend/components/RLTrainingModal.tsx` - Full RL training UI

#### ACT (Action Chunking with Transformers)
**Why**: Designed specifically for bimanual fine manipulation. Proven with 10 minutes of demos.
- Predicts sequences of k actions (reduces effective horizon by k)
- Works with CVAE architecture to handle multimodal demonstrations
- Fast inference (suitable for real-time control)
- **Best for**: Single-step peg-in-hole insertion, precise part placement (simpler tasks)

### Tier 2: Backup Options

#### Diffusion Policy
**Why**: Already proven in the Nextis system. Robust to noise in demonstrations.
- Iterative denoising for action generation
- Handles multimodal action distributions well
- Multi-camera support with auto-resize
- **Best for**: General pick-and-place, backup if SARM/HIL-SERL needs more demos

**Reference**: [HIL-SERL Paper](https://hil-serl.github.io/) | [LeRobot Integration](https://huggingface.co/docs/lerobot/hilserl)

### Tier 3: Future (Data Scaling)

#### NVIDIA GR00T Dreams
**Why**: Multiply 20 demonstrations into thousands using Cosmos world foundation models.
- Generates synthetic trajectory data from single image + language prompt
- Inverse Dynamics Model (IDM) converts 2D dream videos into 3D action trajectories
- 780K synthetic trajectories generated in 11 hours from small demo set
- 40% performance boost when combined with real data

**Current limitations**:
- Only supports SO-100, Franka, GR1, RoboCasa embodiments
- Requires NVIDIA Cosmos infrastructure
- Custom IDM training needed for new embodiments
- Not ready for Feb 9 timeline

**Reference**: [GR00T-Dreams GitHub](https://github.com/NVIDIA/GR00T-Dreams) | [NVIDIA Blog](https://developer.nvidia.com/blog/enhance-robot-learning-with-synthetic-trajectory-data-generated-by-world-foundation-models/)

### Tier 4: Research Frontier

#### Force-Aware Policies
For assembly beyond the demo, force feedback will be essential:

| Approach | Description | Relevance |
|----------|-------------|-----------|
| **Comp-ACT** | Variable compliance control via ACT + VR haptics | High - 20-30 demos, proven on contact-rich tasks |
| **FoAR** | Force-aware reactive policy, dynamic force/vision balance | High - handles sparsely-activated force signals |
| **FILIC** | Dual-loop force-guided IL with impedance control | High - 12-33% improvement with force feedback |
| **DIPCOM** | Diffusion policies for compliant manipulation | Medium - extends Diffusion with force regulation |

**Key insight from research**: Without force feedback, the policy cannot distinguish between free and obstructed motion. Adding any form of force/torque signal substantially improves performance (12-33% in simulation, even more on real robots).

---

## Hardware Roadmap

### Phase 1: YC Demo (Feb 9)
```
Current Setup                    Target Setup
================                 ================
1x STS3215 follower     -->     1x STS3215 follower (backup)
1x STS3215 leader       -->     1x Damiao follower (assembly arm)
                                2x Dynamixel XL330 leaders
```

### Phase 2: Force Sensing (Feb-Mar 2026)
- Add 6-axis F/T sensor to Damiao arm end-effector
- Enable haptic feedback in LeaderAssistService (k_haptic parameter exists, set to 0.0)
- Record force data as part of observation.state
- Train force-conditioned policies (Comp-ACT or FILIC approach)

### Phase 3: Compliance Control (Mar-Apr 2026)
- Implement impedance control for Damiao motors (MIT mode supports torque control)
- Hybrid force-position controller for insertion tasks
- Variable stiffness based on task phase (stiff for approach, compliant for insertion)

### Phase 4: Multi-Robot (Q2 2026)
- Support 4+ arms simultaneously
- Heterogeneous fleet (different arm types working together)
- Shared workspace coordination

---

## Software Roadmap

### Critical for Demo (Week 1-2)

#### 1. Damiao Motor Driver - IN PROGRESS
**Priority**: CRITICAL | **Effort**: 2-3 days
- New directory: `lerobot/src/lerobot/motors/damiao/`
- Reference: `examples_for_damiao/follower.py` (working implementation)
- Protocol: CAN bus via serial bridge (921600 baud)
- **Motor config**: J8009P (base/link1), J4340P (link2/link3), J4310 (link4/link5/gripper)
- Features:
  - Position read/write with POS_VEL mode
  - **Global velocity limiter (0-1)** - critical safety for heavy arm
  - Torque/current read (for load monitoring)
  - Gripper auto-homing via torque detection
  - Safe calibration procedure
- Integration points:
  - Motor bus abstraction (like STS3215 bus)
  - Safety layer (Damiao-specific torque limits)
  - Calibration service (homing, range discovery)
  - UI auto-detection and velocity slider

#### 2. Dynamixel XL330 Leader Integration
**Priority**: HIGH | **Effort**: 1-2 days
- Existing driver: `lerobot/src/lerobot/motors/dynamixel/dynamixel.py`
- Tasks:
  - Test XL330 compatibility with existing driver
  - Configure as leader arm type in settings.yaml
  - Verify position reading accuracy (12-bit encoder)
  - Low-friction backdrivability testing

#### 3. Mixed-Motor Robot Configuration
**Priority**: HIGH | **Effort**: 1 day
- New robot type supporting heterogeneous motor buses
- Settings.yaml: Damiao follower + Dynamixel leader
- Per-arm motor type selection
- Unified calibration across motor types

#### 4. ACT Training Pipeline (Secondary to SARM/HIL-SERL)
**Priority**: LOW | **Effort**: 0.5 days
- ACT is fully implemented in LeRobot but not wired in Nextis UI
- Use as fallback for simpler single-step tasks
- Add ACT presets to training_service.py (like existing Diffusion/SmolVLA/Pi0.5)
- Enable ACT option in TrainModal.tsx (currently disabled with "Coming soon")
- Default config: chunk_size=100, kl_weight=10, num_queries=100

### Post-Demo (Feb-Mar 2026)

#### 5. HIL-SERL Integration
**Priority**: HIGH | **Effort**: 1-2 weeks
- Install: `pip install -e ".[hilserl]"`
- Create URDF for Bi-Umbra/Damiao robot
- Implement GymManipulatorConfig for Nextis robots
- Wire reward classifier training into UI
- Actor-learner mode in HIL service
- Gamepad support for interventions

#### 6. Force Sensing Pipeline
**Priority**: MEDIUM | **Effort**: 2-3 weeks
- F/T sensor driver (serial/USB)
- Add force to observation.state vector
- Record force data in demonstrations
- Force-conditioned policy training (extend ACT/Diffusion)

#### 7. GR00T Dreams Data Augmentation
**Priority**: LOW | **Effort**: 3-4 weeks
- Custom embodiment config for Nextis arms
- IDM training on real demonstration data
- Cosmos API integration for dream generation
- Neural trajectory pipeline

---

## Risk Assessment

### High Risk
| Risk | Impact | Mitigation |
|------|--------|------------|
| Damiao CAN driver bugs | Demo blocked | Fallback to STS3215 arms |
| Dynamixel XL330s arrive late | Worse teleop quality | Use STS3215 leaders (existing) |
| ACT doesn't converge on assembly | No autonomous demo | Use Diffusion (already working) |
| Assembly task too hard for 30-50 demos | Low success rate | Simplify task (pick-and-place only) |

### Medium Risk
| Risk | Impact | Mitigation |
|------|--------|------------|
| Damiao mechanical assembly issues | 1-2 day delay | Start early, have spare parts |
| Camera placement suboptimal | Lower policy performance | Test multiple positions, use 3 cameras |
| Training takes too long | Less iteration time | Use smaller model, fewer steps for testing |

### Low Risk
| Risk | Impact | Mitigation |
|------|--------|------------|
| Frontend UI needs ACT support | Can train via CLI | CLI always available as backup |
| Calibration differences between motors | Slight position errors | Per-motor calibration profiles |

### Fallback Plan (If Everything Goes Wrong)
**Minimum viable demo**: STS3215 arms doing pick-and-place (already working) with Diffusion policy. Narrative: "Look how easy it is to go from teleop to autonomous execution."

---

## Key References & Resources

### Papers
- [HIL-SERL: Precise and Dexterous Robotic Manipulation via Human-in-the-Loop RL](https://hil-serl.github.io/) - Science Robotics 2025
- [ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://tonyzhaozh.github.io/aloha/) - RSS 2023
- [Comp-ACT: Variable Compliance Control from a Few Demonstrations](https://arxiv.org/html/2406.14990v1)
- [FoAR: Force-Aware Reactive Policy for Contact-Rich Manipulation](https://arxiv.org/html/2411.15753v1)
- [FILIC: Dual-Loop Force-Guided IL with Impedance Control](https://arxiv.org/html/2509.17053)
- [DIPCOM: Diffusion Policies for Compliant Manipulation](https://arxiv.org/html/2410.19235v1)
- [GR00T N1: Open Foundation Model for Generalist Humanoid Robots](https://d1qx31qr3h6wln.cloudfront.net/publications/GR00T_1_Whitepaper.pdf)

### Code & Tools
- [LeRobot HIL-SERL Guide](https://huggingface.co/docs/lerobot/hilserl)
- [GR00T-Dreams GitHub](https://github.com/NVIDIA/GR00T-Dreams)
- [DM_Control_Python (Damiao CAN)](https://github.com/cmjang/DM_Control_Python)
- [GELLO Teleoperation Framework](https://github.com/wuphilipp/gello_software)
- [Koch/SO-100 Low-Cost Robot Arm](https://github.com/AlexanderKoch-Koch/low_cost_robot)

### Hardware
- [Damiao J4340P Motor](https://store.foxtech.com/dm-j4340p-2ec-mit-driven-brushless-servo-joint-motor/)
- [Damiao J8009P Motor](https://store.foxtech.com/dm-j8009p-2ec-mit-driven-brushless-servo-joint-motor-with-dual-encoders-for-robotic-arms-actuator-for-robot/)
- [Damiao J4310 Motor](https://store.foxtech.com/dm-j4310-2ec-v1-1-mit-driven-brushless-servo-joint-motor-with-dual-encoders-gear-reduction-for-robotic-arms/)
- [Dynamixel XL330](https://robotis.us/dynamixel-xl330-m288-t/)
- [OpenArm Damiao Motor Kit](https://store.foxtech.com/openarm-damiao-motor-kit/)

---

## Post-YC Vision (Q1-Q2 2026)

### Phase 1: Assembly Excellence
- HIL-SERL integration for 100% success rate assembly
- Force-conditioned policies for delicate operations
- Multi-step task chaining (pick -> orient -> insert -> screw)

### Phase 2: Data Scaling
- GR00T Dreams integration for 100x data augmentation
- Synthetic-to-real transfer for new tasks without demonstrations
- Community dataset sharing (HuggingFace Hub integration)

### Phase 3: Platform Maturity
- One-click setup for new robot types
- Plugin system for custom motors/sensors
- Cloud training option (rent GPU, download policy)
- Mobile app for monitoring and control

### Phase 4: Ecosystem
- Marketplace for robot skills (trained policies)
- Standardized assembly task benchmarks
- Multi-robot coordination for complex assemblies
- Sim-to-real pipeline with Isaac Sim integration

---

*Last updated: January 28, 2026*
