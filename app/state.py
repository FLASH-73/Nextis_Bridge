import threading

from app.core.calibration_service import CalibrationService
from app.core.camera_service import CameraService
from app.core.config import CONFIG_PATH, load_config
from app.core.dataset import DatasetService
from app.core.orchestrator import TaskOrchestrator
from app.core.recorder import DataRecorder
from app.core.teleop_service import TeleoperationService
from app.core.training import TrainingService


class SystemState:
    def __init__(self):
        self.robot = None
        self.leader = None
        self.recorder = None
        self.orchestrator = None
        self.calibration_service = None
        self.camera_service = None
        self.teleop_service = None
        self.dataset_service = None
        self.training_service = None
        self.hil_service = None
        self.reward_classifier_service = None
        self.gvl_reward_service = None
        self.sarm_reward_service = None
        self.rl_service = None
        self.deployment_runtime = None
        self.arm_registry = None  # Arm Manager Service
        self.tool_registry = None
        self.trigger_listener = None
        self.safety_watchdog = None
        self.leader_assists = {} # {arm_prefix: LeaderAssistService}
        self.lock = threading.Lock()

        self.is_initializing = False
        self.init_error = None


    def initialize(self):
        self.is_initializing = True
        self.init_error = None
        try:
             self._inner_initialize()
        except Exception as e:
             import traceback
             print(f"CRITICAL INIT ERROR: {e}")
             traceback.print_exc()
             self.init_error = str(e)
        finally:
             self.is_initializing = False
             print("System Initialization Complete.")

    def _inner_initialize(self):
        print("Initializing System (Fast Startup — no hardware)...")

        # 1. Lightweight services (no hardware deps)
        self.camera_service = CameraService()
        self.camera_service.start_health_monitor()
        self.dataset_service = DatasetService()
        self.training_service = TrainingService()

        # 2. Load config
        load_config()

        # 3. Arm Registry (reads config only, no hardware)
        try:
            from app.core.arm_registry import ArmRegistryService
            self.arm_registry = ArmRegistryService(config_path=str(CONFIG_PATH))
            print(f"Arm Registry initialized: {self.arm_registry.get_status_summary()}")
        except Exception as e:
            print(f"Warning: Arm Registry init failed: {e}")
            self.arm_registry = None

        # 3b. Tool Registry (reads config only, no hardware)
        try:
            from app.core.hardware.tool_registry import ToolRegistryService
            self.tool_registry = ToolRegistryService(config_path=str(CONFIG_PATH))
            print(f"Tool Registry initialized: {len(self.tool_registry.tools)} tools, {len(self.tool_registry.triggers)} triggers")
        except Exception as e:
            print(f"Warning: Tool Registry init failed: {e}")
            self.tool_registry = None

        # 3c. Trigger Listener (not started until explicitly requested)
        try:
            from app.core.hardware.trigger_listener import TriggerListenerService
            self.trigger_listener = TriggerListenerService(self.tool_registry) if self.tool_registry else None
        except Exception as e:
            print(f"Warning: Trigger Listener init failed: {e}")
            self.trigger_listener = None

        # 4. Data Recorder
        self.recorder = DataRecorder(repo_id="roberto/nextis_data", robot_type="bi_umbra_follower")

        # 5. CalibrationService (works with robot=None, uses arm_registry)
        self.calibration_service = CalibrationService(
            robot=None, leader=None, robot_lock=self.lock,
            arm_registry=self.arm_registry
        )

        # 6. TeleoperationService (works with robot=None, uses arm_registry)
        self.teleop_service = TeleoperationService(
            robot=None, leader=None, robot_lock=self.lock,
            leader_assists={}, arm_registry=self.arm_registry,
            camera_service=self.camera_service,
            trigger_listener=self.trigger_listener,
        )

        # 6b. Safety Watchdog (created but not started until arms connect)
        try:
            from app.core.hardware.safety_watchdog import SafetyWatchdog
            if self.arm_registry:
                self.safety_watchdog = SafetyWatchdog(
                    arm_registry=self.arm_registry,
                    safety_layer=self.teleop_service.safety,
                )
        except Exception as e:
            print(f"Warning: Safety watchdog init failed: {e}")
            self.safety_watchdog = None

        # 7. Orchestrator with minimal mock robot
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.is_connected = False
        mock.is_mock = True
        mock.is_calibrated = True
        mock.robot_type = "mock_robot"
        mock.observation_features = {}
        mock.action_features = {}
        self.robot = mock

        self.orchestrator = TaskOrchestrator(self.robot, self.recorder, robot_lock=self.lock)
        self.orchestrator.start()

        # 8. Reward / RL services (lightweight inits)
        from app.core.rl.rewards import RewardClassifierService
        self.reward_classifier_service = RewardClassifierService()
        from app.core.rl.rewards import GVLRewardService
        self.gvl_reward_service = GVLRewardService()
        from app.core.rl.rewards import SARMRewardService
        self.sarm_reward_service = SARMRewardService()

        # 9. HIL Service
        from app.core.hil import HILService
        self.hil_service = HILService(
            teleop_service=self.teleop_service,
            orchestrator=self.orchestrator,
            training_service=self.training_service,
            robot_lock=self.lock
        )

        # 10. Deployment Runtime
        from app.core.deployment import DeploymentRuntime
        self.deployment_runtime = DeploymentRuntime(
            teleop_service=self.teleop_service,
            training_service=self.training_service,
            arm_registry=self.arm_registry,
            camera_service=self.camera_service,
            robot_lock=self.lock,
        )

        # NOTE: Planner is lazy-loaded on first /chat request
        self.planner = None

        print("System ready (connect arms via UI when needed)")

    def reload(self):
        print("Reloading System...")
        # Stop Orchestrator first to stop using the robot
        if self.orchestrator:
            self.orchestrator.stop()

        with self.lock:
            # Disconnect Robot
            if self.robot:
                try:
                    if hasattr(self.robot, 'disconnect'):
                        self.robot.disconnect()
                except Exception as e:
                    print(f"Error disconnecting robot: {e}")

            # Disconnect Leader
            if self.leader:
                try:
                    if hasattr(self.leader, 'disconnect'):
                        self.leader.disconnect()
                except Exception as e:
                    print(f"Error disconnecting leader: {e}")

            # Re-initialize
            import time
            time.sleep(2)
            self.initialize()

    def shutdown(self):
        print("Shutting Down System State...")
        # Stop deployment runtime first (it may be running a control loop)
        if self.deployment_runtime:
            try:
                self.deployment_runtime.stop()
            except Exception as e:
                print(f"Error stopping deployment runtime: {e}")

        if self.orchestrator:
            self.orchestrator.stop()

        # Stop safety watchdog
        if self.safety_watchdog:
            try:
                self.safety_watchdog.stop()
            except Exception as e:
                print(f"Error stopping safety watchdog: {e}")

        # Stop trigger listener
        if self.trigger_listener:
            try:
                self.trigger_listener.stop()
            except Exception as e:
                print(f"Error stopping trigger listener: {e}")

        # Stop camera health monitor, then disconnect all managed cameras
        if self.camera_service:
            try:
                self.camera_service.stop_health_monitor()
                self.camera_service.disconnect_all()
            except Exception as e:
                print(f"Error disconnecting cameras: {e}")

        with self.lock:
            # Disconnect Robot
            if self.robot:
                try:
                    if hasattr(self.robot, 'disconnect'):
                        self.robot.disconnect()
                except Exception as e:
                    print(f"Error disconnecting robot: {e}")

            # Disconnect Leader
            if self.leader:
                try:
                    if hasattr(self.leader, 'disconnect'):
                        self.leader.disconnect()
                except Exception as e:
                    print(f"Error disconnecting leader: {e}")

        # Brief pause to ensure OS releases handles
        import time
        time.sleep(0.5)

    def restart(self):
        print("SYSTEM RESTART REQUESTED. RESTARTING PROCESS...")
        import os
        import sys
        import time

        # Flush buffers
        sys.stdout.flush()
        sys.stderr.flush()

        # Close hardware connections safely if possible
        try:
             self.shutdown() # Clean shutdown (no re-init)
        except Exception as e:
             print(f"Error during shutdown: {e}")

        # os._exit(42) forces exit without cleanup/exception handling, guaranteeing the return code
        os._exit(42)

state = SystemState()
