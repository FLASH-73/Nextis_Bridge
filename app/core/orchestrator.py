import threading
import time
from typing import Dict, List

from app.core.intervention import InterventionEngine
from app.core.recorder import DataRecorder


class TaskOrchestrator:
    def __init__(self, robot, recorder: DataRecorder, robot_lock=None):
        self.robot = robot
        self.recorder = recorder
        self.intervention_engine = InterventionEngine(robot, recorder, robot_lock=robot_lock)

        self.task_chain = []
        self.current_task_index = 0
        self.models = {} # Cache for loaded models
        self.active_policy = None
        self.is_running = False

        # Execution State
        self.is_executing_plan = False
        self.current_plan = []
        self.task_start_time = 0
        self.TASK_DURATION = 5.0 # Seconds per task for simulation

    def load_task_chain(self, tasks: List[str]):
        """
        Receives a list of subtasks from the High-Level Planner (LLM).
        """
        self.task_chain = tasks
        self.current_task_index = 0
        print(f"Task chain loaded: {self.task_chain}")

    def load_model_for_task(self, task_name: str):
        """
        Simulates loading a Diffusion Policy model for a specific task.
        """
        if task_name not in self.models:
            print(f"Loading model for task: {task_name}...")
            # self.models[task_name] = DiffusionPolicy.load(task_name)
            self.models[task_name] = "dummy_model_object"
            time.sleep(0.1) # Simulate load time (minimal to not block too much)
        return self.models[task_name]

    def deploy_policy(self, checkpoint_path: str):
        """
        Load a trained policy from a checkpoint for autonomous execution.

        Args:
            checkpoint_path: Path to the pretrained_model directory containing
                            config.json and model.safetensors
        """
        print(f"Deploying policy from: {checkpoint_path}")

        import json
        from pathlib import Path

        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

        # Read config.json to get policy type
        config_path = checkpoint / "config.json"
        if not config_path.exists():
            raise ValueError(f"config.json not found in checkpoint: {checkpoint_path}")

        with open(config_path) as f:
            config = json.load(f)
        policy_type = config.get("type", "diffusion")
        print(f"  Policy type: {policy_type}")

        # Import LeRobot and get policy class
        import sys
        lerobot_path = Path(__file__).parent.parent.parent / "lerobot" / "src"
        if str(lerobot_path) not in sys.path:
            sys.path.insert(0, str(lerobot_path))

        from lerobot.policies.factory import get_policy_class
        policy_cls = get_policy_class(policy_type)

        # Load pretrained policy using the correct API
        # from_pretrained() loads config automatically and returns policy in eval mode
        print(f"  Loading {policy_cls.__name__} from {checkpoint}...")
        self.deployed_policy = policy_cls.from_pretrained(str(checkpoint))
        self.deployed_policy_path = checkpoint_path
        print(f"  ✅ Policy loaded successfully (device: {self.deployed_policy.config.device})")

        return True

    def start(self):
        self.is_running = True
        self.intervention_engine.start() # Start monitoring for human input

        # Start the main inference loop in a separate thread
        self.inference_thread = threading.Thread(target=self._inference_loop)
        self.inference_thread.start()

    def stop(self):
        self.is_running = False
        self.is_executing_plan = False
        self.intervention_engine.stop()
        if hasattr(self, 'inference_thread') and self.inference_thread.is_alive():
            self.inference_thread.join()
        print("Orchestrator Stopped.")

    def execute_plan(self, plan):
        """
        Non-blocking call to start plan execution.
        """
        print(f"Starting Plan Execution: {plan}")
        self.current_plan = plan
        self.current_task_index = 0
        self.is_executing_plan = True

        # Parse first task
        if self.current_plan:
            first_task = self.current_plan[0].get("task")
            self.active_policy = self.load_model_for_task(first_task)
            self.task_start_time = time.time()

            # Start Recording
            plan_desc = f"execution_{int(time.time())}"
            self.intervention_engine.start_recording(plan_desc)
        else:
             print("Empty plan provided.")
             self.is_executing_plan = False

    def _inference_loop(self):
        print("Starting Inference Loop...")
        while self.is_running:
            # 1. Check if Human is Intervening
            if self.intervention_engine.is_human_controlling:
                # If human is controlling, we pause "autonomous" execution logic
                # The InterventionEngine handles the recording of human actions
                # We might want to PAUSE the plan timer here or just let it drift
                time.sleep(0.01)
                continue

            # 2. Autonomous Execution Logic
            if self.is_executing_plan:
                # Check for Task Completion (Mocked by time for now)
                elapsed = time.time() - self.task_start_time

                if elapsed > self.TASK_DURATION:
                    # Task Complete
                    print(f"Task {self.current_task_index} Complete.")
                    self.current_task_index += 1

                    if self.current_task_index < len(self.current_plan):
                        # Load Next Task
                        next_step = self.current_plan[self.current_task_index]
                        task_name = next_step.get("task")
                        print(f"--> Advancing to: {task_name}")
                        self.active_policy = self.load_model_for_task(task_name)
                        self.task_start_time = time.time()
                    else:
                        # Assessment Complete
                        print("All tasks completed successfully.")
                        self.is_executing_plan = False
                        self.active_policy = None
                        self.intervention_engine.stop_recording()

                # Run Inference (Simulated)
                if self.active_policy and self.robot.is_connected:
                    try:
                        # In a real implementation:
                        # obs = self.robot.get_observation()
                        # action = self.active_policy.select_action(obs)
                        # self.robot.send_action(action)

                        # Record Autonomous Frame (Success Data)
                        # We use the intervention engine's recorder, but mark it as autonomous?
                        # For now, InterventionEngine stops recording when NOT human controlling unless we force it.
                        # We forced it in execute_plan via start_recording.

                        # We need to explicitly record frame here because InterventionEngine loop
                        # mainly checks for Human velocity.

                        if hasattr(self.intervention_engine, 'is_recording_externally') and self.intervention_engine.is_recording_externally:
                             # Capture observation/action for training
                             obs = self.robot.get_observation()
                             # active_policy would give action, here we mock
                             action = getattr(self.robot, 'mock_action', None)
                             # If robot is real, we need real action.
                             # For now, let's just record the state as 'action' (holding position) if no policy output
                             if action is None:
                                 # Use current joint positions as action (holding)
                                 action = obs.get("observation.state", [])

                             self.recorder.save_frame(obs, action)

                    except Exception:
                        # print(f"Inference Error: {e}")
                        pass

            time.sleep(1.0 / 30.0) # 30Hz Control Loop
