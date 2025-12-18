import time
import threading
from typing import List, Dict
from app.core.intervention_controller import InterventionEngine
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
        
    def load_task_chain(self, tasks: List[str]):
        """
        Receives a list of subtasks from the High-Level Planner (LLM).
        Example: ["pick_apple", "move_to_box", "place_apple"]
        """
        self.task_chain = tasks
        self.current_task_index = 0
        print(f"Task chain loaded: {self.task_chain}")

    def load_model_for_task(self, task_name: str):
        """
        Simulates loading a Diffusion Policy model for a specific task.
        In reality, this would load weights from disk/hub.
        """
        if task_name not in self.models:
            print(f"Loading model for task: {task_name}...")
            # self.models[task_name] = DiffusionPolicy.load(task_name)
            self.models[task_name] = "dummy_model_object" 
            time.sleep(0.5) # Simulate load time
        return self.models[task_name]

    def get_current_policy(self):
        if self.current_task_index >= len(self.task_chain):
            return None
        
        current_task = self.task_chain[self.current_task_index]
        return self.load_model_for_task(current_task)

    def advance_task(self):
        """
        Called when a subtask is completed (either by success detection or manual trigger).
        """
        self.current_task_index += 1
        if self.current_task_index < len(self.task_chain):
            print(f"Advancing to next task: {self.task_chain[self.current_task_index]}")
            self.active_policy = self.get_current_policy()
        else:
            print("All tasks completed!")
            self.active_policy = None
            self.stop()

    def start(self):
        self.is_running = True
        self.intervention_engine.start() # Start monitoring for human input
        self.active_policy = self.get_current_policy()
        
        # Start the main inference loop in a separate thread
        self.inference_thread = threading.Thread(target=self._inference_loop)
        self.inference_thread.start()

    def stop(self):
        self.is_running = False
        self.intervention_engine.stop()
        if hasattr(self, 'inference_thread') and self.inference_thread.is_alive():
            self.inference_thread.join()
        print("Orchestrator Stopped.")

    def execute_plan(self, plan):
        """
        Executes a list of high-level tasks sequentially.
        """
        print(f"Executing Plan: {plan}")
        self.current_plan = plan
        self.is_executing_plan = True
        
        # Start Recording Data for Training
        plan_desc = f"execution_{int(time.time())}"
        self.intervention_engine.start_recording(plan_desc)
        
        try:
            for step in plan:
                if not self.is_running: # Use self.is_running for consistency
                    break
                    
                task_name = step.get("task")
                params = step.get("params", {})
                
                print(f"--> Executing Step: {task_name} with {params}")
                
                # Simulate execution time
                # In a real scenario, this would call self.robot.move_to(...) etc.
                time.sleep(2) 
                
                # TODO: Integrate with real robot primitives here
                # if task_name == 'move_to_bin': ...
                
        finally:
            # Ensure we stop recording even if error occurs
            self.intervention_engine.stop_recording()
            self.is_executing_plan = False
            print("Plan Execution Complete.")

    def _inference_loop(self):
        print("Starting Inference Loop...")
        while self.is_running:
            # 1. Check if Human is Intervening
            if self.intervention_engine.is_human_controlling:
                # If human is controlling, we pause inference and just wait
                # The InterventionEngine handles the recording
                time.sleep(0.01)
                continue

            # 2. Run Inference
            if self.active_policy and self.robot.is_connected:
                try:
                    # In a real implementation:
                    # obs = self.robot.get_observation()
                    # action = self.active_policy.select_action(obs)
                    # self.robot.send_action(action)
                    
                    # For now, we simulate "doing work"
                    # If we are recording execution data (autonomous mode), save it
                    if self.is_executing_plan:
                         # Capture data for training (autonomous success data)
                         # self.intervention_engine.record_frame(autonomous=True)
                         pass

                except Exception as e:
                    print(f"Inference Error: {e}")
            
            # 3. Check for Task Completion (Mocked)
            # In reality, we might check a sensor or VLM here
            # if check_success(obs):
            #     self.advance_task()
            
            time.sleep(1.0 / 30.0) # 30Hz Control Loop
