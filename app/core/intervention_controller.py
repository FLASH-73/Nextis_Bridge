import time
import threading
import numpy as np
from app.core.recorder import DataRecorder
from app.core.config import load_config

class InterventionEngine:
    def __init__(self, robot, recorder: DataRecorder, robot_lock=None):
        self.robot = robot
        self.recorder = recorder
        self.robot_lock = robot_lock
        
        # CONFIGURATION
        config = load_config()
        # You might want to add these specific params to settings.yaml later
        self.MOVE_THRESHOLD = config.get("intervention", {}).get("move_threshold", 0.05)
        self.IDLE_TIMEOUT = config.get("intervention", {}).get("idle_timeout", 2.0)
        
        # STATE
        self.is_human_controlling = False
        self.last_human_move_time = time.time()
        self.is_running = False
        
        # Cache for video streaming
        self.latest_observation = {}

    def get_leader_velocity(self):
        """
        Calculates velocity magnitude from the leader arm(s).
        Supports both single-arm and bimanual configurations.
        """
        if self.robot is None or not self.robot.is_connected:
             # print("DEBUG: Robot not connected in get_leader_velocity")
             return 0.0

        try:
            # Access the latest observation from the robot
            if self.robot_lock:
                with self.robot_lock:
                    observation = self.robot.get_observation()
            else:
                observation = self.robot.get_observation()
                
            self.latest_observation = observation # Update cache
            
            max_vel = 0.0
            
            # Check all potential velocity keys
            keys_to_check = [
                "observation.velocity", 
                "observation.velocity_left", 
                "observation.velocity_right"
            ]
            
            for key in keys_to_check:
                if key in observation:
                    val = np.linalg.norm(observation[key])
                    if val > max_vel:
                        max_vel = val
            
            return max_vel

        except Exception as e:
            msg = str(e)
            if "has no calibration registered" in msg:
                # Suppress spam for uncalibrated robot
                pass
            else:
                print(f"Error reading velocity: {e}")
            return 0.0

    def start_recording(self, task_description):
        print(f"External recording started: {task_description}")
        self.is_recording_externally = True
        self.recorder.start_new_episode(task_description)

    def stop_recording(self):
        print("External recording stopped.")
        self.is_recording_externally = False
        self.recorder.stop_episode()

    def start(self):
        self.is_running = True
        self.is_recording_externally = False # Reset flag
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def loop(self):
        print("System Active. Monitoring for human intervention...")
        while self.is_running:
            # print("DEBUG: InterventionEngine loop tick")
            # 1. SENSE
            human_velocity = self.get_leader_velocity()
            # self.latest_observation is updated in get_leader_velocity
            
            # 2. DECIDE
            if human_velocity > self.MOVE_THRESHOLD:
                if not self.is_human_controlling:
                    print("⚠️ HUMAN OVERRIDE ACTIVE - Recording Started")
                    self.is_human_controlling = True
                    self.recorder.start_new_episode("intervention_correction")
                
                self.last_human_move_time = time.time()
            
            elif (time.time() - self.last_human_move_time) > self.IDLE_TIMEOUT:
                if self.is_human_controlling:
                    print("🤖 Human stopped. Stopping recording & Handing control back to AI")
                    self.is_human_controlling = False
                    self.recorder.stop_episode()

            # 3. RECORD (If Human Controlling OR External Recording)
            should_record = self.is_human_controlling or getattr(self, 'is_recording_externally', False)
            
            if should_record and self.robot.is_connected:
                # Capture what the human (or robot) is doing
                try:
                    # For teleop/imitation, the action is often inferred from observation or leader
                    # If capture_action is not defined, use get_observation as a fallback for now
                    if hasattr(self.robot, 'capture_action'):
                        action = self.robot.capture_action()
                    else:
                        # Fallback: In teleop, the "action" is the current state of the follower (if following)
                        # or we might need to implement this properly later.
                        # For now, let's just use observation to prevent crash
                        action = self.robot.get_observation()
                    
                    # Ensure latest_observation is populated
                    if not self.latest_observation:
                         self.latest_observation = self.robot.get_observation()

                    self.recorder.save_frame(self.latest_observation, action)
                except Exception as e:
                    # print(f"Recording error: {e}") # Reduce spam
                    pass

            # 3. ACT (Inference)
            if not self.is_human_controlling:
                # Logic to run AI inference would go here
                pass
            
            time.sleep(1.0 / 30.0) # 30 Hz loop

if __name__ == "__main__":
    print("Initializing Intervention Engine...")
    # Example usage (commented out requires real hardware)
    # robot = make_robot_from_config("so100_follower")
    # robot.connect()
    # recorder = DataRecorder(repo_id="roberto/nextis_test", robot_type="so100_follower")
    # engine = InterventionEngine(robot, recorder)
    # engine.start()