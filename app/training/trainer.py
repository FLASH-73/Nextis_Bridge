import sys
from pathlib import Path
import subprocess
from app.core.config import load_config

# Path to LeRobot's train script
LEROBOT_PATH = Path("lerobot/src/lerobot/scripts/train.py")

class TrainingJob:
    def __init__(self, dataset_repo_id, job_name):
        self.dataset_repo_id = dataset_repo_id
        self.job_name = job_name
        self.config = load_config()
        self.process = None

    def start(self):
        """
        Starts the training process using LeRobot's script.
        """
        training_cfg = self.config.get("training", {})
        policy_type = training_cfg.get("policy", "diffusion")
        batch_size = training_cfg.get("batch_size", 8)
        steps = training_cfg.get("steps", 5000)

        # Construct command
        # python lerobot/scripts/train.py \
        #   policy=diffusion \
        #   dataset_repo_id=... \
        #   env=... \
        #   ...
        
        cmd = [
            sys.executable, str(LEROBOT_PATH),
            f"policy={policy_type}",
            f"dataset_repo_id={self.dataset_repo_id}",
            f"training.batch_size={batch_size}",
            f"training.offline_steps={steps}",
            f"training.online_steps=0", # We are doing offline training first
            f"training.eval_freq=1000",
            f"training.save_freq=1000",
            f"training.save_model=true",
            f"hydra.run.dir=outputs/train/{self.job_name}"
        ]
        
        # Note: We might need to specify the environment or robot config depending on LeRobot's requirements
        # For now, we assume the dataset contains the necessary metadata.
        
        print(f"Starting training job: {self.job_name}")
        print(f"Command: {' '.join(cmd)}")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
    def get_status(self):
        if self.process is None:
            return "NOT_STARTED"
        
        ret = self.process.poll()
        if ret is None:
            return "RUNNING"
        elif ret == 0:
            return "COMPLETED"
        else:
            return "FAILED"

    def get_logs(self):
        if self.process:
            # This is a simple non-blocking read, might need refinement for real-time streaming
            return self.process.stdout.read()
        return ""

if __name__ == "__main__":
    # Example usage
    job = TrainingJob("roberto/nextis_test", "test_run_01")
    # job.start() # Commented out to avoid auto-running heavy compute
    print("Training Job Initialized")
