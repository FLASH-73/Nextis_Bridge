import time
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.utils import make_robot_from_config

class DataRecorder:
    def __init__(self, repo_id, robot_type="so100_follower"):
        self.repo_id = repo_id
        # Use timestamp to separate recording sessions
        self.root = Path(f"data/raw/{int(time.time())}")
        self.fps = 30
        self.robot_type = robot_type
        self.dataset = None

    def start_new_episode(self, task_description):
        """
        Creates a new dataset container if one doesn't exist,
        or prepares the next episode.
        """
        if self.dataset is None:
            # Check if dataset already exists
            if (self.root / "meta/info.json").exists():
                print(f"Loading existing dataset from {self.root}")
                self.dataset = LeRobotDataset(self.repo_id, root=self.root)
            else:
                print(f"Creating new dataset at {self.root}")
                self.dataset = LeRobotDataset.create(
                    repo_id=self.repo_id,
                    fps=self.fps,
                    root=self.root,
                    robot_type=self.robot_type,
                    features={}, 
                    use_videos=True 
                )
        
        # If the dataset supports explicit episode starting, do it here.
        # For LeRobotDataset, adding frames automatically builds the episode,
        # and save_episode() finalizes it.
        print(f"Recording started for task: {task_description}")

    def save_frame(self, observation, action):
        """
        Call this 30 times per second
        """
        if self.dataset:
            # Flexible data mapping
            frame = {"action": action}
            
            # Flatten observation if needed or just pass through known keys
            # LeRobotDataset expects keys like "observation.state", "observation.images.phone"
            for key, value in observation.items():
                # If the key already starts with "observation.", use it
                if key.startswith("observation."):
                    frame[key] = value
                else:
                    # Otherwise assume it's a raw component
                    frame[f"observation.{key}"] = value
            
            self.dataset.add_frame(frame)

    def stop_episode(self):
        if self.dataset:
            self.dataset.save_episode()
            print("Episode saved.")

    def finish_session(self):
        if self.dataset:
            self.dataset.push_to_hub(private=True) # Uploads to HuggingFace
            print("Dataset uploaded to Hub!")