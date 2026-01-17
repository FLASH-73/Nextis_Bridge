
import os
import json
import logging
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

class DatasetService:
    def __init__(self, base_path: str = "/home/roberto/nextis_app/datasets"):
        self.base_path = Path(base_path).expanduser()
        
        self.base_path.mkdir(parents=True, exist_ok=True)
        # We also want to support default lerobot cache if user wants?
        # For now, strict adherence to ~/datasets as requested.

    def list_datasets(self):
        """Scans the base path for LeRobot datasets."""
        datasets = []
        if not self.base_path.exists():
            return []

        # Walk looking for meta/info.json
        # We assume datasets are in subfolders of base_path
        # e.g. base_path/user/repo/meta/info.json
        
        for root, dirs, files in os.walk(self.base_path):
            if "meta" in dirs:
                # Potential dataset root
                dataset_root = Path(root)
                info_path = dataset_root / "meta" / "info.json"
                if info_path.exists():
                    try:
                        with open(info_path, "r") as f:
                            info = json.load(f)
                        
                        # Determine Repo ID relative to base
                        try:
                            rel_path = dataset_root.relative_to(self.base_path)
                            repo_id = str(rel_path)
                        except ValueError:
                            repo_id = dataset_root.name
                            
                        datasets.append({
                            "repo_id": repo_id,
                            "root": str(dataset_root),
                            "fps": info.get("fps"),
                            "robot_type": info.get("robot_type"),
                            "total_episodes": info.get("total_episodes", 0),
                            "total_frames": info.get("total_frames", 0),
                            "total_tasks": info.get("total_tasks", 0)
                        })
                    except Exception as e:
                        logger.warning(f"Failed to read dataset at {dataset_root}: {e}")
                        
        return datasets

    def get_dataset(self, repo_id: str):
        """Loads a dataset by repo_id."""
        # Security: Prevent traversing up
        if ".." in repo_id:
             raise ValueError("Invalid repo_id")
             
        full_path = self.base_path / repo_id
        if not full_path.exists():
             raise FileNotFoundError(f"Dataset {repo_id} not found at {full_path}")
             
        # Load LeRobotDataset
        # We use strict local loading
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=full_path,
            download_videos=False # Local only
        )
        return dataset

    def get_episode_data(self, repo_id: str, episode_index: int):
        """
        Returns cached data for visualization.
        """
        dataset = self.get_dataset(repo_id)
        
        # 1. Get Episode Info from Metadata
        # dataset.meta.episodes is usually a PyArrow Table or Pydict?
        # Let's inspect type or assume similar to HF dataset or dict
        episodes_table = dataset.meta.episodes
        
        # Check if it has 'length' column
        if "length" not in episodes_table:
             # Try converting to dict if it's a Table
             try:
                 episodes_table = episodes_table.to_pydict()
             except:
                 pass
        
        if episode_index >= len(episodes_table["index"]):
             pass # Allow fallback
             
        # Optimziation:
        import numpy as np
        lengths = np.array(episodes_table["length"])
        
        if episode_index < len(lengths):
            start_frame = int(np.sum(lengths[:episode_index]))
            length = int(lengths[episode_index])
        else:
            start_frame = 0
            length = 0
            
        end_frame = start_frame + length
        
        actions = []
        timestamps = []
        
        try:
             if length > 0:
                 chunk = dataset.hf_dataset[start_frame:end_frame]
                 actions = [x.tolist() for x in chunk.get("action", [])]
                 timestamps = chunk.get("timestamp", [])
        except Exception as e:
             logger.warning(f"Failed to load action data for episode {episode_index}: {e}")
        
        # 3. Construct Video Paths
        # LeRobot convention: videos/observation.images.key/chunk-XXX/file-YYY.mp4
        
        # We need to list the cameras (video keys)
        video_keys = [k for k in dataset.features if k.startswith("observation.images")]
        
        video_urls = {}
        for key in video_keys:
            video_urls[key] = f"/api/datasets/{repo_id}/video/{episode_index}/{key}"

        return {
            "index": episode_index,
            "length": length,
            "actions": actions,
            "timestamps": timestamps,
            "videos": video_urls
        }
    def delete_episode(self, repo_id: str, episode_index: int):
        """
        Deletes a specific episode:
        1. Removes from metadata (episodes.parquet).
        2. Renames subsequent video files to maintain contiguous indexing.
        3. Updates info.json total_episodes.
        4. Does NOT remove actual Action/Observation data from chunks (too risky/complex).
        """
        import pandas as pd
        import shutil
        
        dataset_root = self.base_path / repo_id
        if not dataset_root.exists():
             raise FileNotFoundError("Dataset not found")
             
        # 1. Load Episodes Metadata
        # We assume simplified structure (chunk-000/file-000) or we scan.
        # Check `meta/episodes` folder
        episodes_dir = dataset_root / "meta/episodes"
        # Find the specific parquet file? 
        # For simplicity, we assume single-chunk for MVP or we load ALL, concat, and rewrite to single chunk?
        # LeRobot v3 splits. If we rewrite, we might consolidate.
        # Let's try to find where the episode IS.
        
        # Load all into DF
        try:
            df = pd.read_parquet(episodes_dir)
        except:
             raise Exception("Failed to load episodes metadata")
             
        if episode_index not in df["index"].values and episode_index not in df["episode_index"].values:
             raise ValueError("Episode index not found")
             
        # Check total count
        total = len(df)
        
        # 2. Filter and Re-Index
        # Remove the row
        df = df[df["episode_index"] != episode_index]
        
        # Shift indices for subsequent episodes
        # mask = df["episode_index"] > episode_index
        # df.loc[mask, "episode_index"] -= 1
        
        # Note: We must also update 'index' column if it exists?
        # LeRobot 'index' usually means global frame index? No, episode_index is what matters.
        # But we need to shift episode_index to be contiguous.
        
        episodes_to_shift = df[df["episode_index"] > episode_index]["episode_index"].values
        # Create a mapping OLD -> NEW for video renaming
        # We iterate sorted (Ascending) so we process 5->4, 6->5...
        # Wait, if we rename 5->4, and 4 exists (it shouldn't if we process in order), it's fine.
        # But if we rename 6->5, and 5 was just renamed from 6... wait.
        # If we have 0, 1, 2(DEL), 3, 4.
        # We want 0, 1, 3->2, 4->3.
        # We should rename in ASCENDING order (3 then 4).
        # Rename 3 to 2. (2 is free because we deleted it).
        # Rename 4 to 3. (3 is free because we moved it to 2).
        
        shift_map = {old: old - 1 for old in sorted(episodes_to_shift)}
        
        # Apply shift to DF
        df.loc[df["episode_index"] > episode_index, "episode_index"] -= 1
        
        # 3. Write back Metadata
        # We overwrite the entire episodes structure with a SINGLE file to keep it simple and clean.
        # Backup old dir?
        # shutil.copytree(episodes_dir, episodes_dir.with_suffix(".bak"), dirs_exist_ok=True)
        
        # Danger: We are changing the structure from chunked to single file if we do this.
        # But it ensures consistency.
        # Remove old parquet files
        import shutil
        shutil.rmtree(episodes_dir)
        episodes_dir.mkdir()
        (episodes_dir / "chunk-000").mkdir()
        
        # Write new parquet
        output_path = episodes_dir / "chunk-000/file-000.parquet"
        df.to_parquet(output_path)
        
        # 4. Rename Video Files
        # Scan videos/
        video_root = dataset_root / "videos"
        if video_root.exists():
            for key_dir in video_root.iterdir():
                if key_dir.is_dir():
                    # Rename files in this key directory
                    # We expect `episode_XXXXXX.mp4`
                    # We use the shift_map
                    for old_idx, new_idx in shift_map.items():
                        old_file = key_dir / f"episode_{old_idx:06d}.mp4"
                        new_file = key_dir / f"episode_{new_idx:06d}.mp4"
                        
                        if old_file.exists():
                            old_file.rename(new_file)
                            
                    # Remove the deleted episode video
                    del_file = key_dir / f"episode_{episode_index:06d}.mp4"
                    if del_file.exists():
                        del_file.unlink()
                        
        # 5. Update info.json
        info_path = dataset_root / "meta/info.json"
        with open(info_path, "r") as f:
            info = json.load(f)
            
        info["total_episodes"] = len(df)
        # total_frames also decreases?
        # We should subtract the length of the deleted episode.
        # But we don't know it easily unless we looked it up before deleting.
        # df has 'length' usually.
        # We should have calculated total frames from DF sum.
        if "length" in df.columns:
            info["total_frames"] = int(df["length"].sum())
            
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
            
        return {"status": "success", "new_count": len(df)}
