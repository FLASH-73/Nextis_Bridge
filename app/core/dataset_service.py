
import os
import json
import logging
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

# Compute project root relative to this file (app/core/dataset_service.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASETS_PATH = _PROJECT_ROOT / "datasets"

class DatasetService:
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = _DEFAULT_DATASETS_PATH
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
        Reads directly from disk to avoid HuggingFace Hub lookups.
        """
        import pandas as pd
        import numpy as np

        # Security: Prevent traversing up
        if ".." in repo_id:
            raise ValueError("Invalid repo_id")

        dataset_root = self.base_path / repo_id
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset {repo_id} not found at {dataset_root}")

        # 1. Load info.json directly
        info_path = dataset_root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Dataset info not found at {info_path}")

        with open(info_path, "r") as f:
            info = json.load(f)

        fps = info.get("fps", 30)
        features = info.get("features", {})
        total_episodes = info.get("total_episodes", 0)

        # Handle empty datasets (session started but no episodes saved)
        if total_episodes == 0:
            video_keys = [k for k in features.keys() if k.startswith("observation.images")]
            return {
                "index": episode_index,
                "length": 0,
                "actions": [],
                "timestamps": [],
                "videos": {},
                "fps": fps,
                "error": "Dataset has no episodes yet"
            }

        # 2. Load episodes metadata directly from parquet
        episodes_dir = dataset_root / "meta" / "episodes"
        episodes_df = None

        # Try directory first (pandas can read all parquet files in a directory)
        if episodes_dir.exists():
            try:
                episodes_df = pd.read_parquet(episodes_dir)
            except Exception as e:
                logger.warning(f"Failed to load episodes from directory: {e}")

        # Try single file fallback
        if episodes_df is None:
            single_file = episodes_dir / "chunk-000" / "file-000.parquet"
            if single_file.exists():
                try:
                    episodes_df = pd.read_parquet(single_file)
                except Exception as e:
                    logger.warning(f"Failed to load single episodes file: {e}")

        # If still no episodes data, return empty
        if episodes_df is None or len(episodes_df) == 0:
            return {
                "index": episode_index,
                "length": 0,
                "actions": [],
                "timestamps": [],
                "videos": {},
                "fps": fps,
                "error": "No episode metadata found"
            }

        # Determine index column
        if "episode_index" in episodes_df.columns:
            index_col = "episode_index"
        elif "index" in episodes_df.columns:
            index_col = "index"
        else:
            raise ValueError(f"No episode index column found. Columns: {list(episodes_df.columns)}")

        # Get episode length
        lengths = episodes_df["length"].values
        if episode_index < len(lengths):
            start_frame = int(np.sum(lengths[:episode_index]))
            length = int(lengths[episode_index])
        else:
            start_frame = 0
            length = 0

        end_frame = start_frame + length

        # 3. Load action data directly from parquet
        actions = []
        timestamps = []

        try:
            if length > 0:
                data_dir = dataset_root / "data"
                # Load all parquet files in data directory
                data_df = pd.read_parquet(data_dir)

                # Extract the relevant slice
                if len(data_df) > start_frame:
                    chunk_df = data_df.iloc[start_frame:end_frame]

                    # Get action columns
                    if "action" in chunk_df.columns:
                        actions = [x.tolist() if hasattr(x, 'tolist') else list(x) for x in chunk_df["action"].values]

                    # Get timestamps
                    if "timestamp" in chunk_df.columns:
                        timestamps = chunk_df["timestamp"].tolist()
        except Exception as e:
            logger.warning(f"Failed to load action data for episode {episode_index}: {e}")

        # 4. Construct Video URLs from features
        video_keys = [k for k in features.keys() if k.startswith("observation.images")]

        video_urls = {}
        for key in video_keys:
            video_urls[key] = f"/api/datasets/{repo_id}/video/{episode_index}/{key}"

        return {
            "index": episode_index,
            "length": length,
            "actions": actions,
            "timestamps": timestamps,
            "videos": video_urls,
            "fps": fps
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

    def delete_dataset(self, repo_id: str):
        """
        Deletes an entire dataset repository.
        """
        import shutil

        # Security: Prevent traversing up
        if ".." in repo_id:
            raise ValueError("Invalid repo_id")

        dataset_root = self.base_path / repo_id
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset {repo_id} not found")

        # Verify it's actually a dataset (has meta/info.json)
        info_path = dataset_root / "meta" / "info.json"
        if not info_path.exists():
            raise ValueError(f"{repo_id} does not appear to be a valid dataset")

        # Delete the entire directory tree
        shutil.rmtree(dataset_root)

        return {"status": "success", "message": f"Dataset {repo_id} deleted"}
