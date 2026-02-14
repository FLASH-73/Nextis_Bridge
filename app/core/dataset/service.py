"""Dataset browsing, episode inspection, and CRUD operations."""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from app.core.config import DATASETS_DIR
from app.core.dataset.merge import MergeJobManager, MergeValidationResult, MergeJob

logger = logging.getLogger(__name__)


class DatasetService:
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = DATASETS_DIR
        self.base_path = Path(base_path).expanduser()

        self.base_path.mkdir(parents=True, exist_ok=True)

        # Merge functionality delegated to MergeJobManager
        self._merge_manager = MergeJobManager(self.base_path)

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

        # CRITICAL: Sort by episode_index to ensure array indexing matches episode IDs
        # After deletion and renumbering, episodes should be contiguous (0, 1, 2...)
        # but the parquet row order may not match
        episodes_df = episodes_df.sort_values(index_col).reset_index(drop=True)

        # Get episode length - now array indexing matches episode_index
        lengths = episodes_df["length"].values
        if episode_index < len(lengths):
            start_frame = int(np.sum(lengths[:episode_index]))
            length = int(lengths[episode_index])
        else:
            # Episode index out of range
            start_frame = 0
            length = 0
            logger.warning(f"Episode {episode_index} not found. Available: 0-{len(lengths)-1}")

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

        # Get the episode row for timestamp info
        ep_row = episodes_df[episodes_df[index_col] == episode_index] if episodes_df is not None else None

        video_urls = {}
        video_metadata = {}
        for key in video_keys:
            video_urls[key] = f"/api/datasets/{repo_id}/video/{episode_index}/{key}"
            # Check if this is a depth video (by metadata flag or key name)
            feature_info = features.get(key, {})
            is_depth = (
                feature_info.get("info", {}).get("video.is_depth_map", False) or
                key.endswith("_depth")
            )

            # Get timestamp info for seeking in concatenated videos
            # LeRobot stores all episodes in one file, tracked by from_timestamp/to_timestamp
            from_ts = 0.0
            to_ts = None
            if ep_row is not None and not ep_row.empty:
                from_ts_col = f"videos/{key}/from_timestamp"
                to_ts_col = f"videos/{key}/to_timestamp"
                if from_ts_col in ep_row.columns:
                    from_ts = float(ep_row[from_ts_col].iloc[0])
                if to_ts_col in ep_row.columns:
                    to_ts = float(ep_row[to_ts_col].iloc[0])

            video_metadata[key] = {
                "is_depth": is_depth,
                "from_timestamp": from_ts,
                "to_timestamp": to_ts,
            }

        return {
            "index": episode_index,
            "length": length,
            "actions": actions,
            "timestamps": timestamps,
            "videos": video_urls,
            "video_metadata": video_metadata,
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
        episodes_dir = dataset_root / "meta/episodes"

        # Load all into DF
        try:
            df = pd.read_parquet(episodes_dir)
        except:
             raise Exception("Failed to load episodes metadata")

        # Only check episode_index column (the standard LeRobot v3 column name)
        if "episode_index" not in df.columns:
            raise ValueError(f"No episode_index column found. Columns: {list(df.columns)}")

        if episode_index not in df["episode_index"].values:
            raise ValueError(f"Episode index {episode_index} not found in dataset")

        # Check total count
        total = len(df)

        # 2. Filter and Re-Index
        # Remove the row
        df = df[df["episode_index"] != episode_index]

        episodes_to_shift = df[df["episode_index"] > episode_index]["episode_index"].values
        # Create a mapping OLD -> NEW for video renaming
        # Rename in ASCENDING order: 3->2, 4->3, etc.
        shift_map = {old: old - 1 for old in sorted(episodes_to_shift)}

        # Apply shift to DF
        df.loc[df["episode_index"] > episode_index, "episode_index"] -= 1

        # 3. Write back Metadata
        import shutil
        shutil.rmtree(episodes_dir)
        episodes_dir.mkdir()
        (episodes_dir / "chunk-000").mkdir()

        # Sort by episode_index before writing to ensure consistent ordering
        df = df.sort_values("episode_index").reset_index(drop=True)

        # Write new parquet
        output_path = episodes_dir / "chunk-000/file-000.parquet"
        df.to_parquet(output_path)

        # 4. Rename Video Files
        video_root = dataset_root / "videos"
        if video_root.exists():
            for key_dir in video_root.iterdir():
                if key_dir.is_dir():
                    # LeRobot v3 uses chunk directories with file-XXX.mp4 format
                    chunk_dir = key_dir / "chunk-000"  # Single chunk for now
                    if chunk_dir.exists():
                        # Delete the episode's video file FIRST (before renaming)
                        del_file = chunk_dir / f"file-{episode_index:03d}.mp4"
                        if del_file.exists():
                            del_file.unlink()
                            logger.info(f"Deleted video: {del_file}")

                        # Rename subsequent video files to fill the gap
                        for old_idx, new_idx in shift_map.items():
                            old_file = chunk_dir / f"file-{old_idx:03d}.mp4"
                            new_file = chunk_dir / f"file-{new_idx:03d}.mp4"
                            if old_file.exists():
                                old_file.rename(new_file)
                                logger.info(f"Renamed video: {old_file} -> {new_file}")

        # 5. Update info.json
        info_path = dataset_root / "meta/info.json"
        with open(info_path, "r") as f:
            info = json.load(f)

        info["total_episodes"] = len(df)
        if "length" in df.columns:
            info["total_frames"] = int(df["length"].sum())

        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

        return {"status": "success", "new_count": len(df)}

    def delete_dataset(self, repo_id: str):
        """Deletes an entire dataset repository."""
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

    # ── Merge delegation (preserves API) ──

    def validate_merge(self, repo_ids: List[str]) -> MergeValidationResult:
        return self._merge_manager.validate_merge(repo_ids)

    def start_merge_job(self, repo_ids: List[str], output_repo_id: str) -> MergeJob:
        return self._merge_manager.start_merge_job(repo_ids, output_repo_id)

    def get_merge_job_status(self, job_id: str) -> Optional[MergeJob]:
        return self._merge_manager.get_merge_job_status(job_id)
