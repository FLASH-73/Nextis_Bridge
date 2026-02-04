
import os
import json
import logging
import threading
import uuid
from pathlib import Path
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.aggregate import aggregate_datasets, validate_all_metadata

logger = logging.getLogger(__name__)


class MergeJobStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MergeJob:
    job_id: str
    status: MergeJobStatus
    repo_ids: List[str]
    output_repo_id: str
    progress: Dict[str, Any] = field(default_factory=lambda: {"percent": 0, "message": "Pending..."})
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class MergeValidationResult:
    compatible: bool
    datasets: List[Dict[str, Any]]
    merged_info: Optional[Dict[str, Any]]
    errors: List[Dict[str, str]]
    warnings: List[str] = field(default_factory=list)

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

        # Merge job tracking
        self._merge_jobs: Dict[str, MergeJob] = {}
        self._merge_lock = threading.Lock()

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
        
        # Sort by episode_index before writing to ensure consistent ordering
        df = df.sort_values("episode_index").reset_index(drop=True)

        # Write new parquet
        output_path = episodes_dir / "chunk-000/file-000.parquet"
        df.to_parquet(output_path)
        
        # 4. Rename Video Files
        # Scan videos/
        # LeRobot v3 format: videos/{camera_key}/chunk-XXX/file-XXX.mp4
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

    # ==================== MERGE FUNCTIONALITY ====================

    def validate_merge(self, repo_ids: List[str]) -> MergeValidationResult:
        """Validate that datasets can be merged (same fps, robot_type, features)."""
        if len(repo_ids) < 2:
            return MergeValidationResult(
                compatible=False,
                datasets=[],
                merged_info=None,
                errors=[{"type": "insufficient_datasets", "message": "Need at least 2 datasets to merge"}],
                warnings=[]
            )

        datasets_info = []
        all_metadata = []
        errors = []
        warnings = []

        # Load metadata for each dataset
        for repo_id in repo_ids:
            # Security check
            if ".." in repo_id:
                errors.append({"type": "invalid_repo_id", "message": f"Invalid repo_id: {repo_id}"})
                continue

            try:
                full_path = self.base_path / repo_id
                if not full_path.exists():
                    errors.append({"type": "not_found", "message": f"Dataset not found: {repo_id}"})
                    continue

                meta = LeRobotDatasetMetadata(repo_id=repo_id, root=full_path)
                all_metadata.append(meta)
                datasets_info.append({
                    "repo_id": repo_id,
                    "fps": meta.fps,
                    "robot_type": meta.robot_type,
                    "features": list(meta.features.keys()),
                    "total_episodes": meta.total_episodes,
                    "total_frames": meta.total_frames
                })
            except Exception as e:
                errors.append({
                    "type": "load_error",
                    "message": f"Failed to load {repo_id}: {str(e)}"
                })

        if errors:
            return MergeValidationResult(
                compatible=False,
                datasets=datasets_info,
                merged_info=None,
                errors=errors,
                warnings=warnings
            )

        # Validate compatibility using LeRobot's validation
        try:
            fps, robot_type, features = validate_all_metadata(all_metadata)

            # Calculate merged totals
            merged_info = {
                "total_episodes": sum(m.total_episodes for m in all_metadata),
                "total_frames": sum(m.total_frames for m in all_metadata),
                "fps": fps,
                "robot_type": robot_type,
                "features": list(features.keys())
            }

            # Also validate video files exist
            video_valid, missing_videos = self._validate_video_files(repo_ids)
            if not video_valid:
                errors.append({
                    "type": "missing_videos",
                    "message": f"Missing {len(missing_videos)} video file(s): " + ", ".join(missing_videos[:3]) + ("..." if len(missing_videos) > 3 else "")
                })
                return MergeValidationResult(
                    compatible=False,
                    datasets=datasets_info,
                    merged_info=merged_info,  # Still include merged_info for display
                    errors=errors,
                    warnings=warnings
                )

            return MergeValidationResult(
                compatible=True,
                datasets=datasets_info,
                merged_info=merged_info,
                errors=[],
                warnings=warnings
            )

        except ValueError as e:
            # Parse the error message to provide structured feedback
            error_msg = str(e)
            errors.append({
                "type": "compatibility_error",
                "message": error_msg
            })

            return MergeValidationResult(
                compatible=False,
                datasets=datasets_info,
                merged_info=None,
                errors=errors,
                warnings=warnings
            )

    def start_merge_job(self, repo_ids: List[str], output_repo_id: str) -> MergeJob:
        """Start a background merge job."""
        job_id = f"merge_{uuid.uuid4().hex[:8]}"

        job = MergeJob(
            job_id=job_id,
            status=MergeJobStatus.PENDING,
            repo_ids=repo_ids,
            output_repo_id=output_repo_id,
            progress={"percent": 0, "message": "Starting merge..."},
            error=None,
            created_at=datetime.now(),
            completed_at=None
        )

        with self._merge_lock:
            self._merge_jobs[job_id] = job

        # Start background thread
        thread = threading.Thread(
            target=self._execute_merge_job,
            args=(job_id,),
            daemon=True
        )
        thread.start()

        return job

    def _validate_video_files(self, repo_ids: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate all video files exist for the given datasets.
        Returns (is_valid, list_of_missing_files).
        """
        missing_files = []

        for repo_id in repo_ids:
            dataset_path = self.base_path / repo_id
            meta_path = dataset_path / "meta" / "info.json"

            if not meta_path.exists():
                missing_files.append(f"{repo_id}: missing meta/info.json")
                continue

            try:
                with open(meta_path) as f:
                    info = json.load(f)
            except Exception as e:
                missing_files.append(f"{repo_id}: failed to read info.json - {e}")
                continue

            # Get features to find video keys
            features = info.get("features", {})
            video_keys = [k for k in features if features[k].get("dtype") == "video"]

            if not video_keys:
                continue  # No videos in this dataset

            # Load episodes metadata to find video file references
            episodes_dir = dataset_path / "meta" / "episodes"
            if not episodes_dir.exists():
                continue  # No episodes metadata

            try:
                episodes_df = pd.read_parquet(episodes_dir)
            except Exception as e:
                missing_files.append(f"{repo_id}: failed to read episodes metadata - {e}")
                continue

            # Track unique video files to check (avoid checking same file multiple times)
            checked_files = set()

            for video_key in video_keys:
                # Extract camera name from feature key (e.g., "observation.images.camera_1" -> "camera_1")
                camera_name = video_key.replace("observation.images.", "")
                chunk_col = f"videos/{camera_name}/chunk_index"
                file_col = f"videos/{camera_name}/file_index"

                if chunk_col not in episodes_df.columns or file_col not in episodes_df.columns:
                    continue

                # Get unique (chunk, file) pairs
                unique_pairs = set(zip(
                    episodes_df[chunk_col].astype(int),
                    episodes_df[file_col].astype(int)
                ))

                for chunk_idx, file_idx in unique_pairs:
                    video_path = dataset_path / "videos" / camera_name / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
                    file_key = str(video_path)

                    if file_key in checked_files:
                        continue
                    checked_files.add(file_key)

                    if not video_path.exists():
                        rel_path = f"videos/{camera_name}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
                        missing_files.append(f"{repo_id}: {rel_path}")

        return len(missing_files) == 0, missing_files

    def _execute_merge_job(self, job_id: str):
        """Execute merge in background thread."""
        job = self._merge_jobs.get(job_id)
        if not job:
            return

        try:
            with self._merge_lock:
                job.status = MergeJobStatus.IN_PROGRESS
                job.progress = {"percent": 5, "message": "Validating video files..."}

            # PRE-MERGE VALIDATION: Check all source video files exist
            valid, missing = self._validate_video_files(job.repo_ids)
            if not valid:
                error_msg = "Missing video files in source datasets:\n" + "\n".join(missing[:10])
                if len(missing) > 10:
                    error_msg += f"\n...and {len(missing) - 10} more"
                raise ValueError(error_msg)

            with self._merge_lock:
                job.progress = {"percent": 10, "message": "Validating datasets..."}

            # Build paths
            roots = [self.base_path / repo_id for repo_id in job.repo_ids]
            output_path = self.base_path / job.output_repo_id

            # Check output doesn't exist
            if output_path.exists():
                raise ValueError(f"Output dataset '{job.output_repo_id}' already exists")

            with self._merge_lock:
                job.progress = {"percent": 20, "message": "Merging datasets..."}

            # Call LeRobot's aggregate_datasets
            aggregate_datasets(
                repo_ids=job.repo_ids,
                aggr_repo_id=job.output_repo_id,
                roots=roots,
                aggr_root=output_path
            )

            # POST-MERGE VALIDATION: Verify merged dataset has all videos
            with self._merge_lock:
                job.progress = {"percent": 90, "message": "Validating merged dataset..."}

            merged_valid, merged_missing = self._validate_video_files([job.output_repo_id])
            if not merged_valid:
                error_msg = "Merged dataset has missing video files:\n" + "\n".join(merged_missing[:10])
                if len(merged_missing) > 10:
                    error_msg += f"\n...and {len(merged_missing) - 10} more"
                # Clean up failed merge output
                import shutil
                if output_path.exists():
                    shutil.rmtree(output_path)
                raise ValueError(error_msg)

            with self._merge_lock:
                job.status = MergeJobStatus.COMPLETED
                job.progress = {"percent": 100, "message": "Merge complete!"}
                job.completed_at = datetime.now()

            logger.info(f"Merge job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Merge job {job_id} failed: {e}")
            import traceback
            traceback.print_exc()

            with self._merge_lock:
                job.status = MergeJobStatus.FAILED
                job.error = str(e)
                job.progress = {"percent": 0, "message": f"Failed: {str(e)}"}
                job.completed_at = datetime.now()

    def get_merge_job_status(self, job_id: str) -> Optional[MergeJob]:
        """Get status of a merge job."""
        return self._merge_jobs.get(job_id)
