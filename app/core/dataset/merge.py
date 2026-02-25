"""Merge validation and background merge job execution for LeRobot datasets."""

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from lerobot.datasets.aggregate import aggregate_datasets, validate_all_metadata
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

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


class MergeJobManager:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self._merge_jobs: Dict[str, MergeJob] = {}
        self._merge_lock = threading.Lock()

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
