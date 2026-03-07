"""Extract empirical start pose from a checkpoint's training dataset."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

logger = logging.getLogger(__name__)

HIGH_STD_THRESHOLD = 0.15  # rad — flag joints above this


@dataclass
class StartPoseResult:
    """Mean start pose extracted from training episodes."""

    mean: Dict[str, float]
    std: Dict[str, float]
    num_episodes: int
    warnings: List[str] = field(default_factory=list)


def extract_start_pose(checkpoint_path: Union[str, Path]) -> StartPoseResult:
    """Extract mean start pose from the training dataset of a checkpoint.

    Reads the first frame (frame_index == 0) of observation.state from every
    episode, filters to .pos joints only, and returns per-joint mean/std.

    Args:
        checkpoint_path: Path to the pretrained_model checkpoint directory
            (the one containing train_config.json).

    Returns:
        StartPoseResult with per-joint mean, std, episode count, and warnings
        for joints with std > 0.15 rad.

    Raises:
        FileNotFoundError: If train_config.json, dataset, or data files missing.
        ValueError: If dataset lacks observation.state or has no .pos joints.
    """
    cp = Path(checkpoint_path)

    # 1. Resolve dataset root from checkpoint metadata
    train_config_path = cp / "train_config.json"
    if not train_config_path.exists():
        raise FileNotFoundError(
            f"train_config.json not found at {train_config_path}"
        )

    with open(train_config_path) as f:
        train_config = json.load(f)

    dataset_root = train_config.get("dataset", {}).get("root")
    if not dataset_root:
        raise ValueError(
            f"dataset.root not found in {train_config_path}"
        )

    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {dataset_root}"
        )

    # 2. Load joint names from info.json
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info.json not found at {info_path}")

    with open(info_path) as f:
        info = json.load(f)

    all_names = (
        info.get("features", {})
        .get("observation.state", {})
        .get("names")
    )
    if not all_names:
        raise ValueError(
            f"No observation.state names in {info_path}"
        )

    # Filter to .pos joints and record indices
    pos_indices = [i for i, n in enumerate(all_names) if n.endswith(".pos")]
    pos_names = [all_names[i] for i in pos_indices]
    if not pos_names:
        raise ValueError(
            f"No .pos joints found in state names: {all_names}"
        )

    # 3. Read first frames from parquet files
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet data files in {data_dir}")

    start_states = []
    for pf in parquet_files:
        table = pq.read_table(
            pf, columns=["observation.state", "frame_index"]
        )
        mask = pc.equal(table.column("frame_index"), 0)
        first_frames = table.filter(mask)
        for row in first_frames.column("observation.state"):
            start_states.append(row.as_py())

    if not start_states:
        raise ValueError("No episode start frames (frame_index==0) found")

    # 4. Compute mean/std over .pos joints
    all_states = np.array(start_states, dtype=np.float32)
    pos_states = all_states[:, pos_indices]

    mean_values = pos_states.mean(axis=0)
    std_values = pos_states.std(axis=0)

    mean_dict = {name: float(mean_values[j]) for j, name in enumerate(pos_names)}
    std_dict = {name: float(std_values[j]) for j, name in enumerate(pos_names)}

    # 5. Flag high-variance joints
    warnings = []
    for j, name in enumerate(pos_names):
        if std_values[j] > HIGH_STD_THRESHOLD:
            warnings.append(
                f"Joint '{name}' has high start-pose variance: "
                f"std={std_values[j]:.3f} rad (threshold={HIGH_STD_THRESHOLD})"
            )

    if warnings:
        for w in warnings:
            logger.warning("START POSE: %s", w)

    logger.info(
        "Extracted start pose from %d episodes (%d joints): %s",
        len(start_states),
        len(pos_names),
        {k: f"{v:+.3f}" for k, v in mean_dict.items()},
    )

    return StartPoseResult(
        mean=mean_dict,
        std=std_dict,
        num_episodes=len(start_states),
        warnings=warnings,
    )
