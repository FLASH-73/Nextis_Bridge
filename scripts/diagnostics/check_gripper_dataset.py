#!/usr/bin/env python3
"""Diagnostic analysis of gripper channel in a LeRobot v3 dataset.

Inspects both the action gripper (leader command) and observation.state
gripper (follower feedback) to determine whether gripper values were
recorded correctly during teleoperation.

Usage:
    python scripts/diagnostics/check_gripper_dataset.py
    python scripts/diagnostics/check_gripper_dataset.py --dataset Bearing_pickup_cell/v4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """Walk up from this script to find the project root (contains app/ and datasets/)."""
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / "app").is_dir() and (p / "datasets").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError("Could not locate project root (expected app/ and datasets/ dirs)")


def find_dataset(root: Path, rel_path: str) -> Path:
    """Resolve dataset path, trying case variations."""
    candidates = [
        root / "datasets" / rel_path,
        root / "datasets" / rel_path.replace("bearing", "Bearing"),
        root / "datasets" / rel_path.replace("Bearing", "bearing"),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    # Fallback: glob for partial match
    parts = rel_path.split("/")
    pattern = f"*{parts[0].lower().replace('_', '*')}*"
    for d in (root / "datasets").glob(pattern):
        version_dir = d / parts[1] if len(parts) > 1 else d
        if version_dir.is_dir():
            return version_dir
    raise FileNotFoundError(
        f"Dataset not found. Tried: {[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_parquet(dataset_dir: Path) -> pd.DataFrame:
    """Load all parquet data chunks from a LeRobot v3 dataset."""
    data_dir = dataset_dir / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} frames from {len(parquet_files)} parquet file(s)")
    return df


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_gripper(df: pd.DataFrame, column: str, index: int) -> np.ndarray:
    """Extract a single dimension from a list/array column in the dataframe."""
    sample = df[column].iloc[0]
    if isinstance(sample, (list, np.ndarray)):
        return np.array([row[index] for row in df[column]], dtype=np.float64)
    else:
        raise TypeError(
            f"Column '{column}' has unexpected type {type(sample)}. "
            f"Expected list or ndarray."
        )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_episode(
    action_gripper: np.ndarray,
    obs_gripper: np.ndarray,
    episode_idx: int,
) -> dict:
    """Compute per-episode statistics for the gripper channels."""
    n = len(action_gripper)
    mid = n // 2

    def _stats(arr: np.ndarray, label: str) -> dict:
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "frame_0": float(arr[0]),
            "frame_mid": float(arr[mid]),
            "frame_last": float(arr[-1]),
        }

    a_stats = _stats(action_gripper, "action")
    o_stats = _stats(obs_gripper, "obs")

    # Delta at key frames
    deltas = {
        "frame_0": float(action_gripper[0] - obs_gripper[0]),
        "frame_mid": float(action_gripper[mid] - obs_gripper[mid]),
        "frame_last": float(action_gripper[-1] - obs_gripper[-1]),
    }

    # Check for open→close transition
    has_low = np.any(action_gripper < 0.3)
    has_high = np.any(action_gripper > 0.7)
    has_transition = bool(has_low and has_high)

    return {
        "episode": episode_idx,
        "n_frames": n,
        "action": a_stats,
        "obs": o_stats,
        "delta": deltas,
        "has_transition": has_transition,
    }


def print_episode_stats(stats: dict) -> None:
    """Print per-episode statistics."""
    ep = stats["episode"]
    n = stats["n_frames"]
    a = stats["action"]
    o = stats["obs"]
    d = stats["delta"]
    trans = "YES" if stats["has_transition"] else "NO"

    print(f"\n  Episode {ep:3d} ({n:4d} frames)  transition: {trans}")
    print(f"    ACTION  gripper: min={a['min']:.4f}  max={a['max']:.4f}  mean={a['mean']:.4f}"
          f"  | f0={a['frame_0']:.4f}  mid={a['frame_mid']:.4f}  last={a['frame_last']:.4f}")
    print(f"    OBS     gripper: min={o['min']:.4f}  max={o['max']:.4f}  mean={o['mean']:.4f}"
          f"  | f0={o['frame_0']:.4f}  mid={o['frame_mid']:.4f}  last={o['frame_last']:.4f}")
    print(f"    DELTA (a-o):     f0={d['frame_0']:+.4f}  mid={d['frame_mid']:+.4f}  last={d['frame_last']:+.4f}")


def print_summary(all_stats: list[dict]) -> None:
    """Print global summary across all episodes."""
    all_action = []
    all_obs = []
    transition_eps = []
    no_transition_eps = []

    for s in all_stats:
        all_action.extend([s["action"]["min"], s["action"]["max"]])
        all_obs.extend([s["obs"]["min"], s["obs"]["max"]])
        if s["has_transition"]:
            transition_eps.append(s["episode"])
        else:
            no_transition_eps.append(s["episode"])

    # Collect per-episode means for global mean
    action_means = [s["action"]["mean"] for s in all_stats]
    obs_means = [s["obs"]["mean"] for s in all_stats]
    action_mins = [s["action"]["min"] for s in all_stats]
    action_maxs = [s["action"]["max"] for s in all_stats]
    obs_mins = [s["obs"]["min"] for s in all_stats]
    obs_maxs = [s["obs"]["max"] for s in all_stats]

    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL EPISODES")
    print("=" * 80)

    print(f"\n  Action gripper:")
    print(f"    Global min:  {min(action_mins):.4f}")
    print(f"    Global max:  {max(action_maxs):.4f}")
    print(f"    Global mean: {np.mean(action_means):.4f}")

    print(f"\n  Observation gripper:")
    print(f"    Global min:  {min(obs_mins):.4f}")
    print(f"    Global max:  {max(obs_maxs):.4f}")
    print(f"    Global mean: {np.mean(obs_means):.4f}")

    # Range checks
    g_min = min(action_mins)
    g_max = max(action_maxs)
    out_of_range = g_min < 0.0 or g_max > 1.0
    raw_units = g_min < -1.0 or g_max > 2.0

    print(f"\n  Range checks:")
    print(f"    Action gripper exceeds [0, 1]:   {'YES' if out_of_range else 'NO'}"
          f"  (min={g_min:.4f}, max={g_max:.4f})")
    print(f"    Likely raw leader units (>2.0):  {'YES' if raw_units else 'NO'}")

    # Transition analysis
    n_with = len(transition_eps)
    n_without = len(no_transition_eps)
    print(f"\n  Grasp transitions (action goes <0.3 then >0.7 within episode):")
    print(f"    Episodes WITH transition:    {n_with:3d} / {len(all_stats)}")
    print(f"    Episodes WITHOUT transition: {n_without:3d} / {len(all_stats)}")
    if no_transition_eps:
        print(f"    No-transition episodes: {no_transition_eps}")


def print_metadata(dataset_dir: Path, action_gripper_idx: int, obs_gripper_idx: int) -> None:
    """Print dataset metadata and normalization stats."""
    print("\n" + "=" * 80)
    print("DATASET METADATA")
    print("=" * 80)

    # info.json
    info_path = dataset_dir / "meta" / "info.json"
    if info_path.exists():
        info = load_json(info_path)
        print(f"\n  info.json:")
        print(f"    robot_type:     {info.get('robot_type', 'N/A')}")
        print(f"    fps:            {info.get('fps', 'N/A')}")
        print(f"    total_episodes: {info.get('total_episodes', 'N/A')}")
        print(f"    total_frames:   {info.get('total_frames', 'N/A')}")

        features = info.get("features", {})
        action_feat = features.get("action", {})
        obs_feat = features.get("observation.state", {})

        print(f"\n    Action feature names: {action_feat.get('names', 'N/A')}")
        print(f"    Action shape:         {action_feat.get('shape', 'N/A')}")
        print(f"    Obs.state names:      {obs_feat.get('names', 'N/A')}")
        print(f"    Obs.state shape:      {obs_feat.get('shape', 'N/A')}")

        # Verify gripper.pos is present
        action_names = action_feat.get("names", [])
        if action_names and action_gripper_idx < len(action_names):
            print(f"\n    Action gripper name at index {action_gripper_idx}: "
                  f"'{action_names[action_gripper_idx]}'")
        obs_names = obs_feat.get("names", [])
        if obs_names and obs_gripper_idx < len(obs_names):
            print(f"    Obs gripper name at index {obs_gripper_idx}: "
                  f"'{obs_names[obs_gripper_idx]}'")
    else:
        print(f"\n  info.json: NOT FOUND at {info_path}")

    # stats.json
    stats_path = dataset_dir / "meta" / "stats.json"
    if stats_path.exists():
        stats = load_json(stats_path)
        print(f"\n  stats.json normalization stats:")

        for feature_key, gripper_idx, label in [
            ("action", action_gripper_idx, "Action gripper"),
            ("observation.state", obs_gripper_idx, "Obs gripper"),
        ]:
            feat_stats = stats.get(feature_key, {})
            if not feat_stats:
                print(f"\n    {label}: NO STATS FOUND for '{feature_key}'")
                continue

            print(f"\n    {label} ('{feature_key}' index {gripper_idx}):")
            for stat_name in ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]:
                vals = feat_stats.get(stat_name)
                if vals is not None and isinstance(vals, list) and gripper_idx < len(vals):
                    print(f"      {stat_name:5s}: {vals[gripper_idx]:.6f}")

            # Flag suspicious values
            g_min = feat_stats.get("min", [None])[gripper_idx] if "min" in feat_stats else None
            g_max = feat_stats.get("max", [None])[gripper_idx] if "max" in feat_stats else None
            if g_min is not None and g_max is not None:
                if g_min < -1.0 or g_max > 2.0:
                    print(f"      ** FLAG: Out-of-range values suggest raw motor units!")
                elif g_max > 1.0:
                    print(f"      ** NOTE: Max slightly exceeds 1.0 ({g_max:.4f})")
    else:
        print(f"\n  stats.json: NOT FOUND at {stats_path}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_episodes(
    df: pd.DataFrame,
    action_gripper_all: np.ndarray,
    obs_gripper_all: np.ndarray,
    target_episodes: list[int],
    output_path: Path,
) -> None:
    """Plot action vs observation gripper for selected episodes."""
    available = df["episode_index"].unique()
    episodes_to_plot = [e for e in target_episodes if e in available]

    # Fill remaining slots if requested episodes don't exist
    if len(episodes_to_plot) < 5:
        for e in sorted(available):
            if e not in episodes_to_plot:
                episodes_to_plot.append(e)
            if len(episodes_to_plot) >= 5:
                break

    n_plots = min(5, len(episodes_to_plot))
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for ax, ep_idx in zip(axes, episodes_to_plot[:n_plots]):
        mask = df["episode_index"].values == ep_idx
        ep_action = action_gripper_all[mask]
        ep_obs = obs_gripper_all[mask]
        frames = np.arange(len(ep_action))

        ax.plot(frames, ep_action, color="tab:blue", linewidth=1.2, label="Action gripper")
        ax.plot(frames, ep_obs, color="tab:orange", linewidth=1.2, label="Obs gripper")

        # Reference lines
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(0.25, color="red", linestyle="--", linewidth=1.0, alpha=0.7,
                    label="Policy output (~0.25)")

        ax.set_title(f"Episode {ep_idx} ({len(ep_action)} frames)", fontsize=11)
        ax.set_ylabel("Gripper value")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame")
    fig.suptitle("Gripper Channel Analysis: Action vs Observation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def print_diagnosis(all_stats: list[dict], action_gripper_all: np.ndarray) -> None:
    """Print a diagnosis based on the analysis."""
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    g_min = float(np.min(action_gripper_all))
    g_max = float(np.max(action_gripper_all))
    g_mean = float(np.mean(action_gripper_all))

    n_with_transition = sum(1 for s in all_stats if s["has_transition"])
    n_total = len(all_stats)

    if g_max > 2.0:
        print(f"\nDIAGNOSIS: Gripper action values are in raw leader units "
              f"(range {g_min:.2f} to {g_max:.2f}), not normalized 0-1. "
              f"The leader gripper is not being mapped to follower gripper "
              f"convention before recording.")
    elif g_min >= -0.1 and g_max <= 1.1 and n_with_transition > n_total * 0.5:
        print(f"\nDIAGNOSIS: Gripper action values are in normalized 0-1 range "
              f"(min={g_min:.4f}, max={g_max:.4f}) with clear open-to-close "
              f"transitions in {n_with_transition}/{n_total} episodes. "
              f"Gripper recording looks correct.")
        print(f"\nThe policy outputting a constant ~0.25 is NOT a dataset issue. "
              f"Possible causes:")
        print(f"  - Normalization mismatch between training and deployment")
        print(f"  - Policy undertrained on gripper (ACT may weight arm joints more)")
        print(f"  - Action chunking: gripper timing may be lost in the chunk horizon")
        print(f"  - Check if training uses q01/q99 quantile normalization vs min/max")
    elif g_min >= -0.1 and g_max <= 1.1 and n_with_transition <= n_total * 0.1:
        print(f"\nDIAGNOSIS: Gripper action values are in 0-1 range "
              f"(min={g_min:.4f}, max={g_max:.4f}) but show NO grasp transitions "
              f"in {n_total - n_with_transition}/{n_total} episodes. "
              f"The gripper may be recorded as always-open or at a constant value.")
    elif g_min >= -0.1 and g_max <= 1.1:
        print(f"\nDIAGNOSIS: Gripper action values are in 0-1 range "
              f"(min={g_min:.4f}, max={g_max:.4f}). "
              f"Transitions found in {n_with_transition}/{n_total} episodes. "
              f"Dataset has partial grasp coverage — some episodes lack clear "
              f"open-to-close transitions, which may confuse the policy.")
    else:
        print(f"\nDIAGNOSIS: Gripper action values have unusual range "
              f"(min={g_min:.4f}, max={g_max:.4f}). "
              f"Further investigation needed.")

    # Additional detail: check if the constant ~0.25 corresponds to dataset q01
    q01_approx = float(np.percentile(action_gripper_all, 1))
    q10_approx = float(np.percentile(action_gripper_all, 10))
    print(f"\n  Dataset action gripper percentiles: q01={q01_approx:.4f}, q10={q10_approx:.4f}")
    if abs(q01_approx - 0.25) < 0.05:
        print(f"  NOTE: The policy output of ~0.25 matches the dataset q01 ({q01_approx:.4f}).")
        print(f"  If training uses quantile normalization (q01/q99), the policy may be")
        print(f"  stuck predicting the normalized minimum, which denormalizes to ~q01.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Gripper channel diagnostic for LeRobot datasets")
    parser.add_argument(
        "--dataset", default="Bearing_pickup_cell/v4",
        help="Dataset path relative to datasets/ directory (default: Bearing_pickup_cell/v4)",
    )
    args = parser.parse_args()

    root = find_project_root()
    dataset_dir = find_dataset(root, args.dataset)
    print(f"Project root: {root}")
    print(f"Dataset dir:  {dataset_dir}")

    # Indices for gripper in the feature arrays
    ACTION_GRIPPER_IDX = 6  # action shape [7], gripper is last
    OBS_GRIPPER_IDX = 6     # observation.state shape [21], gripper.pos is 7th (index 6)

    # -----------------------------------------------------------------------
    # STEP 3: Metadata inspection (do first so we can verify indices)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: DATASET METADATA")
    print("=" * 80)
    print_metadata(dataset_dir, ACTION_GRIPPER_IDX, OBS_GRIPPER_IDX)

    # -----------------------------------------------------------------------
    # STEP 1: Load and inspect raw values
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: PER-EPISODE GRIPPER ANALYSIS")
    print("=" * 80)

    df = load_parquet(dataset_dir)

    # Extract gripper channels
    action_gripper_all = extract_gripper(df, "action", ACTION_GRIPPER_IDX)
    obs_gripper_all = extract_gripper(df, "observation.state", OBS_GRIPPER_IDX)
    episode_indices = df["episode_index"].values

    # Per-episode analysis
    all_stats = []
    unique_episodes = sorted(df["episode_index"].unique())

    for ep_idx in unique_episodes:
        mask = episode_indices == ep_idx
        ep_action = action_gripper_all[mask]
        ep_obs = obs_gripper_all[mask]
        stats = analyze_episode(ep_action, ep_obs, ep_idx)
        all_stats.append(stats)
        print_episode_stats(stats)

    # -----------------------------------------------------------------------
    # STEP 1 continued: Global summary
    # -----------------------------------------------------------------------
    print_summary(all_stats)

    # -----------------------------------------------------------------------
    # STEP 2: Visualization
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: VISUALIZATION")
    print("=" * 80)

    output_path = Path(__file__).resolve().parent / "gripper_analysis.png"
    target_episodes = [0, 5, 10, 15, 20]
    plot_episodes(df, action_gripper_all, obs_gripper_all, target_episodes, output_path)

    # -----------------------------------------------------------------------
    # STEP 4: Diagnosis
    # -----------------------------------------------------------------------
    print_diagnosis(all_stats, action_gripper_all)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
