#!/usr/bin/env python3
"""Offline diagnostic for the full policy deployment pipeline.

Tests normalization, inference, denormalization, and safety pipeline
against real dataset frames — no robot hardware needed. Catches
dimension mismatches, broken normalization, and config errors before
they become dangerous on hardware.

Supports ACT, Diffusion, and Pi0.5 checkpoints.

Usage:
    python scripts/diagnostics/test_act_deployment.py /path/to/checkpoint/
"""

import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup (same as other scripts/ files)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "lerobot" / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_list(values, n=6):
    """Format a list of floats, showing first N and ellipsis."""
    items = [f"{v:.4f}" for v in values[:n]]
    if len(values) > n:
        items.append("...")
    return "[" + ", ".join(items) + "]"


def _check_mark(ok):
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run_diagnostic(checkpoint_path: Path) -> bool:
    """Run the full deployment diagnostic. Returns True if all checks pass."""

    import numpy as np
    import torch

    failures = []
    warnings = []

    print()
    print("=" * 60)
    print("  ACT Deployment Diagnostic")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")

    # ------------------------------------------------------------------
    # 1. Load checkpoint config.json
    # ------------------------------------------------------------------

    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        print(f"\nFATAL: config.json not found at {config_path}")
        return False

    with open(config_path) as f:
        policy_cfg = json.load(f)

    policy_type = policy_cfg.get("type", "unknown")
    chunk_size = policy_cfg.get("chunk_size", "N/A")
    n_action_steps = policy_cfg.get("n_action_steps", "N/A")
    te_coeff = policy_cfg.get("temporal_ensemble_coeff")
    norm_mapping = policy_cfg.get("normalization_mapping", {})
    input_features = policy_cfg.get("input_features", {})
    output_features = policy_cfg.get("output_features", {})

    print(f"Policy type: {policy_type}")
    print(f"Chunk size: {chunk_size}")
    print(f"Action steps: {n_action_steps}")
    te_str = f"{te_coeff}" if te_coeff is not None else "disabled"
    print(f"Temporal ensembling: {te_str}")

    # Determine dimensions from config
    state_dim = None
    action_dim = None
    image_features = {}
    for key, feat in input_features.items():
        if feat.get("type") == "STATE":
            state_dim = feat["shape"][0] if feat.get("shape") else None
        elif feat.get("type") == "VISUAL":
            image_features[key] = feat
    for key, feat in output_features.items():
        if feat.get("type") == "ACTION":
            action_dim = feat["shape"][0] if feat.get("shape") else None

    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Image features: {list(image_features.keys()) or 'none'}")

    # ------------------------------------------------------------------
    # 2. Load train_config.json and resolve dataset
    # ------------------------------------------------------------------

    train_config_path = checkpoint_path / "train_config.json"
    dataset_root = None
    dataset_repo_id = None

    if train_config_path.exists():
        with open(train_config_path) as f:
            train_config = json.load(f)
        dataset_root = train_config.get("dataset", {}).get("root")
        dataset_repo_id = train_config.get("dataset", {}).get("repo_id")
    else:
        # Fallback: try policy_metadata.json
        for parent_level in range(4):
            p = checkpoint_path
            for _ in range(parent_level):
                p = p.parent
            meta_path = p / "policy_metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                dataset_repo_id = meta.get("dataset_repo_id")
                if dataset_repo_id:
                    dataset_root = str(PROJECT_ROOT / "datasets" / dataset_repo_id)
                break

    # Load dataset info.json for state/action names
    state_names = None
    action_names = None
    dataset_stats = None

    if dataset_root:
        info_path = Path(dataset_root) / "meta" / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                ds_info = json.load(f)
            features = ds_info.get("features", {})
            state_names = features.get("observation.state", {}).get("names")
            action_names = features.get("action", {}).get("names")

        stats_path = Path(dataset_root) / "meta" / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                dataset_stats = json.load(f)

    # Fallback: action names = state names (older datasets)
    if action_names is None and state_names is not None:
        action_names = state_names

    # ------------------------------------------------------------------
    # 3. Load policy
    # ------------------------------------------------------------------

    print()
    print("--- Loading Policy ---")

    try:
        from lerobot.policies.factory import get_policy_class

        policy_cls = get_policy_class(policy_type)
        policy = policy_cls.from_pretrained(str(checkpoint_path))
        policy.eval()
        print(f"Policy loaded: {policy_cls.__name__}")
        device = policy.config.device
        print(f"Device: {device}")
    except Exception as e:
        print(f"FATAL: Failed to load policy: {e}")
        return False

    # ------------------------------------------------------------------
    # 4. Load preprocessor/postprocessor
    # ------------------------------------------------------------------

    preprocessor = None
    postprocessor = None

    try:
        from lerobot.policies.factory import make_pre_post_processors

        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, pretrained_path=str(checkpoint_path)
        )
        print("Preprocessor: loaded")
        print("Postprocessor: loaded")
    except Exception as e:
        print(f"Preprocessor: not available ({e})")
        warnings.append(f"Preprocessor not available: {e}")

    # ------------------------------------------------------------------
    # 5. Print normalization info
    # ------------------------------------------------------------------

    print()
    print("--- Normalization ---")
    print(f"Preprocessor available: {'yes' if preprocessor else 'no'}")

    state_norm = norm_mapping.get("STATE", "unknown")
    action_norm = norm_mapping.get("ACTION", "unknown")
    visual_norm = norm_mapping.get("VISUAL", "unknown")
    print(f"Norm mode (state): {state_norm}")
    print(f"Norm mode (action): {action_norm}")
    print(f"Norm mode (visual): {visual_norm}")

    # Print stats from dataset
    if dataset_stats:
        if "observation.state" in dataset_stats:
            ss = dataset_stats["observation.state"]
            if "mean" in ss:
                print(f"State stats: mean={_fmt_list(ss['mean'])}, std={_fmt_list(ss.get('std', []))}")
            elif "min" in ss:
                print(f"State stats: min={_fmt_list(ss['min'])}, max={_fmt_list(ss['max'])}")
        if "action" in dataset_stats:
            ast = dataset_stats["action"]
            if "mean" in ast:
                print(f"Action stats: mean={_fmt_list(ast['mean'])}, std={_fmt_list(ast.get('std', []))}")
            elif "min" in ast:
                print(f"Action stats: min={_fmt_list(ast['min'])}, max={_fmt_list(ast['max'])}")

    # Also check safetensors stats
    safetensors_path = checkpoint_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    if safetensors_path.exists():
        print(f"Safetensors stats: present ({safetensors_path.name})")
    else:
        print("Safetensors stats: not found (using dataset stats)")

    # ------------------------------------------------------------------
    # 6. Print state/action names
    # ------------------------------------------------------------------

    print()
    print("--- State Names ---")

    if state_names:
        dim_label = f"{len(state_names)}-dim"
        print(f"Training state ({dim_label}): {state_names}")
    else:
        print("Training state names: not available")

    if action_names:
        dim_label = f"{len(action_names)}-dim"
        print(f"Training action ({dim_label}): {action_names}")
    else:
        print("Training action names: not available")

    # Validate dimensions match config
    if state_dim is not None and state_names is not None:
        if len(state_names) != state_dim:
            msg = (
                f"State dim mismatch: config says {state_dim}, "
                f"dataset has {len(state_names)} names"
            )
            print(f"  WARNING: {msg}")
            warnings.append(msg)

    if action_dim is not None and action_names is not None:
        if len(action_names) != action_dim:
            msg = (
                f"Action dim mismatch: config says {action_dim}, "
                f"dataset has {len(action_names)} names"
            )
            print(f"  FAIL: {msg}")
            failures.append(msg)

    # ------------------------------------------------------------------
    # 7. Load dataset frame or generate synthetic observation
    # ------------------------------------------------------------------

    print()
    print("--- Sample Inference (frame 0) ---")

    raw_state = None
    gt_action = None
    frame_loaded = False

    if dataset_root and Path(dataset_root).exists():
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            # Try pyav backend first (more compatible), then torchcodec
            for backend in ("pyav", "torchcodec"):
                try:
                    ds = LeRobotDataset(
                        repo_id=dataset_repo_id or Path(dataset_root).name,
                        root=str(dataset_root),
                        video_backend=backend,
                    )
                    frame = ds[0]
                    break
                except Exception:
                    ds = None
                    frame = None
                    continue

            if frame is not None:
                if "observation.state" in frame:
                    raw_state = frame["observation.state"].numpy()
                    print(f"Dataset frame loaded: {len(ds)} frames, {ds.num_episodes} episodes")
                    frame_loaded = True
                if "action" in frame:
                    gt_action = frame["action"].numpy()
            else:
                print("Dataset exists but could not load frames (video backend issue)")

        except Exception as e:
            print(f"Dataset load failed ({e}), using synthetic data")
    else:
        print("Dataset not found, using synthetic data")

    # Synthetic fallback
    if raw_state is None:
        if state_dim:
            raw_state = np.random.uniform(-1.0, 1.0, size=(state_dim,)).astype(np.float32)
            print(f"Synthetic state: {state_dim}-dim random in [-1, 1]")
        elif state_names:
            raw_state = np.random.uniform(-1.0, 1.0, size=(len(state_names),)).astype(np.float32)
            print(f"Synthetic state: {len(state_names)}-dim random in [-1, 1]")
        else:
            print("FATAL: Cannot determine state dimension")
            return False

    print(
        f"Raw state:        min={raw_state.min():.4f} "
        f"max={raw_state.max():.4f} "
        f"mean={raw_state.mean():.4f}"
    )

    # ------------------------------------------------------------------
    # 8. Build observation and run inference
    # ------------------------------------------------------------------

    # Build raw_batch for preprocessor
    raw_batch = {}
    raw_batch["observation.state"] = torch.tensor(raw_state, dtype=torch.float32)

    # Add synthetic images if policy expects them
    missing_cameras = []
    for key, feat in image_features.items():
        shape = feat.get("shape", [3, 480, 640])
        # For diagnostic, create a random image
        raw_batch[key] = torch.rand(shape, dtype=torch.float32)

    # Normalize via preprocessor or manual
    if preprocessor is not None:
        try:
            policy_obs = preprocessor(raw_batch)
        except Exception as e:
            print(f"Preprocessor failed: {e}")
            print("Falling back to manual normalization...")
            preprocessor = None

    if preprocessor is None:
        # Manual: just add batch dim and move to device
        policy_obs = {}
        for k, v in raw_batch.items():
            policy_obs[k] = v.unsqueeze(0).to(device)

    # Print normalized state stats
    if "observation.state" in policy_obs:
        ns = policy_obs["observation.state"]
        ns_min = float(ns.min())
        ns_max = float(ns.max())
        ns_mean = float(ns.mean())

        norm_ok = -5.0 <= ns_min and ns_max <= 5.0
        print(
            f"Normalized state: min={ns_min:.4f} "
            f"max={ns_max:.4f} "
            f"mean={ns_mean:.4f}  "
            f"{_check_mark(norm_ok)} (expected: ~[-2, 2])"
        )
        if not norm_ok:
            msg = (
                f"Normalized state outside [-5, 5]: "
                f"min={ns_min:.2f}, max={ns_max:.2f}. "
                "Normalization may not be applied correctly."
            )
            if frame_loaded:
                # Real data out of range is a genuine failure
                failures.append(msg)
            else:
                # Synthetic random data won't match training distribution
                msg += " (synthetic data — expected with random input)"
                warnings.append(msg)

    # Check for missing camera features
    if image_features:
        for key in image_features:
            if key not in policy_obs:
                msg = f"Missing camera feature: {key}"
                missing_cameras.append(key)
                warnings.append(msg)
        if missing_cameras:
            print(f"Missing camera features: {missing_cameras}")

    # Run inference
    try:
        policy.reset()
        with torch.no_grad():
            action_tensor = policy.select_action(policy_obs)
    except Exception as e:
        print(f"FATAL: policy.select_action() failed: {e}")
        failures.append(f"Inference failed: {e}")
        _print_result(failures, warnings)
        return False

    # Raw action stats
    if action_tensor is not None:
        at = action_tensor
        at_min = float(at.min())
        at_max = float(at.max())
        at_mean = float(at.mean())
        print(
            f"Raw action out:   min={at_min:.4f} "
            f"max={at_max:.4f} "
            f"mean={at_mean:.4f}"
        )

    # Denormalize action
    denorm_action = None
    if postprocessor is not None:
        try:
            result = postprocessor({"action": action_tensor})
            if isinstance(result, dict):
                denorm_action = result["action"].cpu().numpy()
            else:
                denorm_action = result.cpu().numpy() if hasattr(result, "cpu") else np.array(result)
        except Exception:
            # Some versions expect the tensor directly, not wrapped in a dict
            try:
                result = postprocessor(action_tensor)
                if isinstance(result, dict):
                    denorm_action = result["action"].cpu().numpy()
                else:
                    denorm_action = result.cpu().numpy() if hasattr(result, "cpu") else np.array(result)
            except Exception as e2:
                print(f"Postprocessor failed: {e2}")

    if denorm_action is None:
        # Manual fallback: just convert to numpy
        denorm_action = action_tensor.cpu().numpy()

    # Squeeze to 1D
    if denorm_action.ndim == 3:
        denorm_action = denorm_action[0, 0]
    elif denorm_action.ndim == 2:
        denorm_action = denorm_action[0]

    da_min = float(denorm_action.min())
    da_max = float(denorm_action.max())
    da_mean = float(denorm_action.mean())

    denorm_ok = -math.pi <= da_min and da_max <= math.pi
    print(
        f"Denorm action:    min={da_min:.4f} "
        f"max={da_max:.4f} "
        f"mean={da_mean:.4f}  "
        f"{_check_mark(denorm_ok)} (expected: ~[-pi, pi])"
    )
    if not denorm_ok:
        msg = (
            f"Denormalized action outside [-pi, pi]: "
            f"min={da_min:.4f}, max={da_max:.4f}. "
            "Denormalization may not be applied correctly."
        )
        failures.append(msg)

    # Check action dimension
    expected_action_dim = action_dim or (len(action_names) if action_names else None)
    if expected_action_dim is not None and len(denorm_action) != expected_action_dim:
        msg = (
            f"Action dimension mismatch: policy output has {len(denorm_action)} elements, "
            f"expected {expected_action_dim}"
        )
        print(f"  FAIL: {msg}")
        failures.append(msg)

    # ------------------------------------------------------------------
    # 9. Ground truth comparison
    # ------------------------------------------------------------------

    if gt_action is not None and action_names is not None and frame_loaded:
        print()
        print("Ground truth action from dataset:")

        # Squeeze gt_action if needed
        if gt_action.ndim > 1:
            gt_action = gt_action[0]

        if len(gt_action) == len(denorm_action) and len(gt_action) == len(action_names):
            for i, name in enumerate(action_names):
                gt_val = float(gt_action[i])
                pred_val = float(denorm_action[i])
                delta = abs(pred_val - gt_val)
                ok = delta < 1.0  # Generous threshold — untrained policy won't match exactly
                print(
                    f"  {name:>20s}: {gt_val:+8.4f}  |  "
                    f"Policy predicted: {pred_val:+8.4f}  |  "
                    f"delta = {delta:.4f}  {_check_mark(ok)}"
                )
        else:
            print(
                f"  Cannot compare: gt has {len(gt_action)} dims, "
                f"pred has {len(denorm_action)} dims, "
                f"names has {len(action_names)}"
            )
    elif not frame_loaded:
        print()
        print("Ground truth comparison: skipped (synthetic data)")

    # ------------------------------------------------------------------
    # 10. Safety pipeline simulation
    # ------------------------------------------------------------------

    print()
    print("--- Safety Pipeline Simulation ---")

    # Import velocity limits
    try:
        from app.core.deployment.types import DEFAULT_VELOCITY_LIMITS, FALLBACK_VELOCITY_LIMIT
    except ImportError:
        DEFAULT_VELOCITY_LIMITS = {
            "J8009P": 1.5,
            "J4340P": 2.5,
            "J4310": 3.5,
            "STS3215": 4.0,
        }
        FALLBACK_VELOCITY_LIMIT = 2.0

    loop_hz = 30
    dt = 1.0 / loop_hz

    if action_names is not None and len(denorm_action) == len(action_names):
        # Assume starting from zero position
        current_positions = {name: 0.0 for name in action_names}
        target_positions = {name: float(denorm_action[i]) for i, name in enumerate(action_names)}

        print(f"Max delta per frame ({loop_hz}Hz):")
        for name in action_names:
            target = target_positions[name]
            current = current_positions[name]
            delta = abs(target - current)

            # Guess motor model from joint name
            max_vel = FALLBACK_VELOCITY_LIMIT
            motor_model = "unknown"
            for model, vel in DEFAULT_VELOCITY_LIMITS.items():
                # Simple heuristic: base/link1 → J8009P, link2-3 → J4340P, link4-5 → J4310
                if "base" in name and model == "J8009P":
                    max_vel = vel
                    motor_model = model
                    break
                elif ("link1" in name or "link2" in name) and model == "J8009P":
                    max_vel = vel
                    motor_model = model
                    break
                elif ("link3" in name or "link4" in name) and model == "J4340P":
                    max_vel = vel
                    motor_model = model
                    break
                elif ("link5" in name or "wrist" in name) and model == "J4310":
                    max_vel = vel
                    motor_model = model
                    break

            max_delta = max_vel * dt
            frames_needed = math.ceil(delta / max_delta) if max_delta > 0 and delta > 0 else 0

            print(
                f"  {name:>20s}: {max_delta:.4f}rad  "
                f"(vel_limit={max_vel:.1f} rad/s)  "
                f"frames_to_target={frames_needed}"
            )
    else:
        print("  Cannot simulate: action names not available or dimension mismatch")

    # ------------------------------------------------------------------
    # 11. Result
    # ------------------------------------------------------------------

    _print_result(failures, warnings)
    return len(failures) == 0


def _print_result(failures, warnings):
    """Print final PASS/FAIL summary."""
    print()
    print("=" * 60)

    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    if failures:
        print(f"Failures ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        print()
        print("=== RESULT: FAIL ===")
    else:
        print()
        print("=== RESULT: PASS ===")

    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/checkpoint/")
        print()
        print("Tests the full deployment pipeline offline (no hardware).")
        print("Checkpoint should be a directory containing config.json and model.safetensors.")
        sys.exit(1)

    checkpoint_path = Path(sys.argv[1]).resolve()
    if not checkpoint_path.is_dir():
        print(f"Error: {checkpoint_path} is not a directory")
        sys.exit(1)

    passed = run_diagnostic(checkpoint_path)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
