import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from .types import _PROJECT_ROOT, PolicyType, ValidationResult

logger = logging.getLogger(__name__)


class ValidatorMixin:
    """Mixin providing dataset validation methods for TrainingService."""

    def validate_dataset(self, repo_id: str, policy_type: str) -> ValidationResult:
        """
        Validates that a dataset is compatible with a given policy type.
        Returns validation result with errors, warnings, and detected features.
        """
        errors = []
        warnings = []
        features = {}

        # Security check
        if ".." in repo_id:
            return ValidationResult(
                valid=False,
                errors=["Invalid repo_id: path traversal not allowed"],
                warnings=[],
                features={}
            )

        dataset_path = self.datasets_path / repo_id

        # Check dataset exists
        if not dataset_path.exists():
            return ValidationResult(
                valid=False,
                errors=[f"Dataset not found at {dataset_path}"],
                warnings=[],
                features={}
            )

        # Check info.json exists
        info_path = dataset_path / "meta" / "info.json"
        if not info_path.exists():
            return ValidationResult(
                valid=False,
                errors=["Dataset missing meta/info.json - not a valid LeRobot dataset"],
                warnings=[],
                features={}
            )

        # Load dataset info
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[f"Failed to read info.json: {e}"],
                warnings=[],
                features={}
            )

        features = info.get("features", {})
        total_episodes = info.get("total_episodes", 0)
        total_frames = info.get("total_frames", 0)

        # Check basic dataset validity
        if total_episodes == 0:
            errors.append("Dataset has no episodes")

        if total_frames == 0:
            errors.append("Dataset has no frames")

        # Policy-specific validation
        if policy_type == PolicyType.SMOLVLA.value or policy_type == "smolvla":
            self._validate_for_smolvla(features, info, errors, warnings, dataset_path)
        elif policy_type == PolicyType.DIFFUSION.value or policy_type == "diffusion":
            self._validate_for_diffusion(features, info, errors, warnings)
        elif policy_type == PolicyType.ACT.value or policy_type == "act":
            self._validate_for_act(features, info, errors, warnings)
        elif policy_type == PolicyType.PI05.value or policy_type == "pi05":
            self._validate_for_pi05(features, info, errors, warnings, dataset_path)
        else:
            warnings.append(f"Unknown policy type '{policy_type}', skipping policy-specific validation")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            features={
                "detected": list(features.keys()),
                "total_episodes": total_episodes,
                "total_frames": total_frames,
                "fps": info.get("fps"),
                "robot_type": info.get("robot_type"),
            }
        )

    def _validate_for_smolvla(
        self,
        features: dict,
        info: dict,
        errors: list,
        warnings: list,
        dataset_path: Path
    ):
        """SmolVLA-specific validation."""
        # Required: at least one camera image
        image_keys = [k for k in features.keys() if k.startswith("observation.images")]
        if not image_keys:
            errors.append("SmolVLA requires at least one camera (observation.images.*)")
        else:
            # Check image dimensions
            for key in image_keys:
                feat = features[key]
                shape = feat.get("shape", [])
                if len(shape) >= 2:
                    # Shape could be (C, H, W) or (H, W, C)
                    h, w = shape[-2], shape[-1]
                    if isinstance(feat.get("dtype"), str) and "image" in feat.get("dtype", ""):
                        # Video/image feature
                        pass
                    if h < 480 or w < 480:
                        warnings.append(f"{key} resolution may be low for SmolVLA (recommended: 512x512+)")

        # Required: observation.state
        if "observation.state" not in features:
            errors.append("SmolVLA requires observation.state (robot joint states)")
        else:
            state_shape = features["observation.state"].get("shape", [])
            if state_shape and state_shape[0] > 32:
                warnings.append(f"State dimension ({state_shape[0]}) exceeds SmolVLA max (32), will be truncated")

        # Required: action
        if "action" not in features:
            errors.append("SmolVLA requires action feature")
        else:
            action_shape = features["action"].get("shape", [])
            if action_shape and action_shape[0] > 32:
                warnings.append(f"Action dimension ({action_shape[0]}) exceeds SmolVLA max (32), will be truncated")

        # Optional but recommended: task descriptions
        # Check if task is in features (per-frame) or in episode metadata
        has_task = "task" in features
        if not has_task:
            # Also check episode metadata as fallback
            try:
                import pandas as pd
                episodes_dir = dataset_path / "meta" / "episodes"
                if episodes_dir.exists():
                    df = pd.read_parquet(episodes_dir)
                    if "task" in df.columns or "task_index" in df.columns:
                        has_task = True
            except Exception:
                pass

        if not has_task:
            warnings.append("Dataset missing 'task' descriptions - will use default task prompt")

    def _validate_for_diffusion(self, features: dict, info: dict, errors: list, warnings: list):
        """Diffusion policy validation."""
        # Requires images and actions
        image_keys = [k for k in features.keys() if k.startswith("observation.images") and not k.endswith("_depth")]
        if not image_keys:
            errors.append("Diffusion policy requires at least one camera (observation.images.*)")

        if "action" not in features:
            errors.append("Diffusion policy requires action feature")

        if "observation.state" not in features:
            warnings.append("observation.state recommended for diffusion policy")

        # Inform about auto-resize when cameras have different shapes
        if len(image_keys) > 1:
            shapes = {}
            for k in image_keys:
                shape = tuple(features[k].get("shape", []))
                if shape:
                    shapes[k] = shape

            unique_shapes = set(shapes.values())
            if len(unique_shapes) > 1:
                shape_info = ", ".join([f"{k}: {shapes[k]}" for k in sorted(shapes.keys())])
                warnings.append(
                    f"Cameras have different shapes ({shape_info}). "
                    f"All images will be resized to 480x640 during training."
                )

    def _validate_for_act(self, features: dict, info: dict, errors: list, warnings: list):
        """ACT policy validation."""
        # Similar to diffusion
        image_keys = [k for k in features.keys() if k.startswith("observation.images")]
        if not image_keys:
            errors.append("ACT policy requires at least one camera (observation.images.*)")

        if "action" not in features:
            errors.append("ACT policy requires action feature")

        if "observation.state" not in features:
            errors.append("ACT policy requires observation.state")

    def _validate_for_pi05(
        self,
        features: dict,
        info: dict,
        errors: list,
        warnings: list,
        dataset_path: Path
    ):
        """Pi0.5 policy validation."""
        # Requires images (at least one camera)
        image_keys = [k for k in features.keys() if k.startswith("observation.images") and not k.endswith("_depth")]
        if not image_keys:
            errors.append("Pi0.5 requires at least one camera (observation.images.*)")

        # Requires action
        if "action" not in features:
            errors.append("Pi0.5 requires action feature")
        else:
            action_shape = features["action"].get("shape", [])
            if action_shape and action_shape[0] > 32:
                warnings.append(f"Action dimension ({action_shape[0]}) exceeds Pi0.5 max (32), will be padded/truncated")

        # Requires observation.state
        if "observation.state" not in features:
            warnings.append("Pi0.5 works best with observation.state")
        else:
            state_shape = features["observation.state"].get("shape", [])
            if state_shape and state_shape[0] > 32:
                warnings.append(f"State dimension ({state_shape[0]}) exceeds Pi0.5 max (32), will be padded/truncated")

        # Check for quantile stats in stats.json (required for default normalization)
        stats_path = dataset_path / "meta" / "stats.json"
        has_quantiles = False
        if stats_path.exists():
            try:
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                # Check if action and state have q01/q99
                has_action_quantiles = (
                    "action" in stats and
                    "q01" in stats.get("action", {}) and
                    "q99" in stats.get("action", {})
                )
                has_state_quantiles = True  # Optional
                if "observation.state" in stats:
                    has_state_quantiles = "q01" in stats.get("observation.state", {})
                has_quantiles = has_action_quantiles and has_state_quantiles
            except Exception:
                pass

        if not has_quantiles:
            warnings.append(
                "Dataset missing quantile statistics (q01/q99). "
                "Either compute quantiles first, or use MEAN_STD normalization fallback."
            )

    def _get_dataset_features(self, dataset_repo_id: str) -> dict:
        """Extract feature information from dataset metadata."""
        dataset_path = self.datasets_path / dataset_repo_id / "meta" / "info.json"
        with open(dataset_path) as f:
            info = json.load(f)
        return info.get("features", {})

    def _build_smolvla_rename_map(self, features: dict) -> dict:
        """Build rename_map to translate dataset features to SmolVLA expected format.

        SmolVLA expects: observation.images.camera1, camera2, camera3
        Datasets may have: camera_1, left_cam, wrist, etc.
        """
        rename_map = {}

        # Find all image features (exclude depth cameras - SmolVLA doesn't use them by default)
        image_features = sorted([
            k for k in features.keys()
            if k.startswith("observation.images.")
            and not k.endswith("_depth")
        ])

        # Map to camera1, camera2, camera3, etc.
        for idx, feature_name in enumerate(image_features, start=1):
            expected_name = f"observation.images.camera{idx}"
            if feature_name != expected_name:
                rename_map[feature_name] = expected_name

        return rename_map

    def _build_diffusion_rename_map(self, features: dict) -> dict:
        """Build rename_map for diffusion policy - rename ALL cameras to standard format.

        Since we use resize_shape, all cameras can be used regardless of resolution.
        Images will be resized to a common shape during training.
        """
        rename_map = {}

        # Find all image features (exclude depth cameras)
        image_cameras = []
        for k in sorted(features.keys()):
            if k.startswith("observation.images.") and not k.endswith("_depth"):
                image_cameras.append(k)

        if not image_cameras:
            return rename_map

        # Map ALL cameras to camera1, camera2, etc. format
        for idx, feature_name in enumerate(image_cameras, start=1):
            expected_name = f"observation.images.camera{idx}"
            if feature_name != expected_name:
                rename_map[feature_name] = expected_name

        return rename_map

    def has_quantile_stats(self, repo_id: str) -> dict:
        """Check if dataset has quantile statistics for Pi0.5 training.

        Returns:
            dict with keys: has_quantiles (bool), missing_features (list), message (str)
        """
        dataset_path = self.datasets_path / repo_id
        stats_path = dataset_path / "meta" / "stats.json"

        result = {
            "has_quantiles": False,
            "missing_features": [],
            "message": ""
        }

        if not dataset_path.exists():
            result["message"] = f"Dataset not found: {repo_id}"
            return result

        if not stats_path.exists():
            result["message"] = "No stats.json found in dataset metadata"
            return result

        try:
            with open(stats_path, "r") as f:
                stats = json.load(f)

            required_quantile_keys = ["q01", "q99"]
            features_to_check = ["action", "observation.state"]

            for feature in features_to_check:
                if feature in stats:
                    feature_stats = stats[feature]
                    if not all(k in feature_stats for k in required_quantile_keys):
                        result["missing_features"].append(feature)

            result["has_quantiles"] = len(result["missing_features"]) == 0
            if result["has_quantiles"]:
                result["message"] = "Dataset has quantile statistics"
            else:
                result["message"] = f"Missing quantile stats for: {', '.join(result['missing_features'])}"

        except Exception as e:
            result["message"] = f"Error reading stats: {e}"

        return result

    def compute_quantile_stats(self, repo_id: str) -> dict:
        """Compute quantile statistics for a dataset using LeRobot's augment script.

        This runs synchronously and may take a while for large datasets.
        Returns status dict with success/error message.
        """
        dataset_path = self.datasets_path / repo_id
        if not dataset_path.exists():
            return {"status": "error", "message": f"Dataset not found: {repo_id}"}

        # Build command to run the augment script
        lerobot_src = _PROJECT_ROOT / "lerobot" / "src"
        script_path = lerobot_src / "lerobot" / "datasets" / "v30" / "augment_dataset_quantile_stats.py"

        if not script_path.exists():
            return {"status": "error", "message": f"Quantile augment script not found at {script_path}"}

        cmd = [
            sys.executable,
            str(script_path),
            f"--repo-id={repo_id}",
            f"--root={dataset_path}",
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(lerobot_src)

        logger.info(f"Computing quantile stats for {repo_id}: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=str(_PROJECT_ROOT),
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"Quantile stats computed successfully for {repo_id}")
                return {
                    "status": "success",
                    "message": "Quantile statistics computed successfully"
                }
            else:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                logger.error(f"Quantile computation failed for {repo_id}: {error_msg}")
                return {
                    "status": "error",
                    "message": f"Script failed: {error_msg}"
                }
        except subprocess.TimeoutExpired:
            logger.error(f"Quantile computation timed out for {repo_id}")
            return {
                "status": "error",
                "message": "Quantile computation timed out (>10 minutes)"
            }
        except Exception as e:
            logger.error(f"Quantile computation error for {repo_id}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
