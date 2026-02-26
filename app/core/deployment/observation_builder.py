"""Standalone observation builder for policy deployment.

Extracted from hil/observation.py (HILObservationMixin) and hil/loop.py
(_convert_action_to_dict) into a class with explicit dependencies instead
of mixin self.* lookups.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ObservationBuilder:
    """Builds policy-format observations from raw robot data.

    Handles:
    - Loading training state names from checkpoint metadata
    - Loading normalization statistics (safetensors or dataset stats.json)
    - Transforming raw observations to policy-expected tensor format
    - Converting policy action tensors back to named motor dicts
    """

    def __init__(
        self,
        checkpoint_path: Path,
        policy,
        policy_type: str = "",
        task: str = "",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.policy = policy
        self.policy_type = policy_type
        self.task = task or "Do the task"

        # Caches (cleared via reset_cache)
        self._training_state_names: Optional[List[str]] = None
        self._training_state_names_loaded = False
        self._norm_stats: Optional[dict] = None
        self._norm_stats_loaded = False
        self._pi05_tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_training_state_names(self) -> Optional[List[str]]:
        """Get motor names from the policy's training dataset.

        Reads train_config.json from the checkpoint directory to find the
        original training dataset, then loads its meta/info.json to get
        the observation.state feature names.

        Returns:
            List of motor names like ['left_base.pos', ...] or None.
        """
        if self._training_state_names_loaded:
            return self._training_state_names

        self._training_state_names_loaded = True

        train_config_path = self.checkpoint_path / "train_config.json"
        if not train_config_path.exists():
            logger.warning("train_config.json not found at %s", train_config_path)
            return None

        with open(train_config_path) as f:
            train_config = json.load(f)

        dataset_root = train_config.get("dataset", {}).get("root")
        if not dataset_root:
            logger.warning("dataset.root not found in train_config")
            return None

        info_path = Path(dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            logger.warning("Training dataset info.json not found: %s", info_path)
            return None

        with open(info_path) as f:
            info = json.load(f)

        state_names = info.get("features", {}).get("observation.state", {}).get("names")
        if state_names:
            logger.info(
                "Loaded %d state names from training dataset: %s",
                len(state_names),
                state_names,
            )

        self._training_state_names = state_names
        return state_names

    def load_normalization_stats(self) -> Optional[dict]:
        """Load normalization statistics from checkpoint or training dataset.

        Tries the checkpoint safetensors file first, then falls back to
        the training dataset's meta/stats.json (needed for Pi0.5).

        Returns:
            Dict with keys like 'action.min', 'action.max',
            'observation.state.min', 'observation.state.max' as tensors,
            or None if not found.
        """
        if self._norm_stats_loaded:
            return self._norm_stats

        self._norm_stats_loaded = True

        import torch

        # Try checkpoint safetensors first
        stats_path = (
            self.checkpoint_path
            / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )
        if stats_path.exists():
            try:
                import safetensors.torch as st

                stats = st.load_file(str(stats_path))
                logger.info("Loaded normalization stats from checkpoint (%d keys)", len(stats))
                self._norm_stats = stats
                return stats
            except Exception as e:
                logger.warning("Failed to load safetensors stats: %s", e)

        # Fallback: training dataset stats.json
        logger.debug("Checkpoint stats not found, trying training dataset...")

        metadata_path = self.checkpoint_path / "policy_metadata.json"
        if not metadata_path.exists():
            for parent_level in range(1, 4):
                parent = self.checkpoint_path
                for _ in range(parent_level):
                    parent = parent.parent
                test_path = parent / "policy_metadata.json"
                if test_path.exists():
                    metadata_path = test_path
                    break

        if not metadata_path.exists():
            logger.warning("Policy metadata not found at: %s", metadata_path)
            self._norm_stats = None
            return None

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            dataset_repo_id = metadata.get("dataset_repo_id")
            use_quantile = metadata.get("config", {}).get(
                "use_quantile_normalization", True
            )

            if not dataset_repo_id:
                logger.warning("No dataset_repo_id in policy metadata")
                self._norm_stats = None
                return None

            # Find project root to locate datasets
            project_root = self.checkpoint_path
            while project_root.name != "nextis_app" and project_root.parent != project_root:
                project_root = project_root.parent

            stats_json_path = project_root / "datasets" / dataset_repo_id / "meta" / "stats.json"
            if not stats_json_path.exists():
                logger.warning("Training dataset stats not found: %s", stats_json_path)
                self._norm_stats = None
                return None

            with open(stats_json_path) as f:
                dataset_stats = json.load(f)

            stats: Dict = {}

            if "action" in dataset_stats:
                action_stats = dataset_stats["action"]
                if use_quantile and "q01" in action_stats and "q99" in action_stats:
                    stats["action.min"] = torch.tensor(action_stats["q01"], dtype=torch.float32)
                    stats["action.max"] = torch.tensor(action_stats["q99"], dtype=torch.float32)
                    logger.info("Loaded action stats (quantiles q01/q99)")
                elif "min" in action_stats and "max" in action_stats:
                    stats["action.min"] = torch.tensor(action_stats["min"], dtype=torch.float32)
                    stats["action.max"] = torch.tensor(action_stats["max"], dtype=torch.float32)
                    logger.info("Loaded action stats (min/max)")

            if "observation.state" in dataset_stats:
                state_stats = dataset_stats["observation.state"]
                if use_quantile and "q01" in state_stats and "q99" in state_stats:
                    stats["observation.state.min"] = torch.tensor(
                        state_stats["q01"], dtype=torch.float32
                    )
                    stats["observation.state.max"] = torch.tensor(
                        state_stats["q99"], dtype=torch.float32
                    )
                elif "min" in state_stats and "max" in state_stats:
                    stats["observation.state.min"] = torch.tensor(
                        state_stats["min"], dtype=torch.float32
                    )
                    stats["observation.state.max"] = torch.tensor(
                        state_stats["max"], dtype=torch.float32
                    )

            self._norm_stats = stats if stats else None
            return self._norm_stats

        except Exception as e:
            logger.warning("Failed to load stats from training dataset: %s", e)
            self._norm_stats = None
            return None

    def prepare_observation(self, raw_obs: dict) -> dict:
        """Transform raw robot observation to policy-expected format.

        Robot returns: ``{'camera_1': np.array(H,W,C), 'left_base.pos': 0.5, ...}``
        Policy expects: ``{'observation.images.camera_1': Tensor(1,C,H,W),
                          'observation.state': Tensor(1,N)}``

        Applies normalization:
        - Images: ImageNet-style ``(x - mean) / std``
        - State: MIN_MAX to [-1, 1]
        - Pi0.5: language tokenization
        """
        import torch

        device = self.policy.config.device
        policy_obs: dict = {}
        norm_stats = self.load_normalization_stats()

        # 1. Camera images
        if hasattr(self.policy.config, "image_features") and self.policy.config.image_features:
            for key in self.policy.config.image_features:
                cam_name = key.split(".")[-1]
                if cam_name in raw_obs:
                    img = raw_obs[cam_name]
                    img_tensor = (
                        torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
                    )
                    if norm_stats:
                        mean_key = f"{key}.mean"
                        std_key = f"{key}.std"
                        if mean_key in norm_stats and std_key in norm_stats:
                            mean = norm_stats[mean_key].view(3, 1, 1)
                            std = norm_stats[std_key].view(3, 1, 1)
                            img_tensor = (img_tensor - mean) / (std + 1e-8)
                    policy_obs[key] = img_tensor.unsqueeze(0).to(device)

        # 2. Observation state
        if hasattr(self.policy.config, "robot_state_feature") and self.policy.config.robot_state_feature:
            state_names = self.get_training_state_names()
            if state_names:
                state_values = [float(raw_obs.get(name, 0.0)) for name in state_names]
                state_tensor = torch.tensor(state_values, dtype=torch.float32)

                if (
                    norm_stats
                    and "observation.state.min" in norm_stats
                    and "observation.state.max" in norm_stats
                ):
                    state_min = norm_stats["observation.state.min"]
                    state_max = norm_stats["observation.state.max"]
                    state_range = state_max - state_min
                    dead_motors = state_range.abs() < 1e-6
                    safe_range = torch.where(
                        dead_motors, torch.ones_like(state_range), state_range
                    )
                    state_tensor = (state_tensor - state_min) / safe_range
                    state_tensor = torch.where(
                        dead_motors, torch.full_like(state_tensor, 0.5), state_tensor
                    )
                    state_tensor = state_tensor * 2.0 - 1.0
                    state_tensor = torch.clamp(state_tensor, -1.0, 1.0)

                policy_obs["observation.state"] = state_tensor.unsqueeze(0).to(device)

        # 3. Pi0.5 language tokenization
        if self.policy_type == "pi05" and "observation.state" in policy_obs:
            self._add_pi05_tokenization(policy_obs, device)

        return policy_obs

    def convert_action_to_dict(
        self,
        action,
        raw_obs: dict,
        movement_scale: float = 1.0,
    ) -> dict:
        """Convert policy action tensor to named motor dict.

        1. Handles multi-step diffusion output (3D/2D → 1D)
        2. Denormalizes from [-1, 1] to motor ranges using stats
        3. Handles dead motors (min==max → midpoint)
        4. Applies movement scaling (safety limiter)
        5. Returns dict with named keys from training dataset

        Args:
            action: Policy output tensor or numpy array.
            raw_obs: Raw robot observation for movement scaling.
            movement_scale: 0.1–1.0 safety limiter for action deltas.

        Returns:
            Dict mapping motor names to position values, or empty dict on failure.
        """
        import torch

        if isinstance(action, dict):
            return action

        action_names = self.get_training_state_names()
        if action_names is None:
            logger.warning("No action names available — cannot convert tensor to dict")
            return {}

        # Tensor → numpy, handle multi-step
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = np.array(action)

        if action_np.ndim == 3:
            action_np = action_np[0, 0]
        elif action_np.ndim == 2:
            action_np = action_np[0]
        elif action_np.ndim > 3:
            action_np = action_np.squeeze()

        if len(action_np) != len(action_names):
            logger.warning(
                "Action dimension mismatch: action has %d elements, names has %d",
                len(action_np),
                len(action_names),
            )
            return {}

        # Denormalize
        norm_stats = self.load_normalization_stats()
        if norm_stats and "action.min" in norm_stats and "action.max" in norm_stats:
            action_min = norm_stats["action.min"].cpu().numpy()
            action_max = norm_stats["action.max"].cpu().numpy()
            action_range = action_max - action_min
            dead_motors = np.abs(action_range) < 1e-6
            safe_range = np.where(dead_motors, 1.0, action_range)
            action_mid = (action_max + action_min) / 2.0

            action_np = (action_np + 1.0) / 2.0 * safe_range + action_min
            action_np = np.where(dead_motors, action_mid, action_np)

        # Movement scaling
        if movement_scale < 1.0 and raw_obs is not None:
            if all(name in raw_obs for name in action_names):
                current_state = np.array([float(raw_obs[name]) for name in action_names])
                if len(current_state) == len(action_np):
                    delta = action_np - current_state
                    action_np = current_state + delta * movement_scale

        return {name: float(action_np[i]) for i, name in enumerate(action_names)}

    def reset_cache(self) -> None:
        """Clear cached state names and normalization stats."""
        self._training_state_names = None
        self._training_state_names_loaded = False
        self._norm_stats = None
        self._norm_stats_loaded = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_pi05_tokenization(self, policy_obs: dict, device) -> None:
        """Add Pi0.5 language tokenization to the observation."""
        import torch

        state_tensor = policy_obs["observation.state"].squeeze(0).cpu().numpy()
        max_state_dim = getattr(self.policy.config, "max_state_dim", 32)
        if len(state_tensor) < max_state_dim:
            padded = np.zeros(max_state_dim, dtype=np.float32)
            padded[: len(state_tensor)] = state_tensor
            state_tensor = padded

        discretized = (
            np.digitize(state_tensor, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        )
        discretized = np.clip(discretized, 0, 255)

        state_str = " ".join(map(str, discretized))
        cleaned_task = self.task.strip().replace("_", " ").replace("\n", " ")
        prompt = f"Task: {cleaned_task}, State: {state_str};\nAction: "

        if self._pi05_tokenizer is None:
            from transformers import AutoTokenizer

            self._pi05_tokenizer = AutoTokenizer.from_pretrained(
                "google/paligemma-3b-pt-224"
            )
            self._pi05_tokenizer.padding_side = "right"
            logger.info("Pi0.5 tokenizer loaded")

        max_length = getattr(self.policy.config, "tokenizer_max_length", 200)
        tokenized = self._pi05_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        policy_obs["observation.language.tokens"] = tokenized["input_ids"].to(device)
        policy_obs["observation.language.attention_mask"] = (
            tokenized["attention_mask"].bool().to(device)
        )
