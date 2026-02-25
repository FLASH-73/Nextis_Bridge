"""HIL observation preparation mixin: state names, normalization, and policy observation building."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class HILObservationMixin:
    """Mixin providing observation-related methods for HIL inference."""

    def _get_training_state_names(self) -> list:
        """
        Get motor names from the policy's training dataset.

        The policy checkpoint includes train_config.json which references
        the original training dataset. We need those motor names (e.g., 7 for left arm)
        not the intervention dataset's names (e.g., 14 for both arms).

        Returns:
            List of motor names like ['left_base.pos', 'left_link3.pos', ...] or None
        """
        import json
        from pathlib import Path

        # Return cached result if available
        if hasattr(self, '_cached_training_state_names'):
            return self._cached_training_state_names

        # Get checkpoint path
        checkpoint_path = Path(self.orchestrator.deployed_policy_path)

        # Load train_config.json from checkpoint directory
        train_config_path = checkpoint_path / "train_config.json"
        if not train_config_path.exists():
            print(f"[HIL] WARNING: train_config.json not found at {train_config_path}")
            self._cached_training_state_names = None
            return None

        with open(train_config_path) as f:
            train_config = json.load(f)

        # Get training dataset path
        dataset_root = train_config.get("dataset", {}).get("root")
        if not dataset_root:
            print("[HIL] WARNING: dataset.root not found in train_config")
            self._cached_training_state_names = None
            return None

        # Load training dataset's info.json
        info_path = Path(dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            print(f"[HIL] WARNING: Training dataset info.json not found: {info_path}")
            self._cached_training_state_names = None
            return None

        with open(info_path) as f:
            info = json.load(f)

        # Get state feature names
        state_names = info.get("features", {}).get("observation.state", {}).get("names")
        if state_names:
            print(f"[HIL] Loaded {len(state_names)} state names from training dataset: {state_names}")

        self._cached_training_state_names = state_names
        return state_names

    def _load_normalization_stats(self) -> dict:
        """
        Load normalization statistics from policy checkpoint or training dataset.

        The checkpoint may include a safetensors file with min/max/mean/std values.
        For Pi0.5, the safetensors file is not saved, so we fallback to loading
        stats from the training dataset's meta/stats.json.

        Returns:
            Dict with normalization stats or None if not found
        """
        import json
        from pathlib import Path

        import safetensors.torch as st
        import torch

        # Return cached result if available
        if hasattr(self, '_cached_norm_stats'):
            return self._cached_norm_stats

        # Get checkpoint path
        checkpoint_path = Path(self.orchestrator.deployed_policy_path)
        stats_path = checkpoint_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"

        # Try loading from checkpoint safetensors first
        if stats_path.exists():
            try:
                stats = st.load_file(str(stats_path))
                print(f"[HIL] Loaded normalization stats from checkpoint ({len(stats)} keys)")
                print(f"[HIL DEBUG] Stats keys: {list(stats.keys())}")

                # Log key stats for debugging
                if 'observation.state.min' in stats:
                    print(f"[HIL]   observation.state.min: {stats['observation.state.min'].tolist()}")
                    print(f"[HIL]   observation.state.max: {stats['observation.state.max'].tolist()}")

                # DEBUG: Log action stats (critical for denormalization)
                if 'action.min' in stats:
                    print(f"[HIL DEBUG] action.min: {stats['action.min'].tolist()}")
                    print(f"[HIL DEBUG] action.max: {stats['action.max'].tolist()}")
                    action_range = stats['action.max'] - stats['action.min']
                    print(f"[HIL DEBUG] action range (max-min): {action_range.tolist()}")
                else:
                    print("[HIL DEBUG] WARNING: 'action.min' not found in stats!")

                self._cached_norm_stats = stats
                return stats

            except Exception as e:
                print(f"[HIL] WARNING: Failed to load normalization stats from safetensors: {e}")

        # FALLBACK: Load from training dataset's stats.json (needed for Pi0.5)
        print("[HIL] Checkpoint stats not found, trying training dataset...")

        # Find policy metadata to get training dataset
        metadata_path = checkpoint_path / "policy_metadata.json"
        if not metadata_path.exists():
            # Try parent directories (checkpoints/last/pretrained_model -> policy_dir)
            for parent_level in [1, 2, 3]:
                parent = checkpoint_path
                for _ in range(parent_level):
                    parent = parent.parent
                test_path = parent / "policy_metadata.json"
                if test_path.exists():
                    metadata_path = test_path
                    break

        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)

                dataset_repo_id = metadata.get("dataset_repo_id")
                use_quantile = metadata.get("config", {}).get("use_quantile_normalization", True)

                if dataset_repo_id:
                    # Find datasets path - go up from checkpoint to project root
                    # checkpoint_path: .../training/outputs/policy_dir/checkpoints/last/pretrained_model
                    # datasets_path: .../datasets
                    project_root = checkpoint_path
                    while project_root.name != "nextis_app" and project_root.parent != project_root:
                        project_root = project_root.parent

                    datasets_path = project_root / "datasets"
                    stats_json_path = datasets_path / dataset_repo_id / "meta" / "stats.json"

                    print(f"[HIL] Looking for stats at: {stats_json_path}")

                    if stats_json_path.exists():
                        with open(stats_json_path) as f:
                            dataset_stats = json.load(f)

                        if "action" in dataset_stats:
                            action_stats = dataset_stats["action"]

                            # IMPORTANT: Normalization type MUST match what was used during training!
                            # Pi0.5 uses QUANTILES normalization by default (use_quantile_normalization=True)
                            # Using wrong normalization range shifts all positions to wrong values
                            print(f"[HIL] Training used quantile normalization: {use_quantile}")

                            if use_quantile and "q01" in action_stats and "q99" in action_stats:
                                # Pi0.5 default: use QUANTILES (q01/q99) - must match training!
                                stats = {
                                    "action.min": torch.tensor(action_stats["q01"], dtype=torch.float32),
                                    "action.max": torch.tensor(action_stats["q99"], dtype=torch.float32),
                                }
                                print("[HIL] Loaded action normalization stats (QUANTILES q01/q99 - matching training)")
                                print(f"[HIL DEBUG] action.min (q01): {action_stats['q01']}")
                                print(f"[HIL DEBUG] action.max (q99): {action_stats['q99']}")
                            elif "min" in action_stats and "max" in action_stats:
                                # Fallback to min/max if training used MEAN_STD or quantiles unavailable
                                stats = {
                                    "action.min": torch.tensor(action_stats["min"], dtype=torch.float32),
                                    "action.max": torch.tensor(action_stats["max"], dtype=torch.float32),
                                }
                                print(f"[HIL] Loaded action normalization stats (min/max - {'fallback' if use_quantile else 'matching training'})")
                                print(f"[HIL DEBUG] action.min: {action_stats['min']}")
                                print(f"[HIL DEBUG] action.max: {action_stats['max']}")
                            else:
                                print("[HIL] WARNING: No valid action normalization stats found")
                                stats = {}

                            # CRITICAL: Also load state normalization stats!
                            # Pi0.5 normalizes the input state too - without this, the model receives
                            # raw motor positions instead of normalized values, causing garbage output
                            if "observation.state" in dataset_stats:
                                state_stats = dataset_stats["observation.state"]
                                if use_quantile and "q01" in state_stats and "q99" in state_stats:
                                    stats["observation.state.min"] = torch.tensor(state_stats["q01"], dtype=torch.float32)
                                    stats["observation.state.max"] = torch.tensor(state_stats["q99"], dtype=torch.float32)
                                    print("[HIL] Loaded state normalization stats (QUANTILES q01/q99)")
                                    print(f"[HIL DEBUG] state.min (q01): {state_stats['q01']}")
                                    print(f"[HIL DEBUG] state.max (q99): {state_stats['q99']}")
                                elif "min" in state_stats and "max" in state_stats:
                                    stats["observation.state.min"] = torch.tensor(state_stats["min"], dtype=torch.float32)
                                    stats["observation.state.max"] = torch.tensor(state_stats["max"], dtype=torch.float32)
                                    print("[HIL] Loaded state normalization stats (min/max)")
                                    print(f"[HIL DEBUG] state.min: {state_stats['min']}")
                                    print(f"[HIL DEBUG] state.max: {state_stats['max']}")
                                else:
                                    print("[HIL] WARNING: No valid state normalization stats found - state will NOT be normalized!")
                            else:
                                print("[HIL] WARNING: No observation.state in dataset stats - state will NOT be normalized!")

                            if stats:
                                self._cached_norm_stats = stats
                                return stats
                    else:
                        print(f"[HIL] WARNING: Training dataset stats not found: {stats_json_path}")
                else:
                    print("[HIL] WARNING: No dataset_repo_id in policy metadata")

            except Exception as e:
                print(f"[HIL] WARNING: Failed to load stats from training dataset: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[HIL] WARNING: Policy metadata not found at: {metadata_path}")

        print("[HIL] WARNING: No normalization stats found - actions will NOT be denormalized!")
        self._cached_norm_stats = None
        return None

    def _prepare_policy_observation(self, raw_obs: dict) -> dict:
        """
        Transform robot observation dict to policy-expected format.

        Robot returns: {'camera_1': np.array(H,W,C), 'left_base.pos': 0.5, ...}
        Policy expects: {'observation.images.camera_1': Tensor(C,H,W), 'observation.state': Tensor(N)}

        IMPORTANT: Applies normalization using stats from the policy checkpoint:
        - Images: (x - mean) / std using ImageNet stats
        - State: (x - min) / (max - min) scaled to [-1, 1]
        """
        import torch

        policy = self.orchestrator.deployed_policy
        device = policy.config.device
        policy_obs = {}

        # Load normalization stats from checkpoint
        norm_stats = self._load_normalization_stats()

        # 1. Transform and NORMALIZE camera images
        if hasattr(policy.config, 'image_features') and policy.config.image_features:
            for key in policy.config.image_features:
                # key is like 'observation.images.camera_2'
                cam_name = key.split('.')[-1]  # 'camera_2'

                if cam_name in raw_obs:
                    img = raw_obs[cam_name]  # numpy HWC uint8

                    # HWC → CHW, normalize to [0,1]
                    img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0

                    # Apply ImageNet-style normalization: (x - mean) / std
                    if norm_stats:
                        mean_key = f"{key}.mean"
                        std_key = f"{key}.std"
                        if mean_key in norm_stats and std_key in norm_stats:
                            mean = norm_stats[mean_key].view(3, 1, 1)  # Shape: (3, 1, 1)
                            std = norm_stats[std_key].view(3, 1, 1)
                            img_tensor = (img_tensor - mean) / (std + 1e-8)

                            if not hasattr(self, '_logged_img_norm'):
                                print(f"[HIL] Image {key} normalized: mean={mean.squeeze().tolist()}, std={std.squeeze().tolist()}")
                                self._logged_img_norm = True

                    img_tensor = img_tensor.unsqueeze(0)  # Add batch dim: (1, C, H, W)
                    policy_obs[key] = img_tensor.to(device)

        # 2. Build and NORMALIZE observation.state tensor
        # IMPORTANT: Use motor names from TRAINING dataset, not intervention dataset!
        if hasattr(policy.config, 'robot_state_feature') and policy.config.robot_state_feature:
            # Get motor names from training dataset (correct dimension)
            state_names = self._get_training_state_names()

            if state_names is None:
                # Fallback: use intervention dataset (may cause dimension mismatch!)
                print("[HIL] WARNING: Falling back to intervention dataset state names - may cause dimension mismatch!")
                if hasattr(self.teleop, 'dataset') and self.teleop.dataset is not None:
                    features = self.teleop.dataset.features
                    if 'action' in features and 'names' in features['action']:
                        state_names = features['action']['names']

            if state_names:
                state_values = [float(raw_obs.get(name, 0.0)) for name in state_names]
                state_tensor = torch.tensor(state_values, dtype=torch.float32)

                # Apply MIN_MAX normalization: (x - min) / (max - min) * 2 - 1 → [-1, 1]
                if norm_stats and 'observation.state.min' in norm_stats and 'observation.state.max' in norm_stats:
                    state_min = norm_stats['observation.state.min']
                    state_max = norm_stats['observation.state.max']

                    # Handle dead motors (min == max) - these didn't move during training
                    # Set normalized value to 0.0 (middle of [-1, 1] range) for these motors
                    state_range = state_max - state_min
                    dead_motors = state_range.abs() < 1e-6  # Motors with essentially no range

                    # Replace zero ranges with 1.0 to avoid division by zero
                    safe_range = torch.where(dead_motors, torch.ones_like(state_range), state_range)

                    # Normalize to [0, 1]
                    state_tensor = (state_tensor - state_min) / safe_range

                    # For dead motors, set normalized value to 0.5 (which becomes 0.0 after scaling to [-1, 1])
                    state_tensor = torch.where(dead_motors, torch.full_like(state_tensor, 0.5), state_tensor)

                    # Scale to [-1, 1]
                    state_tensor = state_tensor * 2.0 - 1.0

                    if not hasattr(self, '_logged_state_norm'):
                        dead_count = dead_motors.sum().item()
                        print(f"[HIL] State normalized using MIN_MAX: min={state_min.tolist()}, max={state_max.tolist()}")
                        if dead_count > 0:
                            dead_indices = torch.where(dead_motors)[0].tolist()
                            print(f"[HIL] WARNING: {dead_count} motors have min==max (didn't move in training), indices: {dead_indices}")
                            print("[HIL]   These motors normalized to 0.0 (neutral position)")
                        print(f"[HIL] State after norm: min={state_tensor.min():.3f}, max={state_tensor.max():.3f}, mean={state_tensor.mean():.3f}")
                        self._logged_state_norm = True

                    # Clamp to [-1, 1] to prevent out-of-range values from confusing the policy
                    # This can happen if robot position differs from training range
                    if state_tensor.min() < -1.0 or state_tensor.max() > 1.0:
                        if not hasattr(self, '_logged_clamp_warning'):
                            # Log which motors are out of range (once)
                            print("[HIL] WARNING: Some motors outside training range!")
                            for i, name in enumerate(state_names):
                                val = state_tensor[i].item()
                                if val < -1.0 or val > 1.0:
                                    raw_val = float(raw_obs.get(name, 0.0))
                                    print(f"[HIL]   Motor '{name}': normalized={val:.2f} (raw={raw_val:.1f}, expected=[{state_min[i].item():.1f}, {state_max[i].item():.1f}])")
                            print("[HIL]   Clamping all normalized values to [-1, 1]")
                            self._logged_clamp_warning = True
                        state_tensor = torch.clamp(state_tensor, -1.0, 1.0)

                state_tensor = state_tensor.unsqueeze(0)  # Add batch dim: (1, N)
                policy_obs['observation.state'] = state_tensor.to(device)

        # 3. For Pi0.5: Add language tokenization
        # Pi0.5 is a VLA model that requires tokenized task instruction
        if self.state.policy_type == "pi05":
            task = self.state.task or "Do the task"  # Fallback task

            # Get the normalized state for discretization
            if 'observation.state' in policy_obs:
                state_for_tokenization = policy_obs['observation.state'].squeeze(0).cpu().numpy()

                # Pad state to max_state_dim (32 for Pi0.5)
                max_state_dim = getattr(policy.config, 'max_state_dim', 32)
                if len(state_for_tokenization) < max_state_dim:
                    padded_state = np.zeros(max_state_dim, dtype=np.float32)
                    padded_state[:len(state_for_tokenization)] = state_for_tokenization
                    state_for_tokenization = padded_state

                # Discretize state into 256 bins (Pi0.5 protocol)
                # State should be in [-1, 1] range after normalization
                discretized = np.digitize(
                    state_for_tokenization,
                    bins=np.linspace(-1, 1, 256 + 1)[:-1]
                ) - 1
                discretized = np.clip(discretized, 0, 255)  # Ensure valid range

                # Create Pi0.5 prompt format
                state_str = " ".join(map(str, discretized))
                cleaned_task = task.strip().replace("_", " ").replace("\n", " ")
                prompt = f"Task: {cleaned_task}, State: {state_str};\nAction: "

                # Tokenize using PaLI-Gemma tokenizer (cached for efficiency)
                if not hasattr(self, '_pi05_tokenizer'):
                    from transformers import AutoTokenizer
                    self._pi05_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
                    self._pi05_tokenizer.padding_side = "right"
                    print("[HIL] Pi0.5 tokenizer loaded: google/paligemma-3b-pt-224")

                max_length = getattr(policy.config, 'tokenizer_max_length', 200)
                tokenized = self._pi05_tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt"
                )

                policy_obs['observation.language.tokens'] = tokenized['input_ids'].to(device)
                # IMPORTANT: Pi0.5 expects attention_mask as boolean, not Long
                policy_obs['observation.language.attention_mask'] = tokenized['attention_mask'].bool().to(device)

                if not hasattr(self, '_logged_pi05_tokenization'):
                    print(f"[HIL] Pi0.5 tokenization: prompt length={len(prompt)}, tokens shape={tokenized['input_ids'].shape}")
                    self._logged_pi05_tokenization = True

        # Log transformed observation once
        if not hasattr(self, '_logged_policy_obs'):
            print(f"[HIL] Transformed observation keys: {list(policy_obs.keys())}")
            for k, v in policy_obs.items():
                if hasattr(v, 'shape'):
                    print(f"[HIL]   {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
            self._logged_policy_obs = True

        return policy_obs
