"""SARM (Stage-Aware Reward Modeling) Reward Service.

Trains and uses SARM reward models for HIL-SERL.
SARM learns task progress (0→1) from demonstration datasets,
providing dense rewards for RL training.

Reference: https://arxiv.org/abs/2509.25358
"""

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Default paths
from app.core.config import DATASETS_DIR, MODELS_DIR
_DEFAULT_DATASETS_PATH = DATASETS_DIR
_DEFAULT_MODELS_PATH = MODELS_DIR


@dataclass
class SARMTrainingConfig:
    """Configuration for SARM training."""
    annotation_mode: str = "single_stage"  # Simplest mode, no VLM annotations needed
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 1e-3
    warmup_steps: int = 100
    image_key: str = "observation.images.top"  # Will be auto-detected
    state_key: str = "observation.state"
    n_obs_steps: int = 8
    frame_gap: int = 30  # At 30fps = 1 second between frames
    save_every_epoch: bool = True


@dataclass
class SARMTrainingState:
    """Training state for SARM."""
    status: str = "idle"  # idle, training, completed, failed
    epoch: int = 0
    total_epochs: int = 10
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    stage_loss: float = 0.0
    subtask_loss: float = 0.0
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0


@dataclass
class SARMModel:
    """Metadata for a trained SARM model."""
    name: str
    dataset_repo_id: str
    annotation_mode: str
    num_epochs: int
    final_loss: float
    created_at: str
    image_key: str
    state_key: str


class SARMRewardService:
    """Train and use SARM reward models for HIL-SERL.

    SARM provides stage-aware dense rewards by learning task progress
    from demonstration datasets. Unlike GVL (API-based) or binary
    classifiers, SARM provides:
    - Continuous progress signals [0, 1]
    - Stage awareness (understands subtask structure)
    - Fast local inference (no API latency)
    - Trained on YOUR demos (not generic)
    """

    def __init__(self, datasets_path: Path = None, models_path: Path = None):
        """Initialize SARM reward service.

        Args:
            datasets_path: Path to datasets directory
            models_path: Path to models directory
        """
        self.datasets_path = datasets_path or _DEFAULT_DATASETS_PATH
        self.models_path = (models_path or _DEFAULT_MODELS_PATH) / "sarm_models"
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.training_state = SARMTrainingState()
        self._training_thread = None
        self._stop_training = threading.Event()

        # Lazy-loaded CLIP for inference
        self._clip_model = None
        self._clip_processor = None
        self._device = None

        # Cache for loaded SARM models
        self._loaded_models = {}

        logger.info(f"[SARM] Service initialized, models at: {self.models_path}")

    def _ensure_clip_loaded(self):
        """Lazy-load CLIP model for encoding."""
        if self._clip_model is not None:
            return

        from transformers import CLIPModel, CLIPProcessor

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self._clip_model.to(self._device)
        self._clip_model.eval()

        logger.info(f"[SARM] CLIP loaded on {self._device}")

    def train_sarm(
        self,
        dataset_repo_id: str,
        name: str,
        config: dict = None,
    ) -> dict:
        """Train SARM reward model on demo dataset.

        Runs in background thread. Use get_training_status() to monitor.

        Args:
            dataset_repo_id: Dataset to train on
            name: Name for the trained model
            config: Training configuration overrides

        Returns:
            dict with status
        """
        if self.training_state.status == "training":
            return {"status": "error", "message": "Training already in progress"}

        # Validate name
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            return {"status": "error", "message": "Invalid model name (use alphanumeric, _, -)"}

        # Check if model already exists
        model_path = self.models_path / name
        if model_path.exists():
            return {"status": "error", "message": f"Model '{name}' already exists"}

        # Check dataset exists
        dataset_path = self.datasets_path / dataset_repo_id
        if not dataset_path.exists():
            return {"status": "error", "message": f"Dataset '{dataset_repo_id}' not found"}

        # Parse config
        train_config = SARMTrainingConfig()
        if config:
            for key, value in config.items():
                if hasattr(train_config, key):
                    setattr(train_config, key, value)

        # Start training in background
        self._stop_training.clear()
        self.training_state = SARMTrainingState(
            status="training",
            total_epochs=train_config.num_epochs,
            started_at=time.time(),
        )

        self._training_thread = threading.Thread(
            target=self._training_loop,
            args=(dataset_repo_id, name, train_config),
            daemon=True,
        )
        self._training_thread.start()

        return {"status": "started", "name": name}

    def _training_loop(
        self,
        dataset_repo_id: str,
        name: str,
        config: SARMTrainingConfig,
    ):
        """Background training loop for SARM."""
        try:
            logger.info(f"[SARM] Starting training: {name} on {dataset_repo_id}")

            # Import LeRobot components
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.policies.sarm.configuration_sarm import SARMConfig
            from lerobot.policies.sarm.modeling_sarm import SARMRewardModel
            from lerobot.policies.sarm.processor_sarm import SARMEncodingProcessorStep

            # Load dataset
            dataset_path = self.datasets_path / dataset_repo_id
            dataset = LeRobotDataset(
                repo_id=dataset_repo_id,
                root=dataset_path,
            )

            # Auto-detect image key from dataset
            image_key = config.image_key
            for key in dataset.features:
                if key.startswith("observation.images."):
                    image_key = key
                    break

            logger.info(f"[SARM] Using image_key: {image_key}")

            # Create SARM config for single_stage mode (simplest)
            sarm_config = SARMConfig(
                annotation_mode="single_stage",
                n_obs_steps=config.n_obs_steps,
                frame_gap=config.frame_gap,
                batch_size=config.batch_size,
                image_key=image_key,
                state_key=config.state_key,
                # Single stage = linear progress 0→1
                num_sparse_stages=1,
                sparse_subtask_names=["task"],
                sparse_temporal_proportions=[1.0],
            )

            # Create model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sarm_config.device = str(device)

            model = SARMRewardModel(
                config=sarm_config,
                dataset_stats=dataset.meta.stats if hasattr(dataset.meta, 'stats') else None,
                dataset_meta=dataset.meta,
            )
            model.to(device)
            model.train()

            # Create processor for encoding
            processor = SARMEncodingProcessorStep(
                config=sarm_config,
                image_key=image_key,
                dataset_meta=dataset.meta,
            )
            processor.training = True

            # Create optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

            # Create dataloader
            from torch.utils.data import DataLoader

            # Custom collate that handles delta_timestamps
            def collate_fn(batch):
                # Each item is a dict from dataset
                return batch

            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,  # Avoid multiprocessing issues
                collate_fn=collate_fn,
            )

            total_steps = len(dataloader) * config.num_epochs
            self.training_state.total_steps = total_steps

            # Training loop
            global_step = 0
            for epoch in range(config.num_epochs):
                if self._stop_training.is_set():
                    logger.info("[SARM] Training stopped by user")
                    break

                epoch_loss = 0.0
                epoch_stage_loss = 0.0
                epoch_subtask_loss = 0.0
                num_batches = 0

                for batch_idx, batch_items in enumerate(dataloader):
                    if self._stop_training.is_set():
                        break

                    try:
                        # Process batch through SARM processor
                        processed_batch = self._process_batch_for_sarm(
                            batch_items, processor, image_key, config.state_key, device
                        )

                        if processed_batch is None:
                            continue

                        # Forward pass
                        optimizer.zero_grad()
                        loss, output_dict = model.forward(processed_batch)

                        # Backward pass
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                        # Track losses
                        epoch_loss += loss.item()
                        if "sparse_stage_loss" in output_dict:
                            epoch_stage_loss += output_dict["sparse_stage_loss"].item()
                        if "sparse_subtask_loss" in output_dict:
                            epoch_subtask_loss += output_dict["sparse_subtask_loss"].item()
                        num_batches += 1
                        global_step += 1

                        # Update state
                        self.training_state.step = global_step
                        self.training_state.loss = loss.item()

                    except Exception as e:
                        logger.warning(f"[SARM] Batch error: {e}")
                        continue

                # Epoch complete
                self.training_state.epoch = epoch + 1
                if num_batches > 0:
                    avg_loss = epoch_loss / num_batches
                    self.training_state.loss = avg_loss
                    self.training_state.stage_loss = epoch_stage_loss / num_batches
                    self.training_state.subtask_loss = epoch_subtask_loss / num_batches
                    logger.info(f"[SARM] Epoch {epoch + 1}/{config.num_epochs}, Loss: {avg_loss:.4f}")

            # Save model
            if not self._stop_training.is_set():
                self._save_sarm_model(
                    model, sarm_config, name, dataset_repo_id,
                    config, image_key
                )
                self.training_state.status = "completed"
                self.training_state.completed_at = time.time()
                logger.info(f"[SARM] Training completed: {name}")
            else:
                self.training_state.status = "stopped"

        except Exception as e:
            logger.error(f"[SARM] Training failed: {e}", exc_info=True)
            self.training_state.status = "failed"
            self.training_state.error = str(e)

    def _process_batch_for_sarm(
        self,
        batch_items: list,
        processor: "SARMEncodingProcessorStep",
        image_key: str,
        state_key: str,
        device: torch.device,
    ) -> dict:
        """Process a batch of dataset items for SARM training."""
        try:
            batch_size = len(batch_items)

            # Stack images and states
            images = []
            states = []
            tasks = []
            frame_indices = []
            episode_indices = []

            for item in batch_items:
                # Get image
                img = item.get(image_key)
                if img is None:
                    continue
                if isinstance(img, torch.Tensor):
                    img = img.numpy()
                images.append(img)

                # Get state
                state = item.get(state_key)
                if state is None:
                    state = np.zeros(32)
                if isinstance(state, torch.Tensor):
                    state = state.numpy()
                states.append(state)

                # Get task
                task = item.get("task", "complete the task")
                if isinstance(task, list):
                    task = task[0] if task else "complete the task"
                tasks.append(task)

                # Get indices
                frame_indices.append(item.get("index", 0))
                episode_indices.append(item.get("episode_index", 0))

            if not images:
                return None

            batch_size = len(images)

            # Stack into batched tensors
            # Images: need to handle delta_timestamps format (T, C, H, W) or single (C, H, W)
            processed_images = []
            for img in images:
                if img.ndim == 3:
                    # Single frame (C, H, W) - expand to (1, C, H, W)
                    img = img[np.newaxis, ...]
                # Now img is (T, C, H, W)
                # Pad to total_frames = 1 + n_obs_steps + max_rewind_steps = 13
                total_frames = 13
                if img.shape[0] < total_frames:
                    pad_frames = total_frames - img.shape[0]
                    padding = np.zeros((pad_frames, *img.shape[1:]), dtype=img.dtype)
                    img = np.concatenate([img, padding], axis=0)
                elif img.shape[0] > total_frames:
                    img = img[:total_frames]
                processed_images.append(img)

            images_batch = np.stack(processed_images)  # (B, T, C, H, W)

            # Encode images with CLIP
            video_features = self._encode_images_for_sarm(images_batch, processor)

            # Process states similarly
            processed_states = []
            for state in states:
                if state.ndim == 1:
                    # Single state - expand to (1, D)
                    state = state[np.newaxis, ...]
                # Pad to total_frames
                total_frames = 13
                if state.shape[0] < total_frames:
                    pad_frames = total_frames - state.shape[0]
                    padding = np.zeros((pad_frames, state.shape[-1]), dtype=state.dtype)
                    state = np.concatenate([state, padding], axis=0)
                elif state.shape[0] > total_frames:
                    state = state[:total_frames]
                processed_states.append(state)

            states_batch = np.stack(processed_states)  # (B, T, D)

            # Pad state dim to max_state_dim
            from lerobot.policies.sarm.sarm_utils import pad_state_to_max_dim
            state_tensor = torch.tensor(states_batch, dtype=torch.float32)
            state_features = pad_state_to_max_dim(state_tensor, processor.config.max_state_dim)

            # Encode text (use first task for batch - they should all be the same)
            text_features = processor._encode_text_clip(tasks[0], batch_size)

            # Generate targets for single_stage mode
            # In single_stage: progress = frame_index / episode_length
            # For simplicity, we compute linear progress within each sample
            n_obs_frames = 1 + processor.config.n_obs_steps  # 9
            lengths = torch.full((batch_size,), n_obs_frames, dtype=torch.int32)

            # Targets: For single_stage, just use linear progress 0→1
            # Format: stage (0) + tau (progress)
            sparse_targets = torch.zeros(batch_size, 13, dtype=torch.float32)
            for b in range(batch_size):
                # Linear progress across observation frames
                for t in range(n_obs_frames):
                    tau = t / max(n_obs_frames - 1, 1)
                    sparse_targets[b, t] = 0.0 + tau  # stage 0 + tau

            return {
                "observation": {
                    "video_features": video_features.to(device),
                    "text_features": text_features.to(device),
                    "state_features": state_features.to(device),
                    "lengths": lengths.to(device),
                    "sparse_targets": sparse_targets.to(device),
                }
            }

        except Exception as e:
            logger.warning(f"[SARM] Batch processing error: {e}")
            return None

    def _encode_images_for_sarm(
        self,
        images: np.ndarray,
        processor: "SARMEncodingProcessorStep",
    ) -> torch.Tensor:
        """Encode images with CLIP for SARM."""
        return processor._encode_images_batch(images)

    def _save_sarm_model(
        self,
        model: "SARMRewardModel",
        config: "SARMConfig",
        name: str,
        dataset_repo_id: str,
        train_config: SARMTrainingConfig,
        image_key: str,
    ):
        """Save trained SARM model."""
        model_path = self.models_path / name
        model_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(model.state_dict(), model_path / "model.pt")

        # Save config
        config_dict = {
            "annotation_mode": config.annotation_mode,
            "n_obs_steps": config.n_obs_steps,
            "frame_gap": config.frame_gap,
            "num_sparse_stages": config.num_sparse_stages,
            "sparse_subtask_names": config.sparse_subtask_names,
            "sparse_temporal_proportions": config.sparse_temporal_proportions,
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "max_state_dim": config.max_state_dim,
            "image_key": image_key,
            "state_key": train_config.state_key,
        }
        with open(model_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save metadata
        metadata = {
            "name": name,
            "dataset_repo_id": dataset_repo_id,
            "annotation_mode": config.annotation_mode,
            "num_epochs": train_config.num_epochs,
            "final_loss": self.training_state.loss,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_key": image_key,
            "state_key": train_config.state_key,
        }
        with open(model_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[SARM] Model saved to {model_path}")

    def load_sarm(self, name: str) -> "SARMRewardModel":
        """Load a trained SARM model.

        Args:
            name: Model name

        Returns:
            Loaded SARMRewardModel
        """
        # Check cache
        if name in self._loaded_models:
            return self._loaded_models[name]

        model_path = self.models_path / name
        if not model_path.exists():
            raise ValueError(f"SARM model '{name}' not found")

        # Load config
        with open(model_path / "config.json") as f:
            config_dict = json.load(f)

        # Create SARM config
        from lerobot.policies.sarm.configuration_sarm import SARMConfig
        from lerobot.policies.sarm.modeling_sarm import SARMRewardModel

        sarm_config = SARMConfig(
            annotation_mode=config_dict.get("annotation_mode", "single_stage"),
            n_obs_steps=config_dict.get("n_obs_steps", 8),
            frame_gap=config_dict.get("frame_gap", 30),
            num_sparse_stages=config_dict.get("num_sparse_stages", 1),
            sparse_subtask_names=config_dict.get("sparse_subtask_names", ["task"]),
            sparse_temporal_proportions=config_dict.get("sparse_temporal_proportions", [1.0]),
            hidden_dim=config_dict.get("hidden_dim", 768),
            num_heads=config_dict.get("num_heads", 12),
            num_layers=config_dict.get("num_layers", 8),
            max_state_dim=config_dict.get("max_state_dim", 32),
            image_key=config_dict.get("image_key", "observation.images.top"),
            state_key=config_dict.get("state_key", "observation.state"),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sarm_config.device = str(device)

        # Create model
        model = SARMRewardModel(config=sarm_config)

        # Load weights
        state_dict = torch.load(model_path / "model.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Cache
        self._loaded_models[name] = model

        logger.info(f"[SARM] Loaded model: {name}")
        return model

    def predict_reward(
        self,
        sarm_model: "SARMRewardModel",
        images: list,
        state: np.ndarray,
        task_description: str,
    ) -> float:
        """Predict reward using SARM.

        Args:
            sarm_model: Loaded SARM model
            images: List of recent camera images (history)
            state: Current robot state (joint positions)
            task_description: Task description text

        Returns:
            Progress reward [0, 1]
        """
        self._ensure_clip_loaded()

        try:
            # Prepare images for CLIP encoding
            # Need (B, T, C, H, W) format
            n_obs_steps = sarm_model.config.n_obs_steps

            # Use last n_obs_steps + 1 images
            n_frames = min(len(images), n_obs_steps + 1)
            if n_frames == 0:
                return 0.0

            # Pad if needed
            total_frames = sarm_model.config.total_frames  # 13
            padded_images = []

            for i in range(total_frames):
                if i < n_frames:
                    img = images[-(n_frames - i)]  # Get from end
                else:
                    # Pad with zeros
                    img = np.zeros_like(images[-1]) if images else np.zeros((3, 224, 224), dtype=np.uint8)

                # Ensure correct format (C, H, W)
                if isinstance(img, np.ndarray):
                    if img.ndim == 2:
                        img = np.stack([img, img, img], axis=0)
                    elif img.shape[-1] == 3:  # (H, W, C)
                        img = img.transpose(2, 0, 1)
                padded_images.append(img)

            images_array = np.stack(padded_images)[np.newaxis, ...]  # (1, T, C, H, W)

            # Encode images with CLIP
            video_embeddings = self._encode_images_clip(images_array)

            # Encode text with CLIP
            text_embeddings = self._encode_text_clip(task_description)

            # Prepare state
            if state.ndim == 1:
                state = np.tile(state, (total_frames, 1))  # Repeat for all frames

            from lerobot.policies.sarm.sarm_utils import pad_state_to_max_dim
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state_features = pad_state_to_max_dim(state_tensor, sarm_model.config.max_state_dim)

            # Calculate reward
            reward = sarm_model.calculate_rewards(
                text_embeddings=text_embeddings,
                video_embeddings=video_embeddings,
                state_features=state_features.numpy(),
                return_all_frames=False,
                head_mode="sparse",
            )

            # Clamp to [0, 1]
            if isinstance(reward, np.ndarray):
                reward = float(reward.item() if reward.size == 1 else reward[0])

            return max(0.0, min(1.0, reward))

        except Exception as e:
            logger.warning(f"[SARM] Reward prediction error: {e}")
            return 0.0

    def _encode_images_clip(self, images: np.ndarray) -> np.ndarray:
        """Encode images with CLIP.

        Args:
            images: (B, T, C, H, W) images

        Returns:
            (B, T, 512) encoded features
        """
        from PIL import Image as PILImage

        batch_size, seq_len = images.shape[:2]
        images_flat = images.reshape(batch_size * seq_len, *images.shape[2:])

        # Convert to PIL images
        pil_images = []
        for i in range(images_flat.shape[0]):
            img = images_flat[i]
            if img.shape[0] in [1, 3]:  # (C, H, W)
                img = img.transpose(1, 2, 0)
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)

            # Normalize if needed
            if img.max() > 1.0:
                img = img.astype(np.float32) / 255.0

            pil_img = PILImage.fromarray((img * 255).astype(np.uint8))
            pil_images.append(pil_img)

        # Encode with CLIP
        with torch.no_grad():
            inputs = self._clip_processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            features = self._clip_model.get_image_features(**inputs)
            features = features.cpu().numpy()

        # Reshape back
        features = features.reshape(batch_size, seq_len, -1)
        return features

    def _encode_text_clip(self, text: str) -> np.ndarray:
        """Encode text with CLIP.

        Args:
            text: Task description

        Returns:
            (1, 512) encoded features
        """
        with torch.no_grad():
            inputs = self._clip_processor.tokenizer(
                [text], return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            features = self._clip_model.get_text_features(**inputs)
            features = features.cpu().numpy()

        return features

    def get_model_config(self, name: str) -> dict:
        """Get SARM model configuration.

        Args:
            name: Model name

        Returns:
            Config dict or None if not found
        """
        model_path = self.models_path / name
        config_path = model_path / "config.json"

        if not config_path.exists():
            return None

        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[SARM] Failed to load config for {name}: {e}")
            return None

    def list_sarm_models(self) -> list:
        """List all trained SARM models.

        Returns:
            List of model metadata dicts
        """
        models = []

        if not self.models_path.exists():
            return models

        for model_dir in self.models_path.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        models.append(metadata)
                    except Exception as e:
                        logger.warning(f"[SARM] Failed to load metadata for {model_dir.name}: {e}")

        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return models

    def get_training_status(self) -> dict:
        """Get current training status.

        Returns:
            Training state as dict
        """
        return {
            "status": self.training_state.status,
            "epoch": self.training_state.epoch,
            "total_epochs": self.training_state.total_epochs,
            "step": self.training_state.step,
            "total_steps": self.training_state.total_steps,
            "loss": self.training_state.loss,
            "stage_loss": self.training_state.stage_loss,
            "subtask_loss": self.training_state.subtask_loss,
            "error": self.training_state.error,
            "elapsed_s": time.time() - self.training_state.started_at if self.training_state.started_at else 0,
        }

    def stop_training(self) -> dict:
        """Stop training in progress.

        Returns:
            Status dict
        """
        if self.training_state.status != "training":
            return {"status": "error", "message": "No training in progress"}

        self._stop_training.set()
        return {"status": "stopping"}

    def delete_sarm(self, name: str) -> dict:
        """Delete a SARM model.

        Args:
            name: Model name

        Returns:
            Status dict
        """
        model_path = self.models_path / name
        if not model_path.exists():
            return {"status": "error", "message": f"Model '{name}' not found"}

        # Remove from cache
        if name in self._loaded_models:
            del self._loaded_models[name]

        # Delete files
        import shutil
        shutil.rmtree(model_path)

        logger.info(f"[SARM] Deleted model: {name}")
        return {"status": "deleted", "name": name}

    def is_success(self, reward: float, threshold: float = 0.85) -> bool:
        """Check if reward indicates task success.

        Args:
            reward: Progress reward from predict_reward()
            threshold: Success threshold (default 0.85)

        Returns:
            True if task is considered successful
        """
        return reward >= threshold
