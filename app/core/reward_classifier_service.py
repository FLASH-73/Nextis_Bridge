"""Reward Classifier Service for HIL-SERL.

Manages training and inference of reward classifiers used to provide
automated reward signals during online RL training. Classifiers are
trained on demonstration data: last N frames per episode = "success",
random early frames = "failure".
"""

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# Default paths
_APP_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASETS_PATH = _APP_ROOT / "datasets"
_DEFAULT_MODELS_PATH = _APP_ROOT / "models"


@dataclass
class ClassifierTrainingState:
    status: str = "idle"  # idle, training, completed, failed
    epoch: int = 0
    total_epochs: int = 50
    loss: float = 0.0
    accuracy: float = 0.0
    error: str = ""


@dataclass
class ClassifierInfo:
    name: str
    dataset_repo_id: str
    num_cameras: int
    num_success_frames: int
    num_failure_frames: int
    accuracy: float
    created_at: str
    path: str


class RewardClassifierService:
    """Train and manage reward classifiers for HIL-SERL."""

    def __init__(
        self,
        datasets_path: Path = None,
        models_path: Path = None,
    ):
        self.datasets_path = Path(datasets_path) if datasets_path else _DEFAULT_DATASETS_PATH
        self.models_path = (Path(models_path) if models_path else _DEFAULT_MODELS_PATH) / "reward_classifiers"
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.training_state = ClassifierTrainingState()
        self._training_thread = None
        self._stop_event = threading.Event()

        # Cache loaded classifiers
        self._loaded_classifiers = {}

    def train_classifier(
        self,
        dataset_repo_id: str,
        name: str,
        success_frames_per_episode: int = 5,
        failure_frames_per_episode: int = 10,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        image_size: tuple = (224, 224),
    ) -> dict:
        """Start training a reward classifier from a demonstration dataset.

        Args:
            dataset_repo_id: LeRobot dataset to extract frames from
            name: Name for the trained classifier
            success_frames_per_episode: Number of final frames per episode to label as success
            failure_frames_per_episode: Number of random early frames per episode to label as failure
            epochs: Training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            image_size: Target image resize (H, W)

        Returns:
            dict with status and training info
        """
        if self.training_state.status == "training":
            return {"status": "error", "message": "A classifier is already being trained"}

        # Start training in background
        self._stop_event.clear()
        self.training_state = ClassifierTrainingState(total_epochs=epochs)
        self.training_state.status = "training"

        self._training_thread = threading.Thread(
            target=self._train_classifier_loop,
            args=(dataset_repo_id, name, success_frames_per_episode,
                  failure_frames_per_episode, epochs, batch_size, learning_rate, image_size),
            daemon=True,
        )
        self._training_thread.start()

        return {"status": "started", "name": name}

    def _train_classifier_loop(
        self,
        dataset_repo_id: str,
        name: str,
        success_frames_per_episode: int,
        failure_frames_per_episode: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        image_size: tuple,
    ):
        """Background training loop for reward classifier."""
        try:
            from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
            from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

            logger.info(f"[RewardClassifier] Starting training: {name} from {dataset_repo_id}")

            # 1. Load dataset and extract frames
            dataset_path = self.datasets_path / dataset_repo_id
            images, labels, camera_keys = self._extract_training_data(
                dataset_path, success_frames_per_episode, failure_frames_per_episode, image_size
            )

            if images is None or len(images) == 0:
                raise ValueError("No training data extracted from dataset")

            logger.info(f"[RewardClassifier] Extracted {len(images)} frames "
                        f"({int(labels.sum())} success, {int((1 - labels).sum())} failure)")

            # 2. Create classifier config
            num_cameras = len(camera_keys)
            input_features = {}
            for cam_key in camera_keys:
                from lerobot.configs.types import FeatureType, PolicyFeature
                input_features[cam_key] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, image_size[0], image_size[1]),
                )

            config = RewardClassifierConfig(
                num_classes=2,
                num_cameras=num_cameras,
                model_name="helper2424/resnet10",
                model_type="cnn",
                hidden_dim=256,
                latent_dim=256,
                image_embedding_pooling_dim=8,
                dropout_rate=0.1,
                learning_rate=learning_rate,
                device="cuda" if torch.cuda.is_available() else "cpu",
                input_features=input_features,
                output_features={},
            )

            # 3. Create and train classifier
            classifier = Classifier(config)
            device = torch.device(config.device)
            classifier = classifier.to(device)
            classifier.train()

            optimizer = torch.optim.AdamW(
                classifier.get_optim_params(),
                lr=learning_rate,
                weight_decay=0.01,
            )

            # Prepare DataLoader
            # images shape: (N, num_cameras, C, H, W)
            images_tensor = images.to(device)
            labels_tensor = labels.to(device)
            dataset = TensorDataset(images_tensor, labels_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

            best_accuracy = 0.0

            for epoch in range(epochs):
                if self._stop_event.is_set():
                    logger.info("[RewardClassifier] Training stopped by user")
                    self.training_state.status = "failed"
                    self.training_state.error = "Stopped by user"
                    return

                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0

                for batch_images, batch_labels in dataloader:
                    optimizer.zero_grad()

                    # Build batch dict for classifier
                    batch = {}
                    for i, cam_key in enumerate(camera_keys):
                        batch[cam_key] = batch_images[:, i]  # (B, C, H, W)
                    batch["classifier_label"] = batch_labels.long()

                    loss, info = classifier(batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                    optimizer.step()

                    # Track accuracy
                    with torch.no_grad():
                        output = classifier.predict(
                            [batch[cam_key] for cam_key in camera_keys]
                        )
                        preds = (output.probabilities > 0.5).long().squeeze()
                        correct = (preds == batch_labels.long()).sum().item()
                        epoch_correct += correct
                        epoch_total += len(batch_labels)

                    epoch_loss += loss.item() * len(batch_labels)

                avg_loss = epoch_loss / epoch_total if epoch_total > 0 else 0
                accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0

                self.training_state.epoch = epoch + 1
                self.training_state.loss = avg_loss
                self.training_state.accuracy = accuracy

                if accuracy > best_accuracy:
                    best_accuracy = accuracy

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"[RewardClassifier] Epoch {epoch + 1}/{epochs} - "
                                f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}")

            # 4. Save classifier
            save_path = self.models_path / name
            save_path.mkdir(parents=True, exist_ok=True)

            # Save model weights
            torch.save(classifier.state_dict(), save_path / "classifier.pt")

            # Save config
            config_dict = {
                "num_classes": config.num_classes,
                "num_cameras": config.num_cameras,
                "model_name": config.model_name,
                "model_type": config.model_type,
                "hidden_dim": config.hidden_dim,
                "latent_dim": config.latent_dim,
                "image_embedding_pooling_dim": config.image_embedding_pooling_dim,
                "dropout_rate": config.dropout_rate,
                "learning_rate": config.learning_rate,
                "camera_keys": camera_keys,
                "image_size": list(image_size),
                "input_features": {k: {"type": str(v.type), "shape": list(v.shape)}
                                   for k, v in input_features.items()},
            }
            with open(save_path / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

            # Save metadata
            metadata = {
                "name": name,
                "dataset_repo_id": dataset_repo_id,
                "num_cameras": num_cameras,
                "num_success_frames": int(labels.sum()),
                "num_failure_frames": int((1 - labels).sum()),
                "accuracy": best_accuracy,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epochs": epochs,
            }
            with open(save_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            self.training_state.status = "completed"
            self.training_state.accuracy = best_accuracy
            logger.info(f"[RewardClassifier] Training complete. Accuracy: {best_accuracy:.3f}. "
                        f"Saved to {save_path}")

        except Exception as e:
            logger.exception(f"[RewardClassifier] Training failed: {e}")
            self.training_state.status = "failed"
            self.training_state.error = str(e)

    def _extract_training_data(
        self,
        dataset_path: Path,
        success_frames_per_episode: int,
        failure_frames_per_episode: int,
        image_size: tuple,
    ) -> tuple:
        """Extract labeled frames from a LeRobot dataset.

        Strategy:
        - SUCCESS (label=1): Last N frames of each episode (task completed)
        - FAILURE (label=0): Random frames from first half of each episode

        Returns:
            (images_tensor, labels_tensor, camera_keys)
            images shape: (N, num_cameras, 3, H, W)
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        try:
            dataset = LeRobotDataset(
                repo_id=str(dataset_path.name),
                root=dataset_path,
            )
        except Exception:
            # Try with parent as root
            dataset = LeRobotDataset(
                repo_id=str(dataset_path.name),
                root=dataset_path.parent,
            )

        # Identify camera keys
        camera_keys = sorted([
            key for key in dataset.features.keys()
            if key.startswith("observation.images")
        ])

        if not camera_keys:
            raise ValueError("No camera features found in dataset")

        logger.info(f"[RewardClassifier] Found cameras: {camera_keys}")

        all_images = []
        all_labels = []

        total_episodes = dataset.meta.total_episodes

        for ep_idx in range(total_episodes):
            # Get episode frame indices
            ep_data_indices = [
                i for i in range(len(dataset))
                if dataset[i].get("episode_index", -1) == ep_idx
            ]

            if not ep_data_indices:
                # Try alternative: use episode length from meta
                try:
                    from_idx = dataset.episode_data_index["from"][ep_idx].item()
                    to_idx = dataset.episode_data_index["to"][ep_idx].item()
                    ep_data_indices = list(range(from_idx, to_idx))
                except Exception:
                    continue

            if len(ep_data_indices) < success_frames_per_episode + 2:
                continue

            # SUCCESS frames: last N frames
            success_indices = ep_data_indices[-success_frames_per_episode:]
            for idx in success_indices:
                frame = dataset[idx]
                cam_images = []
                for cam_key in camera_keys:
                    img = frame[cam_key]
                    if isinstance(img, torch.Tensor):
                        # Already tensor (C, H, W) or (H, W, C)
                        if img.dim() == 3 and img.shape[0] != 3:
                            img = img.permute(2, 0, 1)  # HWC -> CHW
                    else:
                        img = torch.from_numpy(np.array(img))
                        if img.dim() == 3 and img.shape[0] != 3:
                            img = img.permute(2, 0, 1)
                    # Resize
                    img = img.float()
                    if img.shape[1:] != image_size:
                        img = F.interpolate(
                            img.unsqueeze(0),
                            size=image_size,
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                    cam_images.append(img)
                all_images.append(torch.stack(cam_images))  # (num_cameras, 3, H, W)
                all_labels.append(1.0)

            # FAILURE frames: random from first half of episode
            first_half = ep_data_indices[:len(ep_data_indices) // 2]
            if len(first_half) > failure_frames_per_episode:
                failure_indices = np.random.choice(first_half, failure_frames_per_episode, replace=False)
            else:
                failure_indices = first_half

            for idx in failure_indices:
                frame = dataset[idx]
                cam_images = []
                for cam_key in camera_keys:
                    img = frame[cam_key]
                    if isinstance(img, torch.Tensor):
                        if img.dim() == 3 and img.shape[0] != 3:
                            img = img.permute(2, 0, 1)
                    else:
                        img = torch.from_numpy(np.array(img))
                        if img.dim() == 3 and img.shape[0] != 3:
                            img = img.permute(2, 0, 1)
                    img = img.float()
                    if img.shape[1:] != image_size:
                        img = F.interpolate(
                            img.unsqueeze(0),
                            size=image_size,
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                    cam_images.append(img)
                all_images.append(torch.stack(cam_images))
                all_labels.append(0.0)

        if not all_images:
            return None, None, camera_keys

        images_tensor = torch.stack(all_images)  # (N, num_cameras, 3, H, W)
        labels_tensor = torch.tensor(all_labels, dtype=torch.float32)

        return images_tensor, labels_tensor, camera_keys

    def load_classifier(self, name: str) -> tuple:
        """Load a trained reward classifier.

        Returns:
            (classifier_model, config_dict) or raises ValueError
        """
        if name in self._loaded_classifiers:
            return self._loaded_classifiers[name]

        model_path = self.models_path / name
        if not model_path.exists():
            raise ValueError(f"Classifier '{name}' not found at {model_path}")

        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
        from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

        # Load config
        with open(model_path / "config.json") as f:
            config_dict = json.load(f)

        # Reconstruct input features
        input_features = {}
        for key, feat_info in config_dict.get("input_features", {}).items():
            input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=tuple(feat_info["shape"]),
            )

        config = RewardClassifierConfig(
            num_classes=config_dict["num_classes"],
            num_cameras=config_dict["num_cameras"],
            model_name=config_dict["model_name"],
            model_type=config_dict["model_type"],
            hidden_dim=config_dict["hidden_dim"],
            latent_dim=config_dict["latent_dim"],
            image_embedding_pooling_dim=config_dict["image_embedding_pooling_dim"],
            dropout_rate=config_dict["dropout_rate"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            input_features=input_features,
            output_features={},
        )

        classifier = Classifier(config)
        state_dict = torch.load(model_path / "classifier.pt", map_location=config.device, weights_only=True)
        classifier.load_state_dict(state_dict)
        classifier = classifier.to(config.device)
        classifier.eval()

        result = (classifier, config_dict)
        self._loaded_classifiers[name] = result
        return result

    def predict_reward(self, classifier, images: list, threshold: float = 0.5) -> float:
        """Predict reward for given camera images.

        Args:
            classifier: Loaded Classifier model
            images: List of image tensors, one per camera (C, H, W)
            threshold: Success threshold for binary classification

        Returns:
            Reward value (0.0 or 1.0)
        """
        with torch.no_grad():
            output = classifier.predict(images)
            prob = output.probabilities.item() if output.probabilities.numel() == 1 else output.probabilities[0, 1].item()
            return 1.0 if prob >= threshold else 0.0

    def list_classifiers(self) -> list:
        """List all available trained reward classifiers."""
        classifiers = []
        if not self.models_path.exists():
            return classifiers

        for model_dir in sorted(self.models_path.iterdir()):
            if not model_dir.is_dir():
                continue
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                classifiers.append(ClassifierInfo(
                    name=metadata.get("name", model_dir.name),
                    dataset_repo_id=metadata.get("dataset_repo_id", "unknown"),
                    num_cameras=metadata.get("num_cameras", 0),
                    num_success_frames=metadata.get("num_success_frames", 0),
                    num_failure_frames=metadata.get("num_failure_frames", 0),
                    accuracy=metadata.get("accuracy", 0.0),
                    created_at=metadata.get("created_at", ""),
                    path=str(model_dir),
                ))
            else:
                # Classifier without metadata (legacy)
                if (model_dir / "classifier.pt").exists():
                    classifiers.append(ClassifierInfo(
                        name=model_dir.name,
                        dataset_repo_id="unknown",
                        num_cameras=0,
                        num_success_frames=0,
                        num_failure_frames=0,
                        accuracy=0.0,
                        created_at="",
                        path=str(model_dir),
                    ))

        return classifiers

    def delete_classifier(self, name: str) -> bool:
        """Delete a trained reward classifier."""
        import shutil

        model_path = self.models_path / name
        if not model_path.exists():
            return False

        # Remove from cache
        self._loaded_classifiers.pop(name, None)

        shutil.rmtree(model_path)
        logger.info(f"[RewardClassifier] Deleted classifier: {name}")
        return True

    def get_training_status(self) -> dict:
        """Get current classifier training status."""
        return {
            "status": self.training_state.status,
            "epoch": self.training_state.epoch,
            "total_epochs": self.training_state.total_epochs,
            "loss": self.training_state.loss,
            "accuracy": self.training_state.accuracy,
            "error": self.training_state.error,
        }

    def stop_training(self):
        """Stop ongoing classifier training."""
        self._stop_event.set()
