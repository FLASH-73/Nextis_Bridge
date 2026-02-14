"""RL types and configuration dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import PROJECT_ROOT, MODELS_DIR, DATASETS_DIR

_APP_ROOT = PROJECT_ROOT
_DEFAULT_MODELS_PATH = MODELS_DIR
_DEFAULT_DATASETS_PATH = DATASETS_DIR


@dataclass
class RLConfig:
    """Configuration for RL training session."""
    # Reward source
    reward_source: str = "sarm"  # "sarm" (recommended), "gvl", or "classifier"
    reward_classifier_name: str = ""  # Name of trained classifier (if using classifier)
    sarm_model_name: str = ""  # Name of trained SARM model (if using SARM)
    task_description: str = ""  # Task description for GVL or SARM

    # Dataset for offline buffer
    dataset_repo_id: str = ""

    # Episode settings
    max_episodes: int = 100
    max_steps_per_episode: int = 300
    fps: int = 30

    # SAC hyperparameters
    discount: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
    batch_size: int = 32
    utd_ratio: int = 1  # Update-to-data ratio

    # Buffer settings
    online_buffer_capacity: int = 100_000
    offline_buffer_capacity: int = 100_000

    # Safety
    movement_scale: float = 0.5  # 0.1-1.0 safety limiter

    # Reset
    reset_position: dict = field(default_factory=dict)  # Motor positions for reset
    reset_time_s: float = 3.0

    # Intervention detection
    intervention_velocity_threshold: float = 0.05

    # GVL settings
    gvl_query_interval: int = 5
    gvl_success_threshold: float = 0.85

    # Training
    warmup_steps: int = 50  # Steps before starting learning
    save_interval_episodes: int = 10

    # Image size for observations
    image_size: tuple = (224, 224)


@dataclass
class RLTrainingState:
    """Live training state for monitoring."""
    status: str = "idle"  # idle, initializing, running, paused, completed, failed
    episode: int = 0
    total_episodes: int = 0
    episode_step: int = 0
    training_step: int = 0
    loss_critic: float = 0.0
    loss_actor: float = 0.0
    loss_temperature: float = 0.0
    avg_reward: float = 0.0
    intervention_rate: float = 0.0
    online_buffer_size: int = 0
    offline_buffer_size: int = 0
    current_reward: float = 0.0
    is_human_intervening: bool = False
    error: str = ""
    total_interventions: int = 0
    total_autonomous_steps: int = 0
    gvl_queries: int = 0
    gvl_avg_latency_ms: float = 0.0
    episode_rewards: list = field(default_factory=list)  # Per-episode total reward
