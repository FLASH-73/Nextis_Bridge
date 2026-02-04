"""GVL (Generative Value Learning) Reward Service.

Uses Gemini API as a zero-shot reward model for HIL-SERL.
Instead of training a binary reward classifier, this sends
camera images to Gemini with a task description and gets
dense reward scores (0.0 → 1.0).

Advantages over binary classifier:
- Zero-shot: no training required
- Dense rewards: 0.0 → 1.0 instead of binary 0/1
- Handles longer task horizons
- Just needs a task description

Reference: GVL (Generative Value Learning) paper approach.
"""

import base64
import io
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GVLConfig:
    """Configuration for GVL reward service."""
    task_description: str = "Pick up the object and place it in the target location"
    query_interval: int = 5  # Query Gemini every N environment steps
    success_threshold: float = 0.85  # Reward above this = episode success
    model_name: str = "gemini-2.0-flash"  # Gemini model to use
    temperature: float = 0.1  # Low temperature for consistent scoring
    max_retries: int = 2  # Retries on API failure
    cache_duration_s: float = 0.5  # Cache rewards for this duration
    image_size: tuple = (224, 224)  # Resize images before sending


@dataclass
class GVLState:
    """State tracking for GVL service."""
    total_queries: int = 0
    total_cost_estimate: float = 0.0  # Rough cost tracking
    avg_latency_ms: float = 0.0
    last_reward: float = 0.0
    last_query_time: float = 0.0
    reward_history: list = field(default_factory=list)


class GVLRewardService:
    """Zero-shot reward prediction using Gemini API.

    Sends camera images + task description to Gemini and gets
    a dense reward score (0.0 to 1.0) indicating task progress.
    """

    def __init__(self, api_key: str = None, config: GVLConfig = None):
        """Initialize GVL reward service.

        Args:
            api_key: Google Gemini API key (or from GOOGLE_API_KEY env var)
            config: GVL configuration
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.config = config or GVLConfig()
        self.state = GVLState()

        # Lazy-init Gemini model
        self._model = None
        self._initialized = False

        # Reward cache to avoid redundant queries
        self._cached_reward = 0.0
        self._cache_time = 0.0
        self._step_counter = 0

        # Latency tracking
        self._latency_window = deque(maxlen=20)

    def _ensure_initialized(self):
        """Lazy-initialize Gemini model on first use."""
        if self._initialized:
            return

        if not self.api_key:
            raise ValueError("No Gemini API key. Set GOOGLE_API_KEY env var or pass api_key.")

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.config.model_name)
            self._initialized = True
            logger.info(f"[GVL] Initialized with model: {self.config.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini: {e}")

    def predict_reward(
        self,
        images: list,
        task_description: str = None,
        force_query: bool = False,
    ) -> float:
        """Predict reward from camera images using Gemini API.

        Args:
            images: List of numpy arrays (H, W, C) uint8 - one per camera
            task_description: Override task description (uses config default if None)
            force_query: Force API query even if cache is fresh

        Returns:
            Reward value between 0.0 and 1.0
        """
        self._step_counter += 1

        # Check if we should query (interval + cache)
        now = time.time()
        cache_fresh = (now - self._cache_time) < self.config.cache_duration_s
        on_interval = (self._step_counter % self.config.query_interval) == 0

        if not force_query and (cache_fresh or not on_interval):
            return self._cached_reward

        # Query Gemini
        try:
            self._ensure_initialized()
            reward = self._query_gemini(images, task_description or self.config.task_description)
            self._cached_reward = reward
            self._cache_time = now
            self.state.last_reward = reward
            self.state.reward_history.append(reward)
            return reward
        except Exception as e:
            logger.warning(f"[GVL] Query failed: {e}, using cached reward")
            return self._cached_reward

    def _query_gemini(self, images: list, task_description: str) -> float:
        """Send images to Gemini and get reward score."""
        import PIL.Image

        start_time = time.time()

        # Build prompt
        prompt = self._build_reward_prompt(task_description)

        # Convert images to PIL for Gemini
        pil_images = []
        for img in images:
            if img is None:
                continue
            if isinstance(img, np.ndarray):
                # Resize if needed
                if img.shape[:2] != self.config.image_size:
                    import cv2
                    img = cv2.resize(img, (self.config.image_size[1], self.config.image_size[0]))
                pil_img = PIL.Image.fromarray(img.astype(np.uint8))
                pil_images.append(pil_img)

        if not pil_images:
            return 0.0

        # Build content for Gemini
        content = pil_images + [prompt]

        # Query with retries
        reward = 0.0
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._model.generate_content(
                    content,
                    generation_config={
                        "temperature": self.config.temperature,
                        "max_output_tokens": 100,
                    },
                )
                reward = self._parse_reward_response(response.text)
                break
            except Exception as e:
                if attempt < self.config.max_retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                logger.warning(f"[GVL] All retries failed: {e}")
                return self._cached_reward

        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        self._latency_window.append(latency_ms)
        self.state.avg_latency_ms = sum(self._latency_window) / len(self._latency_window)
        self.state.total_queries += 1
        self.state.last_query_time = time.time()

        return reward

    def _build_reward_prompt(self, task_description: str) -> str:
        """Build the prompt for Gemini reward prediction."""
        return f"""You are a robot task reward evaluator. Given camera images from a robot workspace, score the task completion progress.

TASK: {task_description}

Score the current state on a scale from 0.0 to 1.0:
- 0.0 = Task has not started, robot is at rest
- 0.1-0.3 = Robot is approaching the target/object
- 0.3-0.5 = Robot has made contact or is grasping
- 0.5-0.7 = Robot is moving the object toward the goal
- 0.7-0.9 = Object is near the target location
- 0.9-1.0 = Task is completed or nearly completed

Respond with ONLY a JSON object: {{"reward": <float between 0.0 and 1.0>, "reason": "<brief explanation>"}}

Be precise and consistent. Focus on the actual task progress visible in the images."""

    def _parse_reward_response(self, response_text: str) -> float:
        """Parse the reward value from Gemini's response."""
        import re

        # Try JSON parsing first
        try:
            # Find JSON in response
            match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                reward = float(data.get("reward", 0.0))
                return max(0.0, min(1.0, reward))  # Clamp to [0, 1]
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Fallback: try to find a float in the response
        try:
            match = re.search(r'(\d+\.?\d*)', response_text)
            if match:
                value = float(match.group(1))
                if 0 <= value <= 1:
                    return value
                elif 0 <= value <= 100:
                    return value / 100.0
        except ValueError:
            pass

        logger.warning(f"[GVL] Could not parse reward from: {response_text[:100]}")
        return 0.0

    def is_success(self, reward: float) -> bool:
        """Check if the reward indicates task success."""
        return reward >= self.config.success_threshold

    def reset(self):
        """Reset per-episode state."""
        self._step_counter = 0
        self._cached_reward = 0.0
        self._cache_time = 0.0

    def get_status(self) -> dict:
        """Get GVL service status."""
        return {
            "initialized": self._initialized,
            "total_queries": self.state.total_queries,
            "avg_latency_ms": round(self.state.avg_latency_ms, 1),
            "last_reward": self.state.last_reward,
            "task_description": self.config.task_description,
            "model": self.config.model_name,
            "query_interval": self.config.query_interval,
            "success_threshold": self.config.success_threshold,
        }

    def update_config(self, **kwargs):
        """Update GVL config parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"[GVL] Updated {key} = {value}")
