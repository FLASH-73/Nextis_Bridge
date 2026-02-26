"""Placeholder RL learner interface for SERL-style online learning.

The full extraction from rl/service.py is complex and deferred.  The
DeploymentRuntime works perfectly without it — HIL_SERL mode just won't
have online learning until this is implemented.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class RLLearner:
    """Placeholder interface for online RL learner.

    TODO: Extract SAC learner logic from rl/service.py including:
    - Online and offline replay buffers
    - Learner thread with SAC updates
    - SARM / GVL / classifier reward computation
    - Multiprocessing queues for transitions and param sync
    """

    def __init__(self, config: Any = None):
        self.config = config
        logger.info("RLLearner created (placeholder — not yet implemented)")

    def add_transition(
        self,
        obs: Dict,
        action: Any,
        reward: float,
        next_obs: Dict,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        raise NotImplementedError("RLLearner.add_transition not yet implemented")

    def compute_reward(self, obs: Dict) -> float:
        """Compute reward for the current observation."""
        raise NotImplementedError("RLLearner.compute_reward not yet implemented")

    def get_metrics(self) -> Dict:
        """Return current training metrics."""
        return {}

    def stop(self) -> None:
        """Stop the learner thread."""
        pass
