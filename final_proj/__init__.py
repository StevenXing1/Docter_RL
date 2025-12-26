"""
Doctor RL: Reinforcement Learning for Hypotension Management

This package provides a custom Gym environment for training RL agents
to manage blood pressure in simulated medical scenarios.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

from gym.envs.registration import register

# Register custom environment
register(
    id="final_proj/RLDocter_v0",
    entry_point="final_proj.envs:DocterEnv",
    max_episode_steps=300,
    reward_threshold=100.0,
)

# Make environment easily accessible
from final_proj.envs.docterEnv import DocterEnv

__all__ = ["DocterEnv"]
