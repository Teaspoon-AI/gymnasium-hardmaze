"""MazeSimulator: A maze navigation simulator for reinforcement learning research.

This package provides maze navigation environments compatible with the Gymnasium API.
The environments simulate a robot navigating through various mazes with different
layouts and objectives.
"""

from gymnasium.envs.registration import register

# Register the environments
register(
    id="HardMaze-v0",
    entry_point="gymnasium_hardmaze.envs:HardMazeEnvV0",
)

__version__ = "0.1.0"
