# Gymnasium HardMaze

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A maze navigation simulator for reinforcement learning research, compatible with the [Gymnasium](https://gymnasium.farama.org/) API.

<p align="center">
  <img src="https://github.com/Teaspoon-AI/gymnasium-hardmaze/raw/main/docs/images/gymnasium_hardmaze_preview.png" alt="Gynasium HardMaze Preview" width="600"/>
</p>

## Overview

Gymnasium HardMaze is a reimplementation of the 'hardmaze' environment used in several neuroevolution research papers. It provides a platform for training and evaluating reinforcement learning agents in navigation tasks of varying complexity.

Key features:
- XML-based maze configuration
- Fully compatible with Gymnasium API
- Deterministic when seeded for reproducible research
- Visualizations for debugging and demonstrations

## Installation

```bash
# Basic installation
pip install gymnasium-hardmaze

# For development
git clone https://github.com/Teaspoon-AI/gymnasium-hardmaze.git
cd HardMaze
pip install -e .
```

## Usage

### Basic Example

```python
import gymnasium as gym
import gymnasium_hardmaze
import numpy as np

# Create default maze environment
env = gym.make("HardMaze-v0", render_mode="human")

# Reset the environment
observation, info = env.reset(seed=42)

for _ in range(1000):
    # Take a random action
    action = env.action_space.sample()

    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # Check if episode is done
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

### Keyboard Control Example

The package includes a keyboard control script for testing environments:

```bash
# Run keyboard agent in the hard maze environment
python -m gymnasium_hardmaze.examples.keyboard_agent
```

## Available Environments

| Environment ID | Description |
|----------------|-------------|
| `HardMaze-v0` | Complex maze with walls and multiple points of interest |

### Observation Space

The observation space is a vector containing:
- Normalized rangefinder readings (distance to walls)
- Radar readings (binary detection of goal)

### Action Space

The action space is a 3-dimensional continuous space:
- `[left_motor, forward, right_motor]` with values between 0 and 1

### Rewards

The default reward function prioritizes reaching the goal, with partial rewards for visiting points of interest along the way.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{gymnasium_hardmaze,
  author = {Stefano Palmieri},
  title = {HardMaze: A Gymnasium-compatible Implementation of hardmaze environment},
  url = {https://github.com/Teaspoon-AI/gymnasium-hardmaze},
  year = {2025},
}
```

## Acknowledgements

This project uses a fitness function derived from Sebastian Risi's 2011 ES-HyperNEAT paper, ported to Python and adapted to work with the Gymnasium API.
