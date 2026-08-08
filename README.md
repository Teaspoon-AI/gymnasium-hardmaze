# Gymnasium HardMaze

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A maze navigation simulator for reinforcement learning research, compatible with the [Gymnasium](https://gymnasium.farama.org/) API.

<p align="center">
  <img src="https://raw.github.com/Teaspoon-AI/gymnasium-hardmaze/main/hardmaze-text.png" alt="Gynasium HardMaze Preview" width="500"/>
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
cd gymnasium-hardmaze
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
| `DualTask-v0` | Risi & Stanley's dual task: navigation and food gathering with one body |

### DualTask-v0

The dual task of Risi & Stanley, *An Enhanced Hypercube-Based Encoding for
Evolving the Placement, Density, and Connectivity of Neurons* (Artificial Life
18(4), 2012, Section 6). One robot — the same body as `HardMaze-v0` — faces two
independent scenarios, one per episode:

- **navigation** — drive a small walled corridor from start to goal using only
  the rangefinders; the pie-slice radars are food sensors and stay dark.
  Fitness: `1 − d_goal` (scaled), exactly 1.0 on reaching the goal.
- **food_gathering** — start at the centre of a walled room and collect four
  food items in sequence, steering by the radar compass that points at the
  current item. Fitness: `(n + (1 − d_food)) / 4`, exactly 1.0 after the
  fourth item.

Select the scenario at construction (`gym.make("DualTask-v0",
scenario="food_gathering")`) or per episode via
`reset(options={"scenario": ...})`. Rewards are per-step fitness deltas, so an
episode's reward sum is the scenario's fitness; the paper averages the two
scenarios and calls the domain solved at a combined 1.0.

The world geometry is Risi's own environment file (`ENV_dual_task.xml`,
provided by Sebastian Risi), shipped verbatim as `data/dualtask_env.xml`: the
nine corridor walls, start and goal, the four food positions, the food room,
and the `1 − d` normaliser (the room's diagonal). The evaluation length (45 s
at the 0.099 s timestep, 454 steps), the 15-unit goal radius and the 20-unit
food radius are reconstructions documented in
`envs/dual_task_env.py`, since the original experiment configuration was never
released.

### Observation Space

The observation space is a vector containing:
- Normalized rangefinder readings (distance to walls)
- Radar readings (binary detection of goal)

### Action Space

The action space is a 3-dimensional continuous space:
- `[left_motor, forward, right_motor]` with values between 0 and 1

### Rewards

The default reward function prioritizes reaching the goal, with partial rewards for visiting points of interest along the way.

## Similar Projects

Several other projects provide maze navigation environments for robotics and AI research:

### [Kheperax](https://github.com/adaptive-intelligent-robotics/Kheperax)
Kheperax is a JAX-based reimplementation of the fastsim simulator that simulates Khepera-like robots in 2D mazes. Key differences from gymnasium-hardmaze:
- **Hardware acceleration**: Fully implemented in JAX for GPU/TPU acceleration and massive parallelization
- **Robot model**: Simulates circular robots with 2 wheels, configurable laser and bumper sensors
- **Optimization focus**: Directly compatible with QDax library for Quality-Diversity optimization
- **Performance**: Designed for high-throughput evolutionary algorithms and population-based methods

### [fastsim_gym](https://github.com/mirandablue/fastsim_gym)
A Gym wrapper for the pyfastsim simulator, also implementing Lehman & Stanley's hard maze. Key differences:
- **Simulation backend**: Uses the pyfastsim C++ simulator for physics simulation
- **Robot configuration**: Features 3 lasers at specific angles (-π/4, 0, π/4) and two bumpers
- **Map format**: Requires binary PBM format maps with specific size constraints
- **API compatibility**: Uses the older Gym API rather than the newer Gymnasium standard

### How gymnasium-hardmaze differs:
- **Modern API**: Built specifically for the Gymnasium API with full compatibility
- **Pure Python**: No external simulator dependencies, making installation and debugging easier
- **Research focus**: Carefully ported from ES-HyperNEAT codebase for accurate replication

## Citation

If you use this software in your research, please cite:

```bibtex
@software{gymnasium-hardmaze,
  author = {Stefano Palmieri},
  title = {HardMaze: A Gymnasium-compatible Implementation of hardmaze environment},
  url = {https://github.com/Teaspoon-AI/gymnasium-hardmaze},
  year = {2025},
}
```

## Acknowledgements

The code in this project is derived from the original source code used in Sebastian Risi's 2011 ES-HyperNEAT paper.

```bibtex
@InProceedings{risi:gecco2011,
  author       = "Sebastian Risi and Kenneth O. Stanley",
  title        = "Enhancing ES-HyperNEAT to Evolve More Complex Regular Neural Networks",
  booktitle    = "Proceedings of the Genetic and Evolutionary Computation Conference (GECCO-2010)",
  year         = 2011,
  publisher    = "ACM",
  url          = "http://eplex.cs.ucf.edu/papers/risi_gecco11.pdf"
}
```
