---
hide-toc: true
firstpage:
lastpage:
---

```{project-logo} _static/img/hardmaze-text.png
:alt: Hardmaze Logo
```

```{project-heading}
A reimplementation of the canonical *Hard Maze* deception benchmark used in quality-diversity and neuro-evolution research.
```

## Installation

```python
pip install gymnasium_hardmaze
```

## Usage

The Gymnasium interface allows to initialize and interact with the Hardmaze environment as follows:

```python
import gymnasium as gym

env = gym.make("HardMaze-v0", render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(1000):
   action = policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
```

## Citation

To cite this project please use:

```bibtex
@software{gymnasium_hardmaze,
  author = {Stefano Palmieri},
  title = {Maze Simulator: A Gymnasium-compatible Implementation of hardmaze environment},
  url = {https://github.com/Farama-Foundation/HardMaze},
  year = {2025},
}
```

```{toctree}
:hidden:
:caption: Introduction
content/basic_usage
```

```{toctree}
:hidden:
:caption: Environments
environments/hardmaze/index
```

```{toctree}
:hidden:
:caption: Development
release_notes
Github <https://github.com/Farama-Foundation/HardMaze>
```
