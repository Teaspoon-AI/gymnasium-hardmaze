---
layout: "contents"
title: Basic Usage
firstpage:
---


## Basic Usage

There is a UI application which allows you to manually control the agent with the arrow keys:

```bash
python -m gymnasium_hardmaze.examples.keyboard_agent
```

## Installation

HardMaze call be installed via `pip`:

```bash
python3 -m pip install gymnasium_hardmaze
```

To modify the codebase or contribute to HardMaze, you would need to install HardMaze from source:

```bash
git clone https://github.com/Farama-Foundation/HardMaze.git
cd HardMaze
python3 -m pip install -e .
```

If using uv for development install `dev` and `docs` groups.

```bash
uv sync --all-groups
```

Now you can build the docs:

```bash
uv run make dirhtml
```

And run tests:
```bash
uv run pytest
```
