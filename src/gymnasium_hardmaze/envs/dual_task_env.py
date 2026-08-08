"""Dual task environment: navigation and food gathering with one body.

The dual task of Risi & Stanley, *An Enhanced Hypercube-Based Encoding for
Evolving the Placement, Density, and Connectivity of Neurons* (Artificial Life
18(4), 2012, Section 6; also GECCO 2011), pairs two independent scenarios that
share the maze robot:

* **Navigation** -- drive from a start point to a goal point through a small
  walled corridor, sensing only with the five rangefinders. The pie-slice
  radars stay dark: in this domain they are food sensors, and there is no
  food. Fitness is ``f_nav = 1 - d_g``, where ``d_g`` is the distance to the
  goal at the end of the evaluation, scaled into [0, 1]; reaching the goal
  scores exactly 1.0.
* **Food gathering** -- start at the centre of a walled room in which one
  piece of food exists at a time, up to four in sequence. The radars act as a
  compass toward the current food item; the rangefinders see only the room's
  boundary. Fitness is ``f_food = (n + (1 - d_f)) / 4`` with ``n`` the number
  of food items collected and ``d_f`` the scaled distance to the next item at
  the end of the evaluation; collecting all four scores exactly 1.0.

The paper evaluates a controller on both scenarios and averages the two
fitness values; the domain is solved at a combined fitness of 1.0. This
environment exposes one scenario per episode -- select it with
``reset(options={"scenario": ...})`` or the constructor -- and reports the
scenario's fitness components in ``info`` so callers can combine them.

The world geometry comes verbatim from Risi's own environment file
(``ENV_dual_task.xml``, provided by Sebastian Risi): nine corridor walls, the
start and goal points, the food positions, and the food room. Two details are
worth knowing:

* The file's ``maxDistance`` (593.246...) is exactly the diagonal of its
  ``AOIRectangle`` -- the food room -- and is the paper's "[scaled] into the
  range [0, 1]" normaliser for both scenarios.
* The file's ``POIPosition`` list holds the four food positions (inside the
  room) plus a copy of the navigation goal; the food sequence is therefore
  deterministic, which is what makes evaluations reproducible. The paper's
  "placed at another random location" describes the agent's-eye view, not the
  evaluation harness.

Reconstructed values, stated as such because the original experiment
configuration was never released: the evaluation time reuses the platform's
shipped maze default (45 s at a 0.099 s timestep, 454 steps per scenario);
the goal radius is 15 units, from the ``SingleGoalPoint`` fitness function
the environment file names; the food radius is 20 units, the platform's
point-of-interest radius.
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from .components.point_of_interest import PointOfInterest
from .components.wall import Wall
from .environment import Environment

# Radius within which the navigation goal counts as reached. From the
# SingleGoalPoint fitness function named by ENV_dual_task.xml.
GOAL_RADIUS = 15.0

# Radius within which a food item counts as eaten. The platform's
# point-of-interest radius; the food positions are the file's POIs.
FOOD_RADIUS = 20.0

# Steps per scenario: the platform's shipped evaluation time (45 s) over its
# timestep (0.099 s), as in the hard maze experiment configuration.
DEFAULT_EVALUATION_STEPS = int(45.0 / 0.099)

SCENARIOS = ("navigation", "food_gathering")


class DualTaskEnvV0(gym.Env, EzPickle):
    """The dual task: navigation or food gathering, one scenario per episode.

    ## Action Space

    `Box(0, 1, (3,), float32)`: `[left_motor, forward_thrust, right_motor]`,
    identical to `HardMaze-v0`. The robot moves ``20 * forward`` units per
    step and turns by ``(left - right) * 18`` degrees.

    ## Observation Space

    `Box(0, 1, (9,), float32)`: five rangefinders followed by four pie-slice
    radar wedges, identical in layout to `HardMaze-v0`. In the navigation
    scenario the radar wedges are always 0 (they sense food, and there is
    none); in the food-gathering scenario exactly one wedge is 1, pointing at
    the current food item.

    ## Rewards

    Per-step deltas of the scenario's fitness, so the episode's reward sum
    equals the paper's scenario fitness: ``1 - d_g`` (navigation, 1.0 on
    reaching the goal) or ``(n + (1 - d_f)) / 4`` (food gathering, 1.0 once
    all four items are collected). Combine the two scenarios' sums yourself --
    the paper averages them.

    ## Episode End

    `terminated` when the goal is reached (navigation) or the fourth food
    item is collected (food gathering); `truncated` after 454 steps (45 s at
    the 0.099 s timestep).

    ## Arguments

    - `scenario`: `"navigation"` or `"food_gathering"`; also selectable per
      episode via `reset(options={"scenario": ...})`.
    - `env_file`: World file; defaults to Risi's `dualtask_env.xml`.
    - `max_steps`: Steps per scenario before truncation.
    """

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(
        self,
        env_file: str = "dualtask_env.xml",
        scenario: str = "navigation",
        max_steps: int = DEFAULT_EVALUATION_STEPS,
        render_mode: Optional[str] = None,
    ):
        EzPickle.__init__(self, env_file, scenario, max_steps, render_mode)
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
        if render_mode is not None:
            raise NotImplementedError("DualTask-v0 does not render yet")
        self.env_file = env_file
        self.scenario = scenario
        self.max_steps = int(max_steps)
        self.render_mode = render_mode

        self.env = Environment(self.env_file)
        if self.env.aoi_rectangle is None:
            raise ValueError(f"{env_file} has no AOIRectangle; not a dual-task world")
        if self.env.max_distance <= 0.0:
            raise ValueError(f"{env_file} has no maxDistance; not a dual-task world")

        self.corridor_walls: List[Wall] = list(self.env.walls)
        self.room_walls: List[Wall] = self._room_boundary(self.env.aoi_rectangle)
        self.foods: List[PointOfInterest] = self._food_sequence()
        if len(self.foods) != 4:
            raise ValueError(
                f"expected 4 food positions inside the room, found {len(self.foods)}"
            )

        x, y, w, h = self.env.aoi_rectangle
        self.room_center: Tuple[float, float] = (x + w / 2.0, y + h / 2.0)

        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        num_sensors = len(self.env.robot.rangefinders) + len(self.env.robot.radars)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_sensors,), dtype=np.float32
        )

        self.time_step = 0.099
        self._steps = 0
        self._foods_eaten = 0
        self._reached_goal = False
        self._previous_fitness = 0.0

    # ------------------------------------------------------------------
    # World derivation
    # ------------------------------------------------------------------
    def _room_boundary(self, rect: Tuple[float, float, float, float]) -> List[Wall]:
        """The food room's four boundary walls.

        The environment file stores the room only as its AOIRectangle; figure
        6b of the paper draws it as a closed box, so the boundary is realised
        as four walls the rangefinders can see and the robot collides with.
        """
        x, y, w, h = rect
        return [
            Wall(x, y, x + w, y),
            Wall(x + w, y, x + w, y + h),
            Wall(x + w, y + h, x, y + h),
            Wall(x, y + h, x, y),
        ]

    def _food_sequence(self) -> List[PointOfInterest]:
        """The four food positions, in file order.

        The file's POI list also carries a copy of the navigation goal;
        membership in the room rectangle is what separates food from goal.
        """
        assert self.env.aoi_rectangle is not None
        x, y, w, h = self.env.aoi_rectangle
        return [
            poi
            for poi in self.env.pois
            if x <= poi.x <= x + w and y <= poi.y <= y + h
        ]

    # ------------------------------------------------------------------
    # Scenario state
    # ------------------------------------------------------------------
    @property
    def _walls(self) -> List[Wall]:
        if self.scenario == "navigation":
            return self.corridor_walls
        return self.room_walls

    @property
    def _current_food(self) -> Optional[PointOfInterest]:
        if self.scenario != "food_gathering" or self._foods_eaten >= len(self.foods):
            return None
        return self.foods[self._foods_eaten]

    def _scenario_fitness(self) -> float:
        """The paper's fitness for the current scenario, at this instant."""
        max_distance = self.env.max_distance
        rx, ry = self.env.robot.location
        if self.scenario == "navigation":
            if self._reached_goal:
                return 1.0
            goal = self.env.goal
            d = float(np.hypot(rx - goal.x, ry - goal.y))
            return max(0.0, 1.0 - d / max_distance)
        if self._foods_eaten >= len(self.foods):
            return 1.0
        food = self.foods[self._foods_eaten]
        d = float(np.hypot(rx - food.x, ry - food.y))
        return (self._foods_eaten + max(0.0, 1.0 - d / max_distance)) / 4.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if options and "scenario" in options:
            scenario = options["scenario"]
            if scenario not in SCENARIOS:
                raise ValueError(
                    f"scenario must be one of {SCENARIOS}, got {scenario!r}"
                )
            self.scenario = scenario

        self.env.reset()
        if self.scenario == "food_gathering":
            self.env.robot.location = self.room_center
            self.env.robot.old_location = self.room_center

        self._steps = 0
        self._foods_eaten = 0
        self._reached_goal = False
        self._previous_fitness = 0.0

        self._update_sensors()
        return self._get_observation(), self._info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_clipped = np.clip(action, 0, 1)
        outputs = [float(v) for v in action_clipped.tolist()]

        self.env.robot.decide_action(outputs, self.time_step)
        self.env.robot.update_position()
        if self._check_collisions():
            self.env.robot.undo()

        self._steps += 1
        terminated = self._update_state()
        self._update_sensors()

        fitness = self._scenario_fitness()
        reward = fitness - self._previous_fitness
        self._previous_fitness = fitness

        truncated = not terminated and self._steps >= self.max_steps
        return self._get_observation(), reward, terminated, truncated, self._info()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _check_collisions(self) -> bool:
        from gymnasium_hardmaze.envs.utils import collide

        return any(collide(wall, self.env.robot) for wall in self._walls)

    def _update_state(self) -> bool:
        rx, ry = self.env.robot.location
        if self.scenario == "navigation":
            goal = self.env.goal
            if float(np.hypot(rx - goal.x, ry - goal.y)) < GOAL_RADIUS:
                self._reached_goal = True
            return self._reached_goal

        food = self._current_food
        if food is not None and float(np.hypot(rx - food.x, ry - food.y)) < FOOD_RADIUS:
            self._foods_eaten += 1
        return self._foods_eaten >= len(self.foods)

    def _update_sensors(self) -> None:
        self.env.robot.update_rangefinders(self._walls)
        food = self._current_food
        if food is not None:
            self.env.robot.update_radars(food)
        else:
            # Navigation, or all food eaten: the food compass has nothing to
            # point at. "using only its rangefinder sensors" -- the radars
            # sense food in this domain, not the goal.
            for radar in self.env.robot.radars:
                radar.detecting = 0

    def _get_observation(self) -> np.ndarray:
        rangefinders = list(self.env.robot.get_rangefinder_observations())
        radars = list(self.env.robot.get_radar_observations())
        return np.array(rangefinders + radars, dtype=np.float32)

    def _info(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "fitness": self._scenario_fitness(),
            "foods_eaten": self._foods_eaten,
            "reached_goal": self._reached_goal,
            "robot_position": self.env.robot.location,
            "steps": self._steps,
        }

    def close(self) -> None:
        pass
