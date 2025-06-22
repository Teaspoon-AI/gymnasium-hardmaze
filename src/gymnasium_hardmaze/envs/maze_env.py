"""Maze navigation environment for reinforcement learning."""

from typing import Any, Dict, List, Optional, Tuple, cast

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

import gymnasium_hardmaze.envs.utils as utils

from .environment import Environment
from .fitness_function_factory import FitnessFunctionFactory
from .rendering.renderer import Renderer

# These values are kept identical to the original C# implementation
# to ensure faithful replication of the benchmark's behavior.
GOAL_REACHED_THRESHOLD = 35.0
POI_REACHED_THRESHOLD = 20.0


class MazeEnv(gym.Env, EzPickle):
    """Maze navigation environment compatible with Gymnasium API.

    The robot must navigate through a maze to reach a goal, optionally
    visiting points of interest along the way.

    Attributes:
        metadata (Dict): Environment metadata for Gymnasium.
        env_file (str): Path to the XML environment file.
        render_mode (Optional[str]): Rendering mode ('human' or 'rgb_array').
        env (Environment): Loaded maze environment.
        pois_reached (List[bool]): Flags for reached points of interest.
        reached_goal (bool): Flag for reaching the goal.
        renderer (Optional[Renderer]): Renderer for visualization.
        action_space (spaces.Box): Action space for the robot.
        observation_space (spaces.Box): Observation space from sensors.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        env_file: str = "default_environment.xml",
        render_mode: Optional[str] = None,
        time_step=0.099,
    ):
        """Initialize the maze environment.

        Args:
            env_file: Path to the XML environment file.
            render_mode: Rendering mode ('human' or 'rgb_array').
            time_step: Time step between control updates (in seconds).
        """
        EzPickle.__init__(self, env_file, render_mode, time_step)
        self.env_file = env_file
        self.render_mode = render_mode

        # Load the environment from XML
        self.env = Environment(self.env_file)

        # Initialize state
        self.pois_reached = [False] * len(self.env.pois)
        self.reached_goal = False

        # Set up renderer if needed
        self.renderer: Optional[Renderer] = None
        if self.render_mode is not None:
            self._setup_renderer()

        # Define action and observation spaces
        # Actions: [left_motor, forward, right_motor]
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        # Observations: rangefinder values + radar values
        num_rangefinders = len(self.env.robot.rangefinders)
        num_radars = len(self.env.robot.radars)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_rangefinders + num_radars,), dtype=np.float32
        )

        self.time_step = 0.099

    def _setup_renderer(self) -> None:
        """Initialize the renderer."""
        if self.renderer is None:
            self.renderer = Renderer(600, 500, "Maze Simulator")

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options for reset.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Initial observation and info.
        """
        super().reset(seed=seed)

        # Reset environment state
        self.env.reset()
        self.pois_reached = [False] * len(self.env.pois)
        self.reached_goal = False

        # Update sensors
        self.env.robot.update_rangefinders(self.env.walls)
        self.env.robot.update_radars(self.env.goal)

        # Get initial observation
        observation = self._get_observation()
        info = {}

        # Render the initial state if needed
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action vector [left_motor, forward, right_motor].

        Returns:
            Tuple: (observation, reward, terminated, truncated, info)
        """
        # Execute action
        self._execute_action(action)

        # Check if we've reached the goal or new POIs
        reached_goal_this_step = self._update_state()

        # Get current observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward()

        # Check if episode is done
        terminated = reached_goal_this_step
        truncated = False

        # Additional info
        info = {
            "pois_reached": self.pois_reached,
            "reached_goal": self.reached_goal,
            "robot_position": self.env.robot.location,
        }

        # Render if needed
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            Optional[np.ndarray]: RGB array of the rendered frame if render_mode is 'rgb_array'.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self) -> Optional[np.ndarray]:
        """Render a frame of the environment.

        Returns:
            Optional[np.ndarray]: RGB array of the rendered frame if
            render_mode is 'rgb_array'.
        """
        if self.renderer is None:
            self._setup_renderer()

        renderer = cast(Renderer, self.renderer)

        renderer.render(self.env)

        if self.render_mode == "rgb_array":
            return renderer.get_rgb_array()
        return None

    def _get_observation(self) -> np.ndarray:
        """Get current observation from the environment.

        Returns:
            np.ndarray: Current sensor readings.
        """
        rangefinder_obs = list(self.env.robot.get_rangefinder_observations())
        radar_obs = list(self.env.robot.get_radar_observations())
        return np.array(rangefinder_obs + radar_obs, dtype=np.float32)

    def _execute_action(self, action: np.ndarray) -> None:
        """Execute the given action.

        Args:
            action: Action vector [left_motor, forward, right_motor].
        """
        # Clip to valid range [0, 1]
        action_clipped: np.ndarray = np.clip(action, 0, 1)

        outputs: List[float] = [float(v) for v in action_clipped.tolist()]

        # Update robot state
        self.env.robot.decide_action(outputs, self.time_step)
        self.env.robot.update_position()

        # Check for collision
        did_collide = self._check_collisions()
        if did_collide:
            self.env.robot.undo()

        # Update sensors
        self.env.robot.update_rangefinders(self.env.walls)
        self.env.robot.update_radars(self.env.goal)

    def _check_collisions(self) -> bool:
        """Check if robot collides with any walls.

        Returns:
            bool: True if collision detected, False otherwise.
        """
        for wall in self.env.walls:
            if utils.collide(wall, self.env.robot):
                return True
        return False

    def _update_state(self) -> bool:
        """Update environment state and check for goal.

        Returns:
            bool: True if goal reached, False otherwise.
        """
        # Check if reached goal
        if utils.robot_distance_from_goal(self.env) < GOAL_REACHED_THRESHOLD:
            self.reached_goal = True
            return True
        else:
            # Update POIs reached
            for i in range(len(self.env.pois)):
                if utils.robot_distance_from_poi(self.env, i) < POI_REACHED_THRESHOLD:
                    self.pois_reached[i] = True
            return False

    def _calculate_reward(self) -> float:
        """Calculate reward based on current state.

        Returns:
            float: Current reward value.
        """
        factory = FitnessFunctionFactory()
        fitness_function = factory.get_fitness_function("hardmaze")
        return fitness_function(self.pois_reached, self.env, self.reached_goal)

    def close(self) -> None:
        """Clean up resources."""
        if self.renderer:
            pygame.quit()

    # ―――――――――――――――――――――――――――――――――――――――――――――――――――
    #  Pickle helpers
    # ―――――――――――――――――――――――――――――――――――――――――――――――――――
    def __getstate__(self):
        """Remove pygame objects (self.renderer) before pickling.

        Everything else is trivially picklable thanks to EzPickle.
        """
        state = self.__dict__.copy()
        # Pygame surfaces are not picklable – drop them.
        state["renderer"] = None
        return state

    def __setstate__(self, state):
        """Restore attributes and rebuild renderer."""
        self.__dict__.update(state)
        if self.render_mode is not None and self.renderer is None:
            # Re-create the pygame window only on demand.
            self._setup_renderer()


class HardMazeEnvV0(MazeEnv):
    """A classic deceptive-reward maze environment for neuroevolution benchmarks.

    ## Description

    The Hard Maze environment is a classic deceptive-reward navigation task, originally
    introduced as a benchmark for neuroevolution and exploration algorithms. This
    environment is a reimplementation of the canonical 'hard maze' used in seminal
    research on Novelty Search and Quality-Diversity.

    The agent, a differential-drive robot, starts at the bottom of a maze and must
    navigate to a goal location at the top. The maze is "deceptive" because a purely
    goal-seeking (greedy) agent will be led into a dead-end, as the shortest path is
    blocked. To succeed, the agent must explore a much longer, seemingly suboptimal
    path that bypasses the trap. This makes the environment an excellent benchmark
    for evaluating an algorithm's ability to handle deception and perform robust
    exploration.

    ## Action Space

    The action space is a `Box(0, 1, (3,), float32)`. The 3-element vector
    corresponds to motor control signals: `[left_motor, forward_thrust, right_motor]`.

    - `forward_thrust`: Controls the robot's forward velocity.
    - `left_motor` and `right_motor`: Control turning. The turning speed is
      proportional to the difference `(left_motor - right_motor)`.

    ## Observation Space

    The observation space is a `Box(0, 1, (9,), float32)`, which is a concatenation of
    the robot's sensor readings:

    - **5 Rangefinders:** These sensors are distributed symmetrically across the robot's
      front, from -90 to +90 degrees. They return the normalized distance to the
      nearest wall in their line of sight. A value of `1.0` means no wall is
      detected within max range, while `0.0` indicates a wall is very close.
    - **4 Radar "Pie-Slices":** These sensors detect the goal. They divide the
      robot's forward view into four 90-degree arcs. Each sensor returns a binary
      value (`0.0` or `1.0`) indicating whether the goal is within its angular
      range and maximum detection distance.

    ## Rewards

    The reward function is sparse and designed to guide the agent through a series
    of waypoints, or Points of Interest (POIs), before reaching the final goal.
    This structure is what makes the environment a "hard" maze.

    - The agent receives a reward of `+1.0` for each POI visited *in the correct sequence*.
    - If the agent fails to reach the next POI in the sequence, its reward for the
      rest of the episode is based on its proximity to that *unreached* POI,
      calculated as `1.0 - (distance / max_distance)`.
    - A large reward of `10.0` is given upon reaching the final goal.

    A simple distance-to-goal reward function would fail in this environment, as it
    would reinforce moving towards the deceptive trap.

    ## Starting State

    The robot starts at a fixed position `(205, 387)` at the bottom of the maze, with a
    fixed heading of 90 degrees (facing upwards).

    ## Episode End

    The episode is considered `terminated` if the robot reaches the goal location
    (i.e., its distance to the goal is less than 35.0 units). There are no other
    termination conditions. `truncated` is always `False`.

    ## Arguments

    ```python
    import gymnasium as gym
    import gymnasium_hardmaze

    env = gym.make("HardMaze-v0", render_mode="human")
    ```

    - `env_file`: The XML file to load the maze layout from. Defaults to
      `hardmaze_env.xml`.
    - `render_mode`: The rendering mode, either `"human"` to display the
      environment or `"rgb_array"` to return frames as numpy arrays.
    - `time_step`: The duration of each simulation step in seconds. Defaults to `0.099`.

    """

    # Accept the full constructor signature used by MazeEnv / EzPickle
    def __init__(
        self,
        env_file: str = "hardmaze_env.xml",
        render_mode: Optional[str] = None,
        time_step: float = 0.099,
    ):
        """Initialize a hard maze environment."""
        # We still force the default XML unless the caller overrides it.
        super().__init__(
            env_file=env_file,
            render_mode=render_mode,
            time_step=time_step,
        )
