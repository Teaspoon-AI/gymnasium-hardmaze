"""Robot component for maze navigation environments."""

import math
import random
import warnings
from typing import List, Tuple

from .radar import Radar
from .rangefinder import RangeFinder


class Robot:
    """Robot agent that navigates through the maze environment.

    The robot has rangefinders for detecting walls and radar sensors for
    detecting the goal. It can move forward and turn based on control inputs.

    Attributes:
        name (str): Identifier for the robot.
        default_speed (float): Base movement speed.
        default_turn_speed (float): Base turning speed.
        actualRange (float): Maximum detection range for sensors.
        default_robot_size (float): Robot radius for collision detection.
        velocity (float): Current forward velocity.
        heading (float): Current heading in radians.
        location (Tuple[float, float]): Current (x, y) position.
        old_location (Tuple[float, float]): Previous (x, y) position.
        time_step (float): Time increment for movement updates.
        heading_noise (float): Amount of noise in heading changes.
        rangefinders (List[RangeFinder]): Distance sensors for detecting walls.
        radars (List[Radar]): Sensors for detecting the goal.
    """

    def __init__(
        self,
        location: Tuple[float, float],
        num_rangefinders: int = 5,
        num_radars: int = 4,
        robot_size: float = 10.5,
        range_distance: float = 40.0,
        time_step: float = 0.099,
        heading_noise: float = 0.0,
        effector_noise: float = 0.0,
        sensor_noise: float = 0.0,
    ):
        """Initialize a robot agent with sensors.

        Args:
            location: Initial (x, y) position.
            num_rangefinders: Number of rangefinder sensors.
            num_radars: Number of radar sensors.
            robot_size: Radius of the robot for collision detection.
            range_distance: Maximum detection range for sensors.
            time_step: Time increment for movement updates.
            heading_noise: Amount of noise in heading changes (0-100).
            effector_noise: Amount of noise in effector changes (0-100).
            sensor_noise: Amount of noise in sensor changes (0-100).
        """
        self.name = "MazeRobotPieSlice"
        self.default_speed = 25.0
        self.default_turn_speed = 9.0
        self.actualRange = range_distance
        self.default_robot_size = robot_size
        self.velocity = 0.0
        self.heading = math.pi / 2
        self.location = location
        self.old_location = location
        self.time_step = time_step
        self.heading_noise = heading_noise
        self.effector_noise = effector_noise
        self.sensor_noise = sensor_noise

        # Warn if non‑zero noise values are provided (feature not implemented yet)
        if effector_noise != 0.0:
            warnings.warn(
                "Effector noise was provided but is not yet implemented; it will be ignored.",
                RuntimeWarning,
            )

        if sensor_noise != 0.0:
            warnings.warn(
                "Sensor noise was provided but is not yet implemented; it will be ignored.",
                RuntimeWarning,
            )

        # Initialize rangefinders
        self.rangefinders: List[RangeFinder] = []
        for i in range(num_rangefinders):
            between_angle = math.pi / 4.0
            final_angle = math.pi / 2 - (between_angle * i)
            self.rangefinders.append(RangeFinder(final_angle, self.actualRange))

        # Initialize radars
        self.radars: List[Radar] = []
        for i in range(num_radars):
            between_angle = math.pi / 2.0
            start_angle = math.pi / 4 - (between_angle * i)
            self.radars.append(Radar(start_angle, start_angle + between_angle))

    def rand_bool(self) -> bool:
        """Return a random boolean value."""
        return bool(random.getrandbits(1))

    def undo(self) -> None:
        """Revert to the previous position."""
        self.location = self.old_location

    def noisy_heading(self) -> float:
        """Add noise to the current heading.

        Returns:
            float: Heading with added noise.
        """
        if self.heading_noise <= 0:
            return self.heading

        handedness = 1 if self.rand_bool() else -1
        max_noise = int(self.heading_noise)
        noise_factor = 0.1 * handedness * random.randint(0, max_noise) / 100.0
        return self.heading + noise_factor

    def decide_action(self, outputs: List[float], time_step: float) -> None:
        """Update robot state based on control outputs.

        Args:
            outputs: Control signals [left_turn, forward, right_turn].
            time_step: Time increment for this action.
        """
        speed = 20.0
        turn_speed = 4.28

        self.velocity = speed * outputs[1]
        self.heading += (outputs[0] - outputs[2]) * turn_speed * time_step

    def update_position(self) -> None:
        """Update position based on current velocity and heading."""
        self.old_location = self.location

        # Apply heading noise
        temp_heading = self.noisy_heading()
        self.heading = temp_heading

        # Calculate movement vector
        dx = math.cos(temp_heading) * self.velocity * self.time_step
        dy = math.sin(temp_heading) * self.velocity * self.time_step

        # Update position
        x = self.location[0] + dx
        y = self.location[1] - dy
        self.location = (x, y)

    def get_rangefinder_observations(self) -> List[float]:
        """Get normalized readings from all rangefinders.

        Returns:
            List[float]: Normalized distance values (0-1) for each rangefinder.
        """
        return [finder.get_value() for finder in self.rangefinders]

    def get_radar_observations(self) -> List[float]:
        """Get readings from all radar sensors.

        Returns:
            List[float]: Binary detection values (0 or 1) for each radar.
        """
        return [radar.get_value() for radar in self.radars]

    def update_rangefinders(self, walls: List) -> None:
        """Update rangefinder readings based on wall positions.

        Args:
            walls: List of wall objects to detect.
        """
        from gymnasium_hardmaze.envs.utils import (  # Import here to avoid circular imports
            raycast,
        )

        for finder in self.rangefinders:
            a1x = self.location[0]
            a1y = self.location[1]
            finder.distance = raycast(
                walls, finder, a1x, a1y, self.heading, self.default_robot_size
            )

    def update_radars(self, goal) -> None:
        """Update radar readings based on goal position.

        Args:
            goal: Goal object to detect.
        """
        from gymnasium_hardmaze.envs.utils import (  # Import here to avoid circular imports
            radar_detect,
        )

        for radar in self.radars:
            r_range = radar.max_range
            start_angle = self.heading + radar.start_angle
            end_angle = self.heading + radar.end_angle
            x, y = self.location
            radar.detecting = int(
                radar_detect(goal, x, y, start_angle, end_angle, r_range)
            )
