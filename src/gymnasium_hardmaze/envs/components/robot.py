"""Robot component for maze navigation environments."""

import math
import random
import warnings
from typing import List, Tuple

from .radar import Radar
from .rangefinder import RangeFinder

# Half the angular spread of the rangefinder array, in radians.
#
# The reference spaces its rangefinders evenly across [-1.3398, 1.3398], i.e.
# +/-76.8 degrees, so the outermost pair look forward-and-out rather than
# straight out to the sides. Widening this to +/-90 degrees costs the robot
# forward coverage in the corridors it has to thread.
SENSOR_HALF_SPREAD = 1.3398

# Heading the robot starts at, in radians.
#
# The reference's environment file records `robot_heading` 0, and that is not
# the value it runs: `hardmaze_exp.xml` sets `overrideTeamFormation`, so
# `initializeRobots` reads the *experiment's* `robot_heading` of 270 degrees
# instead. Headings here are signed the opposite way (see `update_position`,
# which subtracts its y component), so the reference's 270 degrees is +pi/2
# here. Both point the robot up the map, toward the goal.
DEFAULT_START_HEADING = math.pi / 2


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
        heading: float = DEFAULT_START_HEADING,
    ):
        """Initialize a robot agent with sensors.

        Args:
            location: Initial (x, y) position.
            num_rangefinders: Number of rangefinder sensors.
            num_radars: Number of radar sensors.
            robot_size: Radius of the robot for collision detection.
            range_distance: Sensing range measured out from the robot's skin.
            time_step: Time increment for movement updates.
            heading_noise: Amount of noise in heading changes (0-100).
            effector_noise: Amount of noise in effector changes (0-100).
            sensor_noise: Amount of noise in sensor changes (0-100).
            heading: Initial heading in radians. See
                :data:`DEFAULT_START_HEADING`.
        """
        self.name = "MazeRobotPieSlice"
        self.default_speed = 25.0
        self.default_turn_speed = 9.0
        self.actualRange = range_distance
        self.default_robot_size = robot_size
        self.velocity = 0.0
        self.heading = heading
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

        # Rangefinders, spread evenly across the forward arc from left to right.
        # A lone sensor looks straight ahead rather than sitting at one end of
        # the arc, which is where an even spread of one would otherwise put it.
        self.rangefinders: List[RangeFinder] = []
        if num_rangefinders == 1:
            angles = [0.0]
        else:
            spacing = 2.0 * SENSOR_HALF_SPREAD / (num_rangefinders - 1)
            angles = [SENSOR_HALF_SPREAD - spacing * i for i in range(num_rangefinders)]
        for final_angle in angles:
            self.rangefinders.append(
                RangeFinder(final_angle, self.actualRange, self.default_robot_size)
            )

        # Radars, tiling the full circle. Index 0 straddles dead ahead, and the
        # rest follow round, so the array reads as a compass: front, then each
        # side in turn, then behind. The reference orders its wedges the same
        # way, which matters because HyperNEAT-style controllers read geometry
        # off the input layout -- shifting the array by one quadrant silently
        # relabels "the goal is ahead" as "the goal is off to one side".
        self.radars: List[Radar] = []
        for i in range(num_radars):
            between_angle = 2.0 * math.pi / num_radars
            start_angle = -between_angle / 2.0 - (between_angle * i)
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
            finder.distance = raycast(walls, finder, a1x, a1y, self.heading)

    def update_radars(self, goal) -> None:
        """Update radar readings based on goal position.

        Args:
            goal: Goal object to detect.
        """
        from gymnasium_hardmaze.envs.utils import (  # Import here to avoid circular imports
            radar_detect,
        )

        for radar in self.radars:
            start_angle = self.heading + radar.start_angle
            end_angle = self.heading + radar.end_angle
            x, y = self.location
            radar.detecting = int(radar_detect(goal, x, y, start_angle, end_angle))
