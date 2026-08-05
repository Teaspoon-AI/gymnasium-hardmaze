"""Rangefinder component for distance sensing in maze environments."""

# Readings at or below this, after normalization, are reported as 0.0.
#
# From the reference implementation's `Khepera3RobotModel.doAction`, which
# clamps each rangefinder to 1.0 and then applies `if (inputs[j] <= floor)
# inputs[j] = 0.0f` with `floor = 0.25f`. The near field is therefore not
# merely "close", it is invisible, and a controller cannot distinguish a wall
# at the robot's skin from a wall 10 units out.
SENSOR_FLOOR = 0.25


class RangeFinder:
    """Distance sensor for detecting obstacles.

    Measures the distance to the nearest obstacle in its line of sight.

    Distances are measured from the robot's centre, so a reading is never
    smaller than the robot's own radius. Normalization subtracts that radius
    before scaling, which is what puts a reading of 0.0 at the robot's skin and
    1.0 at the far end of its sensing range, rather than leaving the bottom
    quarter of the range permanently unreachable.

    Attributes:
        angle (float): Sensor orientation relative to robot heading (radians).
        actual_range (float): Sensing range measured out from the robot's skin.
        robot_size (float): Radius of the robot the sensor is mounted on.
        max_range (float): Furthest measurable distance from the robot's centre,
            i.e. ``actual_range + robot_size``. This is how far rays are cast
            and the value a reading saturates at.
        distance (float): Current measured distance from the centre, -1 if the
            sensor has not been updated yet.
    """

    def __init__(self, angle: float, actual_range: float, robot_size: float):
        """Initialize a rangefinder sensor.

        Args:
            angle: Sensor orientation relative to robot heading (radians).
            actual_range: Sensing range out from the robot's skin.
            robot_size: Radius of the robot carrying the sensor.
        """
        self.angle = angle
        self.actual_range = actual_range
        self.robot_size = robot_size
        self.max_range = actual_range + robot_size
        self.distance = -1.0

    def get_value(self) -> float:
        """Get the normalized distance value.

        Discounts the robot's own radius, saturates at 1.0, and floors small
        readings to 0.0 (see :data:`SENSOR_FLOOR`).

        Returns:
            float: Normalized distance in range [0, 1].
        """
        value = min((self.distance - self.robot_size) / self.actual_range, 1.0)
        if value <= SENSOR_FLOOR:
            value = 0.0
        return value

    def get_value_raw(self) -> float:
        """Get the raw distance value.

        Returns:
            float: Measured distance from the robot's centre, in environment units.
        """
        return self.distance
