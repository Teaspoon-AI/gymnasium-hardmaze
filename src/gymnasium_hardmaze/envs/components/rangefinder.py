"""Rangefinder component for distance sensing in maze environments."""


class RangeFinder:
    """Distance sensor for detecting obstacles.

    Measures the distance to the nearest obstacle in its line of sight.

    Attributes:
        angle (float): Sensor orientation relative to robot heading (radians).
        max_range (float): Maximum detection distance.
        distance (float): Current measured distance, -1 if nothing detected.
    """

    def __init__(self, angle: float, max_range: float):
        """Initialize a rangefinder sensor.

        Args:
            angle: Sensor orientation relative to robot heading (radians).
            max_range: Maximum detection distance.
        """
        self.angle = angle
        self.max_range = max_range
        self.distance = -1.0

    def get_value(self) -> float:
        """Get the normalized distance value.

        Returns:
            float: Normalized distance in range [0, 1].
        """
        return self.distance / self.max_range

    def get_value_raw(self) -> float:
        """Get the raw distance value.

        Returns:
            float: Measured distance in environment units.
        """
        return self.distance
