"""Radar component for detecting the goal in maze environments."""


class Radar:
    """Radar sensor for detecting objects within an angular range.

    The radar detects whether a target is within its detection range
    and angular field of view.

    Attributes:
        start_angle (float): Starting angle of the detection arc (radians).
        end_angle (float): Ending angle of the detection arc (radians).
        max_range (float): Maximum detection distance.
        detecting (int): Binary value indicating detection (0 or 1).
    """

    def __init__(self, start_angle: float, end_angle: float, max_range: float = 100.0):
        """Initialize a radar sensor.

        Args:
            start_angle: Starting angle of the detection arc (radians).
            end_angle: Ending angle of the detection arc (radians).
            max_range: Maximum detection distance.
        """
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.max_range = max_range
        self.detecting = 0

    def get_value(self) -> int:
        """Get the current detection value.

        Returns:
            int: 1 if target detected, 0 otherwise.
        """
        return self.detecting
