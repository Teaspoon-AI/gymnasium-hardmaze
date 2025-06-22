"""Point of Interest component for maze environments."""


class PointOfInterest:
    """Point of Interest (POI) in the maze environment.

    Represents optional waypoints for the robot to visit.

    Attributes:
        x (float): X-coordinate of the POI.
        y (float): Y-coordinate of the POI.
    """

    def __init__(self, x: float, y: float):
        """Initialize a point of interest.

        Args:
            x: X-coordinate of the POI.
            y: Y-coordinate of the POI.
        """
        self.x = x
        self.y = y
