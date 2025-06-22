"""Goal component for maze environments."""


class Goal:
    """Goal target in the maze environment.

    Represents the target location the robot needs to reach.

    Attributes:
        x (float): X-coordinate of the goal.
        y (float): Y-coordinate of the goal.
    """

    def __init__(self, x: float, y: float):
        """Initialize a goal.

        Args:
            x: X-coordinate of the goal.
            y: Y-coordinate of the goal.
        """
        self.x = x
        self.y = y
