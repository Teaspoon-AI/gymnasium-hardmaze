"""Wall component for maze environments."""

from typing import Tuple


class Wall:
    """Wall obstacle in the maze environment.

    Represented as a line segment between two points.

    Attributes:
        ax (float): X-coordinate of the first endpoint.
        ay (float): Y-coordinate of the first endpoint.
        bx (float): X-coordinate of the second endpoint.
        by (float): Y-coordinate of the second endpoint.
        color (Tuple[int, int, int]): RGB color for rendering.
    """

    def __init__(
        self,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        color: Tuple[int, int, int] = (0, 0, 0),
    ):
        """Initialize a wall.

        Args:
            ax: X-coordinate of the first endpoint.
            ay: Y-coordinate of the first endpoint.
            bx: X-coordinate of the second endpoint.
            by: Y-coordinate of the second endpoint.
            color: RGB color for rendering.
        """
        self.ax = ax
        self.ay = ay
        self.bx = bx
        self.by = by
        self.color = color

    def length_sq(self) -> float:
        """Calculate the squared length of the wall.

        Returns:
            float: Squared length of the wall.
        """
        from gymnasium_hardmaze.envs.utils import (  # Import here to avoid circular imports
            distance,
        )

        return distance(self.ax, self.ay, self.bx, self.by) ** 2
