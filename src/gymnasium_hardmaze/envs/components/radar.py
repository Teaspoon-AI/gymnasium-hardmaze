"""Radar component for detecting the goal in maze environments."""

# How far the detection wedge is drawn, in environment units. Rendering only --
# the radar has no range limit (see `Radar`).
DISPLAY_RANGE = 100.0


class Radar:
    """Radar sensor reporting which quadrant the goal lies in.

    Detection is purely angular and has no distance limit: the radar fires
    whenever the goal falls inside its wedge, however far away it is. So exactly
    one radar of a full set is lit at any moment, and together they act as a
    coarse compass pointing at the goal rather than a proximity sensor.

    This matches the reference implementation, whose `PieSliceSensorArray.update`
    computes the bearing to the goal and lights the matching wedge without
    consulting distance at all. A range limit would leave the whole array dark
    for most of the hard maze -- the goal sits roughly 236 units from the start
    -- depriving a controller of any signal about where it is meant to go.

    Attributes:
        start_angle (float): Starting angle of the detection arc (radians),
            relative to the robot's heading.
        end_angle (float): Ending angle of the detection arc (radians),
            relative to the robot's heading.
        display_range (float): Length the wedge is drawn at. Rendering only;
            it does not affect detection.
        detecting (int): Binary value indicating detection (0 or 1).
    """

    def __init__(
        self,
        start_angle: float,
        end_angle: float,
        display_range: float = DISPLAY_RANGE,
    ):
        """Initialize a radar sensor.

        Args:
            start_angle: Starting angle of the detection arc (radians).
            end_angle: Ending angle of the detection arc (radians).
            display_range: Length the wedge is drawn at; rendering only.
        """
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.display_range = display_range
        self.detecting = 0

    def get_value(self) -> int:
        """Get the current detection value.

        Returns:
            int: 1 if the goal is inside this radar's wedge, 0 otherwise.
        """
        return self.detecting
