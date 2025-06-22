"""Utility functions for maze environments."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from gymnasium_hardmaze.envs.components.rangefinder import RangeFinder
from gymnasium_hardmaze.envs.components.wall import Wall

# avoid circular‑import at runtime: bring these in only for static analysis
if TYPE_CHECKING:
    from gymnasium_hardmaze.envs.components.goal import Goal
    from gymnasium_hardmaze.envs.components.robot import Robot
    from gymnasium_hardmaze.envs.environment import Environment


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        x1: X-coordinate of the first point.
        y1: Y-coordinate of the first point.
        x2: X-coordinate of the second point.
        y2: Y-coordinate of the second point.

    Returns:
        float: Euclidean distance between the points.
    """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def robot_distance_from_poi(env: Environment, i: int) -> float:
    """Calculate distance between robot and a point of interest.

    Args:
        env: Environment containing robot and POIs.
        i: Index of the POI.

    Returns:
        float: Distance between robot and POI.
    """
    robot_x = env.robot.location[0]
    robot_y = env.robot.location[1]
    poi_x = env.pois[i].x
    poi_y = env.pois[i].y
    return distance(robot_x, robot_y, poi_x, poi_y)


def robot_distance_from_goal(env: Environment) -> float:
    """Calculate distance between robot and goal.

    Args:
        env: Environment containing robot and goal.

    Returns:
        float: Distance between robot and goal.
    """
    robot_x = env.robot.location[0]
    robot_y = env.robot.location[1]
    goal_x = env.goal.x
    goal_y = env.goal.y
    return distance(robot_x, robot_y, goal_x, goal_y)


def collide(wall: Wall, robot: Robot) -> bool:
    """Check if robot collides with a wall.

    Args:
        wall: Wall object with endpoints.
        robot: Robot object with location and size.

    Returns:
        bool: True if collision detected, False otherwise.
    """
    a1x = wall.ax
    a1y = wall.ay
    a2x = wall.bx
    a2y = wall.by
    bx = robot.location[0]
    by = robot.location[1]

    rad = robot.default_robot_size
    length_sq = wall.length_sq()

    # Avoid division by zero
    if length_sq < 1e-6:
        return distance(bx, by, a1x, a1y) < rad

    r = ((bx - a1x) * (a2x - a1x) + (by - a1y) * (a2y - a1y)) / length_sq

    # Calculate closest point on line segment
    px = a1x + r * (a2x - a1x)
    py = a1y + r * (a2y - a1y)
    rad_sq = rad * rad

    # Check if closest point is on the line segment
    if 0.0 <= r <= 1.0:
        return distance(bx, by, px, py) ** 2 < rad_sq

    # Check endpoints if closest point is outside the segment
    d1 = distance(bx, by, a1x, a1y) ** 2
    d2 = distance(bx, by, a2x, a2y) ** 2
    return d1 < rad_sq or d2 < rad_sq


def intersection(
    A: tuple[float, float],
    B: tuple[float, float],
    C: tuple[float, float],
    D: tuple[float, float],
) -> tuple[float, float] | None:
    """Calculate intersection point of two line segments.

    Args:
        A: First point of first line segment.
        B: Second point of first line segment.
        C: First point of second line segment.
        D: Second point of second line segment.

    Returns:
        Optional[Tuple[float, float]]: Intersection point or None if no intersection.
    """
    rTop = (A[1] - C[1]) * (D[0] - C[0]) - (A[0] - C[0]) * (D[1] - C[1])
    rBot = (B[0] - A[0]) * (D[1] - C[1]) - (B[1] - A[1]) * (D[0] - C[0])

    sTop = (A[1] - C[1]) * (B[0] - A[0]) - (A[0] - C[0]) * (B[1] - A[1])
    sBot = (B[0] - A[0]) * (D[1] - C[1]) - (B[1] - A[1]) * (D[0] - C[0])

    if abs(rBot) < 1e-6 or abs(sBot) < 1e-6:
        return None

    r = rTop / rBot
    s = sTop / sBot

    if 0 < r < 1 and 0 < s < 1:
        ptx = A[0] + r * (B[0] - A[0])
        pty = A[1] + r * (B[1] - A[1])
        return (ptx, pty)

    return None


def raycast(
    walls: list[Wall],
    finder: RangeFinder,
    a1x: float,
    a1y: float,
    heading: float,
    radius: float,
) -> float:
    """Cast a ray and find the distance to the nearest wall.

    Args:
        walls: List of walls to check for intersection.
        finder: RangeFinder object with angle and max_range.
        a1x: X-coordinate of ray origin.
        a1y: Y-coordinate of ray origin.
        heading: Current heading angle in radians.
        radius: Radius to add to the origin point.

    Returns:
        float: Distance to the nearest wall or max_range if none found.
    """
    shortest_distance = finder.max_range
    length = radius + finder.max_range
    angle = heading + finder.angle
    a2x = a1x + length * math.cos(angle)
    a2y = a1y - length * math.sin(angle)

    for wall in walls:
        p1 = (a1x, a1y)
        p2 = (a2x, a2y)
        p3 = (wall.ax, wall.ay)
        p4 = (wall.bx, wall.by)
        point = intersection(p1, p2, p3, p4)

        if point is not None:
            curr_distance = distance(a1x, a1y, point[0], point[1])
            if curr_distance < shortest_distance:
                shortest_distance = curr_distance

    return shortest_distance


def normalize(angle: float) -> float:
    """Normalize angle to be between -π and π.

    Args:
        angle: Angle in radians.

    Returns:
        float: Normalized angle in radians.
    """
    width = 2 * math.pi
    offset = angle

    return offset - (math.floor(offset / width) * width)


def radar_detect(
    goal: Goal,  # Use Goal class
    x: float,
    y: float,
    start_angle: float,
    end_angle: float,
    r_range: float,
) -> float:
    """Check if goal is within radar detection range and angle.

    Args:
        goal: Goal object with x, y coordinates.
        x: X-coordinate of radar origin.
        y: Y-coordinate of radar origin.
        start_angle: Start angle of radar arc in radians.
        end_angle: End angle of radar arc in radians.
        r_range: Maximum detection range.

    Returns:
        float: 1.0 if goal detected, 0.0 otherwise.
    """
    start_angle = normalize(start_angle)
    end_angle = normalize(end_angle)

    if distance(x, y, goal.x, goal.y) > r_range:
        return 0.0

    angle = normalize(math.atan2(-(goal.y - y), (goal.x - x)))

    # Handle the case where the arc crosses the -π/π boundary
    if start_angle > end_angle:
        return 1.0 if (start_angle <= angle or angle < end_angle) else 0.0
    else:
        return 1.0 if (start_angle <= angle and angle < end_angle) else 0.0
