"""Rendering for maze environments."""

import math
from typing import Any, List

import numpy as np
import pygame

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
# Width of line
LINE_WIDTH = 3
POINT_RAD = 5


class Renderer:
    """Renderer for maze environments using Pygame.

    Handles drawing the robot, walls, points of interest, and goal.

    Attributes:
        disp_width (int): Display width in pixels.
        disp_height (int): Display height in pixels.
        screen (pygame.Surface): Pygame surface for rendering.
    """

    def __init__(self, width: int, height: int, title: str = "Maze Simulator"):
        """Initialize the renderer.

        Args:
            width: Display width in pixels.
            height: Display height in pixels.
            title: Window title.
        """
        pygame.init()
        self.disp_width = width
        self.disp_height = height

        self.screen = pygame.display.set_mode((self.disp_width, self.disp_height))
        self.screen.fill(WHITE)
        pygame.display.set_caption(title)

    def render_walls(self, walls: List[Any]) -> None:
        """Render all walls.

        Args:
            walls: List of Wall objects.
        """
        for wall in walls:
            color = wall.color
            ax = wall.ax
            ay = wall.ay
            bx = wall.bx
            by = wall.by
            pygame.draw.line(self.screen, color, [ax, ay], [bx, by], LINE_WIDTH)

    def render_rangefinder(
        self, x: float, y: float, radius: float, heading: float, finder: Any
    ) -> None:
        """Render a rangefinder sensor ray.

        Args:
            x: X-coordinate of the robot.
            y: Y-coordinate of the robot.
            radius: Robot radius.
            heading: Robot heading in radians.
            finder: RangeFinder object.
        """
        length = finder.distance
        angle = heading + finder.angle
        bx = x + (length) * math.cos(angle)
        by = y - (length) * math.sin(angle)
        pygame.draw.line(self.screen, GREEN, [x, y], [bx, by], LINE_WIDTH)

    def render_radar(
        self, x: float, y: float, radius: float, heading: float, radar: Any
    ) -> None:
        """Render a radar sensor arc.

        Args:
            x: X-coordinate of the robot.
            y: Y-coordinate of the robot.
            radius: Robot radius.
            heading: Robot heading in radians.
            radar: Radar object.
        """
        length = radar.max_range
        start_angle = heading + radar.start_angle
        end_angle = heading + radar.end_angle
        ray_color = (200, 50, 0) if radar.detecting > 0 else (200, 200, 200)

        # Draw an arc by approximating with multiple lines
        angle_step = 0.01
        current_angle = start_angle

        while current_angle < end_angle:
            bx = x + (length) * math.cos(current_angle)
            by = y - (length) * math.sin(current_angle)
            pygame.draw.line(self.screen, ray_color, [x, y], [bx, by], LINE_WIDTH)
            current_angle += angle_step

    def render_robot(self, robot: Any) -> None:
        """Render the robot and its sensors.

        Args:
            robot: Robot object.
        """
        x = int(robot.location[0])
        y = int(robot.location[1])
        radius = robot.default_robot_size

        # Render radar sensors
        for radar in robot.radars:
            self.render_radar(x, y, radius, robot.heading, radar)

        # Render rangefinder sensors
        for finder in robot.rangefinders:
            self.render_rangefinder(x, y, radius, robot.heading, finder)

        # Render robot body
        pygame.draw.circle(self.screen, WHITE, [x, y], int(radius))
        pygame.draw.circle(self.screen, (0, 0, 0), [x, y], int(radius), LINE_WIDTH)

    def render_goal(self, goal: Any) -> None:
        """Render the goal.

        Args:
            goal: Goal object.
        """
        x = int(goal.x)
        y = int(goal.y)
        pygame.draw.circle(self.screen, (0, 100, 0), [x, y], POINT_RAD, LINE_WIDTH)

    def render_pois(self, pois: List[Any]) -> None:
        """Render all points of interest.

        Args:
            pois: List of PointOfInterest objects.
        """
        for poi in pois:
            x = int(poi.x)
            y = int(poi.y)
            pygame.draw.circle(self.screen, (0, 0, 100), [x, y], POINT_RAD, LINE_WIDTH)

    def render_aoi(self, x: int, y: int, width: int, height: int) -> None:
        """Render area of interest rectangle.

        Args:
            x: X-coordinate of top-left corner.
            y: Y-coordinate of top-left corner.
            width: Width of rectangle.
            height: Height of rectangle.
        """
        pygame.draw.rect(self.screen, (255, 0, 0), [x, y, width, height], LINE_WIDTH)

    def render(self, env: Any) -> None:
        """Render the entire environment.

        Args:
            env: Environment object containing robot, walls, POIs, and goal.
        """
        self.screen.fill(WHITE)
        self.render_robot(env.robot)
        self.render_walls(env.walls)
        self.render_pois(env.pois)
        self.render_goal(env.goal)
        pygame.display.update()

    def get_rgb_array(self) -> np.ndarray:
        """Get the rendered scene as an RGB array.

        Returns:
            np.ndarray: RGB array of the current render.
        """
        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2)).astype(np.uint8)
