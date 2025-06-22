"""Environment loading and management for maze navigation."""

from __future__ import annotations

import importlib.resources
import os
from typing import Optional, cast

import bs4
from bs4.element import Tag

from .components.goal import Goal
from .components.point_of_interest import PointOfInterest
from .components.robot import Robot
from .components.wall import Wall


class Environment:
    """Environment for maze navigation."""

    def __init__(self, xml_file: str):
        """Initialize the environment from an XML file.

        Args:
            xml_file: Path to the XML file or resource name under `gymnasium_hardmaze/data/`.

        Raises:
            FileNotFoundError: If the XML file cannot be found.
        """
        # Resolve file path whether it's a relative path or a package resource
        if os.path.isfile(xml_file):
            with open(xml_file, encoding="utf-8") as f:
                self._xml_content = f.read()
        else:
            try:
                # Use importlib.resources for robust access to package data
                self._xml_content = (
                    importlib.resources.files("gymnasium_hardmaze")
                    .joinpath(f"data/{xml_file}")
                    .read_text(encoding="utf-8")
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Could not find environment file or resource: {xml_file}"
                ) from e

        # Parse XML and initialize components
        soup = bs4.BeautifulSoup(self._xml_content, "xml")
        self.populate_walls(soup)
        self.populate_pois(soup)
        self.init_robot(soup)
        self.init_goal(soup)

        group_orientation_tag = cast(Optional[Tag], soup.find("group_orientation"))
        self.group_orientation: float = (
            float(group_orientation_tag.get_text()) if group_orientation_tag else 0.0
        )

        robot_spacing_tag = cast(Optional[Tag], soup.find("robot_spacing"))
        self.robot_spacing: float = (
            float(robot_spacing_tag.get_text()) if robot_spacing_tag else 30.0
        )

        heading_tag = cast(Optional[Tag], soup.find("robot_heading"))
        self.heading: float = float(heading_tag.get_text()) if heading_tag else 0.0

        seed_tag = cast(Optional[Tag], soup.find("seed"))
        self.seed: int = int(seed_tag.get_text()) if seed_tag else 0

    def init_robot(self, soup: bs4.BeautifulSoup) -> None:
        """Initialize the robot from the XML soup.

        Args:
            soup: Parsed BeautifulSoup object containing the environment data.
        """
        start = cast(Optional[Tag], soup.find("start_point"))
        if start is None:
            self.robot = Robot((400.0, 700.0))
            return

        x_tag = cast(Optional[Tag], start.find("x"))
        y_tag = cast(Optional[Tag], start.find("y"))
        try:
            start_x = float(x_tag.get_text()) if x_tag else 400.0
            start_y = float(y_tag.get_text()) if y_tag else 700.0
        except (AttributeError, ValueError):
            start_x, start_y = 400.0, 700.0

        self.robot = Robot((start_x, start_y))

    def init_goal(self, soup: bs4.BeautifulSoup) -> None:
        """Initialize the goal from the XML soup.

        Args:
            soup: Parsed BeautifulSoup object containing the environment data.
        """
        goal_point = cast(Optional[Tag], soup.find("goal_point"))
        if goal_point is None:
            self.goal = Goal(100.0, 100.0)
            return

        x_tag = cast(Optional[Tag], goal_point.find("x"))
        y_tag = cast(Optional[Tag], goal_point.find("y"))
        try:
            goal_x = float(x_tag.get_text()) if x_tag else 100.0
            goal_y = float(y_tag.get_text()) if y_tag else 100.0
        except (AttributeError, ValueError):
            goal_x, goal_y = 100.0, 100.0

        self.goal = Goal(goal_x, goal_y)

    def populate_walls(self, soup: bs4.BeautifulSoup) -> None:
        """Parse and populate walls from the XML soup.

        Args:
            soup: Parsed BeautifulSoup object containing the environment data.
        """
        self.walls: list[Wall] = []

        for cur_wall_pe in soup.find_all("Wall"):
            cur_wall = cast(Tag, cur_wall_pe)
            line_tag = cast(Optional[Tag], cur_wall.find("line"))
            if line_tag is None:
                continue

            p1 = cast(Optional[Tag], line_tag.find("p1"))
            p2 = cast(Optional[Tag], line_tag.find("p2"))
            if p1 is None or p2 is None:
                continue

            try:
                ax = float(cast(Tag, p1.find("x")).get_text())
                ay = float(cast(Tag, p1.find("y")).get_text())
                bx = float(cast(Tag, p2.find("x")).get_text())
                by = float(cast(Tag, p2.find("y")).get_text())
                self.walls.append(Wall(ax, ay, bx, by))
            except (AttributeError, ValueError) as err:
                print(f"Error parsing wall: {err}")

    def populate_pois(self, soup: bs4.BeautifulSoup) -> None:
        """Parse and populate points of interest (POIs) from the XML soup.

        Args:
            soup: Parsed BeautifulSoup object containing the environment data.
        """
        self.pois: list[PointOfInterest] = []

        poi_position = cast(Optional[Tag], soup.find("POIPosition"))
        if poi_position is None:
            return

        for point_pe in poi_position.find_all("Point"):
            point = cast(Tag, point_pe)
            try:
                x_tag = cast(Optional[Tag], point.find("X"))
                y_tag = cast(Optional[Tag], point.find("Y"))
                if x_tag is None or y_tag is None:
                    continue
                x = float(x_tag.get_text())
                y = float(y_tag.get_text())
                self.pois.append(PointOfInterest(x, y))
            except (AttributeError, ValueError) as err:
                print(f"Error parsing POI: {err}")

    def reset(self) -> None:
        """Reset the environment to its initial state.

        Reloads the environment XML and resets the robot to its start position.
        """
        # Re-parse the stored XML content to get a fresh start state
        soup = bs4.BeautifulSoup(self._xml_content, "xml")
        self.init_robot(soup)
