"""Tests for the robot's sensor model.

These pin the properties that make the sensors usable, each of which was wrong
at some point and each of which fails quietly rather than loudly: a rangefinder
that can never read 0, a goal sensor that is dark for most of the maze, a radar
array rotated by one quadrant. None of these raise; they just make the task
unlearnable, so they need tests that assert on values rather than on shapes.
"""

import math

import pytest

from gymnasium_hardmaze.envs.components.goal import Goal
from gymnasium_hardmaze.envs.components.rangefinder import SENSOR_FLOOR, RangeFinder
from gymnasium_hardmaze.envs.components.robot import (
    DEFAULT_START_HEADING,
    SENSOR_HALF_SPREAD,
    Robot,
)
from gymnasium_hardmaze.envs.components.wall import Wall

ROBOT_SIZE = 10.5
ACTUAL_RANGE = 40.0


def make_finder(distance: float) -> RangeFinder:
    finder = RangeFinder(0.0, ACTUAL_RANGE, ROBOT_SIZE)
    finder.distance = distance
    return finder


# -- rangefinder normalization ------------------------------------------------


def test_rangefinder_reaches_one_at_the_far_end_of_its_range():
    """Nothing in sight must read exactly 1.0, not 40/50.5."""
    finder = make_finder(ACTUAL_RANGE + ROBOT_SIZE)
    assert finder.get_value() == pytest.approx(1.0)


def test_rangefinder_reaches_zero_at_the_robots_skin():
    """A wall touching the robot must read 0.0.

    Distances are measured from the centre, so without discounting the radius
    the lowest achievable reading would be 10.5/50.5, and the bottom fifth of
    the range would be unreachable.
    """
    assert make_finder(ROBOT_SIZE).get_value() == pytest.approx(0.0)


def test_rangefinder_saturates_rather_than_exceeding_one():
    assert make_finder(1000.0).get_value() == pytest.approx(1.0)


def test_rangefinder_floors_near_readings_to_zero():
    """Anything normalizing to <= 0.25 collapses to 0.0."""
    just_under = ROBOT_SIZE + ACTUAL_RANGE * SENSOR_FLOOR - 1e-9
    just_over = ROBOT_SIZE + ACTUAL_RANGE * (SENSOR_FLOOR + 0.05)
    assert make_finder(just_under).get_value() == 0.0
    assert make_finder(just_over).get_value() > 0.0


def test_rangefinder_max_range_includes_the_robot_radius():
    finder = RangeFinder(0.0, ACTUAL_RANGE, ROBOT_SIZE)
    assert finder.max_range == pytest.approx(ACTUAL_RANGE + ROBOT_SIZE)


def test_rangefinder_reports_a_wall_beyond_forty_units():
    """A wall between 40 and 50.5 units out is visible.

    Capping reported distance at the sensing range while casting the ray the
    full distance from the centre would discard exactly this band.
    """
    robot = Robot((0.0, 0.0), heading=0.0)
    walls = [Wall(45.0, -100.0, 45.0, 100.0)]
    robot.update_rangefinders(walls)
    middle = robot.rangefinders[len(robot.rangefinders) // 2]
    assert middle.get_value_raw() == pytest.approx(45.0)
    assert 0.0 < middle.get_value() < 1.0


# -- rangefinder geometry -----------------------------------------------------


def test_rangefinders_span_the_reference_arc_left_to_right():
    robot = Robot((0.0, 0.0))
    angles = [finder.angle for finder in robot.rangefinders]
    assert angles[0] == pytest.approx(SENSOR_HALF_SPREAD)
    assert angles[-1] == pytest.approx(-SENSOR_HALF_SPREAD)
    assert angles == sorted(angles, reverse=True), "must run left to right"
    assert angles[len(angles) // 2] == pytest.approx(0.0), "middle looks ahead"


def test_rangefinders_are_evenly_spaced():
    robot = Robot((0.0, 0.0), num_rangefinders=5)
    angles = [finder.angle for finder in robot.rangefinders]
    gaps = [a - b for a, b in zip(angles, angles[1:])]
    assert gaps == pytest.approx([gaps[0]] * len(gaps))


def test_a_single_rangefinder_looks_straight_ahead():
    robot = Robot((0.0, 0.0), num_rangefinders=1)
    assert robot.rangefinders[0].angle == pytest.approx(0.0)


# -- radar --------------------------------------------------------------------


def bearing_case(dx: float, dy: float, heading: float = 0.0):
    """Light the radars for a goal offset ``(dx, dy)`` from the robot."""
    robot = Robot((0.0, 0.0), heading=heading)
    robot.update_radars(Goal(dx, dy))
    return [radar.get_value() for radar in robot.radars]


def test_exactly_one_radar_is_lit():
    """The wedges tile the circle without gaps or overlap."""
    for degrees in range(0, 360, 7):
        radians = math.radians(degrees)
        # y is negated: the package measures bearings counterclockwise on a
        # screen whose y axis points down.
        lit = bearing_case(200.0 * math.cos(radians), -200.0 * math.sin(radians))
        assert sum(lit) == 1, f"bearing {degrees} lit {lit}"


def test_radar_zero_looks_dead_ahead():
    """Index 0 must straddle the heading, not start at it.

    The array is read as a compass, and rotating it by a quadrant relabels
    "the goal is ahead" as "the goal is off to one side".
    """
    assert bearing_case(200.0, 0.0)[0] == 1
    assert bearing_case(200.0, -40.0)[0] == 1, "just left of straight ahead"
    assert bearing_case(200.0, 40.0)[0] == 1, "just right of straight ahead"
    assert bearing_case(-200.0, 0.0)[0] == 0, "directly behind"


def test_radar_has_no_range_limit():
    """Distance must not gate detection.

    The goal starts roughly 236 units from the robot, so a 100-unit limit would
    leave the whole array dark for most of an episode.
    """
    near = bearing_case(30.0, 0.0)
    far = bearing_case(5000.0, 0.0)
    assert near == far
    assert sum(far) == 1


def test_radar_tracks_the_robots_heading():
    """A goal dead ahead stays on radar 0 whichever way the robot faces."""
    for degrees in range(0, 360, 15):
        heading = math.radians(degrees)
        dx, dy = 300.0 * math.cos(heading), -300.0 * math.sin(heading)
        assert bearing_case(dx, dy, heading=heading)[0] == 1


# -- starting pose ------------------------------------------------------------


def test_robot_starts_facing_up_the_map():
    """Forward thrust from the start must decrease y (move up the screen)."""
    robot = Robot((205.0, 387.0))
    assert robot.heading == pytest.approx(DEFAULT_START_HEADING)
    robot.decide_action([0.0, 1.0, 0.0], robot.time_step)
    robot.update_position()
    assert robot.location[1] < 387.0
    assert robot.location[0] == pytest.approx(205.0, abs=1e-9)
