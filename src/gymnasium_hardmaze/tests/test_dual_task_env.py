"""Dual task environment: world loading, scenario semantics, fitness."""

import math

import gymnasium as gym
import numpy as np
import pytest

import gymnasium_hardmaze  # noqa: F401  (registers DualTask-v0)
from gymnasium_hardmaze.envs.dual_task_env import (
    DEFAULT_EVALUATION_STEPS,
    FOOD_RADIUS,
    FOOD_WALL_MARGIN,
    GOAL_RADIUS,
    DualTaskEnvV0,
)


@pytest.fixture
def nav_env():
    env = gym.make("DualTask-v0", scenario="navigation").unwrapped
    yield env
    env.close()


@pytest.fixture
def food_env():
    env = gym.make("DualTask-v0", scenario="food_gathering").unwrapped
    yield env
    env.close()


# ---------------------------------------------------------------------------
# World loading
# ---------------------------------------------------------------------------


def test_world_matches_risis_environment_file(nav_env):
    """The packaged world is Risi's ENV_dual_task.xml, unmodified."""
    assert len(nav_env.corridor_walls) == 9
    assert nav_env.env.robot.location == (255.0, 393.0)
    assert (nav_env.env.goal.x, nav_env.env.goal.y) == (395.0, 184.0)
    assert nav_env.env.max_distance == pytest.approx(593.246155)
    assert nav_env.env.aoi_rectangle == (106.0, 416.0, 479.0, 350.0)


def test_max_distance_is_the_room_diagonal(nav_env):
    """The file's maxDistance is the AOI rectangle's diagonal -- the
    normaliser that scales both scenarios' distances into [0, 1]."""
    _, _, w, h = nav_env.env.aoi_rectangle
    assert math.hypot(w, h) == pytest.approx(nav_env.env.max_distance, abs=1e-4)


def test_food_sequence_is_the_four_room_pois(food_env):
    """POIPosition holds four food positions inside the room plus a copy of
    the navigation goal, which membership in the room filters out."""
    assert [(f.x, f.y) for f in food_env.foods] == [
        (171.0, 517.0),
        (398.0, 475.0),
        (204.0, 682.0),
        (399.0, 629.0),
    ]


def test_room_is_a_closed_box(food_env):
    """Figure 6b draws the food room as a closed box; its boundary must be
    real walls the rangefinders can see and the robot collides with."""
    assert len(food_env.room_walls) == 4
    corners = set()
    for wall in food_env.room_walls:
        corners.add((wall.ax, wall.ay))
        corners.add((wall.bx, wall.by))
    assert corners == {(106.0, 416.0), (585.0, 416.0), (585.0, 766.0), (106.0, 766.0)}


# ---------------------------------------------------------------------------
# Scenario semantics
# ---------------------------------------------------------------------------


def test_navigation_radars_stay_dark(nav_env):
    """The pie-slices are food sensors; with no food they read 0, per "using
    only its rangefinder sensors to detect walls"."""
    obs, _ = nav_env.reset()
    assert np.all(obs[5:] == 0.0)
    for _ in range(20):
        obs, _, _, _, _ = nav_env.step(np.array([0.3, 1.0, 0.6]))
        assert np.all(obs[5:] == 0.0)


def test_food_radar_is_a_compass_toward_the_current_food(food_env):
    """Exactly one wedge lights, and it tracks the *current* food item."""
    obs, info = food_env.reset()
    assert info["scenario"] == "food_gathering"
    assert obs[5:].sum() == 1.0


def test_food_scenario_starts_at_the_room_center(food_env):
    _, info = food_env.reset()
    assert info["robot_position"] == (345.5, 591.0)


def test_scenario_switches_via_reset_options(nav_env):
    _, info = nav_env.reset(options={"scenario": "food_gathering"})
    assert info["scenario"] == "food_gathering"
    _, info = nav_env.reset(options={"scenario": "navigation"})
    assert info["scenario"] == "navigation"


def test_unknown_scenario_is_rejected(nav_env):
    with pytest.raises(ValueError, match="scenario"):
        nav_env.reset(options={"scenario": "quadruped"})
    with pytest.raises(ValueError, match="scenario"):
        DualTaskEnvV0(scenario="quadruped")


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------


def _drive(env, target_fn, max_steps=DEFAULT_EVALUATION_STEPS):
    """Steer toward a moving target until the episode ends.

    ``target_fn(env)`` names the current destination each step, so a chase
    can retarget the moment a food item is eaten or a waypoint is passed.
    Rewards are accumulated so tests can check that their sum equals the
    scenario fitness the paper defines.
    """
    total = 0.0
    info = env._info()
    for _ in range(max_steps):
        tx, ty = target_fn(env)
        rx, ry = env.env.robot.location
        # update_position subtracts the heading's y component, so the world
        # bearing toward smaller y is +pi/2: negate dy when forming the angle.
        bearing = math.atan2(-(ty - ry), tx - rx)
        error = (bearing - env.env.robot.heading + math.pi) % (2 * math.pi) - math.pi
        left = 1.0 if error > 0.05 else 0.0
        right = 1.0 if error < -0.05 else 0.0
        _, reward, terminated, truncated, info = env.step(
            np.array([left, 1.0 if abs(error) < 0.8 else 0.0, right])
        )
        total += reward
        if terminated or truncated:
            return total, terminated, info
    return total, False, info


def test_reward_sum_is_the_paper_fitness_when_idle(nav_env):
    """Standing still, the reward sum telescopes to 1 - d(start, goal)/max."""
    nav_env.reset()
    total = 0.0
    for _ in range(5):
        _, reward, _, _, _ = nav_env.step(np.zeros(3))
        total += reward
    d = math.hypot(255.0 - 395.0, 393.0 - 184.0)
    assert total == pytest.approx(1.0 - d / nav_env.env.max_distance)


def test_collecting_all_food_scores_exactly_one(food_env):
    """A compass-following agent eats all four items; f_food = 1.0."""
    food_env.reset()

    def current_food(env):
        food = env.foods[min(env._foods_eaten, 3)]
        return (food.x, food.y)

    total, terminated, info = _drive(food_env, current_food)
    assert info["foods_eaten"] == 4
    assert terminated
    assert total == pytest.approx(1.0)


def test_food_items_are_eaten_in_sequence(food_env):
    """Driving at food #2 first earns nothing: only the current item counts."""
    food_env.reset()
    second = food_env.foods[1]
    _drive(food_env, lambda env: (second.x, second.y), max_steps=60)
    assert food_env._foods_eaten == 0


def test_navigation_goal_scores_exactly_one(nav_env):
    """Reaching the goal pins f_nav at 1.0 regardless of final distance."""
    nav_env.reset()

    # The corridor turns twice; waypoints trace the passage the walls of
    # ENV_dual_task.xml enclose -- up the start alcove, east along the
    # y ~ 295..340 corridor, into the shaft below the west wall's end
    # (that wall spans y 165..295, so the shaft is entered below y = 295),
    # then north to the goal.
    waypoints = [(253.0, 312.0), (330.0, 316.0), (395.0, 315.0), (395.0, 184.0)]

    progress = {"i": 0}

    def current_waypoint(env):
        rx, ry = env.env.robot.location
        i = progress["i"]
        while (
            i < len(waypoints) - 1
            and math.hypot(rx - waypoints[i][0], ry - waypoints[i][1]) < 12.0
        ):
            i += 1
            progress["i"] = i
        return waypoints[i]

    total, terminated, info = _drive(nav_env, current_waypoint)
    assert terminated, f"scripted driver never reached the goal: {info}"
    assert info["reached_goal"]
    assert total == pytest.approx(1.0)


def test_truncates_at_the_evaluation_limit(nav_env):
    nav_env.reset()
    truncated = False
    for _ in range(DEFAULT_EVALUATION_STEPS + 1):
        _, _, terminated, truncated, _ = nav_env.step(np.zeros(3))
        if terminated or truncated:
            break
    assert truncated
    assert nav_env._steps == DEFAULT_EVALUATION_STEPS


def test_runs_are_deterministic():
    first = gym.make("DualTask-v0", scenario="food_gathering").unwrapped
    second = gym.make("DualTask-v0", scenario="food_gathering").unwrapped
    a, _ = first.reset()
    b, _ = second.reset()
    assert np.array_equal(a, b)
    rng = np.random.default_rng(3)
    for _ in range(50):
        action = rng.random(3).astype(np.float32)
        oa, ra, ta, tra, _ = first.step(action)
        ob, rb, tb, trb, _ = second.step(action)
        assert np.array_equal(oa, ob)
        assert ra == rb and ta == tb and tra == trb
    first.close()
    second.close()


def test_goal_and_food_radii_are_the_documented_reconstructions():
    assert GOAL_RADIUS == 15.0
    assert FOOD_RADIUS == 20.0
    assert DEFAULT_EVALUATION_STEPS == 454


class TestFoodPlacement:
    """The paper's random food rule, and its reproducibility guarantees."""

    def test_fixed_is_the_default_and_replays_the_file(self):
        env = DualTaskEnvV0(scenario="food_gathering")
        assert env.food_placement == "fixed"
        env.reset(seed=0)
        first = [(f.x, f.y) for f in env._episode_foods]
        env.reset(seed=1)
        assert [(f.x, f.y) for f in env._episode_foods] == first
        assert first == [(f.x, f.y) for f in env.foods]
        env.close()

    def test_random_draws_inside_the_room_with_a_wall_margin(self):
        env = DualTaskEnvV0(scenario="food_gathering", food_placement="random")
        x, y, w, h = env.env.aoi_rectangle
        for seed in range(20):
            env.reset(seed=seed)
            assert len(env._episode_foods) == 4
            for food in env._episode_foods:
                assert x + FOOD_WALL_MARGIN <= food.x <= x + w - FOOD_WALL_MARGIN
                assert y + FOOD_WALL_MARGIN <= food.y <= y + h - FOOD_WALL_MARGIN
        env.close()

    def test_random_varies_across_seeds_but_repeats_within_one(self):
        env = DualTaskEnvV0(scenario="food_gathering", food_placement="random")
        env.reset(seed=7)
        seven = [(f.x, f.y) for f in env._episode_foods]
        env.reset(seed=8)
        assert [(f.x, f.y) for f in env._episode_foods] != seven
        env.reset(seed=7)
        assert [(f.x, f.y) for f in env._episode_foods] == seven
        env.close()

    def test_random_food_is_reachable_and_edible(self):
        """A driven robot can still eat randomly placed food."""
        env = DualTaskEnvV0(scenario="food_gathering", food_placement="random")
        env.reset(seed=3)
        _drive(
            env,
            lambda e: (
                e._episode_foods[e._foods_eaten].x,
                e._episode_foods[e._foods_eaten].y,
            ),
        )
        assert env._foods_eaten >= 1
        env.close()

    def test_rejects_unknown_placement(self):
        with pytest.raises(ValueError, match="food_placement"):
            DualTaskEnvV0(food_placement="everywhere")
