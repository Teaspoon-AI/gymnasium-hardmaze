"""Basic tests for the maze environment."""

import gymnasium as gym
import numpy as np
import pytest  # noqa: F401

import gymnasium_hardmaze  # noqa: F401


def test_environment_creation():
    """Test that the environment can be created."""
    env = gym.make("HardMaze-v0")
    assert env is not None
    env.close()


def test_reset_and_step():
    """Test that reset and step functions work properly."""
    env = gym.make("HardMaze-v0")
    observation, info = env.reset(seed=42)

    # Check observation shape
    assert observation.shape == env.observation_space.shape

    # Take a random action
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # Check return values
    assert observation.shape == env.observation_space.shape
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    env.close()


def test_determinism():
    """Test that the environment is deterministic with a fixed seed."""
    env1 = gym.make("HardMaze-v0")
    env2 = gym.make("HardMaze-v0")

    # Reset both environments with the same seed
    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    # Check that initial observations are the same
    np.testing.assert_array_equal(obs1, obs2)

    # Take the same actions and check for same results
    for _ in range(10):
        action = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        obs1, reward1, term1, trunc1, info1 = env1.step(action)
        obs2, reward2, term2, trunc2, info2 = env2.step(action)

        np.testing.assert_array_equal(obs1, obs2)
        assert reward1 == reward2
        assert term1 == term2
        assert trunc1 == trunc2

    env1.close()
    env2.close()


def test_pickling_roundtrip():
    import pickle

    import gymnasium as gym

    env = gym.make("HardMaze-v0")
    dumped = pickle.dumps(env)  # should not raise
    loaded = pickle.loads(dumped)
    obs1, _ = env.reset(seed=123)
    obs2, _ = loaded.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs2)
    env.close()
    loaded.close()


if __name__ == "__main__":
    test_environment_creation()
    test_reset_and_step()
    test_determinism()
    print("All tests passed!")
