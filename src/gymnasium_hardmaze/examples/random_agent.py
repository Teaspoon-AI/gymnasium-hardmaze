#!/usr/bin/env python3
"""Example of a random agent in the maze environment."""

import argparse
import time

import gymnasium as gym

import gymnasium_hardmaze  # noqa: F401


def main():
    """Run a random agent in the maze environment."""
    parser = argparse.ArgumentParser(
        description="Run a random agent in a maze environment"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="HardMaze-v0",
        choices=["HardMaze-v0"],
        help="Environment to use",
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to run"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Maximum steps per episode"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    args = parser.parse_args()

    # Create environment
    render_mode = "human" if args.render else None
    env = gym.make(args.env, render_mode=render_mode)

    # Set seed if provided
    if args.seed is not None:
        env.reset(seed=args.seed)

    for episode in range(args.episodes):
        print(f"Episode {episode + 1}/{args.episodes}")
        observation, info = env.reset()
        total_reward = 0.0

        for step in range(args.steps):
            # Random action
            action = env.action_space.sample()

            # Take a step
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            # Optionally pause to make rendering visible
            if args.render:
                time.sleep(0.01)

            # Check if episode is done
            if terminated or truncated:
                print(
                    f"Episode finished after {step + 1} steps with reward {total_reward}"
                )
                break
        else:
            print(
                f"Episode reached maximum steps ({args.steps}) with reward {total_reward}"
            )

    env.close()


if __name__ == "__main__":
    main()
