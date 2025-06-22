#!/usr/bin/env python3
"""Example of a keyboard-controlled agent in the maze environment."""

import argparse
import time

import gymnasium as gym
import numpy as np
import pygame

import gymnasium_hardmaze  # noqa: F401


def main():
    """Run a keyboard-controlled agent in the maze environment."""
    parser = argparse.ArgumentParser(
        description="Run a keyboard-controlled agent in a maze environment"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="HardMaze-v0",
        choices=["HardMaze-v0"],
        help="Environment to use",
    )
    parser.add_argument(
        "--steps", type=int, default=30000, help="Maximum steps per episode"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    # Initialize pygame for keyboard input
    pygame.init()

    # Create environment (always render in human mode)
    # env = gym.make(args.env, render_mode="human")
    env = gym.make(
        "HardMaze-v0", render_mode="human"
    )  # Force HardMaze which should have walls

    # Set seed if provided
    if args.seed is not None:
        env.reset(seed=args.seed)
    else:
        env.reset()

    print("Use arrow keys to control the robot:")
    print("Left arrow: Turn left")
    print("Right arrow: Turn right")
    print("Up arrow: Move forward")
    print("Down arrow: Stop")
    print("Q: Quit")

    total_reward = 0.0

    for step in range(args.steps):
        # Default action (no movement)
        action = np.array([0, 0.4, 0], dtype=np.float32)

        # Process keyboard input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                env.close()
                return

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = np.array([0.05, 0, 0], dtype=np.float32)
        if keys[pygame.K_RIGHT]:
            action = np.array([0, 0, 0.05], dtype=np.float32)

        # Take a step
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        # Short pause for better control
        time.sleep(0.05)

        # Check if episode is done
        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps with reward {total_reward}")
            break

    env.close()


if __name__ == "__main__":
    main()
