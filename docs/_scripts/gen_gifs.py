"""Generates a GIF for the HardMaze environment."""

from __future__ import annotations

import os
import random

import gymnasium
import numpy as np
from PIL import Image
from tqdm import tqdm

import gymnasium_hardmaze  # noqa: F401

# --- Constants ---
ENV_ID = "HardMaze-v0"
ENV_GROUP = "hardmaze"  # For directory naming
NUM_FRAMES = 400  # How many steps to record for the GIF
SEED = 42

# --- Setup Paths ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "docs", "_static", "videos", ENV_GROUP)
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, f"{ENV_GROUP}.gif")


def generate_gif():
    """Create, run, and save a GIF of the environment."""
    print(f"Generating GIF for {ENV_ID}...")

    try:
        env = gymnasium.make(ENV_ID, render_mode="rgb_array")

        # Check if the environment is renderable
        if "rgb_array" not in env.metadata["render_modes"]:
            print(f"Environment {ENV_ID} does not support 'rgb_array' rendering.")
            return

        # --- Collect frames ---
        frames = []
        state, info = env.reset(seed=SEED)
        random.seed(SEED)  # Seed the random module for action selection

        # Use tqdm for a progress bar
        pbar = tqdm(total=NUM_FRAMES)
        while len(frames) < NUM_FRAMES:
            # Simple heuristic agent: mostly move forward, occasionally turn
            if random.random() < 0.1:  # 10% chance to start a turn
                turn_direction = random.choice([-1, 1])
                turn_duration = random.randint(5, 15)
                for _ in range(turn_duration):
                    if len(frames) >= NUM_FRAMES:
                        break
                    # Action: [left, forward, right]
                    turn_action = (
                        np.array([0.1, 0.1, 0], dtype=np.float32)
                        if turn_direction == -1
                        else np.array([0, 0.1, 0.1], dtype=np.float32)
                    )
                    env.step(turn_action)
                    frame = env.render()

                    # Add this check!
                    if isinstance(frame, np.ndarray):
                        frames.append(Image.fromarray(frame))
                    pbar.update(1)

            # Move forward
            if len(frames) >= NUM_FRAMES:
                break
            forward_action = np.array([0, 0.5, 0], dtype=np.float32)
            observation, reward, terminated, truncated, info = env.step(forward_action)
            frame = env.render()
            if isinstance(frame, np.ndarray):
                frames.append(Image.fromarray(frame))
            pbar.update(1)

            if terminated or truncated:
                print("Episode finished early, resetting.")
                state, info = env.reset(seed=SEED)

        pbar.close()
        env.close()

        # --- Save the GIF ---
        print(f"Saving GIF to {output_path}...")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=50,  # Milliseconds per frame (50ms = 20 FPS)
            loop=0,  # Loop forever
        )
        print("GIF saved successfully!")

    except Exception as e:
        print(f"An error occurred while generating the GIF for {ENV_ID}: {e}")


if __name__ == "__main__":
    generate_gif()
