"""Fitness function factory for maze environments."""

from typing import Any, Callable, Dict, List


def hardmaze_fitness_function(
    reached_pois: List[bool], env: Any, reached_goal: bool
) -> float:
    """Calculate fitness for the hardmaze environment.

    Args:
        reached_pois: List of booleans indicating which POIs have been reached.
        env: Environment object with robot and POI data.
        reached_goal: Boolean indicating if goal has been reached.

    Returns:
        float: Calculated fitness value.
    """
    from gymnasium_hardmaze.envs.utils import robot_distance_from_poi

    fitness = 0.0

    if reached_goal:
        return 10.0

    for i in range(len(reached_pois)):
        # Add 1.0 if the current POI has been reached
        if reached_pois[i]:
            fitness += 1.0
        else:
            discount = robot_distance_from_poi(env, i) / 650.0
            fitness += 1.0 - discount
            break

    return fitness


class FitnessFunctionFactory:
    """Factory for creating fitness functions.

    Provides different fitness functions for evaluating robot performance.
    """

    def __init__(self):
        """Initialize the fitness function factory."""
        self._fitness_functions: Dict[str, Callable] = {
            "hardmaze": hardmaze_fitness_function,
        }

    def get_fitness_function(self, name: str) -> Callable:
        """Get a fitness function by name.

        Args:
            name: Name of the fitness function.

        Returns:
            Callable: Requested fitness function.

        Raises:
            ValueError: If fitness function name is not recognized.
        """
        if name in self._fitness_functions:
            return self._fitness_functions[name]
        else:
            raise ValueError(f"Unknown fitness function: {name}")

    def register_fitness_function(self, name: str, function: Callable) -> None:
        """Register a new fitness function.

        Args:
            name: Name for the fitness function.
            function: Fitness function to register.
        """
        self._fitness_functions[name] = function
