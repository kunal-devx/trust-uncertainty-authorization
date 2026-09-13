"""
VERIFY simulation for graduated authorization.

Simulates the execution outcomes of proposed actions when placed under VERIFY:
- Benign actions always complete successfully.
- Harmful actions are caught/blocked with probability p_catch.

Uses explicitly supplied NumPy Generator for reproducibility.
No global RNG state.
"""

from typing import Tuple
import numpy as np


def verify_action(
    is_harmful: bool,
    p_catch: float,
    rng: np.random.Generator,
) -> bool:
    """
    Simulate the execution outcome of an action under VERIFY.

    Args:
        is_harmful: True if the action is harmful, False if benign.
        p_catch: Probability that a harmful action is caught [0, 1].
        rng: NumPy Generator for stochastic outcomes.

    Returns:
        True if the action ultimately executed, False if blocked/caught.

    Raises:
        ValueError: If p_catch is outside [0, 1].
        TypeError: If p_catch is not numeric or rng is not a Generator.

    Behavior:
        - Benign actions (is_harmful=False) always return True.
          Benign actions do not consume randomness.
        - Harmful actions (is_harmful=True) are caught with probability p_catch.
          If caught, returns False. Otherwise returns True.
    """
    # Validate p_catch
    if not isinstance(p_catch, (int, float)):
        raise TypeError(f"p_catch must be numeric, got {type(p_catch).__name__}")
    if p_catch < 0.0 or p_catch > 1.0:
        raise ValueError(f"p_catch must be in [0, 1], got {p_catch}")

    # Validate RNG
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be np.random.Generator, got {type(rng).__name__}")

    # Benign actions always execute; no randomness consumed
    if not is_harmful:
        return True

    # Harmful actions: caught with probability p_catch
    # If caught (random draw < p_catch), action is blocked (return False)
    # Otherwise action executes (return True)
    random_draw = rng.uniform()
    return random_draw >= p_catch
