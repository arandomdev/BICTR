from typing import Any

import numpy as np
from scipy import constants  # type: ignore


def freeSpacePathloss[T: Any](
    freq: np.floating[T] | float, dist: np.floating[T]
) -> np.floating[T]:
    """Computes the free space pathloss of the E field, i.e without the square"""
    return constants.speed_of_light / (4 * np.pi * dist * freq)
