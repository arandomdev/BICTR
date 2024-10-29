from dataclasses import dataclass

import numpy as np


class Point(object):
    """Point in 3D space."""

    def __init__(self, x: int, y: int, z: int) -> None:
        self._coord = np.array((x, y, z))

    def distance(self, other: "Point") -> float:
        return float(np.linalg.norm(self._coord - other._coord))

    pass


@dataclass
class DelayPath(object):
    delay: float  # Propagation delay in seconds
    gain: float  # Gain in dB
    pass


class PathGenerator(object):
    """Computes the multipath delay paths."""

    def __init__(
        self,
        samplingFreq: int,
        txLoc: Point,
        rxLoc: Point,
        reflectionPoint: Point,
    ) -> None:
        self._samplingFreq = samplingFreq

        self._txLoc = txLoc
        self._rxLoc = rxLoc
        self._reflectionPoint = reflectionPoint

    def generatePaths(self) -> None:
        """Generate the delay paths."""

        # LOS
        # losDelay =
        pass

    pass


class Multipath(object):
    """Filter emulating multipath propagation."""

    def __init__(self, samplingFreq: int) -> None:
        """Initializes the filter.

        Args:
            samplingFreq: The sampling rate in hz.
        """

        self._samplingFreq = samplingFreq
        pass

    pass
