import random
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import scipy.signal  # type: ignore
from scipy import constants  # type: ignore

from lwchm import propagation, signal, spatial


@dataclass
class LWCHMConfiguration(object):
    """Model configuration and constants"""

    # Reflector generation
    refCount: int  # max number of reflectors to generate
    refAttemptPerRing: int
    ringRadiusMin: float
    ringRadiusMax: float
    ringRadiusUncertainty: float  # +- random offset to radius
    ringCount: int  # Number of rings to try between min and max ring radius

    # Ground reflection
    complexRelPermittivityReal: float
    complexRelPermittivityRealStd: float
    complexRelPermittivityImag: float
    complexRelPermittivityImagStd: float
    horizontalPolarization: bool

    # Rayleigh fading
    fadingPaths: int
    fadingDopplerSpread: float


class LWCHM(object):
    """Lunar Wireless Channel Model"""

    def __init__(self, body: spatial.Body, config: LWCHMConfiguration) -> None:
        """Construct a lwchm runner

        Args:
            body: The celestial body
            config: Model configuration
        """
        self._body = body
        self._grid = body.grid
        self._config = config

    def _generateReflectors(
        self,
        fs: float,
        excludeDelaySamples: set[int],
        txCoord: spatial.PointGeo,
        txLoc: spatial.Point3D,
        rxCoord: spatial.PointGeo,
        rxLoc: spatial.Point3D,
        txHeight: float,
        rxHeight: float,
    ) -> tuple[list[spatial.Point3D], list[spatial.PointGeo]]:
        refCount = self._config.refCount
        refAttemptPerRing = self._config.refAttemptPerRing
        ringRadiusMin = self._config.ringRadiusMin
        ringRadiusMax = self._config.ringRadiusMax
        ringRadiusUncertainty = self._config.ringRadiusUncertainty
        ringCount = self._config.ringCount

        excludedDelays = set(excludeDelaySamples)
        txLocArr = np.array(txLoc)
        rxLocArr = np.array(rxLoc)

        rCoords: list[spatial.PointGeo] = []
        currRadius = ringRadiusMin
        radiusStep = (ringRadiusMax - ringRadiusMin) / ringCount

        while currRadius <= ringRadiusMax and len(rCoords) < refCount:
            for _ in range(refAttemptPerRing):
                # Try reflector at random radius and angle
                r = random.uniform(
                    currRadius - ringRadiusUncertainty,
                    currRadius + ringRadiusUncertainty,
                )
                theta = random.uniform(0, 2 * np.pi)
                rCoord = self._body.destination(rxCoord, theta, r)
                rLoc = self._body.geoTo3D(
                    rCoord, self._body.getHeights(np.array([rCoord]))[0]
                )

                # Check if the reflector would overlap with each other
                rDist = np.linalg.norm(txLocArr - np.array(rLoc)) + np.linalg.norm(
                    np.array(rLoc) - rxLocArr
                )
                rDelaySamples = signal.delaySamples(
                    fs, rDist / constants.speed_of_light
                )
                if rDelaySamples in excludedDelays:
                    continue

                # Check LOS
                if self._body.checkLOS(
                    txCoord, txHeight, rCoord, 0
                ) and self._body.checkLOS(rCoord, 0, rxCoord, rxHeight):
                    rCoords.append(rCoord)
                    excludedDelays.add(rDelaySamples)
                    if len(rCoords) == refCount:
                        break

            # Increase radius
            currRadius += radiusStep

        if rCoords:
            # Get heights for each loc
            heights = self._body.getHeights(np.array(rCoords))

            # Convert to from Geo to 3D
            rPoints = [
                self._body.geoTo3D(coord, h) for coord, h in zip(rCoords, heights)
            ]
            return rPoints, rCoords
        else:
            return [], []

    def _generateRayleighFading(
        self, fs: float, carrFs: float, length: int
    ) -> npt.NDArray[np.complex128]:
        """Generate Rayleigh fading signal

        Model from https://doi.org/10.1109/TCOMM.2003.813259
        """

        m = self._config.fadingPaths // 4

        t = np.arange(0, length) / fs
        wd = (
            2
            * np.pi
            * self._config.fadingDopplerSpread
            * carrFs
            / constants.speed_of_light
        )

        # Generate real component
        sig = np.zeros(length, dtype=np.complex128)
        for n in range(1, m + 1):
            theta = random.uniform(-np.pi, np.pi)
            phi = random.uniform(-np.pi, np.pi)
            psi = random.uniform(-np.pi, np.pi)
            angle = (2 * np.pi * n - np.pi + theta) / (4 * m)
            sig += np.cos(psi) * np.cos(wd * t * np.cos(angle) + phi)

        # Imaginary component
        for n in range(1, m + 1):
            theta = random.uniform(-np.pi, np.pi)
            phi = random.uniform(-np.pi, np.pi)
            psi = random.uniform(-np.pi, np.pi)
            angle = (2 * np.pi * n - np.pi + theta) / (4 * m)
            sig += 1j * np.sin(psi) * np.cos(wd * t * np.cos(angle) + phi)

        # Normalize
        sig *= 2 / np.sqrt(m)
        avgPower = cast(np.float64, np.mean(np.square(np.abs(sig))))
        sig *= np.sqrt(1 / avgPower)

        return sig

    def compute(
        self,
        txCoord: spatial.PointGeo,
        rxCoord: spatial.PointGeo,
        txHeight: float,
        rxHeight: float,
        txSig: signal.TransmitSignal,
    ) -> signal.ReceiveSignal | None:
        txWave = txSig.wave
        fs = txSig.fs
        carrFs = txSig.carrierFs

        txGroundHeight, rxGroundHeight = self._body.getHeights(
            np.array((txCoord, rxCoord))
        )
        txLoc = self._body.geoTo3D(txCoord, txGroundHeight + txHeight)
        rxLoc = self._body.geoTo3D(rxCoord, rxGroundHeight + rxHeight)
        txLocArr = np.array(txLoc)
        rxLocArr = np.array(rxLoc)

        delayPaths: dict[np.float64, np.complex128] = {}  # time delay and phasor

        # Compute LOS delay path
        losDist = np.linalg.norm(txLocArr - rxLocArr)
        losDelay = losDist / constants.speed_of_light
        losPl = propagation.freeSpacePathloss(carrFs, losDist)
        if self._body.checkLOS(txCoord, txHeight, rxCoord, rxHeight):
            losPhasor = losPl * np.exp(-1j * 2 * np.pi * carrFs * losDelay)
            delayPaths[losDelay] = losPhasor

        # Compute reflector delay paths
        reflectors = self._generateReflectors(
            fs,
            set((signal.delaySamples(fs, d) for d in delayPaths)),
            txCoord,
            txLoc,
            rxCoord,
            rxLoc,
            txHeight,
            rxHeight,
        )[0]
        if reflectors:
            # Get path distances
            refsArr = np.array(reflectors)
            txToRefDists = np.linalg.norm(txLocArr - refsArr, axis=1)
            refToRxDists = np.linalg.norm(refsArr - rxLocArr, axis=1)
            reflectDists = txToRefDists + refToRxDists

            # Compute time delay
            reflectDelays = reflectDists / constants.speed_of_light

            # Compute pathloss
            reflectPls = np.array(
                [propagation.freeSpacePathloss(carrFs, d) for d in reflectDists]
            )

            # Compute reflection angle with law of cosines
            reflectAngles = (
                np.pi
                - np.arccos(
                    (
                        np.square(txToRefDists)
                        + np.square(refToRxDists)
                        - np.square(losDist)
                    )
                    / (2 * txToRefDists * refToRxDists)
                )
            ) / 2

            # Compute reflection coefficient
            complexRelPermittivities = np.array(
                [
                    random.normalvariate(
                        self._config.complexRelPermittivityReal,
                        self._config.complexRelPermittivityRealStd,
                    )
                    + (
                        1j
                        * random.normalvariate(
                            self._config.complexRelPermittivityImag,
                            self._config.complexRelPermittivityImagStd,
                        )
                    )
                    for _ in range(len(reflectors))
                ]
            )

            if self._config.horizontalPolarization:
                reflectPolarizations = np.sqrt(
                    complexRelPermittivities - np.square(np.cos(reflectAngles))
                )
            else:
                reflectPolarizations = (
                    np.sqrt(complexRelPermittivities - np.square(np.cos(reflectAngles)))
                    / complexRelPermittivities
                )

            reflectCoeffs = (np.sin(reflectAngles) - reflectPolarizations) / (
                np.sin(reflectAngles) + reflectPolarizations
            )

            # Compute phasors
            phases = np.array(
                [random.uniform(0, 2 * np.pi) for _ in range(len(reflectors))]
            )
            reflectPhasors = reflectPls * reflectCoeffs * np.exp(1j * phases)

            delayPaths.update(zip(reflectDelays, reflectPhasors))

        if delayPaths:
            # Apply complex baseband conversion
            if not txSig.modulated:
                minDelay = np.min([d for d in delayPaths])
                for i, (delay, phasor) in enumerate(delayPaths.items()):
                    delayPaths[delay] = phasor * np.exp(
                        -1j * 2 * np.pi * carrFs * (delay - minDelay)
                    )

            # Compute delay samples
            delaySamples = [signal.delaySamples(fs, d) for d in delayPaths]
            delayOffset = min(delaySamples)
            maxDelay = max(delaySamples)

            # Construct fir filter
            firCoeffs = np.zeros(maxDelay - delayOffset + 1, dtype=np.complex128)
            for delaySample, (_, phasor) in zip(delaySamples, delayPaths.items()):
                firCoeffs[delaySample - delayOffset] = phasor

            # Add rayleigh
            rayleighSig = (
                self._generateRayleighFading(txSig.fs, txSig.carrierFs, len(firCoeffs))
                * losPl
                / len(firCoeffs)
            )
            if not txSig.modulated:
                rayleighSig *= np.exp(
                    -1j * 2 * np.pi * carrFs * np.arange(0, len(firCoeffs)) / fs
                )
            firCoeffs += rayleighSig

            firCoeffs /= self._config.refCount + 1

            # Pass signal through filter
            rxWave = scipy.signal.convolve(firCoeffs, txWave)  # type: ignore
            return signal.ReceiveSignal(wave=rxWave)  # type: ignore
        else:
            return None  # No signal due to no paths
