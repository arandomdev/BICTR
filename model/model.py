from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.constants as sci_consts  # type: ignore

# XYZ coordinates in meters
TX_COORD = (0, 0, 2)
RX_COORD = (20, 17, 1.5)
REFLECTION_COORD = (18, 14, 0)

RELATIVE_PERMITTIVITY = 2.75 / sci_consts.epsilon_0  # NU-LHT-2M real permittivity

FS = 12e9

TX_FREQ = 1e9  # 1Ghz


def freeSpacePathloss[T: Any](
    freq: np.floating[T] | float, dist: np.floating[T]
) -> np.floating[T]:
    """Computes the free space pathloss of the E field, i.e without the square"""
    return sci_consts.speed_of_light / (4 * np.pi * dist * freq)


def generateTxSignal() -> npt.NDArray[np.float64]:
    # 1Ghz, 50ns, cosine
    return np.cos(2 * np.pi * TX_FREQ * np.arange(0, 50e-9, 1 / FS))


def twoRay() -> npt.NDArray[np.complex64]:
    txLoc = np.array(TX_COORD, dtype=np.float64)
    rxLoc = np.array(RX_COORD, dtype=np.float64)
    reflectLoc = np.array(REFLECTION_COORD, dtype=np.float64)
    txSig = generateTxSignal()

    # Get path distances
    losDist = np.linalg.norm(txLoc - rxLoc)
    reflectDist = np.linalg.norm(txLoc - reflectLoc) + np.linalg.norm(
        reflectLoc - rxLoc
    )

    # Compute los component
    waveLength = sci_consts.speed_of_light / TX_FREQ
    losSig = (waveLength / 4 / np.pi) * (
        txSig * np.exp(-1j * 2 * np.pi * losDist / waveLength) / losDist
    )

    # Compute reflection coefficient, assume horizontal polarization
    reflectAngle = np.arcsin(TX_COORD[2] / np.linalg.norm(txLoc - reflectLoc))
    reflectPolarization = np.sqrt(
        RELATIVE_PERMITTIVITY - np.square(np.cos(reflectAngle))
    )
    reflectCoeff = (np.sin(reflectAngle) - reflectPolarization) / (
        np.sin(reflectAngle) + reflectPolarization
    )

    delaySpread = (reflectDist - losDist) / sci_consts.speed_of_light
    delaySamples = int(np.round(delaySpread * FS))
    reflectSig = (waveLength * reflectCoeff / 4 / np.pi) * (
        np.pad(txSig, (delaySamples, 0), "constant")
        * np.exp(-1j * 2 * np.pi * reflectDist / waveLength)
        / reflectDist
    )

    # Compute combined signal
    return np.pad(losSig, (0, len(reflectSig) - len(losSig)), "constant") + reflectSig


def multipath() -> npt.NDArray[np.complex64]:
    # Define locations
    txLoc = np.array(TX_COORD, dtype=np.float64)
    rxLoc = np.array(RX_COORD, dtype=np.float64)
    reflectLoc = np.array(REFLECTION_COORD, dtype=np.float64)
    txSig = generateTxSignal()

    # Get path distances
    losDist = np.linalg.norm(txLoc - rxLoc)
    reflectDist = np.linalg.norm(txLoc - reflectLoc) + np.linalg.norm(
        reflectLoc - rxLoc
    )

    # Compute time delay
    losDelay = losDist / sci_consts.speed_of_light
    reflectDelay = reflectDist / sci_consts.speed_of_light

    # Compute pathloss
    losPl = freeSpacePathloss(TX_FREQ, losDist)
    reflectPl = freeSpacePathloss(TX_FREQ, reflectDist)

    # Compute reflection coefficient
    reflectAngle = np.arcsin(TX_COORD[2] / np.linalg.norm(txLoc - reflectLoc))
    reflectPolarization = np.sqrt(
        RELATIVE_PERMITTIVITY - np.square(np.cos(reflectAngle))
    )
    reflectCoeff = (np.sin(reflectAngle) - reflectPolarization) / (
        np.sin(reflectAngle) + reflectPolarization
    )

    # Compute phasors
    losPhasor = losPl * np.exp(-1j * 2 * np.pi * TX_FREQ * losDelay)
    reflectPhasor = (
        reflectPl * reflectCoeff * np.exp(-1j * 2 * np.pi * TX_FREQ * reflectDelay)
    )

    # Construct fir delay taps
    # longest delay -> filter length
    firCoeffs = np.zeros(int(np.floor(reflectDelay * FS)) + 1, dtype=np.complex64)
    firCoeffs[int(np.floor(losDelay * FS))] = losPhasor
    firCoeffs[int(np.floor(reflectDelay * FS))] = reflectPhasor

    # Pass signal through filter
    return np.convolve(firCoeffs, txSig)


def main() -> None:
    twoRaySig = twoRay()
    multipathSig = multipath()

    plt.plot(np.square(np.abs(twoRaySig)), label="two ray")
    plt.plot(np.square(np.abs(multipathSig)), label="multipath")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
