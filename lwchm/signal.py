from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class TransmitSignal(object):
    wave: npt.NDArray[np.float64]  # raw wave samples
    fs: float  # sampling frequency
    carrierFs: float  # carrier frequency


@dataclass
class ReceiveSignal(object):
    wave: npt.NDArray[np.complex128]  # raw wave samples


def delaySamples(fs: float, delay: float | np.floating[Any]) -> int:
    """Compute the number of delay samples given the sampling freq and delay time.

    Args:
        fs: sampling frequency.
        delay: time delay.
    """
    return int(np.round(delay * fs))


def generateBPSKSignal(
    data: bytes, symbolPeriod: float, fs: float, carrFs: float, transmitPower: float
) -> TransmitSignal:
    """Generate a BPSK signal, mainly used for visualization.

    Args:
        data: The data to encode
        symbolPeriod: Symbol period in seconds
        fs: Sampling frequency
        carrFs: Carrier frequency
        transmitPower: Power in dBm
    Returns:
        The generated transmitSignal
    """

    # Compute amplitude
    amp = np.power(10, transmitPower / 20) * np.sqrt(1e-3)

    # convert to phase array
    phases = np.zeros(len(data) * 8)
    for i, byte in enumerate(data):
        for j in range(8):
            phases[i * 8 + j] = np.pi if (byte >> j) & 0x1 else 0

    # upsample to sampling freq
    samplesPerBit = int(symbolPeriod / (1 / fs))
    phasesUpsampled = np.repeat(phases, samplesPerBit)

    # Modulate
    t = np.arange(0, len(phasesUpsampled)) / fs
    signal = amp * np.cos((2 * np.pi * carrFs * t) + phasesUpsampled)
    return TransmitSignal(
        wave=signal,
        fs=fs,
        carrierFs=carrFs,
    )


def generateQPSKSignal(
    data: bytes, symbolPeriod: float, fs: int, carrFs: int, transmitPower: float
) -> TransmitSignal:
    """Generate a QPSK signal.

    Args:
        data: The data to encode
        symbolPeriod: Symbol period in seconds
        fs: Sampling frequency
        carrFs: Carrier frequency
        transmitPower: Power in dBm
    Returns:
        The generated transmitSignal
    """
    # Compute amplitude
    amp = np.power(10, transmitPower / 20) * np.sqrt(1e-3)

    # convert to frequency array
    phaseMap = (np.pi / 4, 3 * np.pi / 4, 7 * np.pi / 4, 5 * np.pi / 4)
    phases = np.zeros(len(data) * 4)
    for i, byte in enumerate(data):
        for j in range(4):
            phases[i * 4 + j] = phaseMap[(byte >> (j * 2)) & 0x3]

    # upsample to sampling freq
    samplesPerBit = int(symbolPeriod / (1 / fs))
    phasesUpsampled = np.repeat(phases, samplesPerBit)

    # Modulate
    t = np.arange(0, len(phasesUpsampled)) / fs
    signal = amp * np.cos((2 * np.pi * carrFs * t) + phasesUpsampled)
    return TransmitSignal(
        wave=signal,
        fs=fs,
        carrierFs=carrFs,
    )


def computeRmsDBM(rxSig: npt.NDArray[np.complex128]) -> float:
    """Compute the VRms and return it in dBm."""

    # TODO: Do I need to move parts without signal?
    vRms = np.sqrt(np.mean(np.square(np.abs(rxSig))))
    return 30 + 20 * np.log10(vRms)  # 1ohm impedance
