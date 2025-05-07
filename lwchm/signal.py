from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class TransmitSignal(object):
    wave: npt.NDArray[np.complex128]  # raw wave samples
    fs: float  # sampling frequency
    carrierFs: float  # carrier frequency
    modulated: bool


@dataclass
class ReceiveSignal(object):
    wave: npt.NDArray[np.complex128]  # raw wave samples


@dataclass
class PSKConfiguration(object):
    data: bytes
    symbolPeriod: float
    fs: float
    carrFs: float
    transmitPower: float
    modulate: bool


def delaySamples(fs: float, delay: float | np.floating[Any]) -> int:
    """Compute the number of delay samples given the sampling freq and delay time.

    Args:
        fs: sampling frequency.
        delay: time delay.
    """
    return int(np.round(delay * fs))


def generateBPSKSignal(config: PSKConfiguration) -> TransmitSignal:
    """Generate a BPSK signal, mainly used for visualization.

    Args:
        config: Signal configuration
    Returns:
        The generated transmitSignal
    """

    # Compute amplitude
    amp = np.power(10, config.transmitPower / 20) * np.sqrt(1e-3)

    # convert to phase array
    phases = np.zeros(len(config.data) * 8)
    for i, byte in enumerate(config.data):
        for j in range(8):
            phases[i * 8 + j] = np.pi if (byte >> j) & 0x1 else 0

    # upsample to sampling freq
    samplesPerBit = round(config.symbolPeriod / (1 / config.fs))
    phasesUpsampled = np.repeat(phases, samplesPerBit)

    if config.modulate:
        # Modulate
        t = np.arange(0, len(phasesUpsampled)) / config.fs
        signal = amp * np.cos((2 * np.pi * config.carrFs * t) + phasesUpsampled)
    else:
        signal = amp * np.exp(1j * phasesUpsampled) / np.sqrt(2)

    return TransmitSignal(
        wave=signal,
        fs=config.fs,
        carrierFs=config.carrFs,
        modulated=config.modulate,
    )


def generateQPSKSignal(config: PSKConfiguration) -> TransmitSignal:
    """Generate a QPSK signal.

    Args:
        config: Signal configuration
    Returns:
        The generated transmitSignal
    """
    # Compute amplitude
    amp = np.power(10, config.transmitPower / 20) * np.sqrt(1e-3)

    # convert to frequency array
    phaseMap = (np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4)
    phases = np.zeros(len(config.data) * 4)
    for i, byte in enumerate(config.data):
        for j in range(4):
            phases[i * 4 + j] = phaseMap[(byte >> (j * 2)) & 0x3]

    # upsample to sampling freq
    samplesPerBit = round(config.symbolPeriod / (1 / config.fs))
    phasesUpsampled = np.repeat(phases, samplesPerBit)

    if config.modulate:
        # Modulate
        t = np.arange(0, len(phasesUpsampled)) / config.fs
        signal = amp * np.cos((2 * np.pi * config.carrFs * t) + phasesUpsampled)
    else:
        signal = amp * np.exp(1j * phasesUpsampled) / np.sqrt(2)

    return TransmitSignal(
        wave=signal,
        fs=config.fs,
        carrierFs=config.carrFs,
        modulated=config.modulate,
    )


def computeRmsDBM(rxSig: npt.NDArray[np.complex128]) -> float:
    """Compute the VRms and return it in dBm."""

    # TODO: Do I need to move parts without signal?
    vRms = np.sqrt(np.mean(np.square(np.abs(rxSig))))
    return 30 + 20 * np.log10(vRms)  # 1ohm impedance
