import pathlib
import random
from dataclasses import dataclass
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.constants as sci_consts  # type: ignore
import scipy.interpolate  # type: ignore
import scipy.io  # type: ignore


@dataclass
class Point3D(object):
    x: float
    y: float
    z: float

    def __array__(
        self, dtype: None = None, copy: bool | None = None
    ) -> npt.NDArray[np.float64]:
        del copy

        return np.array((self.x, self.y, self.z), dtype=dtype)

    def __add__(self, o: "Point3D") -> "Point3D":
        return Point3D(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "Point3D") -> "Point3D":
        return Point3D(self.x - o.x, self.y - o.y, self.z - o.z)


@dataclass
class ReflectorPoint(object):
    loc: Point3D
    angle: float  # angle in radians that was used to generate the location


@dataclass
class TransmitSignal(object):
    wave: npt.NDArray[np.float64]
    fs: float
    carrierFs: float

    symbolStarts: list[int] | None


RANDOM_SEED = "0c4d0fd7-8776-4e90-8d80-86f65bc1cbc5"

# XYZ coordinates in meters
TX_COORD = Point3D(x=0, y=0, z=2)
RX_COORD = Point3D(x=100, y=0, z=1.5)

# Reflector params
# List of tuples, of ring radius in meters and number of reflectors per ring
RING_CONFIG = (
    (1, 3),
    # (20, 5),
    # (40, 7),
    # (60, 9),
    # (80, 13),
)

# Rayleigh Params
FADING_MAX_DOPPLER_SPEED = 0  # m/s
FADING_N_PATHS = 64  # Should be a multiple of 4

# Ground reflection params
RELATIVE_PERMITTIVITY = 2.75 / sci_consts.epsilon_0  # NU-LHT-2M real permittivity

# 5G data from Matlab params
GEN_DATA_FILE = pathlib.Path(R"model/data/data.mat")
GEN_POINT_A = 3e9
GEN_SCS_BANDWIDTH = 50e6
GEN_FS = 7e9

# Impulse signal parameters
IMPULSE_TRANSMIT_POWER = 20  # Transmit power in dBm
IMPULSE_FS = 3e9
IMPULSE_CARR_FS = 500e6
IMPULSE_LENGTH = 100  # Number of samples

# BPSK signal parameters
BPSK_TRANSMIT_POWER = 20  # Transmit power in dBm
BPSK_DATA = 0xA1588DEB
BPSK_DATA_LEN = 32
BPSK_FS = 7e9
BPSK_CARR_FS = 500e6
BPSK_SYMBOL_PERIOD = 1 / BPSK_FS * 20

# QPSK signal parameters, matches DRATS report
QPSK_TRANSMIT_POWER = 30  # Transmit power in dBm
QPSK_DATA = 0xA1588DEB
QPSK_DATA_LEN = 32
QPSK_FS = 3e9
QPSK_CARR_FS = 913e6
QPSK_SYMBOL_PERIOD = 6e-6

# RX sensitivity in dBm
# ETSI TS 138 141-1 V18.7.0
# Table 7.2.5-3 NR Local Area BS reference sensitivity levels
# BW=50mHz, SCS=15kHz, F<3GHz
RX_SENSITIVITY = -86.6
SHOW_RX_SENSITIVITY = True

LOG_ZERO_STUB = -100  # Value to show when taking log of 0


def freeSpacePathloss[T: Any](
    freq: np.floating[T] | float, dist: np.floating[T]
) -> np.floating[T]:
    """Computes the free space pathloss of the E field, i.e without the square"""
    return sci_consts.speed_of_light / (4 * np.pi * dist * freq)


def load5gTxSignal() -> TransmitSignal:
    """Retrieves the IQ samples and the carrier frequency"""
    # Load baseband and metadata
    file = scipy.io.loadmat(str(GEN_DATA_FILE))  # type: ignore
    basebandFs = cast(np.int32, file["waveStruct"]["Fs"][0, 0][0, 0])
    basebandSamples = cast(
        npt.NDArray[np.complex128], file["waveStruct"]["waveform"][0, 0][:, 0]
    )

    basebandTime = np.arange(0, len(basebandSamples) / basebandFs, 1 / basebandFs)
    upsampler = scipy.interpolate.CubicSpline(basebandTime, basebandSamples)

    # upsample for upconversion
    passbandTime = np.arange(0, len(basebandSamples) / basebandFs, 1 / GEN_FS)
    basebandUpsampled = cast(npt.NDArray[np.complex128], upsampler(passbandTime))

    # Upconvert
    carrierFreq = GEN_POINT_A + (GEN_SCS_BANDWIDTH / 2)
    passbandSamples = (
        basebandUpsampled * np.exp(1j * 2 * np.pi * carrierFreq * passbandTime)
    ).real

    return TransmitSignal(
        wave=passbandSamples, fs=GEN_FS, carrierFs=carrierFreq, symbolStarts=None
    )


def generateImpulseSignal() -> TransmitSignal:
    amp = np.power(10, IMPULSE_TRANSMIT_POWER / 20) * np.sqrt(1e-3)

    sig = np.zeros(IMPULSE_LENGTH)
    sig[0] = amp
    return TransmitSignal(
        wave=sig, fs=IMPULSE_FS, carrierFs=IMPULSE_CARR_FS, symbolStarts=None
    )


def generateBPSKSignal() -> TransmitSignal:
    # Compute amplitude
    amp = np.power(10, BPSK_TRANSMIT_POWER / 20) * np.sqrt(1e-3)

    # convert to phase array
    phases = np.array(
        [np.pi if (BPSK_DATA >> i) & 0x1 else 0 for i in range(BPSK_DATA_LEN)]
    )

    # upsample to sampling freq
    samplesPerBit = int(BPSK_SYMBOL_PERIOD / (1 / BPSK_FS))
    phasesUpsampled = cast(npt.NDArray[np.float64], np.repeat(phases, samplesPerBit))
    symbolStarts = [samplesPerBit * i for i in range(len(phases))]

    # Modulate
    t = np.arange(0, len(phasesUpsampled)) / BPSK_FS
    signal = amp * np.cos((2 * np.pi * BPSK_CARR_FS * t) + phasesUpsampled)
    return TransmitSignal(
        wave=signal, fs=BPSK_FS, carrierFs=BPSK_CARR_FS, symbolStarts=symbolStarts
    )


def generateQPSKSignal() -> TransmitSignal:
    # Compute amplitude
    amp = np.power(10, QPSK_TRANSMIT_POWER / 20) * np.sqrt(1e-3)

    # convert to frequency array
    symbols = ((QPSK_DATA >> i) & 0x3 for i in range(0, QPSK_DATA_LEN, 2))
    phaseMap = (np.pi / 4, 3 * np.pi / 4, 7 * np.pi / 4, 5 * np.pi / 4)
    phases = np.array([phaseMap[s] for s in symbols])

    # upsample to sampling freq
    samplesPerBit = int(QPSK_SYMBOL_PERIOD / (1 / QPSK_FS))
    phasesUpsampled = cast(npt.NDArray[np.float64], np.repeat(phases, samplesPerBit))
    symbolStarts = [samplesPerBit * i for i in range(len(phases))]

    # Modulate
    t = np.arange(0, len(phasesUpsampled)) / QPSK_FS
    signal = amp * np.cos((2 * np.pi * QPSK_CARR_FS * t) + phasesUpsampled)
    return TransmitSignal(
        wave=signal, fs=QPSK_FS, carrierFs=QPSK_CARR_FS, symbolStarts=symbolStarts
    )


def generateReflectors() -> list[ReflectorPoint]:
    reflectors: list[ReflectorPoint] = []
    for ringRadius, points in RING_CONFIG:
        for _ in range(points):
            # Random angle
            # TODO: Randomness in radius
            theta = random.uniform(0, 2 * np.pi)
            x = ringRadius * np.cos(theta) + RX_COORD.x
            y = ringRadius * np.sin(theta) + RX_COORD.y
            z = 0
            reflectors.append(ReflectorPoint(loc=Point3D(x=x, y=y, z=z), angle=theta))
    return reflectors


def computeDelaySamples(fs: float, delay: float | np.floating[Any]) -> int:
    """Compute the number of delay samples given the sampling freq and delay time."""
    return int(np.floor(delay * fs))


def generateRayleighFading(
    txSig: TransmitSignal, length: int
) -> npt.NDArray[np.complex128]:
    """Generate Rayleigh fading signal
    Model from https://doi.org/10.1109/TCOMM.2003.813259
    """

    m = FADING_N_PATHS // 4

    t = np.arange(0, length) / txSig.fs
    wd = (
        2
        * np.pi
        * FADING_MAX_DOPPLER_SPEED
        * txSig.carrierFs
        / sci_consts.speed_of_light
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

    return sig


def twoRay(
    txSig: TransmitSignal, reflectors: list[ReflectorPoint]
) -> npt.NDArray[np.complex128]:
    reflectLoc = reflectors[0].loc  # Use the first one

    txWave = txSig.wave
    fs = txSig.fs
    carrFs = txSig.carrierFs

    # Get path distances
    losDist = np.linalg.norm(TX_COORD - RX_COORD)
    reflectDist = np.linalg.norm(TX_COORD - reflectLoc) + np.linalg.norm(
        reflectLoc - RX_COORD
    )

    # Compute los component
    waveLength = sci_consts.speed_of_light / carrFs
    losSig = (waveLength / 4 / np.pi) * (
        txWave * np.exp(-1j * 2 * np.pi * losDist / waveLength) / losDist
    )

    # Compute reflection coefficient, assume horizontal polarization
    reflectAngle = np.arcsin(TX_COORD.z / np.linalg.norm(TX_COORD - reflectLoc))
    reflectPolarization = np.sqrt(
        RELATIVE_PERMITTIVITY - np.square(np.cos(reflectAngle))
    )
    reflectCoeff = (np.sin(reflectAngle) - reflectPolarization) / (
        np.sin(reflectAngle) + reflectPolarization
    )

    delaySpread = (reflectDist - losDist) / sci_consts.speed_of_light
    delaySamples = computeDelaySamples(fs, delaySpread)
    reflectSig = (waveLength * reflectCoeff / 4 / np.pi) * (
        np.pad(txWave, (delaySamples, 0), "constant")
        * np.exp(-1j * 2 * np.pi * reflectDist / waveLength)
        / reflectDist
    )

    # Compute combined signal
    return np.pad(losSig, (0, len(reflectSig) - len(losSig)), "constant") + reflectSig


def model(
    txSig: TransmitSignal,
    reflectors: list[ReflectorPoint],
    removeLosDelay: bool = False,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    txWave = txSig.wave
    fs = txSig.fs
    carrFs = txSig.carrierFs

    # Get path distances
    losDist = np.linalg.norm(TX_COORD - RX_COORD)
    reflectDists = np.array(
        [
            np.linalg.norm(TX_COORD - r.loc) + np.linalg.norm(r.loc - RX_COORD)
            for r in reflectors
        ]
    )

    # Compute time delay
    losDelay = losDist / sci_consts.speed_of_light
    reflectDelays = reflectDists / sci_consts.speed_of_light

    # Compute pathloss
    losPl = freeSpacePathloss(carrFs, losDist)
    reflectPls = np.array([freeSpacePathloss(carrFs, d) for d in reflectDists])

    # Compute reflection coefficient
    reflectAngles = np.arcsin(
        TX_COORD.z / np.array([np.linalg.norm(TX_COORD - r.loc) for r in reflectors])
    )
    reflectPolarizations = np.sqrt(
        RELATIVE_PERMITTIVITY - np.square(np.cos(reflectAngles))
    )
    reflectCoeffs = (np.sin(reflectAngles) - reflectPolarizations) / (
        np.sin(reflectAngles) + reflectPolarizations
    )

    # Compute phasors
    losPhasor = losPl * np.exp(-1j * 2 * np.pi * carrFs * losDelay)
    reflectPhasors = (
        reflectPls * reflectCoeffs * np.exp(-1j * 2 * np.pi * carrFs * reflectDelays)
    )

    # compute delay indices
    losDelaySamples = computeDelaySamples(fs, losDelay)
    reflectDelaysSamples = np.array([computeDelaySamples(fs, d) for d in reflectDelays])

    # Construct fir delay taps
    # longest delay -> filter length
    firCoeffs = np.zeros(reflectDelaysSamples.max() + 1, dtype=np.complex64)
    firCoeffs[losDelaySamples] = losPhasor
    for delay, phasor in zip(reflectDelaysSamples, reflectPhasors):
        # Add phasors in the case that there are duplicate delays
        firCoeffs[delay] += phasor

    if removeLosDelay:
        firCoeffs = firCoeffs[losDelaySamples:]

    # Add rayleigh
    rayleighSig = generateRayleighFading(txSig, len(firCoeffs))
    firCoeffs += rayleighSig

    # Pass signal through filter
    return np.convolve(firCoeffs, txWave), rayleighSig


def computeDB(sig: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Compute the power of a signal in dB"""
    mag = np.square(np.abs(sig))
    return np.multiply(
        np.log10(
            mag,
            out=np.full(mag.shape, LOG_ZERO_STUB / 10, dtype=np.float64),
            where=(mag != 0),
        ),
        10,
    )


def computeDBM(sig: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Compute the power of a signal in dBm"""
    mag = np.square(np.abs(sig)) / 1e-3

    return np.multiply(
        np.log10(
            mag,
            out=np.full(mag.shape, LOG_ZERO_STUB / 10, dtype=np.float64),
            where=(mag != 0),
        ),
        10,
    )


def computeSNR(txSig: npt.NDArray[np.float64], rxSig: npt.NDArray[np.complex128]):
    """Compute the SNR.

    Assumes that the Rx signal does not have los delay

    Args:
        txSig: Transmitted signal
        rxSig: Received signal
    Returns:
        snr in dB
    """

    # Convert tx sig to complex
    txSigComplex = txSig.astype(np.complex128)

    # truncate rx to match size of tx
    rxTrunc = rxSig[0 : len(txSigComplex)]

    # Compute noise
    noise = rxTrunc - txSigComplex

    # Compute SNR
    txPower = computeDBM(txSigComplex)
    noisePower = computeDB(noise)
    snr = txPower - noisePower
    return snr


def main() -> None:
    random.seed(RANDOM_SEED)

    txSig = generateBPSKSignal()
    reflectors = generateReflectors()
    twoRaySig = twoRay(txSig, reflectors)
    rxSig, rayleighSig = model(txSig, reflectors, removeLosDelay=True)

    snr = computeSNR(txSig.wave, rxSig)

    nGraphs = 7
    fig, axs = plt.subplots(nGraphs, sharex=True)  # type: ignore
    axsGen = (a for a in axs)

    ax = next(axsGen)
    ax.plot(computeDBM(twoRaySig))
    if SHOW_RX_SENSITIVITY:
        ax.axhline(RX_SENSITIVITY, color="red")
    ax.set_title("Two Ray")
    ax.set_ylabel("Power [dBm]")

    ax = next(axsGen)
    ax.plot(computeDBM(rxSig))
    if SHOW_RX_SENSITIVITY:
        ax.axhline(RX_SENSITIVITY, color="red")
    ax.set_title("Rx Signal")
    ax.set_ylabel("Power [dBm]")

    ax = next(axsGen)
    ax.plot(snr)
    ax.set_title("SNR [dBm]????")

    ax = next(axsGen)
    ax.plot(computeDBM(txSig.wave))
    ax.set_title("Tx Signal")
    ax.set_ylabel("Power [dBm]")

    ax = next(axsGen)
    ax.plot(txSig.wave)
    ax.set_title("Tx Signal")
    ax.set_ylabel("Amplitude [V]")
    for ss in txSig.symbolStarts or []:
        ax.axvline(ss, color="red")

    ax = next(axsGen)
    ax.plot(np.real(rxSig))
    ax.set_title("Rx Signal")
    ax.set_ylabel("Amplitude [V]")

    ax = next(axsGen)
    ax.plot(np.real(rayleighSig))
    ax.set_title("Rayleigh Signal")
    ax.ticklabel_format(useOffset=False)
    ax.set_ylabel("Amplitude [V]")

    fig.show()
    plt.show()  # type: ignore


if __name__ == "__main__":
    main()
