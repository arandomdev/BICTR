import multiprocessing as mp
import multiprocessing.queues as mpq
import multiprocessing.synchronize as mps
import pathlib
import queue
import random
import signal
from dataclasses import dataclass
from types import FrameType, TracebackType
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pygmt  # type: ignore
import scipy.constants as sci_consts  # type: ignore
import scipy.interpolate  # type: ignore
import scipy.io  # type: ignore
import xarray


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
class PointGeo(object):
    lon: float
    lat: float

    def __array__(
        self, dtype: None = None, copy: bool | None = None
    ) -> npt.NDArray[np.float64]:
        del copy

        return np.array((self.lon, self.lat), dtype=dtype)

    def __eq__(self, value: object) -> bool:
        if isinstance(value, PointGeo):
            if (self.lat == 90 and value.lat == 90) or (
                self.lat == -90 and value.lat == -90
            ):
                # Poles
                return True
            elif self.lat == value.lat and abs(self.lon) == abs(value.lon):
                # Longitude crossing
                return True
            else:
                return self.lat == value.lat and self.lon == value.lon
        else:
            return False


@dataclass
class TransmitSignal(object):
    wave: npt.NDArray[np.float64]
    time: npt.NDArray[np.float64]
    fs: float
    carrierFs: float

    symbolStarts: list[int] | None


@dataclass
class ReceiveSignal(object):
    wave: npt.NDArray[np.complex128]
    time: npt.NDArray[np.float64]


class DelayedKeyboardInterrupt:
    def __enter__(self):
        self.signalReceived = None

        self.oldHandler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig: int, frame: FrameType | None) -> None:
        self.signalReceived = (sig, frame)

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        signal.signal(signal.SIGINT, self.oldHandler)
        if self.signalReceived:
            self.oldHandler(*self.signalReceived)  # type: ignore


RANDOM_SEED = "0c4d0fd7-8776-4e90-8d80-86f65bc1cbc5"

MOON_RADIUS = 1737.4e3

TX_COORD = PointGeo(lon=0, lat=-90)
TX_HEIGHT = 2
RX_COORD = PointGeo(lon=120, lat=-89.5)
RX_HEIGHT = 2

SCAN_RESULTS_PATH = pathlib.Path(R"data/scanMap.nc")
SCAN_REGION = (PointGeo(lon=-180, lat=-89.99), PointGeo(lon=180, lat=-89))
SCAN_BLOCK_SIZE = 0.04  # Block size in degrees
VIEW_REGION = (PointGeo(lon=-180, lat=-90), PointGeo(lon=180, lat=-88.5))

# Reflector params
REFLECTOR_COUNT = 5  # max number of reflectors to generate
REFLECTOR_ATTEMPT_PER_RING = 5
RING_RADIUS_MIN = 5
RING_RADIUS_MAX = 500
RING_COUNT = 10  # Number of rings to try between min and max ring radius
RING_RADIUS_UNCERTAINTY = 15  # +- random offset to radius

# Rayleigh Params
FADING_MAX_DOPPLER_SPEED = 1  # m/s
FADING_N_PATHS = 1024  # Should be a multiple of 4
FADING_POWER = -80  # gain of the rayleigh fading channel in dBm

# Ground reflection params
COMPLEX_RELATIVITY_REAL = 2.75
COMPLEX_RELATIVITY_REAL_STD = 0.115
COMPLEX_RELATIVITY_IMAG = 0.13
COMPLEX_RELATIVITY_IMAG_STD = 0.047

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

LOG_ZERO_STUB = -150  # Value to show when taking log of 0

N_WORKERS = 22


def freeSpacePathloss[T: Any](
    freq: np.floating[T] | float, dist: np.floating[T]
) -> np.floating[T]:
    """Computes the free space pathloss of the E field, i.e without the square"""
    return sci_consts.speed_of_light / (4 * np.pi * dist * freq)


def load5gSignal() -> TransmitSignal:
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

    t = np.arange(0, len(passbandSamples)) / GEN_FS
    return TransmitSignal(
        wave=passbandSamples,
        time=t,
        fs=GEN_FS,
        carrierFs=carrierFreq,
        symbolStarts=None,
    )


def generateImpulseSignal() -> TransmitSignal:
    amp = np.power(10, IMPULSE_TRANSMIT_POWER / 20) * np.sqrt(1e-3)

    sig = np.zeros(IMPULSE_LENGTH)
    sig[0] = amp

    t = np.arange(0, len(sig)) / IMPULSE_FS
    return TransmitSignal(
        wave=sig, time=t, fs=IMPULSE_FS, carrierFs=IMPULSE_CARR_FS, symbolStarts=None
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
        wave=signal,
        time=t,
        fs=BPSK_FS,
        carrierFs=BPSK_CARR_FS,
        symbolStarts=symbolStarts,
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
        wave=signal,
        time=t,
        fs=QPSK_FS,
        carrierFs=QPSK_CARR_FS,
        symbolStarts=symbolStarts,
    )


def computeDestination(loc: PointGeo, bearing: float, distance: float) -> PointGeo:
    """Compute the destination location with bearing and distance.

    Args:
        bearing: direction of travel in radians
        distance: distance of travel in meters
    """
    dist = distance / MOON_RADIUS
    lon1 = np.deg2rad(loc.lon)
    lat1 = np.deg2rad(loc.lat)

    if loc.lat == 90 or loc.lat == -90:
        lon2 = bearing
        lat2 = np.pi / 2 - dist if loc.lat == 90 else -np.pi / 2 + dist
    else:
        lat2 = np.asin(
            np.sin(lat1) * np.cos(dist) + np.cos(lat1) * np.sin(dist) * np.cos(bearing)
        )

        lon2 = lon1 + np.atan2(
            np.sin(bearing) * np.sin(dist) * np.cos(lat1),
            np.cos(dist) - np.sin(lat1) * np.sin(lat2),
        )

    return PointGeo(lon=np.rad2deg(lon2), lat=np.rad2deg(lat2))


def geoTo3D(coord: PointGeo, heightBias: float = 0) -> Point3D:
    inc = np.deg2rad(90 - coord.lat)
    azi = np.deg2rad(coord.lon)
    height = MOON_RADIUS + heightBias
    x = height * np.sin(inc) * np.cos(azi)
    y = height * np.sin(inc) * np.sin(azi)
    z = height * np.cos(inc)
    return Point3D(x=x, y=y, z=z)


def generateReflectors(
    moonGrid: xarray.DataArray, txCoord: PointGeo, rxCoord: PointGeo
) -> tuple[list[Point3D], list[PointGeo]]:
    rCoords: list[PointGeo] = []

    currRadius = RING_RADIUS_MIN
    radiusStep = (RING_RADIUS_MAX - RING_RADIUS_MIN) / RING_COUNT

    while currRadius <= RING_RADIUS_MAX and len(rCoords) < REFLECTOR_COUNT:
        for _ in range(REFLECTOR_ATTEMPT_PER_RING):
            # Try reflector at random radius and angle
            r = random.uniform(
                currRadius - RING_RADIUS_UNCERTAINTY,
                currRadius + RING_RADIUS_UNCERTAINTY,
            )
            theta = random.uniform(0, 2 * np.pi)
            rCoord = computeDestination(rxCoord, theta, r)

            # Check LOS
            if checkLOS(moonGrid, txCoord, TX_HEIGHT, rCoord, 0) and checkLOS(
                moonGrid, rCoord, 0, rxCoord, RX_HEIGHT
            ):
                rCoords.append(rCoord)
                if len(rCoords) == REFLECTOR_COUNT:
                    break

        # Increase radius
        currRadius += radiusStep

    if rCoords:
        # Get heights for each loc
        heights = cast(
            npt.NDArray[np.float64],
            pygmt.grdtrack(  # type: ignore
                moonGrid, points=np.array(rCoords), z_only=True, output_type="numpy"
            )[:, 0],  # type: ignore
        )

        # Convert to from Geo to 3D
        rPoints = [geoTo3D(coord, h) for coord, h in zip(rCoords, heights)]
        return rPoints, rCoords
    else:
        return [], []


def checkLOS(
    moonGrid: xarray.DataArray,
    start: PointGeo,
    startHeightBias: float,
    end: PointGeo,
    endHeightBias: float,
) -> bool:
    """Checks if there is LOS between two Coordinates"""

    # Project from start to end and get terrain height
    track = pygmt.project(  # type: ignore
        center=(start.lon, start.lat),
        endpoint=(end.lon, end.lat),
        generate=0.1,
    )
    trackHeights = cast(
        npt.NDArray[np.float64],
        pygmt.grdtrack(  # type: ignore
            grid=moonGrid,
            points=track,
            z_only=True,
            output_type="numpy",
            newcolname="elevation",
        )[:, 0],
    )

    # Check if the signal intersects with terrain
    los = cast(
        npt.NDArray[np.float64],
        np.linspace(
            trackHeights[0] + startHeightBias,
            trackHeights[-1] + endHeightBias,
            len(trackHeights),
        ),
    )

    diff = los - trackHeights
    return diff.min() >= 0


def computeDelaySamples(fs: float, delay: float | np.floating[Any]) -> int:
    """Compute the number of delay samples given the sampling freq and delay time."""
    return int(np.round(delay * fs))


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
    # gain = np.power(10, FADING_POWER / 20) * np.sqrt(1e-3) / np.sqrt(m)
    # sig *= 2 * gain / np.sqrt(m)
    sig *= 2 / np.sqrt(m)
    avgPower = np.mean(np.square(sig))
    sig *= np.sqrt(1 / avgPower)

    return sig


def twoRay(
    txLoc: Point3D, rxLoc: Point3D, txSig: TransmitSignal, reflectors: list[Point3D]
) -> npt.NDArray[np.complex128]:
    reflectLoc = reflectors[0]  # Use the first one

    txWave = txSig.wave
    fs = txSig.fs
    carrFs = txSig.carrierFs

    # Get path distances
    losDist = np.linalg.norm(txLoc - rxLoc)
    txToRefDist = np.linalg.norm(txLoc - reflectLoc)
    refToRxDist = np.linalg.norm(reflectLoc - rxLoc)
    reflectDist = txToRefDist + refToRxDist

    # Compute los component
    waveLength = sci_consts.speed_of_light / carrFs
    losSig = (waveLength / 4 / np.pi) * (
        txWave * np.exp(-1j * 2 * np.pi * losDist / waveLength) / losDist
    )

    # Compute angle of reflection with law of cosines
    reflectAngle = (
        np.pi
        - np.arccos(
            (np.square(txToRefDist) + np.square(refToRxDist) - np.square(losDist))
            / (2 * txToRefDist * refToRxDist)
        )
    ) / 2

    # Compute reflection coefficient, assume horizontal polarization
    complexRelPermittivity = random.normalvariate(
        COMPLEX_RELATIVITY_REAL, COMPLEX_RELATIVITY_REAL_STD
    ) + (
        1j * random.normalvariate(COMPLEX_RELATIVITY_IMAG, COMPLEX_RELATIVITY_IMAG_STD)
    )
    reflectPolarization = np.sqrt(
        complexRelPermittivity - np.square(np.cos(reflectAngle))
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
    txLoc: Point3D,
    rxLoc: Point3D,
    txSig: TransmitSignal,
    reflectors: list[Point3D],
    hasLOS: bool,
) -> ReceiveSignal | None:
    txWave = txSig.wave
    fs = txSig.fs
    carrFs = txSig.carrierFs

    txLocArr = np.array(txLoc)
    rxLocArr = np.array(rxLoc)

    delayPaths: dict[int, np.complex128] = {}  # delay in filter coeffs: phasor

    # Compute LOS delay path
    losDist = np.linalg.norm(txLocArr - rxLocArr)
    if hasLOS:
        losDelay = losDist / sci_consts.speed_of_light
        losPl = freeSpacePathloss(carrFs, losDist)
        losPhasor = losPl * np.exp(-1j * 2 * np.pi * carrFs * losDelay)
        losDelaySamples = computeDelaySamples(fs, losDelay)
        delayPaths[losDelaySamples] = losPhasor

    # Compute reflector delay paths
    if reflectors:
        # NOTE: Assumes that all reflectors have LOS from TX to RX, as per implementation

        # Get path distances
        refsArr = np.array(reflectors)
        txToRefDists = np.linalg.norm(txLocArr - refsArr, axis=1)
        refToRxDists = np.linalg.norm(refsArr - rxLocArr, axis=1)
        reflectDists = txToRefDists + refToRxDists

        # Compute time delay
        reflectDelays = reflectDists / sci_consts.speed_of_light

        # Compute pathloss
        reflectPls = np.array([freeSpacePathloss(carrFs, d) for d in reflectDists])

        # Compute reflection angle with law of cosines
        reflectAngles = (
            np.pi
            - np.arccos(
                (np.square(txToRefDists) + np.square(refToRxDists) - np.square(losDist))
                / (2 * txToRefDists * refToRxDists)
            )
        ) / 2

        # Compute reflection coefficient
        complexRelPermittivity = random.normalvariate(
            COMPLEX_RELATIVITY_REAL, COMPLEX_RELATIVITY_REAL_STD
        ) + (
            1j
            * random.normalvariate(COMPLEX_RELATIVITY_IMAG, COMPLEX_RELATIVITY_IMAG_STD)
        )
        reflectPolarizations = np.sqrt(
            complexRelPermittivity - np.square(np.cos(reflectAngles))
        )
        reflectCoeffs = (np.sin(reflectAngles) - reflectPolarizations) / (
            np.sin(reflectAngles) + reflectPolarizations
        )

        # Compute phasors
        reflectPhasors = (
            reflectPls
            * reflectCoeffs
            # * np.exp(-1j * 2 * np.pi * carrFs * reflectDelays)
            * np.exp(1j * random.uniform(0, 2 * np.pi))
        )

        # compute delay indices
        reflectDelaysSamples = [computeDelaySamples(fs, d) for d in reflectDelays]

        for delay, phasor in zip(reflectDelaysSamples, reflectPhasors):
            if delay in delayPaths:
                delayPaths[delay] += phasor
            else:
                delayPaths[delay] = phasor

    if delayPaths:
        # Shift all delays to index zero for efficiency
        delayOffset = min(delayPaths.keys())
        maxDelay = max(delayPaths.keys())

        # Construct fir filter
        firCoeffs = np.zeros(maxDelay - delayOffset + 1, dtype=np.complex128)
        for delay, phasor in delayPaths.items():
            firCoeffs[delay - delayOffset] = phasor

        # Add rayleigh
        # TODO: This might require more normalization, maybe instead of fading power, split it across the taps?
        rayleighSig = (
            generateRayleighFading(txSig, len(firCoeffs)) * losPl / len(firCoeffs)
        )
        firCoeffs += rayleighSig

        # Pass signal through filter
        rxWave = np.convolve(firCoeffs, txWave)
        return (
            ReceiveSignal(wave=rxWave, time=np.arange(0, len(rxWave)) / fs),
            firCoeffs,
            rayleighSig,
        )
    else:
        return None  # No signal


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


def computeRmsDBM(rxSig: npt.NDArray[np.complex128]) -> float:
    """Compute the VRms and return it in dBm."""

    # TODO: Do I need to move parts without signal?
    vRms = np.sqrt(np.mean(np.square(np.abs(rxSig))))
    return 30 + 20 * np.log10(vRms)  # 1ohm impedance


def main() -> None:
    random.seed(RANDOM_SEED)

    # txSig = generateImpulseSignal()
    txSig = generateBPSKSignal()
    # txSig = generateQPSKSignal()
    # txSig = load5gSignal()

    # load moon DEM
    moonGrid = pygmt.datasets.load_moon_relief(
        resolution="01m", region=[-180, 180, -90, -89], registration="gridline"
    )

    txHeight, rxHeight = cast(
        tuple[float, float],
        pygmt.grdtrack(  # type: ignore
            moonGrid,
            points=np.array((TX_COORD, RX_COORD)),
            z_only=True,
            output_type="numpy",
        )[:, 0],
    )
    txLoc = geoTo3D(TX_COORD, txHeight + TX_HEIGHT)
    rxLoc = geoTo3D(RX_COORD, rxHeight + RX_HEIGHT)

    refPoints, refCoords = generateReflectors(moonGrid, TX_COORD, RX_COORD)
    if len(refPoints) == 0:
        print("No line of sight reflections")

    rxSig, firCoe, rayleigh = model(txLoc, rxLoc, txSig, refPoints, True)

    # generate map
    fig = pygmt.Figure()
    fig.grdimage(  # type: ignore
        grid=moonGrid,
        projection="G00/-90/12c",
        frame="afg",
    )

    refCoordsArr = np.array(refCoords)
    fig.plot(  # type: ignore
        x=refCoordsArr[:, 0],
        y=refCoordsArr[:, 1],
        style="c0.2c",
        fill="white",
        pen="black",
    )
    fig.plot(  # type: ignore
        x=TX_COORD.lon,
        y=TX_COORD.lat,
        style="i0.3c",
        fill="green",
        pen="black",
    )
    fig.plot(  # type: ignore
        x=RX_COORD.lon,
        y=RX_COORD.lat,
        style="i0.3c",
        fill="red",
        pen="black",
    )
    fig.show()  # type: ignore

    if not rxSig:
        print("No signal from model")
        return

    nGraphs = 4
    if not refPoints:
        nGraphs -= 1

    fig, axs = plt.subplots(nGraphs, figsize=(20, 16))  # type: ignore
    axsGen = (a for a in axs)

    if refPoints:
        twoRaySig = twoRay(txLoc, rxLoc, txSig, refPoints)
        ax = next(axsGen)

        ax.plot(computeDBM(twoRaySig))
        if SHOW_RX_SENSITIVITY:
            ax.axhline(RX_SENSITIVITY, color="red")
        ax.set_title("Two Ray")
        ax.set_ylabel("Power [dBm]")

    ax = next(axsGen)
    ax.plot(rxSig.time, computeDBM(rxSig.wave))
    if SHOW_RX_SENSITIVITY:
        ax.axhline(RX_SENSITIVITY, color="red")
    ax.set_title("Rx Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power [dBm]")

    ax = next(axsGen)
    ax.plot(computeDBM(firCoe), label="Prop")
    ax.plot(computeDBM(rayleigh), label="Rayleigh")
    ax.set_title("Propagation paths")
    # ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power [dBm]")

    # ax = next(axsGen)
    # ax.plot(snr)
    # ax.set_title("SNR [dBm]????")

    ax = next(axsGen)
    ax.plot(computeDBM(txSig.wave))
    ax.set_title("Tx Signal")
    ax.set_ylabel("Power [dBm]")

    # ax = next(axsGen)
    # ax.plot(txSig.time, txSig.wave)
    # ax.set_title("Tx Signal")
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Amplitude [V]")
    # for ss in txSig.symbolStarts or []:
    #     ax.axvline(ss / txSig.fs, color="red")

    # ax = next(axsGen)
    # ax.plot(rxSig.time, np.real(rxSig.wave))
    # ax.set_title("Rx Signal")
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Amplitude [V]")

    # ax = next(axsGen)
    # ax.plot(np.real(rayleighSig))
    # ax.set_title("Rayleigh Signal")
    # ax.set_ylabel("Amplitude [V]")

    # ax = next(axsGen)
    # ax.plot(computeDB(rayleighSig))
    # ax.set_title("Rayleigh Signal")
    # ax.set_ylabel("Power [dB]")
    # print(np.sqrt(np.mean(np.square(np.square(np.abs(rayleighSig))))))

    fig.tight_layout()
    fig.show()
    plt.show()  # type: ignore


def scanRegion() -> None:
    moonGrid = pygmt.datasets.load_moon_relief(
        resolution="01m",
        region=[
            VIEW_REGION[0].lon,
            VIEW_REGION[1].lon,
            VIEW_REGION[0].lat,
            VIEW_REGION[1].lat,
        ],
        registration="gridline",
    )

    # generate list of transceiver locations
    transceivers: list[PointGeo] = [TX_COORD]
    lonAxis = np.arange(SCAN_REGION[0].lon, SCAN_REGION[1].lon, SCAN_BLOCK_SIZE)
    latAxis = np.arange(SCAN_REGION[0].lat, SCAN_REGION[1].lat, SCAN_BLOCK_SIZE)
    for lon in lonAxis:
        for lat in latAxis:
            point = PointGeo(lon=lon, lat=lat)
            if point != TX_COORD:
                transceivers.append(point)

    # Get heights for each transceiver
    transceiverHeights = cast(
        npt.NDArray[np.float64],
        pygmt.grdtrack(  # type: ignore
            moonGrid,
            points=np.array(transceivers),
            z_only=True,
            output_type="numpy",
        )[:, 0],
    )

    # Convert to Point3D
    txLoc = geoTo3D(transceivers[0], transceiverHeights[0] + TX_HEIGHT)
    receiverLocs = [
        geoTo3D(c, h + RX_HEIGHT)
        for c, h in zip(transceivers[1:], transceiverHeights[1:])
    ]

    txSig = generateBPSKSignal()

    # Load results file
    if not SCAN_RESULTS_PATH.exists():
        results = xarray.DataArray(
            dims=["lat", "lon"],
            coords={
                "lon": lonAxis,
                "lat": latAxis,
            },
        )
        results.to_netcdf(SCAN_RESULTS_PATH)  # type: ignore
    else:
        with xarray.open_dataarray(SCAN_RESULTS_PATH) as da:  # type: ignore
            results = da.load()  # type: ignore

    # process each receiver
    writeTimer = 0
    for rxLoc, rxCoord in zip(receiverLocs, transceivers[1:]):
        if not np.isnan(results.loc[rxCoord.lat, rxCoord.lon]):  # type: ignore
            continue

        reflectorLocs = generateReflectors(moonGrid, TX_COORD, rxCoord)[0]
        hasLOS = checkLOS(moonGrid, TX_COORD, TX_HEIGHT, rxCoord, RX_HEIGHT)

        rxSig = model(txLoc, rxLoc, txSig, reflectorLocs, hasLOS)

        if rxSig:
            rxStrength = computeRmsDBM(rxSig.wave)
        else:
            rxStrength = LOG_ZERO_STUB

        results.loc[rxCoord.lat, rxCoord.lon] = rxStrength  # type: ignore
        print(f"Processed {rxCoord}")

        writeTimer = (writeTimer + 1) % 100
        if writeTimer == 0:
            with DelayedKeyboardInterrupt():
                results.to_netcdf(SCAN_RESULTS_PATH)  # type: ignore

    with DelayedKeyboardInterrupt():
        results.to_netcdf(SCAN_RESULTS_PATH)  # type: ignore


def scanRegionMultiWorkerInit(_resultsMem: Any, _cancelEvent: mps.Event) -> None:
    global resultsMem
    global cancelEvent
    resultsMem = _resultsMem
    cancelEvent = _cancelEvent

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def scanRegionMultiWorker(
    txLoc: Point3D,
    txCoord: PointGeo,
    tasks: list[tuple[Point3D, PointGeo]],
    updateQueue: mpq.Queue[int],
) -> None:
    """Child process to scan a region

    Args:
        txLoc: The transmitter location
        tasks: A list of receiver locations
        cancelEvent: A event to cancel the child process and exit
        updateQueue: A feed back mechanism that reports how many tasks were processed
            since the last update from this process
    """

    moonGrid = pygmt.datasets.load_moon_relief(
        resolution="01m",
        region=[
            VIEW_REGION[0].lon,
            VIEW_REGION[1].lon,
            VIEW_REGION[0].lat,
            VIEW_REGION[1].lat,
        ],
        registration="gridline",
    )

    # Construct dataarray from shared memory
    lonAxis = np.arange(SCAN_REGION[0].lon, SCAN_REGION[1].lon, SCAN_BLOCK_SIZE)
    latAxis = np.arange(SCAN_REGION[0].lat, SCAN_REGION[1].lat, SCAN_BLOCK_SIZE)

    resultsArr = np.frombuffer(resultsMem, dtype=np.float64)
    resultsArr = resultsArr.reshape((len(latAxis), len(lonAxis)))

    results = xarray.DataArray(
        resultsArr,
        dims=["lat", "lon"],
        coords={
            "lon": lonAxis,
            "lat": latAxis,
        },
    )

    txSig = generateBPSKSignal()

    tasksProcessed = 0

    for rxLoc, rxCoord in tasks:
        if np.isnan(results.loc[rxCoord.lat, rxCoord.lon]):  # type: ignore
            reflectorLocs = generateReflectors(moonGrid, txCoord, rxCoord)[0]
            hasLOS = checkLOS(moonGrid, TX_COORD, TX_HEIGHT, rxCoord, RX_HEIGHT)

            rxSig = model(txLoc, rxLoc, txSig, reflectorLocs, hasLOS)
            if rxSig:
                rxStrength = computeRmsDBM(rxSig.wave)
                results.loc[rxCoord.lat, rxCoord.lon] = rxStrength  # type: ignore
            else:
                results.loc[rxCoord.lat, rxCoord.lon] = LOG_ZERO_STUB  # type: ignore

        # check cancel and update
        if cancelEvent.is_set():
            break

        tasksProcessed += 1
        if tasksProcessed == 25:
            updateQueue.put(tasksProcessed)
            tasksProcessed = 0

    # Final update before exiting
    updateQueue.put(tasksProcessed)


def scanRegionMulti() -> None:
    print("Initializing")
    moonGrid = pygmt.datasets.load_moon_relief(
        resolution="01m",
        region=[
            VIEW_REGION[0].lon,
            VIEW_REGION[1].lon,
            VIEW_REGION[0].lat,
            VIEW_REGION[1].lat,
        ],
        registration="gridline",
    )

    # generate list of transceiver locations
    transceivers: list[PointGeo] = [TX_COORD]
    lonAxis = np.arange(SCAN_REGION[0].lon, SCAN_REGION[1].lon, SCAN_BLOCK_SIZE)
    latAxis = np.arange(SCAN_REGION[0].lat, SCAN_REGION[1].lat, SCAN_BLOCK_SIZE)
    for lon in lonAxis:
        for lat in latAxis:
            point = PointGeo(lon=lon, lat=lat)
            if point != TX_COORD:
                transceivers.append(point)

    # Get heights for each transceiver
    transceiverHeights = cast(
        npt.NDArray[np.float64],
        pygmt.grdtrack(  # type: ignore
            moonGrid,
            points=np.array(transceivers),
            z_only=True,
            output_type="numpy",
        )[:, 0],
    )

    # Convert to Point3D
    txLoc = geoTo3D(transceivers[0], transceiverHeights[0] + TX_HEIGHT)
    receiverLocs = [
        geoTo3D(c, h + RX_HEIGHT)
        for c, h in zip(transceivers[1:], transceiverHeights[1:])
    ]

    # Load results into shared memory
    if not SCAN_RESULTS_PATH.exists():
        results = xarray.DataArray(
            dims=["lat", "lon"],
            coords={
                "lon": lonAxis,
                "lat": latAxis,
            },
        )
        results.to_netcdf(SCAN_RESULTS_PATH)  # type: ignore
    else:
        with xarray.open_dataarray(SCAN_RESULTS_PATH) as da:  # type: ignore
            results = da.load()  # type: ignore

    resultsMem = mp.RawArray("b", results.nbytes)
    resultsArr = np.frombuffer(resultsMem, dtype=np.float64)
    resultsArr = resultsArr.reshape((len(latAxis), len(lonAxis)))
    np.copyto(resultsArr, results)

    # Replace data array with shared mem
    results = xarray.DataArray(
        resultsArr,
        dims=["lat", "lon"],
        coords={
            "lon": lonAxis,
            "lat": latAxis,
        },
    )

    # Generate tasks
    tasks: list[tuple[Point3D, PointGeo]] = list(zip(receiverLocs, transceivers[1:]))

    cancelEvent = mp.Event()

    with mp.Pool(
        N_WORKERS, scanRegionMultiWorkerInit, initargs=(resultsMem, cancelEvent)
    ) as pool:
        tasksComplete = 0
        mpManager = mp.Manager()
        updateQueue = cast(mpq.Queue[int], mpManager.Queue())

        # Split tasks and create process arguments
        procArgs: list[
            tuple[
                Point3D,
                PointGeo,
                list[tuple[Point3D, PointGeo]],
                mpq.Queue[int],
            ]
        ] = []
        chunkSize, remainder = divmod(len(tasks), N_WORKERS)
        if remainder != 0:
            chunkSize += 1

        for i in range(0, len(tasks), chunkSize):
            procArgs.append((txLoc, TX_COORD, tasks[i : i + chunkSize], updateQueue))

        # create processes
        processes = pool.starmap_async(scanRegionMultiWorker, procArgs)
        print("Started workers")

        try:
            while True:
                # Check that all processes are alive
                if processes.ready():
                    break

                # Update and report counter
                try:
                    taskInc = updateQueue.get_nowait()
                    tasksComplete += taskInc
                    print(
                        f"Completed {tasksComplete} of {len(tasks)} tasks, or {tasksComplete / len(tasks) * 100:0.2f}%"
                    )
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            print("Stopping workers", flush=True)
            cancelEvent.set()

            while True:
                # Check that all processes are alive
                if processes.ready():
                    break

                # Update and report counter
                try:
                    taskInc = updateQueue.get_nowait()
                    tasksComplete += taskInc
                    print(
                        f"Completed {tasksComplete} of {len(tasks)} tasks, or {tasksComplete / len(tasks) * 100:0.2f}%"
                    )
                except queue.Empty:
                    pass

        print("Workers stopped, saving results")
        with DelayedKeyboardInterrupt():
            results.to_netcdf(SCAN_RESULTS_PATH)  # type: ignore

        processes.get()  # Propagate any exceptions

    print("Exiting")


def showHeatmap() -> None:
    moonGrid = pygmt.datasets.load_moon_relief(
        resolution="01m",
        region=[
            VIEW_REGION[0].lon,
            VIEW_REGION[1].lon,
            VIEW_REGION[0].lat,
            VIEW_REGION[1].lat,
        ],
        registration="gridline",
    )

    with xarray.open_dataarray(SCAN_RESULTS_PATH) as da:  # type: ignore
        results = da.load()  # type: ignore

    projection = "G00/-90/12c"
    fig = pygmt.Figure()

    fig.grdcontour(  # type: ignore
        grid=moonGrid,
        # annotation="1000+f8p",  # Annotate contours every 1000 meters
        pen="0.75p,blue",  # Contour line style
        projection=projection,
    )

    # Create a colormap for the secondary data

    pygmt.makecpt(  # type: ignore
        cmap="jet",  # Color palette
        series=[
            results.min().item(),
            results.max().item(),
            0.01,
        ],  # Data range [min, max, increment]
        continuous=True,  # Use continuous colormap
    )

    # Overlay the secondary data as a color map
    fig.grdimage(  # type: ignore
        grid=results,
        cmap=True,  # Use the previously created colormap
        transparency=25,  # Optional transparency level (0-100)
        projection=projection,
    )
    fig.colorbar(frame=["x+lSignal Strength", "y+ldBm"])  # type: ignore

    # Add map frame and labels
    fig.basemap(  # type: ignore
        region=[
            VIEW_REGION[0].lon,
            VIEW_REGION[1].lon,
            VIEW_REGION[0].lat,
            VIEW_REGION[1].lat,
        ],
        projection=projection,
        frame=["afg"],
    )

    fig.show()  # type: ignore


if __name__ == "__main__":
    main()
    # scanRegion()
    # scanRegionMulti()
    # showHeatmap()
