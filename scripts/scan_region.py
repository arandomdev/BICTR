import argparse
import multiprocessing as mp
import multiprocessing.synchronize as mps
import pathlib
import queue
import signal
from types import FrameType, TracebackType
from typing import Any, Literal, cast

import numpy as np
import pygmt  # type: ignore
import xarray as xr

import lwchm.signal
from lwchm import model, spatial

MOON_RADIUS = 1737.4e3

LOG_ZERO_STUB = -150  # Value to show when taking log of 0


BPSK_TRANSMIT_POWER = 20  # Transmit power in dBm
BPSK_DATA = b"deadbeef"
BPSK_DATA_LEN = 32
BPSK_FS = 7e9
BPSK_CARR_FS = 500e6
BPSK_SYMBOL_PERIOD = 1 / BPSK_FS * 20

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

# Ground reflection params
COMPLEX_RELATIVITY_REAL = 2.75
COMPLEX_RELATIVITY_REAL_STD = 0.115
COMPLEX_RELATIVITY_IMAG = 0.13
COMPLEX_RELATIVITY_IMAG_STD = 0.047


class RawArguments(argparse.Namespace):
    body: Literal["moon", "earth"]

    view_region: tuple[float, float, float, float]
    scan_region: tuple[float, float, float, float]
    scan_block_size: float
    tx: tuple[float, float]
    tx_height: float
    rx_height: float

    results_path: pathlib.Path

    workers: int


class Arguments(object):
    body: Literal["moon", "earth"]
    bodyRadius: float

    viewRegion: tuple[spatial.PointGeo, spatial.PointGeo]
    scanRegion: tuple[spatial.PointGeo, spatial.PointGeo]
    scanBlockSize: float  # Coordinate increment amount in XY
    tx: spatial.PointGeo
    txHeight: float
    rxHeight: float

    resultsPath: pathlib.Path

    workers: int  # Number of cores to use

    def __init__(self, args: RawArguments) -> None:
        self.body = args.body
        if self.body == "moon":
            self.bodyRadius = MOON_RADIUS
        else:
            raise NotImplementedError

        self.viewRegion = (
            spatial.PointGeo(args.view_region[0], args.view_region[1]),
            spatial.PointGeo(args.view_region[2], args.view_region[3]),
        )
        self.scanRegion = (
            spatial.PointGeo(args.scan_region[0], args.scan_region[1]),
            spatial.PointGeo(args.scan_region[2], args.scan_region[3]),
        )
        self.scanBlockSize = args.scan_block_size
        self.tx = spatial.PointGeo(args.tx[0], args.tx[1])
        self.txHeight = args.tx_height
        self.rxHeight = args.rx_height
        self.resultsPath = args.results_path

        if args.workers == -1:
            self.workers = mp.cpu_count()
        else:
            self.workers = args.workers


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


def getArgs() -> Arguments:
    parser = argparse.ArgumentParser(
        prog="ScanRegion", description="Scan a region and generate a heatmap."
    )
    parser.add_argument(
        "body", choices=("moon", "earth"), help="Celestial body to use."
    )
    parser.add_argument(
        "view-region",
        metavar=("lon1", "lat1", "lon2", "lat2"),
        type=float,
        nargs=4,
        help="View region to use. The scan region must be within the view region.",
    )
    parser.add_argument(
        "scan-region",
        metavar=("lon1", "lat1", "lon2", "lat2"),
        type=float,
        nargs=4,
        help="View region to use. The scan region must be within the view region.",
    )
    parser.add_argument(
        "scan_block_size",
        type=float,
        help="Coordinate increment amount in both XY.",
    )
    parser.add_argument(
        "tx",
        metavar=("lon", "lat"),
        type=float,
        nargs=2,
        help="Transmitter coordinate.",
    )
    parser.add_argument(
        "tx-height", type=float, help="Height of the transmitter above the ground."
    )
    parser.add_argument(
        "rx-height", type=float, help="Height of the receiver above the ground."
    )
    parser.add_argument(
        "results-path",
        type=pathlib.Path,
        help="File path for results. Should have a .nc extension.",
    )
    parser.add_argument(
        "--workers", type=int, default=-1, help="Number of cores to use."
    )

    rawArgs = parser.parse_args(namespace=RawArguments())
    return Arguments(rawArgs)


def scanRegionMultiWorkerInit(
    _progArgs: Arguments,
    _resultsMem: Any,
    _cancelEvent: mps.Event,
    _updateQueue: "mp.Queue[int]",
) -> None:
    global progArgs
    global resultsMem
    global cancelEvent
    global updateQueue
    progArgs = _progArgs
    resultsMem = _resultsMem
    cancelEvent = _cancelEvent
    updateQueue = _updateQueue

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def scanRegionMultiWorker(
    txCoord: spatial.PointGeo,
    tasks: list[spatial.PointGeo],
) -> None:
    """Child process to scan a region

    Args:
        txLoc: The transmitter location
        tasks: A list of receiver locations
        cancelEvent: A event to cancel the child process and exit
        updateQueue: A feed back mechanism that reports how many tasks were processed
            since the last update from this process
    """

    grid = pygmt.datasets.load_moon_relief(
        resolution="01m",
        region=[  # type: ignore
            progArgs.viewRegion[0].lon,
            progArgs.viewRegion[1].lon,
            progArgs.viewRegion[0].lat,
            progArgs.viewRegion[1].lat,
        ],
        registration="gridline",
    )
    body = spatial.Body(progArgs.bodyRadius, grid)
    config = model.Configuration(
        refCount=REFLECTOR_COUNT,
        refAttemptPerRing=REFLECTOR_ATTEMPT_PER_RING,
        ringRadiusMin=RING_RADIUS_MIN,
        ringRadiusMax=RING_RADIUS_MAX,
        ringRadiusUncertainty=RING_RADIUS_UNCERTAINTY,
        ringCount=RING_COUNT,
        complexRelPermittivityReal=COMPLEX_RELATIVITY_REAL,
        complexRelPermittivityRealStd=COMPLEX_RELATIVITY_REAL_STD,
        complexRelPermittivityImag=COMPLEX_RELATIVITY_IMAG,
        complexRelPermittivityImagStd=COMPLEX_RELATIVITY_IMAG_STD,
        fadingPaths=FADING_N_PATHS,
        fadingDopplerSpread=FADING_MAX_DOPPLER_SPEED,
    )
    chModel = model.LWCHM(body, config)

    # Construct dataarray from shared memory
    lonAxis = np.arange(
        progArgs.scanRegion[0].lon, progArgs.scanRegion[1].lon, progArgs.scanBlockSize
    )
    latAxis = np.arange(
        progArgs.scanRegion[0].lat, progArgs.scanRegion[1].lat, progArgs.scanBlockSize
    )

    resultsArr = np.frombuffer(resultsMem, dtype=np.float64)
    resultsArr = resultsArr.reshape((len(latAxis), len(lonAxis)))

    results = xr.DataArray(
        resultsArr,
        dims=["lat", "lon"],
        coords={
            "lon": lonAxis,
            "lat": latAxis,
        },
    )

    txSig = lwchm.signal.generateBPSKSignal(
        BPSK_DATA, BPSK_SYMBOL_PERIOD, BPSK_FS, BPSK_CARR_FS, BPSK_TRANSMIT_POWER
    )

    tasksProcessed = 0

    for rxCoord in tasks:
        if np.isnan(results.loc[rxCoord.lat, rxCoord.lon]):  # type: ignore
            rxSig = chModel.compute(
                txCoord, rxCoord, progArgs.txHeight, progArgs.rxHeight, txSig
            )
            if rxSig:
                rxStrength = lwchm.signal.computeRmsDBM(rxSig.wave)
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


def main() -> None:
    args = getArgs()

    print("Initializing")
    # generate list of transceiver locations
    transceivers: list[spatial.PointGeo] = [args.tx]
    lonAxis = np.arange(
        args.scanRegion[0].lon, args.scanRegion[1].lon, args.scanBlockSize
    )
    latAxis = np.arange(
        args.scanRegion[0].lat, args.scanRegion[1].lat, args.scanBlockSize
    )
    for lon in lonAxis:
        for lat in latAxis:
            point = spatial.PointGeo(lon=lon, lat=lat)
            if point != args.tx:
                transceivers.append(point)

    # Load results into shared memory
    if not args.resultsPath.exists():
        results = xr.DataArray(
            dims=["lat", "lon"],
            coords={
                "lon": lonAxis,
                "lat": latAxis,
            },
        )
        results.to_netcdf(args.resultsPath)  # type: ignore
    else:
        with xr.open_dataarray(args.resultsPath) as da:  # type: ignore
            results = da.load()  # type: ignore

    resultsMem = mp.RawArray("b", results.nbytes)
    resultsArr = np.frombuffer(resultsMem, dtype=np.float64)
    resultsArr = resultsArr.reshape((len(latAxis), len(lonAxis)))
    np.copyto(resultsArr, results)

    # Replace data array with shared mem
    results = xr.DataArray(
        resultsArr,
        dims=["lat", "lon"],
        coords={
            "lon": lonAxis,
            "lat": latAxis,
        },
    )

    # Generate tasks
    tasks: list[spatial.PointGeo] = list(transceivers[1:])

    cancelEvent = mp.Event()
    updateQueue = cast(mp.Queue[int], mp.Queue())

    with mp.Pool(
        args.workers, scanRegionMultiWorkerInit, initargs=(resultsMem, cancelEvent)
    ) as pool:
        tasksComplete = 0

        # Split tasks and create process arguments
        procArgs: list[
            tuple[
                spatial.PointGeo,
                list[spatial.PointGeo],
            ]
        ] = []
        chunkSize, remainder = divmod(len(tasks), args.workers)
        if remainder != 0:
            chunkSize += 1

        for i in range(0, len(tasks), chunkSize):
            procArgs.append((args.tx, tasks[i : i + chunkSize]))

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
            results.to_netcdf(args.resultsPath)  # type: ignore

        processes.get()  # Propagate any exceptions

    print("Exiting")


if __name__ == "__main__":
    main()
