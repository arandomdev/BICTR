import argparse
import base64
import json
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


class Configuration(object):
    resultsPath: pathlib.Path
    workers: int  # Number of cores to use

    body: Literal["moon", "earth"]
    resolution: str

    scanMode: Literal["region", "track"]
    scanRegion: tuple[spatial.PointGeo, spatial.PointGeo] | None
    scanTrack: pathlib.Path | None

    tx: spatial.PointGeo
    txHeight: float
    rxHeight: float

    modelType: Literal["lwchm"]
    lwchmConf: model.LWCHMConfiguration | None

    signalType: Literal["bpsk", "qpsk"]
    pskConf: lwchm.signal.PSKConfiguration | None

    projection: str

    def __init__(self, raw: dict[str, Any]) -> None:
        self.resultsPath = pathlib.Path(raw["resultsPath"])
        if raw["workers"] == -1:
            self.workers = mp.cpu_count()
        else:
            self.workers = raw["workers"]

        self.body = raw["body"]
        self.resolution = raw["resolution"]

        self.scanMode = raw["scanMode"]
        if self.scanMode == "region":
            self.scanRegion = (
                spatial.PointGeo(raw["scanRegion"][0][0], raw["scanRegion"][0][1]),
                spatial.PointGeo(raw["scanRegion"][1][0], raw["scanRegion"][1][1]),
            )
        elif self.scanMode == "track":
            self.scanTrack = pathlib.Path(raw["scanTrack"])
        else:
            raise NotImplementedError

        self.tx = spatial.PointGeo(raw["tx"][0], raw["tx"][1])
        self.txHeight = raw["txHeight"]
        self.rxHeight = raw["rxHeight"]

        self.modelType = raw["model"]
        if self.modelType == "lwchm":
            self.lwchmConf = model.LWCHMConfiguration(**raw["lwchm"])
        else:
            raise NotImplementedError

        self.signalType = raw["signal"]
        if self.signalType == "bpsk" or self.signalType == "qpsk":
            # Convert base64 to bytes
            data = base64.decodebytes(raw["psk"]["data"].encode())
            rawPskConf = raw["psk"]
            rawPskConf["data"] = data
            self.pskConf = lwchm.signal.PSKConfiguration(**rawPskConf)
        else:
            raise NotImplementedError

        self.projection = raw["projection"]


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


def getArgs() -> Configuration:
    parser = argparse.ArgumentParser(
        prog="ScanRegion", description="Scan a region and generate a heatmap."
    )
    parser.add_argument("config_file", type=pathlib.Path)
    configPath = cast(pathlib.Path, parser.parse_args().config_file)
    return Configuration(json.loads(configPath.read_text()))


def getBodyAndAxes(
    progConfig: Configuration,
) -> tuple[spatial.Body, xr.DataArray, xr.DataArray]:
    assert progConfig.lwchmConf is not None

    if progConfig.scanMode == "region":
        assert progConfig.scanRegion is not None
        body = spatial.Body(
            progConfig.body,
            progConfig.resolution,
            progConfig.scanRegion[0],
            progConfig.scanRegion[1],
            progConfig.lwchmConf.ringRadiusMax
            + progConfig.lwchmConf.ringRadiusUncertainty,
        )
        lonAxis = body.grid.lon.sel(
            lon=slice(progConfig.scanRegion[0].lon, progConfig.scanRegion[1].lon)
        )
        latAxis = body.grid.lat.sel(
            lat=slice(progConfig.scanRegion[0].lat, progConfig.scanRegion[1].lat)
        )

    elif progConfig.scanMode == "track":
        with xr.open_dataarray(progConfig.scanTrack) as track:  # type: ignore
            lonMin = track.lon.min().item()
            latMin = track.lat.min().item()
            lonMax = track.lon.max().item()
            latMax = track.lat.max().item()
            body = spatial.Body(
                progConfig.body,
                progConfig.resolution,
                spatial.PointGeo(lonMin, latMin),
                spatial.PointGeo(lonMax, latMax),
                progConfig.lwchmConf.ringRadiusMax
                + progConfig.lwchmConf.ringRadiusUncertainty,
            )

            track = track.reindex_like(body.grid, method="nearest", tolerance=1e-6)
            nearestLow = track.sel(lat=latMin, lon=lonMin, method="nearest")  # type: ignore
            nearestHigh = track.sel(lat=latMax, lon=lonMax, method="nearest")  # type: ignore
            lonAxis = track.lon.sel(
                lon=slice(nearestLow.lon.item(), nearestHigh.lon.item())
            )
            latAxis = track.lat.sel(
                lat=slice(nearestLow.lat.item(), nearestHigh.lat.item())
            )
    else:
        raise NotImplementedError

    return body, lonAxis, latAxis


def scanRegionMultiWorkerInit(
    _progConfig: Configuration,
    _resultsMem: Any,
    _cancelEvent: mps.Event,
    _updateQueue: "mp.Queue[int]",
) -> None:
    global progConfig
    global resultsMem
    global cancelEvent
    global updateQueue
    progConfig = _progConfig
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

    assert progConfig.modelType == "lwchm"
    assert progConfig.lwchmConf is not None
    body, lonAxis, latAxis = getBodyAndAxes(progConfig)
    chModel = model.LWCHM(body, progConfig.lwchmConf)

    # Construct dataarray from shared memory
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

    assert progConfig.signalType == "bpsk" or progConfig.signalType == "qpsk"
    assert progConfig.pskConf is not None
    if progConfig.signalType == "bpsk":
        txSig = lwchm.signal.generateBPSKSignal(progConfig.pskConf)
    else:
        txSig = lwchm.signal.generateQPSKSignal(progConfig.pskConf)

    tasksProcessed = 0

    for rxCoord in tasks:
        if np.isnan(results.loc[rxCoord.lat, rxCoord.lon]):  # type: ignore
            rxSig = chModel.compute(
                txCoord, rxCoord, progConfig.txHeight, progConfig.rxHeight, txSig
            )
            if rxSig:
                rxStrength = lwchm.signal.computeRmsDBM(rxSig.wave)
                results.loc[rxCoord.lat, rxCoord.lon] = rxStrength  # type: ignore
            else:
                results.loc[rxCoord.lat, rxCoord.lon] = -np.inf  # type: ignore

        # check cancel and update
        if cancelEvent.is_set():
            break

        tasksProcessed += 1
        if tasksProcessed == 25:
            updateQueue.put(tasksProcessed)
            tasksProcessed = 0

    # Final update before exiting
    updateQueue.put(tasksProcessed)


def showHeatmap(
    progConfig: Configuration,
    body: spatial.Body,
    results: xr.DataArray,
) -> None:
    fig = pygmt.Figure()

    maskedResults = results.where(np.isfinite(results), np.nan)

    regionPoints = [
        results.lon.min().item(),
        results.lon.max().item(),
        results.lat.min().item(),
        results.lat.max().item(),
    ]
    fig.grdcontour(  # type: ignore
        grid=body.grid,
        region=regionPoints,
        pen="0.75p,blue",  # Contour line style
        projection=progConfig.projection,
    )

    # Create a colormap for the secondary data

    pygmt.makecpt(  # type: ignore
        cmap="jet",  # Color palette
        series=[
            maskedResults.min().item(),
            maskedResults.max().item(),
            0.01,
        ],  # Data range [min, max, increment]
        continuous=True,  # Use continuous colormap
    )

    # Overlay the secondary data as a color map
    fig.grdimage(  # type: ignore
        grid=maskedResults,
        cmap=True,  # Use the previously created colormap
        transparency=25,  # Optional transparency level (0-100)
        projection=progConfig.projection,
    )
    fig.colorbar(frame=["x+lSignal Strength", "y+ldBm"], position="JBC+o0c/1c")  # type: ignore

    # Add map frame and labels
    fig.basemap(  # type: ignore
        projection=progConfig.projection,
        frame=["afg"],
        map_scale="jBR+w500e",
    )

    fig.show()  # type: ignore


def main() -> None:
    progConfig = getArgs()

    # Create body now incase the relief needs to be downloaded
    print("Initializing")
    body, lonAxis, latAxis = getBodyAndAxes(progConfig)

    # generate list of transceiver locations
    transceivers: list[spatial.PointGeo] = [progConfig.tx]
    if progConfig.scanMode == "region":
        for lon in lonAxis:
            for lat in latAxis:
                point = spatial.PointGeo(lon=lon, lat=lat)
                if point != progConfig.tx:
                    transceivers.append(point)
    elif progConfig.scanMode == "track":
        assert progConfig.scanTrack is not None
        with xr.open_dataarray(progConfig.scanTrack) as track:  # type: ignore
            track = track.reindex_like(body.grid, method="nearest", tolerance=1e-6)
            trackStacked = track.stack(point=["lat", "lon"])
            trackStacked = trackStacked[np.isfinite(trackStacked)]

            for point in trackStacked:
                pointCoord = spatial.PointGeo(point.lon.item(), point.lat.item())
                if pointCoord != progConfig.tx:
                    transceivers.append(pointCoord)
    else:
        raise NotImplementedError

    # Load results into shared memory
    if not progConfig.resultsPath.exists():
        results = xr.DataArray(
            dims=["lat", "lon"],
            coords={
                "lon": lonAxis,
                "lat": latAxis,
            },
        )
        progConfig.resultsPath.parent.mkdir(exist_ok=True, parents=True)
        results.to_netcdf(progConfig.resultsPath)  # type: ignore
    else:
        with xr.open_dataarray(progConfig.resultsPath) as da:  # type: ignore
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
    updateQueue = cast("mp.Queue[int]", mp.Queue())

    with mp.Pool(
        progConfig.workers,
        scanRegionMultiWorkerInit,
        initargs=(progConfig, resultsMem, cancelEvent, updateQueue),
    ) as pool:
        tasksComplete = 0

        # Split tasks and create process arguments
        procArgs: list[
            tuple[
                spatial.PointGeo,
                list[spatial.PointGeo],
            ]
        ] = []
        chunkSize, remainder = divmod(len(tasks), progConfig.workers)
        if remainder != 0:
            chunkSize += 1

        for i in range(0, len(tasks), chunkSize):
            procArgs.append((progConfig.tx, tasks[i : i + chunkSize]))

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
            results.to_netcdf(progConfig.resultsPath)  # type: ignore

        processes.get()  # Propagate any exceptions

    showHeatmap(progConfig, body, results)
    print("Exiting")


if __name__ == "__main__":
    main()
