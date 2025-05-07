import argparse
import csv
import enum
import json
import pathlib
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pygmt  # type: ignore
import xarray as xr

from lwchm import spatial

NO_SIGNAL_LEVEL = -150


class Site(enum.Enum):
    A = 0
    F = 1
    K = 2
    L = 3


class Source(enum.Enum):
    DRATS = 0
    FAST = 1
    AGGRESSIVE = 2
    ITWOM = 3
    ITM = 4


type SiteData = dict[Site, dict[Source, xr.DataArray]]


SITE_NAME_MAP = {
    Site.A: "Site A",
    Site.F: "Site F",
    Site.K: "Site K",
    Site.L: "Site L",
}
SITE_PATH_MAP = {
    Site.A: "siteA",
    Site.F: "siteF",
    Site.K: "siteK",
    Site.L: "siteL",
}

SOURCE_NAME_MAP = {
    Source.DRATS: "DRATS",
    Source.FAST: "BICTR (Fast)",
    Source.AGGRESSIVE: "BICTR (Aggressive)",
    Source.ITWOM: "ITWOM",
    Source.ITM: "ITM",
}

SOURCE_PATH_MAP = {
    Source.DRATS: "drats.png",
    Source.FAST: "fast.png",
    Source.AGGRESSIVE: "aggressive.png",
    Source.ITWOM: "itwom.png",
    Source.ITM: "itm.png",
}

SOURCE_DIFF_PATH_MAP = {
    Source.FAST: "fastDiff.png",
    Source.AGGRESSIVE: "aggressiveDiff.png",
    Source.ITWOM: "itwomDiff.png",
    Source.ITM: "itmDiff.png",
}


@dataclass
class SiteConfig(object):
    drats: pathlib.Path
    fast: pathlib.Path
    aggressive: pathlib.Path
    itwom: pathlib.Path
    itm: pathlib.Path
    viewRegion: tuple[spatial.PointGeo, spatial.PointGeo]


@dataclass
class ProgramConfig(object):
    output: pathlib.Path
    sites: dict[Site, SiteConfig]

    suppressFigures: bool


def getSiteConfig(rawConf: dict[str, Any], site: str) -> SiteConfig:
    return SiteConfig(
        drats=rawConf[site]["drats"],
        fast=rawConf[site]["fast"],
        aggressive=rawConf[site]["aggressive"],
        itwom=rawConf[site]["itwom"],
        itm=rawConf[site]["itm"],
        viewRegion=(
            spatial.PointGeo(
                rawConf[site]["viewRegion"][0][0],
                rawConf[site]["viewRegion"][0][1],
            ),
            spatial.PointGeo(
                rawConf[site]["viewRegion"][1][0],
                rawConf[site]["viewRegion"][1][1],
            ),
        ),
    )


def getConfig() -> ProgramConfig:
    parser = argparse.ArgumentParser("compare_data.py")
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("--suppress-figures", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    with open(args.config) as configFile:
        rawConf = json.load(configFile)

    return ProgramConfig(
        output=pathlib.Path(rawConf["output"]),
        sites={
            Site.A: getSiteConfig(rawConf, "siteA"),
            Site.F: getSiteConfig(rawConf, "siteF"),
            Site.K: getSiteConfig(rawConf, "siteK"),
            Site.L: getSiteConfig(rawConf, "siteL"),
        },
        suppressFigures=args.suppress_figures,
    )


def narrowData(
    viewRegion: tuple[spatial.PointGeo, spatial.PointGeo], da: xr.DataArray
) -> xr.DataArray:
    lonMask = (da.lon >= viewRegion[0].lon) & (da.lon <= viewRegion[1].lon)
    latMask = (da.lat >= viewRegion[0].lat) & (da.lat <= viewRegion[1].lat)
    mask = lonMask & latMask

    da = da.where(mask, drop=True)
    return da


def loadSiteData(
    siteConfig: SiteConfig, body: spatial.Body
) -> dict[Source, xr.DataArray]:
    sources: dict[Source, xr.DataArray] = {}
    with xr.open_dataarray(siteConfig.drats) as da:  # type: ignore
        sources[Source.DRATS] = da.load()  # type: ignore
    with xr.open_dataarray(siteConfig.fast) as da:  # type: ignore
        sources[Source.FAST] = da.load()  # type: ignore
    with xr.open_dataarray(siteConfig.aggressive) as da:  # type: ignore
        sources[Source.AGGRESSIVE] = da.load()  # type: ignore
    with xr.open_dataarray(siteConfig.itwom) as da:  # type: ignore
        sources[Source.ITWOM] = da.load()  # type: ignore
    with xr.open_dataarray(siteConfig.itm) as da:  # type: ignore
        sources[Source.ITM] = da.load()  # type: ignore

    for source, da in sources.items():
        da = narrowData(siteConfig.viewRegion, da)
        if source != Source.DRATS:
            da = da.where(np.isfinite(da), NO_SIGNAL_LEVEL)

        da = da.reindex_like(
            body.grid,
            method="nearest",
            tolerance=1e-6,
            fill_value=-np.inf,  # type: ignore
        )
        sources[source] = da

    return sources


def computeSiteDiff(sources: dict[Source, xr.DataArray]) -> dict[Source, xr.DataArray]:
    assert Source.DRATS in sources

    diffs: dict[Source, xr.DataArray] = {}
    for source, da in sources.items():
        if source == Source.DRATS:
            continue
        diff = da - sources[Source.DRATS]
        diffs[source] = diff.where(np.isfinite(diff), np.nan)

    return diffs


def recordStats(conf: ProgramConfig, siteDiffs: SiteData) -> None:
    with open(conf.output / "stats.csv", mode="w", newline="") as csvFile:
        print(
            "Site  , Model             ,   Mean,   Std\n",
            "-----------------------------------------",
            sep="",
        )
        writer = csv.writer(csvFile)
        writer.writerow(("Site", "Model", "Mean", "Std"))

        for site, diffs in siteDiffs.items():
            means = {
                source: cast(float, da.mean().item()) for (source, da) in diffs.items()
            }
            stds = {
                model: cast(float, da.std().item()) for (model, da) in diffs.items()
            }
            rows = [
                (SITE_NAME_MAP[site], SOURCE_NAME_MAP[s], means[s], stds[s])
                for s in means
            ]

            print(
                "\n".join(
                    "{0}, {1:<18}, {2:>6.2f}, {3:>5.2f}".format(*row) for row in rows
                )
            )

            writer.writerows(rows)


def computeGlobalDiffExtrema(siteDiffData: SiteData) -> tuple[float, float]:
    mins: list[float] = []
    maxs: list[float] = []
    for sources in siteDiffData.values():
        mins.extend(model.min().item() for model in sources.values())  # type: ignore
        maxs.extend(model.max().item() for model in sources.values())  # type: ignore

    return min(mins), max(maxs)


def createCombinedFigure(
    siteData: SiteData,
    conf: ProgramConfig,
    bodies: dict[Site, spatial.Body],
    scaleMin: float,
    scaleMax: float,
    isDiff: bool,
):
    fig = pygmt.Figure()
    pygmt.makecpt(  # type: ignore
        cmap="jet",
        series=[
            scaleMin,
            scaleMax,
            0.01,
        ],
        continuous=True,
    )

    for i, (site, sources) in enumerate(siteData.items()):
        if i == 1:
            fig.shift_origin(xshift="w+2c")
        elif i == 2:
            fig.shift_origin(xshift="-w-2c", yshift="-h-2.5c")
        elif i == 3:
            fig.shift_origin(xshift="w+2c")

        region = [
            conf.sites[site].viewRegion[0].lon,
            conf.sites[site].viewRegion[1].lon,
            conf.sites[site].viewRegion[0].lat,
            conf.sites[site].viewRegion[1].lat,
        ]

        title = f"{SITE_NAME_MAP[site]} Diff" if isDiff else f"{SITE_NAME_MAP[site]}"
        with fig.subplot(  # type: ignore
            nrows=2,
            ncols=2,
            subsize="6c",
            projection="M6c",
            region=region,
            frame="agf",
            sharex="b",
            sharey="l",
            title=title,
            autolabel="+JTC+o0c/0.2c",
            margins="0.2c",
        ):
            with fig.set_panel(0, fixedlabel=SOURCE_NAME_MAP[Source.FAST]):  # type: ignore
                fig.grdcontour(  # type: ignore
                    grid=bodies[site].grid,
                    projection="M?",
                    pen="0.75p,blue",
                    region=region,
                )
                fig.grdimage(  # type: ignore
                    grid=sources[Source.FAST],
                    cmap=True,
                    transparency=25,
                    region=region,
                )
            with fig.set_panel(1, fixedlabel=SOURCE_NAME_MAP[Source.AGGRESSIVE]):  # type: ignore
                fig.grdcontour(  # type: ignore
                    grid=bodies[site].grid,
                    projection="M?",
                    pen="0.75p,blue",
                    region=region,
                )
                fig.grdimage(  # type: ignore
                    grid=sources[Source.AGGRESSIVE],
                    cmap=True,
                    transparency=25,
                    region=region,
                )
            with fig.set_panel(2, fixedlabel=SOURCE_NAME_MAP[Source.ITM]):  # type: ignore
                fig.grdcontour(  # type: ignore
                    grid=bodies[site].grid,
                    projection="M?",
                    pen="0.75p,blue",
                    region=region,
                )
                fig.grdimage(  # type: ignore
                    grid=sources[Source.ITM],
                    cmap=True,
                    transparency=25,
                    region=region,
                )
            with fig.set_panel(3, fixedlabel=SOURCE_NAME_MAP[Source.ITWOM]):  # type: ignore
                fig.grdcontour(  # type: ignore
                    grid=bodies[site].grid,
                    projection="M?",
                    pen="0.75p,blue",
                    region=region,
                )
                fig.grdimage(  # type: ignore
                    grid=sources[Source.ITWOM],
                    cmap=True,
                    transparency=25,
                    region=region,
                )

    if isDiff:
        fig.colorbar(  # type: ignore
            frame=["x+lSignal Strength", "y+lΔdBm"], position="JBC+o-7.25c/1c"
        )
    else:
        fig.colorbar(  # type: ignore
            frame=["x+lSignal Strength", "y+ldBm"], position="JBC+o-7.25c/1c"
        )

    if not conf.suppressFigures:
        fig.show()  # type: ignore

    if isDiff:
        outputPath = conf.output / "combinedDiff.png"
    else:
        outputPath = conf.output / "combined.png"
    outputPath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outputPath)  # type: ignore


def createIndividualFigures(
    siteData: SiteData,
    conf: ProgramConfig,
    bodies: dict[Site, spatial.Body],
    scaleMin: float,
    scaleMax: float,
    isDiff: bool,
) -> None:
    for site, sources in siteData.items():
        sitePath = conf.output / SITE_PATH_MAP[site]
        sitePath.mkdir(exist_ok=True, parents=True)
        region = [
            conf.sites[site].viewRegion[0].lon,
            conf.sites[site].viewRegion[1].lon,
            conf.sites[site].viewRegion[0].lat,
            conf.sites[site].viewRegion[1].lat,
        ]

        for source, da in sources.items():
            fig = pygmt.Figure()

            fig.grdcontour(  # type: ignore
                grid=bodies[site].grid,
                pen="0.75p,blue",
                projection="M15c",
                region=region,
            )

            # Create a colormap for the secondary data
            pygmt.makecpt(  # type: ignore
                cmap="jet",
                series=[
                    scaleMin,
                    scaleMax,
                    0.01,
                ],
                continuous=True,
            )

            # Overlay the secondary data as a color map
            fig.grdimage(  # type: ignore
                grid=da,
                cmap=True,  # Use the previously created colormap
                transparency=25,  # Optional transparency level (0-100)
                projection="M15c",
                region=region,
            )
            if isDiff:
                fig.colorbar(  # type: ignore
                    frame=["x+lSignal Strength", "y+lΔdBm"], position="JBC+o0c/1c"
                )
            else:
                fig.colorbar(  # type: ignore
                    frame=["x+lSignal Strength", "y+ldBm"], position="JBC+o0c/1c"
                )

            # Add map frame and labels
            fig.basemap(  # type: ignore
                region=region,
                projection="M15c",
                frame=["afg", f"+t{SITE_NAME_MAP[site]}: {SOURCE_NAME_MAP[source]}"],
                map_scale="jTR+w500e+f+o0.5+u",
            )

            if not conf.suppressFigures:
                fig.show()  # type: ignore

            if isDiff:
                outputPath = sitePath / SOURCE_DIFF_PATH_MAP[source]
            else:
                outputPath = sitePath / SOURCE_PATH_MAP[source]
            fig.savefig(outputPath)  # type: ignore
    pass


def main() -> None:
    # Load config and data
    conf = getConfig()
    bodies = {
        site: spatial.Body("earth", "01s", conf.viewRegion[0], conf.viewRegion[1])
        for site, conf in conf.sites.items()
    }
    siteData = {
        site: loadSiteData(conf, bodies[site]) for site, conf in conf.sites.items()
    }
    maskedSiteData = {
        site: {
            source: da.where(np.isfinite(da), np.nan) for source, da in sources.items()
        }
        for site, sources in siteData.items()
    }
    siteDiffData = {
        site: computeSiteDiff(sources) for site, sources in siteData.items()
    }

    # Prepare output directory
    conf.output.mkdir(exist_ok=True, parents=True)

    # Stats
    recordStats(conf, siteDiffData)

    minDiff, maxDiff = computeGlobalDiffExtrema(siteDiffData)
    createCombinedFigure(maskedSiteData, conf, bodies, -150, 0, False)
    createCombinedFigure(siteDiffData, conf, bodies, minDiff, maxDiff, True)
    createIndividualFigures(maskedSiteData, conf, bodies, -150, 0, False)
    createIndividualFigures(siteDiffData, conf, bodies, minDiff, maxDiff, True)


if __name__ == "__main__":
    main()
