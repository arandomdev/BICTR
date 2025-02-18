import argparse
import math

EARTH_RAD = 6378137


class ProgramArguments(argparse.Namespace):
    lon: float
    lat: float
    scan_radius: float
    view_port_extra: float


def main() -> None:
    parser = argparse.ArgumentParser("compute_region_bounds.py")
    parser.add_argument("lon", type=float, help="Transmitter longitude")
    parser.add_argument("lat", type=float, help="Transmitter latitude")
    parser.add_argument("scan_radius", type=float, help="Scan radius in meters")
    parser.add_argument(
        "view_port_extra", type=float, help="Additional space for the view port"
    )
    args = parser.parse_args(namespace=ProgramArguments())

    scanDelta = math.degrees(args.scan_radius / EARTH_RAD)

    scanLowLon = math.floor((args.lon - scanDelta) * 1e6) / 1e6
    scanLowLat = math.floor((args.lat - scanDelta) * 1e6) / 1e6

    scanHighLon = math.ceil((args.lon + scanDelta) * 1e6) / 1e6
    scanHighLat = math.ceil((args.lat + scanDelta) * 1e6) / 1e6

    viewDelta = math.degrees((args.scan_radius + args.view_port_extra) / EARTH_RAD)
    viewLowLon = math.floor((args.lon - viewDelta) * 1e6) / 1e6
    viewLowLat = math.floor((args.lat - viewDelta) * 1e6) / 1e6

    viewHighLon = math.ceil((args.lon + viewDelta) * 1e6) / 1e6
    viewHighLat = math.ceil((args.lat + viewDelta) * 1e6) / 1e6

    print(f"Scan Low, {scanLowLon}, {scanLowLat}")
    print(f"Scan High, {scanHighLon}, {scanHighLat}")
    print(f"View Low, {viewLowLon}, {viewLowLat}")
    print(f"View High, {viewHighLon}, {viewHighLat}")


if __name__ == "__main__":
    main()
