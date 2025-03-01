import csv
import pathlib

COMPARE_DIR = pathlib.Path(R"data/compare")


def combine(items: list[pathlib.Path], output: pathlib.Path) -> None:
    with open(output, mode="w", newline="") as combinedFile:
        writer = csv.writer(combinedFile)
        writer.writerow(("site", "Model", "Mean", "Std"))

        for i in items:
            site = i.name
            site = site.replace("_track", "")

            with open(i / "stats.csv", mode="r") as itemStats:
                reader = csv.reader(itemStats)
                next(reader)  # Skip header
                for row in reader:
                    row.insert(0, site)
                    writer.writerow(row)


def main() -> None:
    items = COMPARE_DIR.iterdir()
    items = [i for i in items if (i / "stats.csv").exists()]

    fastItems = [i for i in items if "track" not in i.name]
    aggressiveItems = [i for i in items if "track" in i.name]
    assert len(fastItems) + len(aggressiveItems) == len(items)

    fastItems.sort()
    aggressiveItems.sort()

    combine(fastItems, COMPARE_DIR / "fastStats.csv")
    combine(aggressiveItems, COMPARE_DIR / "aggressiveStats.csv")


if __name__ == "__main__":
    main()
