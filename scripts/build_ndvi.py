"""
Utility script for preparing an NDVI raster from NAIP tiles.

Usage
-----
Example invocation that ingests a folder full of NAIP ZIP archives or GeoTIFFs
and writes an NDVI mosaic next to them:

```
python scripts/build_ndvi.py \
    --input-dir src/ames/data/Parcel_Shapefile/naip \
    --output src/ames/data/naip/ndvi_ames.tif
```
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

try:
    import rasterio
    from rasterio.merge import merge
except ImportError as exc:  # pragma: no cover - import has side-effects only once
    raise ImportError(
        "rasterio is required to build NDVI rasters. "
        "Install project dependencies or `pip install rasterio`."
    ) from exc


LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NDVI raster from NAIP tiles.")
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing NAIP ZIP archives and/or GeoTIFF tiles.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination path for the NDVI GeoTIFF.",
    )
    parser.add_argument(
        "--ndvi-band",
        type=int,
        default=None,
        help=(
            "If provided, treat this band (1-based) as pre-computed NDVI and "
            "write it directly. When omitted, NDVI is derived from red/nir bands."
        ),
    )
    parser.add_argument(
        "--red-band",
        type=int,
        default=1,
        help="Band index (1-based) containing the red channel when computing NDVI.",
    )
    parser.add_argument(
        "--nir-band",
        type=int,
        default=4,
        help="Band index (1-based) containing the near-infrared channel.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console logging level.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file {args.output!s} already exists. "
            "Pass --overwrite to replace it."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="naip_extract_") as tmpdir:
        intermediate = Path(tmpdir)
        tiff_paths = collect_rasters(args.input_dir, intermediate)
        LOGGER.info("Discovered %d raster tiles.", len(tiff_paths))
        ndvi, transform, profile = build_ndvi(
            tiff_paths,
            ndvi_band=args.ndvi_band,
            red_band=args.red_band,
            nir_band=args.nir_band,
        )

    write_geotiff(args.output, ndvi, transform, profile)
    LOGGER.info("NDVI raster written to %s", args.output)


def collect_rasters(input_dir: Path, extract_dir: Path) -> List[Path]:
    """
    Gather GeoTIFF raster paths from a directory of NAIP downloads.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir!s} does not exist.")

    tiffs = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    archives = list(input_dir.glob("*.zip")) + list(input_dir.glob("*.ZIP"))

    for archive in archives:
        LOGGER.debug("Extracting %s", archive)
        try:
            with zipfile.ZipFile(archive) as zf:
                for member in zf.namelist():
                    member_path = Path(member)
                    if member_path.suffix.lower() not in {".tif", ".tiff"}:
                        continue
                    target = extract_dir / member_path.name
                    with zf.open(member) as source, open(target, "wb") as sink:
                        sink.write(source.read())
                    tiffs.append(target)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"Failed to extract {archive!s}: {exc}") from exc

    if not tiffs:
        raise FileNotFoundError(
            f"No GeoTIFF tiles found in {input_dir!s}. "
            "Provide NAIP .tif files or ZIP downloads."
        )

    return tiffs


def build_ndvi(
    tiff_paths: Iterable[Path],
    *,
    ndvi_band: Optional[int],
    red_band: int,
    nir_band: int,
):
    """
    Create an NDVI mosaic from the provided raster tiles.
    """
    datasets = [rasterio.open(path) for path in tiff_paths]
    try:
        mosaic, transform = merge(datasets)
        profile = datasets[0].profile
    finally:
        for ds in datasets:
            ds.close()

    if ndvi_band:
        LOGGER.info("Using band %d as supplied NDVI.", ndvi_band)
        ndvi_array = mosaic[ndvi_band - 1].astype("float32")
    else:
        LOGGER.info(
            "Computing NDVI from red band %d and NIR band %d.", red_band, nir_band
        )
        red = mosaic[red_band - 1].astype("float32")
        nir = mosaic[nir_band - 1].astype("float32")
        denom = nir + red
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi_array = np.where(denom != 0, (nir - red) / denom, np.nan).astype(
                "float32"
            )

    return ndvi_array, transform, profile


def write_geotiff(output: Path, array: np.ndarray, transform, profile) -> None:
    """
    Persist the NDVI array as a single-band GeoTIFF.
    """
    profile = profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        transform=transform,
        nodata=np.nan,
        compress="LZW",
        bigtiff="IF_SAFER",
    )

    with rasterio.open(output, "w", **profile) as dst:
        dst.write(array, 1)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
