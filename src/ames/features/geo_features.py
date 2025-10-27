"""
Spatial helpers for enriching the Ames dataset with parcel-level NDVI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class GeoFeatureEngineer:
    """
    Geospatial feature helper that attaches parcel-level NDVI.

    Notes
    -----
    The implementation keeps things interview-friendly while still working with
    *real* rasters. Parcels are read from the provided shapefile and used to
    sample an NDVI raster (or, when necessary, a red/NIR band pair). The
    resulting column can then be merged onto the main feature matrix.
    """

    def __init__(self, config=None):
        self.config = config or {}
        paths = self.config.get("paths", {}) or {}
        geo_cfg = (
            (self.config.get("data_processing") or {}).get("geo_features") or {}
        )

        self.parcel_path = geo_cfg.get("parcel_shapefile") or paths.get(
            "parcel_shapefile"
        )
        self.parcel_id_column = geo_cfg.get("parcel_id_column", "PARCELID")
        self.features_join_column = geo_cfg.get("features_join_column", "PID")
        self.parcel_id_padding = geo_cfg.get("parcel_id_padding")
        self.output_column = geo_cfg.get("output_column", "parcel_ndvi")

        ndvi_cfg = geo_cfg.get("ndvi") or {}
        self.ndvi_raster = ndvi_cfg.get("raster") or paths.get("ndvi_raster")
        self.ndvi_band = ndvi_cfg.get("band", 1)
        self.nodata_override = ndvi_cfg.get("nodata")

        self.red_raster = ndvi_cfg.get("red_raster") or paths.get("ndvi_red_raster")
        self.nir_raster = ndvi_cfg.get("nir_raster") or paths.get("ndvi_nir_raster")
        self.red_band = ndvi_cfg.get("red_band", 1)
        self.nir_band = ndvi_cfg.get("nir_band", 1)

        self._parcel_frame = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load_reference_layers(self):
        """
        Load the parcel shapefile into a GeoDataFrame (cached).

        Returns
        -------
        geopandas.GeoDataFrame
            Parcels with identifier and geometry columns.

        Raises
        ------
        ValueError
            If the shapefile path is missing from configuration.
        FileNotFoundError
            When the shapefile cannot be found.
        ImportError
            If :mod:`geopandas` is not installed.
        KeyError
            When the configured parcel ID column is absent.
        """
        if self._parcel_frame is not None:
            return self._parcel_frame

        if not self.parcel_path:
            raise ValueError(
                "Parcel shapefile path is not configured. "
                "Set `paths.parcel_shapefile` or "
                "`data_processing.geo_features.parcel_shapefile`."
            )

        try:
            import geopandas as gpd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "geopandas is required to read parcel geometries. "
                "Install it via `pip install geopandas`."
            ) from exc

        shapefile_path = Path(self.parcel_path)
        if shapefile_path.is_dir():
            candidates = list(shapefile_path.glob("*.shp"))
            if not candidates:
                raise FileNotFoundError(
                    f"No shapefile found in directory {shapefile_path!s}."
                )
            shapefile_path = candidates[0]

        if not shapefile_path.exists():
            raise FileNotFoundError(f"Parcel shapefile not found at {shapefile_path!s}.")

        parcels = gpd.read_file(shapefile_path)
        if self.parcel_id_column not in parcels.columns:
            raise KeyError(
                f"Column '{self.parcel_id_column}' not present in {shapefile_path!s}."
            )

        parcels[self.parcel_id_column] = self._normalize_ids(
            parcels[self.parcel_id_column]
        )
        parcels = parcels.dropna(subset=[self.parcel_id_column]).reset_index(drop=True)
        self._parcel_frame = parcels
        return parcels

    def compute_ndvi(self) -> pd.DataFrame:
        """
        Compute average NDVI for each parcel polygon.

        Returns
        -------
        pandas.DataFrame
            Two-column frame with parcel identifiers and NDVI values.
        """
        external_frame = self._load_external_ndvi()
        if external_frame is not None:
            print(
                "[GeoFeatureEngineer] Using external NDVI source with "
                f"{len(external_frame)} rows."
            )
            return external_frame

        parcels = self.load_reference_layers()
        if parcels.empty:
            return pd.DataFrame(columns=[self.parcel_id_column, self.output_column])

        print(
            f"[GeoFeatureEngineer] Loaded {len(parcels)} parcels "
            f"(CRS={parcels.crs}, sample IDs={list(parcels[self.parcel_id_column].head(3))})"
        )
        self._debug_stats = {
            "mask_success": 0,
            "mask_fallback": 0,
            "point_success": 0,
            "point_nan": 0,
        }

        if self.ndvi_raster:
            ndvi_frame = self._ndvi_from_single_raster(
                parcels, Path(self.ndvi_raster), band=self.ndvi_band
            )
        elif self.red_raster and self.nir_raster:
            ndvi_frame = self._ndvi_from_band_pair(
                parcels,
                Path(self.red_raster),
                Path(self.nir_raster),
                red_band=self.red_band,
                nir_band=self.nir_band,
            )
        else:
            LOGGER.warning(
                "No NDVI raster configured; returning NaN for %d parcels.", len(parcels)
            )
            return pd.DataFrame(
                {
                    self.parcel_id_column: parcels[self.parcel_id_column],
                    self.output_column: [np.nan] * len(parcels),
                }
            )

        result = (
            ndvi_frame.dropna(subset=[self.parcel_id_column])
            .groupby(self.parcel_id_column, as_index=False)[self.output_column]
            .mean()
        )
        non_null = result[self.output_column].notna().sum()
        print(
            "[GeoFeatureEngineer] NDVI stats -> "
            f"{non_null}/{len(result)} non-null | "
            f"mask_success={self._debug_stats.get('mask_success', 0)} "
            f"mask_fallback={self._debug_stats.get('mask_fallback', 0)} "
            f"point_success={self._debug_stats.get('point_success', 0)} "
            f"point_nan={self._debug_stats.get('point_nan', 0)}"
        )
        LOGGER.info(
            "Computed NDVI for %d parcels (%d non-null).", len(result), non_null
        )
        return result

    def join_to_features(self, dataframe, geo_features=None):
        """
        Merge NDVI values into the tabular feature dataframe.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Feature matrix containing the join column (defaults to ``PID``).
        geo_features : pandas.DataFrame, optional
            Output of :meth:`compute_ndvi`. When ``None`` or empty the merge is
            skipped.

        Returns
        -------
        pandas.DataFrame or list
            Feature matrix augmented with the NDVI column or the original input
            when a merge cannot be performed.

        Raises
        ------
        KeyError
            If the NDVI frame is missing the parcel ID or NDVI columns.
        """
        if dataframe is None:
            return []
        if geo_features is None or getattr(geo_features, "empty", False):
            LOGGER.info("Geo features missing or empty; skipping NDVI join.")
            return dataframe

        join_col = self.features_join_column
        if join_col not in dataframe.columns:
            LOGGER.warning(
                "Join column '%s' not found on feature dataframe; skipping NDVI join.",
                join_col,
            )
            return dataframe

        geo_df = pd.DataFrame(geo_features).copy()
        required = {self.parcel_id_column, self.output_column}
        missing = required.difference(geo_df.columns)
        if missing:
            raise KeyError(
                f"Geo features missing expected columns: {sorted(missing)}"
            )

        left = dataframe.copy()
        left[join_col] = self._normalize_ids(left[join_col])

        right = geo_df[list(required)].copy()
        right[self.parcel_id_column] = self._normalize_ids(
            right[self.parcel_id_column]
        )

        merged = left.merge(
            right.rename(columns={self.parcel_id_column: join_col}),
            on=join_col,
            how="left",
        )
        return merged

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _load_external_ndvi(self) -> Optional[pd.DataFrame]:
        """
        Load a precomputed NDVI table when configured (CSV or GeoPackage).
        """
        ndvi_cfg = (self.config.get("data_processing") or {}).get("geo_features", {})
        external_cfg = ndvi_cfg.get("external_ndvi") or {}
        path = external_cfg.get("path")
        if not path:
            return None

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(
                f"Configured external NDVI source not found at {path_obj!s}."
            )

        id_column = external_cfg.get("id_column", self.parcel_id_column)
        value_column = external_cfg.get("value_column", self.output_column)
        fmt = external_cfg.get("format")

        print(
            f"[GeoFeatureEngineer] Loading external NDVI from {path_obj} "
            f"(format={fmt or 'auto'}, id_column={id_column}, value_column={value_column})"
        )

        if fmt is None:
            if path_obj.suffix.lower() in {".csv"}:
                fmt = "csv"
            elif path_obj.suffix.lower() in {".gpkg", ".sqlite"}:
                fmt = "gpkg"
            else:
                fmt = "csv"

        if fmt == "csv":
            frame = pd.read_csv(path_obj)
        elif fmt == "gpkg":
            try:
                import geopandas as gpd
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "geopandas is required to read GeoPackage NDVI sources."
                ) from exc
            layer = external_cfg.get("layer")
            frame = gpd.read_file(path_obj, layer=layer)
        else:
            raise ValueError(f"Unsupported external NDVI format '{fmt}'.")

        if id_column not in frame.columns:
            raise KeyError(
                f"External NDVI source is missing ID column '{id_column}'. "
                f"Available columns: {list(frame.columns)}"
            )
        if value_column not in frame.columns:
            raise KeyError(
                f"External NDVI source is missing value column '{value_column}'. "
                f"Available columns: {list(frame.columns)}"
            )

        frame = frame[[id_column, value_column]].copy()
        frame[id_column] = self._normalize_ids(frame[id_column])
        frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
        frame = frame.dropna(subset=[id_column])

        print(
            f"[GeoFeatureEngineer] External NDVI loaded: "
            f"{frame[value_column].notna().sum()} non-null values."
        )

        return frame.rename(
            columns={id_column: self.parcel_id_column, value_column: self.output_column}
        )

    def _ndvi_from_single_raster(
        self, parcels, raster_path: Path, *, band: int
    ) -> pd.DataFrame:
        """
        Sample a pre-computed NDVI raster for every parcel.

        Parameters
        ----------
        parcels : geopandas.GeoDataFrame
            Parcel geometries in their native CRS.
        raster_path : pathlib.Path
            Location of the NDVI raster.
        band : int
            Band index (1-based) that contains NDVI values.

        Returns
        -------
        pandas.DataFrame
            NDVI values keyed by parcel identifier.
        """
        import geopandas as gpd  # noqa: F401  (import for type checkers)
        import rasterio
        from rasterio.mask import mask
        from shapely.geometry import mapping

        print(
            f"[GeoFeatureEngineer] Using NDVI raster at {raster_path} (band={band})"
        )
        if not raster_path.exists():
            raise FileNotFoundError(f"NDVI raster not found at {raster_path!s}.")

        with rasterio.open(raster_path) as src:
            print(
                f"[GeoFeatureEngineer] Raster CRS={src.crs}, bounds={src.bounds}, "
                f"transform={src.transform}, nodata={src.nodata}"
            )
            target = self._project_parcels(parcels, src.crs)
            nodata = (
                self.nodata_override if self.nodata_override is not None else src.nodata
            )

            values: list[float] = []
            for geom in target.geometry:
                if geom is None or geom.is_empty:
                    values.append(np.nan)
                    continue

                values.append(
                    self._sample_geometry_mean(src, geom, band=band, nodata=nodata)
                )

        return pd.DataFrame(
            {
                self.parcel_id_column: parcels[self.parcel_id_column],
                self.output_column: values,
            }
        )

    def _ndvi_from_band_pair(
        self,
        parcels,
        red_raster: Path,
        nir_raster: Path,
        *,
        red_band: int,
        nir_band: int,
    ) -> pd.DataFrame:
        """
        Compute NDVI from red and near-infrared rasters.

        Parameters
        ----------
        parcels : geopandas.GeoDataFrame
            Parcel geometries in their native CRS.
        red_raster : pathlib.Path
            GeoTIFF containing the red band.
        nir_raster : pathlib.Path
            GeoTIFF containing the near-infrared band.
        red_band : int
            1-based index of the red band.
        nir_band : int
            1-based index of the NIR band.

        Returns
        -------
        pandas.DataFrame
            NDVI values keyed by parcel identifier.
        """
        import rasterio
        from rasterio.mask import mask
        from shapely.geometry import mapping

        if not red_raster.exists():
            raise FileNotFoundError(f"Red band raster not found at {red_raster!s}.")
        if not nir_raster.exists():
            raise FileNotFoundError(f"NIR band raster not found at {nir_raster!s}.")

        with rasterio.open(red_raster) as red_src, rasterio.open(nir_raster) as nir_src:
            if red_src.crs != nir_src.crs:
                raise ValueError("Red and NIR rasters must share the same CRS.")
            if red_src.transform != nir_src.transform:
                raise ValueError(
                    "Red and NIR rasters must share the same geotransform/footprint."
                )

            target = self._project_parcels(parcels, red_src.crs)

            red_nodata = (
                self.nodata_override if self.nodata_override is not None else red_src.nodata
            )
            nir_nodata = (
                self.nodata_override if self.nodata_override is not None else nir_src.nodata
            )

            values: list[float] = []
            for geom in target.geometry:
                if geom is None or geom.is_empty:
                    values.append(np.nan)
                    continue

                ndvi_value = self._sample_ndvi_from_pair(
                    red_src,
                    nir_src,
                    geom,
                    red_band=red_band,
                    nir_band=nir_band,
                    red_nodata=red_nodata,
                    nir_nodata=nir_nodata,
                )
                values.append(ndvi_value)

        return pd.DataFrame(
            {
                self.parcel_id_column: parcels[self.parcel_id_column],
                self.output_column: values,
            }
        )

    def _sample_geometry_mean(self, src, geom, *, band: int, nodata) -> float:
        """
        Sample raster values under a geometry, falling back to centroid sampling.
        """
        from rasterio.mask import mask
        from shapely.geometry import mapping

        try:
            data, _ = mask(
                src,
                [mapping(geom)],
                crop=True,
                indexes=band,
                filled=True,
                nodata=nodata,
            )
        except ValueError:
            data = None

        if data is not None:
            arr = data[0].astype("float32")
            valid = np.ones(arr.shape, dtype=bool)
            if nodata is not None and np.isfinite(nodata):
                valid &= arr != nodata
            if np.issubdtype(arr.dtype, np.floating):
                valid &= ~np.isnan(arr)
            if valid.any():
                self._debug_stats["mask_success"] = (
                    self._debug_stats.get("mask_success", 0) + 1
                )
                return float(arr[valid].mean())

        # Fall back to representative point sampling when polygon extraction fails.
        self._debug_stats["mask_fallback"] = (
            self._debug_stats.get("mask_fallback", 0) + 1
        )
        point = geom.representative_point()
        sample = next(src.sample([(point.x, point.y)], indexes=band), [np.nan])
        value = sample[0]
        if nodata is not None and np.isfinite(nodata) and value == nodata:
            self._debug_stats["point_nan"] = self._debug_stats.get("point_nan", 0) + 1
            return np.nan
        if not np.isfinite(value):
            self._debug_stats["point_nan"] = self._debug_stats.get("point_nan", 0) + 1
            return np.nan
        self._debug_stats["point_success"] = (
            self._debug_stats.get("point_success", 0) + 1
        )
        return float(value)

    def _sample_ndvi_from_pair(
        self,
        red_src,
        nir_src,
        geom,
        *,
        red_band: int,
        nir_band: int,
        red_nodata,
        nir_nodata,
    ) -> float:
        """
        Compute NDVI for a geometry using separate red and NIR rasters.
        """
        from rasterio.mask import mask
        from shapely.geometry import mapping

        try:
            red_data, _ = mask(
                red_src,
                [mapping(geom)],
                crop=True,
                indexes=red_band,
                filled=True,
                nodata=red_nodata,
            )
            nir_data, _ = mask(
                nir_src,
                [mapping(geom)],
                crop=True,
                indexes=nir_band,
                filled=True,
                nodata=nir_nodata,
            )
        except ValueError:
            red_data = nir_data = None

        if red_data is not None and nir_data is not None:
            red_arr = red_data[0].astype("float32")
            nir_arr = nir_data[0].astype("float32")
            valid = np.ones(red_arr.shape, dtype=bool)
            if red_nodata is not None and np.isfinite(red_nodata):
                valid &= red_arr != red_nodata
            if nir_nodata is not None and np.isfinite(nir_nodata):
                valid &= nir_arr != nir_nodata
            denom = nir_arr + red_arr
            valid &= denom != 0
            if valid.any():
                ndvi = (nir_arr - red_arr) / denom
                ndvi = ndvi.astype("float32")
                ndvi = ndvi[valid]
                if ndvi.size:
                    self._debug_stats["mask_success"] = (
                        self._debug_stats.get("mask_success", 0) + 1
                    )
                    return float(np.nanmean(ndvi))

        # Fallback to representative point sampling.
        self._debug_stats["mask_fallback"] = (
            self._debug_stats.get("mask_fallback", 0) + 1
        )
        point = geom.representative_point()
        red_sample = next(
            red_src.sample([(point.x, point.y)], indexes=red_band), [np.nan]
        )[0]
        nir_sample = next(
            nir_src.sample([(point.x, point.y)], indexes=nir_band), [np.nan]
        )[0]

        if (
            (red_nodata is not None and np.isfinite(red_nodata) and red_sample == red_nodata)
            or (nir_nodata is not None and np.isfinite(nir_nodata) and nir_sample == nir_nodata)
        ):
            self._debug_stats["point_nan"] = self._debug_stats.get("point_nan", 0) + 1
            return np.nan
        if not np.isfinite(red_sample) or not np.isfinite(nir_sample):
            self._debug_stats["point_nan"] = self._debug_stats.get("point_nan", 0) + 1
            return np.nan

        denom = nir_sample + red_sample
        if denom == 0:
            self._debug_stats["point_nan"] = self._debug_stats.get("point_nan", 0) + 1
            return np.nan
        self._debug_stats["point_success"] = (
            self._debug_stats.get("point_success", 0) + 1
        )
        return float((nir_sample - red_sample) / denom)

    def _normalize_ids(self, series: Iterable) -> pd.Series:
        """
        Clean parcel identifiers and optionally left-pad them.

        Parameters
        ----------
        series : Iterable
            Series-like collection of identifiers.

        Returns
        -------
        pandas.Series
            Normalized IDs as strings (with leading zeros preserved when
            ``parcel_id_padding`` is configured).
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        def _clean(value):
            if pd.isna(value):
                return None
            if isinstance(value, (int, np.integer)):
                text = f"{value:d}"
            elif isinstance(value, (float, np.floating)):
                if not np.isfinite(value):
                    return None
                rounded = int(round(float(value)))
                if np.isclose(value, rounded, atol=1e-6):
                    text = f"{rounded:d}"
                else:
                    text = str(value).strip()
            else:
                text = str(value).strip()

            if not text:
                return None
            if self.parcel_id_padding:
                text = text.zfill(self.parcel_id_padding)
            return text

        return series.apply(_clean)

    def _project_parcels(self, parcels, target_crs):
        """
        Reproject parcel geometries to match the raster CRS.

        Parameters
        ----------
        parcels : geopandas.GeoDataFrame
            Parcel geometries.
        target_crs : rasterio.crs.CRS or dict
            Raster coordinate reference system.

        Returns
        -------
        geopandas.GeoDataFrame
            Parcels reprojected to the raster CRS.
        """
        if target_crs is None:
            return parcels
        if getattr(parcels, "crs", None) is None:
            return parcels
        if parcels.crs == target_crs:
            return parcels
        return parcels.to_crs(target_crs)
