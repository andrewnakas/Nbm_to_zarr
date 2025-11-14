"""Region job processor for NBM CONUS forecast data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests

from nbm_to_zarr.base.region_job import ProcessingRegion, RegionJob, SourceFileCoord
from nbm_to_zarr.base.template_config import DataVariableConfig, TemplateConfig


@dataclass
class NbmConusSourceFileCoord(SourceFileCoord):
    """Coordinate representing a single NBM CONUS forecast file."""

    init_time: pd.Timestamp
    forecast_hour: int
    region: str = "co"  # CONUS region code

    def download_url(self) -> str:
        """Return the NOMADS download URL for this file.

        Format: https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/
                blend.YYYYMMDD/HH/core/blend.tHHz.core.fXXX.co.grib2
        """
        date_str = self.init_time.strftime("%Y%m%d")
        cycle_str = self.init_time.strftime("%H")
        forecast_str = f"{self.forecast_hour:03d}"

        return (
            f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/"
            f"blend.{date_str}/{cycle_str}/core/"
            f"blend.t{cycle_str}z.core.f{forecast_str}.{self.region}.grib2"
        )

    def index_url(self) -> str:
        """Return the index file URL."""
        return f"{self.download_url()}.idx"


class NbmConusForecastRegionJob(RegionJob[NbmConusSourceFileCoord, DataVariableConfig]):
    """Process NBM CONUS forecast data for a temporal region."""

    # Variable mapping from standard names to GRIB2 parameter names
    # These will need to be adjusted based on actual NBM GRIB2 structure
    VARIABLE_MAPPING = {
        "t2m": {"name": "TMP", "level": "2 m above ground"},
        "dpt2m": {"name": "DPT", "level": "2 m above ground"},
        "tmax": {"name": "TMAX", "level": "2 m above ground"},
        "tmin": {"name": "TMIN", "level": "2 m above ground"},
        "u10m": {"name": "UGRD", "level": "10 m above ground"},
        "v10m": {"name": "VGRD", "level": "10 m above ground"},
        "u80m": {"name": "UGRD", "level": "80 m above ground"},
        "v80m": {"name": "VGRD", "level": "80 m above ground"},
        "gust": {"name": "GUST", "level": "surface"},
        "tp": {"name": "APCP", "level": "surface"},
        "prate": {"name": "PRATE", "level": "surface"},
        "snow": {"name": "ASNOW", "level": "surface"},
        "tcc": {"name": "TCDC", "level": "entire atmosphere"},
        "ceil": {"name": "HGT", "level": "cloud ceiling"},
        "vis": {"name": "VIS", "level": "surface"},
        "dswrf": {"name": "DSWRF", "level": "surface"},
        "dlwrf": {"name": "DLWRF", "level": "surface"},
        "sp": {"name": "PRES", "level": "surface"},
        "rh2m": {"name": "RH", "level": "2 m above ground"},
    }

    def generate_source_file_coords(self) -> list[NbmConusSourceFileCoord]:
        """Generate source file coordinates for the processing region.

        NBM is updated hourly with forecasts extending out to 36 hours.
        """
        coords = []

        # Generate init times at hourly intervals
        current_time = self.processing_region.init_time_start
        while current_time <= self.processing_region.init_time_end:
            # For each init time, generate forecast hours 0-36
            for forecast_hour in range(37):  # 0-36 inclusive
                coords.append(
                    NbmConusSourceFileCoord(
                        init_time=current_time,
                        forecast_hour=forecast_hour,
                        region="co",
                    )
                )

            current_time += timedelta(hours=1)

        return coords

    def download_file(self, source_coord: NbmConusSourceFileCoord) -> Path:
        """Download a GRIB2 file from NOMADS."""
        url = source_coord.download_url()

        # Create filename from source coordinate
        date_str = source_coord.init_time.strftime("%Y%m%d")
        cycle_str = source_coord.init_time.strftime("%H")
        forecast_str = f"{source_coord.forecast_hour:03d}"
        filename = f"blend.t{cycle_str}z.core.f{forecast_str}.{source_coord.region}.grib2"

        # Create subdirectory for this date
        download_path = self.download_dir / date_str / cycle_str
        download_path.mkdir(parents=True, exist_ok=True)

        file_path = download_path / filename

        # Download if not already cached
        if not file_path.exists():
            print(f"  Downloading {filename} ({source_coord.forecast_hour}h forecast)...")

            # Retry logic for network issues
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, stream=True, timeout=60)
                    response.raise_for_status()

                    # Write to temporary file first
                    temp_path = file_path.with_suffix('.tmp')
                    with open(temp_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    # Move to final location
                    temp_path.rename(file_path)
                    break

                except (requests.RequestException, IOError) as e:
                    if attempt < max_retries - 1:
                        print(f"  Attempt {attempt + 1} failed: {e}. Retrying...")
                        continue
                    else:
                        raise
        else:
            print(f"  Using cached {filename}")

        return file_path

    def read_data(
        self, file_path: Path, source_coord: NbmConusSourceFileCoord
    ) -> dict[str, np.ndarray]:
        """Read data from GRIB2 file using rasterio.

        Returns a dictionary mapping variable names to numpy arrays.
        """
        data_dict: dict[str, np.ndarray] = {}

        try:
            with rasterio.open(file_path) as src:
                # Read metadata for all bands
                tags_list = [src.tags(i) for i in range(1, src.count + 1)]

                # Process each requested variable
                for var_config in self.data_vars:
                    if var_config.name not in self.VARIABLE_MAPPING:
                        continue

                    var_info = self.VARIABLE_MAPPING[var_config.name]

                    # Find matching band
                    found = False
                    for band_idx, tags in enumerate(tags_list, start=1):
                        # Match by GRIB parameter name
                        # Try different possible tag names
                        grib_name = tags.get("GRIB_ELEMENT", "")
                        if not grib_name:
                            grib_name = tags.get("GRIB_COMMENT", "")
                        if not grib_name:
                            grib_name = tags.get("long_name", "")

                        if var_info["name"] in grib_name:
                            # Read the band data
                            data = src.read(band_idx)

                            # Handle missing values
                            nodata = src.nodata
                            if nodata is not None:
                                data = np.where(data == nodata, np.nan, data)

                            data_dict[var_config.name] = data
                            found = True
                            break

                    if not found:
                        print(f"  Warning: Variable {var_config.name} ({var_info['name']}) not found in GRIB file")

        except Exception as e:
            print(f"  Error reading GRIB file {file_path}: {e}")
            raise

        return data_dict

    @classmethod
    def operational_update_jobs(
        cls,
        template_config: TemplateConfig[DataVariableConfig],
        data_vars: list[DataVariableConfig],
        output_path: Path,
    ) -> list[NbmConusForecastRegionJob]:
        """Create jobs for operational updates.

        For NBM, we process the most recent available forecast cycle.
        """
        # Get current time
        now = pd.Timestamp.now(tz="UTC")

        # NBM data has some latency, so look back a few hours to ensure data availability
        # Round down to the nearest hour
        init_time = now.floor("H") - timedelta(hours=2)

        # Create a single job for the most recent forecast
        processing_region = ProcessingRegion(
            init_time_start=init_time,
            init_time_end=init_time,
        )

        return [
            cls(
                template_config=template_config,
                processing_region=processing_region,
                data_vars=data_vars,
                output_path=output_path,
            )
        ]
