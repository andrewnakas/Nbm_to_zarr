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

    # Variable mapping from standard names to actual NBM GRIB2 element names
    # Based on inspection of NBM GRIB2 files
    VARIABLE_MAPPING = {
        "t2m": {"grib_element": "T", "short_name": "2-HTGL"},
        "dpt2m": {"grib_element": "Td", "short_name": "2-HTGL"},
        "tmax": {"grib_element": "T", "short_name": "2-HTGL"},  # May need different logic
        "tmin": {"grib_element": "T", "short_name": "2-HTGL"},  # May need different logic
        "u10m": {"grib_element": "WindSpd", "short_name": "10-HTGL", "wind_component": "u"},
        "v10m": {"grib_element": "WindSpd", "short_name": "10-HTGL", "wind_component": "v"},
        "u80m": {"grib_element": "WindSpd", "short_name": "80-HTGL", "wind_component": "u"},
        "v80m": {"grib_element": "WindSpd", "short_name": "80-HTGL", "wind_component": "v"},
        "gust": {"grib_element": "WindGust", "short_name": "10-HTGL"},
        "tp": {"grib_element": "QPF01", "short_name": "0-SFC"},
        "prate": {"grib_element": "QPF01", "short_name": "0-SFC"},  # Same as tp, just different name
        "snow": {"grib_element": "SnowAmt01", "short_name": "0-SFC"},
        "tcc": {"grib_element": "TCDC", "short_name": "0-RESERVED"},
        "ceil": {"grib_element": "CEIL", "short_name": "0-RESERVED"},
        "vis": {"grib_element": "VIS", "short_name": "0-SFC"},
        "dswrf": {"grib_element": "DSWRF", "short_name": "0-SFC"},
        "dlwrf": {"grib_element": "DSWRF", "short_name": "0-SFC"},  # NBM may not have DLWRF
        "sp": {"grib_element": "PRES", "short_name": "0-SFC"},  # May not exist
        "rh2m": {"grib_element": "RH", "short_name": "2-HTGL"},
    }

    def generate_source_file_coords(self) -> list[NbmConusSourceFileCoord]:
        """Generate source file coordinates for the processing region.

        NBM is updated hourly with forecasts extending out to 36 hours.
        Note: f000 (analysis) files often don't exist, so we start from f001.
        """
        coords = []

        # Generate init times at hourly intervals
        current_time = self.processing_region.init_time_start
        while current_time <= self.processing_region.init_time_end:
            # For each init time, generate forecast hours 1-36
            # (skip f000 as it often doesn't exist)
            for forecast_hour in range(1, 37):  # 1-36 inclusive
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

        # Storage for wind data (need both speed and direction for U/V calculation)
        wind_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # level -> (speed, direction)

        try:
            with rasterio.open(file_path) as src:
                # Read metadata for all bands
                tags_list = [src.tags(i) for i in range(1, src.count + 1)]

                # First pass: collect wind speed and direction data
                for band_idx, tags in enumerate(tags_list, start=1):
                    grib_element = tags.get("GRIB_ELEMENT", "")
                    short_name = tags.get("GRIB_SHORT_NAME", "")

                    if grib_element == "WindSpd":
                        data = src.read(band_idx)
                        nodata = src.nodata
                        if nodata is not None:
                            data = np.where(data == nodata, np.nan, data)

                        if short_name not in wind_data:
                            wind_data[short_name] = [None, None]
                        wind_data[short_name][0] = data  # speed

                    elif grib_element == "WindDir":
                        data = src.read(band_idx)
                        nodata = src.nodata
                        if nodata is not None:
                            data = np.where(data == nodata, np.nan, data)

                        if short_name not in wind_data:
                            wind_data[short_name] = [None, None]
                        wind_data[short_name][1] = data  # direction

                # Process each requested variable
                for var_config in self.data_vars:
                    if var_config.name not in self.VARIABLE_MAPPING:
                        continue

                    var_info = self.VARIABLE_MAPPING[var_config.name]
                    grib_element = var_info["grib_element"]
                    short_name = var_info.get("short_name", "")

                    # Handle wind components specially
                    if "wind_component" in var_info:
                        if short_name in wind_data:
                            speed, direction = wind_data[short_name]
                            if speed is not None and direction is not None:
                                # Convert wind direction (from) to radians
                                # Direction is "from", so add 180 to get "to" direction
                                dir_rad = np.deg2rad(direction + 180)

                                if var_info["wind_component"] == "u":
                                    # U component (east-west)
                                    data_dict[var_config.name] = speed * np.sin(dir_rad)
                                else:
                                    # V component (north-south)
                                    data_dict[var_config.name] = speed * np.cos(dir_rad)
                            else:
                                print(f"  Warning: Missing wind speed or direction for {var_config.name}")
                        else:
                            print(f"  Warning: Wind data not found for level {short_name}")
                        continue

                    # For non-wind variables, find matching band
                    found = False
                    for band_idx, tags in enumerate(tags_list, start=1):
                        elem = tags.get("GRIB_ELEMENT", "")
                        sname = tags.get("GRIB_SHORT_NAME", "")

                        # Match both element and short_name if specified
                        if elem == grib_element:
                            if not short_name or short_name in sname:
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
                        print(f"  Warning: Variable {var_config.name} ({grib_element}) not found in GRIB file")

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
        init_time = now.floor("h") - timedelta(hours=2)

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
