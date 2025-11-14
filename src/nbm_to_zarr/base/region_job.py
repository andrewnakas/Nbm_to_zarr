"""Base classes for regional data processing jobs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
import xarray as xr

from nbm_to_zarr.base.template_config import DataVariableConfig, TemplateConfig


class SourceFileCoord(ABC):
    """Abstract base class representing a source file coordinate."""

    @abstractmethod
    def download_url(self) -> str:
        """Return the URL to download the file."""
        ...

    @abstractmethod
    def index_url(self) -> str:
        """Return the URL to the index file."""
        ...


@dataclass
class ProcessingRegion:
    """Defines a temporal processing region."""

    init_time_start: pd.Timestamp
    init_time_end: pd.Timestamp


SourceFileCoordT = TypeVar("SourceFileCoordT", bound=SourceFileCoord)
DataVarT = TypeVar("DataVarT", bound=DataVariableConfig)


class RegionJob(ABC, Generic[SourceFileCoordT, DataVarT]):
    """Base class for processing a temporal region of data."""

    def __init__(
        self,
        template_config: TemplateConfig[DataVarT],
        processing_region: ProcessingRegion,
        data_vars: list[DataVarT],
        output_path: Path,
        download_dir: Path | None = None,
    ) -> None:
        """Initialize the region job.

        Args:
            template_config: Template configuration defining dataset structure
            processing_region: Temporal region to process
            data_vars: List of data variables to process
            output_path: Path to output Zarr store
            download_dir: Optional directory for downloaded files
        """
        self.template_config = template_config
        self.processing_region = processing_region
        self.data_vars = data_vars
        self.output_path = output_path
        self.download_dir = download_dir or Path("/tmp/nbm_downloads")
        self.download_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def generate_source_file_coords(self) -> list[SourceFileCoordT]:
        """Generate source file coordinates for the processing region."""
        ...

    @abstractmethod
    def download_file(self, source_coord: SourceFileCoordT) -> Path:
        """Download a file and return the local path."""
        ...

    @abstractmethod
    def read_data(
        self, file_path: Path, source_coord: SourceFileCoordT
    ) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Read data from the file and return arrays for each variable."""
        ...

    def apply_transformations(
        self, data: dict[str, np.ndarray], var_config: DataVarT
    ) -> dict[str, np.ndarray]:
        """Apply transformations like bit rounding to data arrays."""
        if var_config.keepbits is not None:
            for var_name in data:
                data[var_name] = self._round_to_n_bits(data[var_name], var_config.keepbits)
        return data

    @staticmethod
    def _round_to_n_bits(data: np.ndarray, n_bits: int) -> np.ndarray:
        """Round data to n significant bits for better compression."""
        # Implementation of bit rounding algorithm
        if not np.issubdtype(data.dtype, np.floating):
            return data

        # Handle NaN and inf values
        mask = np.isfinite(data)
        if not np.any(mask):
            return data

        result = data.copy()
        finite_data = result[mask]

        # Get mantissa precision
        mantissa = np.frexp(finite_data)[0]
        precision = 2.0 ** (n_bits - 24)  # Assuming float32

        # Round to precision
        rounded = np.around(mantissa / precision) * precision
        result[mask] = np.ldexp(rounded, np.frexp(finite_data)[1])

        return result

    def get_indices(self, source_coord: SourceFileCoordT) -> dict[str, int]:
        """Get dataset indices for a source coordinate.

        Subclasses should override this if they have specific indexing logic.
        Default implementation assumes source_coord has init_time and forecast_hour attributes.
        """
        # This is a generic implementation that assumes certain attributes
        # Subclasses can override for custom behavior
        if hasattr(source_coord, 'init_time') and hasattr(source_coord, 'forecast_hour'):
            return {
                'init_time': source_coord.init_time,
                'forecast_hour': source_coord.forecast_hour,
            }
        raise NotImplementedError("Subclass must override get_indices() method")

    def process(self) -> xr.Dataset:
        """Process the region and return the populated dataset."""
        # Generate source file coordinates
        source_coords = self.generate_source_file_coords()

        # Create dimension coordinates
        init_times = pd.date_range(
            start=self.processing_region.init_time_start,
            end=self.processing_region.init_time_end,
            freq="1H",
        )

        # Build empty dataset
        ds = self.template_config.get_template(
            append_dim_start=init_times[0],
            append_dim_periods=len(init_times),
            append_dim_freq="1H",
        )

        # Create lead_time coordinate values
        lead_times = pd.to_timedelta(np.arange(self.template_config.dimensions['lead_time']), unit='h')
        ds = ds.assign_coords(lead_time=lead_times)

        print(f"Processing {len(source_coords)} source files...")

        # Process each source file
        processed_count = 0
        for source_coord in source_coords:
            try:
                # Download file
                print(f"Downloading: {source_coord.download_url()}")
                file_path = self.download_file(source_coord)

                # Read data
                result = self.read_data(file_path, source_coord)

                # Handle backward compatibility
                if isinstance(result, tuple):
                    data_dict, metadata = result
                else:
                    data_dict = result
                    metadata = {}

                # Get indices for this source coordinate
                indices = self.get_indices(source_coord)
                init_time = indices['init_time']
                forecast_hour = indices['forecast_hour']

                # Find the init_time index
                init_idx = np.where(ds.init_time.values == init_time)[0]
                if len(init_idx) == 0:
                    print(f"Warning: init_time {init_time} not found in dataset")
                    continue
                init_idx = init_idx[0]

                # Apply transformations and populate dataset
                for var_config in self.data_vars:
                    if var_config.name in data_dict:
                        transformed_data = self.apply_transformations(
                            {var_config.name: data_dict[var_config.name]}, var_config
                        )

                        # Populate the dataset with the data at the correct indices
                        data_array = transformed_data[var_config.name]
                        ds[var_config.name].values[init_idx, forecast_hour, :, :] = data_array

                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count}/{len(source_coords)} files")

            except Exception as e:
                print(f"Error processing {source_coord}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"Successfully processed {processed_count}/{len(source_coords)} files")
        return ds

    @classmethod
    @abstractmethod
    def operational_update_jobs(
        cls,
        template_config: TemplateConfig[DataVarT],
        data_vars: list[DataVarT],
        output_path: Path,
    ) -> list[RegionJob[SourceFileCoordT, DataVarT]]:
        """Create jobs for operational updates."""
        ...
