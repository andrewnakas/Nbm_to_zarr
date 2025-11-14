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

        # Process each source file
        for source_coord in source_coords:
            try:
                # Download file
                file_path = self.download_file(source_coord)

                # Read data
                result = self.read_data(file_path, source_coord)

                # Handle backward compatibility
                if isinstance(result, tuple):
                    data_dict, metadata = result
                else:
                    data_dict = result
                    metadata = {}

                # Apply transformations and populate dataset
                for var_config in self.data_vars:
                    if var_config.name in data_dict:
                        transformed_data = self.apply_transformations(
                            {var_config.name: data_dict[var_config.name]}, var_config
                        )
                        # Populate the dataset with the data
                        # This is a simplified version - actual implementation would need
                        # proper indexing based on init_time and lead_time
                        pass

            except Exception as e:
                print(f"Error processing {source_coord}: {e}")
                continue

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
