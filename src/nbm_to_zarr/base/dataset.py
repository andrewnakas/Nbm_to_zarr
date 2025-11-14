"""Base dataset orchestrator class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import xarray as xr

from nbm_to_zarr.base.region_job import RegionJob, SourceFileCoord
from nbm_to_zarr.base.template_config import DataVariableConfig, TemplateConfig

SourceFileCoordT = TypeVar("SourceFileCoordT", bound=SourceFileCoord)
DataVarT = TypeVar("DataVarT", bound=DataVariableConfig)


class Dataset(ABC, Generic[SourceFileCoordT, DataVarT]):
    """Base class for dataset orchestration."""

    @property
    @abstractmethod
    def template_config(self) -> TemplateConfig[DataVarT]:
        """Return the template configuration."""
        ...

    @property
    @abstractmethod
    def region_job_class(self) -> type[RegionJob[SourceFileCoordT, DataVarT]]:
        """Return the region job class."""
        ...

    @property
    def dataset_id(self) -> str:
        """Return the dataset ID."""
        return self.template_config.dataset_attributes.id

    def operational_update(self, output_dir: Path) -> None:
        """Run an operational update of the dataset."""
        output_path = output_dir / f"{self.dataset_id}.zarr"

        # Get jobs for operational update
        jobs = self.region_job_class.operational_update_jobs(
            template_config=self.template_config,
            data_vars=self.template_config.data_vars,
            output_path=output_path,
        )

        # Process each job
        for job in jobs:
            ds = job.process()
            self._save_to_zarr(ds, output_path)

    def _save_to_zarr(self, ds: xr.Dataset, output_path: Path) -> None:
        """Save dataset to Zarr format."""
        # Remove problematic attributes
        for var in ds.coords:
            if "units" in ds[var].attrs and var in ["init_time", "valid_time"]:
                del ds[var].attrs["units"]

        # Clean encoding
        encoding = {}
        for var in list(ds.data_vars) + list(ds.coords):
            var_encoding = {}
            if "compressor" in ds[var].encoding:
                var_encoding["compressor"] = ds[var].encoding["compressor"]
            if "chunks" in ds[var].encoding:
                var_encoding["chunks"] = ds[var].encoding["chunks"]
            encoding[var] = var_encoding

        # Determine write mode
        mode = "a" if output_path.exists() else "w"

        # Write to Zarr
        ds.to_zarr(
            output_path,
            mode=mode,
            encoding=encoding,
            consolidated=True,
            zarr_format=2,
        )
