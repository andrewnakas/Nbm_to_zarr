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

        print(f"\n{'='*60}")
        print(f"Starting operational update for {self.dataset_id}")
        print(f"Output path: {output_path}")
        print(f"{'='*60}\n")

        # Get jobs for operational update
        try:
            jobs = self.region_job_class.operational_update_jobs(
                template_config=self.template_config,
                data_vars=self.template_config.data_vars,
                output_path=output_path,
            )
            print(f"Created {len(jobs)} job(s) to process\n")
        except Exception as e:
            print(f"ERROR: Failed to create jobs: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Process each job
        for i, job in enumerate(jobs, 1):
            print(f"\n{'='*60}")
            print(f"Processing job {i}/{len(jobs)}")
            print(f"{'='*60}\n")

            try:
                ds = job.process()

                if ds is None:
                    print("ERROR: Job returned None dataset")
                    continue

                print(f"\n{'='*60}")
                print(f"Saving dataset to Zarr...")
                print(f"{'='*60}\n")

                self._save_to_zarr(ds, output_path)

                print(f"\n✅ Successfully saved to {output_path}")

            except Exception as e:
                print(f"\nERROR: Failed to process job {i}: {e}")
                import traceback
                traceback.print_exc()
                raise

    def _save_to_zarr(self, ds: xr.Dataset, output_path: Path) -> None:
        """Save dataset to Zarr format."""
        import zarr

        print(f"Dataset info:")
        print(f"  Dimensions: {dict(ds.dims)}")
        print(f"  Variables: {list(ds.data_vars.keys())}")
        print(f"  Coordinates: {list(ds.coords.keys())}")

        # Remove problematic attributes
        for var in ds.coords:
            if "units" in ds[var].attrs and var in ["init_time", "valid_time"]:
                del ds[var].attrs["units"]

        # Build encoding with proper compressor
        from numcodecs import Zstd

        encoding = {}
        for var in list(ds.data_vars) + list(ds.coords):
            var_encoding = {
                "compressor": Zstd(level=3),
            }

            # Get chunks from the variable if available
            if hasattr(ds[var], 'chunks') and ds[var].chunks is not None:
                # Convert dask chunks to dict
                chunks_dict = dict(zip(ds[var].dims, [c[0] if isinstance(c, tuple) else c for c in ds[var].chunks]))
                var_encoding["chunks"] = tuple(chunks_dict.get(dim, ds.dims[dim]) for dim in ds[var].dims)

            encoding[var] = var_encoding

        # Determine write mode
        mode = "a" if output_path.exists() else "w"

        print(f"Write mode: {mode}")
        print(f"Output path: {output_path}")

        try:
            # Write to Zarr
            ds.to_zarr(
                output_path,
                mode=mode,
                encoding=encoding,
                consolidated=True,
                compute=True,
            )
            print(f"✅ Successfully wrote {output_path}")

        except Exception as e:
            print(f"ERROR during Zarr write: {e}")
            import traceback
            traceback.print_exc()
            raise
