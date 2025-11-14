#!/usr/bin/env python3
"""Clean up old forecast data to maintain rolling storage."""

import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import xarray as xr


def cleanup_old_forecasts(max_age_hours: int = 24) -> None:
    """Remove forecast data older than the specified age.

    Args:
        max_age_hours: Maximum age of forecasts to keep in hours
    """
    data_dir = Path("data")
    cutoff_time = pd.Timestamp.now(tz="UTC") - timedelta(hours=max_age_hours)

    print(f"Cleaning up forecasts older than {cutoff_time}")

    # Process each Zarr dataset
    for zarr_path in data_dir.glob("*.zarr"):
        try:
            # Open the dataset
            ds = xr.open_zarr(zarr_path, consolidated=True)

            # Get the append dimension (usually 'init_time')
            append_dim = "init_time"
            if append_dim not in ds.dims:
                print(f"Warning: {zarr_path} does not have '{append_dim}' dimension")
                ds.close()
                continue

            # Find indices to keep
            init_times = pd.DatetimeIndex(ds[append_dim].values)
            keep_mask = init_times >= cutoff_time

            if keep_mask.sum() == 0:
                print(f"Warning: All data in {zarr_path} is older than cutoff")
                ds.close()
                continue

            if keep_mask.all():
                print(f"No cleanup needed for {zarr_path}")
                ds.close()
                continue

            # Select only recent data
            ds_recent = ds.isel({append_dim: keep_mask})

            # Create a temporary path
            temp_path = zarr_path.parent / f"{zarr_path.name}.tmp"

            # Save the filtered dataset
            ds_recent.to_zarr(temp_path, mode="w", consolidated=True)

            ds.close()
            ds_recent.close()

            # Replace the old dataset
            shutil.rmtree(zarr_path)
            temp_path.rename(zarr_path)

            removed_count = (~keep_mask).sum()
            print(f"Removed {removed_count} old forecast(s) from {zarr_path.name}")

        except Exception as e:
            print(f"Error processing {zarr_path}: {e}")
            continue


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Clean up old forecast data")
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=24,
        help="Maximum age of forecasts to keep in hours (default: 24)",
    )

    args = parser.parse_args()
    cleanup_old_forecasts(args.max_age_hours)


if __name__ == "__main__":
    main()
