# NBM to Zarr

Automated conversion of NOAA's National Blend of Models (NBM) weather forecast data into Zarr format for cloud-native access.

## Overview

This repository automates the download and conversion of NBM CONUS forecast data from GRIB2 format to Zarr, enabling efficient cloud-native access to weather forecast data. The system runs on a rolling hourly basis, maintaining recent forecast data.

## Features

- **Hourly Updates**: Automatically fetches and processes new NBM forecast data every hour
- **Cloud-Optimized**: Converts GRIB2 data to Zarr format with optimized chunking and compression
- **Rolling Storage**: Maintains a 24-hour rolling window of forecast data
- **Comprehensive Variables**: Includes 20+ meteorological variables (temperature, wind, precipitation, etc.)

## NBM Data Details

- **Model**: National Blend of Models (NBM) CONUS
- **Resolution**: 2.5 km
- **Update Frequency**: Hourly
- **Forecast Length**: 36 hours (hourly forecasts)
- **Domain**: CONUS (Continental United States)

## Installation

```bash
# Clone the repository
git clone https://github.com/andrewnakas/Nbm_to_zarr.git
cd Nbm_to_zarr

# Install using pip
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

```bash
# List available datasets
nbm-zarr list-datasets

# Show dataset information
nbm-zarr info noaa-nbm-conus-forecast

# Generate template configuration
nbm-zarr update-template --output-dir ./data

# Run operational update
nbm-zarr operational-update --dataset-id noaa-nbm-conus-forecast --output-dir ./data
```

### Automated Updates

The repository includes a GitHub Actions workflow that runs hourly to:
1. Fetch the latest NBM forecast data
2. Convert to Zarr format
3. Maintain a 24-hour rolling window
4. Generate data catalog for easy access

## Data Structure

The Zarr datasets include the following dimensions:
- `init_time`: Forecast initialization time
- `lead_time`: Forecast lead time (0-36 hours)
- `y`: North-south grid dimension (1597 points)
- `x`: East-west grid dimension (2345 points)

### Available Variables

The datasets include 20+ meteorological variables:
- **Temperature**: 2m temperature, dewpoint, max/min temperature
- **Wind**: 10m and 80m wind components, wind gusts
- **Precipitation**: Total precipitation, precipitation rate, snow accumulation
- **Cloud**: Total cloud cover, ceiling height
- **Radiation**: Downward shortwave/longwave radiation
- **Visibility**: Surface visibility
- **Pressure**: Surface pressure

## Architecture

The project follows a modular architecture:

- `src/nbm_to_zarr/base/`: Base classes for template configuration and data processing
- `src/nbm_to_zarr/noaa/nbm_conus/forecast/`: NBM CONUS-specific implementations
- `.github/workflows/`: Automated update workflows

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Data provided by NOAA National Centers for Environmental Prediction
- Based on the architecture patterns from dynamical.org
