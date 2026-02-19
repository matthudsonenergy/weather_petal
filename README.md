# Historical Weather Data Project (Open-Meteo)

This repository provides a ready-to-run Python workflow for:

- Geocoding a location name to coordinates with Open-Meteo Geocoding API
- Fetching historical weather data from Open-Meteo Archive API
- Transforming API response into Pandas DataFrames
- Exporting CSV outputs
- Plotting selected weather metrics with Matplotlib

## Project Layout

```text
weather_petal/
├── weather_historical.py
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- Internet access (for API requests)

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the script with:

```bash
python weather_historical.py \
  --location "Seattle, US" \
  --start-date 2024-01-01 \
  --end-date 2024-01-15 \
  --metrics "temperature_2m,precipitation,wind_speed_10m,temperature_2m_max,precipitation_sum" \
  --timezone "America/Los_Angeles" \
  --output-dir output
```

### Required Arguments

- `--location`: Location string (city, city-country, etc.)
- `--start-date`: Start date (`YYYY-MM-DD`)
- `--end-date`: End date (`YYYY-MM-DD`)
- `--metrics`: Comma-separated metric names

### Optional Arguments

- `--timezone`: IANA timezone or `auto` (default: `auto`)
- `--output-dir`: Output folder path (default: `output`)
- `--timeout`: HTTP timeout in seconds (default: `30`)

## Supported Metrics

### Hourly

- `temperature_2m`
- `relative_humidity_2m`
- `apparent_temperature`
- `precipitation`
- `rain`
- `snowfall`
- `snow_depth`
- `cloud_cover`
- `surface_pressure`
- `visibility`
- `wind_speed_10m`
- `wind_direction_10m`

### Daily

- `temperature_2m_max`
- `temperature_2m_min`
- `temperature_2m_mean`
- `apparent_temperature_max`
- `apparent_temperature_min`
- `precipitation_sum`
- `rain_sum`
- `snowfall_sum`
- `wind_speed_10m_max`

## Output

The script creates the output directory (if missing) and writes:

- `hourly_weather.csv` when hourly metrics are requested
- `daily_weather.csv` when daily metrics are requested
- `hourly_weather.png` when hourly metrics are requested
- `daily_weather.png` when daily metrics are requested

## Error Handling

The script includes explicit error handling for:

- Invalid date formats
- Invalid date ranges
- Unsupported metric names
- Failed geocoding lookups
- API request errors and timeouts
- Missing expected fields in API responses

On failure, an error message is printed to stderr and the script exits with code `2`.

## Local Execution Notes

- Open-Meteo is free and does not require an API key for this use case.
- If requests fail, verify network connectivity and rerun with a larger timeout, e.g. `--timeout 60`.
- Use a narrow date range first to validate your metric selection and output format.

## Example: Quick Test

```bash
python weather_historical.py \
  --location "New York, US" \
  --start-date 2024-02-01 \
  --end-date 2024-02-07 \
  --metrics "temperature_2m,relative_humidity_2m,wind_speed_10m"
```
