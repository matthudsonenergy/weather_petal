# Historical Weather Data Project (Open-Meteo)

This repository provides a ready-to-run Python workflow for:

- Geocoding a ZIP/postal code to coordinates with Open-Meteo Geocoding API
- Fetching historical weather data from Open-Meteo Archive API
- Transforming API response into Pandas DataFrames
- Exporting CSV outputs
- Plotting selected weather metrics with Matplotlib (temperature in F)

## Project Layout

```text
weather_petal/
├── weather_historical.py
├── google_apps_script/
│   ├── ten_day_hourly_forecast.gs
│   └── README.md
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
  --zip-code "98101" \
  --start-date 2024-01-01 \
  --end-date 2024-01-15 \
  --metrics "temperature_2m,precipitation,wind_speed_10m,temperature_2m_max,precipitation_sum" \
  --timezone "America/Los_Angeles" \
  --output-dir output
```

### Required Arguments

- `--zip-code`: ZIP/postal code string (for example `98101`)
- `--start-date`: Start date (`YYYY-MM-DD`)
- `--end-date`: End date (`YYYY-MM-DD`)
- `--metrics`: Comma-separated metric names

### Optional Arguments

- `--timezone`: IANA timezone or `auto` (default: `auto`)
- `--output-dir`: Output folder path (default: `output`)
- `--timeout`: HTTP timeout in seconds (default: `30`)
- `--llm-summary`: Send CSV-derived weather data to Hugging Face and generate an extreme-weather summary
- `--hf-model`: Hugging Face model id for `--llm-summary` (default: `mistralai/Mistral-7B-Instruct-v0.3`)
- `--hf-token-env`: Environment variable name for your Hugging Face token (default: `HF_TOKEN`)
- `--llm-max-rows`: Maximum CSV rows per frame included in LLM context (default: `240`)

Temperature values are requested in Fahrenheit by default.

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

Each run creates a new subfolder under `--output-dir` to avoid overwriting previous results.

Folder pattern:

`<zip_code_slug>_<start-date>_to_<end-date>_<YYYYMMDD_HHMMSS>`

Example:

`output/10001_2024-02-01_to_2024-02-07_20260219_150530/`

Inside that run folder, the script writes:

- `hourly_weather.csv` when hourly metrics are requested
- `daily_weather.csv` when daily metrics are requested
- `hourly_weather.png` when hourly metrics are requested
- `below_freezing_weather.png` when hourly Fahrenheit metrics include values at or below 32°F (2 subplots: below-freezing temperature line + consecutive-hour streak bars)
- `daily_weather.png` when daily metrics are requested
- `llm_extreme_weather_summary.md` when `--llm-summary` is enabled

## LLM Extreme Weather Summary (Hugging Face)

Set your token in an environment variable (default is `HF_TOKEN`):

```bash
export HF_TOKEN=your_hf_token_here
```

Run with `--llm-summary` to generate `llm_extreme_weather_summary.md` in the run output folder:

```bash
python weather_historical.py \
  --zip-code "98101" \
  --start-date 2024-01-01 \
  --end-date 2024-01-15 \
  --metrics "temperature_2m,precipitation,wind_speed_10m" \
  --llm-summary \
  --hf-model "mistralai/Mistral-7B-Instruct-v0.3"
```

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

## Daily Forecast Email (Google Workspace)

If you want a 10-day hourly forecast emailed daily, use the Google Apps Script version in:

- `google_apps_script/ten_day_hourly_forecast.gs`
- `google_apps_script/README.md`

## Example: Quick Test

```bash
python weather_historical.py \
  --zip-code "10001" \
  --start-date 2024-02-01 \
  --end-date 2024-02-07 \
  --metrics "temperature_2m,relative_humidity_2m,wind_speed_10m"
```
