#!/usr/bin/env python3
"""Fetch, process, and plot historical weather data from Open-Meteo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Frequently used metrics from the Open-Meteo API.
SUPPORTED_HOURLY_METRICS = {
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "cloud_cover",
    "surface_pressure",
    "visibility",
    "wind_speed_10m",
    "wind_direction_10m",
}

SUPPORTED_DAILY_METRICS = {
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
}


class WeatherClientError(Exception):
    """Application-level exception for predictable user-facing failures."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch historical weather data for a location using Open-Meteo, "
            "then process and plot selected metrics."
        )
    )
    parser.add_argument("--location", required=True, help="Location name, e.g. 'Chicago, US'.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument(
        "--metrics",
        required=True,
        help=(
            "Comma-separated metric names. Metrics can be hourly (temperature_2m, wind_speed_10m) "
            "or daily (temperature_2m_max, precipitation_sum)."
        ),
    )
    parser.add_argument(
        "--timezone",
        default="auto",
        help="Timezone in IANA format (e.g. America/New_York) or 'auto'. Default: auto.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for CSV and PNG files. Default: output",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds. Default: 30",
    )
    return parser.parse_args()


def validate_date_range(start_date: str, end_date: str) -> None:
    try:
        start = pd.to_datetime(start_date, format="%Y-%m-%d")
        end = pd.to_datetime(end_date, format="%Y-%m-%d")
    except ValueError as exc:
        raise WeatherClientError("Dates must use YYYY-MM-DD format.") from exc

    if start > end:
        raise WeatherClientError("Start date must be before or equal to end date.")


def normalize_metrics(metrics_csv: str) -> Tuple[List[str], List[str]]:
    metrics = [m.strip() for m in metrics_csv.split(",") if m.strip()]
    if not metrics:
        raise WeatherClientError("At least one metric must be provided via --metrics.")

    invalid = [
        m for m in metrics if m not in SUPPORTED_HOURLY_METRICS and m not in SUPPORTED_DAILY_METRICS
    ]
    if invalid:
        raise WeatherClientError(
            f"Unsupported metrics: {', '.join(invalid)}. "
            "Please use supported hourly/daily Open-Meteo metric names."
        )

    hourly_metrics = [m for m in metrics if m in SUPPORTED_HOURLY_METRICS]
    daily_metrics = [m for m in metrics if m in SUPPORTED_DAILY_METRICS]
    return hourly_metrics, daily_metrics


def geocode_location(location: str, timeout: int) -> Tuple[float, float, str]:
    params = {"name": location, "count": 1, "language": "en", "format": "json"}
    try:
        response = requests.get(GEOCODING_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WeatherClientError(f"Failed to geocode location '{location}': {exc}") from exc

    data = response.json()
    results = data.get("results")
    if not results:
        raise WeatherClientError(f"No geocoding results found for '{location}'.")

    hit = results[0]
    lat = hit.get("latitude")
    lon = hit.get("longitude")
    name = hit.get("name", location)
    country = hit.get("country", "")
    label = f"{name}, {country}".strip(", ")

    if lat is None or lon is None:
        raise WeatherClientError(f"Invalid geocoding response for '{location}'.")

    return float(lat), float(lon), label


def fetch_archive_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str,
    hourly_metrics: Sequence[str],
    daily_metrics: Sequence[str],
    timeout: int,
) -> Dict:
    params: Dict[str, str | float] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
    }
    if hourly_metrics:
        params["hourly"] = ",".join(hourly_metrics)
    if daily_metrics:
        params["daily"] = ",".join(daily_metrics)

    try:
        response = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WeatherClientError(f"Failed to fetch archive weather data: {exc}") from exc

    payload = response.json()
    if "hourly" not in payload and "daily" not in payload:
        raise WeatherClientError("Archive response does not contain hourly or daily weather data.")
    return payload


def build_dataframes(payload: Dict, hourly_metrics: Sequence[str], daily_metrics: Sequence[str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}

    if hourly_metrics:
        hourly = payload.get("hourly", {})
        time_values = hourly.get("time")
        if not time_values:
            raise WeatherClientError("Hourly metrics requested but no hourly time values returned.")
        hourly_df = pd.DataFrame({"time": pd.to_datetime(time_values)})
        for metric in hourly_metrics:
            hourly_df[metric] = hourly.get(metric)
        frames["hourly"] = hourly_df

    if daily_metrics:
        daily = payload.get("daily", {})
        time_values = daily.get("time")
        if not time_values:
            raise WeatherClientError("Daily metrics requested but no daily time values returned.")
        daily_df = pd.DataFrame({"time": pd.to_datetime(time_values)})
        for metric in daily_metrics:
            daily_df[metric] = daily.get(metric)
        frames["daily"] = daily_df

    return frames


def save_dataframes(frames: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, frame in frames.items():
        frame.to_csv(output_dir / f"{key}_weather.csv", index=False)


def plot_frame(frame: pd.DataFrame, metrics: Sequence[str], title: str, image_path: Path) -> None:
    if not metrics:
        return

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3.8 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        axis = axes[idx]
        axis.plot(frame["time"], frame[metric], linewidth=1.6)
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
        axis.set_title(metric)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(image_path, dpi=140)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    validate_date_range(args.start_date, args.end_date)
    hourly_metrics, daily_metrics = normalize_metrics(args.metrics)

    latitude, longitude, resolved_location = geocode_location(args.location, timeout=args.timeout)
    payload = fetch_archive_data(
        latitude=latitude,
        longitude=longitude,
        start_date=args.start_date,
        end_date=args.end_date,
        timezone=args.timezone,
        hourly_metrics=hourly_metrics,
        daily_metrics=daily_metrics,
        timeout=args.timeout,
    )

    frames = build_dataframes(payload, hourly_metrics=hourly_metrics, daily_metrics=daily_metrics)
    output_dir = Path(args.output_dir)
    save_dataframes(frames, output_dir)

    if hourly_metrics:
        plot_frame(
            frame=frames["hourly"],
            metrics=hourly_metrics,
            title=f"Hourly weather metrics for {resolved_location}",
            image_path=output_dir / "hourly_weather.png",
        )
    if daily_metrics:
        plot_frame(
            frame=frames["daily"],
            metrics=daily_metrics,
            title=f"Daily weather metrics for {resolved_location}",
            image_path=output_dir / "daily_weather.png",
        )

    print("Completed successfully.")
    print(f"Location: {resolved_location} ({latitude:.4f}, {longitude:.4f})")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Saved files in: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WeatherClientError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
