#!/usr/bin/env python3
"""Fetch, process, and plot historical weather data from Open-Meteo."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

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

FAHRENHEIT_METRICS = {
    "temperature_2m",
    "apparent_temperature",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
}


class WeatherClientError(Exception):
    """Application-level exception for predictable user-facing failures."""


def metric_label(metric: str) -> str:
    if metric == "temperature_2m":
        return "Temperature"
    return metric


def slugify(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    compact = "_".join(part for part in normalized.split("_") if part)
    return compact or "location"


def build_run_output_dir(base_output_dir: Path, location: str, start_date: str, end_date: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{slugify(location)}_{start_date}_to_{end_date}_{timestamp}"
    return base_output_dir / folder_name


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
    def extract_result(payload: Dict) -> Tuple[float, float, str] | None:
        results = payload.get("results")
        if not results:
            return None

        hit = results[0]
        lat = hit.get("latitude")
        lon = hit.get("longitude")
        if lat is None or lon is None:
            return None

        name = hit.get("name", location)
        country = hit.get("country", "")
        label = f"{name}, {country}".strip(", ")
        return float(lat), float(lon), label

    parts = [part.strip() for part in location.split(",") if part.strip()]
    city_hint = parts[0] if parts else location.strip()
    country_hint = parts[1].upper() if len(parts) > 1 and len(parts[1]) == 2 else None

    attempts: List[Dict[str, str | int]] = [
        {"name": location, "count": 1, "language": "en", "format": "json"},
    ]
    if city_hint and city_hint != location:
        attempts.append({"name": city_hint, "count": 1, "language": "en", "format": "json"})
    if country_hint:
        attempts.append(
            {
                "name": city_hint,
                "count": 1,
                "language": "en",
                "format": "json",
                "countryCode": country_hint,
            }
        )

    last_request_error: requests.RequestException | None = None
    for params in attempts:
        try:
            response = requests.get(GEOCODING_URL, params=params, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            last_request_error = exc
            continue

        resolved = extract_result(response.json())
        if resolved is not None:
            return resolved

    if last_request_error is not None:
        raise WeatherClientError(f"Failed to geocode location '{location}': {last_request_error}") from last_request_error
    raise WeatherClientError(
        f"No geocoding results found for '{location}'. Try formats like 'New York' or 'New York, US'."
    )


def choose_weather_api_url(start_date: str) -> str:
    start = pd.to_datetime(start_date, format="%Y-%m-%d")
    today = pd.Timestamp.now().normalize()
    return FORECAST_URL if start >= today else ARCHIVE_URL


def fetch_weather_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str,
    hourly_metrics: Sequence[str],
    daily_metrics: Sequence[str],
    timeout: int,
) -> Dict:
    weather_api_url = choose_weather_api_url(start_date=start_date)
    params: Dict[str, str | float] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
        "temperature_unit": "fahrenheit",
    }
    if hourly_metrics:
        params["hourly"] = ",".join(hourly_metrics)
    if daily_metrics:
        params["daily"] = ",".join(daily_metrics)

    try:
        response = requests.get(weather_api_url, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WeatherClientError(f"Failed to fetch weather data: {exc}") from exc

    payload = response.json()
    if "hourly" not in payload and "daily" not in payload:
        raise WeatherClientError("Weather response does not contain hourly or daily weather data.")
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
        label = metric_label(metric)
        axis.plot(frame["time"], frame[metric], linewidth=1.6)
        if metric in FAHRENHEIT_METRICS:
            axis.set_ylabel(f"{label} (°F)")
        else:
            axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.set_title(label)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(image_path, dpi=140)
    plt.close(fig)


def plot_below_freezing_frame(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    title: str,
    image_path: Path,
    freezing_point_f: float = 32.0,
) -> bool:
    freezing_metrics = [metric for metric in metrics if metric in FAHRENHEIT_METRICS and metric in frame.columns]
    if not freezing_metrics:
        return False

    primary_metric = "temperature_2m" if "temperature_2m" in frame.columns else freezing_metrics[0]
    subfreezing_series = frame[primary_metric].where(frame[primary_metric] <= freezing_point_f)
    subfreezing = subfreezing_series.notna()
    if not subfreezing.any():
        return False

    streak_starts: List[pd.Timestamp] = []
    streak_lengths: List[int] = []
    current_start: pd.Timestamp | None = None
    current_length = 0
    previous_time: pd.Timestamp | None = None

    for timestamp, is_subfreezing in zip(frame["time"], subfreezing):
        if is_subfreezing:
            is_continuation = (
                previous_time is not None
                and pd.notna(previous_time)
                and pd.notna(timestamp)
                and timestamp - previous_time == pd.Timedelta(hours=1)
                and current_start is not None
            )
            if not is_continuation:
                if current_start is not None and current_length > 0:
                    streak_starts.append(current_start)
                    streak_lengths.append(current_length)
                current_start = timestamp
                current_length = 1
            else:
                current_length += 1
        else:
            if current_start is not None and current_length > 0:
                streak_starts.append(current_start)
                streak_lengths.append(current_length)
            current_start = None
            current_length = 0
        previous_time = timestamp

    if current_start is not None and current_length > 0:
        streak_starts.append(current_start)
        streak_lengths.append(current_length)

    if not streak_starts:
        return False

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    temp_axis, streak_axis = axes
    primary_label = metric_label(primary_metric)
    temp_axis.plot(frame["time"], subfreezing_series, linewidth=1.8)
    temp_axis.set_ylabel(f"{primary_label} (°F)")
    temp_axis.set_title(f"{primary_label} (below freezing only)")
    temp_axis.grid(alpha=0.25)

    streak_axis.bar(streak_starts, streak_lengths, width=pd.Timedelta(hours=8))
    streak_axis.set_ylabel("Consecutive hours")
    streak_axis.set_xlabel("Streak start time")
    streak_axis.set_title(f"{primary_label} consecutive below-freezing streaks")
    streak_axis.grid(alpha=0.25, axis="y")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(image_path, dpi=140)
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    validate_date_range(args.start_date, args.end_date)
    hourly_metrics, daily_metrics = normalize_metrics(args.metrics)

    latitude, longitude, resolved_location = geocode_location(args.location, timeout=args.timeout)
    payload = fetch_weather_data(
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
    output_root = Path(args.output_dir)
    output_dir = build_run_output_dir(
        base_output_dir=output_root,
        location=resolved_location,
        start_date=args.start_date,
        end_date=args.end_date,
    )
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
    if hourly_metrics:
        plot_below_freezing_frame(
            frame=frames["hourly"],
            metrics=hourly_metrics,
            title=f"Below-freezing hourly metrics for {resolved_location}",
            image_path=output_dir / "below_freezing_weather.png",
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
