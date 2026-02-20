#!/usr/bin/env python3
"""Fetch, process, and plot historical weather data from Open-Meteo."""

from __future__ import annotations

import argparse
import os
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
HUGGINGFACE_ROUTER_CHAT_COMPLETIONS_URL = (
    "https://router.huggingface.co/hf-inference/models/{model}/v1/chat/completions"
)
HUGGINGFACE_OPENAI_COMPAT_CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions"

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
            "Fetch historical weather data for a ZIP code using Open-Meteo, "
            "then process and plot selected metrics."
        )
    )
    parser.add_argument("--zip-code", required=True, help="ZIP/postal code, e.g. '98101'.")
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
    parser.add_argument(
        "--llm-summary",
        action="store_true",
        help=(
            "Generate an LLM summary focused on extreme weather events using CSV data. "
            "Requires a Hugging Face token."
        ),
    )
    parser.add_argument(
        "--hf-model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Hugging Face model ID used with --llm-summary.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable name that stores your Hugging Face token. Default: HF_TOKEN",
    )
    parser.add_argument(
        "--llm-max-rows",
        type=int,
        default=240,
        help="Max CSV rows per frame included in LLM context. Default: 240",
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


def geocode_zip_code(zip_code: str, timeout: int) -> Tuple[float, float, str]:
    def extract_result(payload: Dict) -> Tuple[float, float, str] | None:
        results = payload.get("results")
        if not results:
            return None

        hit = results[0]
        lat = hit.get("latitude")
        lon = hit.get("longitude")
        if lat is None or lon is None:
            return None

        name = hit.get("name", zip_code)
        postal_code = hit.get("postcode") or hit.get("postcodes")
        country = hit.get("country", "")
        if isinstance(postal_code, list):
            postal_code = postal_code[0] if postal_code else ""
        label_parts = [str(postal_code).strip() if postal_code else "", name, country]
        label = ", ".join(part for part in label_parts if part).strip(", ")
        return float(lat), float(lon), label

    attempts: List[Dict[str, str | int]] = [
        {"name": zip_code, "count": 1, "language": "en", "format": "json"},
    ]

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
        raise WeatherClientError(f"Failed to geocode ZIP code '{zip_code}': {last_request_error}") from last_request_error
    raise WeatherClientError(
        f"No geocoding results found for ZIP code '{zip_code}'."
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


def summarize_numeric_extremes(frame: pd.DataFrame, frame_name: str) -> str:
    if frame.empty:
        return f"{frame_name}: no rows available."

    lines: List[str] = [f"{frame_name} extreme snapshots:"]
    working = frame.copy()
    if "time" in working.columns:
        working["time"] = pd.to_datetime(working["time"], errors="coerce")
    numeric_cols = [col for col in working.columns if col != "time" and pd.api.types.is_numeric_dtype(working[col])]
    if not numeric_cols:
        lines.append("- no numeric columns found")
        return "\n".join(lines)

    for col in numeric_cols:
        non_null = working[["time", col]].dropna() if "time" in working.columns else working[[col]].dropna()
        if non_null.empty:
            continue

        max_row = non_null.loc[non_null[col].idxmax()]
        min_row = non_null.loc[non_null[col].idxmin()]
        p95 = non_null[col].quantile(0.95)
        p05 = non_null[col].quantile(0.05)

        max_time = max_row["time"] if "time" in non_null.columns else "n/a"
        min_time = min_row["time"] if "time" in non_null.columns else "n/a"
        lines.append(
            f"- {col}: min={min_row[col]:.2f} at {min_time}, max={max_row[col]:.2f} at {max_time}, "
            f"p05={p05:.2f}, p95={p95:.2f}"
        )

    return "\n".join(lines)


def build_llm_prompt(
    frames: Dict[str, pd.DataFrame],
    location_label: str,
    start_date: str,
    end_date: str,
    max_rows: int,
) -> str:
    prompt_parts: List[str] = [
        f"Location: {location_label}",
        f"Date range: {start_date} to {end_date}",
        "Task: Identify notable and potentially extreme weather events from the CSV data.",
        "Return a concise report with sections: Executive Summary, Extreme Events, Risk Notes, and Follow-up Checks.",
        "Use concrete timestamps and values whenever possible.",
    ]

    for frame_name, frame in frames.items():
        clipped = frame.head(max_rows).copy()
        if "time" in clipped.columns:
            clipped["time"] = pd.to_datetime(clipped["time"], errors="coerce").astype(str)
        prompt_parts.append("")
        prompt_parts.append(summarize_numeric_extremes(frame=frame, frame_name=frame_name))
        prompt_parts.append(f"{frame_name} CSV sample (first {len(clipped)} rows):")
        prompt_parts.append(clipped.to_csv(index=False))

    return "\n".join(prompt_parts)


def extract_hf_generated_text(payload: object) -> str:
    if isinstance(payload, dict):
        if "generated_text" in payload and isinstance(payload["generated_text"], str):
            return payload["generated_text"].strip()
        if "choices" in payload and isinstance(payload["choices"], list):
            choices = payload["choices"]
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content.strip()
                    if isinstance(content, list):
                        chunks: List[str] = []
                        for item in content:
                            if isinstance(item, dict):
                                chunk = item.get("text")
                                if isinstance(chunk, str):
                                    chunks.append(chunk)
                        if chunks:
                            return "".join(chunks).strip()
                text = choices[0].get("text")
                if isinstance(text, str):
                    return text.strip()

    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            text = first.get("generated_text")
            if isinstance(text, str):
                return text.strip()

    return ""


def extract_hf_error_message(payload: object) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, str):
            return error.strip()

        message = payload.get("message")
        if isinstance(message, str):
            return message.strip()

    return ""


def generate_huggingface_extreme_weather_summary(
    frames: Dict[str, pd.DataFrame],
    output_dir: Path,
    location_label: str,
    start_date: str,
    end_date: str,
    model: str,
    token_env_name: str,
    max_rows: int,
    timeout: int,
) -> Path:
    if max_rows <= 0:
        raise WeatherClientError("--llm-max-rows must be greater than 0.")

    token = os.getenv(token_env_name)
    if not token:
        raise WeatherClientError(
            f"LLM summary requested, but token env var '{token_env_name}' is not set."
        )

    prompt = build_llm_prompt(
        frames=frames,
        location_label=location_label,
        start_date=start_date,
        end_date=end_date,
        max_rows=max_rows,
    )

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Primary path: Hugging Face router chat completions endpoint.
    chat_payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a meteorological analyst. Focus on extreme weather patterns and anomalies. "
                    "Do not invent values. If confidence is limited, state uncertainty explicitly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 700,
        "temperature": 0.1,
    }

    # Primary path: server-selected provider via OpenAI-compatible router endpoint.
    openai_compat_payload = {
        "model": model,
        "messages": chat_payload["messages"],
        "max_tokens": chat_payload["max_tokens"],
        "temperature": chat_payload["temperature"],
    }
    response = requests.post(
        HUGGINGFACE_OPENAI_COMPAT_CHAT_COMPLETIONS_URL,
        headers=headers,
        json=openai_compat_payload,
        timeout=timeout,
    )

    # Fallback path: force hf-inference provider route for models not yet mapped on /v1.
    if response.status_code in {400, 404, 422}:
        response = requests.post(
            HUGGINGFACE_ROUTER_CHAT_COMPLETIONS_URL.format(model=model),
            headers=headers,
            json=chat_payload,
            timeout=timeout,
        )

    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        message = ""
        try:
            message = extract_hf_error_message(response.json())
        except ValueError:
            message = ""

        if response.status_code == 410:
            details = (
                f" Details: {message}" if message else ""
            )
            raise WeatherClientError(
                "Hugging Face returned 410 Gone for this model/provider route. "
                "Try a different --hf-model or a model with another available inference provider."
                f"{details}"
            ) from exc

        if message:
            raise WeatherClientError(
                f"Hugging Face API request failed ({response.status_code}): {message}"
            ) from exc
        raise WeatherClientError(f"Hugging Face API request failed: {exc}") from exc

    payload = response.json()
    if isinstance(payload, dict) and "error" in payload:
        raise WeatherClientError(f"Hugging Face API error: {payload['error']}")

    summary_text = extract_hf_generated_text(payload)
    if not summary_text:
        raise WeatherClientError("Hugging Face API returned no generated summary text.")

    summary_path = output_dir / "llm_extreme_weather_summary.md"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    return summary_path


def main() -> int:
    args = parse_args()
    validate_date_range(args.start_date, args.end_date)
    hourly_metrics, daily_metrics = normalize_metrics(args.metrics)

    latitude, longitude, resolved_location = geocode_zip_code(args.zip_code, timeout=args.timeout)
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
        location=args.zip_code,
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

    summary_path: Path | None = None
    if args.llm_summary:
        summary_path = generate_huggingface_extreme_weather_summary(
            frames=frames,
            output_dir=output_dir,
            location_label=resolved_location,
            start_date=args.start_date,
            end_date=args.end_date,
            model=args.hf_model,
            token_env_name=args.hf_token_env,
            max_rows=args.llm_max_rows,
            timeout=args.timeout,
        )

    print("Completed successfully.")
    print(f"Location: {resolved_location} ({latitude:.4f}, {longitude:.4f})")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Saved files in: {output_dir.resolve()}")
    if summary_path is not None:
        print(f"LLM summary: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WeatherClientError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
