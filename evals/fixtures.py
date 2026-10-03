"""Deterministic weather fixtures for evals."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone


def _hourly_times(hours: int = 72, utc_offset: int = 19800) -> list[str]:
    tz = timezone(timedelta(seconds=utc_offset))
    start = datetime.now(tz).replace(minute=0, second=0, microsecond=0)
    return [(start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(hours)]


def make_weather(overrides: dict | None = None) -> dict:
    times = _hourly_times()
    n = len(times)
    base = {
        "latitude": 23.26,
        "longitude": 77.4,
        "utc_offset_seconds": 19800,
        "timezone": "Asia/Kolkata",
        "hourly_units": {
            "temperature_2m": "°C",
            "wind_gusts_10m": "km/h",
            "uv_index": "",
            "apparent_temperature": "°C",
            "precipitation": "mm",
            "weather_code": "wmo code",
            "pressure_msl": "hPa",
            "precipitation_probability": "%",
        },
        "hourly": {
            "time": times,
            "temperature_2m": [28.0] * n,
            "wind_gusts_10m": [10.0] * n,
            "uv_index": [3.0] * n,
            "apparent_temperature": [30.0] * n,
            "precipitation": [0.0] * n,
            "weather_code": [0] * n,
            "pressure_msl": [1010.0] * n,
            "precipitation_probability": [20] * n,
        },
    }
    if overrides:
        data = deepcopy(base)
        for k, v in overrides.items():
            if k == "hourly" and isinstance(v, dict):
                data["hourly"].update(v)
            else:
                data[k] = v
        return data
    return base


def patch_metric_series(payload: dict, metric: str, value: float | int, window_indices: slice | None = None) -> dict:
    data = deepcopy(payload)
    series = list(data["hourly"][metric])
    if window_indices is None:
        series = [value] * len(series)
    else:
        for i in range(len(series)):
            if i in range(window_indices.start or 0, window_indices.stop or len(series)):
                series[i] = value
    data["hourly"][metric] = series
    return data
