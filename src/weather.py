"""Geocoding and Open-Meteo forecast fetch."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import httpx
from pydantic import BaseModel, Field

GEocode_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class GeocodeResult(BaseModel):
    name: str
    latitude: float
    longitude: float
    country_code: str | None = None
    admin1: str | None = None


class HourlyForecast(BaseModel):
    time: list[str]
    model_config = {"extra": "allow"}


class ForecastPayload(BaseModel):
    latitude: float
    longitude: float
    utc_offset_seconds: int
    timezone: str
    hourly: dict[str, Any]
    hourly_units: dict[str, str] = Field(default_factory=dict)


async def geocode_search(
    name: str,
    country_code: str | None = None,
    count: int = 5,
    client: httpx.AsyncClient | None = None,
) -> list[GeocodeResult]:
    params: dict[str, Any] = {"name": name, "count": count, "language": "en", "format": "json"}
    if country_code:
        params["country_code"] = country_code

    own = client is None
    if own:
        client = httpx.AsyncClient(timeout=8.0)
    try:
        r = await client.get(GEocode_URL, params=params)
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        out: list[GeocodeResult] = []
        for item in results[:count]:
            out.append(
                GeocodeResult(
                    name=item.get("name", name),
                    latitude=float(item["latitude"]),
                    longitude=float(item["longitude"]),
                    country_code=item.get("country_code"),
                    admin1=item.get("admin1"),
                )
            )
        return out
    finally:
        if own:
            await client.aclose()


async def fetch_forecast(
    latitude: float,
    longitude: float,
    hourly_vars: list[str],
    forecast_days: int = 3,
    client: httpx.AsyncClient | None = None,
) -> ForecastPayload:
    if not hourly_vars:
        hourly_vars = ["temperature_2m"]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(sorted(set(hourly_vars))),
        "timezone": "auto",
        "forecast_days": forecast_days,
    }

    own = client is None
    if own:
        client = httpx.AsyncClient(timeout=8.0)

    last_err: Exception | None = None
    for attempt in range(2):
        try:
            r = await client.get(FORECAST_URL, params=params)
            if r.status_code >= 500 and attempt == 0:
                await asyncio.sleep(0.3)
                continue
            r.raise_for_status()
            raw = r.json()
            return ForecastPayload.model_validate(raw)
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            last_err = e
            if attempt == 0:
                await asyncio.sleep(0.3)
                continue
            raise
        except Exception as e:
            last_err = e
            raise
    if last_err:
        raise last_err
    raise RuntimeError("forecast fetch failed")


GeocodeFn = Callable[..., Any]
FetchForecastFn = Callable[..., Any]
