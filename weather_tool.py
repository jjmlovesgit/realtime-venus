"""Small, keyless current-weather lookup for U.S. ZIP codes."""

from __future__ import annotations

import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


_ZIP_RE = re.compile(r"(?<!\d)(\d{5})(?!\d)")
_WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with light hail",
    99: "thunderstorm with heavy hail",
}

_HEADERS = {"User-Agent": "Realtime-Venus-local-weather-tool/1.0"}


def extract_us_zip(delegate_text: str) -> str:
    matches = set(_ZIP_RE.findall(delegate_text))
    if len(matches) != 1:
        raise ValueError("Weather tool request must contain exactly one five-digit U.S. ZIP code.")
    return matches.pop()


def _get_json(url: str) -> dict:
    request = Request(url, headers=_HEADERS)
    try:
        with urlopen(request, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Weather service request failed: {exc}") from exc


def current_weather_by_zip(zip_code: str) -> str:
    """Return a concise current-conditions summary for a U.S. ZIP code."""
    if not re.fullmatch(r"\d{5}", zip_code):
        raise ValueError("ZIP code must contain five digits.")

    geo_query = urlencode({"name": zip_code, "count": 10, "language": "en", "format": "json"})
    geo = _get_json(f"https://geocoding-api.open-meteo.com/v1/search?{geo_query}")
    locations = [
        item for item in geo.get("results", [])
        if item.get("country_code") == "US" and zip_code in item.get("postcodes", [])
    ]
    if not locations:
        raise ValueError(f"I couldn't find U.S. ZIP code {zip_code}.")
    location = locations[0]

    params = urlencode(
        {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "auto",
        }
    )
    weather = _get_json(f"https://api.open-meteo.com/v1/forecast?{params}")
    current = weather.get("current")
    if not isinstance(current, dict):
        raise RuntimeError("Weather service returned no current conditions.")

    conditions = _WEATHER_CODES.get(current.get("weather_code"), "conditions unavailable")
    city = location.get("name", "Unknown location")
    state = location.get("admin1")
    place = f"{city}, {state}" if state else city
    return (
        f"Current weather for ZIP {zip_code} ({place}): {conditions}, "
        f"{current['temperature_2m']:.0f} degrees Fahrenheit, feels like "
        f"{current['apparent_temperature']:.0f}, humidity {current['relative_humidity_2m']:.0f} percent, "
        f"wind {current['wind_speed_10m']:.0f} miles per hour, "
        f"precipitation {current['precipitation']:.2f} millimeters."
    )


def run_weather_delegate(delegate_text: str) -> str:
    """Validate the model's requested tool and perform its ZIP lookup."""
    try:
        request = json.loads(delegate_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Weather tool request was not valid JSON.") from exc
    if not isinstance(request, dict) or request.get("tool") != "weather_by_zip":
        raise ValueError("Only the weather_by_zip tool is enabled.")
    zip_code = request.get("zip_code")
    if not isinstance(zip_code, str) or not re.fullmatch(r"\d{5}", zip_code):
        raise ValueError("Weather tool requires a five-digit U.S. ZIP code.")
    return current_weather_by_zip(zip_code)
