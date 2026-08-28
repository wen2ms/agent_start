import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
from langchain.tools import ToolRuntime, tool
from rag.rag_summarize import RagSummarizeService
from utils.config_handler import rag_config


@dataclass
class UserContext:
    user_id: str


def _get_user(user_id: str, encoding: str = "utf-8") -> dict[str, str] | None:
    with Path(rag_config["users_csv_path"]).open("r", encoding=encoding, newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row["user_id"] == user_id:
                return row
    return None


def _weather_code_to_description(code: int) -> str:
    descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snowfall",
        73: "Moderate snowfall",
        75: "Heavy snowfall",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return descriptions.get(code, f"Weather code {code}")


_rag_summarize_service = RagSummarizeService()


@tool(description="Search the trusted residential energy-management knowledge base.")
def energy_knowledge_search(question: str) -> str:
    return _rag_summarize_service.summarize(question)


@tool(description="Return the current user's city or service region.")
def get_user_location(runtime: ToolRuntime[UserContext]) -> str:
    user = _get_user(runtime.context.user_id)
    if user is None:
        return "User location is unavailable."
    city = user.get("city", "").strip()
    if not city:
        return "User location is unavailable."
    return city


@tool(description="Return the current user's energy-account identifier.")
def get_account_id(runtime: ToolRuntime[UserContext]) -> str:
    user = _get_user(runtime.context.user_id)
    if user is None:
        return "Energy account is unavailable."
    account_id = user.get("account_id", "").strip()
    if not account_id:
        return "Energy account is unavailable."
    return account_id


@tool(description="Return the current month in YYYY-MM format.")
def get_current_month(runtime: ToolRuntime[UserContext]) -> str:
    user = _get_user(runtime.context.user_id)
    if user is not None:
        timezone_name = user.get("timezone", "").strip()
        if timezone_name:
            now = datetime.now(ZoneInfo(timezone_name))
            return now.strftime("%Y-%m")
    return datetime.now().astimezone().strftime("%Y-%m")


@tool(description="Return current weather conditions for a specific city.")
def get_current_weather(city: str) -> dict[str, Any]:
    geocoding_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    geocoding_response.raise_for_status()
    geocoding_data = geocoding_response.json()
    results = geocoding_data.get("results")
    if not results:
        return {"error": f"Could not find location: {city}"}
    location = results[0]
    latitude = location["latitude"]
    longitude = location["longitude"]
    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
            "timezone": "auto",
        },
        timeout=10,
    )
    weather_response.raise_for_status()
    weather_data = weather_response.json()
    current = weather_data["current"]
    weather_code = int(current["weather_code"])
    return {
        "city": location["name"],
        "country": location.get("country"),
        "latitude": latitude,
        "longitude": longitude,
        "timezone": weather_data.get("timezone"),
        "temperature_c": current["temperature_2m"],
        "apparent_temperature_c": current["apparent_temperature"],
        "relative_humidity_percent": current["relative_humidity_2m"],
        "precipitation_mm": current["precipitation"],
        "wind_speed_kmh": current["wind_speed_10m"],
        "weather_code": weather_code,
        "description": _weather_code_to_description(weather_code),
    }


if __name__ == "__main__":
    print("\n" + "=" * 20 + "Knowledge Search" + "=" * 20)
    result = energy_knowledge_search.invoke({"question": "How can I reduce air conditioning energy consumption?"})
    print(result)

    print("\n" + "=" * 20 + "Weather" + "=" * 20)
    result = get_current_weather.invoke({"city": "Shanghai"})
    print(result)
