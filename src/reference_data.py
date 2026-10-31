"""Lookup helpers for the Flight Delay Advisor sidebar.

Flight features and route distances are sourced from the processed BTS dataset.
Airport names are loaded from the public AirportMiles CC0 directory when available;
a small local map is used as a fallback if the network is unavailable.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import FLIGHTS_PATH

AIRPORT_DIRECTORY_URL = "https://airportmiles.com/downloads/airports.csv"
CARRIER_DIRECTORY_URL = "https://www.bts.gov/topics/airlines-and-airports/airline-codes"

# Common US BTS carriers. Codes shown in the selector are always taken from the
# actual local BTS dataset; unknown codes remain selectable by code.
AIRLINE_NAMES: dict[str, str] = {
    "AA": "American Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "C5": "CommuteAir",
    "DL": "Delta Air Lines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "G7": "GoJet Airlines",
    "HA": "Hawaiian Airlines",
    "MQ": "Envoy Air",
    "NK": "Spirit Airlines",
    "OH": "PSA Airlines",
    "OO": "SkyWest Airlines",
    "PT": "Piedmont Airlines",
    "QX": "Horizon Air",
    "SY": "Sun Country Airlines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "YV": "Mesa Airlines",
    "YX": "Republic Airways",
    "9E": "Endeavor Air",
    "9K": "Cape Air",
    "AX": "Trans States Airlines (kode historis)",
    "EV": "ExpressJet (kode historis)",
    "ZW": "Air Wisconsin",
}

# Fallback for common airports if the online directory cannot be reached.
# Format: IATA: (official/commonly used airport name, served city, region/state, country).
FALLBACK_AIRPORTS: dict[str, tuple[str, str, str, str]] = {
    "ATL": ("Hartsfield-Jackson Atlanta International Airport", "Atlanta", "Georgia", "United States"),
    "AUS": ("Austin-Bergstrom International Airport", "Austin", "Texas", "United States"),
    "BNA": ("Nashville International Airport", "Nashville", "Tennessee", "United States"),
    "BOS": ("Boston Logan International Airport", "Boston", "Massachusetts", "United States"),
    "BWI": ("Baltimore/Washington International Airport", "Baltimore", "Maryland", "United States"),
    "CLE": ("Cleveland Hopkins International Airport", "Cleveland", "Ohio", "United States"),
    "CLT": ("Charlotte Douglas International Airport", "Charlotte", "North Carolina", "United States"),
    "DCA": ("Ronald Reagan Washington National Airport", "Washington", "District of Columbia", "United States"),
    "DEN": ("Denver International Airport", "Denver", "Colorado", "United States"),
    "DFW": ("Dallas Fort Worth International Airport", "Dallas-Fort Worth", "Texas", "United States"),
    "DTW": ("Detroit Metropolitan Wayne County Airport", "Detroit", "Michigan", "United States"),
    "EWR": ("Newark Liberty International Airport", "Newark / New York", "New Jersey", "United States"),
    "FLL": ("Fort Lauderdale-Hollywood International Airport", "Fort Lauderdale", "Florida", "United States"),
    "HNL": ("Daniel K. Inouye International Airport", "Honolulu", "Hawaii", "United States"),
    "IAD": ("Washington Dulles International Airport", "Washington", "Virginia", "United States"),
    "IAH": ("George Bush Intercontinental Airport", "Houston", "Texas", "United States"),
    "JAX": ("Jacksonville International Airport", "Jacksonville", "Florida", "United States"),
    "JFK": ("John F. Kennedy International Airport", "New York", "New York", "United States"),
    "LAS": ("Harry Reid International Airport", "Las Vegas", "Nevada", "United States"),
    "LAX": ("Los Angeles International Airport", "Los Angeles", "California", "United States"),
    "LGA": ("LaGuardia Airport", "New York", "New York", "United States"),
    "MCI": ("Kansas City International Airport", "Kansas City", "Missouri", "United States"),
    "MCO": ("Orlando International Airport", "Orlando", "Florida", "United States"),
    "MDW": ("Chicago Midway International Airport", "Chicago", "Illinois", "United States"),
    "MEM": ("Memphis International Airport", "Memphis", "Tennessee", "United States"),
    "MIA": ("Miami International Airport", "Miami", "Florida", "United States"),
    "MSP": ("Minneapolis-Saint Paul International Airport", "Minneapolis-St. Paul", "Minnesota", "United States"),
    "MSY": ("Louis Armstrong New Orleans International Airport", "New Orleans", "Louisiana", "United States"),
    "OAK": ("Oakland San Francisco Bay Airport", "Oakland", "California", "United States"),
    "ORD": ("Chicago O'Hare International Airport", "Chicago", "Illinois", "United States"),
    "PBI": ("Palm Beach International Airport", "West Palm Beach", "Florida", "United States"),
    "PDX": ("Portland International Airport", "Portland", "Oregon", "United States"),
    "PHL": ("Philadelphia International Airport", "Philadelphia", "Pennsylvania", "United States"),
    "PHX": ("Phoenix Sky Harbor International Airport", "Phoenix", "Arizona", "United States"),
    "PIT": ("Pittsburgh International Airport", "Pittsburgh", "Pennsylvania", "United States"),
    "RDU": ("Raleigh-Durham International Airport", "Raleigh-Durham", "North Carolina", "United States"),
    "SAN": ("San Diego International Airport", "San Diego", "California", "United States"),
    "SAT": ("San Antonio International Airport", "San Antonio", "Texas", "United States"),
    "SDF": ("Louisville Muhammad Ali International Airport", "Louisville", "Kentucky", "United States"),
    "SEA": ("Seattle-Tacoma International Airport", "Seattle", "Washington", "United States"),
    "SFO": ("San Francisco International Airport", "San Francisco", "California", "United States"),
    "SJC": ("San Jose Mineta International Airport", "San Jose", "California", "United States"),
    "SLC": ("Salt Lake City International Airport", "Salt Lake City", "Utah", "United States"),
    "SMF": ("Sacramento International Airport", "Sacramento", "California", "United States"),
    "STL": ("St. Louis Lambert International Airport", "St. Louis", "Missouri", "United States"),
    "TPA": ("Tampa International Airport", "Tampa", "Florida", "United States"),
}


def load_airport_directory(timeout_seconds: float = 5.0) -> dict[str, dict[str, str]]:
    """Load all scheduled-flight airport labels from the public CC0 CSV.

    This is optional metadata only. Failure to reach the directory is safe because
    the local fallback below remains available.
    """
    try:
        response = requests.get(
            AIRPORT_DIRECTORY_URL,
            timeout=(2.5, timeout_seconds),
            headers={"User-Agent": "FlightDelayPassengerRightsAdvisor/1.0"},
        )
        response.raise_for_status()
        frame = pd.read_csv(BytesIO(response.content), dtype=str).fillna("")
        required = {"iata", "name"}
        if not required.issubset(frame.columns):
            return {}
        data: dict[str, dict[str, str]] = {}
        for _, row in frame.iterrows():
            code = str(row.get("iata", "")).strip().upper()
            if len(code) != 3 or not code.isalpha():
                continue
            data[code] = {
                "name": str(row.get("name", "")).strip(),
                "municipality": str(row.get("municipality", "")).strip(),
                "served_city": str(row.get("served_city", "")).strip(),
                "country": str(row.get("country", "")).strip(),
                "region": str(row.get("region", "")).strip(),
            }
        return data
    except Exception:
        return {}


def build_route_reference(frame: pd.DataFrame) -> dict[str, Any]:
    """Build carrier, airport, route-distance, and route-option lists from BTS rows."""
    required = {"OP_CARRIER", "ORIGIN", "DEST", "DISTANCE"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset BTS belum memuat kolom referensi: {', '.join(missing)}")

    data = frame[list(required)].copy()
    for column in ("OP_CARRIER", "ORIGIN", "DEST"):
        data[column] = data[column].astype("string").str.strip().str.upper()
    data["DISTANCE"] = pd.to_numeric(data["DISTANCE"], errors="coerce")
    data = data.dropna(subset=["OP_CARRIER", "ORIGIN", "DEST", "DISTANCE"])
    data = data[(data["DISTANCE"] > 0) & (data["ORIGIN"].str.len() == 3) & (data["DEST"].str.len() == 3)]
    if data.empty:
        raise ValueError("Dataset BTS tidak memiliki kombinasi carrier/rute yang valid.")

    carriers = data["OP_CARRIER"].value_counts().index.tolist()
    origin_counts = data["ORIGIN"].value_counts()
    origins = origin_counts.index.tolist()

    routes = (
        data.groupby(["ORIGIN", "DEST"], as_index=False)
        .agg(distance_miles=("DISTANCE", "median"), flights=("DISTANCE", "size"))
        .sort_values(["ORIGIN", "flights", "DEST"], ascending=[True, False, True])
    )
    destinations_by_origin: dict[str, list[str]] = {}
    distance_by_route: dict[tuple[str, str], int] = {}
    for row in routes.itertuples(index=False):
        origin = str(row.ORIGIN)
        destination = str(row.DEST)
        destinations_by_origin.setdefault(origin, []).append(destination)
        distance_by_route[(origin, destination)] = int(round(float(row.distance_miles)))

    return {
        "carriers": carriers,
        "origins": origins,
        "destinations_by_origin": destinations_by_origin,
        "distance_by_route": distance_by_route,
    }


def airport_label(code: str, directory: dict[str, dict[str, str]] | None = None) -> str:
    """Return a searchable 'IATA — name, city, region, country' label."""
    code = str(code).strip().upper()
    directory = directory or {}
    entry = directory.get(code)
    if entry:
        name = entry.get("name") or code
        city = entry.get("served_city") or entry.get("municipality")
        details = [city, entry.get("region"), entry.get("country")]
        details = list(dict.fromkeys(item.strip() for item in details if item and item.strip()))
        suffix = f" · {', '.join(details)}" if details else ""
        return f"{code} — {name}{suffix}"
    fallback = FALLBACK_AIRPORTS.get(code)
    if fallback:
        name, city, region, country = fallback
        return f"{code} — {name} · {city}, {region}, {country}"
    return f"{code} — Nama bandara tidak tersedia pada direktori lokal"


def parse_carrier_directory(html: str) -> dict[str, str]:
    """Parse the official BTS airline-code table into {carrier_code: airline_name}."""
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for header_position, row in enumerate(rows):
            cells = row.find_all(["th", "td"])
            headers = ["".join(ch for ch in cell.get_text(" ", strip=True).lower() if ch.isalnum()) for cell in cells]
            code_index = next((i for i, h in enumerate(headers) if h in {"carrier", "carriercode", "code"}), None)
            name_index = next((i for i, h in enumerate(headers) if h in {"carriername", "airline", "airlinename", "name"} or ("name" in h and ("carrier" in h or "airline" in h))), None)
            if code_index is None or name_index is None:
                continue
            found: dict[str, str] = {}
            for data_row in rows[header_position + 1 :]:
                values = [cell.get_text(" ", strip=True) for cell in data_row.find_all(["th", "td"])]
                if len(values) <= max(code_index, name_index):
                    continue
                code = values[code_index].strip().upper()
                name = values[name_index].strip()
                if code and name and len(code) <= 3 and all(ch.isalnum() for ch in code):
                    found[code] = name
            if found:
                return found
    return {}


def load_carrier_directory(timeout_seconds: float = 5.0) -> dict[str, str]:
    """Load the official BTS airline-name directory; safe empty fallback on network failure."""
    try:
        response = requests.get(
            CARRIER_DIRECTORY_URL,
            timeout=(2.5, timeout_seconds),
            headers={"User-Agent": "FlightDelayPassengerRightsAdvisor/1.0"},
        )
        response.raise_for_status()
        return parse_carrier_directory(response.text)
    except Exception:
        return {}


def carrier_label(code: str, directory: dict[str, str] | None = None) -> str:
    """Return a searchable airline display label, retaining the BTS carrier code."""
    code = str(code).strip().upper()
    directory = directory or {}
    name = directory.get(code) or AIRLINE_NAMES.get(code)
    if name:
        return f"{code} — {name}"
    return f"{code} — nama maskapai tidak tersedia; pilih berdasarkan kode"
