"""Synthetic airline bookings data and a one-week demo.

Each day's file simulates something that really happens with files
delivered over SFTP:

  day 1  normal (becomes the baseline)
  day 2  normal day-to-day variation                   -> expect PASS
  day 3  passenger_email mostly missing (CRM column)   -> expect WARN
  day 4  source switched fares to USD                  -> expect BLOCK
  day 5  fare_amount renamed to total_fare             -> expect BLOCK
  day 6  partial file: transfer cut off                -> expect BLOCK
  day 7  email gap again, after a person said day 3
         was a known, harmless issue                   -> expect PASS

All data is generated; no real airline or passenger data is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROUTES = [
    ("MCT", "DXB"), ("MCT", "SLL"), ("MCT", "KHI"), ("MCT", "CMB"),
    ("MCT", "DOH"), ("SLL", "MCT"), ("DXB", "MCT"), ("MCT", "COK"),
]
CHANNELS = ["web", "app", "agent", "call_center"]
CHANNEL_P = [0.45, 0.35, 0.15, 0.05]
STATUSES = ["PAID", "PENDING", "REFUNDED"]
STATUS_P = [0.90, 0.07, 0.03]


def make_bookings(day: int, rows: int = 1200, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed if seed is not None else 1000 + day)
    route_idx = rng.integers(0, len(ROUTES), size=rows)
    pax = rng.choice([1, 1, 1, 2, 2, 3, 4], size=rows)
    base_fare = rng.gamma(shape=6.0, scale=8.0, size=rows) + 15  # OMR
    return pd.DataFrame(
        {
            "booking_id": [f"B{day:02d}{i:05d}" for i in range(rows)],
            "booking_date": f"2026-10-{day:02d}",
            "origin": [ROUTES[i][0] for i in route_idx],
            "destination": [ROUTES[i][1] for i in route_idx],
            "fare_amount": np.round(base_fare * pax, 3),
            "currency": "OMR",
            "pax_count": pax,
            "channel": rng.choice(CHANNELS, size=rows, p=CHANNEL_P),
            "payment_status": rng.choice(STATUSES, size=rows, p=STATUS_P),
            "passenger_email": [f"pax{day:02d}{i:05d}@example.com" for i in range(rows)],
        }
    )


@dataclass
class Scenario:
    day: int
    title: str
    expected: str


SCENARIOS = [
    Scenario(1, "normal day (baseline)", "LOAD"),
    Scenario(2, "normal day-to-day variation", "PASS"),
    Scenario(3, "passenger_email mostly missing", "WARN"),
    Scenario(4, "fares switched to USD", "BLOCK"),
    Scenario(5, "fare_amount renamed to total_fare", "BLOCK"),
    Scenario(6, "partial file (transfer cut off)", "BLOCK"),
    Scenario(7, "email gap again, now a known issue", "PASS"),
]


def make_day(day: int) -> pd.DataFrame:
    rng = np.random.default_rng(42 + day)
    if day in (1, 2):
        return make_bookings(day, rows=int(rng.integers(1150, 1260)))

    df = make_bookings(day, rows=int(rng.integers(1150, 1260)))
    if day in (3, 7):
        mask = rng.random(len(df)) < 0.35
        df.loc[mask, "passenger_email"] = None
    elif day == 4:
        df["fare_amount"] = np.round(df["fare_amount"] * 2.6, 2)  # OMR -> USD
        df["currency"] = "USD"
    elif day == 5:
        df = df.rename(columns={"fare_amount": "total_fare"})
    elif day == 6:
        df = df.head(int(len(df) * 0.3))
    return df


def write_week(landing: str | Path) -> list[Path]:
    landing = Path(landing)
    landing.mkdir(parents=True, exist_ok=True)
    paths = []
    for scenario in SCENARIOS:
        path = landing / f"bookings_2026-10-{scenario.day:02d}.csv"
        make_day(scenario.day).to_csv(path, index=False)
        paths.append(path)
    return paths
