from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_market_data(csv_path: str | Path, timestamp_col: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if timestamp_col not in frame.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=False)
    frame = frame.sort_values(timestamp_col).reset_index(drop=True)
    return frame


def save_dataframe(frame: pd.DataFrame, csv_path: str | Path) -> Path:
    target = ensure_parent_dir(csv_path)
    frame.to_csv(target, index=False)
    return target


def generate_synthetic_market_data(days: int, freq_minutes: int, seed: int) -> pd.DataFrame:
    periods = int(days * 24 * 60 / freq_minutes)
    timestamps = pd.date_range("2025-01-01", periods=periods, freq=f"{freq_minutes}min")
    rng = np.random.default_rng(seed)

    hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0
    day_of_week = timestamps.dayofweek.to_numpy()
    year_progress = np.arange(periods) / max(periods - 1, 1)

    load = (
        90
        + 18 * np.sin(2 * np.pi * hour / 24 - 0.7)
        + 8 * np.cos(2 * np.pi * day_of_week / 7)
        + 6 * year_progress
        + rng.normal(0, 3.0, size=periods)
    )

    solar_shape = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    solar = np.maximum(0.0, 35 * solar_shape + rng.normal(0, 2.0, size=periods))

    wind = (
        22
        + 10 * np.sin(2 * np.pi * np.arange(periods) / (96 * 5) + 1.2)
        + rng.normal(0, 4.0, size=periods)
    )
    wind = np.maximum(0.0, wind)

    temperature = (
        14
        + 10 * np.sin(2 * np.pi * hour / 24 - 1.4)
        + 4 * np.sin(2 * np.pi * year_progress)
        + rng.normal(0, 1.5, size=periods)
    )

    evening_peak = ((hour >= 18) & (hour <= 21)).astype(float)
    low_net_load = (load - wind - solar) < np.quantile(load - wind - solar, 0.15)
    spike_mask = rng.random(periods) < 0.02

    price = (
        55
        + 1.35 * (load - load.mean())
        - 0.85 * wind
        - 1.10 * solar
        + 0.9 * temperature
        + 18 * evening_peak
        - 25 * low_net_load.astype(float)
        + spike_mask * rng.uniform(70, 180, size=periods)
        + rng.normal(0, 6.0, size=periods)
    )
    price = np.clip(price, -50.0, 500.0)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "load": np.round(load, 4),
            "wind": np.round(wind, 4),
            "solar": np.round(solar, 4),
            "temperature": np.round(temperature, 4),
            "price": np.round(price, 4),
        }
    )
