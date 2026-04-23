from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from .config import DataConfig, FeaturesConfig


@dataclass(slots=True)
class FeatureFrame:
    frame: pd.DataFrame
    feature_columns: list[str]


def build_feature_frame(
    raw_frame: pd.DataFrame,
    data_cfg: DataConfig,
    feature_cfg: FeaturesConfig,
) -> FeatureFrame:
    frame = raw_frame.copy()
    timestamp_col = data_cfg.timestamp_col
    target_col = data_cfg.target_col

    if timestamp_col not in frame.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    if target_col not in frame.columns:
        raise ValueError(f"Missing target column: {target_col}")

    frame = frame.sort_values(timestamp_col).reset_index(drop=True)

    numeric_cols = set([target_col, *data_cfg.known_feature_cols, *feature_cfg.lag_source_cols])
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    feature_columns: list[str] = []
    ts = pd.to_datetime(frame[timestamp_col])

    if feature_cfg.add_calendar_features:
        hour = ts.dt.hour + ts.dt.minute / 60.0
        day_of_week = ts.dt.dayofweek
        day_of_year = ts.dt.dayofyear

        calendar_features = {
            "hour_sin": (2 * math.pi * hour / 24).map(math.sin),
            "hour_cos": (2 * math.pi * hour / 24).map(math.cos),
            "dow_sin": (2 * math.pi * day_of_week / 7).map(math.sin),
            "dow_cos": (2 * math.pi * day_of_week / 7).map(math.cos),
            "doy_sin": (2 * math.pi * day_of_year / 366).map(math.sin),
            "doy_cos": (2 * math.pi * day_of_year / 366).map(math.cos),
            "is_weekend": day_of_week.isin([5, 6]).astype(int),
        }
        for name, series in calendar_features.items():
            frame[name] = series
            feature_columns.append(name)

    for col in data_cfg.known_feature_cols:
        if col not in frame.columns:
            raise ValueError(f"Missing known feature column: {col}")
        feature_columns.append(col)

    for source_col in feature_cfg.lag_source_cols:
        if source_col not in frame.columns:
            raise ValueError(f"Missing lag source column: {source_col}")

        for lag in feature_cfg.lags:
            feature_name = f"{source_col}_lag_{lag}"
            frame[feature_name] = frame[source_col].shift(lag)
            feature_columns.append(feature_name)

        shifted = frame[source_col].shift(1)
        for window in feature_cfg.rolling_windows:
            mean_name = f"{source_col}_roll_mean_{window}"
            std_name = f"{source_col}_roll_std_{window}"
            frame[mean_name] = shifted.rolling(window).mean()
            frame[std_name] = shifted.rolling(window).std(ddof=0)
            feature_columns.extend([mean_name, std_name])

    result_cols = [timestamp_col, target_col, *feature_columns]
    frame = frame[result_cols]
    frame = frame.dropna().reset_index(drop=True)
    return FeatureFrame(frame=frame, feature_columns=feature_columns)
