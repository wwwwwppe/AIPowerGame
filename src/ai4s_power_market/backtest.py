from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AppConfig
from .data import ensure_parent_dir
from .features import build_feature_frame
from .modeling import TabularPriceForecaster
from .storage import DynamicProgramDispatcher


@dataclass(slots=True)
class BacktestArtifacts:
    summary: pd.DataFrame
    predictions: pd.DataFrame
    dispatch: pd.DataFrame
    feature_columns: list[str]
    output_dir: Path


@dataclass(slots=True)
class FoldWindow:
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def _build_folds(n_rows: int, config: AppConfig) -> list[FoldWindow]:
    cfg = config.backtest
    windows: list[FoldWindow] = []
    cursor = cfg.min_train_size

    while cursor + cfg.test_size <= n_rows:
        if cfg.train_mode == "sliding":
            train_start = max(0, cursor - cfg.min_train_size)
        else:
            train_start = 0

        windows.append(
            FoldWindow(
                fold_id=len(windows) + 1,
                train_start=train_start,
                train_end=cursor,
                test_start=cursor,
                test_end=cursor + cfg.test_size,
            )
        )
        cursor += cfg.step_size

    if config.backtest.max_folds > 0:
        windows = windows[-config.backtest.max_folds :]
        windows = [
            FoldWindow(
                fold_id=index + 1,
                train_start=window.train_start,
                train_end=window.train_end,
                test_start=window.test_start,
                test_end=window.test_end,
            )
            for index, window in enumerate(windows)
        ]

    if not windows:
        raise ValueError(
            "Not enough rows after feature engineering to construct rolling folds. "
            "Reduce min_train_size/test_size or use more data."
        )

    return windows


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _timestamped_output_dir(base_dir: Path) -> Path:
    run_id = datetime.now().strftime("backtest_%Y%m%d_%H%M%S")
    output_dir = base_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_backtest(raw_frame: pd.DataFrame, config: AppConfig, output_dir: Path | None = None) -> BacktestArtifacts:
    feature_frame = build_feature_frame(raw_frame, config.data, config.features)
    frame = feature_frame.frame
    feature_columns = feature_frame.feature_columns

    folds = _build_folds(len(frame), config)
    dispatcher = DynamicProgramDispatcher(config.storage)

    summary_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[pd.DataFrame] = []
    dispatch_rows: list[pd.DataFrame] = []

    for fold in folds:
        train_df = frame.iloc[fold.train_start : fold.train_end].reset_index(drop=True)
        test_df = frame.iloc[fold.test_start : fold.test_end].reset_index(drop=True)

        X_train = train_df[feature_columns]
        y_train = train_df[config.data.target_col]
        X_test = test_df[feature_columns]
        y_test = test_df[config.data.target_col].to_numpy(dtype=float)

        forecaster = TabularPriceForecaster(config.model)
        fit_result = forecaster.fit(X_train, y_train)
        y_pred = forecaster.predict(X_test)

        timestamps = test_df[config.data.timestamp_col]
        realized_dispatch = dispatcher.optimize(y_pred, y_test, timestamps=timestamps)
        oracle_dispatch = dispatcher.optimize(y_test, y_test, timestamps=timestamps)

        prediction_frame = pd.DataFrame(
            {
                "fold_id": fold.fold_id,
                "timestamp": timestamps,
                "actual_price": y_test,
                "predicted_price": y_pred,
            }
        )
        prediction_rows.append(prediction_frame)

        realized_frame = realized_dispatch.schedule.copy()
        realized_frame["fold_id"] = fold.fold_id
        realized_frame["policy"] = "predicted_dispatch"
        dispatch_rows.append(realized_frame)

        oracle_frame = oracle_dispatch.schedule.copy()
        oracle_frame["fold_id"] = fold.fold_id
        oracle_frame["policy"] = "oracle_dispatch"
        dispatch_rows.append(oracle_frame)

        summary_rows.append(
            {
                "fold_id": fold.fold_id,
                "model_name": fit_result.model_name,
                "train_rows": fit_result.n_rows,
                "test_rows": len(test_df),
                "mae": _mae(y_test, y_pred),
                "rmse": _rmse(y_test, y_pred),
                "realized_revenue": realized_dispatch.summary["realized_revenue"],
                "oracle_revenue": oracle_dispatch.summary["realized_revenue"],
                "revenue_gap": oracle_dispatch.summary["realized_revenue"] - realized_dispatch.summary["realized_revenue"],
                "end_soc_mwh": realized_dispatch.summary["end_soc_mwh"],
                "terminal_gap_mwh": realized_dispatch.summary["terminal_gap_mwh"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    dispatch = pd.concat(dispatch_rows, ignore_index=True)

    base_output_dir = config.resolve_path(config.paths.results_dir)
    resolved_output_dir = output_dir or _timestamped_output_dir(base_output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(ensure_parent_dir(resolved_output_dir / "fold_summary.csv"), index=False)
    predictions.to_csv(ensure_parent_dir(resolved_output_dir / "predictions.csv"), index=False)
    dispatch.to_csv(ensure_parent_dir(resolved_output_dir / "dispatch.csv"), index=False)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config.config_path),
        "feature_columns": feature_columns,
        "mean_mae": float(summary["mae"].mean()),
        "mean_rmse": float(summary["rmse"].mean()),
        "mean_realized_revenue": float(summary["realized_revenue"].mean()),
        "mean_oracle_revenue": float(summary["oracle_revenue"].mean()),
        "storage": asdict(config.storage),
        "backtest": asdict(config.backtest),
    }
    (resolved_output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return BacktestArtifacts(
        summary=summary,
        predictions=predictions,
        dispatch=dispatch,
        feature_columns=feature_columns,
        output_dir=resolved_output_dir,
    )
