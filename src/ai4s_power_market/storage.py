from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StorageConfig


@dataclass(slots=True)
class DispatchResult:
    schedule: pd.DataFrame
    summary: dict[str, float]


class DynamicProgramDispatcher:
    """Approximate deterministic storage dispatch using discretized dynamic programming."""

    def __init__(self, config: StorageConfig) -> None:
        self.config = config
        self.interval_hours = config.interval_minutes / 60.0
        self.terminal_penalty_per_mwh = 1_000.0
        self.soc_levels = self._build_soc_levels()
        self.power_levels = self._build_power_levels()

    def _build_soc_levels(self) -> np.ndarray:
        step = self.config.soc_grid_mwh
        max_soc = self.config.energy_capacity_mwh
        levels = np.arange(0.0, max_soc + step * 0.5, step)
        return np.unique(np.round(levels, 8))

    def _build_power_levels(self) -> np.ndarray:
        step = self.config.power_grid_mw
        levels = np.arange(
            -self.config.max_charge_mw,
            self.config.max_discharge_mw + step * 0.5,
            step,
        )
        levels = np.append(levels, [0.0, -self.config.max_charge_mw, self.config.max_discharge_mw])
        levels = np.clip(levels, -self.config.max_charge_mw, self.config.max_discharge_mw)
        return np.unique(np.round(levels, 8))

    def _soc_transition(self, soc_mwh: float, power_mw: float) -> float | None:
        if power_mw >= 0:
            next_soc = soc_mwh - (power_mw / self.config.discharge_efficiency) * self.interval_hours
        else:
            charge_mw = -power_mw
            next_soc = soc_mwh + charge_mw * self.config.charge_efficiency * self.interval_hours

        if next_soc < -1e-9 or next_soc > self.config.energy_capacity_mwh + 1e-9:
            return None

        return float(np.clip(next_soc, 0.0, self.config.energy_capacity_mwh))

    def _nearest_soc_index(self, soc_mwh: float) -> int:
        return int(np.argmin(np.abs(self.soc_levels - soc_mwh)))

    def optimize(
        self,
        forecast_prices: np.ndarray | pd.Series,
        actual_prices: np.ndarray | pd.Series | None = None,
        timestamps: pd.Series | pd.Index | None = None,
    ) -> DispatchResult:
        forecast = np.asarray(forecast_prices, dtype=float)
        actual = np.asarray(actual_prices if actual_prices is not None else forecast_prices, dtype=float)

        if forecast.shape != actual.shape:
            raise ValueError("forecast_prices and actual_prices must have the same shape")

        n_steps = len(forecast)
        n_soc = len(self.soc_levels)
        dp = np.full((n_steps + 1, n_soc), -np.inf)
        action_choice = np.zeros((n_steps, n_soc), dtype=float)
        next_choice = np.zeros((n_steps, n_soc), dtype=int)

        dp[n_steps, :] = -np.abs(self.soc_levels - self.config.terminal_soc_mwh) * self.terminal_penalty_per_mwh

        for step in range(n_steps - 1, -1, -1):
            for soc_idx, soc in enumerate(self.soc_levels):
                best_value = -np.inf
                best_action = 0.0
                best_next_idx = soc_idx

                for power_mw in self.power_levels:
                    next_soc = self._soc_transition(soc, power_mw)
                    if next_soc is None:
                        continue

                    next_idx = self._nearest_soc_index(next_soc)
                    step_value = forecast[step] * power_mw * self.interval_hours + dp[step + 1, next_idx]

                    if step_value > best_value:
                        best_value = step_value
                        best_action = power_mw
                        best_next_idx = next_idx

                dp[step, soc_idx] = best_value
                action_choice[step, soc_idx] = best_action
                next_choice[step, soc_idx] = best_next_idx

        current_idx = self._nearest_soc_index(self.config.initial_soc_mwh)
        rows: list[dict[str, float | str]] = []
        current_soc = float(self.soc_levels[current_idx])

        for step in range(n_steps):
            power_mw = float(action_choice[step, current_idx])
            next_idx = int(next_choice[step, current_idx])
            next_soc = float(self.soc_levels[next_idx])

            timestamp = None if timestamps is None else pd.Timestamp(timestamps[step])
            rows.append(
                {
                    "timestamp": timestamp,
                    "forecast_price": float(forecast[step]),
                    "actual_price": float(actual[step]),
                    "power_to_grid_mw": power_mw,
                    "soc_start_mwh": current_soc,
                    "soc_end_mwh": next_soc,
                    "forecast_revenue": float(forecast[step] * power_mw * self.interval_hours),
                    "realized_revenue": float(actual[step] * power_mw * self.interval_hours),
                }
            )

            current_idx = next_idx
            current_soc = next_soc

        schedule = pd.DataFrame(rows)
        summary = {
            "steps": float(n_steps),
            "forecast_revenue": float(schedule["forecast_revenue"].sum()),
            "realized_revenue": float(schedule["realized_revenue"].sum()),
            "start_soc_mwh": float(schedule["soc_start_mwh"].iloc[0]) if not schedule.empty else self.config.initial_soc_mwh,
            "end_soc_mwh": float(schedule["soc_end_mwh"].iloc[-1]) if not schedule.empty else self.config.initial_soc_mwh,
            "terminal_gap_mwh": float(abs(current_soc - self.config.terminal_soc_mwh)),
        }
        return DispatchResult(schedule=schedule, summary=summary)
