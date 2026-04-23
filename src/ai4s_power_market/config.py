from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(slots=True)
class PathsConfig:
    results_dir: str
    default_data_path: str


@dataclass(slots=True)
class DataConfig:
    timestamp_col: str
    target_col: str
    known_feature_cols: list[str]


@dataclass(slots=True)
class FeaturesConfig:
    lag_source_cols: list[str]
    lags: list[int]
    rolling_windows: list[int]
    add_calendar_features: bool


@dataclass(slots=True)
class ModelConfig:
    model_type: str
    random_state: int
    learning_rate: float
    max_depth: int
    max_iter: int


@dataclass(slots=True)
class BacktestConfig:
    min_train_size: int
    test_size: int
    step_size: int
    dispatch_horizon: int
    max_folds: int
    train_mode: str


@dataclass(slots=True)
class StorageConfig:
    energy_capacity_mwh: float
    max_charge_mw: float
    max_discharge_mw: float
    charge_efficiency: float
    discharge_efficiency: float
    initial_soc_mwh: float
    terminal_soc_mwh: float
    interval_minutes: int
    soc_grid_mwh: float
    power_grid_mw: float


@dataclass(slots=True)
class DemoConfig:
    days: int
    freq_minutes: int
    seed: int


@dataclass(slots=True)
class AppConfig:
    root_dir: Path
    config_path: Path
    paths: PathsConfig
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    backtest: BacktestConfig
    storage: StorageConfig
    demo: DemoConfig

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.root_dir / path


def load_config(config_path: str | Path, project_root: str | Path | None = None) -> AppConfig:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        if project_root is not None:
            cfg_path = Path(project_root) / cfg_path
        else:
            cfg_path = cfg_path.resolve()

    root_dir = Path(project_root).resolve() if project_root is not None else cfg_path.parent
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))

    return AppConfig(
        root_dir=root_dir,
        config_path=cfg_path.resolve(),
        paths=PathsConfig(**raw["paths"]),
        data=DataConfig(**raw["data"]),
        features=FeaturesConfig(**raw["features"]),
        model=ModelConfig(**raw["model"]),
        backtest=BacktestConfig(**raw["backtest"]),
        storage=StorageConfig(**raw["storage"]),
        demo=DemoConfig(**raw["demo"]),
    )
