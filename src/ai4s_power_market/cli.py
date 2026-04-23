from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import run_backtest
from .config import load_config
from .data import generate_synthetic_market_data, load_market_data, save_dataframe
from .logging_utils import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI4S power market baseline scaffold")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the TOML config file, relative to the project root by default.",
    )

    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("generate-demo-data", help="Generate synthetic market data for smoke tests.")
    demo_parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    demo_parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path. Defaults to config.paths.default_data_path.",
    )

    backtest_parser = subparsers.add_parser("backtest", help="Run the baseline rolling backtest.")
    backtest_parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    backtest_parser.add_argument(
        "--data",
        default=None,
        help="Optional data CSV path. Defaults to config.paths.default_data_path.",
    )
    backtest_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to a timestamped folder under config.paths.results_dir.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    config_path = args.config or "configs/baseline.toml"
    config = load_config(config_path, project_root=PROJECT_ROOT)

    if args.command == "generate-demo-data":
        output_path = config.resolve_path(args.output or config.paths.default_data_path)
        frame = generate_synthetic_market_data(
            days=config.demo.days,
            freq_minutes=config.demo.freq_minutes,
            seed=config.demo.seed,
        )
        target = save_dataframe(frame, output_path)
        LOGGER.info("Synthetic data saved to %s", target)
        print(target)
        return 0

    if args.command == "backtest":
        data_path = config.resolve_path(args.data or config.paths.default_data_path)
        raw_frame = load_market_data(data_path, config.data.timestamp_col)
        output_dir = None if args.output_dir is None else Path(args.output_dir)
        if output_dir is not None and not output_dir.is_absolute():
            output_dir = config.resolve_path(str(output_dir))

        artifacts = run_backtest(raw_frame, config, output_dir=output_dir)
        LOGGER.info("Backtest results saved to %s", artifacts.output_dir)
        print(artifacts.output_dir)
        print(artifacts.summary.to_string(index=False))
        return 0

    parser.print_help()
    return 1
