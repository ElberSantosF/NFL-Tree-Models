"""Command line interface of the workbench.

    python -m nfl_trees run configs/decision_tree.yaml
    python -m nfl_trees run --all
    python -m nfl_trees models
    python -m nfl_trees configs
    python -m nfl_trees leaderboard
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ExperimentConfig, list_configs
from .paths import CONFIG_DIR


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def cmd_run(args: argparse.Namespace) -> int:
    from .experiment import run

    refs = list(args.configs)
    if args.all:
        if refs:
            print("pass either a list of configs or --all, not both")
            return 1
        refs = [str(CONFIG_DIR / f"{name}.yaml") for name in list_configs()]
        if not refs:
            print(f"no configs in {CONFIG_DIR} -- create an experiment YAML first")
            return 1
    if not refs:
        print("nothing to run: pass a config or use --all")
        return 1

    failures = 0
    for ref in refs:
        try:
            result = run(ExperimentConfig.load(ref), save=not args.no_save)
            print(result.summary())
        except Exception as exc:  # one broken config must not stop the batch
            failures += 1
            logging.error("failed on '%s': %s", ref, exc)
    return 1 if failures else 0


def cmd_models(args: argparse.Namespace) -> int:
    from .models import list_models

    specs = list_models()
    if not specs:
        print(
            "empty catalog: no model registered yet.\n"
            "Register the first one with @register in src/nfl_trees/models.py "
            "(the module itself carries the example), then create its config in configs/."
        )
        return 0

    print(f"{'model':<24} {'ok':<4} description")
    print("-" * 96)
    for spec in specs:
        flag = "yes" if spec.available else "no"
        print(f"{spec.name:<24} {flag:<4} {spec.summary}")
    return 0


def cmd_configs(args: argparse.Namespace) -> int:
    names = list_configs()
    if not names:
        print(f"no configs in {CONFIG_DIR}")
        return 0
    for name in names:
        try:
            cfg = ExperimentConfig.load(name)
            print(f"{name:<32} {cfg.model.type:<24} {cfg.task:<15} {cfg.data.target}")
        except Exception as exc:
            print(f"{name:<32} INVALID: {exc}")
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    from .experiment import load_leaderboard

    df = load_leaderboard()
    if df.empty:
        print("empty leaderboard: run an experiment first")
        return 0
    cols = ["experiment", "model", "target", "primary_metric", "primary_value", "n_test"]
    print(df[cols].to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = df.to_markdown(index=False)
        except ImportError:  # tabulate not installed
            text = df.to_csv(index=False)
        args.output.write_text(text, encoding="utf-8")
        print(f"\nsaved to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nfl-trees", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="log at DEBUG level")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one or more experiments")
    p_run.add_argument("configs", nargs="*", help="config paths or names")
    p_run.add_argument("--all", action="store_true", help="run every config in configs/")
    p_run.add_argument("--no-save", action="store_true", help="do not write to results/")
    p_run.set_defaults(func=cmd_run)

    p_models = sub.add_parser("models", help="list the model catalog")
    p_models.set_defaults(func=cmd_models)

    p_configs = sub.add_parser("configs", help="list the available experiments")
    p_configs.set_defaults(func=cmd_configs)

    p_lb = sub.add_parser("leaderboard", help="show the run comparison")
    p_lb.add_argument("--output", type=Path, help="write it as markdown")
    p_lb.set_defaults(func=cmd_leaderboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
