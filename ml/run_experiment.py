"""ml/run_experiment.py: hedge research implementation.

Functions are documented at their point of definition.
Historical outputs are research estimates, not executable orders.
"""


import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ml.automatic_dataset import (
    PROJECT_ROOT,
    build_dataset,
)


def run_command(command):
    """Perform the run command operation for this research workflow.

    Args:
        command: Input to the run_command operation.
    """
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def run_experiment(
    ric,
    start_date,
    end_date,
    test_start,
    cost_tag=None,
    download_start=None,
    download_end=None,
    shares=1000,
    horizons=None,
):
    """Perform the run experiment operation for this research workflow.

    Args:
        ric: Input to the run_experiment operation.
        start_date: Input to the run_experiment operation.
        end_date: Input to the run_experiment operation.
        test_start: Input to the run_experiment operation.
        cost_tag: Input to the run_experiment operation.
        download_start: Input to the run_experiment operation.
        download_end: Input to the run_experiment operation.
        shares: Input to the run_experiment operation.
        horizons: Input to the run_experiment operation.

    Returns:
        Computed result or status for the caller.
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    test = pd.Timestamp(test_start).normalize()

    if not start < test <= end:
        raise ValueError(
            "Required: start < test_start <= end."
        )

    if (download_start is None) != (download_end is None):
        raise ValueError(
            "Specify both --download-start and "
            "--download-end, or neither."
        )

    if horizons is None:
        horizons = [21, 63, 126]

    if shares <= 0 or any(h <= 0 for h in horizons):
        raise ValueError(
            "Shares and horizons must be positive."
        )

    print("\n--- EXPERIMENT CONFIGURATION ---", flush=True)
    print(f"Underlying: {ric}", flush=True)
    print(f"Dataset start: {start.date()}", flush=True)
    print(f"Dataset end: {end.date()}", flush=True)
    print(f"Test start: {test.date()}", flush=True)

    # Stage 1: Download new historical observations
    # only if a download range was supplied.
    if download_start is not None:
        dl_start = pd.Timestamp(
            download_start
        ).normalize()

        dl_end = pd.Timestamp(
            download_end
        ).normalize()

        if not start <= dl_start <= dl_end <= end:
            raise ValueError(
                "Download period must be inside "
                "the selected dataset period."
            )

        print(
            "\n--- STAGE 1: HISTORICAL LSEG BACKTEST ---",
            flush=True
        )

        command = [
            sys.executable,
            str(PROJECT_ROOT / "run_historical_pipeline.py"),
            "--ric", ric,
            "--start", str(dl_start.date()),
            "--end", str(dl_end.date()),
            "--shares", str(shares),
            "--horizons",
            *[str(h) for h in horizons],
        ]

        run_command(command)

    # Stage 2: Consolidate compatible historical runs.
    print(
        "\n--- STAGE 2: DATASET CONSOLIDATION ---",
        flush=True
    )

    dataset_path = Path(
        build_dataset(
            ric=ric,
            start_date=start,
            end_date=end,
            cost_tag=cost_tag,
        )
    ).resolve()

    # Stage 3: Perform walk-forward ML evaluation.
    experiment_name = (
        f"{dataset_path.stem}"
        f"_TEST{test:%Y%m%d}"
    )

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ml_evaluations"
        / experiment_name
    )

    print(
        "\n--- STAGE 3: WALK-FORWARD ML ---",
        flush=True
    )

    command = [
        sys.executable,
        "-m",
        "ml.walk_forward_validation",
        "--dataset",
        str(dataset_path),
        "--test-start",
        str(test.date()),
        "--output-dir",
        str(output_dir),
    ]

    run_command(command)

    print("\n--- EXPERIMENT COMPLETED ---")
    print(f"Dataset: {dataset_path}")
    print(f"Evaluation: {output_dir}")

    return dataset_path, output_dir


def main():
    """Run the module entry point and its workflow.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run historical hedging data collection, "
            "consolidation and walk-forward ML."
        )
    )

    parser.add_argument(
        "--ric",
        required=True
    )

    parser.add_argument(
        "--start",
        required=True
    )

    parser.add_argument(
        "--end",
        required=True
    )

    parser.add_argument(
        "--test-start",
        required=True
    )

    parser.add_argument(
        "--cost-tag",
        default=None
    )

    parser.add_argument(
        "--download-start",
        default=None
    )

    parser.add_argument(
        "--download-end",
        default=None
    )

    parser.add_argument(
        "--shares",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[21, 63, 126]
    )

    args = parser.parse_args()

    run_experiment(
        ric=args.ric,
        start_date=args.start,
        end_date=args.end,
        test_start=args.test_start,
        cost_tag=args.cost_tag,
        download_start=args.download_start,
        download_end=args.download_end,
        shares=args.shares,
        horizons=args.horizons,
    )


if __name__ == "__main__":
    main()
