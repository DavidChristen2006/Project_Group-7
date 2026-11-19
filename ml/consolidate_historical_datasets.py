
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUNS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical_runs"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_datasets"
)


RUN_PREFIXES = [
    "AAPL_O_20220103_20221201_H21-63-126_S1000_K500_TC",
    "AAPL_O_20230403_20230601_H21-63-126_S1000_K500_TC",
    "AAPL_O_20230703_20231201_H21-63-126_S1000_K500_TC",
]


KEY_COLUMNS = [
    "Underlying RIC",
    "Decision Date",
    "Hedge Horizon Days",
    "Strategy",
]


def find_run(prefix):
    matches = [
        path
        for path in RUNS_DIR.glob(f"{prefix}*")
        if path.is_dir()
        and (path / "historical_ml_features.parquet").exists()
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one run for {prefix}, "
            f"found {len(matches)}."
        )

    return matches[0]


def consolidate_historical_datasets():
    frames = []
    cost_tags = set()
    schemas = []

    for prefix in RUN_PREFIXES:
        run_dir = find_run(prefix)

        cost_tag = run_dir.name.rsplit("_TC", 1)[-1]

        cost_tags.add(cost_tag)

        frame = pd.read_parquet(
            run_dir / "historical_ml_features.parquet"
        )

        required = KEY_COLUMNS + [
            "Outcome Date",
            "Final Return",
            "Maximum Drawdown",
            "Total Transaction Costs",
        ]

        missing = [
            column
            for column in required
            if column not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"{run_dir.name}: missing columns {missing}"
            )

        schemas.append(set(frame.columns))

        print(
            f"Loaded {len(frame)} observations "
            f"from {run_dir.name}"
        )

        frames.append(frame)

    if len(cost_tags) != 1:
        raise ValueError(
            "Incompatible transaction-cost configurations."
        )

    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError(
            "Historical ML feature schemas differ. "
            "Review the generating pipelines."
        )

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    combined["Decision Date"] = pd.to_datetime(
        combined["Decision Date"]
    )

    combined["Outcome Date"] = pd.to_datetime(
        combined["Outcome Date"]
    )

    if combined[KEY_COLUMNS].isna().any().any():
        raise ValueError(
            "Missing strategy identification values."
        )

    if combined[KEY_COLUMNS].duplicated().any():
        duplicates = combined.loc[
            combined[KEY_COLUMNS].duplicated(
                keep=False
            ),
            KEY_COLUMNS
        ]

        raise ValueError(
            "Duplicate strategy observations:\n"
            f"{duplicates.to_string(index=False)}"
        )

    if not (
        combined["Outcome Date"]
        > combined["Decision Date"]
    ).all():
        raise ValueError(
            "Found outcomes on or before decision dates."
        )

    combined = combined.sort_values(
        KEY_COLUMNS
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "aapl_historical_hedging_2022_2023_full.parquet"
    )

    combined.to_parquet(
        output_path,
        index=False
    )

    print("\n--- CONSOLIDATED DATASET ---")
    print(f"Observations: {len(combined)}")
    print(
        "Decision dates: "
        f"{combined['Decision Date'].nunique()}"
    )

    print("\nCoverage by horizon:")
    print(
        combined.groupby(
            "Hedge Horizon Days"
        )["Strategy"].count()
    )

    print(f"\nSaved to: {output_path}")

    return combined


if __name__ == "__main__":
    consolidate_historical_datasets()
