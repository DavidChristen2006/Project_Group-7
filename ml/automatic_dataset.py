
import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUNS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical_runs"
)

DATASETS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_datasets"
)

UNIQUE_COLUMNS = [
    "Underlying RIC",
    "Decision Date",
    "Hedge Horizon Days",
    "Strategy",
]


def discover_historical_runs(ric, cost_tag=None):
    if not RUNS_DIR.is_dir():
        raise FileNotFoundError(
            f"Historical runs directory not found: {RUNS_DIR}"
        )

    prefix = ric.replace(".", "_") + "_"
    paths = []

    for folder in sorted(RUNS_DIR.iterdir()):
        if not folder.is_dir():
            continue

        if not folder.name.startswith(prefix):
            continue

        if "_TC" not in folder.name:
            print(f"Skipping legacy run: {folder.name}")
            continue

        actual_tag = folder.name.rsplit("_TC", 1)[-1]

        if cost_tag is not None and actual_tag != cost_tag:
            continue

        path = folder / "historical_ml_features.parquet"

        if path.is_file():
            paths.append(path)

    return paths


def build_dataset(
    ric,
    start_date,
    end_date,
    cost_tag=None,
):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if end < start:
        raise ValueError(
            "End date must not precede start date."
        )

    paths = discover_historical_runs(
        ric,
        cost_tag=cost_tag,
    )

    if not paths:
        raise ValueError(
            f"No historical runs found for {ric}."
        )

    candidates = []

    required = set(
        UNIQUE_COLUMNS + ["Outcome Date"]
    )

    for path in paths:
        df = pd.read_parquet(path).copy()

        if not required.issubset(df.columns):
            raise ValueError(
                f"Missing required columns in {path}"
            )

        df["Decision Date"] = pd.to_datetime(
            df["Decision Date"]
        ).dt.normalize()

        df["Outcome Date"] = pd.to_datetime(
            df["Outcome Date"]
        ).dt.normalize()

        df = df.loc[
            (df["Underlying RIC"] == ric)
            & (df["Decision Date"] >= start)
            & (df["Decision Date"] <= end)
        ].copy()

        if df.empty:
            continue

        if df.duplicated(UNIQUE_COLUMNS).any():
            raise ValueError(
                f"Duplicate observations within {path}"
            )

        tag = path.parent.name.rsplit("_TC", 1)[-1]

        candidates.append({
            "path": path,
            "data": df,
            "tag": tag,
            "schema": set(df.columns),
        })

    if not candidates:
        raise ValueError(
            "No observations in the requested period."
        )

    tags = {item["tag"] for item in candidates}

    if len(tags) != 1:
        raise ValueError(
            "Multiple transaction-cost configurations found. "
            "Specify --cost-tag."
        )

    # Prioritize runs with more observations.
    # This puts the finalized 2022 run before
    # its smaller development runs.
    candidates.sort(
        key=lambda item: (
            -len(item["data"]),
            str(item["path"]),
        )
    )

    reference_schema = candidates[0]["schema"]

    frames = []
    selected_keys = set()

    for item in candidates:
        df = item["data"]
        path = item["path"]

        keys = pd.MultiIndex.from_frame(
            df[UNIQUE_COLUMNS]
        ).tolist()

        new_mask = [
            key not in selected_keys
            for key in keys
        ]

        new_df = df.loc[new_mask].copy()

        if new_df.empty:
            print(
                f"Skipping overlapping run: "
                f"{path.parent.name}"
            )
            continue

        if item["schema"] != reference_schema:
            raise ValueError(
                "Incompatible schema contains additional "
                "observations and cannot be silently skipped: "
                f"{path.parent.name}"
            )

        frames.append(new_df)

        new_keys = pd.MultiIndex.from_frame(
            new_df[UNIQUE_COLUMNS]
        ).tolist()

        selected_keys.update(new_keys)

        print(
            f"Loaded {len(new_df)} observations "
            f"from {path.parent.name}"
        )

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    if combined.duplicated(UNIQUE_COLUMNS).any():
        raise ValueError(
            "Duplicate observations remain."
        )

    if (
        combined["Outcome Date"]
        <= combined["Decision Date"]
    ).any():
        raise ValueError(
            "Invalid outcome dates."
        )

    combined = combined.sort_values(
        UNIQUE_COLUMNS
    ).reset_index(drop=True)

    DATASETS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_ric = ric.replace(".", "_")
    tag = next(iter(tags))

    output_path = (
        DATASETS_DIR
        / (
            f"{safe_ric}_{start:%Y%m%d}_"
            f"{end:%Y%m%d}_TC{tag}.parquet"
        )
    )

    combined.to_parquet(
        output_path,
        index=False
    )

    print("\n--- AUTOMATIC DATASET ---")
    print(f"Underlying: {ric}")
    print(f"Observations: {len(combined)}")
    print(
        "Decision dates: "
        f"{combined['Decision Date'].nunique()}"
    )
    print(f"Cost tag: {tag}")
    print(f"Saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ric",
        required=True,
    )

    parser.add_argument(
        "--start",
        required=True,
    )

    parser.add_argument(
        "--end",
        required=True,
    )

    parser.add_argument(
        "--cost-tag",
        default=None,
    )

    args = parser.parse_args()

    build_dataset(
        ric=args.ric,
        start_date=args.start,
        end_date=args.end,
        cost_tag=args.cost_tag,
    )


if __name__ == "__main__":
    main()
