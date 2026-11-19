
"""Locate NVIDIA records across the existing historical data pipeline.

Run:
    py -m data_pipeline.diagnose_nvda_pipeline

Read-only: no API calls, no changes to existing datasets.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"

FILES = {
    "Historical membership grid": (
        ROOT
        / "historical_universe/equity_collection_candidate"
        / "equity_membership_grid_candidate.parquet"
    ),
    "Historical ML features": (
        ROOT
        / "research_features/equity_features_v1"
        / "point_in_time_equity_features.parquet"
    ),
    "Forward 63-day outcomes": (
        ROOT
        / "research_outcomes/equity_forward_63d_v1"
        / "point_in_time_equity_forward_outcomes.parquet"
    ),
}

PRICE_ROOT = ROOT / "market_history_v2" / "series"

SEARCH_TEXT = "NVDA|NVIDIA"


def inspect_dataset(name: str, path: Path) -> None:
    print(f"\n--- {name.upper()} ---")

    if not path.is_file():
        print("FILE NOT FOUND:", path)
        return

    frame = pd.read_parquet(path)

    print("Total rows:", len(frame))
    print("Columns:", frame.columns.tolist())

    text_columns = [
        column
        for column in frame.columns
        if (
            pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
            or isinstance(
                frame[column].dtype,
                pd.CategoricalDtype,
            )
        )
    ]

    matches = pd.Series(
        False,
        index=frame.index,
    )

    matching_columns = []

    for column in text_columns:
        mask = (
            frame[column]
            .astype("string")
            .str.contains(
                SEARCH_TEXT,
                case=False,
                regex=True,
                na=False,
            )
        )

        if mask.any():
            matching_columns.append(column)
            matches |= mask

    found = frame.loc[matches].copy()

    print("NVIDIA-related rows:", len(found))
    print("Matching columns:", matching_columns)

    if found.empty:
        return

    useful_columns = []

    for column in frame.columns:
        label = str(column).lower()

        if (
            column in matching_columns
            or "date" in label
            or "membership" in label
            or "candidate" in label
            or "status" in label
        ):
            useful_columns.append(column)

    useful_columns = useful_columns[:15]

    print("\nFirst matching records:")
    print(
        found[useful_columns]
        .head(5)
        .to_string(index=False)
    )

    if "Decision Date" in found.columns:
        dates = pd.to_datetime(
            found["Decision Date"],
            errors="coerce",
        )

        if dates.notna().any():
            print("First decision date:", dates.min().date())
            print("Last decision date:", dates.max().date())

            requested = found.loc[
                dates.dt.normalize().eq(
                    pd.Timestamp("2024-12-02")
                )
            ]

            print(
                "Matching rows on 2024-12-02:",
                len(requested),
            )

            if not requested.empty:
                print(
                    requested[useful_columns]
                    .head(5)
                    .to_string(index=False)
                )


def inspect_price_cache() -> None:
    print("\n--- NVIDIA PRICE CACHE ---")

    if not PRICE_ROOT.is_dir():
        print("Price directory not found:", PRICE_ROOT)
        return

    folders = [
        path
        for path in PRICE_ROOT.iterdir()
        if (
            "NVDA" in path.name.upper()
            or "NVIDIA" in path.name.upper()
        )
    ]

    print("Matching price folders:", len(folders))

    for folder in folders:
        print("\nFolder:", folder.name)

        files = (
            list(folder.rglob("*.parquet"))
            if folder.is_dir()
            else [folder]
        )

        print("Parquet files:", len(files))

        for path in files[:10]:
            print(" ", path.relative_to(PRICE_ROOT))

    if not folders:
        print(
            "No NVIDIA-named price folder found. "
            "This does not exclude an alternative identifier."
        )


def main() -> None:
    print("\n--- NVIDIA PIPELINE DIAGNOSTIC ---")

    for name, path in FILES.items():
        inspect_dataset(name, path)

    inspect_price_cache()

    print(
        "\nRead-only diagnostic complete."
    )


if __name__ == "__main__":
    main()
