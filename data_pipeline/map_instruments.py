
"""Map ever-member RICs to LSEG reference and security identifiers.

Requires data_pipeline/identifier_mapping.py with map_instruments_batch().
Uses the configured default LSEG Platform session (ld.open_session()).
Never merges historical securities merely because they share identifiers.
"""

import argparse
import hashlib
from pathlib import Path

import lseg.data as ld
import pandas as pd

from data_pipeline.identifier_mapping import map_instruments_batch


ROOT = Path("data/processed/ever_member_universe")
MASTER_FILE = ROOT / "ever_member_instruments.parquet"
OUTPUT = ROOT / "instrument_mapping"
CACHE = OUTPUT / "cache" / "symbol_conversion_v2"
BATCH_SIZE = 20

MAPPING_COLUMNS = [
    "Underlying RIC",
    "Common Name",
    "Primary Issue RIC",
    "Organization PermID",
    "ISIN",
    "CUSIP",
    "SEDOL",
    "Issuer PermID",
    "Mapping Status",
]

OUTPUT_RENAMES = {
    "Primary Issue RIC": "Primary RIC (reference only)",
    "Organization PermID": "Organization ID (not security ID)",
}


def cache_name(rics):
    digest = hashlib.sha256(
        "\n".join(rics).encode("utf-8")
    ).hexdigest()[:16]

    return CACHE / f"batch_{digest}.parquet"


def validate_batch(frame, rics):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Mapping response is not a DataFrame.")

    missing = [
        column
        for column in MAPPING_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Mapping response is missing columns: {missing}"
        )

    result = frame[MAPPING_COLUMNS].copy()

    result["Underlying RIC"] = (
        result["Underlying RIC"]
        .astype("string")
        .str.strip()
    )

    if result["Underlying RIC"].isna().any():
        raise ValueError("Mapping contains missing RIC keys.")

    if result["Underlying RIC"].duplicated().any():
        raise ValueError("Mapping contains duplicate RIC keys.")

    if (
        len(result) != len(rics)
        or set(result["Underlying RIC"]) != set(rics)
    ):
        raise ValueError(
            "Mapping RICs differ from the requested batch."
        )

    return (
        result
        .set_index("Underlying RIC")
        .loc[rics]
        .reset_index()
    )


def get_batch(rics, refresh=False):
    path = cache_name(rics)

    if path.exists() and not refresh:
        try:
            result = validate_batch(
                pd.read_parquet(path),
                rics,
            )

            print(f"Cached: {path.name}")
            return result

        except Exception as error:
            print(
                f"Ignoring invalid cache {path.name}: {error}"
            )

    result = validate_batch(
        map_instruments_batch(rics),
        rics,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = path.with_name(
        f"{path.stem}.tmp.parquet"
    )

    result.to_parquet(
        tmp_path,
        index=False,
    )

    tmp_path.replace(path)

    print(
        f"Downloaded and cached: {len(result)} rows"
    )

    return result


def failed_mapping(ric, error):
    record = {
        column: pd.NA
        for column in MAPPING_COLUMNS
    }

    record["Underlying RIC"] = ric
    record["Mapping Status"] = "REQUEST_FAILED"

    failure = {
        "Underlying RIC": ric,
        "Error": str(error),
    }

    return pd.DataFrame([record]), [failure]


def collect_batch(rics, refresh=False):
    """Isolate problematic RICs instead of losing an entire batch."""

    try:
        return get_batch(rics, refresh=refresh), []

    except Exception as error:
        print(
            f"FAILED ({len(rics)} RICs): "
            f"{type(error).__name__}: {error}"
        )

        if len(rics) == 1:
            return failed_mapping(rics[0], error)

        midpoint = len(rics) // 2

        print(
            "Splitting failed batch to isolate "
            "the affected RICs."
        )

        left, left_failures = collect_batch(
            rics[:midpoint],
            refresh,
        )

        right, right_failures = collect_batch(
            rics[midpoint:],
            refresh,
        )

        combined = pd.concat(
            [left, right],
            ignore_index=True,
        )

        return combined, left_failures + right_failures


def review_flags(series):
    """Convert existing review flags safely to boolean."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    return (
        series
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0", "yes", "y"])
    )


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--limit",
        type=int,
        help="Pilot: first N exact RICs",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Map every RIC",
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download batches instead of using the v2 cache",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")

    master = pd.read_parquet(MASTER_FILE)

    if "Underlying RIC" not in master.columns:
        raise ValueError(
            "Input master is missing 'Underlying RIC'."
        )

    master = master.copy()

    master["Underlying RIC"] = (
        master["Underlying RIC"]
        .astype("string")
        .str.strip()
    )

    if (
        master["Underlying RIC"].isna().any()
        or master["Underlying RIC"].eq("").any()
    ):
        raise ValueError(
            "Missing or blank exact RICs in the input master."
        )

    if master["Underlying RIC"].duplicated().any():
        raise ValueError(
            "Duplicate exact RICs in the input master."
        )

    master = (
        master
        .sort_values("Underlying RIC")
        .reset_index(drop=True)
    )

    if args.limit is not None:
        master = master.head(args.limit).copy()

    rics = master["Underlying RIC"].tolist()

    destination = (
        OUTPUT
        if args.all
        else OUTPUT / "pilot"
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n--- INSTRUMENT IDENTIFIER MAPPING (V2) ---")

    print("RICs requested:", len(rics))

    print(
        "Reference fields: "
        "TR.CommonName, TR.PrimaryRIC, TR.OrganizationID"
    )

    print(
        "Security IDs: Symbol Conversion "
        "(ISIN, CUSIP, SEDOL, PermID)"
    )

    print("Session: configured default LSEG session")
    print("Cache:", CACHE)
    print("Output:", destination)

    frames = []
    failures = []

    ld.open_session()

    try:
        for offset in range(
            0,
            len(rics),
            BATCH_SIZE,
        ):
            batch = rics[
                offset:offset + BATCH_SIZE
            ]

            print(
                f"\nBatch {offset // BATCH_SIZE + 1}: "
                f"{offset + 1}-{offset + len(batch)}"
            )

            frame, batch_failures = collect_batch(
                batch,
                refresh=args.refresh,
            )

            frames.append(frame)
            failures.extend(batch_failures)

    finally:
        ld.close_session()

    mapped = pd.concat(
        frames,
        ignore_index=True,
    )

    mapped = validate_batch(
        mapped,
        rics,
    )

    mapped = mapped.rename(
        columns=OUTPUT_RENAMES
    )

    collisions = (
        set(mapped.columns) & set(master.columns)
    ) - {"Underlying RIC"}

    if collisions:
        raise ValueError(
            "Mapping columns already present in master: "
            f"{sorted(collisions)}"
        )

    result = master.merge(
        mapped,
        on="Underlying RIC",
        how="left",
        validate="one_to_one",
    )

    # Shared identifiers indicate potential aliases.
    # They never authorize merging historical securities.

    for identifier in ("ISIN", "CUSIP"):
        populated = result[identifier].notna()

        counts = (
            result.loc[populated]
            .groupby(identifier)["Underlying RIC"]
            .nunique()
        )

        shared_ids = counts.index[
            counts > 1
        ]

        result[
            f"Shared {identifier} Across RICs"
        ] = result[identifier].isin(shared_ids)

    if "Identifier Review Required" in result.columns:
        previous_flags = review_flags(
            result["Identifier Review Required"]
        )

    else:
        previous_flags = pd.Series(
            False,
            index=result.index,
        )

    result["Identifier Review Required"] = (
        previous_flags
        | result["Shared ISIN Across RICs"]
        | result["Shared CUSIP Across RICs"]
        | result["Mapping Status"].ne("IDENTIFIER_MAPPED")
    )

    # Save full mapping.
    result.to_parquet(
        destination / "instrument_mapping.parquet",
        index=False,
    )

    result.to_csv(
        destination / "instrument_mapping.csv",
        index=False,
    )

    # Save instruments requiring manual review.
    result.loc[
        result["Identifier Review Required"]
    ].to_csv(
        destination / "identifier_review.csv",
        index=False,
    )

    # Save failed LSEG requests.
    pd.DataFrame(
        failures,
        columns=[
            "Underlying RIC",
            "Error",
        ],
    ).to_csv(
        destination / "request_failures.csv",
        index=False,
    )

    print("\n--- MAPPING SUMMARY ---")

    print(
        result["Mapping Status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        "With ISIN:",
        int(result["ISIN"].notna().sum()),
    )

    print(
        "With CUSIP:",
        int(result["CUSIP"].notna().sum()),
    )

    print(
        "Flagged for review:",
        int(
            result["Identifier Review Required"].sum()
        ),
    )

    print(
        "Failed requests:",
        len(failures),
    )

    print("\nSample:")

    print(
        result[
            [
                "Underlying RIC",
                "Common Name",
                "ISIN",
                "CUSIP",
                "Mapping Status",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )

    print("\nSaved in:", destination)

    print(
        "No instrument aliases have been merged."
    )

    if not args.all:
        print(
            "Pilot output is separate; "
            "full-universe output remains untouched."
        )


if __name__ == "__main__":
    main()
