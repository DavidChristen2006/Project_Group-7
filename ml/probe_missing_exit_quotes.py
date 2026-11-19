
from pathlib import Path

import pandas as pd
import lseg.data as ld


RUN_NAME = (
    "AAPL_O_20150102_20150601_H21_S1000_K500_"
    "TC00a56a790ea1_EXP6500-12500_STEP5000_"
    "CFG7d3bc7d1e923"
)

ROOT = Path("data/processed")

inventory = pd.read_parquet(
    ROOT / "candidate_inventories" / RUN_NAME
    / "historical_candidates.parquet"
)

coverage = pd.read_parquet(
    ROOT / "candidate_backtests" / RUN_NAME
    / "candidate_coverage.parquet"
)


# Candidate IDs may repeat across decision dates
# or hedge horizons. Use the full candidate identity.

KEYS = [
    "Underlying RIC",
    "Decision Date",
    "Hedge Horizon Days",
    "Candidate ID",
]

for frame in (inventory, coverage):
    frame["Candidate ID"] = (
        frame["Candidate ID"].astype(str)
    )

    frame["Decision Date"] = pd.to_datetime(
        frame["Decision Date"]
    ).dt.normalize()

    frame["Hedge Horizon Days"] = (
        pd.to_numeric(
            frame["Hedge Horizon Days"],
            errors="raise",
        ).astype(int)
    )

missing = [
    col for col in KEYS
    if col not in inventory.columns
    or col not in coverage.columns
]

if missing:
    raise ValueError(
        f"Missing merge columns: {missing}"
    )

if inventory.duplicated(KEYS).any():
    raise ValueError(
        "Duplicate full candidate identities "
        "in inventory."
    )

if coverage.duplicated(KEYS).any():
    raise ValueError(
        "Duplicate full candidate identities "
        "in coverage."
    )

incomplete = coverage.loc[
    coverage["Status"] == "Incomplete"
].copy()

merged = incomplete.merge(
    inventory,
    on=KEYS,
    how="left",
    suffixes=("_coverage", ""),
    validate="one_to_one",
    indicator=True,
)

if not merged["_merge"].eq("both").all():
    raise ValueError(
        "Some incomplete candidates could not "
        "be matched to the inventory."
    )

print(
    "Incomplete candidates matched:",
    len(merged),
)


rics = set()

for _, row in merged.iterrows():
    reason = str(row["Reason"])

    if "Short Call" in reason:
        rics.add(str(row["Short Call RIC"]))

    if "Short Put" in reason:
        rics.add(str(row["Short Put RIC"]))


print("\n--- CONTRACTS TO RECHECK ---")

for ric in sorted(rics):
    print(ric)


ld.open_session()

try:
    for ric in sorted(rics):
        print(f"\n--- {ric} ---")

        try:
            history = ld.get_history(
                universe=ric,
                fields=["BID", "ASK"],
                interval="daily",
                start="2015-05-27",
                end="2015-06-08",
            )

            if history is None or history.empty:
                print("No history returned.")
                continue

            if isinstance(history.columns, pd.MultiIndex):
                for level in range(
                    history.columns.nlevels
                ):
                    if ric in history.columns.get_level_values(
                        level
                    ):
                        history = history.xs(
                            ric,
                            axis=1,
                            level=level,
                        )
                        break

            history.index = pd.to_datetime(
                history.index
            ).normalize()

            print(history.to_string())

            target = pd.Timestamp("2015-06-02")

            if target in history.index:
                print(
                    "Exit-date observation found:",
                    history.loc[target].to_dict(),
                )
            else:
                print(
                    "Exit-date observation not returned."
                )

        except Exception as exc:
            print(
                type(exc).__name__,
                str(exc),
            )

finally:
    ld.close_session()


print("\nNo files modified.")
