
"""Retrieve actual LSEG deltas for selected ETF option structures.

The caller must already have an open LSEG session.

Uses TR.DELTA. Missing or invalid Greeks are rejected
rather than replaced with assumed values.
"""

from __future__ import annotations

from math import isfinite

import pandas as pd


def calculate_net_deltas(
    structures: dict,
    leg_deltas: dict[str, float],
) -> dict[str, float]:
    """Calculate signed option-position delta per structure."""

    if not structures:
        raise ValueError("No option structures supplied")

    normalized = {
        str(ric).strip().upper(): float(delta)
        for ric, delta in leg_deltas.items()
    }

    result = {}

    for name, structure in structures.items():

        net = 0.0

        for leg in structure.legs:

            ric = leg.ric.strip().upper()

            if ric not in normalized:
                raise ValueError(
                    f"Missing LSEG delta for {ric}"
                )

            delta = normalized[ric]

            if (
                not isfinite(delta)
                or not -1.0 <= delta <= 1.0
            ):
                raise ValueError(
                    f"Invalid delta for {ric}: {delta}"
                )

            if leg.right == "put" and delta > 0.001:
                raise ValueError(
                    f"Unexpected positive put delta: {ric}"
                )

            if leg.right == "call" and delta < -0.001:
                raise ValueError(
                    f"Unexpected negative call delta: {ric}"
                )

            # Long leg: +delta
            # Short leg: -delta
            net += leg.side * delta

        if not isfinite(net) or net >= -0.01:
            raise ValueError(
                f"{name}: net delta {net:.4f} "
                "is not a usable bearish hedge"
            )

        result[name] = net

    return result


def retrieve_structure_option_deltas(
    structures: dict,
) -> tuple[dict[str, float], dict[str, float]]:
    """Request LSEG deltas and aggregate by option structure."""

    import lseg.data as ld

    rics = sorted({
        leg.ric
        for structure in structures.values()
        for leg in structure.legs
    })

    if not rics:
        raise ValueError(
            "No option contracts to query"
        )

    raw = ld.get_data(
        universe=rics,
        fields=["TR.DELTA"],
    )

    if (
        raw is None
        or raw.empty
        or not {
            "Instrument",
            "Delta",
        }.issubset(raw.columns)
    ):
        raise ValueError(
            "LSEG did not return Instrument "
            "and Delta fields"
        )

    work = raw[
        ["Instrument", "Delta"]
    ].copy()

    work["Instrument"] = (
        work["Instrument"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    work["Delta"] = pd.to_numeric(
        work["Delta"],
        errors="coerce",
    )

    if (
        work["Instrument"].isna().any()
        or work["Instrument"].duplicated().any()
    ):
        raise ValueError(
            "Missing or duplicate option identifiers"
        )

    if work["Delta"].isna().any():
        raise ValueError(
            "One or more option deltas are missing"
        )

    leg_deltas = dict(
        zip(
            work["Instrument"].astype(str),
            work["Delta"].astype(float),
        )
    )

    net_deltas = calculate_net_deltas(
        structures,
        leg_deltas,
    )

    return net_deltas, leg_deltas
