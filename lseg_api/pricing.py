"""lseg_api/pricing.py: hedge research implementation.

Functions are documented at their point of definition.
Historical outputs are research estimates, not executable orders.
"""

import lseg.data as ld
import pandas as pd


def get_delayed_streaming_snapshot(
    universe,
    fields
):
    """Get delayed streaming snapshot from the current context.

    Args:
        universe: Input to the get_delayed_streaming_snapshot operation.
        fields: Input to the get_delayed_streaming_snapshot operation.

    Returns:
        Computed result or status for the caller.
    """
    if isinstance(universe, str):
        universe = [universe]

    delayed_universe = [
        ric if ric.startswith("/") else f"/{ric}"
        for ric in universe
    ]

    stream = ld.content.pricing.Definition(
        universe=delayed_universe,
        fields=fields
    ).get_stream()

    try:
        stream.open(
            with_updates=False
        )

        data = stream.get_snapshot()

    finally:
        stream.close()

    if data is None:
        return pd.DataFrame()

    return data.copy()