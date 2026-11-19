import lseg.data as ld
import pandas as pd


def get_delayed_streaming_snapshot(
    universe,
    fields
):
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