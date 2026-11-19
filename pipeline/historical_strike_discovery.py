
from decimal import (
    Decimal,
    ROUND_CEILING,
    ROUND_FLOOR,
)

import math


def generate_extended_strikes(
    spot,
    minimum_strike_pct=0.65,
    maximum_strike_pct=1.25,
    strike_step=5.0,
    max_strikes=60,
):
    """
    Generate a strike grid within a percentage
    range of the underlying stock price.

    The returned strikes are multiples of
    strike_step and lie within the requested
    price range.

    This function does not query LSEG.
    """

    values = [
        spot,
        minimum_strike_pct,
        maximum_strike_pct,
        strike_step,
    ]

    if not all(
        isinstance(value, (int, float))
        and math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "Strike parameters must be finite numbers."
        )

    if spot <= 0:
        raise ValueError(
            "Stock price must be positive."
        )

    if (
        minimum_strike_pct <= 0
        or maximum_strike_pct <= minimum_strike_pct
    ):
        raise ValueError(
            "Invalid strike percentage range."
        )

    if strike_step <= 0:
        raise ValueError(
            "Strike step must be positive."
        )

    if (
        not isinstance(max_strikes, int)
        or isinstance(max_strikes, bool)
        or max_strikes <= 0
    ):
        raise ValueError(
            "max_strikes must be a positive integer."
        )

    spot_decimal = Decimal(str(spot))

    lower = (
        spot_decimal
        * Decimal(str(minimum_strike_pct))
    )

    upper = (
        spot_decimal
        * Decimal(str(maximum_strike_pct))
    )

    step = Decimal(str(strike_step))

    first_index = int(
        (lower / step).to_integral_value(
            rounding=ROUND_CEILING
        )
    )

    last_index = int(
        (upper / step).to_integral_value(
            rounding=ROUND_FLOOR
        )
    )

    count = max(
        0,
        last_index - first_index + 1,
    )

    if count > max_strikes:
        raise ValueError(
            f"Strike grid contains {count} strikes, "
            f"exceeding the limit of {max_strikes}. "
            "Increase strike_step or narrow the range."
        )

    strikes = [
        float(step * index)
        for index in range(
            first_index,
            last_index + 1,
        )
    ]

    return strikes
