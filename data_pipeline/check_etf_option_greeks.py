
"""Read-only LSEG option Greek availability check.
"""

import lseg.data as ld


OPTION_RICS = [
    "SPYN192770000.U",  # Long 700 put
    "SPYN192762500.U",  # Short 625 put
    "SPYB192773500.U",  # Short 735 call
    "SPYB192781000.U",  # Long 810 call
]

FIELDS = [
    "TR.DELTA",
    "DELTA",
    "TR.IMPLIEDVOLATILITY",
]


def main():
    print("\n--- SPY OPTION GREEKS CHECK ---")

    ld.open_session()

    try:
        for field in FIELDS:

            print(f"\nField: {field}")

            try:
                result = ld.get_data(
                    universe=OPTION_RICS,
                    fields=[field],
                )

                if result is None or result.empty:
                    print("No data returned")
                    continue

                print(
                    result.to_string(
                        index=False
                    )
                )

            except Exception as exc:
                print(
                    f"{type(exc).__name__}: {exc}"
                )

    finally:
        ld.close_session()

    print("\nGreek availability check complete.")


if __name__ == "__main__":
    main()
