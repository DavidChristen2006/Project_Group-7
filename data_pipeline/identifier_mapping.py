
import pandas as pd
import lseg.data as ld

from lseg.data.content import symbol_conversion


REFERENCE_FIELDS = [
    "TR.CommonName",
    "TR.PrimaryRIC",
    "TR.OrganizationID",
]


def map_instruments_batch(rics):
    rics = list(dict.fromkeys(str(ric).strip() for ric in rics))

    if not rics:
        raise ValueError("No RICs supplied.")

    requested = pd.DataFrame({"Underlying RIC": rics})

    # 1. Retrieve reference information.
    reference = ld.get_data(
        universe=rics,
        fields=REFERENCE_FIELDS,
    )

    reference = reference.rename(
        columns={
            "Instrument": "Underlying RIC",
            "Company Common Name": "Common Name",
        }
    )

    reference_columns = [
        "Underlying RIC",
        "Common Name",
        "Primary Issue RIC",
        "Organization PermID",
    ]

    for column in reference_columns:
        if column not in reference.columns:
            reference[column] = pd.NA

    reference = reference[reference_columns].copy()

    if reference["Underlying RIC"].isna().any():
        raise ValueError("Reference response has missing RIC keys.")

    if reference["Underlying RIC"].duplicated().any():
        raise ValueError("Duplicate reference RICs returned.")

    # 2. Retrieve instrument identifiers.
    response = symbol_conversion.Definition(
        symbols=rics,
        from_symbol_type=symbol_conversion.SymbolTypes.RIC,
        to_symbol_types=[
            symbol_conversion.SymbolTypes.ISIN,
            symbol_conversion.SymbolTypes.CUSIP,
            symbol_conversion.SymbolTypes.SEDOL,
            symbol_conversion.SymbolTypes.OA_PERM_ID,
        ],
    ).get_data()

    identifiers = response.data.df.copy()

    # The requested RIC is the DataFrame index.
    identifiers.index.name = "Underlying RIC"
    identifiers = identifiers.reset_index()

    identifiers = identifiers.rename(
        columns={
            "IssueISIN": "ISIN",
            "IssuerOAPermID": "Issuer PermID",
        }
    )

    identifier_columns = [
        "Underlying RIC",
        "ISIN",
        "CUSIP",
        "SEDOL",
        "Issuer PermID",
    ]

    for column in identifier_columns:
        if column not in identifiers.columns:
            identifiers[column] = pd.NA

    identifiers = identifiers[identifier_columns].copy()

    if identifiers["Underlying RIC"].duplicated().any():
        raise ValueError("Ambiguous duplicate conversion results.")

    # 3. Merge on the requested RIC, never row position.
    result = requested.merge(
        reference,
        on="Underlying RIC",
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        identifiers,
        on="Underlying RIC",
        how="left",
        validate="one_to_one",
    )

    # 4. Normalize identifier strings.
    for column in ["ISIN", "CUSIP", "SEDOL"]:
        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    # 5. Classify mapping quality.
    has_identifier = (
        result["ISIN"].notna()
        | result["CUSIP"].notna()
    )

    has_reference = (
        result["Common Name"].notna()
        | result["Primary Issue RIC"].notna()
        | result["Organization PermID"].notna()
        | result["Issuer PermID"].notna()
    )

    result["Mapping Status"] = "UNRESOLVED"

    result.loc[
        has_reference, "Mapping Status"
    ] = "REFERENCE_ONLY"

    result.loc[
        has_identifier, "Mapping Status"
    ] = "IDENTIFIER_MAPPED"

    # 6. Integrity checks.
    if len(result) != len(rics):
        raise RuntimeError("Mapping changed the instrument count.")

    if result["Underlying RIC"].duplicated().any():
        raise RuntimeError("Duplicate instruments in final mapping.")

    return result


def main():
    test_rics = [
        "AAPL.OQ",
        "AAPL.O",
        "MSFT.O",
        "IBM.N",
        "AABA.OQ^J19",
    ]

    ld.open_session()

    try:
        result = map_instruments_batch(test_rics)

        print("\n--- IDENTIFIER MAPPING TEST ---")
        print(result.to_string(index=False))

        print("\n--- MAPPING SUMMARY ---")
        print(result["Mapping Status"].value_counts())

        print("\nWith ISIN:", result["ISIN"].notna().sum())
        print("With CUSIP:", result["CUSIP"].notna().sum())

    finally:
        ld.close_session()


if __name__ == "__main__":
    main()
