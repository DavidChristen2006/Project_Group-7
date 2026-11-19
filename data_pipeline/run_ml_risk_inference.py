
"""Generate 63-session research forecasts from point-in-time features.

Uses fitted research models; no LSEG calls or hedge recommendations.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_pipeline.train_equity_63d_baseline import FEATURES, KEY


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE = (
    ROOT
    / "data/processed/research_features/equity_features_v1"
    / "point_in_time_equity_features.parquet"
)

REGRESSION_DIR = (
    ROOT
    / "data/processed/ml_baselines/equity_63d_suite_v1"
    / "full/selected_models"
)

CLASSIFICATION_DIR = (
    ROOT
    / "data/processed/ml_baselines/equity_63d_classifier_suite_v1"
    / "models"
)

DEFAULT_OUTPUT_DIR = (
    ROOT / "data/processed/ml_inference/equity_63d_v1"
)

REGRESSION_TARGETS = {
    "pred_max_loss_from_entry_63d":
        "target_max_loss_from_entry_63d",
    "pred_max_drawdown_63d":
        "target_max_drawdown_63d",
    "pred_realized_vol_63d":
        "target_realized_vol_63d",
}

CLASSIFICATION_TARGETS = {
    10: "loss_at_least_10pct_63d",
    15: "loss_at_least_15pct_63d",
    20: "loss_at_least_20pct_63d",
}

HISTORICAL_RIC_ALIASES = {
    "NVDA.O": "NVDA.OQ",
}


def boolean_mask(
    series: pd.Series,
    name: str,
) -> pd.Series:

    values = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    result = values.map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    })

    if result.isna().any():
        raise ValueError(
            f"{name} has invalid boolean values"
        )

    return result.astype(bool)


def load_source(
    source: Path,
    membership_policy: str,
) -> pd.DataFrame:

    if not source.is_file():
        raise FileNotFoundError(source)

    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)

    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)

    else:
        raise ValueError(
            "Source must be .csv or .parquet"
        )

    missing = (
        set(KEY + FEATURES)
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"Missing feature columns: {sorted(missing)}"
        )

    frame = frame.copy()

    frame["Decision Date"] = pd.to_datetime(
        frame["Decision Date"],
        errors="raise",
    ).dt.normalize()

    frame["Underlying RIC"] = (
        frame["Underlying RIC"]
        .astype("string")
        .str.strip()
    )

    if (
        frame[KEY].isna().any().any()
        or frame["Underlying RIC"].eq("").any()
        or frame.duplicated(KEY).any()
    ):
        raise ValueError(
            "Missing or duplicate decision-date/RIC keys"
        )

    if "Feature Candidate" in frame.columns:
        frame = frame.loc[
            boolean_mask(
                frame["Feature Candidate"],
                "Feature Candidate",
            )
        ].copy()

    if membership_policy == "observed":

        if "Membership Source" not in frame.columns:
            raise ValueError(
                "Missing Membership Source. For live "
                "feature inputs use --membership-policy all."
            )

        frame = frame.loc[
            frame["Membership Source"].eq(
                "OBSERVED_SNAPSHOT_MATCH"
            )
        ].copy()

    if frame.empty:
        raise ValueError(
            "No eligible feature observations"
        )

    for column in FEATURES:

        original = frame[column]

        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        if (
            original.notna()
            & converted.isna()
        ).any():
            raise ValueError(
                f"Non-numeric values in {column}"
            )

        frame[column] = converted.replace(
            [np.inf, -np.inf],
            np.nan,
        )

    if frame[FEATURES].isna().all(
        axis=1
    ).any():
        raise ValueError(
            "At least one row has no usable predictors"
        )

    return frame


def select_rows(
    frame: pd.DataFrame,
    decision_date: str,
    ric: str | None,
) -> pd.DataFrame:

    if ric is not None:
        requested_ric = ric.strip().upper()

        historical_ric = HISTORICAL_RIC_ALIASES.get(
            requested_ric,
            requested_ric,
        )

        frame = frame.loc[
            frame["Underlying RIC"]
            .str.upper()
            .eq(historical_ric)
        ]

        if frame.empty:
            raise ValueError(
                f"No eligible historical features for "
                f"{requested_ric} "
                f"(resolved as {historical_ric})"
            )

        if historical_ric != requested_ric:
            print(
                f"Historical RIC mapping: "
                f"{requested_ric} -> {historical_ric}"
            )

    chosen_date = (
        frame["Decision Date"].max()
        if decision_date.lower() == "latest"
        else pd.Timestamp(
            decision_date
        ).normalize()
    )

    selected = frame.loc[
        frame["Decision Date"].eq(chosen_date)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No eligible observations on "
            f"{chosen_date.date()}"
        )

    return selected.sort_values(
        KEY
    ).reset_index(drop=True)


def load_model(path: Path):

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing fitted model: {path}"
        )

    # Only load model artifacts created in our own project.
    model = joblib.load(path)

    names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if (
        names is not None
        and list(names) != FEATURES
    ):
        raise ValueError(
            f"Feature order mismatch: {path}"
        )

    return model


def validate_predictions(
    predictions,
    count: int,
    name: str,
) -> np.ndarray:

    result = np.asarray(
        predictions,
        dtype=float,
    ).reshape(-1)

    if (
        len(result) != count
        or not np.isfinite(result).all()
    ):
        raise ValueError(
            f"Invalid model predictions: {name}"
        )

    return result


def project_nested_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    """Project probabilities onto P10 >= P15 >= P20.

    Equal-weight, least-squares isotonic projection.
    This ensures logical ordering, not calibration.
    """

    values = np.asarray(
        probabilities,
        dtype=float,
    )

    if (
        values.ndim != 2
        or values.shape[1] != 3
    ):
        raise ValueError(
            "Expected three probability columns"
        )

    if (
        not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
    ):
        raise ValueError(
            "Probabilities must be within [0, 1]"
        )

    adjusted = np.empty_like(values)

    for i, row in enumerate(values):

        blocks = []

        for probability in row:

            blocks.append(
                [float(probability), 1]
            )

            while (
                len(blocks) > 1
                and blocks[-2][0] < blocks[-1][0]
            ):
                right_mean, right_weight = (
                    blocks.pop()
                )

                left_mean, left_weight = (
                    blocks.pop()
                )

                weight = (
                    left_weight + right_weight
                )

                mean = (
                    left_mean * left_weight
                    + right_mean * right_weight
                ) / weight

                blocks.append(
                    [mean, weight]
                )

        adjusted[i] = [
            mean
            for mean, weight in blocks
            for _ in range(weight)
        ]

    if (
        (adjusted[:, 0] + 1e-12 < adjusted[:, 1]).any()
        or (adjusted[:, 1] + 1e-12 < adjusted[:, 2]).any()
    ):
        raise AssertionError(
            "Probability ordering correction failed"
        )

    return adjusted


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
    )

    parser.add_argument(
        "--date",
        default="latest",
    )

    parser.add_argument(
        "--ric",
        default=None,
    )

    parser.add_argument(
        "--membership-policy",
        choices=("observed", "all"),
        default="observed",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    source = args.source.resolve()

    frame = load_source(
        source,
        args.membership_policy,
    )

    chosen = select_rows(
        frame,
        args.date,
        args.ric,
    )

    x = chosen[FEATURES]

    output_columns = KEY.copy()

    if "Membership Source" in chosen.columns:
        output_columns.append(
            "Membership Source"
        )

    output = chosen[
        output_columns
    ].copy()

    model_files = {}

    # Three continuous risk forecasts.
    for output_name, target in REGRESSION_TARGETS.items():

        path = (
            REGRESSION_DIR
            / f"{target}.joblib"
        )

        model = load_model(path)

        predictions = validate_predictions(
            model.predict(x),
            len(x),
            target,
        )

        if "drawdown" in target:
            predictions = np.minimum(
                predictions,
                0.0,
            )
        else:
            predictions = np.maximum(
                predictions,
                0.0,
            )

        output[output_name] = predictions

        model_files[target] = str(path)

    # Three independently trained breach probabilities.
    raw_probability_columns = []

    for threshold, target in CLASSIFICATION_TARGETS.items():

        path = (
            CLASSIFICATION_DIR
            / f"{target}.joblib"
        )

        model = load_model(path)

        if list(model.classes_) != [0, 1]:
            raise ValueError(
                f"Invalid classifier classes: {target}"
            )

        probability = validate_predictions(
            model.predict_proba(x)[:, 1],
            len(x),
            target,
        )

        if (
            (probability < 0).any()
            or (probability > 1).any()
        ):
            raise ValueError(
                f"Probability out of bounds: {target}"
            )

        output[
            f"raw_prob_loss_ge_{threshold}pct_63d"
        ] = probability

        raw_probability_columns.append(
            probability
        )

        model_files[target] = str(path)

    raw = np.column_stack(
        raw_probability_columns
    )

    violations = (
        (raw[:, 0] + 1e-12 < raw[:, 1])
        | (raw[:, 1] + 1e-12 < raw[:, 2])
    )

    corrected = project_nested_probabilities(
        raw
    )

    for i, threshold in enumerate(
        CLASSIFICATION_TARGETS
    ):
        output[
            f"prob_loss_ge_{threshold}pct_63d"
        ] = corrected[:, i]

    output[
        "probability_order_corrected"
    ] = violations

    date_string = (
        chosen["Decision Date"]
        .iloc[0]
        .date()
        .isoformat()
    )

    ric_suffix = (
        "_"
        + re.sub(
            r"[^A-Za-z0-9_.-]",
            "_",
            args.ric,
        )
        if args.ric
        else ""
    )

    destination = args.output_dir.resolve()

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        destination
        / f"risk_forecasts_{date_string}{ric_suffix}.csv"
    )

    manifest_file = (
        destination
        / f"risk_forecasts_{date_string}{ric_suffix}_manifest.json"
    )

    output.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "research_only": True,
        "decision_date": date_string,
        "rows": len(output),
        "feature_count": len(FEATURES),
        "feature_columns": FEATURES,
        "membership_policy": args.membership_policy,
        "feature_source": str(source),
        "model_artifacts": model_files,
        "raw_probability_order_violations": int(
            violations.sum()
        ),
        "correction": (
            "Equal-weight least-squares isotonic projection"
        ),
        "limitations": [
            "These probabilities are not independently calibrated.",
            "Models were selected using limited historical validation.",
            "The historical test period was already examined.",
            "No option payoffs or transaction costs are calculated.",
            "Live features must match the original 30 definitions.",
            "No future outcome targets are used for prediction.",
        ],
    }

    manifest_file.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n--- 63-DAY ML RISK INFERENCE ---"
    )

    print(
        "Decision date:",
        date_string,
    )

    print(
        "Eligible securities:",
        len(output),
    )

    print(
        "Loaded regression models:",
        len(REGRESSION_TARGETS),
    )

    print(
        "Loaded classification models:",
        len(CLASSIFICATION_TARGETS),
    )

    print(
        "Original probability-order violations:",
        int(violations.sum()),
    )

    print(
        "Violations after correction: 0"
    )

    display_columns = [
        "Underlying RIC",
        "pred_max_loss_from_entry_63d",
        "pred_max_drawdown_63d",
        "pred_realized_vol_63d",
        "prob_loss_ge_10pct_63d",
        "prob_loss_ge_15pct_63d",
        "prob_loss_ge_20pct_63d",
    ]

    print(
        "\nHighest estimated 20% loss risk:"
    )

    print(
        output.nlargest(
            min(10, len(output)),
            "prob_loss_ge_20pct_63d",
        )[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        "\nPredictions:",
        output_file,
    )

    print(
        "Audit manifest:",
        manifest_file,
    )

    print(
        "RESEARCH ONLY: no hedge strategy recommended."
    )


if __name__ == "__main__":
    main()
