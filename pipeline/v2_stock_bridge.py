"""Read-only, explicitly point-in-time bridge to market_history_v2 stock prices.

Use unadjusted historical trading closes for option-hedge cash accounting.
A paired adjusted series is REQUIRED to detect unexplained within-horizon
changes in corporate-action price basis. Never substitute an adjusted close
for an unavailable unadjusted historical trading close.

This module does not authenticate to LSEG, download, alter, or save data.
It does not establish historical membership or validate option deliverables;
those are separate prerequisites to a multi-stock backtest.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class StockBasisError(ValueError):
    """Stock prices cannot safely be used on the historical share basis."""


def _date(value: str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None:
        raise ValueError("Invalid or timezone-aware trading date")
    return result.normalize()


def _series_dir(ric: str, mode: str, project_root: Path) -> Path:
    if not isinstance(ric, str) or not ric.strip() or ric != ric.strip():
        raise ValueError("Require an exact nonblank historical stock RIC")
    if mode not in ("unadjusted", "adjusted"):
        raise ValueError("mode must be unadjusted or adjusted")
    safe_ric = re.sub(r"[^A-Za-z0-9_.-]", "_", ric)
    digest = hashlib.sha256(ric.encode("utf-8")).hexdigest()[:8]
    return (
        project_root / "data" / "processed" / "market_history_v2"
        / "series" / f"{safe_ric}_{digest}" / mode
    )


def _matching_files(
    ric: str,
    mode: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    project_root: Path,
) -> list[Path]:
    folder = _series_dir(ric, mode, project_root)
    candidates: list[tuple[int, pd.Timestamp, Path]] = []
    for file in folder.glob("*.parquet"):
        match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet", file.name)
        if match is None:
            continue
        left, right = (_date(value) for value in match.groups())
        if left <= start <= end <= right:
            candidates.append(((right - left).days, right, file))
    # Prefer the *narrowest* covering file to avoid unnecessary data reads.
    # Tie break deterministically in favor of later collection end dates.
    return [p for _, _, p in sorted(candidates, key=lambda x: (x[0], -x[1].value, str(x[2])))]


def _read_mode(
    ric: str,
    mode: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    project_root: Path,
) -> pd.Series:
    files = _matching_files(ric, mode, start, end, project_root)
    if not files:
        raise FileNotFoundError(
            f"No market_history_v2 {mode} series covering "
            f"{ric}, {start.date()}–{end.date()}. "
            "Collect this exact RIC and date range; do not substitute another security."
        )
    errors: list[str] = []
    for path in files:
        try:
            frame = pd.read_parquet(path)
            required = {"Date", "RIC", "Adjustment Mode", "Close", "Close Field"}
            if not required.issubset(frame.columns) or frame.empty:
                raise StockBasisError("Missing required v2 fields or empty file")
            if not frame["RIC"].astype(str).eq(ric).all():
                raise StockBasisError("Unexpected RIC in v2 file")
            if not frame["Adjustment Mode"].astype(str).eq(mode).all():
                raise StockBasisError("Unexpected adjustment mode in v2 file")
            if not frame["Close Field"].isin(("TRDPRC_1", "OFF_CLOSE")).all():
                raise StockBasisError("Unknown underlying close field")
            dates = pd.to_datetime(frame["Date"], errors="raise")
            if dates.isna().any() or dates.dt.tz is not None:
                raise StockBasisError("Invalid or timezone-aware v2 dates")
            dates = dates.dt.normalize()
            if dates.duplicated().any():
                raise StockBasisError("Duplicate stock trading dates")
            close = pd.to_numeric(frame["Close"], errors="coerce").to_numpy(dtype=float)
            series = pd.Series(close, index=pd.DatetimeIndex(dates), name="Close").sort_index()
            series = series.loc[start:end]
            if series.empty or series.index[0] != start or series.index[-1] != end:
                raise StockBasisError("Incomplete decision-to-outcome coverage")
            if not np.isfinite(series.to_numpy()).all() or (series <= 0).any():
                raise StockBasisError("Missing or invalid stock closes inside holding window")
            return series
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append(f"{path.name}: {exc}")
    raise StockBasisError(
        f"No usable {mode} history for {ric} in {start.date()}–{end.date()}: "
        + "; ".join(errors)
    )


def load_v2_stock_window(
    ric: str,
    decision_date: str | pd.Timestamp,
    outcome_date: str | pd.Timestamp,
    *,
    hedge_horizon_days: int,
    expected_spot: float,
    project_root: str | Path | None = None,
    max_basis_drift: float = 0.002,
    large_return_guard: float = 0.45,
) -> pd.Series:
    """Return verified *unadjusted* trading closes including both endpoints.

    Do not use an unadjusted series spanning split/rights/distribution events
    unless quantity and option contract deliverables are explicitly modeled.
    A ratio drift or large return flags the window for manual investigation.
    This is an intentionally conservative guard, not proof of no corporate actions.
    """
    start, end = _date(decision_date), _date(outcome_date)
    if start >= end:
        raise ValueError("Decision must precede outcome date")
    if not isinstance(hedge_horizon_days, (int, np.integer)) or hedge_horizon_days <= 0:
        raise ValueError("Hedge horizon must be a positive integer")
    if not np.isfinite(expected_spot) or expected_spot <= 0:
        raise ValueError("Expected entry spot must be positive and finite")
    if not (0 <= max_basis_drift < 1 and 0 < large_return_guard < 1):
        raise ValueError("Invalid price-basis thresholds")
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    raw = _read_mode(ric, "unadjusted", start, end, root)
    adj = _read_mode(ric, "adjusted", start, end, root)
    if not raw.index.equals(adj.index):
        raise StockBasisError("Adjusted and unadjusted dates differ in the hedge window")
    if len(raw) != hedge_horizon_days + 1:
        raise StockBasisError(
            f"Expected {hedge_horizon_days + 1} sessions, found {len(raw)}"
        )
    if not np.isclose(raw.iloc[0], expected_spot, rtol=1e-5, atol=1e-4):
        raise StockBasisError(
            f"Historical trading-price basis mismatch: v2 {raw.iloc[0]:.5f}, "
            f"candidate entry spot {expected_spot:.5f}"
        )
    ratio = raw.to_numpy(dtype=float) / adj.to_numpy(dtype=float)
    if np.max(np.abs(ratio / ratio[0] - 1)) > max_basis_drift:
        raise StockBasisError(
            "Adjusted/unadjusted price-basis ratio changes during holding period; "
            "verify corporate actions, share quantities, and contract terms"
        )
    if raw.pct_change().iloc[1:].abs().gt(large_return_guard).any():
        raise StockBasisError(
            "Large daily move in unadjusted trading prices; verify no stock split "
            "or other corporate action before accounting for option contracts"
        )
    return raw.copy()
