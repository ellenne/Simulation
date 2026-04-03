"""
Missing-value audit and defensive cleaning for Simulation datasets.

Policy (used by ml_baseline_pipeline and documented in DataExploration.ipynb):
- Tabular: median imputation from TRAIN for numeric predictors; mode for object
  predictors; test rows use TRAIN statistics only (no leakage from test).
- NLP: NaN/blank complaint text -> placeholder string so vectorizers never see NaN.
- Rows with missing TARGET in train are dropped (should not occur in shipped CSVs).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def audit_csvs(data_dir: Path) -> pd.DataFrame:
    """Summarize NaN counts and blank text fields for every CSV under data_dir."""
    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(path)
        na_total = int(df.isna().sum().sum())
        blank_text = 0
        for c in df.select_dtypes(include=["object"]).columns:
            blank_text += int((df[c].astype(str).str.strip() == "").sum())
        rows.append(
            {
                "file": path.name,
                "rows": len(df),
                "cols": df.shape[1],
                "na_cells": na_total,
                "blank_object_cells": blank_text,
                "needs_handling": na_total > 0 or blank_text > 0,
            }
        )
    return pd.DataFrame(rows)


def impute_tabular_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    id_col: str = "id",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Impute missing predictors using TRAIN only. Returns copies; original frames unchanged.
    """
    tr = train.copy()
    te = test.copy()
    log: dict[str, Any] = {"imputed_cells_train": 0, "imputed_cells_test": 0, "dropped_train_rows": 0}

    if target_col in tr.columns:
        miss_y = tr[target_col].isna()
        if miss_y.any():
            log["dropped_train_rows"] = int(miss_y.sum())
            tr = tr.loc[~miss_y].copy()

    feat_cols = [c for c in tr.columns if c not in (id_col, target_col)]

    for col in feat_cols:
        if col not in te.columns:
            continue
        if pd.api.types.is_numeric_dtype(tr[col]):
            med = tr[col].median()
            if pd.isna(med):
                med = 0.0
            m_tr = tr[col].isna()
            m_te = te[col].isna()
            if m_tr.any():
                n = int(m_tr.sum())
                tr.loc[m_tr, col] = med
                log["imputed_cells_train"] += n
            if m_te.any():
                n = int(m_te.sum())
                te.loc[m_te, col] = med
                log["imputed_cells_test"] += n
        else:
            mode = tr[col].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else ""
            m_tr = tr[col].isna()
            m_te = te[col].isna()
            if m_tr.any():
                n = int(m_tr.sum())
                tr.loc[m_tr, col] = fill
                log["imputed_cells_train"] += n
            if m_te.any():
                n = int(m_te.sum())
                te.loc[m_te, col] = fill
                log["imputed_cells_test"] += n

    return tr, te, log


def clean_complaint_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    text_col: str = "complaint_text",
    target_col: str = "category_label",
    placeholder: str = "[empty complaint]",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Strip text, replace NaN/empty complaint_text with placeholder; drop train rows
    with missing target.
    """
    tr = train.copy()
    te = test.copy()
    log: dict[str, Any] = {
        "filled_empty_train": 0,
        "filled_empty_test": 0,
        "dropped_train_rows": 0,
    }

    if target_col in tr.columns:
        bad = tr[target_col].isna()
        if bad.any():
            log["dropped_train_rows"] = int(bad.sum())
            tr = tr.loc[~bad].copy()

    for name, df in ("train", tr), ("test", te):
        if text_col not in df.columns:
            continue
        s = df[text_col].fillna("").astype(str).str.strip()
        empty = s == ""
        n = int(empty.sum())
        s = s.mask(empty, placeholder)
        df[text_col] = s
        if name == "train":
            log["filled_empty_train"] = n
        else:
            log["filled_empty_test"] = n

    return tr, te, log


def format_tabular_cleaning_log(log: dict[str, Any]) -> str:
    parts = []
    if log.get("dropped_train_rows", 0):
        parts.append(f"dropped {log['dropped_train_rows']} train row(s) with missing target")
    ntr, nte = log.get("imputed_cells_train", 0), log.get("imputed_cells_test", 0)
    if ntr or nte:
        parts.append(f"imputed {ntr} train + {nte} test predictor cell(s) (train-derived stats)")
    if not parts:
        return "Data cleaning: no missing predictor/target values; no imputation applied."
    return "Data cleaning: " + "; ".join(parts) + "."


def format_complaint_cleaning_log(log: dict[str, Any]) -> str:
    parts = []
    if log.get("dropped_train_rows", 0):
        parts.append(f"dropped {log['dropped_train_rows']} train row(s) with missing category")
    ft, fe = log.get("filled_empty_train", 0), log.get("filled_empty_test", 0)
    if ft or fe:
        parts.append(f"replaced {ft} train + {fe} test empty/NaN text field(s) with placeholder")
    if not parts:
        return "Data cleaning: complaint text present; no placeholder fills."
    return "Data cleaning: " + "; ".join(parts) + "."
