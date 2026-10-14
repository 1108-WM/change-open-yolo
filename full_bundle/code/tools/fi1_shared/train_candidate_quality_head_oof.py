"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import numpy as np

from tools.audit_train_candidate_quality_dataset import NATIVE_SOURCE, TRACK_FEATURES, feature_schema

def _as_float(row: dict, field: str) -> float:
    value = row.get(field)
    return float("nan") if value is None else float(value)

def feature_matrix(rows: list[dict], group: str, protocol_name: str) -> tuple[np.ndarray, list[str]]:
    groups = feature_schema(protocol_name)["groups"]
    if group not in groups:
        raise ValueError(f"unknown feature group {group}")
    names, columns = ["candidate_source=d2b_track"], [np.asarray([float(row["candidate_source"] != NATIVE_SOURCE) for row in rows])]
    for field in groups[group]["numeric"]:
        values = np.asarray([_as_float(row, field) for row in rows], dtype=np.float64)
        columns.append(values)
        names.append(field)
        if field in TRACK_FEATURES:
            columns.append(np.isnan(values).astype(np.float64))
            names.append(f"{field}__missing")
    return np.column_stack(columns), names

def canonicalize_predictions(target: str, prediction: np.ndarray) -> np.ndarray:
    """Use one [0, 1] prediction contract for metrics and OOF consumers."""
    values = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{target}: model produced non-finite predictions")
    return np.clip(values, 0.0, 1.0)
