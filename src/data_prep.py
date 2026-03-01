"""Vorverarbeitung der HUPA-UCM-Daten für TML- und DL-Benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    STEP_MINUTES,
    HISTORY_STEPS,
    HORIZON_STEPS,
    MAX_CGM_GAP_MINUTES,
)

# -----------------------------
# Merkmal-Sets (Feature Sets)
# -----------------------------
FEATURE_SETS: dict[str, list[str]] = {
    "E0": ["glucose"],  # Nur CGM
    "E1": [  # Kernsignale (4 Merkmale)
        "glucose",
        "basal_rate",
        "bolus_volume_delivered",
        "carb_input",
    ],
    "E2": [  # Vollständig mit Wearable-Daten (6 Merkmale)
        "glucose",
        "basal_rate",
        "bolus_volume_delivered",
        "carb_input",
        "heart_rate",
        "steps",
    ],
    "E3": [  # E2 + Kalorien (7 Merkmale, Mohajeri-Ansatz)
        "glucose",
        "basal_rate",
        "bolus_volume_delivered",
        "carb_input",
        "heart_rate",
        "steps",
        "calories",
    ],
    "E4": [  # Kombiniertes Insulin (3 Merkmale, Mohajeri-Ansatz)
        "glucose",
        "insulin_total",  # basal_rate + bolus kombiniert
        "carb_input",
    ],
}

# Aggregationsrichtlinie (Single Source of Truth)
# Ereignis-Spalten -> sum, kontinuierliche Spalten -> mean
AGG_POLICY: dict[str, str] = {
    # Kontinuierliche Signale -> mean
    "glucose": "mean",
    "heart_rate": "mean",
    "basal_rate": "mean",
    # Ereignissignale -> sum
    "bolus_volume_delivered": "sum",
    "carb_input": "sum",
    "steps": "sum",
    "calories": "sum",
    # Abgeleitetes Merkmal (für E4)
    "insulin_total": "sum",  # basal_rate + bolus kombiniert
}

# Abgeleitet aus AGG_POLICY
EVENT_LIKE_COLS = tuple(k for k, v in AGG_POLICY.items() if v == "sum")
CONTINUOUS_COLS = tuple(k for k, v in AGG_POLICY.items() if v == "mean")


@dataclass(frozen=True)
class BenchmarkProtocol:
    """
    Ein Protokoll = eine feste, reproduzierbare Datenrichtlinie für alle Experimente/Modelle.
    """
    step_minutes: int = STEP_MINUTES

    # Interpolation limit for CGM/HR gaps (in minutes).
    max_cgm_gap_minutes: Optional[int] = MAX_CGM_GAP_MINUTES

    interpolate_hr: bool = True
    require_hr_for_trim: bool = True

    # Scientific safeguard against "synthetic windows":
    # minimum fraction of truly observed CGM points in each history window.
    min_obs_frac: float = 0.5


# -----------------------------
# Hilfsfunktionen
# -----------------------------
def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        raise ValueError("Column 'time' expected in HUPA CSV.")
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").set_index("time")
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("Failed to create DatetimeIndex from 'time'.")
    return out


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c == "time":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _agg_policy(cols: list[str]) -> dict[str, str]:
    """Aggregationsrichtlinie für Spalten aus AGG_POLICY ableiten."""
    return {c: AGG_POLICY.get(c, "mean") for c in cols}



def _resample_to_grid(df: pd.DataFrame, step_minutes: int) -> pd.DataFrame:
    df_num = df.select_dtypes(include=[np.number]).copy()
    if df_num.empty:
        raise ValueError("No numeric columns found after parsing.")
    return df_num.resample(f"{step_minutes}min").agg(_agg_policy(list(df_num.columns))).sort_index()


def _interpolate_internal_gaps(
    s: pd.Series,
    step_minutes: int,
    max_gap_minutes: Optional[int],
) -> pd.Series:
    """
    Nur interne Lücken interpolieren; keine Randextrapolation.
    """
    if max_gap_minutes is None:
        return s.interpolate(method="time", limit_direction="both", limit_area="inside")

    limit_steps = max(1, int(max_gap_minutes / step_minutes))
    return s.interpolate(
        method="time",
        limit=limit_steps,
        limit_direction="both",
        limit_area="inside",
    )


def _trim_valid_range(df: pd.DataFrame, require_hr: bool) -> pd.DataFrame:
    if "glucose" not in df.columns:
        raise ValueError("Column 'glucose' expected after resampling.")

    req_cols = ["glucose"]
    if require_hr and "heart_rate" in df.columns:
        req_cols.append("heart_rate")

    valid = df[req_cols].notna().all(axis=1)
    if not bool(valid.any()):
        raise ValueError("No valid segment found: required signals are missing everywhere.")

    first = valid.idxmax()
    last = valid.iloc[::-1].idxmax()
    return df.loc[first:last].copy()


# -----------------------------
# Vorverarbeitung
# -----------------------------
def preprocess_patient_df(
    df_raw: pd.DataFrame,
    protocol: BenchmarkProtocol = BenchmarkProtocol(),
) -> pd.DataFrame:
    """
    Baseline-Vorverarbeitung für faires Benchmarking.
    """
    df = _to_datetime_index(df_raw)
    df = _coerce_numeric(df)

    # Resampling auf reguläres 5-Minuten-Raster
    df = _resample_to_grid(df, step_minutes=protocol.step_minutes)

    # Fehlende Spalten ergänzen (damit Feature-Sets nicht abstürzen)
    for c in EVENT_LIKE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in CONTINUOUS_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Beobachtete CGM-Punkte vor Interpolation markieren (wissenschaftliche Absicherung)
    df["glucose_obs"] = df["glucose"].notna().astype(np.int8)

    # CGM (und optional HR) nur in internen Lücken interpolieren
    df["glucose"] = _interpolate_internal_gaps(
        df["glucose"], protocol.step_minutes, protocol.max_cgm_gap_minutes
    )

    if protocol.interpolate_hr and "heart_rate" in df.columns:
        df["heart_rate"] = _interpolate_internal_gaps(
            df["heart_rate"], protocol.step_minutes, protocol.max_cgm_gap_minutes
        )

    # Ränder ohne gültiges CGM (und optional HR) abschneiden
    df = _trim_valid_range(df, require_hr=protocol.require_hr_for_trim)

    # Ereignis-Signale: fehlende Werte mit 0 füllen
    for c in EVENT_LIKE_COLS:
        df[c] = df[c].fillna(0.0)

    # insulin_total für E4-Kompatibilität erzeugen
    df["insulin_total"] = df["basal_rate"] + df["bolus_volume_delivered"]

    return df


# -----------------------------
# Fensterbildung (einheitlich für DL/TML)
# -----------------------------
def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    history_steps: int = HISTORY_STEPS,
    horizon_steps: int = HORIZON_STEPS,
    target_col: str = "glucose",
    as_sequence: bool = True,
    min_obs_frac: float = 0.0,
    obs_col: str = "glucose_obs",
    return_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]:
    """
    Einheitlicher Fensterbauer:
      - DL: as_sequence=True  -> X Form (N, T, F)
      - TML: as_sequence=False -> X Form (N, T*F)
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    values = df[feature_cols].to_numpy(dtype=np.float32)
    target = df[target_col].to_numpy(dtype=np.float32)

    n = len(df)
    if n < history_steps + horizon_steps:
        raise ValueError("Not enough rows for given history_steps + horizon_steps.")

    # Window k covers indices [k, k + history_steps - 1]
    # Target for window k is at index (k + history_steps - 1) + horizon_steps
    # This gives exactly PH minutes ahead from the last point in the window
    x_view = np.lib.stride_tricks.sliding_window_view(values, window_shape=history_steps, axis=0)
    # x_view shape is (n - history_steps + 1, F, history_steps), need (N, T, F)
    x_view = np.transpose(x_view, (0, 2, 1))  # -> (n_total, T, F)
    n_total = n - history_steps - horizon_steps + 1
    x_windows = x_view[:n_total]  # (n_total, T, F)
    y = target[history_steps - 1 + horizon_steps : history_steps - 1 + horizon_steps + n_total]

    # Basisvalidierung: keine NaNs
    mask_nan = ~np.isnan(x_windows).any(axis=(1, 2)) & ~np.isnan(y)

    # Wissenschaftliche Absicherung: Mindestanteil beobachteter CGM-Punkte
    mask_obs = np.ones(n_total, dtype=bool)
    if min_obs_frac > 0.0:
        if obs_col not in df.columns:
            raise ValueError(f"obs_col '{obs_col}' not found in df (needed for min_obs_frac).")

        obs = df[obs_col].to_numpy(dtype=np.int16)
        obs_roll = (
            pd.Series(obs, index=df.index)
            .rolling(window=history_steps, min_periods=history_steps)
            .sum()
            .to_numpy()
        )
        # For window k, history ends at index k+history_steps-1
        obs_in_window = obs_roll[history_steps - 1 : history_steps - 1 + n_total]
        min_obs = int(np.ceil(history_steps * float(min_obs_frac)))
        mask_obs = (obs_in_window >= min_obs)

    mask = mask_nan & mask_obs
    x_windows = x_windows[mask]
    y = y[mask]

    if x_windows.shape[0] == 0:
        raise ValueError("No valid windows remain after NaN / min-obs filtering.")

    if not as_sequence:
        x_windows = x_windows.reshape(x_windows.shape[0], -1)

    if not return_stats:
        return x_windows, y

    stats = {
        "n_windows_possible": int(n_total),
        "n_windows_after_nan": int(mask_nan.sum()),
        "n_windows_kept": int(mask.sum()),
        "dropped_nan": int(n_total - mask_nan.sum()),
        "dropped_min_obs": int(mask_nan.sum() - mask.sum()),
        "min_obs_frac": float(min_obs_frac),
    }
    return x_windows, y, stats


def make_windows_sequence(
    df: pd.DataFrame,
    feature_cols: list[str],
    history_steps: int = HISTORY_STEPS,
    horizon_steps: int = HORIZON_STEPS,
    target_col: str = "glucose",
    min_obs_frac: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    return make_windows(
        df,
        feature_cols=feature_cols,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        target_col=target_col,
        as_sequence=True,
        min_obs_frac=min_obs_frac,
    )


def make_windows_tabular(
    df: pd.DataFrame,
    feature_cols: list[str],
    history_steps: int = HISTORY_STEPS,
    horizon_steps: int = HORIZON_STEPS,
    target_col: str = "glucose",
    min_obs_frac: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    return make_windows(
        df,
        feature_cols=feature_cols,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        target_col=target_col,
        as_sequence=False,
        min_obs_frac=min_obs_frac,
    )


# -----------------------------
# Datensatz-Aufbau (eine Funktion für DL/TML)
# -----------------------------
def _iter_patient_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("HUPA*P.csv"))
    if not files:
        raise FileNotFoundError(f"No HUPA*P.csv files found in {data_dir.resolve()}.")
    return files


def build_global_dataset(
    *,
    data_dir: Path = DATA_DIR,
    protocol: BenchmarkProtocol = BenchmarkProtocol(),
    feature_set: Literal["E0", "E1", "E2", "E3", "E4"] = "E1",
    feature_cols: Optional[list[str]] = None,
    as_sequence: bool = True,
    history_steps: int = HISTORY_STEPS,
    horizon_steps: int = HORIZON_STEPS,
    target_col: str = "glucose",
    skip_patients_with_no_windows: bool = True,
    allowed_patients: Optional[set[str]] = None,
    return_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]] | tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """
    Globalen Datensatz aufbauen:
      - DL: as_sequence=True  -> X (N, T, F)
      - TML: as_sequence=False -> X (N, T*F)
    """
    if feature_cols is None:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature_set '{feature_set}'. Available: {list(FEATURE_SETS)}")
        feature_cols = FEATURE_SETS[feature_set]

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    patient_ids: list[str] = []

    # Statistiken
    dataset_stats = {
        "n_patients": 0,
        "n_patients_skipped": 0,
        "n_windows_possible": 0,
        "n_windows_kept": 0,
        "dropped_nan": 0,
        "dropped_min_obs": 0,
        "per_patient": {},
    }

    for p in _iter_patient_files(data_dir):
        # Nur erlaubte Patienten (für fairen E0 vs E1 Vergleich)
        if allowed_patients is not None and p.stem not in allowed_patients:
            continue

        df_raw = pd.read_csv(p, sep=";")
        df_prep = preprocess_patient_df(df_raw, protocol=protocol)

        try:
            xp, yp, stats = make_windows(
                df_prep,
                feature_cols=feature_cols,
                history_steps=history_steps,
                horizon_steps=horizon_steps,
                target_col=target_col,
                as_sequence=as_sequence,
                min_obs_frac=protocol.min_obs_frac,
                return_stats=True,
            )
        except ValueError:
            if skip_patients_with_no_windows:
                dataset_stats["n_patients_skipped"] += 1
                continue
            raise

        all_x.append(xp)
        all_y.append(yp)
        patient_ids.extend([p.stem] * len(yp))

        # Statistiken aggregieren
        dataset_stats["n_patients"] += 1
        dataset_stats["n_windows_possible"] += stats["n_windows_possible"]
        dataset_stats["n_windows_kept"] += stats["n_windows_kept"]
        dataset_stats["dropped_nan"] += stats["dropped_nan"]
        dataset_stats["dropped_min_obs"] += stats["dropped_min_obs"]
        dataset_stats["per_patient"][p.stem] = stats

    if not all_x:
        raise ValueError("Dataset is empty: no patient produced valid windows after filtering.")

    result = (
        np.concatenate(all_x, axis=0),
        np.concatenate(all_y, axis=0),
        np.asarray(patient_ids),
        feature_cols,
    )

    if return_stats:
        return result + (dataset_stats,)
    return result


# -----------------------------
# Diagnose
# -----------------------------
def analyze_sparsity(df: pd.DataFrame, exclude_cols: Optional[list[str]] = None) -> dict[str, float]:
    """
    Anteil der Nullen je numerischem Merkmal.
    """
    if exclude_cols is None:
        exclude_cols = ["glucose", "glucose_obs"]  # Beobachtungsmarker ausschließen

    sparsity: dict[str, float] = {}
    num_cols = set(df.select_dtypes(include=[np.number]).columns)

    for col in df.columns:
        if col in num_cols and col not in exclude_cols:
            total = len(df)
            sparsity[col] = float((df[col] == 0).sum()) * 100.0 / total if total else 0.0

    return sparsity
