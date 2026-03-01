"""Trainings- und Evaluierungshilfen."""
from __future__ import annotations

import numpy as np
from keras import callbacks
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import ES_MIN_DELTA, ES_PATIENCE


ScalerType = StandardScaler | MinMaxScaler


def scale_features_3d(
    x_train: np.ndarray,
    x_test: np.ndarray,
    scaler_type: str = "standard",
) -> tuple[np.ndarray, np.ndarray, ScalerType]:
    """3D-Sequenzdaten skalieren (Samples, Zeitschritte, Merkmale)."""
    n_samples_tr, n_steps, n_features = x_train.shape
    n_samples_te = x_test.shape[0]

    x_train_2d = x_train.reshape(-1, n_features)
    x_test_2d = x_test.reshape(-1, n_features)

    scaler: ScalerType = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train_2d)
    x_test_scaled = scaler.transform(x_test_2d)

    x_train_scaled = x_train_scaled.reshape(n_samples_tr, n_steps, n_features)
    x_test_scaled = x_test_scaled.reshape(n_samples_te, n_steps, n_features)

    return x_train_scaled, x_test_scaled, scaler


def train_val_test_split_temporal_by_patient(
    x: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    history_steps: int = 12,
    horizon_steps: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Temporale Aufteilung je Patient mit Embargo gegen Fensterüberlappung.
    """
    embargo = history_steps - 1 + horizon_steps

    x_train, x_val, x_test = [], [], []
    y_train, y_val, y_test = [], [], []

    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        xp, yp = x[mask], y[mask]
        n = len(xp)

        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train_stop = max(0, train_end - embargo)
        val_stop = max(train_end, val_end - embargo)

        if train_stop < 1 or (val_stop - train_end) < 1 or (n - val_end) < 1:
            continue

        x_train.append(xp[:train_stop])
        y_train.append(yp[:train_stop])
        x_val.append(xp[train_end:val_stop])
        y_val.append(yp[train_end:val_stop])
        x_test.append(xp[val_end:])
        y_test.append(yp[val_end:])

    if not x_train:
        raise ValueError("No patients left after embargo split.")

    return (
        np.concatenate(x_train),
        np.concatenate(x_val),
        np.concatenate(x_test),
        np.concatenate(y_train),
        np.concatenate(y_val),
        np.concatenate(y_test),
    )


def split_with_patient_ids(
    x: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    history_steps: int = 12,
    horizon_steps: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Temporale Aufteilung mit Embargo — gibt zusätzlich Patienten-IDs für den Testsatz zurück."""
    embargo = history_steps - 1 + horizon_steps

    x_train, x_val, x_test = [], [], []
    y_train, y_val, y_test = [], [], []
    pid_test: list[np.ndarray] = []

    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        xp, yp = x[mask], y[mask]
        n = len(xp)

        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train_stop = max(0, train_end - embargo)
        val_stop = max(train_end, val_end - embargo)

        if train_stop < 1 or (val_stop - train_end) < 1 or (n - val_end) < 1:
            continue

        x_train.append(xp[:train_stop])
        y_train.append(yp[:train_stop])
        x_val.append(xp[train_end:val_stop])
        y_val.append(yp[train_end:val_stop])
        x_test.append(xp[val_end:])
        y_test.append(yp[val_end:])
        pid_test.append(np.full(n - val_end, pid))

    if not x_train:
        raise ValueError("No patients left after embargo split.")

    return (
        np.concatenate(x_train),
        np.concatenate(x_val),
        np.concatenate(x_test),
        np.concatenate(y_train),
        np.concatenate(y_val),
        np.concatenate(y_test),
        np.concatenate(pid_test),
    )


def get_callbacks(
    patience: int = ES_PATIENCE,
    min_delta: float = ES_MIN_DELTA,
    monitor: str = "val_loss",
) -> list[callbacks.Callback]:
    """Standard-Callbacks für DL-Training: EarlyStopping + ReduceLROnPlateau."""
    return [
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=0,
        ),
    ]

