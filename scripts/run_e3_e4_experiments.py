"""E3/E4-Experimente: erweiterte Merkmal-Sets nach Mohajeri."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from config import (
    HORIZONS,
    SEED,
    STEP_MINUTES,
    HISTORY_STEPS,
    CSV_DIR,
    EPOCHS,
    BATCH_SIZE,
)
from src.data_prep import BenchmarkProtocol, build_global_dataset
from src.training import train_val_test_split_temporal_by_patient, scale_features_3d, get_callbacks
from src.metrics import calculate_metrics, clarke_error_grid_analysis
from src.ml_models import build_ridge, build_random_forest, build_xgboost
from src.models import build_cnn_model, build_lstm_model, build_crnn_model


np.random.seed(SEED)
tf.random.set_seed(SEED)


E3_E4_FEATURE_SETS = ["E3", "E4"]

TML_MODELS: dict[str, Callable] = {
    "Ridge": build_ridge,
    "RandomForest": build_random_forest,
    "XGBoost": build_xgboost,
}

DL_MODELS: dict[str, Callable] = {
    "CNN": build_cnn_model,
    "LSTM": build_lstm_model,
    "CRNN": build_crnn_model,
}


def run_tml_experiment(
    feature_set: Literal["E3", "E4"],
    horizon_minutes: int,
    model_name: str,
    model_builder: Callable,
) -> dict:
    """Ein TML-Experiment für E3/E4 ausführen."""
    horizon_steps = horizon_minutes // STEP_MINUTES
    protocol = BenchmarkProtocol()

    x, y, patient_ids, feature_cols, dataset_stats = build_global_dataset(
        feature_set=feature_set,
        as_sequence=False,
        horizon_steps=horizon_steps,
        protocol=protocol,
        return_stats=True,
    )

    x_train, x_val, x_test, y_train, _, y_test = train_val_test_split_temporal_by_patient(
        x, y, patient_ids,
        history_steps=HISTORY_STEPS,
        horizon_steps=horizon_steps,
    )

    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)

    model = model_builder(random_state=SEED)
    model.fit(x_train_sc, y_train)

    y_pred = model.predict(x_test_sc)

    metrics = calculate_metrics(y_test, y_pred)
    clarke = clarke_error_grid_analysis(y_test, y_pred)

    return {
        "feature_set": feature_set,
        "horizon": horizon_minutes,
        "model": model_name,
        "model_type": "TML",
        "n_features": len(feature_cols),
        "n_patients": dataset_stats["n_patients"],
        "n_patients_skipped": dataset_stats["n_patients_skipped"],
        "n_windows_possible": dataset_stats["n_windows_possible"],
        "n_windows_kept": dataset_stats["n_windows_kept"],
        "dropped_nan": dataset_stats["dropped_nan"],
        "dropped_min_obs": dataset_stats["dropped_min_obs"],
        "n_train": len(x_train),
        "n_val": len(x_val),
        "n_test": len(x_test),
        "history_steps": HISTORY_STEPS,
        "horizon_steps": horizon_steps,
        "embargo_steps": HISTORY_STEPS - 1 + horizon_steps,
        "train_frac": 0.7,
        "val_frac": 0.15,
        **metrics,
        "clarke_A": clarke["A"],
        "clarke_B": clarke["B"],
        "clarke_AB": clarke["AB"],
        "clarke_C": clarke["C"],
        "clarke_D": clarke["D"],
        "clarke_E": clarke["E"],
    }


def run_dl_experiment(
    feature_set: Literal["E3", "E4"],
    horizon_minutes: int,
    model_name: str,
    model_builder: Callable,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    verbose: int = 0,
) -> dict:
    """Ein DL-Experiment für E3/E4 ausführen."""
    horizon_steps = horizon_minutes // STEP_MINUTES
    protocol = BenchmarkProtocol()

    x, y, patient_ids, feature_cols, dataset_stats = build_global_dataset(
        feature_set=feature_set,
        as_sequence=True,
        horizon_steps=horizon_steps,
        protocol=protocol,
        return_stats=True,
    )

    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split_temporal_by_patient(
        x, y, patient_ids,
        history_steps=HISTORY_STEPS,
        horizon_steps=horizon_steps,
    )

    x_train_sc, x_test_sc, scaler = scale_features_3d(x_train, x_test)
    x_val_sc = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)

    n_steps, n_features = x_train_sc.shape[1], x_train_sc.shape[2]
    model = model_builder(n_steps, n_features)

    history = model.fit(
        x_train_sc,
        y_train,
        validation_data=(x_val_sc, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(),
        verbose=verbose,
    )

    epochs_trained = len(history.history.get("loss", []))
    y_pred = model.predict(x_test_sc, verbose=0).flatten()

    metrics = calculate_metrics(y_test, y_pred)
    clarke = clarke_error_grid_analysis(y_test, y_pred)

    return {
        "feature_set": feature_set,
        "horizon": horizon_minutes,
        "model": model_name,
        "model_type": "DL",
        "n_features": len(feature_cols),
        "n_patients": dataset_stats["n_patients"],
        "n_patients_skipped": dataset_stats["n_patients_skipped"],
        "n_windows_possible": dataset_stats["n_windows_possible"],
        "n_windows_kept": dataset_stats["n_windows_kept"],
        "dropped_nan": dataset_stats["dropped_nan"],
        "dropped_min_obs": dataset_stats["dropped_min_obs"],
        "n_train": len(x_train),
        "n_val": len(x_val),
        "n_test": len(x_test),
        "history_steps": HISTORY_STEPS,
        "horizon_steps": horizon_steps,
        "embargo_steps": HISTORY_STEPS - 1 + horizon_steps,
        "train_frac": 0.7,
        "val_frac": 0.15,
        "epochs_trained": epochs_trained,
        **metrics,
        "clarke_A": clarke["A"],
        "clarke_B": clarke["B"],
        "clarke_AB": clarke["AB"],
        "clarke_C": clarke["C"],
        "clarke_D": clarke["D"],
        "clarke_E": clarke["E"],
    }


def main() -> None:
    """Alle E3/E4-Experimente ausführen (TML + DL)."""
    results: list[dict] = []
    
    # TML-Experimente
    total_tml = len(E3_E4_FEATURE_SETS) * len(HORIZONS) * len(TML_MODELS)
    current = 0
    
    print(f"=== E3/E4 TML Experiments: {total_tml} ===\n")
    
    for feature_set in E3_E4_FEATURE_SETS:
        for horizon in HORIZONS:
            for model_name, model_builder in TML_MODELS.items():
                current += 1
                print(f"[{current}/{total_tml}] {model_name} | {feature_set} | PH={horizon}...", end=" ")
                
                start = time.time()
                try:
                    result = run_tml_experiment(
                        feature_set,  # type: ignore[arg-type]
                        horizon,
                        model_name,
                        model_builder,
                    )
                    elapsed = time.time() - start
                    result["fit_seconds"] = round(elapsed, 2)
                    print(f"RMSE={result['rmse']:.2f} ({elapsed:.1f}s)")
                    results.append(result)
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue
    
    # DL-Experimente
    total_dl = len(E3_E4_FEATURE_SETS) * len(HORIZONS) * len(DL_MODELS)
    current = 0
    
    print(f"\n=== E3/E4 DL Experiments: {total_dl} ===\n")
    
    for feature_set in E3_E4_FEATURE_SETS:
        for horizon in HORIZONS:
            for model_name, model_builder in DL_MODELS.items():
                current += 1
                print(f"[{current}/{total_dl}] {model_name} | {feature_set} | PH={horizon}...", end=" ", flush=True)
                
                start = time.time()
                try:
                    result = run_dl_experiment(
                        feature_set,  # type: ignore[arg-type]
                        horizon,
                        model_name,
                        model_builder,
                    )
                    elapsed = time.time() - start
                    result["fit_seconds"] = round(elapsed, 2)
                    print(f"RMSE={result['rmse']:.2f} ({elapsed:.1f}s)")
                    results.append(result)
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue
    
    # Ergebnisse speichern
    df = pd.DataFrame(results)
    output_path = CSV_DIR / "e3_e4_benchmark.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    # Zusammenfassung
    print("\n=== Summary by Feature Set ===")
    summary = df.groupby(["feature_set", "model_type"])[["rmse", "mae", "r2", "clarke_AB"]].mean()
    print(summary.round(2))


if __name__ == "__main__":
    main()
