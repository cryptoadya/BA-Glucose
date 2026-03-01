"""DL-Benchmark: CNN + LSTM + CRNN × E0/E1/E2 × PH 30/60/120."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    HORIZONS,
    SEED,
    STEP_MINUTES,
    HISTORY_STEPS,
    CSV_DIR,
    EPOCHS,
    BATCH_SIZE,
)
from src.data_prep import (
    BenchmarkProtocol,
    FEATURE_SETS,
    build_global_dataset,
)
from src.training import (
    train_val_test_split_temporal_by_patient,
    scale_features_3d,
    get_callbacks,
)
from src.metrics import calculate_metrics, clarke_error_grid_analysis
from src.models import build_cnn_model, build_lstm_model, build_crnn_model


np.random.seed(SEED)
tf.random.set_seed(SEED)


DL_MODELS: dict[str, Callable] = {
    "CNN": build_cnn_model,
    "LSTM": build_lstm_model,
    "CRNN": build_crnn_model,
}


def run_dl_experiment(
    feature_set: Literal["E0", "E1", "E2"],
    horizon_minutes: int,
    model_name: str,
    model_builder: Callable,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    verbose: int = 0,
) -> dict:
    """Ein DL-Experiment ausführen und Metriken + Datensatzstatistiken zurückgeben."""
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
    """Alle DL-Experimente ausführen."""
    results: list[dict] = []
    total = len(FEATURE_SETS) * len(HORIZONS) * len(DL_MODELS)
    current = 0

    print(f"=== DL Benchmark: {total} experiments ===\n")

    for feature_set in FEATURE_SETS:
        for horizon in HORIZONS:
            for model_name, model_builder in DL_MODELS.items():
                current += 1
                print(f"[{current}/{total}] {model_name} | {feature_set} | PH={horizon}...", end=" ", flush=True)

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

    df = pd.DataFrame(results)
    output_path = CSV_DIR / "dl_benchmark_with_embargo.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")

    print("\n=== Summary by PH ===")
    summary = df.groupby("horizon")[["rmse", "mae", "r2", "clarke_AB"]].mean()
    print(summary.round(2))


if __name__ == "__main__":
    main()

