"""TML-Benchmark: Ridge + Random Forest + XGBoost × E0/E1/E2 × PH 30/60/120."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, Callable

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import HORIZONS, SEED, STEP_MINUTES, HISTORY_STEPS, CSV_DIR
from src.data_prep import (
    BenchmarkProtocol,
    FEATURE_SETS,
    build_global_dataset,
)
from src.training import train_val_test_split_temporal_by_patient
from src.metrics import calculate_metrics, clarke_error_grid_analysis
from src.ml_models import build_ridge, build_random_forest, build_xgboost


TML_MODELS: dict[str, Callable] = {
    "Ridge": build_ridge,
    "RandomForest": build_random_forest,
    "XGBoost": build_xgboost,
}


def run_tml_experiment(
    feature_set: Literal["E0", "E1", "E2"],
    horizon_minutes: int,
    model_name: str,
    model_builder: Callable,
) -> dict:
    """Ein TML-Experiment ausführen und Metriken + Datensatzstatistiken zurückgeben."""
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


def main() -> None:
    """Alle TML-Experimente ausführen."""
    results: list[dict] = []
    total = len(FEATURE_SETS) * len(HORIZONS) * len(TML_MODELS)
    current = 0

    print(f"=== TML Benchmark: {total} experiments ===\n")

    for feature_set in FEATURE_SETS:
        for horizon in HORIZONS:
            for model_name, model_builder in TML_MODELS.items():
                current += 1
                print(f"[{current}/{total}] {model_name} | {feature_set} | PH={horizon}...", end=" ")

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

    df = pd.DataFrame(results)
    output_path = CSV_DIR / "tml_benchmark_with_embargo.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")

    print("\n=== Summary by PH ===")
    summary = df.groupby("horizon")[["rmse", "mae", "r2", "clarke_AB"]].mean()
    print(summary.round(2))


if __name__ == "__main__":
    main()

