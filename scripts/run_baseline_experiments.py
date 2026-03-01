"""Baseline-Experimente: Persistenz und linearer Trend."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import HORIZONS, STEP_MINUTES, HISTORY_STEPS, CSV_DIR
from src.data_prep import BenchmarkProtocol, build_global_dataset
from src.training import train_val_test_split_temporal_by_patient
from src.metrics import calculate_metrics, clarke_error_grid_analysis


def persistence_baseline(x_test: np.ndarray) -> np.ndarray:
    """Letzten Glukosewert vorhersagen (Persistenz-Basislinie).
    
    Für 3D-Daten (N, T, F): letzten Zeitschritt, erstes Merkmal (Glukose).
    Für 2D-Daten (N, T*F): Glukose des letzten Zeitschritts.
    """
    if x_test.ndim == 3:
        # (N, T, F) — letzten Zeitschritt nehmen, Glukose ist Merkmal 0
        return x_test[:, -1, 0]
    else:
        # (N, T*F) — Glukose am letzten Zeitschritt
        # Für E0 (F=1) ist es die letzte Spalte
        # Glukose ist immer das erste Merkmal, wiederholt an jedem Zeitschritt
        n_features = 1  # E0 has only glucose
        return x_test[:, -n_features]  # Last glucose value


def linear_trend_baseline(x_test: np.ndarray, horizon_steps: int) -> np.ndarray:
    """Vorhersage mittels linearer Extrapolation der letzten zwei Glukosewerte."""

    if x_test.ndim == 3:
        # (N, T, F) — Glukose ist Merkmal 0
        g_last = x_test[:, -1, 0]
        g_prev = x_test[:, -2, 0]
    else:
        # (N, T*F) — für E0 mit F=1
        g_last = x_test[:, -1]
        g_prev = x_test[:, -2]
    
    slope = g_last - g_prev  # Change per step
    return g_last + horizon_steps * slope


def run_baseline_experiment(horizon_minutes: int) -> list[dict]:
    """Persistenz- und Trend-Baselines für einen Horizont ausführen."""
    horizon_steps = horizon_minutes // STEP_MINUTES
    protocol = BenchmarkProtocol()
    
    # E0 verwenden (nur Glukose) — Baselines nutzen ohnehin nur Glukose
    x, y, patient_ids, feature_cols, dataset_stats = build_global_dataset(
        feature_set="E0",
        as_sequence=True,  # Need 3D for correct indexing
        horizon_steps=horizon_steps,
        protocol=protocol,
        return_stats=True,
    )
    
    # Gleiche Aufteilung wie Modelle (mit Embargo)
    _, _, x_test, _, _, y_test = train_val_test_split_temporal_by_patient(
        x, y, patient_ids,
        history_steps=HISTORY_STEPS,
        horizon_steps=horizon_steps,
    )
    
    results = []
    
    # Persistenz-Basislinie
    y_pred_persist = persistence_baseline(x_test)
    metrics_persist = calculate_metrics(y_test, y_pred_persist)
    clarke_persist = clarke_error_grid_analysis(y_test, y_pred_persist)
    
    results.append({
        "feature_set": "E0",
        "horizon": horizon_minutes,
        "model": "Persistence",
        "model_type": "Baseline",
        "n_features": 1,
        "n_patients": dataset_stats["n_patients"],
        "n_test": len(x_test),
        "history_steps": HISTORY_STEPS,
        "horizon_steps": horizon_steps,
        "embargo_steps": HISTORY_STEPS - 1 + horizon_steps,
        **metrics_persist,
        "clarke_A": clarke_persist["A"],
        "clarke_B": clarke_persist["B"],
        "clarke_AB": clarke_persist["AB"],
        "clarke_C": clarke_persist["C"],
        "clarke_D": clarke_persist["D"],
        "clarke_E": clarke_persist["E"],
    })
    
    # Linearer Trend-Basislinie
    y_pred_trend = linear_trend_baseline(x_test, horizon_steps)
    metrics_trend = calculate_metrics(y_test, y_pred_trend)
    clarke_trend = clarke_error_grid_analysis(y_test, y_pred_trend)
    
    results.append({
        "feature_set": "E0",
        "horizon": horizon_minutes,
        "model": "LinearTrend",
        "model_type": "Baseline",
        "n_features": 1,
        "n_patients": dataset_stats["n_patients"],
        "n_test": len(x_test),
        "history_steps": HISTORY_STEPS,
        "horizon_steps": horizon_steps,
        "embargo_steps": HISTORY_STEPS - 1 + horizon_steps,
        **metrics_trend,
        "clarke_A": clarke_trend["A"],
        "clarke_B": clarke_trend["B"],
        "clarke_AB": clarke_trend["AB"],
        "clarke_C": clarke_trend["C"],
        "clarke_D": clarke_trend["D"],
        "clarke_E": clarke_trend["E"],
    })
    
    return results


def main() -> None:
    """Naive Baselines für alle Horizonte ausführen."""
    results: list[dict] = []
    
    print("=== Naive Baselines ===\n")
    
    for horizon in HORIZONS:
        print(f"PH={horizon}...", end=" ")
        start = time.time()
        
        horizon_results = run_baseline_experiment(horizon)
        for r in horizon_results:
            r["fit_seconds"] = 0.0  # No training
        results.extend(horizon_results)
        
        elapsed = time.time() - start
        persist_rmse = horizon_results[0]["rmse"]
        trend_rmse = horizon_results[1]["rmse"]
        print(f"Persistence RMSE={persist_rmse:.2f}, LinearTrend RMSE={trend_rmse:.2f} ({elapsed:.1f}s)")
    
    # Ergebnisse speichern
    df = pd.DataFrame(results)
    output_path = CSV_DIR / "baseline_benchmark.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    # Zusammenfassung
    print("\n=== Summary ===")
    summary = df.pivot(index="horizon", columns="model", values=["rmse", "mae", "clarke_AB"])
    print(summary.round(2))


if __name__ == "__main__":
    main()
