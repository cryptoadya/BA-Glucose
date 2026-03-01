"""Vorhersagen generieren, speichern und Clarke-EGA-Abbildungen erzeugen."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import SEED, STEP_MINUTES, HISTORY_STEPS, EPOCHS, BATCH_SIZE
from src.data_prep import BenchmarkProtocol, build_global_dataset
from src.ml_models import build_random_forest, build_xgboost
from src.training import scale_features_3d, get_callbacks, split_with_patient_ids
from src.models import build_lstm_model
from src.metrics import clarke_error_grid_analysis
from src.visualization import plot_clarke_error_grid, plot_clarke_zones_stacked

# Zufallsgenerator initialisieren
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Ausgabeverzeichnis
PREDICTIONS_DIR = Path(__file__).parent.parent / "results" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures" / "thesis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)




# =============================================================================
# Baselines
# =============================================================================
def persistence_baseline(x_test: np.ndarray) -> np.ndarray:
    """Letzten Glukosewert vorhersagen."""
    if x_test.ndim == 3:
        return x_test[:, -1, 0]
    return x_test[:, -1]


# =============================================================================
# Vorhersagen als CSV speichern
# =============================================================================
def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
    model_name: str,
    feature_set: str,
    horizon: int,
    metadata: dict,
) -> Path:
    """Vorhersagen als CSV-Datei speichern."""
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "patient_id": patient_ids,
        "sample_idx": range(len(y_true)),
    })
    
    filename = f"{model_name.lower()}_{feature_set.lower()}_ph{horizon}.csv"
    filepath = PREDICTIONS_DIR / filename
    df.to_csv(filepath, index=False)
    
    # Metadaten speichern
    meta_file = PREDICTIONS_DIR / f"{model_name.lower()}_{feature_set.lower()}_ph{horizon}_meta.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return filepath


# =============================================================================
# Trainieren und Vorhersagen generieren
# =============================================================================
def get_persistence_predictions(horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Persistenz-Basislinie-Vorhersagen generieren."""
    print(f"\n{'='*60}")
    print(f"Persistence Baseline — PH={horizon} min")
    print(f"{'='*60}")
    
    protocol = BenchmarkProtocol()
    horizon_steps = horizon // STEP_MINUTES
    
    # Datensatz mit E1-Merkmalen aufbauen
    x, y, patient_ids, feature_cols, stats = build_global_dataset(
        feature_set="E1",
        as_sequence=True,
        horizon_steps=horizon_steps,
        protocol=protocol,
        return_stats=True,
    )
    
    # Aufteilung
    _, _, x_test, _, _, y_test, pid_test = split_with_patient_ids(
        x, y, patient_ids,
        history_steps=HISTORY_STEPS,
        horizon_steps=horizon_steps,
    )
    
    # Persistenz: letzter Glukosewert
    start = time.time()
    y_pred = persistence_baseline(x_test)
    elapsed = time.time() - start
    
    # Metriken
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    clarke = clarke_error_grid_analysis(y_test, y_pred)
    
    print(f"  Samples: {len(y_test):,}")
    print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    print(f"  Clarke A+B: {clarke['AB']:.1f}%")
    print(f"  Time: {elapsed:.2f}s")
    
    metadata = {
        "model": "Persistence",
        "feature_set": "E1",
        "horizon": horizon,
        "n_samples": len(y_test),
        "rmse": float(rmse),
        "mae": float(mae),
        "clarke_A": float(clarke["A"]),
        "clarke_B": float(clarke["B"]),
        "clarke_AB": float(clarke["AB"]),
        "clarke_C": float(clarke["C"]),
        "clarke_D": float(clarke["D"]),
        "clarke_E": float(clarke["E"]),
        "fit_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
    }
    
    return y_test, y_pred, pid_test, metadata


def get_tml_predictions(model_name: str, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """TML-Modell trainieren und Vorhersagen generieren."""
    print(f"\n{'='*60}")
    print(f"{model_name}/E1 — PH={horizon} min")
    print(f"{'='*60}")
    
    protocol = BenchmarkProtocol()
    horizon_steps = horizon // STEP_MINUTES
    
    # Datensatz mit E1-Merkmalen aufbauen
    x, y, patient_ids, feature_cols, stats = build_global_dataset(
        feature_set="E1",
        as_sequence=True,
        horizon_steps=horizon_steps,
        protocol=protocol,
        return_stats=True,
    )
    
    # Aufteilung
    x_train, x_val, x_test, y_train, y_val, y_test, pid_test = split_with_patient_ids(
        x, y, patient_ids,
        history_steps=HISTORY_STEPS,
        horizon_steps=horizon_steps,
    )
    
    print(f"  Train: {len(x_train):,}, Val: {len(x_val):,}, Test: {len(x_test):,}")
    
    # Für TML flach machen
    x_train_flat = x_train.reshape(len(x_train), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)
    
    # Skalierung
    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train_flat)
    x_test_sc = scaler.transform(x_test_flat)
    
    # Modell erstellen
    if model_name == "RandomForest":
        model = build_random_forest(random_state=SEED)
    elif model_name == "XGBoost":
        model = build_xgboost(random_state=SEED)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Trainieren
    print(f"  Training {model_name}...", flush=True)
    start = time.time()
    model.fit(x_train_sc, y_train)
    elapsed = time.time() - start
    
    # Vorhersage
    y_pred = model.predict(x_test_sc)
    
    # Metriken
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    clarke = clarke_error_grid_analysis(y_test, y_pred)
    
    print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    print(f"  Clarke A+B: {clarke['AB']:.1f}%")
    print(f"  Time: {elapsed:.1f}s")
    
    metadata = {
        "model": model_name,
        "feature_set": "E1",
        "horizon": horizon,
        "n_samples": len(y_test),
        "rmse": float(rmse),
        "mae": float(mae),
        "clarke_A": float(clarke["A"]),
        "clarke_B": float(clarke["B"]),
        "clarke_AB": float(clarke["AB"]),
        "clarke_C": float(clarke["C"]),
        "clarke_D": float(clarke["D"]),
        "clarke_E": float(clarke["E"]),
        "fit_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
    }
    
    return y_test, y_pred, pid_test, metadata


def get_lstm_predictions(horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """LSTM trainieren und Vorhersagen generieren."""
    print(f"\n{'='*60}")
    print(f"LSTM/E1 — PH={horizon} min")
    print(f"{'='*60}")
    
    protocol = BenchmarkProtocol()
    horizon_steps = horizon // STEP_MINUTES
    
    # Datensatz mit E1-Merkmalen aufbauen
    x, y, patient_ids, feature_cols, stats = build_global_dataset(
        feature_set="E1",
        as_sequence=True,
        horizon_steps=horizon_steps,
        protocol=protocol,
        return_stats=True,
    )
    
    # Aufteilung
    x_train, x_val, x_test, y_train, y_val, y_test, pid_test = split_with_patient_ids(
        x, y, patient_ids,
        history_steps=HISTORY_STEPS,
        horizon_steps=horizon_steps,
    )
    
    print(f"  Train: {len(x_train):,}, Val: {len(x_val):,}, Test: {len(x_test):,}")
    
    # Skalierung
    x_train_sc, x_test_sc, scaler = scale_features_3d(x_train, x_test)
    x_val_sc = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)
    
    # LSTM erstellen
    n_steps, n_features = x_train_sc.shape[1], x_train_sc.shape[2]
    model = build_lstm_model(n_steps, n_features)
    
    # Trainieren
    print(f"  Training LSTM ({EPOCHS} epochs max)...", flush=True)
    start = time.time()
    
    history = model.fit(
        x_train_sc, y_train,
        validation_data=(x_val_sc, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(),
        verbose=1,
    )
    
    elapsed = time.time() - start
    epochs_trained = len(history.history.get("loss", []))
    
    # Vorhersage
    y_pred = model.predict(x_test_sc, verbose=0).flatten()
    
    # Metriken
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    clarke = clarke_error_grid_analysis(y_test, y_pred)
    
    print(f"  Epochs trained: {epochs_trained}")
    print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    print(f"  Clarke A+B: {clarke['AB']:.1f}%")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    metadata = {
        "model": "LSTM",
        "feature_set": "E1",
        "horizon": horizon,
        "n_samples": len(y_test),
        "epochs_trained": epochs_trained,
        "rmse": float(rmse),
        "mae": float(mae),
        "clarke_A": float(clarke["A"]),
        "clarke_B": float(clarke["B"]),
        "clarke_AB": float(clarke["AB"]),
        "clarke_C": float(clarke["C"]),
        "clarke_D": float(clarke["D"]),
        "clarke_E": float(clarke["E"]),
        "fit_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
    }
    
    return y_test, y_pred, pid_test, metadata





# =============================================================================
# Hauptprogramm
# =============================================================================
def main():
    """Modelle trainieren, Vorhersagen speichern, Abbildungen erzeugen."""
    print("=" * 70)
    print("STAGE 2: Generate Predictions and Clarke EGA Figures")
    print("=" * 70)
    print(f"Output directory: {PREDICTIONS_DIR}")
    print(f"Estimated time: ~35 minutes")
    print()
    
    total_start = time.time()
    all_results = []
    
    # =========================================================================
    # 1. Persistenz-Baselines
    # =========================================================================
    for horizon in [30, 60]:
        y_true, y_pred, pid, meta = get_persistence_predictions(horizon)
        path = save_predictions(y_true, y_pred, pid, "Persistence", "E1", horizon, meta)
        print(f"  ✓ Saved predictions: {path}")
        
        all_results.append({
            "label": f"Persistence (PH={horizon})",
            "horizon": horizon,
            "y_true": y_true,
            "y_pred": y_pred,
            "clarke": clarke_error_grid_analysis(y_true, y_pred),
            "metadata": meta,
        })
    
    # =========================================================================
    # 2. RandomForest/E1 für PH=30
    # =========================================================================
    y_true, y_pred, pid, meta = get_tml_predictions("RandomForest", 30)
    path = save_predictions(y_true, y_pred, pid, "RandomForest", "E1", 30, meta)
    print(f"  ✓ Saved predictions: {path}")
    
    all_results.append({
        "label": "RandomForest/E1 (PH=30)",
        "horizon": 30,
        "y_true": y_true,
        "y_pred": y_pred,
        "clarke": clarke_error_grid_analysis(y_true, y_pred),
        "metadata": meta,
    })
    
    # =========================================================================
    # 3. XGBoost/E1 für PH=60
    # =========================================================================
    y_true, y_pred, pid, meta = get_tml_predictions("XGBoost", 60)
    path = save_predictions(y_true, y_pred, pid, "XGBoost", "E1", 60, meta)
    print(f"  ✓ Saved predictions: {path}")
    
    all_results.append({
        "label": "XGBoost/E1 (PH=60)",
        "horizon": 60,
        "y_true": y_true,
        "y_pred": y_pred,
        "clarke": clarke_error_grid_analysis(y_true, y_pred),
        "metadata": meta,
    })
    
    # =========================================================================
    # 4. LSTM/E1 für PH=30 und PH=60
    # =========================================================================
    for horizon in [30, 60]:
        y_true, y_pred, pid, meta = get_lstm_predictions(horizon)
        path = save_predictions(y_true, y_pred, pid, "LSTM", "E1", horizon, meta)
        print(f"  ✓ Saved predictions: {path}")
        
        all_results.append({
            "label": f"LSTM/E1 (PH={horizon})",
            "horizon": horizon,
            "y_true": y_true,
            "y_pred": y_pred,
            "clarke": clarke_error_grid_analysis(y_true, y_pred),
            "metadata": meta,
        })
    
    # =========================================================================
    # 5. Clarke-EGA-Abbildungen erzeugen
    # =========================================================================
    print(f"\n{'='*60}")
    print("Generating Clarke EGA Figures")
    print(f"{'='*60}")
    
    for result in all_results:
        label = result["label"].replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        filename = f"clarke_ega_{label.lower()}.png"
        plot_clarke_error_grid(
            result["y_true"],
            result["y_pred"],
            title=f"Clarke EGA: {result['label']}",
            filepath=FIGURES_DIR / filename,
        )
    
    # =========================================================================
    # 6. Gestapeltes Balkendiagramm erzeugen
    # =========================================================================
    print(f"\n{'='*60}")
    print("Generating Stacked Bar Charts")
    print(f"{'='*60}")
    
    # PH=30 Vergleich
    ph30_results = [r for r in all_results if r["horizon"] == 30]
    plot_clarke_zones_stacked(ph30_results, FIGURES_DIR / "clarke_zones_stacked_ph30.png")
    
    # PH=60 Vergleich
    ph60_results = [r for r in all_results if r["horizon"] == 60]
    plot_clarke_zones_stacked(ph60_results, FIGURES_DIR / "clarke_zones_stacked_ph60.png")
    
    # =========================================================================
    # Zusammenfassung
    # =========================================================================
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"\nPredictions saved to: {PREDICTIONS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    
    print("\n--- Predictions Files ---")
    for f in sorted(PREDICTIONS_DIR.glob("*.csv")):
        print(f"  • {f.name}")
    
    print("\n--- Clarke EGA Results ---")
    for r in all_results:
        print(f"  {r['label']:30s} A+B={r['clarke']['AB']:.1f}%  RMSE={r['metadata']['rmse']:.2f}")


if __name__ == "__main__":
    main()
