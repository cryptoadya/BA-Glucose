"""Pro-Patient-Evaluation mit Bootstrap-Konfidenzintervallen."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import HORIZONS, SEED, STEP_MINUTES, HISTORY_STEPS, CSV_DIR, EPOCHS, BATCH_SIZE
from src.data_prep import BenchmarkProtocol, FEATURE_SETS, build_global_dataset
from src.ml_models import build_ridge, build_random_forest, build_xgboost
from src.training import split_with_patient_ids


FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)



def calculate_per_patient_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
) -> pd.DataFrame:
    """Metriken für jeden Patienten einzeln berechnen."""
    records = []
    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        yt, yp = y_true[mask], y_pred[mask]
        
        mse = mean_squared_error(yt, yp)
        records.append({
            "patient": pid,
            "n_samples": len(yt),
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(yt, yp),
            "r2": r2_score(yt, yp) if len(yt) > 1 else 0.0,
        })
    
    return pd.DataFrame(records)


def macro_average(df: pd.DataFrame) -> dict[str, float]:
    """Makro-Mittelwert: einfacher Durchschnitt über Patienten."""
    return {
        "rmse_macro": df["rmse"].mean(),
        "mae_macro": df["mae"].mean(),
        "r2_macro": df["r2"].mean(),
        "rmse_std": df["rmse"].std(),
        "mae_std": df["mae"].std(),
    }


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap-95%-Konfidenzintervall durch Resampling."""
    rng = np.random.default_rng(seed)
    boot_means = []
    
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(sample.mean())
    
    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    
    return {
        "mean": values.mean(),
        "ci_lower": np.percentile(boot_means, alpha * 100),
        "ci_upper": np.percentile(boot_means, (1 - alpha) * 100),
    }


# =============================================================================
# Baselines
# =============================================================================
def persistence_baseline(x_test: np.ndarray) -> np.ndarray:
    """Letzten Glukosewert vorhersagen."""
    if x_test.ndim == 3:
        return x_test[:, -1, 0]
    return x_test[:, -1]


def linear_trend_baseline(x_test: np.ndarray, horizon_steps: int) -> np.ndarray:
    """Lineare Extrapolation."""
    if x_test.ndim == 3:
        g_last = x_test[:, -1, 0]
        g_prev = x_test[:, -2, 0]
    else:
        g_last = x_test[:, -1]
        g_prev = x_test[:, -2]
    
    slope = g_last - g_prev
    return g_last + horizon_steps * slope


# =============================================================================
# TML-Evaluation
# =============================================================================
def run_tml_evaluation(
    feature_set: str = "E0",
    horizons: list[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pro-Patient-Evaluation für TML-Modelle ausführen."""
    if horizons is None:
        horizons = HORIZONS
    
    protocol = BenchmarkProtocol()
    all_per_patient = []
    all_summary = []
    
    models = {
        "Persistence": None,
        "LinearTrend": None,
        "Ridge": build_ridge,
        "RandomForest": build_random_forest,
        "XGBoost": build_xgboost,
    }
    
    for horizon in horizons:
        print(f"\n=== TML: PH={horizon} min ===")
        horizon_steps = horizon // STEP_MINUTES
        
        x, y, patient_ids, feature_cols, stats = build_global_dataset(
            feature_set=feature_set,
            as_sequence=True,
            horizon_steps=horizon_steps,
            protocol=protocol,
            return_stats=True,
        )
        
        x_train, x_val, x_test, y_train, y_val, y_test, pid_test = split_with_patient_ids(
            x, y, patient_ids,
            history_steps=HISTORY_STEPS,
            horizon_steps=horizon_steps,
        )
        
        n_train, n_test = len(x_train), len(x_test)
        x_train_flat = x_train.reshape(n_train, -1)
        x_test_flat = x_test.reshape(n_test, -1)
        
        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train_flat)
        x_test_sc = scaler.transform(x_test_flat)
        
        for model_name, builder in models.items():
            print(f"  {model_name}...", end=" ")
            start = time.time()
            
            if model_name == "Persistence":
                y_pred = persistence_baseline(x_test)
            elif model_name == "LinearTrend":
                y_pred = linear_trend_baseline(x_test, horizon_steps)
            else:
                model = builder(random_state=SEED)
                model.fit(x_train_sc, y_train)
                y_pred = model.predict(x_test_sc)
            
            pp_df = calculate_per_patient_metrics(y_test, y_pred, pid_test)
            pp_df["model"] = model_name
            pp_df["horizon"] = horizon
            pp_df["feature_set"] = feature_set
            all_per_patient.append(pp_df)
            
            rmse_micro = np.sqrt(mean_squared_error(y_test, y_pred))
            macro = macro_average(pp_df)
            rmse_ci = bootstrap_ci(pp_df["rmse"].values)
            mae_ci = bootstrap_ci(pp_df["mae"].values)
            
            summary_row = {
                "model": model_name,
                "horizon": horizon,
                "feature_set": feature_set,
                "n_patients": len(pp_df),
                "n_test_samples": len(y_test),
                "rmse_micro": rmse_micro,
                "mae_micro": mean_absolute_error(y_test, y_pred),
                "rmse_macro": macro["rmse_macro"],
                "mae_macro": macro["mae_macro"],
                "rmse_std": macro["rmse_std"],
                "mae_std": macro["mae_std"],
                "rmse_ci_lower": rmse_ci["ci_lower"],
                "rmse_ci_upper": rmse_ci["ci_upper"],
                "mae_ci_lower": mae_ci["ci_lower"],
                "mae_ci_upper": mae_ci["ci_upper"],
            }
            all_summary.append(summary_row)
            
            elapsed = time.time() - start
            print(f"RMSE macro={macro['rmse_macro']:.2f} [{rmse_ci['ci_lower']:.2f}, {rmse_ci['ci_upper']:.2f}] ({elapsed:.1f}s)")
    
    return pd.concat(all_per_patient, ignore_index=True), pd.DataFrame(all_summary)


# =============================================================================
# DL-Evaluation (LSTM)
# =============================================================================
def run_dl_evaluation(
    feature_set: str = "E1",
    horizons: list[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pro-Patient-Evaluation für LSTM ausführen."""
    # Lazy-Import, um TensorFlow nur bei Bedarf zu laden
    import tensorflow as tf
    from src.training import scale_features_3d, get_callbacks
    from src.models import build_lstm_model
    
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    if horizons is None:
        horizons = [30, 60]
    
    protocol = BenchmarkProtocol()
    all_per_patient = []
    all_summary = []
    
    for horizon in horizons:
        print(f"\n=== LSTM/E1: PH={horizon} min ===")
        horizon_steps = horizon // STEP_MINUTES
        
        x, y, patient_ids, feature_cols, stats = build_global_dataset(
            feature_set=feature_set,
            as_sequence=True,
            horizon_steps=horizon_steps,
            protocol=protocol,
            return_stats=True,
        )
        
        x_train, x_val, x_test, y_train, y_val, y_test, pid_test = split_with_patient_ids(
            x, y, patient_ids,
            history_steps=HISTORY_STEPS,
            horizon_steps=horizon_steps,
        )
        
        print(f"Dataset: {len(x):,} windows, {stats['n_patients']} patients")
        print(f"Train: {len(x_train):,}, Val: {len(x_val):,}, Test: {len(x_test):,}")
        
        x_train_sc, x_test_sc, scaler = scale_features_3d(x_train, x_test)
        x_val_sc = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)
        
        n_steps, n_features = x_train_sc.shape[1], x_train_sc.shape[2]
        model = build_lstm_model(n_steps, n_features)
        
        print("Training LSTM...", flush=True)
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
        print(f"Trained {epochs_trained} epochs in {elapsed:.1f}s")
        
        y_pred = model.predict(x_test_sc, verbose=0).flatten()
        
        pp_df = calculate_per_patient_metrics(y_test, y_pred, pid_test)
        pp_df["model"] = "LSTM"
        pp_df["horizon"] = horizon
        pp_df["feature_set"] = feature_set
        all_per_patient.append(pp_df)
        
        rmse_micro = np.sqrt(mean_squared_error(y_test, y_pred))
        macro = macro_average(pp_df)
        rmse_ci = bootstrap_ci(pp_df["rmse"].values)
        mae_ci = bootstrap_ci(pp_df["mae"].values)
        
        summary_row = {
            "model": "LSTM",
            "horizon": horizon,
            "feature_set": feature_set,
            "n_patients": len(pp_df),
            "n_test_samples": len(y_test),
            "epochs_trained": epochs_trained,
            "fit_seconds": elapsed,
            "rmse_micro": rmse_micro,
            "mae_micro": mean_absolute_error(y_test, y_pred),
            "rmse_macro": macro["rmse_macro"],
            "mae_macro": macro["mae_macro"],
            "rmse_std": macro["rmse_std"],
            "mae_std": macro["mae_std"],
            "rmse_ci_lower": rmse_ci["ci_lower"],
            "rmse_ci_upper": rmse_ci["ci_upper"],
            "mae_ci_lower": mae_ci["ci_lower"],
            "mae_ci_upper": mae_ci["ci_upper"],
        }
        all_summary.append(summary_row)
        
        print(f"RMSE macro={macro['rmse_macro']:.2f} [{rmse_ci['ci_lower']:.2f}, {rmse_ci['ci_upper']:.2f}]")
    
    return pd.concat(all_per_patient, ignore_index=True), pd.DataFrame(all_summary)


# =============================================================================
# Hauptprogramm
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Per-patient evaluation for TML and DL models")
    parser.add_argument("--dl", action="store_true", help="Include LSTM evaluation")
    parser.add_argument("--dl-only", action="store_true", help="Run only LSTM evaluation")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Per-Patient Evaluation: Macro-Average and Bootstrap CI")
    print("=" * 60)
    
    all_per_patient = []
    all_summary = []
    
    # TML-Evaluation
    if not args.dl_only:
        pp_tml, summary_tml = run_tml_evaluation(feature_set="E0", horizons=HORIZONS)
        all_per_patient.append(pp_tml)
        all_summary.append(summary_tml)
    
    # DL-Evaluation
    if args.dl or args.dl_only:
        pp_dl, summary_dl = run_dl_evaluation(feature_set="E1", horizons=[30, 60])
        all_per_patient.append(pp_dl)
        all_summary.append(summary_dl)
    
    # Zusammenführen und speichern
    per_patient_df = pd.concat(all_per_patient, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)
    
    pp_path = CSV_DIR / "per_patient_metrics.csv"
    summary_path = CSV_DIR / "macro_average_summary.csv"
    
    per_patient_df.to_csv(pp_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n✓ Per-patient metrics saved to: {pp_path}")
    print(f"✓ Summary with CI saved to: {summary_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY: Micro vs Macro RMSE (with 95% CI)")
    print("=" * 60)
    
    cols = ["model", "horizon", "rmse_micro", "rmse_macro", "rmse_ci_lower", "rmse_ci_upper"]
    print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
