"""Globale Konstanten und Hyperparameter."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
CSV_DIR = RESULTS_DIR / "csv"
FIGURES_DIR = RESULTS_DIR / "figures"

def ensure_dirs() -> None:
    """Ausgabeverzeichnisse erstellen, falls nicht vorhanden."""
    for dir_path in [RESULTS_DIR, MODELS_DIR, CSV_DIR, FIGURES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


ensure_dirs()

# Zeitkonstanten
STEP_MINUTES: int = 5
HISTORY_MINUTES: int = 60
HORIZON_MINUTES: int = 30
HORIZONS: list[int] = [30, 60, 120]
HISTORY_STEPS: int = HISTORY_MINUTES // STEP_MINUTES
HORIZON_STEPS: int = HORIZON_MINUTES // STEP_MINUTES

# Vorverarbeitung
MAX_CGM_GAP_MINUTES: int = 30

# DL-Training
EPOCHS: int = 50
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.001
ES_PATIENCE: int = 10
ES_MIN_DELTA: float = 0.001

# Modell-Hyperparameter
LSTM_UNITS: tuple[int, int] = (64, 32)
LSTM_DROPOUT: float = 0.2
CNN_FILTERS: tuple[int, int] = (64, 32)
CNN_KERNEL_SIZE: int = 3

XGB_N_ESTIMATORS: int = 200
XGB_MAX_DEPTH: int = 6
XGB_LEARNING_RATE: float = 0.1

SEED: int = 42

