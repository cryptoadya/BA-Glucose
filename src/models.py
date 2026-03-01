"""Neuronale Netzarchitekturen für die Glukosevorhersage."""
from __future__ import annotations

import keras
from keras import layers


def build_lstm_model(
    n_steps: int,
    n_features: int,
    lstm_units: tuple[int, int] = (64, 32),
    dropout: float = 0.2,
    dense_units: int = 16,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Zweischichtiges LSTM-Modell."""
    model = keras.Sequential([
        layers.Input(shape=(n_steps, n_features)),
        layers.LSTM(lstm_units[0], return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(lstm_units[1]),
        layers.Dropout(dropout),
        layers.Dense(dense_units, activation="relu"),
        layers.Dense(1),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model





def build_cnn_model(
    n_steps: int,
    n_features: int,
    filters: tuple[int, int] = (64, 32),
    kernel_size: int = 3,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> keras.Model:
    """1D-CNN-Modell für Zeitreihen."""
    model = keras.Sequential([
        layers.Input(shape=(n_steps, n_features)),
        layers.Conv1D(filters[0], kernel_size=kernel_size, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters[1], kernel_size=kernel_size, activation="relu", padding="same"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_crnn_model(
    n_steps: int,
    n_features: int,
    filters: int = 64,
    lstm_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> keras.Model:
    """CNN + LSTM Hybridmodell."""
    model = keras.Sequential([
        layers.Input(shape=(n_steps, n_features)),
        layers.Conv1D(filters, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(lstm_units, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model





MODEL_REGISTRY = {
    "lstm": build_lstm_model,
    "cnn": build_cnn_model,
    "crnn": build_crnn_model,
}


def get_model(name: str, n_steps: int, n_features: int, **kwargs) -> keras.Model:
    """Modell nach Name aus der Registry abrufen."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](n_steps, n_features, **kwargs)
