"""Traditionelle ML-Modelle für die Glukosevorhersage."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


def build_ridge(alpha: float = 1.0, random_state: int = 42) -> Ridge:
    """Ridge-Regression — lineare Basislinie."""
    return Ridge(alpha=alpha, random_state=random_state)


def build_random_forest(
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_leaf: int = 4,
    n_jobs: int = -1,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Random-Forest-Regressor."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def build_xgboost(
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    tree_method: str = "hist",
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> XGBRegressor:
    """XGBoost-Regressor."""
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method=tree_method,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )


ML_MODEL_REGISTRY = {
    "ridge": build_ridge,
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
}


def get_ml_model(name: str, **kwargs) -> Ridge | RandomForestRegressor | XGBRegressor:
    """ML-Modell nach Name abrufen."""
    name = name.lower()
    if name not in ML_MODEL_REGISTRY:
        available = list(ML_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return ML_MODEL_REGISTRY[name](**kwargs)


def get_all_ml_models() -> dict[str, Ridge | RandomForestRegressor | XGBRegressor]:
    """Alle verfügbaren ML-Modelle mit Standardparametern abrufen."""
    return {
        "Ridge": build_ridge(),
        "Random Forest": build_random_forest(),
        "XGBoost": build_xgboost(),
    }


HYPERPARAM_GRIDS = {
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [2, 4, 8],
    },
    "xgboost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
    },
}

