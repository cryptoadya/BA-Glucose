"""Metriken zur Bewertung der Glukosevorhersagequalität."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Standard-Regressionsmetriken berechnen."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def clarke_zone(true_val: float, pred_val: float) -> str:
  """
  Clarke-Error-Grid-Zone für ein Wertepaar (Referenz=true_val, Vorhersage=pred_val).
  """

  r = float(true_val)
  p = float(pred_val)

  # Zone A: innerhalb 20% ODER beide in Hypoglykämie (<=70)
  if (r <= 70 and p <= 70) or (1.2 * r >= p >= 0.8 * r):
    return "A"

  # Zone E: gegenteilige Behandlung
  if (r >= 180 and p <= 70) or (r <= 70 and p >= 180):
    return "E"

  # Zone C: Überkorrektur
  if ((70 <= r <= 290) and (p >= r + 110)) or ((130 <= r <= 180) and (p <= (7 / 5) * r - 182)):
    return "C"

  # Zone D: fehlende Erkennung
  if (r >= 240 and (70 <= p <= 180)) or (r <= 175 / 3 and (180 >= p >= 70)) or (
    (175 / 3 <= r <= 70) and (p >= (6 / 5) * r)):
    return "D"

  # Sonst Zone B
  return "B"


def clarke_error_grid_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | list[str]]:
    """Clarke-Error-Grid-Zonenanteile berechnen."""
    zones = [clarke_zone(t, p) for t, p in zip(y_true, y_pred)]
    total = len(zones)

    return {
        "A": zones.count("A") / total * 100,
        "B": zones.count("B") / total * 100,
        "C": zones.count("C") / total * 100,
        "D": zones.count("D") / total * 100,
        "E": zones.count("E") / total * 100,
        "AB": (zones.count("A") + zones.count("B")) / total * 100,
        "zones": zones,
    }


def glycemic_category(glucose: float) -> str:
    """Glukosewert in klinische Kategorie einordnen."""
    if glucose < 54:
        return "severe_hypo"
    elif glucose < 70:
        return "hypo"
    elif glucose <= 180:
        return "normal"
    elif glucose <= 250:
        return "hyper"
    else:
        return "severe_hyper"


def _category_sensitivity(
    true_cats: list[str],
    pred_cats: list[str],
    target_cats: list[str],
) -> float:
    """Sensitivität für bestimmte Kategorien."""
    true_positives = sum(
        t in target_cats and p in target_cats for t, p in zip(true_cats, pred_cats)
    )
    actual_positives = sum(t in target_cats for t in true_cats)
    if actual_positives == 0:
        return 0.0
    return true_positives / actual_positives * 100


def category_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Genauigkeit der glykämischen Kategorienvorhersage berechnen."""
    true_cats = [glycemic_category(v) for v in y_true]
    pred_cats = [glycemic_category(v) for v in y_pred]

    correct = sum(t == p for t, p in zip(true_cats, pred_cats))
    total = len(true_cats)

    return {
        "category_accuracy": correct / total * 100,
        "hypo_sensitivity": _category_sensitivity(true_cats, pred_cats, ["hypo", "severe_hypo"]),
        "hyper_sensitivity": _category_sensitivity(true_cats, pred_cats, ["hyper", "severe_hyper"]),
    }


def full_evaluation(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Vollständige Evaluation mit allen Metriken."""
    basic = calculate_metrics(y_true, y_pred)
    clarke = clarke_error_grid_analysis(y_true, y_pred)
    category = category_accuracy(y_true, y_pred)

    clarke_clean = {k: v for k, v in clarke.items() if k != "zones"}
    return {**basic, **clarke_clean, **category}
