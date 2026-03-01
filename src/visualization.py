"""Visualisierungsfunktionen für Ergebnisse."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import clarke_zone, clarke_error_grid_analysis


# ─────────────────────────────────────────────────────────────
# Clarke Error Grid
# ─────────────────────────────────────────────────────────────

# Kanonische Zonenfarben
ZONE_COLORS: dict[str, str] = {
    "A": "#2E7D32",  # Dark green
    "B": "#8BC34A",  # Light green
    "C": "#FFC107",  # Amber
    "D": "#FF5722",  # Deep orange
    "E": "#B71C1C",  # Dark red
}


def _draw_clarke_zone_boundaries(ax: plt.Axes, lw: float = 1.5) -> None:
    """Kanonische Clarke-EGA-Zonengrenzen zeichnen."""
    ax.plot([0, 175 / 3], [70, 70], "-", c="black", lw=lw)
    ax.plot([175 / 3, 400 / 1.2], [70, 400], "-", c="black", lw=lw)
    ax.plot([70, 70], [84, 400], "-", c="black", lw=lw)
    ax.plot([0, 70], [180, 180], "-", c="black", lw=lw)
    ax.plot([70, 290], [180, 400], "-", c="black", lw=lw)
    ax.plot([70, 70], [0, 56], "-", c="black", lw=lw)
    ax.plot([70, 400], [56, 320], "-", c="black", lw=lw)
    ax.plot([180, 180], [0, 70], "-", c="black", lw=lw)
    ax.plot([180, 400], [70, 70], "-", c="black", lw=lw)
    ax.plot([240, 240], [70, 180], "-", c="black", lw=lw)
    ax.plot([240, 400], [180, 180], "-", c="black", lw=lw)
    ax.plot([130, 180], [0, 70], "-", c="black", lw=lw)


def _add_zone_labels(ax: plt.Axes, fontsize: int = 18) -> None:
    """Zonenbeschriftungen mit kanonischen Farben."""
    positions = [
        (30, 15, "A"), (370, 260, "B"), (280, 370, "B"),
        (160, 370, "C"), (160, 15, "C"),
        (30, 140, "D"), (370, 120, "D"),
        (30, 370, "E"), (370, 15, "E"),
    ]
    for x, y, zone in positions:
        ax.text(x, y, zone, fontsize=fontsize, fontweight="bold",
                color=ZONE_COLORS[zone])


def plot_clarke_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Clarke Error Grid",
    filepath: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 10),
) -> None:
    """
    Clarke-Error-Grid mit farbigen Punkten nach Zone plotten.

    Wird entweder in *filepath* gespeichert oder via ``plt.show()`` angezeigt.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    clarke = clarke_error_grid_analysis(y_true, y_pred)
    zones = np.array([clarke_zone(t, p) for t, p in zip(y_true, y_pred)])

    fig, ax = plt.subplots(figsize=figsize)

    # Punkte nach Zone (A oben)
    for zone in ["E", "D", "C", "B", "A"]:
        mask = zones == zone
        if mask.sum() > 0:
            ax.scatter(
                y_true[mask], y_pred[mask],
                c=ZONE_COLORS[zone], s=10, alpha=0.5, edgecolors="none",
                label=f"Zone {zone} ({mask.sum():,} pts, "
                      f"{mask.sum() / len(y_true) * 100:.1f}%)",
            )

    ax.set_title(f"{title}\nA+B = {clarke['AB']:.1f}%",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Referenz-Glukose (mg/dL)", fontsize=12)
    ax.set_ylabel("Vorhergesagte Glukose (mg/dL)", fontsize=12)
    ax.set_xlim([0, 400])
    ax.set_ylim([0, 400])
    ax.set_aspect(1.0)
    ax.set_facecolor("white")

    # Diagonale
    ax.plot([0, 400], [0, 400], "--", c="gray", linewidth=1.5, alpha=0.7)

    _draw_clarke_zone_boundaries(ax)
    _add_zone_labels(ax)

    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  ✓ Saved: {filepath.name}")
    else:
        plt.show()


def plot_clarke_zones_stacked(
    all_results: list[dict],
    filepath: Optional[Path] = None,
    title: str = "Clarke Error Grid Zone Distribution",
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """
    Gestapeltes Balkendiagramm der Clarke-Zonenverteilung.

    *all_results* ist eine Liste von Dicts mit Schlüsseln ``"label"`` und
    ``"clarke"`` (Ergebnis von ``clarke_error_grid_analysis``).
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = [r["label"] for r in all_results]
    zone_keys = ["A", "B", "C", "D", "E"]

    bottom = np.zeros(len(labels))

    for zone in zone_keys:
        values = [r["clarke"][zone] for r in all_results]
        bars = ax.bar(
            labels, values, bottom=bottom,
            label=f"Zone {zone}", color=ZONE_COLORS[zone],
            edgecolor="black", linewidth=0.5,
        )
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val > 3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom[j] + val / 2,
                    f"{val:.1f}%", ha="center", va="center", fontsize=9,
                    fontweight="bold" if zone == "A" else "normal",
                    color="white" if zone in ("A", "E") else "black",
                )
        bottom += values

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(title="Zone", loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  ✓ Saved: {filepath.name}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────
# Allgemeine Hilfsplots
# ─────────────────────────────────────────────────────────────

def plot_training_history(
    history: dict,
    title: str = "Training History",
    figsize: tuple[int, int] = (12, 4),
) -> None:
    """Trainings-Loss- und MAE-Kurven plotten."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(history["loss"], label="Train", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title(f"{title} — Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "mae" in history:
        axes[1].plot(history["mae"], label="Train", linewidth=2)
        if "val_mae" in history:
            axes[1].plot(history["val_mae"], label="Validation", linewidth=2)
        axes[1].set_ylabel("MAE (mg/dL)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title(f"{title} — MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_clarke_zones_bar(
    clarke_results: dict[str, float],
    title: str = "Clarke Error Grid Distribution",
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """Balkendiagramm der Clarke-Zonenanteile."""
    zones = ["A", "B", "C", "D", "E"]
    values = [clarke_results.get(z, 0) for z in zones]
    colors = [ZONE_COLORS[z] for z in zones]

    plt.figure(figsize=figsize)
    bars = plt.bar(zones, values, color=colors, alpha=0.7, edgecolor="black")

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            fontsize=10,
        )

    plt.ylabel("Percentage (%)", fontsize=12)
    plt.xlabel("Clarke Zone", fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim([0, max(values) * 1.15])
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_points: int = 200,
    title: str = "Predictions vs Actual",
    figsize: tuple[int, int] = (14, 5),
) -> None:
    """Zeitreihenplot: Vorhersagen vs. tatsächliche Werte."""
    plt.figure(figsize=figsize)

    idx = range(min(n_points, len(y_true)))

    plt.plot(idx, y_true[: len(idx)], label="Actual", linewidth=2, alpha=0.8)
    plt.plot(idx, y_pred[: len(idx)], label="Predicted", linewidth=2, alpha=0.8)

    plt.axhline(y=70, color="red", linestyle="--", alpha=0.5, label="Hypo threshold")
    plt.axhline(y=180, color="orange", linestyle="--", alpha=0.5, label="Hyper threshold")
    plt.fill_between(idx, 70, 180, alpha=0.1, color="green", label="Target range")

    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Glucose (mg/dL)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "rmse",
    title: str = "Model Comparison",
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Balkendiagramm zum Modellvergleich."""
    df_sorted = results_df.sort_values(metric)

    plt.figure(figsize=figsize)
    colors = plt.colormaps["viridis"](np.linspace(0.2, 0.8, len(df_sorted)))

    bars = plt.barh(df_sorted["model"], df_sorted[metric], color=colors, alpha=0.8)

    for bar, val in zip(bars, df_sorted[metric]):
        plt.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=10)

    plt.xlabel(metric.upper(), fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()


def plot_patient_results(
    df: pd.DataFrame,
    metric: str = "rmse",
    figsize: tuple[int, int] = (14, 6),
) -> None:
    """Pro-Patient-Ergebnisse plotten."""
    plt.figure(figsize=figsize)

    x = range(len(df))
    plt.bar(x, df[metric], alpha=0.7, edgecolor="black")

    mean_val = df[metric].mean()
    plt.axhline(y=mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}", linewidth=2)

    plt.xticks(x, df["patient"], rotation=45, ha="right", fontsize=8)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.title(f"Per-Patient {metric.upper()}", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()
