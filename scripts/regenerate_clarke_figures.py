"""Clarke-EGA-Abbildungen aus gespeicherten Vorhersagen regenerieren."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.metrics import clarke_error_grid_analysis
from src.visualization import plot_clarke_error_grid, ZONE_COLORS

# Pfade
PREDICTIONS_DIR = Path(__file__).parent.parent / "results" / "predictions"
FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures" / "thesis"


def main():
    """Alle Clarke-EGA-Abbildungen aus gespeicherten Vorhersagen regenerieren."""
    print("=" * 60)
    print("Regenerating Clarke EGA Figures (Colored)")
    print("=" * 60)

    # Alle Vorhersage-Dateien finden
    pred_files = sorted(PREDICTIONS_DIR.glob("*.csv"))

    if not pred_files:
        print("No prediction files found!")
        return

    print(f"Found {len(pred_files)} prediction files\n")

    all_results = []

    for pred_file in pred_files:
        # Dateinamen parsen: model_featureset_phX.csv
        name_parts = pred_file.stem.split('_')
        model = name_parts[0].title()
        if model == 'Randomforest':
            model = 'RandomForest'
        elif model == 'Xgboost':
            model = 'XGBoost'

        # Horizont aus Dateiname extrahieren
        horizon = int(name_parts[-1].replace('ph', ''))
        feature_set = name_parts[1].upper()

        # Vorhersagen laden
        df = pd.read_csv(pred_file)
        y_true = df['y_true'].values
        y_pred = df['y_pred'].values

        # Titel und Dateiname erstellen
        title = f"Clarke EGA: {model}/{feature_set} (PH={horizon})"
        fig_name = f"clarke_ega_{model.lower()}_{feature_set.lower()}_ph{horizon}.png"

        # Plot erzeugen
        plot_clarke_error_grid(y_true, y_pred, title, FIGURES_DIR / fig_name)

        # Für gestapeltes Balkendiagramm speichern
        clarke = clarke_error_grid_analysis(y_true, y_pred)
        all_results.append({
            'label': f"{model}/{feature_set} (PH={horizon})",
            'horizon': horizon,
            'clarke': clarke,
        })

    # Gestapelte Balkendiagramme erzeugen
    print("\nGenerating stacked bar charts...")

    for horizon in [30, 60]:
        ph_results = [r for r in all_results if r['horizon'] == horizon]
        if ph_results:
            fig, ax = plt.subplots(figsize=(12, 6))

            models = [r["label"] for r in ph_results]
            zones = ["A", "B", "C", "D", "E"]

            bottom = np.zeros(len(models))

            for zone in zones:
                values = [r["clarke"][zone] for r in ph_results]
                bars = ax.bar(models, values, bottom=bottom, label=f'Zone {zone}',
                             color=ZONE_COLORS[zone], edgecolor="black", linewidth=0.5)

                for j, (bar, val) in enumerate(zip(bars, values)):
                    if val > 3:
                        ax.text(bar.get_x() + bar.get_width()/2, bottom[j] + val/2,
                               f"{val:.1f}%", ha="center", va="center", fontsize=9,
                               fontweight="bold" if zone == "A" else "normal",
                               color='white' if zone in ('A', 'E') else 'black')

                bottom += values

            ax.set_ylabel("Percentage (%)", fontsize=12)
            ax.set_title(f"Clarke Error Grid Zone Distribution (PH={horizon} min)", fontsize=14, fontweight='bold')
            ax.legend(title="Zone", loc="upper right")
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3, axis="y")

            plt.xticks(rotation=15, ha="right")
            plt.tight_layout()

            filepath = FIGURES_DIR / f"clarke_zones_stacked_ph{horizon}.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"  ✓ Saved: {filepath.name}")

    print("\n" + "=" * 60)
    print("✓ All Clarke EGA figures regenerated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
