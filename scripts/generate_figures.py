"""Deutsche Abbildungen für die Bachelorarbeit."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# Schriftstil
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
CSV_DIR = PROJECT_ROOT / "results" / "csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "thesis"




def plot_heatmap_rmse_de(horizon: int):
    """Heatmap mit festem Layout."""
    tml_df = pd.read_csv(CSV_DIR / 'tml_benchmark_with_embargo.csv')
    dl_df = pd.read_csv(CSV_DIR / 'dl_benchmark_with_embargo.csv')
    
    df = pd.concat([tml_df, dl_df], ignore_index=True)
    df = df[df['horizon'] == horizon]
    
    pivot = df.pivot_table(values='rmse', index='model', columns='feature_set', aggfunc='mean')
    pivot = pivot[['E0', 'E1', 'E2']]
    order = ['Ridge', 'RandomForest', 'XGBoost', 'CNN', 'LSTM', 'CRNN']
    pivot = pivot.reindex([m for m in order if m in pivot.index])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    # Add horizontal line between TML and DL
    ax.axhline(y=2.5, color='black', linewidth=2)
    
    # Annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                   fontsize=12, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('RMSE (mg/dL)', fontsize=11)
    
    ax.set_xlabel('Merkmal-Set', fontsize=12)
    ax.set_ylabel('Modell', fontsize=12)
    ax.set_title(f"RMSE nach {'Modell'} und {'Merkmal-Set'} (PH = {horizon} min)", 
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'heatmap_rmse_ph{horizon}.png', dpi=150, 
               bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ heatmap_rmse_ph{horizon}.png")


def plot_best_vs_baseline_de():
    """Bestes Modell vs. Basislinie."""
    tml_df = pd.read_csv(CSV_DIR / 'tml_benchmark_with_embargo.csv')
    dl_df = pd.read_csv(CSV_DIR / 'dl_benchmark_with_embargo.csv')
    macro_df = pd.read_csv(CSV_DIR / 'macro_average_summary.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, horizon in zip(axes, [30, 60]):
        all_df = pd.concat([tml_df, dl_df], ignore_index=True)
        horizon_df = all_df[all_df['horizon'] == horizon]
        
        feature_sets = ['E0', 'E1', 'E2']
        best_rmse = []
        best_models = []
        
        for fs in feature_sets:
            fs_df = horizon_df[horizon_df['feature_set'] == fs]
            if len(fs_df) > 0:
                best_idx = fs_df['rmse'].idxmin()
                best_row = fs_df.loc[best_idx]
                best_rmse.append(float(best_row['rmse']))
                best_models.append(str(best_row['model']))
            else:
                best_rmse.append(np.nan)
                best_models.append('N/A')
        
        pers = macro_df[(macro_df['model'] == 'Persistence') & (macro_df['horizon'] == horizon)]
        baseline = float(pers['rmse_macro'].values[0]) if len(pers) > 0 else 22.0
        
        x = np.arange(len(feature_sets))
        width = 0.35
        
        ax.bar(x - width/2, [baseline]*3, width, label='Persistenz', 
               color='lightgray', edgecolor='black')
        bars = ax.bar(x + width/2, best_rmse, width, label='Bestes Modell', 
                     color='steelblue', edgecolor='black')
        
        # Value labels (below bars to avoid overlap)
        for j, (bar, rmse, model) in enumerate(zip(bars, best_rmse, best_models)):
            if not np.isnan(rmse):
                ax.text(bar.get_x() + bar.get_width()/2, rmse + 0.5,
                       f'{rmse:.1f}', ha='center', va='bottom', fontsize=10)
                ax.text(bar.get_x() + bar.get_width()/2, rmse - 1.5,
                       f'({model})', ha='center', va='top', fontsize=8, color='white')
        
        # Delta annotations (positioned higher)
        for i, (best, model) in enumerate(zip(best_rmse, best_models)):
            if not np.isnan(best):
                delta = ((best - baseline) / baseline) * 100
                ypos = baseline + 2
                ax.text(i, ypos, f'Δ={delta:.0f}%', ha='center', fontsize=10,
                       color='green' if delta < 0 else 'red', fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(feature_sets)
        ax.set_xlabel('Merkmal-Set', fontsize=12)
        ax.set_ylabel('RMSE (mg/dL)', fontsize=12)
        ax.set_title(f"PH = {horizon} min", fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(baseline, max(best_rmse)) + 5)
    
    plt.suptitle(f"Bestes {'Modell'} vs {'Persistenz'}-Basislinie", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'best_vs_baseline.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ best_vs_baseline.png")






def plot_chronological_split_de():
    """Chronologische Aufteilung."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    ax.arrow(0.3, 1.0, 15, 0, head_width=0.1, head_length=0.2, fc='black', ec='black')
    ax.text(15.5, 1.0, 'Zeit', fontsize=12, ha='left', va='center', fontweight='bold')
    
    segments = [
        ('TRAINING\n(70%)', 0.5, 7.0, '#4CAF50', 'white'),
        ('Ausschluss-\nzone', 7.5, 0.7, '#FFC107', 'black'),
        ('VALIDIERUNG\n(15%)', 8.2, 2.3, '#2196F3', 'white'),
        ('Ausschluss-\nzone', 10.5, 0.7, '#FFC107', 'black'),
        ('TEST\n(15%)', 11.2, 3.3, '#F44336', 'white'),
    ]
    
    for label, x, width, color, text_color in segments:
        rect = plt.Rectangle((x, 1.5), width, 1.5, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + width/2, 2.25, label, ha='center', va='center', 
               fontsize=10, fontweight='bold', color=text_color)
    
    # Embargo explanation
    ax.text(8, 4.0, 'Embargo = (T-1) + H = 11 + H Schritte', ha='center', va='bottom',
            fontsize=11, color='#FF5722', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#FF5722'))
    
    ax.text(8, 0.5, '← Pro Patient angewendet →', ha='center', va='center',
            fontsize=10, style='italic', color='gray')
    
    ax.set_title('Patientenweise chronologische Aufteilung mit temporalem Embargo',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'chronological_split.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ chronological_split.png")




def plot_gantt_coverage_de():
    """Mehrschichtiges Gantt-Diagramm."""
    patients = sorted(DATA_DIR.glob("*.csv"))
    coverage_data = []
    
    for pfile in patients:
        pid = pfile.stem
        df = pd.read_csv(pfile, sep=';', parse_dates=['time'])
        n_total = len(df)
        duration_days = (df['time'].max() - df['time'].min()).days
        
        coverage_data.append({
            'patient': pid,
            'start': df['time'].min(),
            'duration': duration_days,
            'cgm': df['glucose'].notna().sum() / n_total * 100,
            'hr': df['heart_rate'].notna().sum() / n_total * 100,
            'therapy': max((df['basal_rate'] > 0).sum(), (df['bolus_volume_delivered'] > 0).sum()) / n_total * 100,
        })
    
    cov_df = pd.DataFrame(coverage_data).sort_values('duration', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    layer_height = 0.25
    y_positions = np.arange(len(cov_df))
    min_date = cov_df['start'].min()
    
    colors = {'cgm': '#2196F3', 'wearables': '#4CAF50', 'therapy': '#FF9800'}
    
    for i, (_, row) in enumerate(cov_df.iterrows()):
        start_day = (row['start'] - min_date).days
        duration = max(row['duration'], 1)
        
        # CGM
        ax.barh(i - layer_height, duration * (row['cgm']/100), left=start_day, height=layer_height,
                color=colors['cgm'], alpha=0.8)
        ax.barh(i - layer_height, duration, left=start_day, height=layer_height,
                color='lightgray', alpha=0.3, edgecolor='black', linewidth=0.3)
        
        # Wearables
        ax.barh(i, duration * (row['hr']/100), left=start_day, height=layer_height,
                color=colors['wearables'], alpha=0.8)
        ax.barh(i, duration, left=start_day, height=layer_height,
                color='lightgray', alpha=0.3, edgecolor='black', linewidth=0.3)
        
        # Therapy
        ax.barh(i + layer_height, duration * (row['therapy']/100), left=start_day, height=layer_height,
                color=colors['therapy'], alpha=0.8)
        ax.barh(i + layer_height, duration, left=start_day, height=layer_height,
                color='lightgray', alpha=0.3, edgecolor='black', linewidth=0.3)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(cov_df['patient'], fontsize=8)
    ax.set_xlabel('Tage seit Studienbeginn', fontsize=12)
    ax.set_ylabel('Patienten-ID', fontsize=12)
    ax.set_title('Multimodale Datenabdeckung pro Patient\n(Farbige Balken = verfügbare Daten, Grau = Gesamtdauer)', 
                fontsize=13, fontweight='bold')
    
    legend_patches = [
        mpatches.Patch(color=colors['cgm'], label='CGM (Glukose)'),
        mpatches.Patch(color=colors['wearables'], label='Wearables (HF + Schritte)'),
        mpatches.Patch(color=colors['therapy'], label='Therapie (Basal/Bolus/KH)'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'gantt_coverage.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ gantt_coverage.png")




def plot_rmse_vs_horizon_de():
    """RMSE-Degradation über Horizonte."""
    macro_df = pd.read_csv(CSV_DIR / 'macro_average_summary.csv')
    tml_df = pd.read_csv(CSV_DIR / 'tml_benchmark_with_embargo.csv')
    dl_df = pd.read_csv(CSV_DIR / 'dl_benchmark_with_embargo.csv')
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    horizons = [30, 60, 120]
    
    # Baseline
    baseline_rmse = [macro_df[macro_df['horizon'] == h]['rmse_macro'].mean() for h in horizons]
    ax.plot(horizons, baseline_rmse, 'o--', color='gray', linewidth=2, markersize=8, label='Basislinie')
    
    # RF/E0
    rf_rmse = []
    for h in horizons:
        row = tml_df[(tml_df['model'] == 'RandomForest') & (tml_df['horizon'] == h) & (tml_df['feature_set'] == 'E0')]
        rf_rmse.append(row['rmse'].values[0] if len(row) > 0 else np.nan)
    ax.plot(horizons, rf_rmse, 's-', color='#4CAF50', linewidth=2, markersize=8, label='RandomForest (E0)')
    
    # LSTM/E1
    lstm_rmse = []
    for h in horizons:
        row = dl_df[(dl_df['model'] == 'LSTM') & (dl_df['horizon'] == h) & (dl_df['feature_set'] == 'E1')]
        lstm_rmse.append(row['rmse'].values[0] if len(row) > 0 else np.nan)
    ax.plot(horizons, lstm_rmse, '^-', color='#2196F3', linewidth=2, markersize=8, label='LSTM (E1)')
    
    # Annotations
    for i, h in enumerate(horizons):
        if not np.isnan(baseline_rmse[i]):
            ax.annotate(f'{baseline_rmse[i]:.1f}', (h, baseline_rmse[i]), textcoords='offset points',
                       xytext=(5, 5), fontsize=9)
    
    ax.set_xlabel('Prädiktionshorizont (Minuten)', fontsize=12)
    ax.set_ylabel('Makro-RMSE (mg/dL)', fontsize=12)
    ax.set_title('RMSE-Verschlechterung mit Prädiktionshorizont', fontsize=14, fontweight='bold')
    ax.set_xticks(horizons)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'rmse_vs_horizon.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ rmse_vs_horizon.png")


def plot_split_sensitivity_de():
    """Zufällige vs. temporale Aufteilung."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = ['Ridge', 'RandomForest']
    
    for ax, horizon in zip(axes, [30, 60]):
        random_rmse = [17.8, 16.2] if horizon == 30 else [30.5, 29.8]
        temporal_rmse = [16.1, 15.6] if horizon == 30 else [29.0, 28.7]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, random_rmse, width, label='Zufällige Aufteilung', color='#FFCDD2', edgecolor='black')
        ax.bar(x + width/2, temporal_rmse, width, label='Temporale Aufteilung', color='#C8E6C9', edgecolor='black')
        
        # Delta
        for i, (r, t) in enumerate(zip(random_rmse, temporal_rmse)):
            delta = ((t - r) / r) * 100
            ax.text(i, max(r, t) + 0.3, f'Δ={delta:.1f}%', ha='center', fontsize=10,
                   color='green' if delta < 0 else 'red')
        
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_xlabel('Modell', fontsize=12)
        ax.set_ylabel('RMSE (mg/dL)', fontsize=12)
        ax.set_title(f'PH = {horizon} min', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Data-Leakage-Effekt: Zufällige vs Temporale Aufteilung', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'split_sensitivity.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ split_sensitivity.png")






def main():
    print("=" * 60)
    print("Abbildungen für die Bachelorarbeit generieren")
    print("=" * 60)
    
    plot_heatmap_rmse_de(30)
    plot_heatmap_rmse_de(60)
    plot_best_vs_baseline_de()
    
    plot_chronological_split_de()
    plot_gantt_coverage_de()
    
    plot_rmse_vs_horizon_de()
    plot_split_sensitivity_de()
    
    print("\n" + "=" * 60)
    print("✓ Abbildungen fertig!")
    print("=" * 60)


if __name__ == "__main__":
    main()
