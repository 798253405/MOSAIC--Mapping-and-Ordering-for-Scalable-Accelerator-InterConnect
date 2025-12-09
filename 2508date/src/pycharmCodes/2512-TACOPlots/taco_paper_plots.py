#!/usr/bin/env python3
"""
TACO Paper Plots - Overview + 3 Dimension Figures
- Figure 1: Overview (精华图) - 3 models × 2 NoC × 2 data types
- Figure 2: NoC Size Comparison
- Figure 3: DNN Model Comparison
- Figure 4: Coding Compare
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import os

# Set font (use DejaVu Sans as fallback if Arial not available)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
rcParams['font.size'] = 8

# ============== CONFIGURATION ==============
# 9 methods
METHODS = ['O0', 'O1', 'O2', 'C1', 'C2', 'O1+C1', 'O1+C2', 'O2+C1', 'O2+C2']
METHOD_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12',
                 '#1abc9c', '#e67e22', '#34495e', '#16a085']

# Models
MODELS = ['LeNet', 'DarkNet', 'VGG']

# NoC sizes
NOC_SIZES = ['MC2_4×4', 'MC4_4×4', 'MC4_8×8', 'MC8_8×8']
NOC_SIZES_SHORT = ['MC2_4×4', 'MC8_8×8']  # For overview

# Data types
DATA_TYPES = ['Trained\nFloat-32', 'Trained\nFixed-8', 'Random\nFloat-32', 'Random\nFixed-8']
DATA_TYPES_SHORT = ['Trained\nFloat-32', 'Trained\nFixed-8']  # For overview


def generate_mock_data(baseline=100, seed=42):
    """Generate realistic mock data for BT reduction"""
    np.random.seed(seed)

    # Reduction rates (approximate based on paper)
    reduction_rates = {
        'O0': 0,           # Baseline
        'O1': 15,          # Affiliated ordering ~15%
        'O2': 30,          # Separated ordering ~30%
        'C1': 5,           # Full Bus-Invert ~5%
        'C2': 25,          # Partitioned Bus-Invert ~25%
        'O1+C1': 18,       # Combined
        'O1+C2': 35,       # Combined
        'O2+C1': 33,       # Combined
        'O2+C2': 50,       # Best combined ~50%
    }

    data = {}
    for model in MODELS:
        data[model] = {}
        model_factor = {'LeNet': 1.0, 'DarkNet': 1.1, 'VGG': 0.95}[model]

        for noc in NOC_SIZES:
            data[model][noc] = {}
            noc_factor = 1.0 + NOC_SIZES.index(noc) * 0.05

            for dtype in DATA_TYPES:
                # Fixed-8 has better reduction
                dtype_factor = 1.2 if 'Fixed' in dtype else 1.0
                # Trained has slightly better reduction
                train_factor = 1.05 if 'Trained' in dtype else 1.0

                values = []
                for method in METHODS:
                    base_reduction = reduction_rates[method]
                    # Add variation
                    variation = np.random.uniform(-3, 3)
                    final_reduction = base_reduction * model_factor * dtype_factor * train_factor + variation
                    final_reduction = max(0, min(65, final_reduction))  # Clamp
                    bt_value = baseline * (1 - final_reduction / 100)
                    values.append(bt_value)

                data[model][noc][dtype] = values

    return data


def plot_overview(data, save_path):
    """
    Figure 1: Overview (精华图)
    3 rows (models) × 2 areas (NoC) × 2 subplots (data types) = 12 subplots
    """
    fig = plt.figure(figsize=(14, 10))

    noc_short = NOC_SIZES_SHORT
    dtype_short = DATA_TYPES_SHORT

    # 3 rows × 4 columns
    rows = len(MODELS)
    cols = len(noc_short) * len(dtype_short)

    current_label = ord('a')

    for row_idx, model in enumerate(MODELS):
        for area_idx, noc in enumerate(noc_short):
            for sub_idx, dtype in enumerate(dtype_short):
                # Calculate subplot position
                col_idx = area_idx * len(dtype_short) + sub_idx
                ax_idx = row_idx * cols + col_idx + 1
                ax = plt.subplot(rows, cols, ax_idx)

                # Get data
                values = data[model][noc][dtype]

                # Plot bars
                x = np.arange(len(METHODS))
                bars = ax.bar(x, values, color=METHOD_COLORS, width=0.7,
                             edgecolor='white', linewidth=0.5, alpha=0.85)

                # Title for first row
                if row_idx == 0:
                    title = f'{noc}\n{dtype}'
                    ax.set_title(title, fontsize=8, fontweight='bold', pad=3)

                # Model label on left
                if col_idx == 0:
                    ax.text(-0.35, 0.5, model, transform=ax.transAxes,
                           fontsize=10, fontweight='bold', rotation=90,
                           va='center', ha='right', color='darkblue')

                # Y-axis
                if col_idx == 0:
                    ax.set_ylabel('BT (×10⁷)', fontsize=7)

                # X-axis
                ax.set_xticks(x)
                ax.set_xticklabels(METHODS, fontsize=5.5, rotation=45, ha='right')
                ax.set_ylim(0, 120)

                # Grid
                ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
                ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Subplot label
                label = f'({chr(current_label)})'
                ax.text(0.02, 0.95, label, transform=ax.transAxes,
                       fontsize=7, fontweight='bold', va='top', ha='left')
                current_label += 1

    # Legend at top
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[i], alpha=0.85)
               for i in range(len(METHODS))]
    fig.legend(handles, METHODS, loc='upper center', ncol=9,
              frameon=True, fontsize=8, bbox_to_anchor=(0.5, 0.98))

    plt.suptitle('Overview: BT Reduction Across Models, NoC Sizes, and Data Types',
                fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_noc_comparison(data, save_path):
    """
    Figure 2: NoC Size Comparison
    Fixed model (LeNet), compare different NoC sizes
    4 rows (data types) × 4 cols (NoC sizes) = 16 subplots
    """
    model = 'LeNet'
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))

    current_label = ord('a')

    for row_idx, dtype in enumerate(DATA_TYPES):
        for col_idx, noc in enumerate(NOC_SIZES):
            ax = axes[row_idx, col_idx]

            values = data[model][noc][dtype]
            x = np.arange(len(METHODS))
            bars = ax.bar(x, values, color=METHOD_COLORS, width=0.7,
                         edgecolor='white', linewidth=0.5, alpha=0.85)

            # Add value labels on top
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.1f}', ha='center', va='bottom', fontsize=6)

            # Title
            if row_idx == 0:
                ax.set_title(noc, fontsize=10, fontweight='bold')

            # Data type label on left
            if col_idx == 0:
                ax.text(-0.3, 0.5, dtype.replace('\n', ' '),
                       transform=ax.transAxes, fontsize=9, fontweight='bold',
                       rotation=90, va='center', ha='right', color='darkblue')

            # Y-axis
            if col_idx == 0:
                ax.set_ylabel('Bit Transitions (×10⁷)', fontsize=8)

            # X-axis
            ax.set_xticks(x)
            ax.set_xticklabels(METHODS, fontsize=7, rotation=45, ha='right')
            ax.set_ylim(0, 130)

            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Subplot label
            label = f'({chr(current_label)})'
            ax.text(0.02, 0.95, label, transform=ax.transAxes,
                   fontsize=8, fontweight='bold', va='top', ha='left')
            current_label += 1

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[i], alpha=0.85)
               for i in range(len(METHODS))]
    fig.legend(handles, METHODS, loc='upper center', ncol=9,
              frameon=True, fontsize=9, bbox_to_anchor=(0.5, 0.98))

    plt.suptitle(f'NoC Size Comparison ({model} Model)',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0.05, 0, 1, 0.94])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_model_comparison(data, save_path):
    """
    Figure 3: DNN Model Comparison
    Fixed NoC (MC4_8×8), compare different models
    """
    noc = 'MC4_8×8'
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    current_label = ord('a')

    # Only use 2 data types for cleaner presentation
    dtypes_to_show = ['Trained\nFloat-32', 'Trained\nFixed-8']

    for row_idx, dtype in enumerate(dtypes_to_show):
        for col_idx, model in enumerate(MODELS):
            ax = axes[row_idx, col_idx]

            values = data[model][noc][dtype]
            x = np.arange(len(METHODS))
            bars = ax.bar(x, values, color=METHOD_COLORS, width=0.7,
                         edgecolor='white', linewidth=0.5, alpha=0.85)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.1f}', ha='center', va='bottom', fontsize=6)

            # Title
            if row_idx == 0:
                ax.set_title(model, fontsize=11, fontweight='bold')

            # Data type label
            if col_idx == 0:
                ax.text(-0.3, 0.5, dtype.replace('\n', ' '),
                       transform=ax.transAxes, fontsize=9, fontweight='bold',
                       rotation=90, va='center', ha='right', color='darkblue')

            # Y-axis
            if col_idx == 0:
                ax.set_ylabel('Bit Transitions (×10⁷)', fontsize=8)

            # X-axis
            ax.set_xticks(x)
            ax.set_xticklabels(METHODS, fontsize=7, rotation=45, ha='right')
            ax.set_ylim(0, 130)

            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Subplot label
            label = f'({chr(current_label)})'
            ax.text(0.02, 0.95, label, transform=ax.transAxes,
                   fontsize=8, fontweight='bold', va='top', ha='left')
            current_label += 1

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[i], alpha=0.85)
               for i in range(len(METHODS))]
    fig.legend(handles, METHODS, loc='upper center', ncol=9,
              frameon=True, fontsize=9, bbox_to_anchor=(0.5, 0.98))

    plt.suptitle(f'DNN Model Comparison ({noc})',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0.05, 0, 1, 0.94])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_coding_compare(data, save_path):
    """
    Figure 4: Coding Compare
    Fixed model (LeNet) and NoC (MC2_4×4), detailed comparison
    """
    model = 'LeNet'
    noc = 'MC2_4×4'

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    current_label = ord('a')

    for idx, dtype in enumerate(DATA_TYPES):
        row_idx = idx // 2
        col_idx = idx % 2
        ax = axes[row_idx, col_idx]

        values = data[model][noc][dtype]
        baseline = values[0]

        x = np.arange(len(METHODS))
        bars = ax.bar(x, values, color=METHOD_COLORS, width=0.65,
                     edgecolor='black', linewidth=0.8, alpha=0.85)

        # Add value and reduction percentage labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            # Value on top
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
            # Reduction percentage below
            if i > 0:
                reduction = (baseline - val) / baseline * 100
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                       f'-{reduction:.1f}%', ha='center', va='center',
                       fontsize=6, color='white', fontweight='bold')

        # Title
        ax.set_title(dtype.replace('\n', ' '), fontsize=11, fontweight='bold')

        # Y-axis
        ax.set_ylabel('Bit Transitions (×10⁷)', fontsize=9)

        # X-axis
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, fontsize=8, rotation=45, ha='right')
        ax.set_ylim(0, 130)

        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Subplot label
        label = f'({chr(current_label)})'
        ax.text(0.02, 0.95, label, transform=ax.transAxes,
               fontsize=9, fontweight='bold', va='top', ha='left')
        current_label += 1

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[i], alpha=0.85,
                             edgecolor='black', linewidth=0.8)
               for i in range(len(METHODS))]
    fig.legend(handles, METHODS, loc='upper center', ncol=9,
              frameon=True, fontsize=9, bbox_to_anchor=(0.5, 0.98))

    plt.suptitle(f'Coding Method Comparison ({model}, {noc})',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate mock data
    print("Generating mock data...")
    data = generate_mock_data()

    # Generate all figures
    print("\nGenerating figures...")

    # Figure 1: Overview
    plot_overview(data, os.path.join(script_dir, 'fig1_overview.pdf'))

    # Figure 2: NoC Size Comparison
    plot_noc_comparison(data, os.path.join(script_dir, 'fig2_noc_comparison.pdf'))

    # Figure 3: DNN Model Comparison
    plot_model_comparison(data, os.path.join(script_dir, 'fig3_model_comparison.pdf'))

    # Figure 4: Coding Compare
    plot_coding_compare(data, os.path.join(script_dir, 'fig4_coding_compare.pdf'))

    print("\nAll figures generated successfully!")
    print(f"Output directory: {script_dir}")


if __name__ == "__main__":
    main()
