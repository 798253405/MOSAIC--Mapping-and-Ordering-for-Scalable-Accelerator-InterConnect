#!/usr/bin/env python3
"""
TACO Paper Plots v2 - Redesigned Figure Set
12 figures with reduction-based labels (value - 100%).

Usage:
    python v.py              # Generate all figures
    python v.py --verify     # Run verification only
    python v.py --fig 1 5 12 # Generate specific figures
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import rcParams
import os
import sys
import csv

# ============== CONFIGURATION ==============
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
rcParams['font.size'] = 8

METHODS = ['O0', 'O1', 'O2', 'C1', 'C2', 'O1+C1', 'O1+C2', 'O2+C1', 'O2+C2']
METHOD_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12',
                 '#1abc9c', '#e67e22', '#34495e', '#16a085']

# CSV case order -> display order: case1=O0,case2=C1,case3=C2,case4=O1,case5=O2,...
CASE_TO_DISPLAY_ORDER = [0, 3, 4, 1, 2, 5, 6, 7, 8]

CASE_TO_METHOD = {
    'case1_default': 'O0',
    'case2_TACOall128BitInvert': 'C1',
    'case3_PartialBusInvert': 'C2',
    'case4_affiliatedordering': 'O1',
    'case5_seperratedordering': 'O2',
    'case6_affiliatedordering_TACOall128BitInvert': 'O1+C1',
    'case7_affiliatedordering_PartialBusInvert': 'O1+C2',
    'case8_seperratedordering_TACOall128BitInvert': 'O2+C1',
    'case9_seperratedordering_PartialBusInvert': 'O2+C2',
}

MODELS = ['LeNet', 'AlexNet', 'DarkNet-19']
NOC_SIZES = ['2mc_4x4', '4mc_4x4', '4mc_8x8', '8mc_8x8']
NOC_LABELS = ['MC2 4x4', 'MC4 4x4', 'MC4 8x8', 'MC8 8x8']
EVAL_MODES = ['randomeval-float', 'randomeval-fixed', 'fulleval-float', 'fulleval-fixed']

BLOCK_OFFSETS = {
    'randomeval-float': 0,
    'randomeval-fixed': 10,
    'fulleval-float': 20,
    'fulleval-fixed': 30,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILENAME = '2026TACO-FromESWEEK - 260323DIFF NN Model.csv'
CSV_PATH = os.path.join(SCRIPT_DIR, CSV_FILENAME)


# ============== DATA PARSING (same as original) ==============

def parse_csv(csv_path):
    data = {}
    for m in MODELS:
        data[m] = {}
        for n in NOC_SIZES:
            data[m][n] = {}
            for e in EVAL_MODES:
                data[m][n][e] = {'float_bt': [], 'fixed_bt': []}

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    i = 0
    while i < len(all_rows):
        row = all_rows[i]
        if len(row) < 7:
            i += 1
            continue
        cell0 = row[0].strip()
        if cell0 in MODELS:
            model_name = cell0
            block_rows = []
            j = i
            while j < len(all_rows) and len(all_rows[j]) >= 7 and all_rows[j][0].strip() == model_name:
                block_rows.append(all_rows[j])
                j += 1
            if len(block_rows) == 9:
                noc = block_rows[0][3].strip()
                _parse_block(data, model_name, noc, block_rows)
            i = j
        else:
            i += 1
    return data


def _parse_block(data, model, noc, rows):
    for eval_mode, offset in BLOCK_OFFSETS.items():
        float_bt_list = []
        fixed_bt_list = []
        for row in rows:
            if offset + 6 >= len(row):
                continue
            try:
                float_bt = int(row[offset + 5].strip())
                fixed_bt = int(row[offset + 6].strip())
            except (ValueError, IndexError):
                continue
            float_bt_list.append(float_bt)
            fixed_bt_list.append(fixed_bt)
        if len(float_bt_list) == 9:
            data[model][noc][eval_mode]['float_bt'] = float_bt_list
            data[model][noc][eval_mode]['fixed_bt'] = fixed_bt_list


def reorder_to_display(values):
    return [values[i] for i in CASE_TO_DISPLAY_ORDER]


def normalize(values):
    baseline = values[0]
    if baseline == 0:
        return [0] * len(values)
    return [v / baseline * 100 for v in values]


def get_normalized_display(data, model, noc, eval_mode, metric='float_bt'):
    raw = data[model][noc][eval_mode][metric]
    reordered = reorder_to_display(raw)
    return normalize(reordered)


def reduction_label(val):
    """Format value as reduction from baseline: 85.3 -> '-14.7%', 100 -> '0.0%'"""
    r = val - 100
    if abs(r) < 0.05:
        return '0.0%'
    return f'{r:.1f}%'


# ============== PLOTTING HELPERS ==============

def setup_subplot_v2(ax, values, row_idx, col_idx,
                     row_label=None, col_label=None,
                     show_values=False, show_reduction_inside=False,
                     ylim_max=110, subplot_label=None):
    """Common subplot setup with reduction-based labels."""
    x = np.arange(len(METHODS))
    bars = ax.bar(x, values, color=METHOD_COLORS, width=0.7,
                  edgecolor='white', linewidth=0.5, alpha=0.85)

    if show_values:
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=5, fontweight='bold')

    if show_reduction_inside:
        for i, (bar, val) in enumerate(zip(bars, values)):
            if i > 0:
                r = 100 - val
                if r > 3:  # only show if enough space
                    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() / 2,
                            f'-{r:.1f}%', ha='center', va='center',
                            fontsize=5, color='white', fontweight='bold')

    if row_idx == 0 and col_label:
        ax.set_title(col_label, fontsize=8, fontweight='bold', pad=3)

    if col_idx == 0 and row_label:
        ax.text(-0.35, 0.5, row_label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', rotation=90,
                va='center', ha='right', color='darkblue')

    if col_idx == 0:
        ax.set_ylabel('Normalized BT (%)', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=5.5, rotation=45, ha='right')
    ax.set_ylim(0, ylim_max)
    ax.axhline(y=100, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if subplot_label:
        ax.text(0.02, 0.95, subplot_label, transform=ax.transAxes,
                fontsize=7, fontweight='bold', va='top', ha='left')
    return bars


def add_legend(fig, bbox_y=0.98):
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[i], alpha=0.85)
               for i in range(len(METHODS))]
    fig.legend(handles, METHODS, loc='upper center', ncol=9,
              frameon=True, fontsize=8, bbox_to_anchor=(0.5, bbox_y))


def save_fig(fig, filename, save_dir=None):
    if save_dir is None:
        save_dir = SCRIPT_DIR
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


# ============== OVERVIEW FIGURES (Fig 1-4) ==============

def _plot_overview(data, save_dir, eval_mode, metric, title_suffix, filename):
    """Generic overview: 3 models x 4 NoC."""
    fig = plt.figure(figsize=(16, 10))
    label_idx = ord('a')

    for row_idx, model in enumerate(MODELS):
        for col_idx, (noc, noc_label) in enumerate(zip(NOC_SIZES, NOC_LABELS)):
            ax = plt.subplot(3, 4, row_idx * 4 + col_idx + 1)
            values = get_normalized_display(data, model, noc, eval_mode, metric)
            setup_subplot_v2(ax, values, row_idx, col_idx,
                            row_label=model, col_label=noc_label,
                            show_values=True, subplot_label=f'({chr(label_idx)})')
            label_idx += 1

    add_legend(fig)
    plt.suptitle(f'BT Reduction: {title_suffix}',
                fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    save_fig(fig, filename, save_dir)


def plot_fig01(data, save_dir):
    """Fig 1: Overview Trained Float"""
    _plot_overview(data, save_dir, 'fulleval-float', 'float_bt',
                   'Trained Weights, Float-32 BT', 'v2_fig01_overview_trained_float.pdf')


def plot_fig02(data, save_dir):
    """Fig 2: Overview Trained Fixed"""
    _plot_overview(data, save_dir, 'fulleval-fixed', 'fixed_bt',
                   'Trained Weights, Fixed-8 BT', 'v2_fig02_overview_trained_fixed.pdf')


def plot_fig03(data, save_dir):
    """Fig 3: Overview Random Float"""
    _plot_overview(data, save_dir, 'randomeval-float', 'float_bt',
                   'Random Weights, Float-32 BT', 'v2_fig03_overview_random_float.pdf')


def plot_fig04(data, save_dir):
    """Fig 4: Overview Random Fixed"""
    _plot_overview(data, save_dir, 'randomeval-fixed', 'fixed_bt',
                   'Random Weights, Fixed-8 BT', 'v2_fig04_overview_random_fixed.pdf')


# ============== FIG 5: Combined NoC-avg Reduction ==============

def plot_fig05(data, save_dir):
    """Fig 5: BT Reduction % - 3 models x 4 cols, NoC-averaged, 8 methods (skip O0), error bars"""
    configs = [
        ('fulleval-float', 'float_bt', 'Trained\nFloat-32'),
        ('fulleval-fixed', 'fixed_bt', 'Trained\nFixed-8'),
        ('randomeval-float', 'float_bt', 'Random\nFloat-32'),
        ('randomeval-fixed', 'fixed_bt', 'Random\nFixed-8'),
    ]

    methods_no_baseline = METHODS[1:]
    colors_no_baseline = METHOD_COLORS[1:]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    label_idx = ord('a')

    for row_idx, model in enumerate(MODELS):
        for col_idx, (eval_mode, metric, col_title) in enumerate(configs):
            ax = axes[row_idx, col_idx]

            all_reductions = []
            for noc in NOC_SIZES:
                normed = get_normalized_display(data, model, noc, eval_mode, metric)
                reductions = [100 - v for v in normed[1:]]  # skip O0
                all_reductions.append(reductions)

            all_reductions = np.array(all_reductions)
            means = np.mean(all_reductions, axis=0)
            yerr_low = means - np.min(all_reductions, axis=0)
            yerr_high = np.max(all_reductions, axis=0) - means

            x = np.arange(len(methods_no_baseline))
            bars = ax.bar(x, means, color=colors_no_baseline, width=0.7,
                         edgecolor='white', linewidth=0.5, alpha=0.85,
                         yerr=[yerr_low, yerr_high], capsize=2, error_kw={'linewidth': 0.8})

            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=5, fontweight='bold')

            if row_idx == 0:
                ax.set_title(col_title, fontsize=9, fontweight='bold')
            if col_idx == 0:
                ax.text(-0.35, 0.5, model, transform=ax.transAxes,
                       fontsize=10, fontweight='bold', rotation=90,
                       va='center', ha='right', color='darkblue')
                ax.set_ylabel('BT Reduction (%)', fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(methods_no_baseline, fontsize=5.5, rotation=45, ha='right')
            ax.set_ylim(0, 80)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.text(0.02, 0.95, f'({chr(label_idx)})', transform=ax.transAxes,
                   fontsize=7, fontweight='bold', va='top')
            label_idx += 1

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors_no_baseline[i], alpha=0.85)
               for i in range(len(methods_no_baseline))]
    fig.legend(handles, methods_no_baseline, loc='upper center', ncol=8,
              frameon=True, fontsize=8, bbox_to_anchor=(0.5, 0.98))
    plt.suptitle('BT Reduction (%, avg across NoC, error bars = min/max, higher = better)',
                fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    save_fig(fig, 'v2_fig05_combined_noc_avg.pdf', save_dir)


# ============== FIG 6: Heatmap ==============

def plot_fig06(data, save_dir):
    """Fig 6: Heatmap - reduction values"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    for ax, eval_mode, metric, title in [
        (ax1, 'fulleval-float', 'float_bt', 'Float-32 BT Reduction (%)'),
        (ax2, 'fulleval-fixed', 'fixed_bt', 'Fixed-8 BT Reduction (%)')
    ]:
        matrix = []
        row_labels = []
        for model in MODELS:
            for noc, noc_label in zip(NOC_SIZES, NOC_LABELS):
                values = get_normalized_display(data, model, noc, eval_mode, metric)
                # Convert to reduction: 100 - normalized
                reductions = [v - 100 for v in values]
                matrix.append(reductions)
                row_labels.append(f'{model}\n{noc_label}')

        matrix = np.array(matrix)

        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-70, vmax=5)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, fontsize=8, rotation=45, ha='right')
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                color = 'white' if val < -40 else 'black'
                text = f'{val:.1f}%' if abs(val) > 0.05 else '0.0%'
                ax.text(j, i, text, ha='center', va='center',
                       fontsize=6, color=color, fontweight='bold')

        # Highlight best (most negative) per row
        for i in range(matrix.shape[0]):
            best_j = np.argmin(matrix[i, 1:]) + 1
            ax.add_patch(plt.Rectangle((best_j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor='blue', linewidth=2))

    fig.colorbar(im, ax=ax2, location='right', label='BT Reduction (%)',
                shrink=0.8, pad=0.02)
    plt.suptitle('Heatmap: BT Reduction Across All Configurations (Trained Weights)',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'v2_fig06_heatmap.pdf', save_dir)


# ============== FIG 7: Grouped by Model (2x2) ==============

def plot_fig07(data, save_dir):
    """Fig 7: Methods grouped, models as colored bars, 2x2 (random/trained x float/fixed)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    model_colors = ['#3498db', '#e74c3c', '#2ecc71']
    bar_width = 0.25
    label_idx = ord('a')

    configs = [
        (0, 0, 'randomeval-float', 'float_bt', 'Float-32 Random'),
        (0, 1, 'randomeval-fixed', 'fixed_bt', 'Fixed-8 Random'),
        (1, 0, 'fulleval-float', 'float_bt', 'Float-32 Trained'),
        (1, 1, 'fulleval-fixed', 'fixed_bt', 'Fixed-8 Trained'),
    ]

    for row_idx, col_idx, eval_mode, metric, title in configs:
        ax = axes[row_idx, col_idx]
        x = np.arange(len(METHODS))

        for m_idx, model in enumerate(MODELS):
            all_vals = []
            for noc in NOC_SIZES:
                vals = get_normalized_display(data, model, noc, eval_mode, metric)
                all_vals.append(vals)
            means = np.mean(all_vals, axis=0)
            offset = (m_idx - 1) * bar_width
            ax.bar(x + offset, means, bar_width, label=model,
                  color=model_colors[m_idx], alpha=0.85,
                  edgecolor='white', linewidth=0.5)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Method', fontsize=9)
        ax.set_ylabel('Normalized BT (%)', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, fontsize=8, rotation=45, ha='right')
        ax.set_ylim(0, 115)
        ax.axhline(y=100, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=8)
        ax.text(0.02, 0.95, f'({chr(label_idx)})', transform=ax.transAxes,
               fontsize=9, fontweight='bold', va='top')
        label_idx += 1

    plt.suptitle('Method Comparison Grouped by Model (avg across NoC)',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, 'v2_fig07_grouped_by_model.pdf', save_dir)


# ============== FIG 8: Radar/Spider ==============

def plot_fig08(data, save_dir):
    """Fig 8: Radar chart - 1x2 (Float / Fixed).
    Each panel: 3 models x 2 eval modes (trained=solid, random=dashed) = 6 polygons.
    Same color per model, solid=trained, dashed=random.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(polar=True))

    methods_no_baseline = METHODS[1:]
    N = len(methods_no_baseline)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Colors per model: deep=trained, light=random (same hue)
    model_styles = [
        # (trained_color, random_color, trained_lw, random_lw)
        ('#1a5276', '#85c1e9', 2.0, 1.3),  # blue deep / light
        ('#922b21', '#f1948a', 2.0, 1.3),  # red deep / light
        ('#1e8449', '#82e0aa', 2.0, 1.3),  # green deep / light
    ]

    panels = [
        (ax1, 'float_bt', 'fulleval-float', 'randomeval-float', 'Float-32 BT'),
        (ax2, 'fixed_bt', 'fulleval-fixed', 'randomeval-fixed', 'Fixed-8 BT'),
    ]

    for ax, metric, trained_eval, random_eval, title in panels:
        for m_idx, model in enumerate(MODELS):
            t_color, r_color, t_lw, r_lw = model_styles[m_idx]
            for eval_mode, color, lw, alpha_fill, suffix in [
                (trained_eval, t_color, t_lw, 0.10, 'Trained'),
                (random_eval, r_color, r_lw, 0.08, 'Random'),
            ]:
                all_vals = []
                for noc in NOC_SIZES:
                    normed = get_normalized_display(data, model, noc, eval_mode, metric)
                    reductions = [100 - v for v in normed[1:]]
                    all_vals.append(reductions)
                means = np.mean(all_vals, axis=0).tolist()
                means += means[:1]

                ax.plot(angles, means, '-', linewidth=lw,
                       label=f'{model} {suffix}', color=color, markersize=0)
                ax.fill(angles, means, alpha=alpha_fill, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(methods_no_baseline, fontsize=7)
        ax.set_ylim(0, 90)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=6.5,
                 ncol=1, framealpha=0.9)

    plt.suptitle('Method Reduction Profiles: Trained (solid) vs Random (dashed), avg across NoC',
                fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, 'v2_fig08_radar.pdf', save_dir)


# ============== FIG 9: Stacked - Coding as baseline ==============

def plot_fig09(data, save_dir):
    """Fig 9: Stacked - C as baseline (blue), +O1 (orange), +O2 (red).
    6 bars: C1, C1+O1, C1+O2, C2, C2+O1, C2+O2
    Dashed reference lines connect related bars.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    label_idx = ord('a')

    bar_labels = ['C1', 'C1+O1', 'C1+O2', 'C2', 'C2+O1', 'C2+O2']

    COLOR_CODING = '#3498db'   # blue - coding only
    COLOR_O1 = '#e67e22'       # orange - +O1
    COLOR_O2 = '#e74c3c'       # red - +O2

    # (coding_display_idx, combined_display_idx, ordering_type)
    # ordering_type: None=coding only, 'O1', 'O2'
    stacked_config = [
        (3, None, None),   # C1 alone
        (3, 5, 'O1'),      # C1 + O1 -> display idx 5 is O1+C1
        (3, 7, 'O2'),      # C1 + O2 -> display idx 7 is O2+C1
        (4, None, None),   # C2 alone
        (4, 6, 'O1'),      # C2 + O1 -> display idx 6 is O1+C2
        (4, 8, 'O2'),      # C2 + O2 -> display idx 8 is O2+C2
    ]

    for row_idx, (eval_mode, metric, row_title) in enumerate([
        ('fulleval-float', 'float_bt', 'Float-32'),
        ('fulleval-fixed', 'fixed_bt', 'Fixed-8')
    ]):
        for col_idx, model in enumerate(MODELS):
            ax = axes[row_idx, col_idx]

            coding_parts = []
            ordering_adds = []

            for noc in NOC_SIZES:
                normed = get_normalized_display(data, model, noc, eval_mode, metric)
                c_parts = []
                o_adds = []
                for coding_idx, combined_idx, _ in stacked_config:
                    coding_reduction = 100 - normed[coding_idx]
                    if combined_idx is not None:
                        combined_reduction = 100 - normed[combined_idx]
                        ordering_additional = max(0, combined_reduction - coding_reduction)
                    else:
                        ordering_additional = 0
                    c_parts.append(coding_reduction)
                    o_adds.append(ordering_additional)
                coding_parts.append(c_parts)
                ordering_adds.append(o_adds)

            coding_means = np.mean(coding_parts, axis=0)
            ordering_means = np.mean(ordering_adds, axis=0)
            totals = coding_means + ordering_means

            x = np.arange(len(bar_labels))
            bar_width = 0.6

            # Draw bars with per-bar ordering color
            for i in range(len(bar_labels)):
                _, _, otype = stacked_config[i]
                # Blue coding base
                ax.bar(x[i], coding_means[i], bar_width, color=COLOR_CODING, alpha=0.85)
                # Ordering addition with type-specific color
                if ordering_means[i] > 0:
                    ocolor = COLOR_O1 if otype == 'O1' else COLOR_O2
                    ax.bar(x[i], ordering_means[i], bar_width, bottom=coding_means[i],
                          color=ocolor, alpha=0.85)

            # Dashed reference lines
            hw = bar_width / 2 * 1.1  # slightly wider than bar
            # C1 group: bars 0(C1), 1(C1+O1), 2(C1+O2)
            c1_h = totals[0]        # C1 height
            c1o1_h = totals[1]      # C1+O1 height
            # C1+O1: dashed at C1 height
            ax.plot([x[1] - hw, x[1] + hw], [c1_h, c1_h],
                   color='black', linestyle='--', linewidth=1, alpha=0.7)
            # C1+O2: dashed at C1 height and C1+O1 height
            ax.plot([x[2] - hw, x[2] + hw], [c1_h, c1_h],
                   color='black', linestyle='--', linewidth=1, alpha=0.7)
            ax.plot([x[2] - hw, x[2] + hw], [c1o1_h, c1o1_h],
                   color='gray', linestyle=':', linewidth=1, alpha=0.6)

            # C2 group: bars 3(C2), 4(C2+O1), 5(C2+O2)
            c2_h = totals[3]        # C2 height
            c2o1_h = totals[4]      # C2+O1 height
            # C2+O1: dashed at C2 height
            ax.plot([x[4] - hw, x[4] + hw], [c2_h, c2_h],
                   color='black', linestyle='--', linewidth=1, alpha=0.7)
            # C2+O2: dashed at C2 height and C2+O1 height
            ax.plot([x[5] - hw, x[5] + hw], [c2_h, c2_h],
                   color='black', linestyle='--', linewidth=1, alpha=0.7)
            ax.plot([x[5] - hw, x[5] + hw], [c2o1_h, c2o1_h],
                   color='gray', linestyle=':', linewidth=1, alpha=0.6)

            # Value labels on top
            for i in range(len(bar_labels)):
                ax.text(x[i], totals[i] + 1, f'{totals[i]:.1f}%',
                       ha='center', va='bottom', fontsize=6.5, fontweight='bold')

            if row_idx == 0:
                ax.set_title(model, fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.text(-0.3, 0.5, row_title, transform=ax.transAxes,
                       fontsize=10, fontweight='bold', rotation=90,
                       va='center', ha='right', color='darkblue')
                ax.set_ylabel('BT Reduction (%)', fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels, fontsize=7, rotation=45, ha='right')
            ax.set_ylim(0, 80)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if row_idx == 0 and col_idx == 2:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor=COLOR_CODING, alpha=0.85, label='Coding only'),
                    Patch(facecolor=COLOR_O1, alpha=0.85, label='+ O1 (Affiliated)'),
                    Patch(facecolor=COLOR_O2, alpha=0.85, label='+ O2 (Separated)'),
                ]
                ax.legend(handles=legend_elements, fontsize=7, loc='upper right')
            ax.text(0.02, 0.95, f'({chr(label_idx)})', transform=ax.transAxes,
                   fontsize=8, fontweight='bold', va='top')
            label_idx += 1

    plt.suptitle('Coding as Baseline + Ordering Contribution (avg across NoC)',
                fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    save_fig(fig, 'v2_fig09_stacked_coding_baseline.pdf', save_dir)


# ============== FIG 10: Overall of Overall ==============

def plot_fig10(data, save_dir):
    """Fig 10: Overall of Overall - avg across ALL models x ALL NoC x trained+random.
    1x2 layout: Float panel / Fixed panel, 9 bars each with error bars.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    label_idx = ord('a')

    for ax, metric, eval_modes, title in [
        (ax1, 'float_bt', ['fulleval-float', 'randomeval-float'],
         'Float-32 BT'),
        (ax2, 'fixed_bt', ['fulleval-fixed', 'randomeval-fixed'],
         'Fixed-8 BT'),
    ]:
        # Collect across ALL models x ALL NoC x both eval modes
        all_values = []
        for model in MODELS:
            for noc in NOC_SIZES:
                for eval_mode in eval_modes:
                    vals = get_normalized_display(data, model, noc, eval_mode, metric)
                    all_values.append(vals)

        all_values = np.array(all_values)  # shape: (3*4*2, 9) = (24, 9)
        means = np.mean(all_values, axis=0)
        stds = np.std(all_values, axis=0)
        mins = np.min(all_values, axis=0)
        maxs = np.max(all_values, axis=0)
        yerr = [means - mins, maxs - means]

        x = np.arange(len(METHODS))
        bars = ax.bar(x, means, color=METHOD_COLORS, width=0.65,
                     edgecolor='black', linewidth=0.8, alpha=0.85,
                     yerr=yerr, capsize=3, error_kw={'linewidth': 1})

        for i, (bar, val) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 3,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        n_configs = all_values.shape[0]
        ax.set_title(f'{title}  (n = {n_configs} configs)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Normalized BT (%)', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, fontsize=9, rotation=45, ha='right')
        ax.set_ylim(0, 120)
        ax.axhline(y=100, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.95, f'({chr(label_idx)})', transform=ax.transAxes,
               fontsize=10, fontweight='bold', va='top')
        label_idx += 1

    plt.suptitle('OVERALL AVERAGE: BT Reduction Across All Models, NoC Sizes & Weight Types',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, 'v2_fig10_overall_of_overall.pdf', save_dir)


# ============== VERIFICATION (same as original) ==============

def verify_round1(data):
    print("\n" + "=" * 60)
    print("VERIFICATION ROUND 1: Data Parsing & Normalization")
    print("=" * 60)
    results = []

    print("\n--- Step 1.1: Raw Value Spot Checks ---")
    spot_checks = [
        ('LeNet', '2mc_4x4', 'randomeval-float', 0, 'float_bt', 21178213),
        ('LeNet', '2mc_4x4', 'randomeval-float', 0, 'fixed_bt', 5849232),
        ('LeNet', '2mc_4x4', 'fulleval-fixed', 8, 'float_bt', 14312833),
        ('LeNet', '2mc_4x4', 'fulleval-fixed', 8, 'fixed_bt', 1797865),
        ('AlexNet', '2mc_4x4', 'randomeval-float', 0, 'float_bt', 611628356),
        ('AlexNet', '4mc_8x8', 'fulleval-float', 2, 'float_bt', 673129521),
        ('DarkNet-19', '2mc_4x4', 'randomeval-float', 0, 'float_bt', 9675028725),
        ('DarkNet-19', '4mc_4x4', 'fulleval-fixed', 6, 'fixed_bt', 285746310),
        ('DarkNet-19', '4mc_8x8', 'fulleval-float', 0, 'float_bt', 10237906514),
    ]
    for model, noc, eval_mode, case_idx, metric, expected in spot_checks:
        actual = data[model][noc][eval_mode][metric][case_idx]
        status = "PASS" if actual == expected else "FAIL"
        results.append((f"Raw {model}/{noc}/{eval_mode}/case{case_idx+1}/{metric}", status, f"expected={expected}, actual={actual}"))
        symbol = "OK" if status == "PASS" else "XX"
        print(f"  [{symbol}] {model}/{noc}/{eval_mode} case{case_idx+1} {metric}: expected={expected}, got={actual}")

    print("\n--- Step 1.2: Normalization Checks ---")
    norm_checks = [
        ('LeNet', '2mc_4x4', 'randomeval-float', 8, 'float_bt', 16210178 / 21178213 * 100),
        ('DarkNet-19', '4mc_8x8', 'fulleval-float', 4, 'float_bt', 7671770503 / 10237906514 * 100),
        ('AlexNet', '8mc_8x8', 'randomeval-float', 3, 'float_bt', 528950598 / 606144593 * 100),
    ]
    for model, noc, eval_mode, case_idx, metric, expected_pct in norm_checks:
        raw = data[model][noc][eval_mode][metric]
        actual_pct = raw[case_idx] / raw[0] * 100
        diff = abs(actual_pct - expected_pct)
        status = "PASS" if diff < 0.01 else "FAIL"
        results.append((f"Norm {model}/{noc}/case{case_idx+1}/{metric}", status, f"diff={diff:.6f}"))
        symbol = "OK" if status == "PASS" else "XX"
        print(f"  [{symbol}] Norm {model}/{noc} case{case_idx+1}: expected={expected_pct:.4f}%, got={actual_pct:.4f}%")

    print("\n--- Step 1.3: Dimension Counts ---")
    total_points = 0
    zero_baselines = []
    for model in MODELS:
        for noc in NOC_SIZES:
            for eval_mode in EVAL_MODES:
                for metric in ['float_bt', 'fixed_bt']:
                    vals = data[model][noc][eval_mode][metric]
                    total_points += len(vals)
                    if vals[0] == 0:
                        zero_baselines.append(f"{model}/{noc}/{eval_mode}/{metric}")

    expected_total = 3 * 4 * 4 * 2 * 9
    ok = total_points == expected_total
    results.append(("Dimension count", "PASS" if ok else "FAIL", f"{total_points}/{expected_total}"))
    print(f"  [{'OK' if ok else 'XX'}] Total: {total_points} (expected {expected_total})")
    ok2 = len(zero_baselines) == 0
    results.append(("Non-zero baselines", "PASS" if ok2 else "FAIL", ""))
    print(f"  [{'OK' if ok2 else 'XX'}] All baselines non-zero: {ok2}")

    passed = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n--- Round 1: {passed}/{len(results)} PASSED ---")
    return results


def verify_round2(data, r1_results):
    print("\n" + "=" * 60)
    print("VERIFICATION ROUND 2: Meta-Verification")
    print("=" * 60)
    results = []

    print("\n--- Step 2.1: Independent Re-parse ---")
    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
        raw_lines = f.readlines()

    for line_idx, model, expected_f, expected_x in [
        (14, 'LeNet', 21178213, 5849232),
        (59, 'AlexNet', 611628356, 190688720),
        (104, 'DarkNet-19', 9675028725, 2941466924),
    ]:
        parts = raw_lines[line_idx].strip().split(',')
        pf, px = int(parts[5]), int(parts[6])
        af = data[model]['2mc_4x4']['randomeval-float']['float_bt'][0]
        ax = data[model]['2mc_4x4']['randomeval-float']['fixed_bt'][0]
        ok = (pf == af == expected_f and px == ax == expected_x)
        results.append((f"Re-parse {model}", "PASS" if ok else "FAIL", ""))
        print(f"  [{'OK' if ok else 'XX'}] {model}: float={pf}=={af}, fixed={px}=={ax}")

    print("\n--- Step 2.2: Monotonicity Sanity ---")
    o2c2_best = 0
    fixed_better = 0
    c2_better = 0
    total_c = 0
    total_p = 0

    for model in MODELS:
        for noc in NOC_SIZES:
            for eval_mode in EVAL_MODES:
                for metric in ['float_bt', 'fixed_bt']:
                    vals = reorder_to_display(data[model][noc][eval_mode][metric])
                    normed = normalize(vals)
                    if np.argmin(normed[1:]) + 1 == 8:
                        o2c2_best += 1
                    c1_r = 100 - normed[3]
                    c2_r = 100 - normed[4]
                    if c2_r > c1_r:
                        c2_better += 1
                    total_c += 1

            float_n = get_normalized_display(data, model, noc, 'fulleval-float', 'float_bt')
            fixed_n = get_normalized_display(data, model, noc, 'fulleval-fixed', 'fixed_bt')
            for i in range(1, 9):
                if (100 - fixed_n[i]) > (100 - float_n[i]):
                    fixed_better += 1
                total_p += 1

    for label, count, total, threshold in [
        ("O2+C2 best >80%", o2c2_best, total_c, 80),
        ("C2 > C1 >90%", c2_better, total_c, 90),
        ("Fixed > Float >70%", fixed_better, total_p, 70),
    ]:
        pct = count / total * 100
        ok = pct >= threshold
        results.append((label, "PASS" if ok else "WARN", f"{pct:.1f}%"))
        print(f"  [{'OK' if ok else '!!'}] {label}: {pct:.1f}% ({count}/{total})")

    print("\n--- Step 2.3: Reduction Label Check ---")
    # Verify reduction_label function
    test_cases = [(100, '0.0%'), (85.3, '-14.7%'), (65.0, '-35.0%')]
    for val, expected in test_cases:
        actual = reduction_label(val)
        ok = actual == expected
        results.append((f"reduction_label({val})", "PASS" if ok else "FAIL", f"{actual}"))
        print(f"  [{'OK' if ok else 'XX'}] reduction_label({val}): expected={expected}, got={actual}")

    passed = sum(1 for _, s, _ in results if s == "PASS")
    warned = sum(1 for _, s, _ in results if s == "WARN")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"\n--- Round 2: {passed} PASS, {warned} WARN, {failed} FAIL ---")
    return results


# ============== MAIN ==============

def main():
    args = sys.argv[1:]

    print("=" * 60)
    print("TACO Paper Plots v2 - Redesigned Figure Set")
    print("=" * 60)

    print(f"\nParsing CSV: {CSV_PATH}")
    data = parse_csv(CSV_PATH)
    print("CSV parsed successfully.")

    verify_only = '--verify' in args
    fig_filter = None
    if '--fig' in args:
        fig_idx = args.index('--fig')
        fig_filter = [int(x) for x in args[fig_idx + 1:] if x.isdigit()]

    print("\nRunning verification...")
    r1 = verify_round1(data)
    r2 = verify_round2(data, r1)

    all_results = r1 + r2
    failures = [(n, d) for n, s, d in all_results if s == "FAIL"]
    if failures:
        print(f"\n!! {len(failures)} FAILED !!")
        for n, d in failures:
            print(f"  FAILED: {n} - {d}")

    if verify_only:
        print("\nVerification complete (--verify mode).")
        return

    save_dir = SCRIPT_DIR
    print(f"\nGenerating figures to: {save_dir}")

    plot_functions = {
        1: ('Overview: Trained Float', plot_fig01),
        2: ('Overview: Trained Fixed', plot_fig02),
        3: ('Overview: Random Float', plot_fig03),
        4: ('Overview: Random Fixed', plot_fig04),
        5: ('NoC-avg Reduction', plot_fig05),
        6: ('Heatmap', plot_fig06),
        7: ('Grouped by Model', plot_fig07),
        8: ('Radar/Spider', plot_fig08),
        9: ('Stacked: Coding + Ordering', plot_fig09),
        10: ('Overall of Overall', plot_fig10),
    }

    for num, (name, func) in plot_functions.items():
        if fig_filter and num not in fig_filter:
            continue
        print(f"\n  Fig {num}: {name}...")
        try:
            func(data, save_dir)
        except Exception as e:
            print(f"  ERROR Fig {num}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll done!")


if __name__ == "__main__":
    main()
