#!/usr/bin/env python3
"""
128-Token Small LLM Performance Analysis on NoC - Comprehensive Visualization
Updated with complete legend in subplot (e)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def create_dataframe():
    """Create DataFrame from provided 128-token configuration data"""
    data = {
        'NoC_Size': ['MC2_4x4'] * 6 + ['MC8_8x8'] * 6 + ['MC32_16x16'] * 6 + ['MC128_32x32'] * 6,
        'Strategy': ['baseline', 'samos', 'affiliated', 'separated', 'mosaic-1', 'mosaic-2'] * 4,
        'Total_Cycles': [
            # MC2_4x4
            6321329, 6670838, 6321329, 6321329, 6670838, 6670838,
            # MC8_8x8
            1601064, 1601679, 1601064, 1601064, 1601679, 1601679,
            # MC32_16x16
            420393, 397033, 420393, 420393, 397033, 397033,
            # MC128_32x32
            108320, 103120, 108320, 108320, 103120, 103120
        ],
        'Avg_Hops': [
            # MC2_4x4
            1.71, 1.68, 1.71, 1.71, 1.68, 1.68,
            # MC8_8x8
            1.71, 1.68, 1.71, 1.71, 1.68, 1.68,
            # MC32_16x16
            1.71, 1.69, 1.71, 1.71, 1.69, 1.69,
            # MC128_32x32
            1.72, 1.70, 1.72, 1.72, 1.70, 1.70
        ],
        'BitTransitions': [
            # MC2_4x4
            2996483603, 2937201979, 2104938412, 2048925106, 2063408240, 2008985952,
            # MC8_8x8
            2996456395, 2939766001, 2105881456, 2049770511, 2068054420, 2013128138,
            # MC32_16x16
            2997578083, 2949234598, 2110651477, 2054537810, 2076402021, 2021025947,
            # MC128_32x32
            2998282220, 2976012302, 2110862399, 2054636629, 2095626629, 2039702294
        ]
    }

    df = pd.DataFrame(data)

    # Map strategies to display names
    strategy_map = {
        'baseline': 'Baseline',
        'samos': 'SAMOS',
        'affiliated': 'Affiliated',
        'separated': 'Separated',
        'mosaic-1': 'MOSAIC-1',
        'mosaic-2': 'MOSAIC-2'
    }
    df['Strategy'] = df['Strategy'].map(strategy_map)

    # Map NoC sizes for display
    noc_map = {
        'MC2_4x4': '4×4',
        'MC8_8x8': '8×8',
        'MC32_16x16': '16×16',
        'MC128_32x32': '32×32'
    }
    df['NoC_Display'] = df['NoC_Size'].map(noc_map)

    return df


def create_comprehensive_analysis():
    """Create comprehensive analysis figure with 5 subplots"""

    df = create_dataframe()

    # Set up the figure
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')

    fig = plt.figure(figsize=(18, 8))

    # Define color schemes
    colors = {
        'Baseline': '#003f5c',  # 深海蓝
        'SAMOS': '#edc948',  # 金黄
        'Affiliated': '#665191',  # 紫罗兰
        'Separated': '#a05195',  # 洋红
        'MOSAIC-1': '#d45087',  # 玫瑰红
        'MOSAIC-2': '#ff7c43'  # 珊瑚橙
    }

    # Create gridspec
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    strategies = ['Baseline', 'SAMOS', 'Affiliated', 'Separated', 'MOSAIC-1', 'MOSAIC-2']
    noc_order = ['4×4', '8×8', '16×16', '32×32']

    # (a) Execution Cycles - WITH SAMOS LABELS ONLY
    ax1 = fig.add_subplot(gs[0, 0])
    bar_width = 0.13
    x_base = np.arange(len(noc_order))

    baseline_values = []
    for noc in noc_order:
        noc_data = df[(df['NoC_Display'] == noc) & (df['Strategy'] == 'Baseline')]
        baseline_values.append(noc_data['Total_Cycles'].values[0] if len(noc_data) > 0 else 0)

    for i, strategy in enumerate(strategies):
        values = []
        for noc in noc_order:
            noc_data = df[(df['NoC_Display'] == noc) & (df['Strategy'] == strategy)]
            values.append(noc_data['Total_Cycles'].values[0] if len(noc_data) > 0 else 0)

        bars = ax1.bar(x_base + i * bar_width, values, bar_width,
                       label=strategy, color=colors[strategy], alpha=0.85)

        # Add labels ONLY for SAMOS
        if strategy == 'SAMOS':
            for j, (val, baseline_val) in enumerate(zip(values, baseline_values)):
                if baseline_val > 0 and val != baseline_val:
                    reduction = ((baseline_val - val) / baseline_val) * 100
                    y_pos = val + max(baseline_values) * 0.01

                    if abs(reduction) > 0.01:
                        if j < 2:  # First two bars - normal black
                            text = f'+{abs(reduction):.1f}%' if reduction < 0 else f'-{reduction:.1f}%'
                            ax1.text(x_base[j] + i * bar_width, y_pos, text,
                                     ha='center', va='bottom', fontsize=7,
                                     color='black', rotation=90)
                        else:  # Last two bars - prominent red
                            text = f'+{abs(reduction):.1f}%' if reduction < 0 else f'-{reduction:.1f}%'
                            ax1.text(x_base[j] + i * bar_width, y_pos, text,
                                     ha='center', va='bottom', fontsize=12,
                                     color='red', rotation=90, fontweight='bold')

    ax1.set_xlabel('NoC Configuration', fontweight='bold')
    ax1.set_ylabel('Execution Cycles', fontweight='bold')
    ax1.set_title('(a) Execution Cycles (128 Tokens)', fontweight='bold', fontsize=11)
    ax1.set_xticks(x_base + bar_width * 2.5)
    ax1.set_xticklabels(noc_order, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # (b) Energy Consumption - NO LEGEND
    ax2 = fig.add_subplot(gs[0, 1])

    for i, strategy in enumerate(strategies):
        values = []
        for noc in noc_order:
            noc_data = df[(df['NoC_Display'] == noc) & (df['Strategy'] == strategy)]
            values.append(noc_data['BitTransitions'].values[0] / 1e9 if len(noc_data) > 0 else 0)

        bars = ax2.bar(x_base + i * bar_width, values, bar_width,
                       label=strategy, color=colors[strategy], alpha=0.85)

    ax2.set_xlabel('NoC Configuration', fontweight='bold')
    ax2.set_ylabel('Total Bit Transitions (G)', fontweight='bold')
    ax2.set_title('(b) Link Energy Consumption (128 Tokens)', fontweight='bold', fontsize=11)
    ax2.set_xticks(x_base + bar_width * 2.5)
    ax2.set_xticklabels(noc_order, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # (c) Average Hops - NO LEGEND
    ax3 = fig.add_subplot(gs[1, 0])

    for i, strategy in enumerate(strategies):
        values = []
        for noc in noc_order:
            noc_data = df[(df['NoC_Display'] == noc) & (df['Strategy'] == strategy)]
            values.append(noc_data['Avg_Hops'].values[0] if len(noc_data) > 0 else 0)

        bars = ax3.bar(x_base + i * bar_width, values, bar_width,
                       label=strategy, color=colors[strategy], alpha=0.85)

        if strategy == 'SAMOS':
            baseline_values_hops = []
            for noc in noc_order:
                baseline_data = df[(df['NoC_Display'] == noc) & (df['Strategy'] == 'Baseline')]
                baseline_values_hops.append(baseline_data['Avg_Hops'].values[0] if len(baseline_data) > 0 else 0)

            for j, (samos_val, baseline_val) in enumerate(zip(values, baseline_values_hops)):
                if baseline_val > 0:
                    reduction = ((baseline_val - samos_val) / baseline_val) * 100
                    ax3.text(x_base[j] + i * bar_width, samos_val + 0.02,
                             f'-{reduction:.1f}%',
                             ha='center', va='bottom', fontsize=9,
                             color='darkgreen', fontweight='bold')

    ax3.set_xlabel('NoC Configuration', fontweight='bold')
    ax3.set_ylabel('Average Hops per Flit', fontweight='bold')
    ax3.set_title('(c) Network Communication Distance (128 Tokens)', fontweight='bold', fontsize=11)
    ax3.set_xticks(x_base + bar_width * 2.5)
    ax3.set_xticklabels(noc_order, rotation=15, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(1.5, 1.8)

    # (d) Power Reduction - NO LEGEND but WITH EFFECTIVE RANGE
    ax4 = fig.add_subplot(gs[1, 1])

    strategies_to_plot = ['SAMOS', 'Affiliated', 'Separated', 'MOSAIC-1', 'MOSAIC-2']
    markers = ['o', 's', '^', 'D', 'v']

    for j, strategy in enumerate(strategies_to_plot):
        improvements = []
        for noc in noc_order:
            noc_data = df[df['NoC_Display'] == noc]
            baseline_val = noc_data[noc_data['Strategy'] == 'Baseline']['BitTransitions'].values[0]
            strategy_val = noc_data[noc_data['Strategy'] == strategy]['BitTransitions'].values[0]
            improvement = ((baseline_val - strategy_val) / baseline_val) * 100
            improvements.append(improvement)

        line = ax4.plot(range(len(noc_order)), improvements,
                        marker=markers[j], linewidth=2, markersize=8,
                        label=strategy, color=colors[strategy], alpha=0.9)

        for i, y in enumerate(improvements):
            if i == 0 or i == len(improvements) - 1:
                ax4.text(i, y, f'{y:.1f}%', ha='center', va='bottom', fontsize=7)

    # Effective range for 16×16 and 32×32
    ax4.axvspan(1.5, 3.5, alpha=0.2, color='yellow')
    ax4.text(2.5, 36, 'Effective Range', ha='center', fontsize=10,
             color='darkred', fontweight='bold')

    ax4.set_xlabel('NoC Configuration', fontweight='bold')
    ax4.set_ylabel('Bit Transitions Reduction (%)', fontweight='bold')
    ax4.set_title('(d) Power Efficiency Improvement (128 Tokens)', fontweight='bold', fontsize=11)
    ax4.set_xticks(range(len(noc_order)))
    ax4.set_xticklabels(noc_order, rotation=15, ha='right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-5, 40)

    # (e) Performance-Power Trade-off with COMPLETE LEGEND
    ax5 = fig.add_subplot(gs[0:2, 2])
    ax5_twin = ax5.twinx()

    x_positions = np.arange(len(noc_order))
    bar_width_e = 0.25

    # Get data for all strategies
    all_cycles = {}
    all_bittrans = {}

    for strategy in strategies:
        cycles = []
        bittrans = []
        for noc in noc_order:
            noc_data = df[df['NoC_Display'] == noc]
            cycles.append(noc_data[noc_data['Strategy'] == strategy]['Total_Cycles'].values[0])
            bittrans.append(noc_data[noc_data['Strategy'] == strategy]['BitTransitions'].values[0])
        all_cycles[strategy] = cycles
        all_bittrans[strategy] = bittrans

    # Plot ONLY the 3 main strategies as visible bars
    bars1 = ax5.bar(x_positions - bar_width_e, all_cycles['Baseline'], bar_width_e,
                    color=colors['Baseline'], alpha=0.7, label='Baseline')
    bars2 = ax5.bar(x_positions, all_cycles['SAMOS'], bar_width_e,
                    color=colors['SAMOS'], alpha=0.7, label='SAMOS')
    bars3 = ax5.bar(x_positions + bar_width_e, all_cycles['MOSAIC-2'], bar_width_e,
                    color=colors['MOSAIC-2'], alpha=0.7, label='MOSAIC-2')

    # Add SAMOS labels with consistent styling
    baseline_cycles = all_cycles['Baseline']
    samos_cycles = all_cycles['SAMOS']

    for i, (baseline_c, samos_c) in enumerate(zip(baseline_cycles, samos_cycles)):
        if baseline_c != samos_c:
            reduction = ((baseline_c - samos_c) / baseline_c) * 100
            if abs(reduction) > 0.01:
                y_offset = max(baseline_cycles) * 0.02

                if i < 2:  # First two - normal black
                    text = f'+{abs(reduction):.1f}%' if reduction < 0 else f'-{reduction:.1f}%'
                    ax5.text(x_positions[i], samos_c + y_offset, text,
                             ha='center', va='bottom', fontsize=7,
                             color='black', rotation=90)
                else:  # Last two - prominent red
                    text = f'+{abs(reduction):.1f}%' if reduction < 0 else f'-{reduction:.1f}%'
                    ax5.text(x_positions[i], samos_c + y_offset / 2, text,
                             ha='center', va='bottom', fontsize=12,
                             color='red', fontweight='bold')

    # Calculate and plot power reduction for ALL strategies
    baseline_bt = all_bittrans['Baseline']

    for strategy in ['SAMOS', 'Affiliated', 'Separated', 'MOSAIC-1', 'MOSAIC-2']:
        reduction = [(b - s) / b * 100 for b, s in zip(baseline_bt, all_bittrans[strategy])]

        marker_map = {'SAMOS': 'o', 'Affiliated': 's', 'Separated': '^',
                      'MOSAIC-1': 'D', 'MOSAIC-2': 'v'}

        ax5_twin.plot(x_positions, reduction, marker=marker_map[strategy],
                      linewidth=2, markersize=8, color=colors[strategy],
                      label=f'{strategy} BT↓', alpha=0.9, linestyle='-')

    # Formatting
    ax5.set_xlabel('NoC Configuration', fontweight='bold')
    ax5.set_ylabel('Execution Cycles', fontweight='bold', color='black')
    ax5_twin.set_ylabel('Bit Transitions Reduction (%)', fontweight='bold', color='darkgreen')
    ax5.set_title('(e) Summary of Latency and BT Reduction Rate (128 Tokens)', fontweight='bold', fontsize=11)

    ax5.set_xticks(x_positions)
    ax5.set_xticklabels(noc_order, rotation=15, ha='right')
    ax5.tick_params(axis='y', labelcolor='black')
    ax5_twin.tick_params(axis='y', labelcolor='darkgreen')
    ax5.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Create manual comprehensive legend for ALL strategies
    legend_elements = [
        # Bars (Cycles)
        Patch(facecolor=colors['Baseline'], alpha=0.7, label='Baseline'),
        Patch(facecolor=colors['SAMOS'], alpha=0.7, label='SAMOS'),
        Patch(facecolor=colors['Affiliated'], alpha=0.7, label='Affiliated'),
        Patch(facecolor=colors['Separated'], alpha=0.7, label='Separated'),
        Patch(facecolor=colors['MOSAIC-1'], alpha=0.7, label='MOSAIC-1'),
        Patch(facecolor=colors['MOSAIC-2'], alpha=0.7, label='MOSAIC-2'),
        # Lines (BT Reduction)
        Line2D([0], [0], color=colors['SAMOS'], marker='o', markersize=8,
               linewidth=2, label='SAMOS BT↓'),
        Line2D([0], [0], color=colors['Affiliated'], marker='s', markersize=8,
               linewidth=2, label='Affiliated BT↓'),
        Line2D([0], [0], color=colors['Separated'], marker='^', markersize=8,
               linewidth=2, label='Separated BT↓'),
        Line2D([0], [0], color=colors['MOSAIC-1'], marker='D', markersize=8,
               linewidth=2, label='MOSAIC-1 BT↓'),
        Line2D([0], [0], color=colors['MOSAIC-2'], marker='v', markersize=8,
               linewidth=2, label='MOSAIC-2 BT↓')
    ]

    # Create comprehensive legend INSIDE the plot
    ax5.legend(handles=legend_elements,
               loc='center right', bbox_to_anchor=(0.98, 0.5),
               fontsize=6, ncol=2, framealpha=0.95,
               title='All Strategies', title_fontsize=7)

    ax5.grid(True, alpha=0.3)
    ax5_twin.set_ylim(-5, 35)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_filename = '128token_noc_performance_analysis.pdf'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Analysis saved to: {output_filename}")

    plt.savefig('128token_noc_performance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"PNG version saved to: 128token_noc_performance_analysis.png")

    plt.show()


if __name__ == "__main__":
    create_comprehensive_analysis()