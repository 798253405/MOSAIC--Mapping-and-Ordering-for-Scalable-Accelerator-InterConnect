#!/usr/bin/env python3
"""
LeNet Performance Analysis on NoC - Comprehensive Visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_dataframe():
    """Create DataFrame from provided LeNet data"""
    data = {
        'NoC_Size': ['4x4'] * 6 + ['8x8'] * 6 + ['16x16'] * 6 + ['32x32'] * 6,
        'Strategy': ['baseline', 'samos', 'affiliated', 'separated', 'mosaic-1', 'mosaic-2'] * 4,
        'Total_Cycles': [
            # 4x4
            50221, 46274, 50221, 50221, 46274, 46274,
            # 8x8
            13166, 12580, 13166, 13166, 12580, 12580,
            # 16x16
            3967, 3924, 3967, 3967, 3924, 3924,
            # 32x32
            1883, 1883, 1883, 1883, 1883, 1883
        ],
        'Avg_Hops': [
            # 4x4
            1.72, 1.66, 1.72, 1.72, 1.66, 1.66,
            # 8x8
            1.72, 1.68, 1.72, 1.72, 1.68, 1.68,
            # 16x16
            1.73, 1.71, 1.73, 1.73, 1.71, 1.71,
            # 32x32
            1.73, 1.73, 1.73, 1.73, 1.73, 1.73
        ],
        'BitTransitions': [
            # 4x4
            21458990, 21011516, 17821086, 14958865, 17390662, 14601405,
            # 8x8
            21383552, 21011986, 17763324, 14943957, 17513034, 14712847,
            # 16x16
            21339814, 21172280, 17908451, 15074887, 17785726, 14959713,
            # 32x32
            20743205, 20743205, 17824210, 15032907, 17824210, 15032907
        ]
    }

    df = pd.DataFrame(data)

    # Map strategies to display names
    strategy_map = {
        'baseline': 'Baseline',
        'samos': 'Travel Time',
        'affiliated': 'Affiliated',
        'separated': 'Separated',
        'mosaic-1': 'MOSAIC-1',
        'mosaic-2': 'MOSAIC-2'
    }
    df['Strategy'] = df['Strategy'].map(strategy_map)

    # Map NoC sizes for display
    noc_map = {
        '4x4': '4×4',
        '8x8': '8×8',
        '16x16': '16×16',
        '32x32': '32×32'
    }
    df['NoC_Display'] = df['NoC_Size'].map(noc_map)

    return df


def create_comprehensive_analysis():
    """Create comprehensive analysis figure with 5 subplots for LeNet"""

    df = create_dataframe()

    # Set up the figure with compatible style
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')

    fig = plt.figure(figsize=(20, 8))

    # Define colors
    colors = {
        'Baseline': '#003f5c',  # 深海蓝
        'Travel Time': '#edc948',
        'Affiliated': '#665191',  # 紫罗兰
        'Separated': '#a05195',  # 洋红
        'MOSAIC-1': '#d45087',  # 玫瑰红
        'MOSAIC-2': '#ff7c43'  # 珊瑚橙
    }

    # Import and create gridspec
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    strategies = ['Baseline', 'Travel Time', 'Affiliated', 'Separated', 'MOSAIC-1', 'MOSAIC-2']
    noc_order = ['4×4', '8×8', '16×16', '32×32']

    # (a) Execution Cycles
    ax1 = fig.add_subplot(gs[0, 0])
    bar_width = 0.13
    x_base = np.arange(len(noc_order))

    for i, strategy in enumerate(strategies):
        values = []
        for noc in noc_order:
            noc_data = df[(df['NoC_Display'] == noc) & (df['Strategy'] == strategy)]
            values.append(noc_data['Total_Cycles'].values[0] if len(noc_data) > 0 else 0)

        bars = ax1.bar(x_base + i * bar_width, values, bar_width,
                       label=strategy, color=colors[strategy], alpha=0.85)

    ax1.set_xlabel('NoC Configuration', fontweight='bold')
    ax1.set_ylabel('Execution Cycles', fontweight='bold')
    ax1.set_title('(a) Execution Cycles', fontweight='bold', fontsize=11)
    ax1.set_xticks(x_base + bar_width * 2.5)
    ax1.set_xticklabels(noc_order, rotation=15, ha='right')
    # NO LEGEND for subplot (a)
    ax1.grid(True, alpha=0.3, axis='y')

    # (b) Energy Consumption (Bit Transitions)
    ax2 = fig.add_subplot(gs[0, 1])

    for i, strategy in enumerate(strategies):
        values = []
        for noc in noc_order:
            noc_data = df[(df['NoC_Display'] == noc) & (df['Strategy'] == strategy)]
            values.append(noc_data['BitTransitions'].values[0] / 1e6 if len(noc_data) > 0 else 0)  # Convert to M

        bars = ax2.bar(x_base + i * bar_width, values, bar_width,
                       label=strategy, color=colors[strategy], alpha=0.85)

    ax2.set_xlabel('NoC Configuration', fontweight='bold')
    ax2.set_ylabel('Total Bit Transitions (M)', fontweight='bold')
    ax2.set_title('(b) Link Energy Consumption', fontweight='bold', fontsize=11)
    ax2.set_xticks(x_base + bar_width * 2.5)
    ax2.set_xticklabels(noc_order, rotation=15, ha='right')
    # NO LEGEND for subplot (b)
    ax2.grid(True, alpha=0.3, axis='y')

    # (c) Average Hops
    ax3 = fig.add_subplot(gs[1, 0])

    for i, strategy in enumerate(strategies):
        values = []
        for noc in noc_order:
            noc_data = df[(df['NoC_Display'] == noc) & (df['Strategy'] == strategy)]
            values.append(noc_data['Avg_Hops'].values[0] if len(noc_data) > 0 else 0)

        bars = ax3.bar(x_base + i * bar_width, values, bar_width,
                       label=strategy, color=colors[strategy], alpha=0.85)

        # Add percentage reduction labels for Travel Time only
        if strategy == 'Travel Time':
            baseline_values = []
            for noc in noc_order:
                baseline_data = df[(df['NoC_Display'] == noc) & (df['Strategy'] == 'Baseline')]
                baseline_values.append(baseline_data['Avg_Hops'].values[0] if len(baseline_data) > 0 else 0)

            for j, (traveltime_val, baseline_val) in enumerate(zip(values, baseline_values)):
                if baseline_val > 0:
                    reduction = ((baseline_val - traveltime_val) / baseline_val) * 100
                    # Show all values, including 0%
                    ax3.text(x_base[j] + i * bar_width, traveltime_val + 0.02,
                             f'-{reduction:.1f}%',
                             ha='center', va='bottom', fontsize=7,
                             color='darkgreen', fontweight='bold')

    ax3.set_xlabel('NoC Configuration', fontweight='bold')
    ax3.set_ylabel('Average Hops per Flit', fontweight='bold')
    ax3.set_title('(c) Network Communication Distance', fontweight='bold', fontsize=11)
    ax3.set_xticks(x_base + bar_width * 2.5)
    ax3.set_xticklabels(noc_order, rotation=15, ha='right')
    # NO LEGEND for subplot (c)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(1.5, 1.85)  # Adjust y-axis for better visibility

    # (d) Power Reduction vs Baseline
    ax4 = fig.add_subplot(gs[1, 1])

    strategies_to_plot = ['Travel Time', 'Affiliated', 'Separated', 'MOSAIC-1', 'MOSAIC-2']
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

        # Add percentage labels for key points
        for i, y in enumerate(improvements):
            if i == 0 or i == len(improvements) - 1:  # First and last points
                ax4.text(i, y, f'{y:.1f}%', ha='center', va='bottom', fontsize=7)

    # Highlight Travel Time effectiveness
    ax4.axvspan(-0.5, 2.5, alpha=0.3, color='yellow')
    ax4.text(1, -2, 'Travel Time Effective Range\n(≤16×16 NoC for LeNet)', ha='center', fontsize=8,
             color='darkred', fontweight='bold')

    ax4.set_xlabel('NoC Configuration', fontweight='bold')
    ax4.set_ylabel('Bit Transitions Reduction (%)', fontweight='bold')
    ax4.set_title('(d) Power Efficiency Improvement', fontweight='bold', fontsize=11)
    ax4.set_xticks(range(len(noc_order)))
    ax4.set_xticklabels(noc_order, rotation=15, ha='right')
    # NO LEGEND for subplot (d)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-5, 35)

    # (e) Performance-Power Trade-off
    ax5 = fig.add_subplot(gs[0:2, 2])
    ax5_twin = ax5.twinx()

    # Compare Baseline, Travel Time, and MOSAIC-2
    x_positions = np.arange(len(noc_order))
    bar_width_e = 0.25

    # Get data for three strategies
    baseline_cycles = []
    traveltime_cycles = []
    mosaic2_cycles = []
    baseline_bittrans = []
    traveltime_bittrans = []
    mosaic2_bittrans = []
    separated_bittrans = []  # Add for Separated strategy

    for noc in noc_order:
        noc_data = df[df['NoC_Display'] == noc]
        baseline_cycles.append(noc_data[noc_data['Strategy'] == 'Baseline']['Total_Cycles'].values[0])
        traveltime_cycles.append(noc_data[noc_data['Strategy'] == 'Travel Time']['Total_Cycles'].values[0])
        mosaic2_cycles.append(noc_data[noc_data['Strategy'] == 'MOSAIC-2']['Total_Cycles'].values[0])
        baseline_bittrans.append(noc_data[noc_data['Strategy'] == 'Baseline']['BitTransitions'].values[0])
        traveltime_bittrans.append(noc_data[noc_data['Strategy'] == 'Travel Time']['BitTransitions'].values[0])
        mosaic2_bittrans.append(noc_data[noc_data['Strategy'] == 'MOSAIC-2']['BitTransitions'].values[0])
        separated_bittrans.append(
            noc_data[noc_data['Strategy'] == 'Separated']['BitTransitions'].values[0])  # Add Separated data

    # Plot cycles
    bars1 = ax5.bar(x_positions - bar_width_e, baseline_cycles, bar_width_e,
                    color=colors['Baseline'], alpha=0.7, label='Baseline')
    bars2 = ax5.bar(x_positions, traveltime_cycles, bar_width_e,
                    color=colors['Travel Time'], alpha=0.7, label='Travel Time')
    bars3 = ax5.bar(x_positions + bar_width_e, mosaic2_cycles, bar_width_e,
                    color=colors['MOSAIC-2'], alpha=0.7, label='MOSAIC-2')

    # Calculate and plot power reduction
    traveltime_reduction = [(b - s) / b * 100 for b, s in zip(baseline_bittrans, traveltime_bittrans)]
    mosaic2_reduction = [(b - m) / b * 100 for b, m in zip(baseline_bittrans, mosaic2_bittrans)]
    separated_reduction = [(b - s) / b * 100 for b, s in
                           zip(baseline_bittrans, separated_bittrans)]  # Add Separated reduction

    line1 = ax5_twin.plot(x_positions, traveltime_reduction, 'o-', linewidth=2.5,
                          markersize=9, color=colors['Travel Time'], label='Travel Time Power Reduction')
    line2 = ax5_twin.plot(x_positions, mosaic2_reduction, 's-', linewidth=2.5,
                          markersize=9, color=colors['MOSAIC-2'], label='MOSAIC-2 Power Reduction')
    line3 = ax5_twin.plot(x_positions, separated_reduction, '^-', linewidth=2.5,
                          markersize=9, color=colors['Separated'],
                          label='Separated Power Reduction')  # Add Separated line

    # Labels and formatting
    ax5.set_xlabel('NoC Configuration', fontweight='bold')
    ax5.set_ylabel('Execution Cycles', fontweight='bold', color='black')
    ax5_twin.set_ylabel('Bit Transitions Reduction (%)', fontweight='bold', color='darkgreen')
    ax5.set_title('(e) Summary of Latency and BT Reduction Rate', fontweight='bold', fontsize=11)

    ax5.set_xticks(x_positions)
    ax5.set_xticklabels(noc_order, rotation=15, ha='right')
    ax5.tick_params(axis='y', labelcolor='black')
    ax5_twin.tick_params(axis='y', labelcolor='darkgreen')

    # Import necessary modules for legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Create comprehensive legend elements
    legend_elements = [
        # Bar chart colors (6 strategies)
        Patch(facecolor=colors['Baseline'], alpha=0.85, label='Baseline'),
        Patch(facecolor=colors['Travel Time'], alpha=0.85, label='Travel Time'),
        Patch(facecolor=colors['Affiliated'], alpha=0.85, label='Affiliated'),
        Patch(facecolor=colors['Separated'], alpha=0.85, label='Separated'),
        Patch(facecolor=colors['MOSAIC-1'], alpha=0.85, label='MOSAIC-1'),
        Patch(facecolor=colors['MOSAIC-2'], alpha=0.85, label='MOSAIC-2'),
        # All line indicators from subplot (d) - 5 lines
        Line2D([0], [0], color=colors['Travel Time'], marker='o', markersize=8,
               linewidth=2, label='Travel Time '),
        Line2D([0], [0], color=colors['Affiliated'], marker='s', markersize=8,
               linewidth=2, label='Affiliated '),
        Line2D([0], [0], color=colors['Separated'], marker='^', markersize=8,
               linewidth=2, label='Separated '),
        Line2D([0], [0], color=colors['MOSAIC-1'], marker='D', markersize=8,
               linewidth=2, label='MOSAIC-1 '),
        Line2D([0], [0], color=colors['MOSAIC-2'], marker='v', markersize=8,
               linewidth=2, label='MOSAIC-2 ')
    ]

    # Add the comprehensive legend to subplot (e)
    ax5.legend(handles=legend_elements,
               loc='center right',
               bbox_to_anchor=(0.98, 0.5),
               fontsize=10,
               title='Strategies',
               title_fontsize=11,
               frameon=True,
               fancybox=True)

    ax5.grid(True, alpha=0.3)
    ax5_twin.set_ylim(-5, 35)

    # Adjust layout
    plt.tight_layout()

    # Save as PDF (vector format) for publication
    plt.savefig('lenetSmall_noc_performance_analysis.pdf', dpi=150, bbox_inches='tight')
    print(f"PDF saved to: lenet_noc_performance_analysis.pdf")

    plt.show()


if __name__ == "__main__":
    create_comprehensive_analysis()