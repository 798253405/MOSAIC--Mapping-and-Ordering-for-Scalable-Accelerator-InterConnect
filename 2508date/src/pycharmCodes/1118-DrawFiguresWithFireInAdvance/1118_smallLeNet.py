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
        'NoC_Size': ['2_4x4'] * 8 + ['4_8x8'] * 8 + ['4_16x16'] * 8 + ['4_32x32'] * 8,
        'Strategy': ['baseline', 'TravelTime', 'affiliated', 'separated', 'Combo-1', 'Combo-2', 'mosaic-1', 'mosaic-2'] * 4,
        'Total_Cycles': [
            # 2_4x4
            50221, 46274, 50221, 50221, 46274, 46274, 45747, 45747,

            # 4_8x8
            13166, 12580, 13166, 13166, 12580, 12580, 12443, 12443,
            # 4_16x16
            3967, 3924, 3967, 3967, 3924, 3924, 3968, 3968,
            # 4_32x32
            1883, 1883, 1883, 1883, 1883, 1883, 1873, 1873,
        ],
        'Avg_Hops': [
            # 2_4x4
            1.72, 1.66, 1.72, 1.72, 1.66, 1.66, 1.67, 1.67,

            # 4_8x8
            1.72, 1.68,1.72, 1.72, 1.68, 1.68, 1.68, 1.68,
            # 4_16x16
            1.73, 1.71, 1.73, 1.73, 1.71, 1.71, 1.71, 1.71,
            # 4_32x32
            1.73, 1.73, 1.73, 1.73, 1.73, 1.73, 1.73, 1.73,
        ],
        'BitTransitions': [
            # 2_4x4
            21458990, 21011516, 17822580, 14955099, 17390962, 14596957, 17471918, 14648401,

            # 4_8x8
            21383552, 21011986, 17764991, 14940041, 17514002, 14710299, 17557224, 14750390,
            # 4_16x16
            21339814, 21172280, 17906924, 15073509, 17785388, 14955901, 17784506, 14941146,
            # 4_32x32
            20743205, 20743205, 17827385, 15027037, 17827037, 15027101, 17390032, 14702085,
        ],
        'Total_Flits': [74554] * 32  # All have the same total flits
    }

    df = pd.DataFrame(data)

    # Map strategies to display names
    strategy_map = {
        'baseline': 'Baseline',
        'TravelTime': 'TravelTime',
        'affiliated': 'Affiliated',
        'separated': 'Separated',
        'Combo-1': 'Combo-1',
        'Combo-2': 'Combo-2',
        'mosaic-1': 'MOSAIC-1',
        'mosaic-2': 'MOSAIC-2'
    }
    df['Strategy'] = df['Strategy'].map(strategy_map)

    # Map NoC sizes for display
    noc_map = {
        '2_4x4': 'MC2_4×4',
        '4_8x8': 'MC4_8×8',
        '4_16x16': 'MC4_16×16',
        '4_32x32': 'MC4_32×32'
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

    fig = plt.figure(figsize=(20, 10))

    # Define colors
    colors = {
        'Baseline': '#003f5c',  # 深海蓝
        'TravelTime': '#edc948',
        'Affiliated': '#665191',  # 紫罗兰
        'Separated': '#a05195',  # 洋红
        'Combo-1': '#bc5090',  # 新增
        'Combo-2': '#ef5675',  # 新增
        'MOSAIC-1': '#009E73',  # 深绿色
        'MOSAIC-2': '#4ECDC4'  # 青色
    }




    # Create gridspec
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.3)

    strategies = ['Baseline', 'TravelTime', 'Affiliated', 'Separated',  'Combo-1', 'Combo-2','MOSAIC-1', 'MOSAIC-2']
    noc_order = ['MC2_4×4',  'MC4_8×8', 'MC4_16×16', 'MC4_32×32']

    # (a) Execution Cycles
    ax1 = fig.add_subplot(gs[0, 0])
    bar_width = 0.10
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
    ax1.set_xticks(x_base + bar_width * 3.5)
    ax1.set_xticklabels(noc_order, rotation=15, ha='right')
    ax1.legend(loc='upper right', fontsize=7, ncol=2)
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
    ax2.set_xticks(x_base + bar_width * 3.5)
    ax2.set_xticklabels(noc_order, rotation=15, ha='right')
    ax2.legend(loc='upper left', fontsize=7, ncol=2)
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

        # # Add percentage reduction labels for TravelTime only
        # if strategy == 'TravelTime':
        #     baseline_values = []
        #     for noc in noc_order:
        #         baseline_data = df[(df['NoC_Display'] == noc) & (df['Strategy'] == 'Baseline')]
        #         baseline_values.append(baseline_data['Avg_Hops'].values[0] if len(baseline_data) > 0 else 0)
        #
        #     for j, (TravelTime_val, baseline_val) in enumerate(zip(values, baseline_values)):
        #         if baseline_val > 0:
        #             reduction = ((baseline_val - TravelTime_val) / baseline_val) * 100
        #             # Show all values, including 0%
        #             ax3.text(x_base[j] + i * bar_width, TravelTime_val + 0.15,
        #                      f'-{reduction:.1f}%',
        #                      ha='center', va='bottom', fontsize=7,
        #                      color='darkgreen', fontweight='bold')

    ax3.set_xlabel('NoC Configuration', fontweight='bold')
    ax3.set_ylabel('Average Hops per Flit', fontweight='bold')
    ax3.set_title('(c) Network Communication Distance', fontweight='bold', fontsize=11)
    ax3.set_xticks(x_base + bar_width * 3.5)
    ax3.set_xticklabels(noc_order, rotation=15, ha='right')
    ax3.legend(loc='upper left', fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3, axis='y')

    # (d) Power Reduction vs Baseline
    ax4 = fig.add_subplot(gs[1, 1])

    strategies_to_plot = ['TravelTime', 'Affiliated', 'Separated', 'Combo-1', 'Combo-2', 'MOSAIC-1', 'MOSAIC-2']
    markers = ['o', 's', '^', 'p', 'h', 'D', 'v']

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

        # # Add percentage labels for key points
        # for i, y in enumerate(improvements):
        #     if i == 0 or i == len(improvements) - 1:  # First and last points
        #         ax4.text(i, y, f'{y:.1f}%', ha='center', va='bottom', fontsize=7)

    ax4.set_xlabel('NoC Configuration', fontweight='bold')
    ax4.set_ylabel('Bit Transitions Reduction (%)', fontweight='bold')
    ax4.set_title('(d) Power Efficiency Improvement', fontweight='bold', fontsize=11)
    ax4.set_xticks(range(len(noc_order)))
    ax4.set_xticklabels(noc_order, rotation=15, ha='right')
    ax4.legend(loc='best', fontsize=7)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-5, 35)

    # (e) Performance-Power Trade-off
    ax5 = fig.add_subplot(gs[0:2, 2])
    ax5_twin = ax5.twinx()

    # Compare Baseline, TravelTime, Combo-1, and MOSAIC-1
    x_positions = np.arange(len(noc_order))
    bar_width_e = 0.20

    # Get data for four strategies
    baseline_cycles = []
    TravelTime_cycles = []
    combo1_cycles = []
    mosaic1_cycles = []
    baseline_bittrans = []
    TravelTime_bittrans = []
    mosaic1_bittrans = []
    affiliated_bittrans = []  # Add for Affiliated strategy
    combo1_bittrans = []  # Add for Combo-1 strategy

    for noc in noc_order:
        noc_data = df[df['NoC_Display'] == noc]
        baseline_cycles.append(noc_data[noc_data['Strategy'] == 'Baseline']['Total_Cycles'].values[0])
        TravelTime_cycles.append(noc_data[noc_data['Strategy'] == 'TravelTime']['Total_Cycles'].values[0])
        combo1_cycles.append(noc_data[noc_data['Strategy'] == 'Combo-1']['Total_Cycles'].values[0])
        mosaic1_cycles.append(noc_data[noc_data['Strategy'] == 'MOSAIC-1']['Total_Cycles'].values[0])
        baseline_bittrans.append(noc_data[noc_data['Strategy'] == 'Baseline']['BitTransitions'].values[0])
        TravelTime_bittrans.append(noc_data[noc_data['Strategy'] == 'TravelTime']['BitTransitions'].values[0])
        mosaic1_bittrans.append(noc_data[noc_data['Strategy'] == 'MOSAIC-1']['BitTransitions'].values[0])
        affiliated_bittrans.append(
            noc_data[noc_data['Strategy'] == 'Affiliated']['BitTransitions'].values[0])  # Add Affiliated data
        combo1_bittrans.append(
            noc_data[noc_data['Strategy'] == 'Combo-1']['BitTransitions'].values[0])  # Add Combo-1 data

    # Plot cycles - 4 bars
    bars1 = ax5.bar(x_positions - bar_width_e * 1.5, baseline_cycles, bar_width_e,
                    color=colors['Baseline'], alpha=0.7, label='Baseline')
    bars2 = ax5.bar(x_positions - bar_width_e * 0.5, TravelTime_cycles, bar_width_e,
                    color=colors['TravelTime'], alpha=0.7, label='TravelTime')
    bars3 = ax5.bar(x_positions + bar_width_e * 0.5, combo1_cycles, bar_width_e,
                    color=colors['Combo-1'], alpha=0.7, label='Combo-1')
    bars4 = ax5.bar(x_positions + bar_width_e * 1.5, mosaic1_cycles, bar_width_e,
                    color=colors['MOSAIC-1'], alpha=0.7, label='MOSAIC-1')

    # Calculate and plot power reduction
    TravelTime_reduction = [(b - s) / b * 100 for b, s in zip(baseline_bittrans, TravelTime_bittrans)]
    mosaic1_reduction = [(b - m) / b * 100 for b, m in zip(baseline_bittrans, mosaic1_bittrans)]
    affiliated_reduction = [(b - a) / b * 100 for b, a in
                           zip(baseline_bittrans, affiliated_bittrans)]  # Add Affiliated reduction
    combo1_reduction = [(b - c) / b * 100 for b, c in
                           zip(baseline_bittrans, combo1_bittrans)]  # Add Combo-1 reduction

    line1 = ax5_twin.plot(x_positions, TravelTime_reduction, 'o-', linewidth=2.5,
                          markersize=9, color=colors['TravelTime'], label='TravelTime Power Reduction')
    line2 = ax5_twin.plot(x_positions, mosaic1_reduction, 's-', linewidth=2.5,
                          markersize=9, color=colors['MOSAIC-1'], label='MOSAIC-1 Power Reduction')
    line3 = ax5_twin.plot(x_positions, affiliated_reduction, '^-', linewidth=2.5,
                          markersize=9, color=colors['Affiliated'],
                          label='Affiliated Power Reduction')  # Add Affiliated line
    line4 = ax5_twin.plot(x_positions, combo1_reduction, 'D-', linewidth=2.5,
                          markersize=9, color=colors['Combo-1'],
                          label='Combo-1 Power Reduction')  # Add Combo-1 line



    # Labels and formatting
    ax5.set_xlabel('NoC Configuration', fontweight='bold')
    ax5.set_ylabel('Execution Cycles', fontweight='bold', color='black')
    ax5_twin.set_ylabel('Bit Transitions Reduction (%)', fontweight='bold', color='darkgreen')
    ax5.set_title('(e) Performance-Power Trade-off', fontweight='bold', fontsize=11)

    ax5.set_xticks(x_positions)
    ax5.set_xticklabels(noc_order, rotation=15, ha='right')
    ax5.tick_params(axis='y', labelcolor='black')
    ax5_twin.tick_params(axis='y', labelcolor='darkgreen')

    # Get all legend elements
    bars_legend = ax5.get_legend_handles_labels()

    # Combine bar chart and line chart legends
    combined_handles = bars_legend[0] + [line1[0], line2[0], line3[0], line4[0]]
    combined_labels = bars_legend[1] + ['TravelTime Reduction', 'MOSAIC-1 Reduction', 'Affiliated Reduction', 'Combo-1 Reduction']

    # Create combined legend
    ax5.legend(combined_handles, combined_labels, loc='best', fontsize=7, ncol=1)

    ax5.grid(True, alpha=0.3)
    ax5_twin.set_ylim(-5, 35)

    # Adjust layout
    plt.tight_layout()

    # Save as PDF (vector format) for publication
    plt.savefig('lenet_noc_performance_analysis.pdf', dpi=150, bbox_inches='tight')
    print(f"PDF saved to: lenet_noc_performance_analysis.pdf")

    plt.show()


if __name__ == "__main__":
    create_comprehensive_analysis()