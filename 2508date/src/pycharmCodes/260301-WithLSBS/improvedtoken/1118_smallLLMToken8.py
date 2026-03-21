#!/usr/bin/env python3
"""
Small LLM Performance Analysis on NoC - Comprehensive Visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_dataframe():
    """Create DataFrame from provided data"""
    data = {
        'NoC_Size': ['2_4x4'] * 8 + ['4_8x8'] * 8 + ['4_16x16'] * 8 + ['4_32x32'] * 8,
        'Strategy': ['baseline', 'TravelTime', 'affiliated', 'separated', 'mosaic-1', 'mosaic-2', 'mosaic_new-1', 'mosaic_new-2'] * 4,
        'Total_Cycles': [
            # 2_4x4
            420525, 397033, 420525, 420525, 397033, 397033, 328296, 328296,
            # 4_8x8
            108283, 103120, 108283, 108283, 103120, 103120, 85382, 85382,
            # 4_16x16
            28625, 28625, 28625, 28625, 28625, 28625, 23682, 23682,
            # 4_32x32
            11588, 11588, 11588, 11588, 11588, 11588, 9572, 9572,
        ],
        'Avg_Hops': [
            # 2_4x4
            1.72, 1.69, 1.72, 1.72, 1.69, 1.69, 1.69, 1.69,
            # 4_8x8
            1.72, 1.70, 1.72, 1.72, 1.70, 1.70, 1.70, 1.70,
            # 4_16x16
            1.73, 1.73, 1.73, 1.73, 1.73, 1.73, 1.73, 1.73,
            # 4_32x32
            1.73, 1.73, 1.73, 1.73, 1.73, 1.73, 1.73, 1.73,
        ],
        'BitTransitions': [
            # 2_4x4
            187594617, 184478613, 132051283, 128341675, 129880247, 126243317, 130266197, 126637265,
            # 4_8x8
            187981955, 186305559, 132344558, 128616426, 131131595, 127445142, 131338550, 127663255,
            # 4_16x16
            188659706, 188659706, 132932483, 129148839, 132932483, 129148839, 133078378, 129344788,
            # 4_32x32
            188484220, 188484220, 132879145, 128923774, 132879145, 128923774, 133045579, 129208722,
        ]
    }

    df = pd.DataFrame(data)

    # Map strategies to display names
    strategy_map = {
        'baseline': 'Baseline',
        'TravelTime': 'TravelTime',
        'affiliated': 'Affiliated',
        'separated': 'Separated',
        'mosaic-1': 'Combo-1',
        'mosaic-2': 'Combo-2',
        'mosaic_new-1': 'MOSAIC-1',
        'mosaic_new-2': 'MOSAIC-2'
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
    """Create comprehensive analysis figure with 5 subplots"""

    df = create_dataframe()

    # Set up the figure with compatible style
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')

    fig = plt.figure(figsize=(20, 10))  # Reduced height for tighter layout

    # Define color schemes - Choose your preferred scheme here

    # Option 1: Nature Publishing Group Style (蓝绿色系) - 推荐
    colors_nature = {
        'Baseline': '#0173B2',  # 深蓝
        'TravelTime': '#56B4E9',  # 天蓝
        'Affiliated': '#009E73',  # 蓝绿
        'Separated': '#F0E442',  # 黄色
        'MOSAIC-1': '#E69F00',  # 橙色
        'MOSAIC-2': '#CC79A7'  # 粉紫
    }

    # Option 2: IEEE Style (经典蓝红色系)
    colors_ieee = {
        'Baseline': '#1f77b4',  # 标准蓝
        'TravelTime': '#ff7f0e',  # 橙色
        'Affiliated': '#2ca02c',  # 绿色
        'Separated': '#d62728',  # 红色
        'MOSAIC-1': '#9467bd',  # 紫色
        'MOSAIC-2': '#8c564b'  # 棕色
    }

    # Option 3: Science/Cell Style (冷色调专业)
    colors_science = {
        'Baseline': '#003f5c',  # 深海蓝
        'TravelTime': '#2f4b7c',  # 皇家蓝
        'Affiliated': '#665191',  # 紫罗兰
        'Separated': '#a05195',  # 洋红
        'MOSAIC-1': '#d45087',  # 玫瑰红
        'MOSAIC-2': '#ff7c43'  # 珊瑚橙
    }

    # Option 4: Original (原始配色)
    colors_original = {
    'Baseline': '#003f5c',     # 深海蓝
    'TravelTime': '#edc948',   # 金黄
    'Affiliated': '#665191',   # 紫罗兰
    'Separated': '#a05195',    # 洋红
    'Combo-1': '#bc5090',      # 新增
    'Combo-2': '#ef5675',      # 新增
    'MOSAIC-1': '#009E73',     # 深绿色
    'MOSAIC-2': '#4ECDC4'      # 青色
    }
    # SELECT YOUR PREFERRED COLOR SCHEME HERE
    colors = colors_original # Change this to colors_ieee, colors_science, or colors_original

    # Create gridspec
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.3)

    strategies = ['Baseline', 'TravelTime', 'Affiliated', 'Separated', 'Combo-1', 'Combo-2', 'MOSAIC-1', 'MOSAIC-2']
    noc_order = ['MC2_4×4', 'MC4_8×8', 'MC4_16×16', 'MC4_32×32']

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

   # (b) Standalone Method Improvement (4×4 NoC, 8-token LLM)
    ax2 = fig.add_subplot(gs[0, 1])

    base_cyc = 420525
    base_bt  = 187594617

    standalone_methods = ['TravelTime', 'Affiliated', 'Separated', 'FIA', 'Routing Switch', 'LSBS-A']
    cyc_vals = [397033, 420525, 420525, 357329, 415068, 420525]
    bt_vals  = [184478613, 132051283, 128341675, 187617877, 187603775, 126881215]

    cyc_red = [(base_cyc - v) / base_cyc * 100 for v in cyc_vals]
    bt_red  = [(base_bt - v) / base_bt * 100 for v in bt_vals]

    x_m = np.arange(len(standalone_methods))
    w = 0.35

    bars_cyc = ax2.bar(x_m - w/2, cyc_red, w, label='Cycle Reduction',
                       color='#2166ac', edgecolor='black', linewidth=0.4)
    bars_bt  = ax2.bar(x_m + w/2, bt_red, w, label='BT Reduction',
                       color='#b2182b', edgecolor='black', linewidth=0.4)

    for bar in bars_cyc:
        h = bar.get_height()
        #if abs(h) > 0.5:
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5,
               f'{h:.4f}%', ha='center', va='bottom', fontsize=7, color='#2166ac')
    for bar in bars_bt:
        h = bar.get_height()
        #if abs(h) > 0.5:
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5,
               f'{h:.4f}%', ha='center', va='bottom', fontsize=7, color='#b2182b')

    ax2.axhline(y=0, color='gray', linewidth=0.5)
    ax2.set_xlabel('Standalone Method', fontweight='bold')
    ax2.set_ylabel('Improvement, higher is better (%)', fontweight='bold')
    ax2.set_title('(b) Standalone Method Improvement', fontweight='bold', fontsize=11)
    ax2.set_xticks(x_m)
    ax2.set_xticklabels(standalone_methods, rotation=25, ha='right', fontsize=8)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(-2, 38)

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
        #             ax3.text(x_base[j] + i * bar_width, TravelTime_val + 0.1,
        #                      f'-{reduction:.1f}%',
        #                      ha='center', va='bottom', fontsize=15,
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
    affiliated_bittrans = []
    combo1_bittrans = []

    for noc in noc_order:
        noc_data = df[df['NoC_Display'] == noc]
        baseline_cycles.append(noc_data[noc_data['Strategy'] == 'Baseline']['Total_Cycles'].values[0])
        TravelTime_cycles.append(noc_data[noc_data['Strategy'] == 'TravelTime']['Total_Cycles'].values[0])
        combo1_cycles.append(noc_data[noc_data['Strategy'] == 'Combo-1']['Total_Cycles'].values[0])
        mosaic1_cycles.append(noc_data[noc_data['Strategy'] == 'MOSAIC-1']['Total_Cycles'].values[0])
        baseline_bittrans.append(noc_data[noc_data['Strategy'] == 'Baseline']['BitTransitions'].values[0])
        TravelTime_bittrans.append(noc_data[noc_data['Strategy'] == 'TravelTime']['BitTransitions'].values[0])
        mosaic1_bittrans.append(noc_data[noc_data['Strategy'] == 'MOSAIC-1']['BitTransitions'].values[0])
        affiliated_bittrans.append(noc_data[noc_data['Strategy'] == 'Affiliated']['BitTransitions'].values[0])
        combo1_bittrans.append(noc_data[noc_data['Strategy'] == 'Combo-1']['BitTransitions'].values[0])

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
    affiliated_reduction = [(b - a) / b * 100 for b, a in zip(baseline_bittrans, affiliated_bittrans)]
    combo1_reduction = [(b - c) / b * 100 for b, c in zip(baseline_bittrans, combo1_bittrans)]

    line1 = ax5_twin.plot(x_positions, TravelTime_reduction, 'o-', linewidth=2.5,
                          markersize=9, color=colors['TravelTime'], label='TravelTime Power Reduction')
    line2 = ax5_twin.plot(x_positions, mosaic1_reduction, 's-', linewidth=2.5,
                          markersize=9, color=colors['MOSAIC-1'], label='MOSAIC-1 Power Reduction')
    line3 = ax5_twin.plot(x_positions, affiliated_reduction, '^-', linewidth=2.5,
                          markersize=9, color=colors['Affiliated'], label='Affiliated Power Reduction')
    line4 = ax5_twin.plot(x_positions, combo1_reduction, 'D-', linewidth=2.5,
                          markersize=9, color=colors['Combo-1'], label='Combo-1 Power Reduction')

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
    combined_handles = bars_legend[0] + [line1[0], line3[0], line4[0], line2[0]]
    combined_labels = bars_legend[1] + ['TravelTime', 'Affiliated', 'Combo-1', 'MOSAIC-1']

    # Create combined legend
    ax5.legend(combined_handles, combined_labels, loc='best', fontsize=7, ncol=1)
    ax5.grid(True, alpha=0.3)
    ax5_twin.set_ylim(-5, 35)

    # Adjust layout
    plt.tight_layout()

    # Save figure as PDF (vector format for papers)
    plt.savefig('noc_performance_analysis.pdf', dpi=150, bbox_inches='tight')
    print(f"Analysis saved to: noc_performance_analysis.pdf")

    plt.show()


if __name__ == "__main__":
    create_comprehensive_analysis()
