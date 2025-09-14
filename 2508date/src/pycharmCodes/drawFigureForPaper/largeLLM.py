#!/usr/bin/env python3
"""
128-Token Small LLM Performance Analysis on NoC - Comprehensive Visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_dataframe():
    """Create DataFrame from provided 128-token configuration data"""
    data = {
        'NoC_Size': ['2_4x4'] * 6 + ['4_4x4'] * 6 + ['4_8x8'] * 6 + ['4_16x16'] * 6 + ['4_32x32'] * 6,
        'Strategy': ['baseline', 'TravelTime', 'affiliated', 'separated', 'mosaic-1', 'mosaic-2'] * 5,
        'Total_Cycles': [
            # 2_4x4
            6321329, 6670838, 6321329, 6321329, 6670838, 6670838,
            # 4_4x4
            7070360, 6777316, 7070360, 7070360, 6777316, 6777316,
            # 4_8x8
            2510068, 2364128, 2510068, 2510068, 2364128, 2364128,
            # 4_16x16
            2464564, 2453258, 2464564, 2464564, 2453258, 2453258,
            # 4_32x32
            2583446, 2571731, 2583446, 2583446, 2571731, 2571731
        ],
        'Avg_Hops': [
            # 2_4x4
            1.71, 1.68, 1.71, 1.71, 1.68, 1.68,
            # 4_4x4
            1.33, 1.32, 1.33, 1.33, 1.32, 1.32,
            # 4_8x8
            2.13, 2.09, 2.13, 2.13, 2.09, 2.09,
            # 4_16x16
            4.06, 4.05, 4.06, 4.06, 4.05, 4.05,
            # 4_32x32
            8.04, 8.01, 8.04, 8.04, 8.01, 8.01
        ],
        'BitTransitions': [
            # 2_4x4
            2996483603, 2937201979, 2104938412, 2048925106, 2063408240, 2008985952,
            # 4_4x4
            2329050107, 2301856884, 1636809877, 1593298336, 1619139887, 1576234065,
            # 4_8x8
            3731334719, 3658959294, 2663524326, 2594007524, 2609421510, 2541136617,
            # 4_16x16
            7104767572, 7078061034, 5061418287, 4926463734, 5040310504, 4907593037,
            # 4_32x32
            14050826458, 13988441060, 9987811185, 9719675332, 9948873053, 9681220918
        ]
    }

    df = pd.DataFrame(data)

    # Map strategies to display names
    strategy_map = {
        'baseline': 'Baseline',
        'TravelTime': 'TravelTime',
        'affiliated': 'Affiliated',
        'separated': 'Separated',
        'mosaic-1': 'MOSAIC-1',
        'mosaic-2': 'MOSAIC-2'
    }
    df['Strategy'] = df['Strategy'].map(strategy_map)

    # Map NoC sizes for display
    noc_map = {
        '2_4x4': 'MC2_4×4',
        '4_4x4': 'MC4_4×4',
        '4_8x8': 'MC4_8×8',
        '4_16x16': 'MC4_16×16',
        '4_32x32': 'MC4_32×32'
    }
    df['NoC_Display'] = df['NoC_Size'].map(noc_map)

    return df


def create_comprehensive_analysis():
    """Create comprehensive analysis figure with 5 subplots"""

    df = create_dataframe()

    # Set up the figure with compatible style - larger size
    try:
        plt.style.use('seaborn-darkgrid')  # Try older seaborn style name
    except:
        plt.style.use('default')  # Fallback to default

    fig = plt.figure(figsize=(20, 8))

    # Define color schemes
    colors_original = {
        'Baseline': '#003f5c',  # 深海蓝
        'TravelTime': '#edc948',  # 金黄
        'Affiliated': '#665191',  # 紫罗兰
        'Separated': '#a05195',  # 洋红
        'MOSAIC-1': '#d45087',  # 玫瑰红
        'MOSAIC-2': '#ff7c43'  # 珊瑚橙
    }

    colors = colors_original

    # Create gridspec
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    strategies = ['Baseline', 'TravelTime', 'Affiliated', 'Separated', 'MOSAIC-1', 'MOSAIC-2']
    noc_order = ['MC2_4×4', 'MC4_4×4', 'MC4_8×8', 'MC4_16×16', 'MC4_32×32']

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
    ax1.set_title('(a) Execution Cycles (128 Token)', fontweight='bold', fontsize=11)
    ax1.set_xticks(x_base + bar_width * 2.5)
    ax1.set_xticklabels(noc_order, rotation=15, ha='right')
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, axis='y')

    # (b) Energy Consumption (Bit Transitions)
    ax2 = fig.add_subplot(gs[0, 1])

    for i, strategy in enumerate(strategies):
        values = []
        for noc in noc_order:
            noc_data = df[(df['NoC_Display'] == noc) & (df['Strategy'] == strategy)]
            values.append(noc_data['BitTransitions'].values[0] / 1e9 if len(noc_data) > 0 else 0)  # Convert to G

        bars = ax2.bar(x_base + i * bar_width, values, bar_width,
                       label=strategy, color=colors[strategy], alpha=0.85)

    ax2.set_xlabel('NoC Configuration', fontweight='bold')
    ax2.set_ylabel('Total Bit Transitions (G)', fontweight='bold')
    ax2.set_title('(b) Energy Consumption (128 Token)', fontweight='bold', fontsize=11)
    ax2.set_xticks(x_base + bar_width * 2.5)
    ax2.set_xticklabels(noc_order, rotation=15, ha='right')
    ax2.legend(loc='upper left', fontsize=9, ncol=2)
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

        # Add percentage reduction labels for TravelTime only
        if strategy == 'TravelTime':
            baseline_values = []
            for noc in noc_order:
                baseline_data = df[(df['NoC_Display'] == noc) & (df['Strategy'] == 'Baseline')]
                baseline_values.append(baseline_data['Avg_Hops'].values[0] if len(baseline_data) > 0 else 0)

            for j, (TravelTime_val, baseline_val) in enumerate(zip(values, baseline_values)):
                if baseline_val > 0:
                    reduction = ((baseline_val - TravelTime_val) / baseline_val) * 100
                    ax3.text(x_base[j] + i * bar_width, TravelTime_val + 0.1,
                             f'-{reduction:.1f}%',
                             ha='center', va='bottom', fontsize=9,
                             color='darkgreen', fontweight='bold')

    ax3.set_xlabel('NoC Configuration', fontweight='bold')
    ax3.set_ylabel('Average Hops per Flit', fontweight='bold')
    ax3.set_title('(c) Network Communication Distance (128 Token)', fontweight='bold', fontsize=11)
    ax3.set_xticks(x_base + bar_width * 2.5)
    ax3.set_xticklabels(noc_order, rotation=15, ha='right')
    ax3.legend(loc='upper left', fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3, axis='y')

    # (d) Power Reduction vs Baseline
    ax4 = fig.add_subplot(gs[1, 1])

    strategies_to_plot = ['TravelTime', 'Affiliated', 'Separated', 'MOSAIC-1', 'MOSAIC-2']
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

    # Highlight TravelTime effectiveness
    ax4.axvspan(-0.5, 4.5, alpha=0.3, color='yellow')
    ax4.text(1, -3, 'TravelTime Effective Range  )', ha='center', fontsize=10,
             color='darkred', fontweight='bold')

    ax4.set_xlabel('NoC Configuration', fontweight='bold')
    ax4.set_ylabel('Bit Transitions Reduction (%)', fontweight='bold')
    ax4.set_title('(d) Power Efficiency Improvement (128 Token)', fontweight='bold', fontsize=11)
    ax4.set_xticks(range(len(noc_order)))
    ax4.set_xticklabels(noc_order, rotation=15, ha='right')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-5, 35)

    # (e) Performance-Power Trade-off - UPDATED WITH SEPARATED ORDERING
    ax5 = fig.add_subplot(gs[0:2, 2])
    ax5_twin = ax5.twinx()

    # Compare Baseline, TravelTime, and MOSAIC-2 (bars) + Separated (line)
    x_positions = np.arange(len(noc_order))
    bar_width_e = 0.25

    # Get data for all strategies
    baseline_cycles = []
    TravelTime_cycles = []
    mosaic2_cycles = []
    baseline_bittrans = []
    TravelTime_bittrans = []
    mosaic2_bittrans = []
    separated_bittrans = []

    for noc in noc_order:
        noc_data = df[df['NoC_Display'] == noc]
        baseline_cycles.append(noc_data[noc_data['Strategy'] == 'Baseline']['Total_Cycles'].values[0])
        TravelTime_cycles.append(noc_data[noc_data['Strategy'] == 'TravelTime']['Total_Cycles'].values[0])
        mosaic2_cycles.append(noc_data[noc_data['Strategy'] == 'MOSAIC-2']['Total_Cycles'].values[0])
        baseline_bittrans.append(noc_data[noc_data['Strategy'] == 'Baseline']['BitTransitions'].values[0])
        TravelTime_bittrans.append(noc_data[noc_data['Strategy'] == 'TravelTime']['BitTransitions'].values[0])
        mosaic2_bittrans.append(noc_data[noc_data['Strategy'] == 'MOSAIC-2']['BitTransitions'].values[0])
        separated_bittrans.append(noc_data[noc_data['Strategy'] == 'Separated']['BitTransitions'].values[0])

    # Plot execution cycles as bars
    bars1 = ax5.bar(x_positions - bar_width_e, baseline_cycles, bar_width_e,
                    color=colors['Baseline'], alpha=0.7, label='Baseline')
    bars2 = ax5.bar(x_positions, TravelTime_cycles, bar_width_e,
                    color=colors['TravelTime'], alpha=0.7, label='TravelTime')
    bars3 = ax5.bar(x_positions + bar_width_e, mosaic2_cycles, bar_width_e,
                    color=colors['MOSAIC-2'], alpha=0.7, label='MOSAIC-2')

    # Add TravelTime cycle reduction percentages
    for i, (baseline_c, TravelTime_c) in enumerate(zip(baseline_cycles, TravelTime_cycles)):
        if baseline_c != TravelTime_c:
            reduction = ((baseline_c - TravelTime_c) / baseline_c) * 100
            if reduction > 0:
                y_offset = max(baseline_cycles) * 0.02
                ax5.text(x_positions[i], TravelTime_c + y_offset,
                         f'-{reduction:.1f}%',
                         ha='center', va='bottom',
                         fontsize=9, color='darkblue',
                         fontweight='bold', rotation=90)

    # Calculate and plot power reduction for THREE strategies
    TravelTime_reduction = [(b - s) / b * 100 for b, s in zip(baseline_bittrans, TravelTime_bittrans)]
    mosaic2_reduction = [(b - m) / b * 100 for b, m in zip(baseline_bittrans, mosaic2_bittrans)]
    separated_reduction = [(b - s) / b * 100 for b, s in zip(baseline_bittrans, separated_bittrans)]

    # Plot all three power reduction lines - ORDER: TravelTime, Separated, MOSAIC-2
    line1 = ax5_twin.plot(x_positions, TravelTime_reduction, 'o-', linewidth=2.5,
                          markersize=9, color=colors['TravelTime'], label='TravelTime ', alpha=0.9)
    line2 = ax5_twin.plot(x_positions, separated_reduction, '^-', linewidth=2.5,
                          markersize=9, color=colors['Separated'], label='Separated ', alpha=0.9)
    line3 = ax5_twin.plot(x_positions, mosaic2_reduction, 's-', linewidth=2.5,
                          markersize=9, color=colors['MOSAIC-2'], label='MOSAIC-2  ', alpha=0.9)



    # Labels and formatting
    ax5.set_xlabel('NoC Configuration', fontweight='bold')
    ax5.set_ylabel('Execution Cycles', fontweight='bold', color='black')
    ax5_twin.set_ylabel('Bit Transitions Reduction (%)', fontweight='bold', color='darkgreen')
    ax5.set_title('(e) Performance-Power Trade-off (128 Token)', fontweight='bold', fontsize=11)

    ax5.set_xticks(x_positions)
    ax5.set_xticklabels(noc_order, rotation=15, ha='right')
    ax5.tick_params(axis='y', labelcolor='black')
    ax5_twin.tick_params(axis='y', labelcolor='darkgreen')

    # Combined legend - merge both axes legends
    handles1, labels1 = ax5.get_legend_handles_labels()
    handles2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(handles1 + handles2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.75, 0.78), fontsize=9, ncol=1)

    ax5.grid(True, alpha=0.3)
    ax5_twin.set_ylim(-5, 35)

    # Adjust layout
    plt.tight_layout()

    # Save figure as PDF (vector format for papers)
    output_filename = '128token_noc_performance_analysis.pdf'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Analysis saved to: {output_filename}")

    # Also save as high-resolution PNG for presentations
    plt.savefig('128token_noc_performance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"PNG version saved to: 128token_noc_performance_analysis.png")

    plt.show()


if __name__ == "__main__":
    create_comprehensive_analysis()