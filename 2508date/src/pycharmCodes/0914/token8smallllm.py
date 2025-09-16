#!/usr/bin/env python3
"""
Small LLM Performance Analysis on NoC - Updated with New Data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_dataframe():
    """Create DataFrame from updated data"""
    data = {
        'NoC_Size': ['MC2_4x4'] * 6 + ['MC8_8x8'] * 6 + ['MC32_16x16'] * 6 + ['MC128_32x32'] * 6,
        'Strategy': ['baseline', 'samos', 'affilated', 'seperated', 'mosaic-1', 'mosaic-2'] * 4,
        'Total_Cycles': [
            # MC2_4x4
            420525, 397033, 420525, 420525, 397033, 397033,
            # MC8_8x8
            108283, 103120, 103120, 108283, 103120, 103120,
            # MC32_16x16
            28625, 28625, 28625, 28625, 28625, 28625,
            # MC128_32x32
            11588, 11588, 11588, 11588, 11588, 11588
        ],
        'Avg_Hops': [
            # MC2_4x4
            1.72, 1.69, 1.72, 1.72, 1.69, 1.69,
            # MC8_8x8
            1.72, 1.71, 1.71, 1.72, 1.71, 1.71,
            # MC32_16x16
            1.73, 1.73, 1.73, 1.73, 1.73, 1.73,
            # MC128_32x32
            1.73, 1.73, 1.73, 1.73, 1.73, 1.73
        ],
        'BitTransitions': [
            # MC2_4x4
            187594617, 184478613, 132051283, 128341675, 129880247, 126243317,
            # MC8_8x8
            187981955, 186305559, 131131595, 128616426, 131131595, 127445142,
            # MC32_16x16
            188659706, 188659706, 132932483, 129148839, 132932483, 129148839,
            # MC128_32x32
            188484220, 188484220, 132879145, 128923774, 132879145, 128923774
        ]
    }

    df = pd.DataFrame(data)

    # Map strategies to display names
    strategy_map = {
        'baseline': 'Baseline',
        'samos': 'Travel-Time',
        'affilated': 'Affiliated',
        'seperated': 'Separated',
        'mosaic-1': 'MOSAIC-1',
        'mosaic-2': 'MOSAIC-2'
    }
    df['Strategy'] = df['Strategy'].map(strategy_map)

    # Map NoC sizes for display - 去掉MC前缀，只保留尺寸
    noc_display_map = {
        'MC2_4x4': '4×4',
        'MC8_8x8': '8×8',
        'MC32_16x16': '16×16',
        'MC128_32x32': '32×32'
    }
    df['NoC_Display'] = df['NoC_Size'].map(noc_display_map)

    return df


def create_comprehensive_analysis():
    """Create comprehensive analysis figure with 5 subplots"""

    df = create_dataframe()

    # Set up the figure with compatible style
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')

    fig = plt.figure(figsize=(20, 8))

    # Color scheme
    colors = {
        'Baseline': '#003f5c',
        'Travel-Time': '#edc948',
        'Affiliated': '#665191',
        'Separated': '#a05195',
        'MOSAIC-1': '#d45087',
        'MOSAIC-2': '#ff7c43'
    }

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    strategies = ['Baseline', 'Travel-Time', 'Affiliated', 'Separated', 'MOSAIC-1', 'MOSAIC-2']
    noc_order = ['4×4', '8×8', '16×16', '32×32']  # 简化的NoC尺寸

    # (a) Execution Cycles - NO LEGEND
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
    ax1.set_title('(a) Execution Cycles (8 Tokens)', fontweight='bold', fontsize=11)
    ax1.set_xticks(x_base + bar_width * 2.5)
    ax1.set_xticklabels(noc_order, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # (b) Energy Consumption - NO LEGEND
    ax2 = fig.add_subplot(gs[0, 1])

    for i, strategy in enumerate(strategies):
        values = []
        for noc in noc_order:
            noc_data = df[(df['NoC_Display'] == noc) & (df['Strategy'] == strategy)]
            values.append(noc_data['BitTransitions'].values[0] / 1e6 if len(noc_data) > 0 else 0)

        bars = ax2.bar(x_base + i * bar_width, values, bar_width,
                       label=strategy, color=colors[strategy], alpha=0.85)

    ax2.set_xlabel('NoC Configuration', fontweight='bold')
    ax2.set_ylabel('Total Bit Transitions (M)', fontweight='bold')
    ax2.set_title('(b) Link Energy Consumption (8 Tokens)', fontweight='bold', fontsize=11)
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

        # Add percentage reduction labels for Travel-Time only
        if strategy == 'Travel-Time':
            baseline_values = []
            for noc in noc_order:
                baseline_data = df[(df['NoC_Display'] == noc) & (df['Strategy'] == 'Baseline')]
                baseline_values.append(baseline_data['Avg_Hops'].values[0] if len(baseline_data) > 0 else 0)

            for j, (traveltime_val, baseline_val) in enumerate(zip(values, baseline_values)):
                if baseline_val > 0:
                    reduction = ((baseline_val - traveltime_val) / baseline_val) * 100
                    ax3.text(x_base[j] + i * bar_width, traveltime_val + 0.01,
                             f'-{reduction:.1f}%',
                             ha='center', va='bottom', fontsize=15,
                             color='darkgreen', fontweight='bold')

    ax3.set_xlabel('NoC Configuration', fontweight='bold')
    ax3.set_ylabel('Average Hops per Flit', fontweight='bold')
    ax3.set_title('(c) Network Communication Distance (8 Tokens)', fontweight='bold', fontsize=11)
    ax3.set_xticks(x_base + bar_width * 2.5)
    ax3.set_xticklabels(noc_order, rotation=15, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    # (d)  vs Baseline - NO LEGEND
    ax4 = fig.add_subplot(gs[1, 1])

    strategies_to_plot = ['Travel-Time', 'Affiliated', 'Separated', 'MOSAIC-1', 'MOSAIC-2']
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
            if i == 0 or i == len(improvements) - 1:
                ax4.text(i, y, f'{y:.1f}%', ha='center', va='bottom', fontsize=7)

    # Highlight Travel-Time effectiveness for first two NoC configs
    ax4.axvspan(-0.5, 1.5, alpha=0.3, color='yellow')
    ax4.text(0.75, -3, 'Travel-Time Effective Range\n(≤8×8 NoC for this LLM)', ha='center',
             fontsize=12, color='darkred', fontweight='bold')

    ax4.set_xlabel('NoC Configuration', fontweight='bold')
    ax4.set_ylabel('Bit Transitions Reduction (%)', fontweight='bold')
    ax4.set_title('(d) Power Efficiency Improvement (8 Tokens)', fontweight='bold', fontsize=11)
    ax4.set_xticks(range(len(noc_order)))
    ax4.set_xticklabels(noc_order, rotation=15, ha='right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-5, 35)

    # (e) Performance-Power Trade-off - WITH ALL LEGENDS
    ax5 = fig.add_subplot(gs[0:2, 2])
    ax5_twin = ax5.twinx()

    # Compare Baseline, Travel-Time, and MOSAIC-1
    x_positions = np.arange(len(noc_order))
    bar_width_e = 0.25

    # Get data for three strategies
    baseline_cycles = []
    traveltime_cycles = []
    mosaic1_cycles = []
    baseline_bittrans = []
    traveltime_bittrans = []
    mosaic1_bittrans = []

    for noc in noc_order:
        noc_data = df[df['NoC_Display'] == noc]
        baseline_cycles.append(noc_data[noc_data['Strategy'] == 'Baseline']['Total_Cycles'].values[0])
        traveltime_cycles.append(noc_data[noc_data['Strategy'] == 'Travel-Time']['Total_Cycles'].values[0])
        mosaic1_cycles.append(noc_data[noc_data['Strategy'] == 'MOSAIC-1']['Total_Cycles'].values[0])
        baseline_bittrans.append(noc_data[noc_data['Strategy'] == 'Baseline']['BitTransitions'].values[0])
        traveltime_bittrans.append(noc_data[noc_data['Strategy'] == 'Travel-Time']['BitTransitions'].values[0])
        mosaic1_bittrans.append(noc_data[noc_data['Strategy'] == 'MOSAIC-1']['BitTransitions'].values[0])

    # Plot cycles bars
    bars1 = ax5.bar(x_positions - bar_width_e, baseline_cycles, bar_width_e,
                    color=colors['Baseline'], alpha=0.7, label='Baseline')
    bars2 = ax5.bar(x_positions, traveltime_cycles, bar_width_e,
                    color=colors['Travel-Time'], alpha=0.7, label='Travel-Time')
    bars3 = ax5.bar(x_positions + bar_width_e, mosaic1_cycles, bar_width_e,
                    color=colors['MOSAIC-1'], alpha=0.7, label='MOSAIC-1')

    # Add cycle reduction labels for Travel-Time
    for i, (baseline_c, traveltime_c) in enumerate(zip(baseline_cycles, traveltime_cycles)):
        reduction = ((baseline_c - traveltime_c) / baseline_c) * 100
        if reduction > 0:  # Only show if there's actual reduction
            y_offset = 17000 if i == 0 else 5000
            ax5.text(x_positions[i], traveltime_c + y_offset,
                     f'-{reduction:.1f}%',
                     ha='center', va='bottom',
                     fontsize=13, color='darkblue',
                     fontweight='bold', rotation=90)

    # Calculate and plot  lines
    traveltime_reduction = [(b - s) / b * 100 for b, s in zip(baseline_bittrans, traveltime_bittrans)]
    mosaic1_reduction = [(b - m) / b * 100 for b, m in zip(baseline_bittrans, mosaic1_bittrans)]

    line1 = ax5_twin.plot(x_positions, traveltime_reduction, 'o-', linewidth=2.5,
                          markersize=9, color=colors['Travel-Time'], label='Travel-Time BT Reduction')
    line2 = ax5_twin.plot(x_positions, mosaic1_reduction, 's-', linewidth=2.5,
                          markersize=9, color=colors['MOSAIC-1'], label='MOSAIC-1 BT Reduction')

    # Add Separated line
    separated_bittrans = []
    for noc in noc_order:
        noc_data = df[df['NoC_Display'] == noc]
        separated_bittrans.append(noc_data[noc_data['Strategy'] == 'Separated']['BitTransitions'].values[0])

    separated_reduction = [(b - s) / b * 100 for b, s in zip(baseline_bittrans, separated_bittrans)]

    line3 = ax5_twin.plot(x_positions, separated_reduction, '^-', linewidth=2.5,
                          markersize=9, color=colors['Separated'], label='Separated BT Reduction')

    # Labels and formatting
    ax5.set_xlabel('NoC Configuration', fontweight='bold')
    ax5.set_ylabel('Execution Cycles', fontweight='bold', color='black')
    ax5_twin.set_ylabel('Bit Transitions Reduction (%)', fontweight='bold', color='darkgreen')
    ax5.set_title('(e) Summary of Latency and BT Reduction Rate (8 Tokens)', fontweight='bold', fontsize=11)

    ax5.set_xticks(x_positions)
    ax5.set_xticklabels(noc_order, rotation=15, ha='right')
    ax5.tick_params(axis='y', labelcolor='black')
    ax5_twin.tick_params(axis='y', labelcolor='darkgreen')

    # Create comprehensive legend for ALL strategies
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # 创建所有策略的图例项（柱状图样式）
    strategy_patches = []
    strategy_labels = []

    # 添加所有6个策略的图例
    for strategy in strategies:
        patch = Patch(color=colors[strategy], alpha=0.7, label=strategy)
        strategy_patches.append(patch)
        strategy_labels.append(strategy)

    # 创建所有折线图的图例项
    line_legends = []
    line_labels_custom = []

    # 为图(d)中的所有策略创建折线图例
    marker_dict = {'Travel-Time': 'o', 'Affiliated': 's', 'Separated': '^',
                   'MOSAIC-1': 'D', 'MOSAIC-2': 'v'}

    for strategy in ['Travel-Time', 'Affiliated', 'Separated', 'MOSAIC-1', 'MOSAIC-2']:
        line = Line2D([0], [0], color=colors[strategy], linewidth=2.5,
                      marker=marker_dict[strategy],
                      markersize=8, label=f'{strategy} ')
        line_legends.append(line)
        line_labels_custom.append(f'{strategy} ')

    # 组合所有图例元素
    all_handles = strategy_patches + line_legends
    all_labels = strategy_labels + line_labels_custom

    # 创建组合图例，分两列显示
    ax5.legend(all_handles, all_labels,
               loc='center right', bbox_to_anchor=(0.99, 0.5),
               fontsize=8, ncol=2, framealpha=0.95,
               title='Strategies & ', title_fontsize=9)

    ax5.grid(True, alpha=0.3)
    ax5_twin.set_ylim(-5, 35)

    # Adjust layout
    plt.tight_layout()

    # Save figure as PDF
    plt.savefig('noc_performance_analysis_updated.pdf', dpi=150, bbox_inches='tight')
    print(f"Analysis saved to: noc_performance_analysis_updated.pdf")

    plt.show()


if __name__ == "__main__":
    create_comprehensive_analysis()