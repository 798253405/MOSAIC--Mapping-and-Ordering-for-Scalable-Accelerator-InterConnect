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
        'NoC_Size': ['2_4x4'] * 6 + ['4_4x4'] * 6 + ['4_8x8'] * 6 + ['4_16x16'] * 6 + ['4_32x32'] * 6,
        'Strategy': ['baseline', 'TravelTime', 'affiliated', 'separated', 'mosaic-1', 'mosaic-2'] * 5,
        'Total_Cycles': [
            # 2_4x4
            420525, 397033, 420525, 420525, 397033, 397033,
            # 4_4x4
            445080, 425779, 445080, 445080, 425779, 425779,
            # 4_8x8
            162251, 153608, 162251, 162251, 153608, 153608,
            # 4_16x16
            159753, 159753, 159753, 159753, 159753, 159753,
            # 4_32x32
            168154, 168154, 168154, 168154, 168154, 168154
        ],
        'Avg_Hops': [
            # 2_4x4
            1.72, 1.69, 1.72, 1.72, 1.69, 1.69,
            # 4_4x4
            1.33, 1.32, 1.33, 1.33, 1.32, 1.32,
            # 4_8x8
            2.14, 2.11, 2.14, 2.14, 2.11, 2.11,
            # 4_16x16
            4.09, 4.09, 4.09, 4.09, 4.09, 4.09,
            # 4_32x32
            8.06, 8.06, 8.06, 8.06, 8.06, 8.06
        ],
        'BitTransitions': [
            # 2_4x4
            187594617, 184478613, 132051283, 128341675, 129880247, 126243317,
            # 4_4x4
            145777160, 144310646, 102431525, 99568406, 101406659, 98548476,
            # 4_8x8
            233601619, 230675328, 166801208, 162125635, 164635347, 159958076,
            # 4_16x16
            446351192, 446351192, 318665356, 308790976, 318665356, 308790976,
            # 4_32x32
            877441916, 877441916, 624509061, 604536890, 624509061, 604536890
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

    fig = plt.figure(figsize=(20, 8))  # Increased from 18,11 to 20,12

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
    'TravelTime': '#edc948',        # 金黄
    'Affiliated': '#665191',    # 紫罗兰
    'Separated': '#a05195',     # 洋红
    'MOSAIC-1': '#d45087',     # 玫瑰红
    'MOSAIC-2': '#ff7c43'      # 珊瑚橙
    }
    # SELECT YOUR PREFERRED COLOR SCHEME HERE
    colors = colors_original # Change this to colors_ieee, colors_science, or colors_original

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
    ax1.set_title('(a) Execution Cycles', fontweight='bold', fontsize=11)
    ax1.set_xticks(x_base + bar_width * 2.5)
    ax1.set_xticklabels(noc_order, rotation=15, ha='right')
    ax1.legend(loc='upper right', fontsize=13, ncol=2)
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
    ax2.set_title('(b) Energy Consumption', fontweight='bold', fontsize=11)
    ax2.set_xticks(x_base + bar_width * 2.5)
    ax2.set_xticklabels(noc_order, rotation=15, ha='right')
    ax2.legend(loc='upper left', fontsize=13, ncol=2)
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
                    # Show all values, including 0%
                    ax3.text(x_base[j] + i * bar_width, TravelTime_val + 0.1,
                             f'-{reduction:.1f}%',
                             ha='center', va='bottom', fontsize=15,
                             color='darkgreen', fontweight='bold')

    ax3.set_xlabel('NoC Configuration', fontweight='bold')
    ax3.set_ylabel('Average Hops per Flit', fontweight='bold')
    ax3.set_title('(c) Network Communication Distance', fontweight='bold', fontsize=11)
    ax3.set_xticks(x_base + bar_width * 2.5)
    ax3.set_xticklabels(noc_order, rotation=15, ha='right')
    ax3.legend(loc='upper left', fontsize=13, ncol=2)
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

    # Highlight TravelTime effectiveness for first three NoC configs (2_4x4, 4_4x4, 4_8x8)
    ax4.axvspan(-0.5, 2.5, alpha=0.3, color='yellow')  # Increased alpha from 0.15 to 0.3
    ax4.text(1, -3, 'TravelTime Effective Range 8×8 NoC \n(≤for this small LLM configuration)', ha='center', fontsize=12,
             color='darkred', fontweight='bold')

    ax4.set_xlabel('NoC Configuration', fontweight='bold')
    ax4.set_ylabel('Bit Transitions Reduction (%)', fontweight='bold')
    ax4.set_title('(d) Power Efficiency Improvement', fontweight='bold', fontsize=11)
    ax4.set_xticks(range(len(noc_order)))
    ax4.set_xticklabels(noc_order, rotation=15, ha='right')
    ax4.legend(loc='best', fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-5, 35)

    # (e) Performance-Power Trade-off
    ax5 = fig.add_subplot(gs[0:2, 2])
    ax5_twin = ax5.twinx()

    # Compare Baseline, TravelTime, and MOSAIC-2
    x_positions = np.arange(len(noc_order))
    bar_width_e = 0.25

    # Get data for three strategies
    baseline_cycles = []
    TravelTime_cycles = []
    mosaic2_cycles = []
    baseline_bittrans = []
    TravelTime_bittrans = []
    mosaic2_bittrans = []

    for noc in noc_order:
        noc_data = df[df['NoC_Display'] == noc]
        baseline_cycles.append(noc_data[noc_data['Strategy'] == 'Baseline']['Total_Cycles'].values[0])
        TravelTime_cycles.append(noc_data[noc_data['Strategy'] == 'TravelTime']['Total_Cycles'].values[0])
        mosaic2_cycles.append(noc_data[noc_data['Strategy'] == 'MOSAIC-2']['Total_Cycles'].values[0])
        baseline_bittrans.append(noc_data[noc_data['Strategy'] == 'Baseline']['BitTransitions'].values[0])
        TravelTime_bittrans.append(noc_data[noc_data['Strategy'] == 'TravelTime']['BitTransitions'].values[0])
        mosaic2_bittrans.append(noc_data[noc_data['Strategy'] == 'MOSAIC-2']['BitTransitions'].values[0])

    # Plot cycles
    bars1 = ax5.bar(x_positions - bar_width_e, baseline_cycles, bar_width_e,
                    color=colors['Baseline'], alpha=0.7, label='Baseline')
    bars2 = ax5.bar(x_positions, TravelTime_cycles, bar_width_e,
                    color=colors['TravelTime'], alpha=0.7, label='TravelTime')
    bars3 = ax5.bar(x_positions + bar_width_e, mosaic2_cycles, bar_width_e,
                    color=colors['MOSAIC-2'], alpha=0.7, label='MOSAIC-2')
    for i, (baseline_c, TravelTime_c) in enumerate(zip(baseline_cycles, TravelTime_cycles)):
        reduction = ((baseline_c - TravelTime_c) / baseline_c) * 100
        if (i == 1):
            ax5.text(x_positions[i], TravelTime_c + 17000,
                     f'-{reduction:.1f}%',
                     ha='center', va='bottom',
                     fontsize=13, color='darkblue',
                     fontweight='bold', rotation=90)
        else:
            ax5.text(x_positions[i], TravelTime_c + 5000,
                     f'-{reduction:.1f}%',
                     ha='center', va='bottom',
                     fontsize=13, color='darkblue',
                     fontweight='bold', rotation=90)
    # Calculate and plot power reduction
    TravelTime_reduction = [(b - s) / b * 100 for b, s in zip(baseline_bittrans, TravelTime_bittrans)]
    mosaic2_reduction = [(b - m) / b * 100 for b, m in zip(baseline_bittrans, mosaic2_bittrans)]

    line1 = ax5_twin.plot(x_positions, TravelTime_reduction, 'o-', linewidth=2.5,
                          markersize=9, color=colors['TravelTime'])
    line2 = ax5_twin.plot(x_positions, mosaic2_reduction, 's-', linewidth=2.5,
                          markersize=9, color=colors['MOSAIC-2'])
    separated_bittrans = []
    for noc in noc_order:
        noc_data = df[df['NoC_Display'] == noc]
        separated_bittrans.append(noc_data[noc_data['Strategy'] == 'Separated']['BitTransitions'].values[0])

    separated_reduction = [(b - s) / b * 100 for b, s in zip(baseline_bittrans, separated_bittrans)]

    # 在 line1 和 line2 之后添加第三条线
    line3 = ax5_twin.plot(x_positions, separated_reduction, '^-', linewidth=2.5,
                          markersize=9, color=colors['Separated'])


    # Key findings text box
    textstr = '''MOSAIC-2: 28.6% BT reduction and ~5% cycle 
    
    reduction simultaneously'''
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax5.text(0.68, 0.75, textstr, transform=ax5.transAxes, fontsize=8,
             verticalalignment='center', bbox=props, ha='center')

    # 在绘制条形图后添加



    # Labels and formatting
    ax5.set_xlabel('NoC Configuration', fontweight='bold')
    ax5.set_ylabel('Execution Cycles', fontweight='bold', color='black')
    ax5_twin.set_ylabel('Bit Transitions Reduction (%)', fontweight='bold', color='darkgreen')
    ax5.set_title('(e) Performance-Power Trade-off', fontweight='bold', fontsize=11)

    ax5.set_xticks(x_positions)
    ax5.set_xticklabels(noc_order, rotation=15, ha='right')
    ax5.tick_params(axis='y', labelcolor='black')
    ax5_twin.tick_params(axis='y', labelcolor='darkgreen')

    # 获取所有图例元素
    bars_legend = ax5.get_legend_handles_labels()
    lines_legend = ax5_twin.get_legend_handles_labels()

    # 组合柱状图和折线图的图例
    combined_handles = bars_legend[0] + [line1[0], line2[0], line3[0]]
    combined_labels = bars_legend[1] + ['TravelTime Reduction', 'MOSAIC-2 Reduction', 'Separated Reduction']

    # 创建组合图例
    ax5.legend(combined_handles, combined_labels, loc='center right',bbox_to_anchor=(1, 0.58),   fontsize=10, ncol=1)
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