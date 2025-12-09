import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 7

# 创建图形 - 3行，每行3个区域
fig = plt.figure(figsize=(18, 12))

# 主配置
noc_configs = ['MC2 6×4', 'MC4 8×8', 'MC8 8×8']
data_types = ['Float-32\nTrained', 'Float-32\nRandom',
              'Fixed-8\nTrained', 'Fixed-8\nRandom']
methods = ['Baseline', 'O1', 'O2']
colors = ['#e67e22', '#f39c12', '#27ae60']  # 橙-黄-绿

# 数据
data_values = [100, 90, 80]

# 用于子图标号
subplot_labels = []
current_label = ord('a')

# 创建3行（对应3个不同的场景或模型）
rows = 3
for row in range(rows):
    # 每行标题
    if row == 0:
        row_title = 'LeNet Model'
    elif row == 1:
        row_title = 'DarkNet Model'
    else:
        row_title = 'VGG Model'

    # 计算当前行的起始位置
    row_base = row * 4

    # 在每行创建3个区域
    for area_idx, noc_config in enumerate(noc_configs):
        # 在每个区域创建4个子图
        for sub_idx, data_type in enumerate(data_types):
            # 计算子图位置 (row, col)
            # 每个区域占4列，所以col = area_idx * 4 + sub_idx
            ax_idx = row * 12 + area_idx * 4 + sub_idx + 1
            ax = plt.subplot(rows, 12, ax_idx)

            # 添加随机扰动使数据看起来更真实
            data = np.array(data_values) * (0.98 + np.random.rand(3) * 0.04)

            x = np.arange(3)
            bars = ax.bar(x, data, color=colors, width=0.7,
                          edgecolor='white', linewidth=0.8, alpha=0.9)

            # 添加数值标签
            for bar, val in zip(bars, data):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{val:.1f}',
                        ha='center', va='bottom', fontsize=6.5,
                        fontweight='bold')

            # 第一行：数据类型标题
            if row == 0:
                ax.set_title(data_type, fontsize=8,
                             fontweight='bold', pad=4)

            # 每个区域的第一个子图：添加NoC配置标签
            if sub_idx == 0:
                # 在左侧添加NoC配置文字
                ax.text(-0.35, 0.5, noc_config,
                        transform=ax.transAxes,
                        fontsize=9, fontweight='bold',
                        rotation=90, va='center', ha='right')

            # 最左侧：添加行标题
            if area_idx == 0 and sub_idx == 0:
                ax.text(-0.65, 0.5, row_title,
                        transform=ax.transAxes,
                        fontsize=11, fontweight='bold',
                        rotation=90, va='center', ha='right',
                        color='darkblue')

            # Y轴标签（每个区域最左边的子图）
            if sub_idx == 0:
                ax.set_ylabel('Bit Trans.\n(×10⁷)', fontsize=7)
            else:
                ax.set_ylabel('')

            # X轴标签
            ax.set_xticks(x)
            ax.set_xticklabels(methods, fontsize=6.5, rotation=0)

            # 设置Y轴范围
            ax.set_ylim(0, 120)

            # 网格
            ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)

            # 美化边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # 添加子图标号
            label = f'({chr(current_label)})'
            ax.text(0.02, 0.98, label,
                    transform=ax.transAxes,
                    fontsize=8, fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.8,
                              edgecolor='gray', linewidth=0.5))
            current_label += 1

# 添加整体图例
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.9,
                         edgecolor='white', linewidth=0.8)
           for i in range(3)]
fig.legend(handles, methods,
           loc='upper center',
           ncol=3,
           frameon=True,
           fontsize=10,
           bbox_to_anchor=(0.5, 0.98),
           columnspacing=1.5,
           handlelength=1.5)

# 添加区域分隔线
for row in range(rows):
    for sep in [4, 8]:  # 在第4和第8列后面画分隔线
        line_x = sep / 12
        fig.add_artist(plt.Line2D([line_x, line_x],
                                  [0.05 + row / 3, 0.05 + (row + 1) / 3 - 0.02],
                                  transform=fig.transFigure,
                                  color='gray', linewidth=1.5,
                                  linestyle='--', alpha=0.5))

plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.03,
                    hspace=0.4, wspace=0.5)

plt.savefig('benchmark_style_layout.pdf', dpi=300, bbox_inches='tight')
plt.savefig('benchmark_style_layout.png', dpi=300, bbox_inches='tight')
print("✓ 图片已保存")
plt.show()