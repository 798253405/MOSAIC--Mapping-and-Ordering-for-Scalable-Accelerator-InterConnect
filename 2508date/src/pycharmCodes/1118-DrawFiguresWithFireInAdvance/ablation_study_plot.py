import matplotlib.pyplot as plt
import numpy as np
import os

# 配置方法名称和对应的数据 (BT Reduction, Cycle Reduction)
# 请填入你的实际数据
methods_data = {
    'Baseline': (0, 0),  # 通常baseline作为参考点设置为(0, 0)
    'TravelTime': (4.800590842, 0.7427558971),
    'Affiliated': (0, 29.59760809),
    'Separated': (0, 31.47287419),
    'Combo-1': (4.800590842, 30.10575806),
    'Combo-2': (4.800590842, 31.97097056),
    'FireInAdvance': (14.86706056, 0.01624343422),
    'RoutingSwitch': (0.9970457903, 0.00550084975),
    'MOSAIC-1': (21.17614476, 30.01322457),
    'MOSAIC-2': (21.17614476, 31.88007482)
}

# 提取数据
method_names = list(methods_data.keys())
bt_reductions = [methods_data[m][0] for m in method_names]
cycle_reductions = [methods_data[m][1] for m in method_names]

# 创建图形
plt.figure(figsize=(12, 8))

# 为不同的方法类别使用不同的颜色和标记
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

# 绘制散点图
for i, method in enumerate(method_names):
    x, y = methods_data[method]
    plt.scatter(x, y, c=colors[i], marker=markers[i], s=800,
                label=method, alpha=0.7, edgecolors='black', linewidth=1.5)
    # 添加标签，调整位置避免重合，字体调大
    if method == 'RoutingSwitch':
        plt.annotate(method, (x, y), xytext=(-60, 15), textcoords='offset points',
                    fontsize=13, fontweight='bold')
    elif method == 'TravelTime':
        plt.annotate(method, (x, y), xytext=(5, 15), textcoords='offset points',
                    fontsize=13, fontweight='bold')
    elif method == 'Baseline':
        plt.annotate(method, (x, y), xytext=(-80, 5), textcoords='offset points',
                    fontsize=13, fontweight='bold')
    elif method == 'MOSAIC-1':
        plt.annotate(method, (x, y), xytext=(5, -12), textcoords='offset points',
                    fontsize=13, fontweight='bold')
    elif method == 'MOSAIC-2':
        plt.annotate(method, (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=13, fontweight='bold')
    elif method == 'Combo-1':
        plt.annotate(method, (x, y), xytext=(10, 20), textcoords='offset points',
                    fontsize=13, fontweight='bold')
    elif method == 'Combo-2':
        plt.annotate(method, (x, y), xytext=(10, -20), textcoords='offset points',
                    fontsize=13, fontweight='bold')
    elif method == 'Affiliated':
        plt.annotate(method, (x, y), xytext=(-80, 5), textcoords='offset points',
                    fontsize=13, fontweight='bold')
    elif method == 'Separated':
        plt.annotate(method, (x, y), xytext=(5, 10), textcoords='offset points',
                    fontsize=13, fontweight='bold')
    else:
        plt.annotate(method, (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=13, fontweight='bold')

# 添加参考线（baseline）
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# 设置标签和标题
plt.xlabel('Cycle Reduction (%)', fontsize=14, fontweight='bold')
plt.ylabel('BT Reduction (%)', fontsize=14, fontweight='bold')

# 设置固定的坐标轴范围，让数据点分布更均匀
plt.xlim(-5,27)  # Cycle Reduction 范围
plt.ylim(-5, 40)  # BT Reduction 范围

# 添加网格
plt.grid(True, alpha=0.3, linestyle='--')

# 添加图例，增加间距和大小避免重叠，使用2列布局
plt.legend(loc='right', fontsize=11, framealpha=0.9,
           ncol=2,            # 2列布局
           markerscale=1.2,  # 图例中标记的大小
           labelspacing=3.0,  # 增加标签之间的垂直间距 (1.2 * 2.5)
           columnspacing=1.5, # 列之间的间距
           borderpad=1.0,     # 图例边框内边距
           handletextpad=0.8) # 图例标记和文字之间的间距

# 调整布局
plt.tight_layout()

# 获取脚本所在目录，保存到同一目录下
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(script_dir, 'ablation_study.pdf')
png_path = os.path.join(script_dir, 'ablation_study.png')

# 保存图形
plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.savefig(png_path, dpi=300, bbox_inches='tight')

print(f"图形已保存为:\n{pdf_path}\n{png_path}")

# 显示图形（如果在支持的环境中）
# plt.show()