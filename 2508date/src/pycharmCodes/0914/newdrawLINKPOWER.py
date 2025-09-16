import matplotlib.pyplot as plt
import numpy as np

# NoC sizes
noc_sizes = ['4×4', '8×8', '16×16', '32×32']

# 功耗数据 (mW)
data = {
    'Original link power NOCS 2007': [102.144, 476.672, 2042.88, 8443.904],
    'Link power NOCS 2007 after MOSAIC-2': [68.738, 321.618, 1408.728, 5806.259],
    'Original link power SoCC 2025': [33.216, 155.008, 664.32, 2745.856],
    'Link power SoCC 2025 after MOSAIC-2': [22.352, 104.586, 458.101, 1888.125],
    'Four-ordering': [8.852/4*2, 8.852/4*8, 8.852/4*32, 8.852/4*128]
}

# 面积数据 (kGE)
area_data = {
    'Routers': [2008.64, 8034.56, 32138.24, 128552.96],
    'Ordering units': [51.64/4*2, 51.64/4*8, 51.64/4*32, 51.64/4*128]
}

# 创建图形和主轴
fig, ax1 = plt.subplots(figsize=(16, 7))

# ====== 柱状图（左Y轴）======
x = np.arange(len(noc_sizes))
width = 0.12
colors_bar = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4A7C59']

# 绘制柱状图
for i, (label, values) in enumerate(data.items()):
    offset = (i - 2) * width
    bars = ax1.bar(x + offset, values, width, label=label, color=colors_bar[i],
                   edgecolor='black', linewidth=0.5, alpha=0.8)

    # 显示Original link power NOCS 2007的数值
    # 显示Original link power NOCS 2007的数值
    if label == 'Original link power NOCS 2007':
        for j, bar in enumerate(bars):
            height = bar.get_height()
            label_text = f'{height:.2f}'
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 100,
                     label_text, ha='center', va='bottom', fontsize=20,
                     weight='bold', color='#2E86AB')

    # 显示Four-ordering的数值
    elif label == 'Four-ordering':
        for j, bar in enumerate(bars):
            height = bar.get_height()
            # 由于值很小，需要在较高位置显示，并用引线连接
            y_pos = 160  # 固定高度显示
            bar_x = bar.get_x() + bar.get_width() / 2.

            # 添加引线
            ax1.plot([bar_x, bar_x], [height, y_pos - 50],
                     'k-', alpha=0.3, linewidth=1)

            # 显示数值
            if j == 1:  # 第一个显示完整标签
                label_text = f'{height:.2f} '
            else:
                label_text = f'{height:.2f}'

            ax1.text(bar_x, y_pos, label_text,
                     ha='center', va='bottom', fontsize=20,
                     weight='bold', color='#4A7C59')

# 设置左Y轴

ax1.set_ylabel('Power Consumption (mW)', fontsize=20, fontweight='bold', color='black')
ax1.set_ylim(0, max(data['Original link power NOCS 2007'][3],
                    data['Link power NOCS 2007 after MOSAIC-2'][3]) * 1.15)
ax1.tick_params(axis='y', labelcolor='black', labelsize=16)
ax1.set_xticks(x)
ax1.set_xticklabels(noc_sizes, fontsize=20)

# ====== 折线图（右Y轴）======
ax2 = ax1.twinx()

# 绘制ref2面积折线
line1 = ax2.plot(x, area_data['Routers'], 'o-', color='red',
                 linewidth=3, markersize=12, label='Area of routers',
                 markeredgecolor='darkred', markeredgewidth=2)

# 绘制socc4 units面积折线
line2 = ax2.plot(x, area_data['Ordering units'], 's-', color='blue',
                 linewidth=3, markersize=12, label='Area of ordering units',
                 markeredgecolor='red', markeredgewidth=2)

# 在折线上添加数值标签
for i, (x_val, y_val) in enumerate(zip(x, area_data['Routers'])):
    if y_val > 10000:
        label_text = f'{y_val / 1000:.2f}k'
    else:
        label_text = f'{y_val:.2f}'
    ax2.text(x_val, y_val + y_val * 0.05, label_text,
             ha='center', va='bottom', fontsize=20,
             fontweight='bold', color='red')

 

# 设置右Y轴 - 从负值开始以显示底部的线
ax2.set_ylabel('Area (kGE)', fontsize=20, fontweight='bold', color='black')
ax2.set_ylim(-10000, max(area_data['Routers']) * 1.15)
ax2.tick_params(axis='y', labelcolor='black', labelsize=16)

# 在Y轴0处添加水平参考线
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)



# 添加网格
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# 合并图例
bars_legend = ax1.get_legend_handles_labels()
lines_legend = ax2.get_legend_handles_labels()

ax1.legend(bars_legend[0] + lines_legend[0],
           bars_legend[1] + lines_legend[1],
           loc='upper left', fontsize=16, framealpha=0.95, ncol=2)

# 美化Y轴刻度
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}' if x >= 0 else ''))

plt.tight_layout()
plt.show()