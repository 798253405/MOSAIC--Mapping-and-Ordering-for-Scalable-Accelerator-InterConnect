import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

noc_sizes = ['4×4', '8×8', '16×16', '32×32']
x = np.arange(len(noc_sizes))

# Area data (kGE)
routers = [2008.64, 8034.56, 32138.24, 128552.96]
ordering = [25.82, 103.28, 413.12, 1652.48]

# Power data (mW) - MOSAIC-2 results
orig_nocs = [102.144, 476.672, 2042.88, 8443.904]
after_nocs = [68.738, 321.618, 1408.728, 5806.259]
orig_socc = [33.216, 155.008, 664.32, 2745.856]
after_socc = [22.352, 104.586, 458.101, 1888.125]
ordering_power = [4.426, 17.704, 70.816, 283.264]

# ============ Figure A: Hardware Cost ============
fig1, ax1 = plt.subplots(figsize=(8, 4.5))

width = 0.3
bars1 = ax1.bar(x - width/2, routers, width, label='Router area',
                color='#2E86AB', edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x + width/2, ordering, width, label='Ordering unit area',
                color='#F18F01', edgecolor='black', linewidth=0.5)

for bar in bars1:
    h = bar.get_height()
    txt = f'{h/1000:.1f}k' if h >= 1000 else f'{h:.0f}'
    ax1.text(bar.get_x() + bar.get_width()/2, h, txt,
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2E86AB')

for bar in bars2:
    h = bar.get_height()
    txt = f'{h/1000:.1f}k' if h >= 1000 else f'{h:.1f}'
    ax1.text(bar.get_x() + bar.get_width()/2, h, txt,
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='#C73E1D')

ax1.set_yscale('log')
ax1.set_ylabel('Area (kGE, log scale)', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(noc_sizes, fontsize=11)
ax1.set_xlabel('NoC configuration', fontsize=12)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
ax1.set_ylim(10, 300000)

plt.tight_layout()
plt.savefig('fig14a_hw_cost.pdf', dpi=300, bbox_inches='tight')
plt.close()

# ============ Figure B: Link Power Reduction ============
fig2, ax2 = plt.subplots(figsize=(8, 5))

width = 0.18

b1 = ax2.bar(x - 1.5*width, orig_nocs, width, label='Before MOSAIC (NOCS 2007)',
        color='#5B2C8C', edgecolor='black', linewidth=0.5)
b2 = ax2.bar(x - 0.5*width, after_nocs, width, label='After MOSAIC (NOCS 2007)',
        color='#C4A8E0', edgecolor='black', linewidth=0.5)
b_oh1 = ax2.bar(x - 0.5*width, ordering_power, width, bottom=after_nocs,
        color='#4A7C59', edgecolor='black', linewidth=0.5)
b3 = ax2.bar(x + 0.5*width, orig_socc, width, label='Before MOSAIC (SoCC 2025)',
        color='#B8262C', edgecolor='black', linewidth=0.5)
b4 = ax2.bar(x + 1.5*width, after_socc, width, label='After MOSAIC (SoCC 2025)',
        color='#E8A0A3', edgecolor='black', linewidth=0.5)
ax2.bar(x + 1.5*width, ordering_power, width, bottom=after_socc,
        color='#4A7C59', edgecolor='black', linewidth=0.5)

# Annotation: overhead on top, saved on bottom
for i in range(len(noc_sizes)):
    red_pct = (orig_nocs[i] - after_nocs[i]) / orig_nocs[i] * 100
    oh_pct = ordering_power[i] / orig_nocs[i] * 100
    ax2.text(x[i], orig_nocs[i] * 1.6, f'+{oh_pct:.1f}% overhead\n−{red_pct:.1f}% saved',
             ha='center', va='center', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.85))

handles = [b1, b2, b3, b4, b_oh1]
labels = ['Before MOSAIC (NOCS 2007)', 'After MOSAIC (NOCS 2007)',
          'Before MOSAIC (SoCC 2025)', 'After MOSAIC (SoCC 2025)',
          'Ordering unit overhead']
ax2.legend(handles, labels, fontsize=8.5, loc='upper left', ncol=1)

ax2.set_yscale('log')
ax2.set_ylabel('Power consumption (mW, log scale)', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(noc_sizes, fontsize=11)
ax2.set_xlabel('NoC configuration', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.set_ylim(1, 30000)

plt.tight_layout()
plt.savefig('fig14b_link_power.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Done. fig14a_hw_cost.pdf and fig14b_link_power.pdf saved.")
