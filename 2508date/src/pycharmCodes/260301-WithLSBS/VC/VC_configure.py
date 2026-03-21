import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
})

# Data
vc_labels = ['2', '4', '8']
baseline_cycles = [393272, 420525, 403752]
mosaic_cycles   = [386158, 328296, 322184]
baseline_bt     = [187594845, 187594617, 187547093]
mosaic_bt       = [133369742, 130266197, 129676204]

# Compute reductions (%)
cycle_red = [(b - m) / b * 100 for b, m in zip(baseline_cycles, mosaic_cycles)]
bt_red    = [(b - m) / b * 100 for b, m in zip(baseline_bt, mosaic_bt)]

x = np.arange(len(vc_labels))
w = 0.32

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.4))

# --- Left: Total Cycles ---
bars1 = ax1.bar(x - w/2, [c/1e3 for c in baseline_cycles], w, label='Baseline',
                color='#9e9e9e', edgecolor='#666666', linewidth=0.5)
bars2 = ax1.bar(x + w/2, [c/1e3 for c in mosaic_cycles], w, label='MOSAIC',
                color='#2e7d32', edgecolor='#1b5e20', linewidth=0.5)

# Add reduction % labels on top of MOSAIC bars
for i, (bar, red) in enumerate(zip(bars2, cycle_red)):
    ax1.annotate(f'−{red:.1f}%',
                 xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 4), textcoords='offset points',
                 ha='center', va='bottom', fontsize=7, fontweight='bold',
                 color='#2e7d32')

ax1.set_xlabel('VCs per virtual network')
ax1.set_ylabel('Total cycles (×10³)')
ax1.set_xticks(x)
ax1.set_xticklabels(vc_labels)
ax1.set_ylim(0, 480)
ax1.legend(loc='upper right', frameon=True, edgecolor='#cccccc', fancybox=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('(a) Latency', fontsize=10, pad=6)

# --- Right: Bit Transitions ---
bars3 = ax2.bar(x - w/2, [b/1e6 for b in baseline_bt], w, label='Baseline',
                color='#9e9e9e', edgecolor='#666666', linewidth=0.5)
bars4 = ax2.bar(x + w/2, [b/1e6 for b in mosaic_bt], w, label='MOSAIC',
                color='#1565c0', edgecolor='#0d47a1', linewidth=0.5)

for i, (bar, red) in enumerate(zip(bars4, bt_red)):
    ax2.annotate(f'−{red:.1f}%',
                 xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 4), textcoords='offset points',
                 ha='center', va='bottom', fontsize=7, fontweight='bold',
                 color='#1565c0')

ax2.set_xlabel('VCs per virtual network')
ax2.set_ylabel('Bit transitions (×10⁶)')
ax2.set_xticks(x)
ax2.set_xticklabels(vc_labels)
ax2.set_ylim(0, 220)
ax2.legend(loc='upper right', frameon=True, edgecolor='#cccccc', fancybox=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('(b) Bit transitions', fontsize=10, pad=6)

plt.tight_layout(w_pad=2.5)
plt.savefig('vc_sensitivity.pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('vc_sensitivity.png', bbox_inches='tight', pad_inches=0.05)
print("Done. Saved to current directory.")
print(f"\nCycle reduction: {[f'{r:.1f}%' for r in cycle_red]}")
print(f"BT reduction:    {[f'{r:.1f}%' for r in bt_red]}")
