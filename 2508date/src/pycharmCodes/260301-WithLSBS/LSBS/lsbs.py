import matplotlib.pyplot as plt
import numpy as np

categories = ['4-bit', '8-bit', '12-bit']

baseline = [187594617, 187594617, 187594617]
lsbs     = [187130335, 187035312, 187054881]
lsbs_a   = [128168647, 126977206, 126881215]

x = np.arange(len(categories))
width = 0.22

fig, ax = plt.subplots(figsize=(6, 4))

b1 = ax.bar(x - width, baseline, width, label='Baseline', color='#4472C4', edgecolor='black', linewidth=0.5)
b2 = ax.bar(x,         lsbs,     width, label='LSBS',     color='#ED7D31', edgecolor='black', linewidth=0.5)
b3 = ax.bar(x + width, lsbs_a,   width, label='LSBS-A',   color='#70AD47', edgecolor='black', linewidth=0.5)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1e6,
                f'{h/1e6:.1f}M', ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_ylabel('Bit Transitions', fontsize=11)
ax.set_xlabel('LSBS Saturation Bits', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 230e6)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1e6:.0f}M'))
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=10, loc='lower left')
plt.tight_layout()
plt.savefig('lsbs_bt_llm1.pdf', dpi=300, bbox_inches='tight')
plt.savefig('lsbs_bt_llm1.png', dpi=300, bbox_inches='tight')
plt.show()
