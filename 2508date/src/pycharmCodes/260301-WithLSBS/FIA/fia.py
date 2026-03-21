import matplotlib.pyplot as plt
import numpy as np

# Reordered: Default → Fire15(25%) → Fire10(50%) → Fire5(75%)
categories = ['Baseline', '25%', '50%', '75%']

llm1 = [420525, 389231, 354043, 357329]
lenet = [50221, 48267, 47587, 46799]

# --- Figure 1: LLM1 ---
fig1, ax1 = plt.subplots(figsize=(5, 3.5))
bars1 = ax1.bar(categories, llm1, width=0.5, color='#4472C4', edgecolor='black', linewidth=0.5)
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 3000,
             f'{h:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_ylabel('Execution Cycles', fontsize=11)
ax1.set_xlabel('Fire-In-Advance Ratio', fontsize=11)
#ax1.set_title('LLM1 (4×4 Mesh)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 470000)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
fig1.tight_layout()
fig1.savefig('fia_bt_llm1.pdf', dpi=300, bbox_inches='tight')
fig1.savefig('fia_bt_llm1.png', dpi=300, bbox_inches='tight')
# --- Figure 2: LeNet ---
fig2, ax2 = plt.subplots(figsize=(5, 3.5))
bars2 = ax2.bar(categories, lenet, width=0.5, color='#ED7D31', edgecolor='black', linewidth=0.5)
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 400,
             f'{h:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_ylabel('Execution Cycles', fontsize=11)
ax2.set_xlabel('Fire-In-Advance Ratio', fontsize=11)
#ax2.set_title('LeNet (4×4 Mesh)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 56000)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
fig2.tight_layout()
fig2.savefig('fia_bt_lenet.pdf', dpi=300, bbox_inches='tight')
fig2.savefig('fia_bt_lenet.png', dpi=300, bbox_inches='tight')

print("Done!")
