import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(2, 1, figsize=(8, 1.6), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05})

# Colors
c_req   = '#4472C4'
c_resp  = '#ED7D31'
c_comp  = '#70AD47'
c_mc    = '#9B59B6'
c_overlap = '#A8D5A0'
c_fia   = '#E03030'

def draw_block(ax, x, w, y, h, color, label, fontsize=6, textcolor='white', alpha=1.0):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black', linewidth=0.6, alpha=alpha)
    ax.add_patch(rect)
    if label:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=textcolor)

H = 0.5
Y = 0.15
REQ_W = 1.0
MC_W  = 0.6
RESP_W = 1.0
COMP_W = 2.2

ADVANCE_RATIO = 0.75
EXEC_RATIO = 1 - ADVANCE_RATIO

# ============================================================
# (a) Default
# ============================================================
ax1 = axes[0]
ax1.set_xlim(-2.5, 12.5)
ax1.set_ylim(-0.05, 0.85)
ax1.axis('off')

x = 0
for i in range(2):
    draw_block(ax1, x, REQ_W, Y, H, c_req, f'Req{i}', 7); x += REQ_W
    draw_block(ax1, x, MC_W, Y, H, c_mc, 'MA', 7); x += MC_W
    draw_block(ax1, x, RESP_W, Y, H, c_resp, f'Resp{i}', 7); x += RESP_W
    draw_block(ax1, x, COMP_W, Y, H, c_comp, f'Compute{i}', 7); x += COMP_W

# Task 2: partial (only Req2 + ...)
draw_block(ax1, x, REQ_W, Y, H, c_req, 'Req2', 7); x += REQ_W
ax1.text(x + 0.3, Y + H/2, '...', ha='center', va='center', fontsize=10, fontweight='bold', color='black')

ax1.annotate('', xy=(12.2, 0.05), xytext=(0, 0.05),
             arrowprops=dict(arrowstyle='->', color='black', lw=0.7))

# ============================================================
# (b) Fire-In-Advance
# ============================================================
ax2 = axes[1]
ax2.set_xlim(-2.5, 12.5)
ax2.set_ylim(-0.15, 1.1)
ax2.axis('off')

comp_exec = COMP_W * EXEC_RATIO
comp_adv  = COMP_W * ADVANCE_RATIO

fia_xs = []

x = 0
draw_block(ax2, x, REQ_W, Y, H, c_req, 'Req0', 7); x += REQ_W
draw_block(ax2, x, MC_W, Y, H, c_mc, 'MA', 7); x += MC_W
draw_block(ax2, x, RESP_W, Y, H, c_resp, 'Resp0', 7); x += RESP_W

draw_block(ax2, x, comp_exec, Y, H, c_comp, 'Comp', 6); x += comp_exec
fia_xs.append(x)
draw_block(ax2, x, comp_adv, Y + H + 0.02, H * 0.5, c_overlap, 'Overlapped Comp0', 6, 'black', 0.7)

draw_block(ax2, x, REQ_W, Y, H, c_req, 'Req1', 7); x += REQ_W
draw_block(ax2, x, MC_W, Y, H, c_mc, 'MA', 7); x += MC_W
draw_block(ax2, x, RESP_W, Y, H, c_resp, 'Resp1', 7); x += RESP_W

draw_block(ax2, x, comp_exec, Y, H, c_comp, 'Comp', 6); x += comp_exec
fia_xs.append(x)
draw_block(ax2, x, comp_adv, Y + H + 0.02, H * 0.5, c_overlap, 'Overlapped Comp1', 6, 'black', 0.7)

# Task 2: partial
draw_block(ax2, x, REQ_W, Y, H, c_req, 'Req2', 7); x += REQ_W
ax2.text(x + 0.3, Y + H/2, '...', ha='center', va='center', fontsize=10, fontweight='bold', color='black')

# Red dashed FIA lines + text
for i, fx in enumerate(fia_xs):
    ax2.plot([fx, fx], [Y - 0.1, Y + H + 0.02 + H*0.5 + 0.05],
             ls='--', color=c_fia, lw=1.2, alpha=0.8)
    ax2.text(fx, Y - 0.12, 'Fire-In-Advance', ha='center', va='top',
             fontsize=5.5, fontweight='bold', color=c_fia)

ax2.annotate('', xy=(12.2, 0.05), xytext=(0, 0.05),
             arrowprops=dict(arrowstyle='->', color='black', lw=0.7))

# Legend
legend_items = [
    mpatches.Patch(facecolor=c_req, edgecolor='black', label='Req Travel'),
    mpatches.Patch(facecolor=c_mc, edgecolor='black', label='Memory Access'),
    mpatches.Patch(facecolor=c_resp, edgecolor='black', label='Resp Travel'),
    mpatches.Patch(facecolor=c_comp, edgecolor='black', label='PE Compute'),
    mpatches.Patch(facecolor=c_overlap, edgecolor='black', label='Overlapped Compute'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=5, fontsize=6.5,
           frameon=True, edgecolor='gray', handlelength=1.2, handletextpad=0.3,
           columnspacing=1.0, borderpad=0.3)

plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.18)
plt.savefig('fia_principle.pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.savefig('fia_principle.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.show()
