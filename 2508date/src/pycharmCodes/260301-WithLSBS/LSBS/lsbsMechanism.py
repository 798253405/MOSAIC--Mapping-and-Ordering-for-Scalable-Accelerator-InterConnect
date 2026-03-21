import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(8, 1.8))
ax.axis('off')
ax.set_xlim(0, 6.9)
ax.set_ylim(0.55, 4.5)

# Colors
c_sign  = '#9B59B6'
c_exp   = '#4472C4'
c_mant  = '#70AD47'
c_lbit  = '#E03030'

BW = 0.42
BH = 0.55
GAP = 0.03
DOTS_W = 0.5

def draw_box(ax, x, w, y, h, color, label, fontsize=15, textcolor='white', hatch=None):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black', linewidth=0.6)
    ax.add_patch(rect)
    if hatch:
        rect_h = plt.Rectangle((x, y), w, h, facecolor='none', edgecolor='white',
                                linewidth=0, hatch=hatch, alpha=0.5)
        ax.add_patch(rect_h)
    if label:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=textcolor)

def draw_bit(ax, x, y, val, color, hatch=None):
    draw_box(ax, x, BW, y, BH, color, str(val), fontsize=15, hatch=hatch)

def draw_dots(ax, x, y):
    ax.text(x, y + BH/2, '...', ha='center', va='center',
            fontsize=20, fontweight='bold', color='black')

SIGN_W = BW
EXP_W = 1.8
UPPER_W = BW * 4 + GAP * 3

lbit_val = 1
lsb_left  = [0, 1]
lsb_right = [0, 0]
lsb_sat_left  = [1, 1]
lsb_sat_right = [1, 1]

# ============================================================
# Row 1: Default
# ============================================================
y1 = 3.0
x = 0

draw_box(ax, x, SIGN_W, y1, BH, c_sign, 'S', 15); x += SIGN_W + GAP
draw_box(ax, x, EXP_W, y1, BH, c_exp, '8-bit Exp', 14); x += EXP_W + GAP
draw_box(ax, x, UPPER_W, y1, BH, c_mant, '...', 18); x += UPPER_W + GAP

lbit_x = x
draw_bit(ax, x, y1, lbit_val, c_mant); x += BW + GAP
highlight = plt.Rectangle((lbit_x - 0.02, y1 - 0.02), BW + 0.04, BH + 0.04,
                            facecolor='none', edgecolor=c_lbit, linewidth=2.5, zorder=5)
ax.add_patch(highlight)

lsb_start_x = x
for b in lsb_left:
    draw_bit(ax, x, y1, b, c_mant); x += BW + GAP
draw_dots(ax, x + 0.15, y1); x += DOTS_W
for b in lsb_right:
    draw_bit(ax, x, y1, b, c_mant); x += BW + GAP
lsb_end_x = x - GAP

ax.annotate('bit l+1', xy=(lbit_x + BW/2, y1 + BH + 0.03),
            xytext=(lbit_x + BW/2, y1 + BH + 0.5),
            ha='center', fontsize=15, fontweight='bold', color=c_lbit,
            arrowprops=dict(arrowstyle='->', color=c_lbit, lw=1.5))

ax.annotate('', xy=(lsb_start_x, y1 - 0.08), xytext=(lsb_end_x, y1 - 0.08),
            arrowprops=dict(arrowstyle='<->', color=c_lbit, lw=1.5))
ax.text((lsb_start_x + lsb_end_x)/2, y1 - 0.3, 'l bits',
        ha='center', fontsize=15, fontweight='bold', color=c_lbit)

row1_end = x

# ============================================================
# Arrow + LSBS label
# ============================================================
mid_x = row1_end / 2
ax.annotate('', xy=(mid_x, 1.85), xytext=(mid_x, 2.35),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(mid_x - 0.6, 2.1, 'LSBS', ha='right', va='center', fontsize=17, fontweight='bold', color='black')
ax.text(mid_x + 0.8, 2.1, 'bit l+1 = 1 \u2192 l bits all 1\nbit l+1 = 0 \u2192 l bits all 0',
        ha='left', va='center', fontsize=14,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF2CC', edgecolor='gray', lw=0.5))

# ============================================================
# Row 2: LSBS
# ============================================================
y2 = 0.8
x = 0

draw_box(ax, x, SIGN_W, y2, BH, c_sign, 'S', 15); x += SIGN_W + GAP
draw_box(ax, x, EXP_W, y2, BH, c_exp, '8-bit Exp', 14); x += EXP_W + GAP
draw_box(ax, x, UPPER_W, y2, BH, c_mant, '...', 18); x += UPPER_W + GAP

lbit_x2 = x
draw_bit(ax, x, y2, lbit_val, c_mant); x += BW + GAP
highlight2 = plt.Rectangle((lbit_x2 - 0.02, y2 - 0.02), BW + 0.04, BH + 0.04,
                             facecolor='none', edgecolor=c_lbit, linewidth=2.5, zorder=5)
ax.add_patch(highlight2)

for b in lsb_sat_left:
    draw_bit(ax, x, y2, b, c_mant, hatch='//'); x += BW + GAP
draw_dots(ax, x + 0.15, y2); x += DOTS_W
for b in lsb_sat_right:
    draw_bit(ax, x, y2, b, c_mant, hatch='//'); x += BW + GAP

plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
plt.savefig('lsbs_mechanism.pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.savefig('lsbs_mechanism.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.show()
