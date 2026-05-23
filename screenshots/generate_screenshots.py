"""
Generate realistic Streamlit dashboard screenshot mockups using matplotlib.
Saved as PNG for embedding in README.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

Path("screenshots").mkdir(exist_ok=True)

BG      = "#0D1B2A"
CARD    = "#0A2540"
BORDER  = "#1E3A52"
WHITE   = "#FFFFFF"
MUTED   = "#7B99B5"
ACCENT  = "#5DCAA5"
GOLD    = "#F5A623"
BLUE    = "#5BB8F5"
RED     = "#E94560"
NAVY    = "#0F3460"

def styled_fig(w=14, h=8):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    return fig

def card(ax, x, y, w, h, title, value, subtitle="", color=ACCENT, delta=None):
    box = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.02",
                         facecolor=CARD, edgecolor=BORDER, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x+w/2, y+h*0.72, title, ha="center", va="center",
            color=MUTED, fontsize=8, fontweight="normal")
    ax.text(x+w/2, y+h*0.42, value, ha="center", va="center",
            color=color, fontsize=16, fontweight="bold")
    if subtitle:
        ax.text(x+w/2, y+h*0.18, subtitle, ha="center", va="center",
                color=MUTED, fontsize=7)

# ── SCREENSHOT 1: Main dashboard ─────────────────────────────────
fig = styled_fig(14, 9)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 14); ax.set_ylim(0, 9)
ax.axis("off"); ax.set_facecolor(BG)

# Top bar
bar = FancyBboxPatch((0,8.3), 14, 0.7, boxstyle="square",
                     facecolor=NAVY, edgecolor="none")
ax.add_patch(bar)
ax.text(0.3, 8.65, "📊  Credit Default Model — Monitoring Dashboard", color=WHITE,
        fontsize=13, fontweight="bold", va="center")
ax.text(13.7, 8.65, "May 23, 2026  |  Last run: 09:00 IST", color=MUTED,
        fontsize=8, va="center", ha="right")

# KPI cards row
cards_data = [
    (0.3, 7.0, 3.1, 1.0, "DATASET DRIFT", "🔴  DETECTED", "5 / 6 features", RED),
    (3.7, 7.0, 3.1, 1.0, "DRIFTED FEATURES", "5 / 6", "83% of feature set", GOLD),
    (7.1, 7.0, 3.1, 1.0, "ESTIMATED ROC-AUC", "0.9828", "NannyML CBPE", ACCENT),
    (10.5,7.0, 3.1, 1.0, "ALERT TRIGGERED", "✅  None", "Within threshold", ACCENT),
]
for cx, cy, cw, ch, title, val, sub, col in cards_data:
    card(ax, cx, cy, cw, ch, title, val, sub, col)

# Section label
ax.text(0.3, 6.75, "Feature Drift Status  (Evidently AI — DataDriftPreset)", color=BLUE,
        fontsize=9, fontweight="bold")

# Feature drift chips
feats = [("age","🔴 DRIFTED",RED), ("limit_bal","🔴 DRIFTED",RED),
         ("pay_delay","🔴 DRIFTED",RED), ("bill_amt","🔴 DRIFTED",RED),
         ("pay_amt","🔴 DRIFTED",RED), ("education","✅ Stable",ACCENT)]
for i, (feat, status, col) in enumerate(feats):
    cx = 0.3 + (i % 3) * 4.5
    cy = 6.25 - (i // 3) * 0.65
    chip = FancyBboxPatch((cx, cy), 4.0, 0.52, boxstyle="round,pad=0.02",
                          facecolor=CARD, edgecolor=col, linewidth=1.2)
    ax.add_patch(chip)
    ax.text(cx+0.18, cy+0.26, feat, color=WHITE, fontsize=9,
            fontweight="bold", va="center")
    ax.text(cx+3.82, cy+0.26, status, color=col, fontsize=8,
            fontweight="bold", va="center", ha="right")

# NannyML AUC chart
ax.text(0.3, 4.8, "Estimated Performance Over Time  (NannyML CBPE — no labels needed)",
        color=BLUE, fontsize=9, fontweight="bold")
ax_nml = fig.add_axes([0.022, 0.27, 0.57, 0.24], facecolor=CARD)
chunks = range(5)
auc_vals = [0.984, 0.981, 0.983, 0.982, 0.983]
threshold = [0.947] * 5
ax_nml.plot(chunks, auc_vals, "o-", color=BLUE, linewidth=2.5, markersize=6, label="Est. ROC-AUC")
ax_nml.plot(chunks, threshold, "--", color=RED, linewidth=1.5, label="Alert threshold (0.947)")
ax_nml.fill_between(chunks, threshold, auc_vals, alpha=0.1, color=BLUE)
ax_nml.set_facecolor(CARD); ax_nml.set_xlim(-0.3, 4.3); ax_nml.set_ylim(0.88, 1.01)
for spine in ax_nml.spines.values(): spine.set_edgecolor(BORDER)
ax_nml.tick_params(colors=MUTED, labelsize=8)
ax_nml.set_xlabel("Production Chunk", color=MUTED, fontsize=8)
ax_nml.set_ylabel("Est. ROC-AUC", color=MUTED, fontsize=8)
ax_nml.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=WHITE)

# Feature distributions (right panel)
ax.text(8.7, 4.8, "Feature Distributions — Reference vs Production",
        color=BLUE, fontsize=9, fontweight="bold")
for i, (feat, shift) in enumerate([("pay_delay", 2.5), ("bill_amt", 25000), ("age", 12)]):
    axi = fig.add_axes([0.625 + i*0.125, 0.27, 0.115, 0.24], facecolor=CARD)
    np.random.seed(i)
    ref_vals = np.random.normal(3, 2, 300)
    prod_vals = np.random.normal(3 + shift * 0.3, 2.2, 300)
    axi.hist(ref_vals, bins=18, alpha=0.65, color=NAVY, label="Ref")
    axi.hist(prod_vals, bins=18, alpha=0.65, color=RED, label="Prod")
    axi.set_facecolor(CARD); axi.set_title(feat, color=WHITE, fontsize=7, pad=3)
    for spine in axi.spines.values(): spine.set_edgecolor(BORDER)
    axi.tick_params(colors=MUTED, labelsize=6)
    if i == 0: axi.legend(fontsize=6, facecolor=CARD, edgecolor=BORDER, labelcolor=WHITE)

# Footer
ax.text(0.3, 0.15, "Built with Evidently AI + NannyML  ·  FastAPI + Streamlit  ·  GCP Cloud Run",
        color=MUTED, fontsize=8)

plt.savefig("screenshots/dashboard_main.png", dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close()
print("Saved: screenshots/dashboard_main.png")

# ── SCREENSHOT 2: Evidently HTML report ──────────────────────────
fig = styled_fig(14, 8)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 14); ax.set_ylim(0, 8)
ax.axis("off"); ax.set_facecolor(BG)

bar = FancyBboxPatch((0,7.35), 14, 0.65, boxstyle="square", facecolor=NAVY, edgecolor="none")
ax.add_patch(bar)
ax.text(0.3, 7.67, "📋  Evidently AI — Data Drift Report  (drift_report.html)", color=WHITE,
        fontsize=12, fontweight="bold", va="center")

# Summary box
summ = FancyBboxPatch((0.3, 6.4), 13.4, 0.75, boxstyle="round,pad=0.03",
                      facecolor=CARD, edgecolor=RED, linewidth=2)
ax.add_patch(summ)
ax.text(0.6, 6.78, "⚠️  Dataset Drift Detected", color=RED, fontsize=11, fontweight="bold")
ax.text(5.5, 6.78, "Drifted columns: 5 / 6   (83%)", color=WHITE, fontsize=10)
ax.text(10.5, 6.78, "Share: 0.833", color=GOLD, fontsize=10, fontweight="bold")

# Column drift table
cols_data = [
    ("Feature", "Type", "Stat Test", "Drift Score", "Drifted?", WHITE, True),
    ("age", "num", "KS", "0.312", "🔴 Yes", RED, False),
    ("limit_bal","num","KS","0.287","🔴 Yes", RED, False),
    ("pay_delay","num","KS","0.445","🔴 Yes", RED, False),
    ("bill_amt", "num","KS","0.298","🔴 Yes", RED, False),
    ("pay_amt",  "num","KS","0.271","🔴 Yes", RED, False),
    ("education","cat","Chi²","0.031","✅ No",  ACCENT, False),
]
col_x = [0.3, 3.2, 5.8, 8.0, 10.2, 11.8]
for row_i, row in enumerate(cols_data):
    y_r = 6.15 - row_i * 0.52
    bg_row = "#112233" if row_i % 2 == 0 and not row[6] else CARD
    if not row[6]:
        bg_patch = FancyBboxPatch((0.28, y_r-0.1), 13.44, 0.47,
                                  boxstyle="square", facecolor=bg_row, edgecolor="none")
        ax.add_patch(bg_patch)
    for ci, (val, xp) in enumerate(zip(row[:6], col_x)):
        col_c = row[5] if ci == 4 else (WHITE if row[6] else MUTED if ci in [1,2] else WHITE)
        fw = "bold" if row[6] or ci == 0 else "normal"
        ax.text(xp, y_r+0.13, str(val), color=col_c, fontsize=9, fontweight=fw)

plt.savefig("screenshots/evidently_report.png", dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close()
print("Saved: screenshots/evidently_report.png")
