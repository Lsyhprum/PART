#!/usr/bin/env python3
"""Dual-axis comparison charts: read ac_sweep_results.json and generate, for both
the C-sweep and A-sweep, a chart with "QPS@0.9 bars (left y-axis) + Indexing time
line (right y-axis)"; also produce a 1x2 combined figure (left=C-sweep, right=A-sweep)
at the end. This script only reads the JSON, never reruns experiments, and never
modifies any existing script, image, or categorical C++ code.
Files: ac_sweep_c_dual.png / ac_sweep_a_dual.png (kept) + ac_sweep_dual_combined.png (new)."""
import json
import os
import glob as _glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = "./_log"
JSON_PATH = os.path.join(LOG_DIR, "ac_sweep_results.json")

# ====================== Tunable parameters (tweak here) ======================
FIGSIZE_SINGLE   = (9, 5)     # single-figure size (width, height)
FIGSIZE_COMBINED = (16, 5)    # combined figure (1x2) size
DPI = 140                     # output resolution
FS_TITLE  = 36                # subplot title font size
FS_AXIS   = 36                # axis label font size
FS_TICK   = 22                # tick label font size (shared by x/y axes)
FS_ANNOT  = 20                # bar-top / line annotation font size
FS_LEGEND = 26                # legend font size
FS_SCI   = 18                # scientific-notation exponent (x10^n) font size
LW = 4                      # line width
MS = 20                        # line marker size
BAR_WIDTH = 0.55              # bar width
COLOR_QPS       = "#2c7fb8"   # bar (QPS) default color
COLOR_IDX       = "#000000"   # line (Indexing) color (black)
# C-sweep / A-sweep bar styling (color + hatch)
BAR_COLOR_C  = "#2c7fb8"   # C-sweep bar color (blue)
BAR_COLOR_A  = "#2ca25f"   # A-sweep bar color (green, avoids orange line)
BAR_HATCH_C  = "x"         # C-sweep bar hatch (diagonal cross)
BAR_HATCH_A  = "+"         # A-sweep bar hatch (orthogonal cross)

# ---- Font (Times): register .ttf files from the directory and set as global default ----
FONT_DIR = os.environ.get("AC_SWEEP_FONT_DIR", "")  # empty -> use matplotlib default font
def _register_fonts():
    from matplotlib import font_manager as _fm
    if not os.path.isdir(FONT_DIR):
        return
    ttfs = sorted(_glob.glob(os.path.join(FONT_DIR, "*.ttf")))
    for f in ttfs:
        try:
            _fm.fontManager.addfont(f)
        except Exception:
            pass
    if ttfs:
        try:
            name = _fm.FontProperties(fname=ttfs[0]).get_name()
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["font.serif"] = [name]
        except Exception:
            pass
_register_fonts()


def plot_dualaxis(ax1, ax2, cfgs, xlabels, title, bar_color=COLOR_QPS, bar_hatch="", xlabel=""):
    """Draw on the given ax1 (left axis, QPS bars) / ax2 (right axis, Indexing line); does not create a figure or call savefig."""
    x = list(range(len(cfgs)))
    qps = [d["qps90"] for d in cfgs]
    idx = [d["index_s"] for d in cfgs]

    bars = ax1.bar(x, qps, color=bar_color, edgecolor="black", linewidth=0.6,
                   width=BAR_WIDTH, hatch=bar_hatch, label="QPS")
    line, = ax2.plot(x, idx, marker="s", color=COLOR_IDX, linewidth=LW,
                     markersize=MS, label="Index time")

    # Axis labels and ranges
    ax1.set_ylabel("QPS", fontsize=FS_AXIS)
    ax2.set_ylabel("Index time (s)", fontsize=FS_AXIS)
    ax1.tick_params(axis="y", labelsize=FS_TICK)
    ax2.tick_params(axis="y", labelsize=FS_TICK)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax1.yaxis.get_offset_text().set_fontsize(FS_SCI)
    ax2.yaxis.get_offset_text().set_fontsize(FS_SCI)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=FS_TICK)
    ax1.set_ylim(0, max(qps) * 1.18)
    if min(idx) > 0:
        ax2.set_ylim(0, max(idx) * 1.25)
    else:
        ax2.set_ylim(min(idx) * 1.2, max(idx) * 1.25)

    # Bar-top value annotation (white box to keep it readable over the bars)
    bbox = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
    for b, v in zip(bars, qps):
        ax1.text(b.get_x() + b.get_width() / 2, v + max(qps) * 0.015,
                 f"{v:,.0f}", ha="center", va="bottom", fontsize=FS_ANNOT, bbox=bbox)

    ax1.grid(axis="y", alpha=0.3)
    if title:
        ax1.set_title(title, fontsize=FS_TITLE)
    ax1.set_xlabel(xlabel, fontsize=FS_AXIS)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=FS_LEGEND, framealpha=1.0)


def plot_single(cfgs, xlabels, title, out_path, bar_color=COLOR_QPS, bar_hatch="", xlabel=""):
    """Wrapper: builds its own figure, calls plot_dualaxis, and saves; preserves the original standalone-figure behavior."""
    fig, ax1 = plt.subplots(figsize=FIGSIZE_SINGLE)
    plot_dualaxis(ax1, ax1.twinx(), cfgs, xlabels, title, bar_color, bar_hatch, xlabel=xlabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path)
    print("saved", out_path, "/", pdf_path)


def main():
    if not os.path.exists(JSON_PATH):
        raise SystemExit(f"missing {JSON_PATH}; run ac_sweep_analyze.py first")
    data = json.load(open(JSON_PATH))
    csweep = [d for d in data if d["A"] == 1 and d["C"] != 10]
    asweep = [d for d in data if d["C"] == 10]
    csweep.sort(key=lambda d: d["C"])
    asweep.sort(key=lambda d: d["A"])

    # Keep the two standalone figures (keep long titles, add x-axis labels)
    plot_single(csweep, [f"{d['C']}" for d in csweep],
                "C-sweep (A=1 fixed): QPS@0.9recall vs Indexing time (OpenAI1536 d=1536 N~999k)",
                os.path.join(LOG_DIR, "ac_sweep_c_dual.png"),
                bar_color=BAR_COLOR_C, bar_hatch=BAR_HATCH_C, xlabel="Cardinality")
    plot_single(asweep, [f"{d['A']}" for d in asweep],
                "A-sweep (C=10 fixed): QPS@0.9recall vs Indexing time (OpenAI1536 d=1536 N~999k)",
                os.path.join(LOG_DIR, "ac_sweep_a_dual.png"),
                bar_color=BAR_COLOR_A, bar_hatch=BAR_HATCH_A, xlabel="Number of Attributes")

    # Combined figure: 1x2 subplots (left A-sweep, right C-sweep); no subplot titles, numeric x-axis, each with an x-axis label
    fig, (axL, axR) = plt.subplots(1, 2, figsize=FIGSIZE_COMBINED)
    plot_dualaxis(axL, axL.twinx(), asweep, [f"{d['A']}" for d in asweep],
                  "", xlabel="Number of Attributes", bar_color=BAR_COLOR_A, bar_hatch=BAR_HATCH_A)
    plot_dualaxis(axR, axR.twinx(), csweep, [f"{d['C']}" for d in csweep],
                  "", xlabel="Cardinality", bar_color=BAR_COLOR_C, bar_hatch=BAR_HATCH_C)
    fig.tight_layout()
    out = os.path.join(LOG_DIR, "ac_sweep_dual_combined.png")
    fig.savefig(out, dpi=DPI)
    pdf_out = os.path.splitext(out)[0] + ".pdf"
    fig.savefig(pdf_out)
    print("saved", out, "/", pdf_out)


if __name__ == "__main__":
    main()
