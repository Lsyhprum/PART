#!/usr/bin/env python3
"""双轴对比图：读 ac_sweep_results.json，为 C-sweep 与 A-sweep 各生成一张
「QPS@0.9 柱形图（左 y 轴）+ Indexing time 折线（右 y 轴）」对比图；
并在末尾生成一张 1×2 合并图（左=C-sweep，右=A-sweep）。
仅读取 JSON，不重跑实验、不修改任何既有脚本/图片与 categorical C++ 代码。
文件：ac_sweep_c_dual.png / ac_sweep_a_dual.png（保留）+ ac_sweep_dual_combined.png（新增）。"""
import json
import os
import glob as _glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = "/home/yq/SIGMOD2027/_log"
JSON_PATH = os.path.join(LOG_DIR, "ac_sweep_results.json")

# ====================== 可调参数（微调入口，改这里即可） ======================
FIGSIZE_SINGLE   = (9, 5)     # 独立图尺寸（宽, 高）
FIGSIZE_COMBINED = (16, 5)    # 合并图(1×2)尺寸
DPI = 140                     # 输出分辨率
FS_TITLE  = 36                # 子图标题字号
FS_AXIS   = 36                # 坐标轴标签字号
FS_TICK   = 22                # 刻度标签字号（x/y 轴通用）
FS_ANNOT  = 20                # 柱顶/折线数值标注字号
FS_LEGEND = 26                # 图例字号
FS_SCI   = 18                # 科学计数法指数(×10^n)字号
LW = 4                      # 折线线宽
MS = 20                        # 折线 marker 大小
BAR_WIDTH = 0.55              # 柱宽
COLOR_QPS       = "#2c7fb8"   # 柱(QPS)默认颜色
COLOR_IDX       = "#000000"   # 折线(Indexing)颜色（黑）
# C-sweep / A-sweep 柱形区分样式（颜色 + 花纹）
BAR_COLOR_C  = "#2c7fb8"   # C-sweep 柱颜色（蓝）
BAR_COLOR_A  = "#2ca25f"   # A-sweep 柱颜色（绿，避开橙色折线）
BAR_HATCH_C  = "x"         # C-sweep 柱花纹（斜方格 / 对角交叉）
BAR_HATCH_A  = "+"         # A-sweep 柱花纹（正方格 / 正交十字）

# ---- 字体（Times）：注册指定目录下的 .ttf 并设为全局默认字体 ----
FONT_DIR = "/home/yq/_env/times"
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
    """在给定 ax1(左轴 QPS 柱) / ax2(右轴 Indexing 折线) 上作画；不建 fig、不 savefig。"""
    x = list(range(len(cfgs)))
    qps = [d["qps90"] for d in cfgs]
    idx = [d["index_s"] for d in cfgs]

    bars = ax1.bar(x, qps, color=bar_color, edgecolor="black", linewidth=0.6,
                   width=BAR_WIDTH, hatch=bar_hatch, label="QPS")
    line, = ax2.plot(x, idx, marker="s", color=COLOR_IDX, linewidth=LW,
                     markersize=MS, label="Index time")

    # 轴标签与范围
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

    # 柱顶数值标注（白色底框，避免压在柱面上不可读）
    bbox = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
    for b, v in zip(bars, qps):
        ax1.text(b.get_x() + b.get_width() / 2, v + max(qps) * 0.015,
                 f"{v:,.0f}", ha="center", va="bottom", fontsize=FS_ANNOT, bbox=bbox)

    ax1.grid(axis="y", alpha=0.3)
    if title:
        ax1.set_title(title, fontsize=FS_TITLE)
    ax1.set_xlabel(xlabel, fontsize=FS_AXIS)

    # 合并图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=FS_LEGEND, framealpha=1.0)


def plot_single(cfgs, xlabels, title, out_path, bar_color=COLOR_QPS, bar_hatch="", xlabel=""):
    """包装：自建 fig 调 plot_dualaxis 并 savefig，维持原独立图行为。"""
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

    # 保留两张独立图（长标题保留，补 x 轴标签）
    plot_single(csweep, [f"{d['C']}" for d in csweep],
                "C-sweep (A=1 fixed): QPS@0.9recall vs Indexing time (OpenAI1536 d=1536 N~999k)",
                os.path.join(LOG_DIR, "ac_sweep_c_dual.png"),
                bar_color=BAR_COLOR_C, bar_hatch=BAR_HATCH_C, xlabel="Cardinality")
    plot_single(asweep, [f"{d['A']}" for d in asweep],
                "A-sweep (C=10 fixed): QPS@0.9recall vs Indexing time (OpenAI1536 d=1536 N~999k)",
                os.path.join(LOG_DIR, "ac_sweep_a_dual.png"),
                bar_color=BAR_COLOR_A, bar_hatch=BAR_HATCH_A, xlabel="Number of Attributes")

    # 合并图：1×2 子图（左 A-sweep，右 C-sweep）；去子标题、x 轴纯数字、各加 x 轴标签
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
