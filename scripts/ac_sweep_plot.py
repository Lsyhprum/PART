#!/usr/bin/env python3
"""读 ac_sweep_results.json，绘制：
 (1) A-sweep (固定 C=10, A=1..4) 三联条形图：QPS@0.9 / Indexing time / max recall；
 (2) A/C 合并对照图：选择率 sel=(1/C)^A → QPS@0.9（log-log）与 → max recall，
     证明 A 与 C 的影响完全由 selectivity 决定。
仅绘图，不改任何库代码。"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = "/home/yq/SIGMOD2027/_log"
data = json.load(open(f"{LOG_DIR}/ac_sweep_results.json"))
csweep = [d for d in data if d["A"] == 1 and d["C"] != 10]
asweep = [d for d in data if d["C"] == 10]
for s in (csweep, asweep):
    s.sort(key=lambda d: d["A"] if d["C"] == 10 else d["C"])

def bar3(axs, cfgs, xticklabels, title_prefix):
    qps = [d["qps90"] for d in cfgs]
    idx = [d["index_s"] for d in cfgs]
    mrec = [d["maxrecall"] for d in cfgs]
    x = list(range(len(cfgs)))
    w = 0.62
    def panel(ax, vals, color, ylabel, title, fmt, logy=False):
        ax.bar(x, vals, width=w, color=color, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(xticklabels, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12); ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        vmax = max(vals)
        for b, v in zip(ax.patches, vals):
            ax.text(b.get_x()+b.get_width()/2, v + vmax*0.015, fmt.format(v),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, vmax*1.15)
    panel(axs[0], qps, "#2c7fb8", "QPS @ 90% recall", f"{title_prefix}: QPS@0.9recall", "{:,.0f}")
    panel(axs[1], idx, "#d95f0e", "Indexing time (s)", f"{title_prefix}: index build time", "{:.1f}")
    panel(axs[2], mrec, "#1b9e77", "max achievable recall", f"{title_prefix}: max recall", "{:.3f}",
          logy=False)
    lo = min(mrec) - (max(mrec)-min(mrec))*0.4 - 0.005
    axs[2].set_ylim(max(0.85, lo), 1.005)

# ---- 图1：A-sweep 三联条形图 ----
fig1, axs = plt.subplots(1, 3, figsize=(17, 4.8))
bar3(axs, asweep, [f"A={d['A']}" for d in asweep], "A-sweep (C=10 fixed)")
fig1.suptitle("A-sweep — effect of attribute count A (C=10 fixed, OpenAI1536 d=1536 N~999k)",
              fontsize=14, fontweight="bold")
fig1.tight_layout(rect=[0, 0, 1, 0.94])
fig1.savefig(f"{LOG_DIR}/ac_sweep_a_hist.png", dpi=140)
print("saved", f"{LOG_DIR}/ac_sweep_a_hist.png")

# ---- 图2：A/C 合并——选择率决定一切 ----
fig2, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
def scatter(ax, cfgs, ykey, color, label):
    xs = [d["sel"] for d in cfgs]
    ys = [d[ykey] for d in cfgs]
    ax.scatter(xs, ys, s=90, color=color, edgecolor="black", zorder=3, label=label)
    for d in cfgs:
        ax.annotate(f"A{d['A']}C{d['C']}", (d["sel"], d[ykey]),
                    textcoords="offset points", xytext=(5, 4), fontsize=9)
allpts = csweep + asweep
scatter(axL, csweep, "qps90", "#2c7fb8", "C-sweep (A=1)")
scatter(axL, asweep, "qps90", "#d95f0e", "A-sweep (C=10)")
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xlabel("selectivity  sel = (1/C)^A  (log)", fontsize=12)
axL.set_ylabel("QPS @ 90% recall  (log)", fontsize=12)
axL.set_title("Selectivity governs QPS@0.9  (A & C collapse onto one curve)", fontsize=12)
axL.grid(True, which="both", alpha=0.3); axL.legend()

scatter(axR, csweep, "maxrecall", "#2c7fb8", "C-sweep (A=1)")
scatter(axR, asweep, "maxrecall", "#d95f0e", "A-sweep (C=10)")
axR.set_xscale("log")
axR.set_xlabel("selectivity  sel = (1/C)^A  (log)", fontsize=12)
axR.set_ylabel("max achievable recall", fontsize=12)
axR.set_ylim(0.85, 1.005)
axR.set_title("Selectivity governs max recall (ceiling drops < sel~1e-4)", fontsize=12)
axR.grid(True, which="both", alpha=0.3); axR.legend()

fig2.suptitle("A and C affect performance ONLY through selectivity (OpenAI1536)",
              fontsize=14, fontweight="bold")
fig2.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig(f"{LOG_DIR}/ac_sweep_combined.png", dpi=140)
print("saved", f"{LOG_DIR}/ac_sweep_combined.png")
