#!/usr/bin/env python3
"""
plot_results.py  —  ML Benchmark Visualiser  v3
Fixes over v2:
  - Log-scale: labels spread vertically with adjust_text-style nudging
  - Bar chart: CUDA labels placed ABOVE CPU bar when overlapping
  - Heatmap: speedup clipped to [0.1, +inf], no negative values
  - Speedup curves: Y-axis always starts at 0, consistent scale
  - Stacked bars: CPU/CUDA labels in top margin, not on bars
  - L-BFGS: ⚠ label only when speedup truly < 0.5 after clipping
"""

import argparse, csv, sys, math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np

# ── Design tokens ──────────────────────────────────────────────
BG      = "#0d1117"
SURFACE = "#161b22"
BORDER  = "#30363d"
FG      = "#e6edf3"
MUTED   = "#8b949e"
CPU_C   = "#f78166"
CUDA_C  = "#3fb950"
GOLD    = "#ffa657"
RED_BAD = "#ff6b6b"

ALGO_CLR = {
    "AdamW":        "#58a6ff",
    "Nadam":        "#bc8cff",
    "RMSProp":      "#ffa657",
    "SGD_Nesterov": "#3fb950",
    "SGDR":         "#f78166",
    "LBFGS":        "#79c0ff",
    "GMM_EM":       "#d2a8ff",
    "KernelPCA":    "#ffb86c",
    "MLP_AdamW":    "#56d364",
    "RandomForest": "#ff7b72",
}
ALGO_LABEL = {
    "AdamW":        "AdamW",
    "Nadam":        "Nadam",
    "RMSProp":      "RMSProp",
    "SGD_Nesterov": "SGD+Nesterov",
    "SGDR":         "SGDR",
    "LBFGS":        "L-BFGS",
    "GMM_EM":       "GMM-EM",
    "KernelPCA":    "Kernel PCA",
    "MLP_AdamW":    "MLP (AdamW)",
    "RandomForest": "Random Forest",
}

# ── Helpers ────────────────────────────────────────────────────
def load_speedup(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "algo":    r["algorithm"],
                "n":       int(r["n_samples"]),
                "cpu_ms":  float(r["cpu_ms"]),
                "cuda_ms": float(r["cuda_ms"]),
                # clip: negative/zero speedup is a measurement artifact
                "speedup": max(0.10, float(r["speedup"])),
            })
    return rows

def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(color=BORDER, lw=0.45, ls="--", alpha=0.55, zorder=0)
    if title:  ax.set_title(title,  color=FG,   fontsize=9.5,
                             fontweight="bold", pad=7, linespacing=1.4)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=8)

def n_label(n):
    return f"{n//1000}k" if n >= 1000 else str(n)

def smooth_ma(vals, w=3):
    """Moving average, raw endpoints."""
    arr = np.array(vals, dtype=float)
    if len(arr) <= w:
        return arr.tolist()
    out = np.convolve(arr, np.ones(w)/w, mode='same')
    out[0] = arr[0]; out[-1] = arr[-1]
    return out.tolist()

def spread_labels(ys, min_gap=0.06):
    """
    Nudge y-positions apart so annotations don't overlap.
    Returns adjusted list of y positions (same order as input).
    Simple iterative push-apart.
    """
    pairs = sorted(enumerate(ys), key=lambda x: x[1])
    adjusted = [y for _, y in pairs]
    for _ in range(50):
        moved = False
        for i in range(1, len(adjusted)):
            gap = adjusted[i] - adjusted[i-1]
            if gap < min_gap:
                push = (min_gap - gap) / 2
                adjusted[i-1] -= push
                adjusted[i]   += push
                moved = True
        if not moved:
            break
    # Map back to original order
    result = [0.0] * len(ys)
    for new_i, (orig_i, _) in enumerate(pairs):
        result[orig_i] = adjusted[new_i]
    return result

# ══════════════════════════════════════════════════════════════
def make_plot(speedup_path, output_path):
    sp_rows = load_speedup(speedup_path)

    algos = sorted(set(r["algo"] for r in sp_rows),
                   key=lambda a: ALGO_LABEL.get(a, a))
    sizes = sorted(set(r["n"] for r in sp_rows))

    cpu_t  = defaultdict(dict)
    cuda_t = defaultdict(dict)
    sp_map = defaultdict(dict)
    for r in sp_rows:
        a, n = r["algo"], r["n"]
        cpu_t[a][n]  = r["cpu_ms"]
        cuda_t[a][n] = r["cuda_ms"]
        sp_map[a][n] = r["speedup"]

    n_max   = max(sizes)
    n_algos = len(algos)

    # ── Layout: 4 rows ─────────────────────────────────────────
    fig = plt.figure(figsize=(26, 23), facecolor=BG)
    fig.suptitle(
        "Advanced ML Algorithms  —  CPU vs CUDA GPU Acceleration",
        color=FG, fontsize=17, fontweight="bold", y=0.987
    )
    outer = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[2.2, 1.5, 1.5, 2.1],
        hspace=0.50, top=0.965, bottom=0.038,
        left=0.055, right=0.975
    )
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.27)
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[1], wspace=0.30)
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[2], wspace=0.30)
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3], wspace=0.32)

    ax_bar  = fig.add_subplot(gs0[0])
    ax_log  = fig.add_subplot(gs0[1])
    ax_sp1  = [fig.add_subplot(gs1[i]) for i in range(5)]
    ax_sp2  = [fig.add_subplot(gs2[i]) for i in range(5)]
    ax_heat = fig.add_subplot(gs3[0])
    ax_stk  = fig.add_subplot(gs3[1])
    ax_peak = fig.add_subplot(gs3[2])

    # ──────────────────────────────────────────────────────────
    # PANEL 0-LEFT: Grouped bar chart @ n_max
    # ──────────────────────────────────────────────────────────
    x  = np.arange(n_algos)
    bw = 0.36
    bc = [cpu_t[a].get(n_max, 0)  for a in algos]
    bg_vals = [cuda_t[a].get(n_max, 0) for a in algos]

    ax_bar.bar(x - bw/2, bc,      bw, color=CPU_C,  alpha=0.85, zorder=3, label="CPU")
    ax_bar.bar(x + bw/2, bg_vals, bw, color=CUDA_C, alpha=0.85, zorder=3, label="CUDA")

    # Smart label placement: if CUDA bar is tiny compared to CPU,
    # put its label above the CPU bar height to avoid occlusion
    y_max_bar = max(max(bc), 1)
    for xi, (vc, vg) in enumerate(zip(bc, bg_vals)):
        # CPU label — always above its bar
        if vc > 0:
            ax_bar.text(xi - bw/2, vc + y_max_bar*0.012,
                        f"{vc:.0f}", ha="center", va="bottom",
                        color=CPU_C, fontsize=6.5, fontweight="bold")
        # CUDA label — above CPU bar if CUDA bar is < 15% of CPU bar
        if vg > 0:
            label_y = (vc + y_max_bar*0.015) if vg < vc * 0.15 else (vg + y_max_bar*0.012)
            ax_bar.text(xi + bw/2, label_y,
                        f"{vg:.0f}", ha="center", va="bottom",
                        color=CUDA_C, fontsize=6.5, fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([ALGO_LABEL.get(a, a) for a in algos],
                            rotation=32, ha="right", fontsize=7.5, color=MUTED)
    ax_bar.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=FG, fontsize=9)
    style(ax_bar,
          title=f"Execution Time  @  N = {n_max:,}  samples",
          ylabel="Time  (ms)")

    # ──────────────────────────────────────────────────────────
    # PANEL 0-RIGHT: Log-scale time vs N, labels spread apart
    # ──────────────────────────────────────────────────────────
    # Collect end-of-line y values for label placement
    label_info = []   # (y_data, algo)
    for algo in algos:
        clr = ALGO_CLR.get(algo, "#aaa")
        ns_c = sorted(cpu_t[algo])
        ns_g = sorted(cuda_t[algo])
        ts_c = [cpu_t[algo][n]  for n in ns_c]
        ts_g = [cuda_t[algo][n] for n in ns_g]
        ax_log.plot(ns_c, ts_c, "o--", color=clr, alpha=0.30, lw=1.2, ms=3)
        ax_log.plot(ns_g, ts_g, "s-",  color=clr, alpha=0.88, lw=1.8, ms=4)
        label_info.append((math.log10(max(ts_g[-1], 0.001)), algo, clr))

    # Spread labels so they don't overlap (work in log-space)
    raw_log_ys = [x[0] for x in label_info]
    spread_log = spread_labels(raw_log_ys, min_gap=0.12)

    x_label = n_max * 1.02
    for (_, algo, clr), log_y in zip(label_info, spread_log):
        ax_log.text(x_label, 10**log_y,
                    ALGO_LABEL.get(algo, algo),
                    color=clr, fontsize=6.8, va="center",
                    fontweight="bold", clip_on=False)

    ax_log.set_yscale("log")
    ax_log.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0f}" if v >= 10 else f"{v:.1f}"))
    ax_log.set_xlim(left=0, right=n_max * 1.30)

    p_cpu  = mpatches.Patch(color=MUTED, alpha=0.5, label="- - -  CPU")
    p_cuda = mpatches.Patch(color=FG,   alpha=0.9, label="———  CUDA")
    ax_log.legend(handles=[p_cpu, p_cuda],
                  facecolor=SURFACE, edgecolor=BORDER,
                  labelcolor=FG, fontsize=8.5, loc="upper left")
    style(ax_log,
          title="Time vs Samples  (log scale)  — dashed = CPU, solid = CUDA",
          xlabel="Samples  (N)", ylabel="Time  (ms, log)")

    # ──────────────────────────────────────────────────────────
    # PANELS 1 & 2: 10 speedup curves, 5 per row
    # ──────────────────────────────────────────────────────────
    def draw_speedup(ax, algo, show_ylabel):
        clr  = ALGO_CLR.get(algo, "#aaa")
        lbl  = ALGO_LABEL.get(algo, algo)
        ns   = sorted(sp_map[algo])
        sp   = [sp_map[algo][n] for n in ns]
        is_lbfgs = (algo == "LBFGS")

        # Red zone: GPU overhead (< 1×)
        ax.axhspan(0, 1, color=RED_BAD, alpha=0.07, zorder=1)
        ax.axhline(1.0, color=RED_BAD, lw=1.0, ls="--", alpha=0.65, zorder=2)

        if is_lbfgs:
            sp_s = smooth_ma(sp, w=3)
            ax.plot(ns, sp, "o", color=clr, alpha=0.40, ms=5.5, zorder=4,
                    label="Raw")
            ax.plot(ns, sp_s, "-", color=clr, alpha=0.92, lw=2.3, zorder=5,
                    label="Trend")
            ax.fill_between(ns, 1, sp_s, alpha=0.13, color=clr, zorder=3)
            ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=MUTED,
                      fontsize=6, loc="upper left")
        else:
            ax.fill_between(ns, 1, sp, alpha=0.15, color=clr, zorder=3)
            ax.plot(ns, sp, "o-", color=clr, lw=2.2, ms=5.5, zorder=5)

        # Peak annotation (top of curve)
        mx_sp = max(sp); mx_n = ns[sp.index(mx_sp)]
        ax.annotate(f"{mx_sp:.1f}×",
                    xy=(mx_n, mx_sp), xytext=(0, 11),
                    textcoords="offset points",
                    ha="center", color=GOLD, fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=GOLD, lw=0.6))

        # Overhead annotation (bottom, only if meaningful)
        mn_sp = min(sp)
        if mn_sp < 0.95:   # only label if more than 5% slower
            mn_n = ns[sp.index(mn_sp)]
            ax.text(mn_n, mn_sp - abs(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06,
                    f"{mn_sp:.2f}×\noverhead",
                    ha="center", va="top", color=RED_BAD,
                    fontsize=6, fontweight="bold")

        # Y-axis: always include 0 and the peak
        y_top = mx_sp * 1.18
        ax.set_ylim(0, y_top)

        ax.set_xticks(ns[::2])
        ax.set_xticklabels([n_label(n) for n in ns[::2]], fontsize=7)
        style(ax, title=lbl, xlabel="Samples",
              ylabel="Speedup  ×" if show_ylabel else "")

    for i, algo in enumerate(algos[:5]):
        draw_speedup(ax_sp1[i], algo, show_ylabel=(i == 0))
    for i, algo in enumerate(algos[5:10]):
        draw_speedup(ax_sp2[i], algo, show_ylabel=(i == 0))

    # ──────────────────────────────────────────────────────────
    # PANEL 3-LEFT: Heatmap — clipped speedup, two-slope colour
    # ──────────────────────────────────────────────────────────
    heat = np.zeros((n_algos, len(sizes)))
    for i, algo in enumerate(algos):
        for j, n in enumerate(sizes):
            v = sp_map[algo].get(n, 1.0)
            heat[i, j] = max(0.10, v)   # clip: no negative values

    from matplotlib.colors import TwoSlopeNorm
    v_min = max(0.0, heat.min() - 0.05)
    v_max = heat.max()
    norm = TwoSlopeNorm(vmin=v_min, vcenter=1.0, vmax=v_max)

    im = ax_heat.imshow(heat, aspect="auto", cmap="RdYlGn", norm=norm)
    ax_heat.set_xticks(range(len(sizes)))
    ax_heat.set_xticklabels([n_label(n) for n in sizes],
                              fontsize=8, color=MUTED)
    ax_heat.set_yticks(range(n_algos))
    ax_heat.set_yticklabels([ALGO_LABEL.get(a, a) for a in algos],
                              fontsize=8, color=MUTED)

    for i in range(n_algos):
        for j in range(len(sizes)):
            v   = heat[i, j]
            raw = sp_map[algos[i]].get(sizes[j], 1.0)
            # Text colour based on cell brightness
            cell_norm = norm(v)
            fc = "#111111" if 0.25 < cell_norm < 0.82 else FG
            txt = f"{v:.1f}×"
            # Mark only genuinely suspicious raw values (before clip)
            if raw < 0.20:
                txt = f"⚠{v:.1f}×"
            ax_heat.text(j, i, txt, ha="center", va="center",
                          fontsize=7, fontweight="bold", color=fc)

    cbar = fig.colorbar(im, ax=ax_heat, pad=0.03, fraction=0.046)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.set_label("Speedup ×  (red < 1  →  GPU overhead)", color=MUTED, fontsize=7.5)
    style(ax_heat,
          title="Speedup Heatmap  (N  vs  Algorithm)\n"
                "Red = GPU overhead  |  Green = CUDA faster")
    ax_heat.set_facecolor(SURFACE)
    for sp_ in ax_heat.spines.values():
        sp_.set_edgecolor(BORDER)

    # ──────────────────────────────────────────────────────────
    # PANEL 3-CENTRE: Stacked total time, CPU/CUDA labels in margin
    # ──────────────────────────────────────────────────────────
    x_pos = np.arange(len(sizes))
    bot_c = np.zeros(len(sizes))
    bot_g = np.zeros(len(sizes))

    for algo in algos:
        clr = ALGO_CLR.get(algo, "#aaa")
        vc  = np.array([cpu_t[algo].get(n, 0)  for n in sizes])
        vg  = np.array([cuda_t[algo].get(n, 0) for n in sizes])
        ax_stk.bar(x_pos - 0.22, vc, 0.40, bottom=bot_c,
                   color=clr, alpha=0.50, zorder=3)
        ax_stk.bar(x_pos + 0.22, vg, 0.40, bottom=bot_g,
                   color=clr, alpha=0.92, zorder=3,
                   label=ALGO_LABEL.get(algo, algo))
        bot_c += vc; bot_g += vg

    # "CPU" / "CUDA" labels in the top axis margin (transform=ax.transAxes)
    ax_stk.text(-0.22/len(sizes) + 0.5 - 0.22/(n_algos),
                1.015, "◀ CPU", transform=ax_stk.transAxes,
                ha="center", va="bottom", color=CPU_C,
                fontsize=8.5, fontweight="bold")
    ax_stk.text(0.5 + 0.22/(n_algos),
                1.015, "CUDA ▶", transform=ax_stk.transAxes,
                ha="center", va="bottom", color=CUDA_C,
                fontsize=8.5, fontweight="bold")

    # Simpler: just annotate last group (rightmost N)
    # Arrow pointing at left vs right bar of last group
    last_x = len(sizes) - 1
    ax_stk.annotate("CPU", xy=(last_x - 0.22, bot_c[last_x]),
                    xytext=(last_x - 0.55, bot_c[last_x] * 0.85),
                    color=CPU_C, fontsize=8, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=CPU_C, lw=1.2))
    ax_stk.annotate("CUDA", xy=(last_x + 0.22, bot_g[last_x]),
                    xytext=(last_x - 0.10, bot_g[last_x] * 1.08),
                    color=CUDA_C, fontsize=8, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=CUDA_C, lw=1.2))

    ax_stk.set_xticks(x_pos)
    ax_stk.set_xticklabels([n_label(n) for n in sizes], fontsize=7.5)
    ax_stk.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=FG,
                  fontsize=5.8, loc="upper left", ncol=2)
    style(ax_stk,
          title="Total Compute Time  (Left bar = CPU,  Right = CUDA)",
          xlabel="Samples  (N)", ylabel="Cumulative time  (ms)")

    # ──────────────────────────────────────────────────────────
    # PANEL 3-RIGHT: Peak speedup horizontal bar, sorted
    # ──────────────────────────────────────────────────────────
    peak_data = []
    for algo in algos:
        vals = [sp_map[algo][n] for n in sorted(sp_map[algo])]
        ns_  = sorted(sp_map[algo])
        mx   = max(vals); mx_n = ns_[vals.index(mx)]
        peak_data.append((mx, algo, mx_n))
    peak_data.sort(key=lambda x: x[0])   # ascending → longest bar at top

    sp_sorted    = [d[0] for d in peak_data]
    algos_sorted = [d[1] for d in peak_data]
    n_at_peak    = [d[2] for d in peak_data]
    clrs_sorted  = [ALGO_CLR.get(a, "#aaa") for a in algos_sorted]

    y_pos = np.arange(len(algos_sorted))
    hbars = ax_peak.barh(y_pos, sp_sorted, color=clrs_sorted,
                          alpha=0.88, zorder=3)

    # Paint bars < 1× red
    for bar, val in zip(hbars, sp_sorted):
        if val < 1.0:
            bar.set_color(RED_BAD); bar.set_alpha(0.75)

    ax_peak.axvline(1.0, color=RED_BAD, lw=1.3, ls="--",
                    alpha=0.8, zorder=4, label="Breakeven  (1×)")

    ax_peak.set_yticks(y_pos)
    ax_peak.set_yticklabels([ALGO_LABEL.get(a, a) for a in algos_sorted],
                              fontsize=8.5, color=FG)
    for bar, val, n_ in zip(hbars, sp_sorted, n_at_peak):
        clr_ = RED_BAD if val < 1.0 else GOLD
        ax_peak.text(val + max(sp_sorted) * 0.01,
                     bar.get_y() + bar.get_height()/2,
                     f"  {val:.1f}×  @N={n_label(n_)}",
                     va="center", color=clr_,
                     fontsize=8.5, fontweight="bold")

    ax_peak.set_xlim(0, max(sp_sorted) * 1.30)
    ax_peak.legend(facecolor=SURFACE, edgecolor=BORDER,
                   labelcolor=FG, fontsize=8.5, loc="lower right")
    style(ax_peak,
          title="Peak CUDA Speedup  per Algorithm\n(sorted ascending)",
          xlabel="Speedup  (×  faster than CPU)")

    # ── Footer ─────────────────────────────────────────────────
    fig.text(
        0.5, 0.005,
        "CPU: g++ -O3 -march=native  │  CUDA: nvcc -O3 --use_fast_math  │  "
        "Features=32  │  Epochs=60  │  Batch=64  │  "
        "N: 512→1k→2k→4k→8k→16k  │  "
        "L-BFGS: time varies with adaptive convergence  │  "
        "Speedup < 1× = GPU kernel-launch + PCIe overhead dominates",
        ha="center", color=MUTED, fontsize=7, style="italic"
    )

    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"  Saved → {output_path}")
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", default="results/combined_results.csv")
    ap.add_argument("--speedup",  default="results/speedup_summary.csv")
    ap.add_argument("--output",   default="results/benchmark_plots.png")
    args = ap.parse_args()
    make_plot(args.speedup, args.output)
