# -*- coding: utf-8 -*-
"""
metric_b_mean_radius.py
Metric B: mean radius per class (+/- std).
Low-N classes (N < MIN_N) shown with hollow markers.

Outputs:
  fig1B_page_comparison.png
  fig2B_crossgroup_day4.png
  fig3B_timecourse.png
  fig3B_byday.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from utils import (
    CLASSES, CLASS_COLORS, FIG1_CONDITIONS, FIG2_LAYOUT, FIG3_GROUPS,
    MIN_N, load_condition, class_mean_std, class_counts,
    overlay_hollow_marker, apply_style, save_figure,
)

COL     = "radius_um"
YLABEL  = "Mean radius (um)"
DAY_MAP = {"Day 0": 0, "Day 1": 1, "Day 4": 4}
MARKERS = ["o", "s", "^"]


def _hollow_handle():
    return mlines.Line2D([], [], marker="o", markerfacecolor="none",
                         markeredgecolor="#555", linestyle="none",
                         markersize=6, label="N<{}: unreliable".format(MIN_N))


# -- Figure 1B ----------------------------------------------------------------

def fig1B():
    apply_style()
    labels   = list(FIG1_CONDITIONS.keys())
    folders  = list(FIG1_CONDITIONS.values())
    datasets = [load_condition(f) for f in folders]
    totals   = [len(df) for df in datasets]
    all_ms   = [class_mean_std(df, COL) for df in datasets]
    all_cnts = [class_counts(df) for df in datasets]

    offsets = np.linspace(-0.2, 0.2, len(labels))
    fig, ax = plt.subplots(figsize=(13, 4.5))

    for i, (means, stds) in enumerate(all_ms):
        cnts = all_cnts[i]
        for c in CLASSES:
            if means[c] > 0:
                ax.errorbar(c + offsets[i], means[c], yerr=stds[c],
                            fmt=MARKERS[i], color=CLASS_COLORS[c],
                            markeredgecolor="#333", markeredgewidth=0.5,
                            markersize=6, capsize=3, capthick=0.8, linewidth=0.8)
                if cnts[c] < MIN_N:
                    overlay_hollow_marker(ax, c + offsets[i], means[c],
                                          CLASS_COLORS[c], markersize=8)

    cond_handles = [
        mlines.Line2D([], [], marker=MARKERS[i], color="#555", linestyle="none",
                      markersize=8, label="{} (N={:,})".format(lbl, tot))
        for i, (lbl, tot) in enumerate(zip(labels, totals))
    ]
    ax.legend(handles=cond_handles + [_hollow_handle()], fontsize=8, framealpha=0.9)
    ax.set_xticks(range(16))
    ax.set_xticklabels([str(c) for c in CLASSES])
    ax.set_xlabel("Class")
    ax.set_ylabel(YLABEL)
    ax.set_title("Figure 1B -- Mean radius: PAGE (day 3) vs heated conditions (day 4)")
    ax.set_xlim(-0.6, 15.6)
    fig.tight_layout()
    save_figure(fig, "fig1B_page_comparison.png")
    plt.close(fig)


# -- Figure 2B ----------------------------------------------------------------

def fig2B():
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey=True)
    fig.suptitle("Figure 2B -- Mean radius per class: cross-group comparison (day 4)", fontsize=11)

    for row in range(2):
        for col in range(2):
            lbl, folder = FIG2_LAYOUT[row][col]
            df          = load_condition(folder)
            means, stds = class_mean_std(df, COL)
            cnts        = class_counts(df)
            ax          = axes[row][col]

            for c in CLASSES:
                if means[c] > 0:
                    ax.errorbar(c, means[c], yerr=stds[c],
                                fmt="o", color=CLASS_COLORS[c],
                                markeredgecolor="#333", markeredgewidth=0.5,
                                markersize=6, capsize=3, capthick=0.8, linewidth=0.8)
                    if cnts[c] < MIN_N:
                        overlay_hollow_marker(ax, c, means[c], CLASS_COLORS[c], markersize=8)

            ax.set_title("{}\n(N={:,})".format(lbl.replace("\n", ", "), len(df)), fontsize=9)
            ax.set_xticks(range(16))
            ax.set_xticklabels([str(c) for c in CLASSES], fontsize=7)
            if col == 0:
                ax.set_ylabel(YLABEL)
            if row == 1:
                ax.set_xlabel("Class")

    fig.text(0.01, 0.01, "open circle = N<{}: unreliable".format(MIN_N),
             fontsize=7, color="#cc0000", va="bottom")
    fig.tight_layout()
    save_figure(fig, "fig2B_crossgroup_day4.png")
    plt.close(fig)


# -- Figure 3B (x = class) ----------------------------------------------------

def fig3B():
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle("Figure 3B -- Mean radius per class over time (per group)", fontsize=11)

    for idx, (gname, timepoints) in enumerate(FIG3_GROUPS.items()):
        ax          = axes[idx // 2][idx % 2]
        n_t         = len(timepoints)
        offsets     = np.linspace(-0.2, 0.2, n_t)
        tp_datasets = [(day_lbl, load_condition(folder)) for day_lbl, folder in timepoints]

        for i, (day_lbl, df) in enumerate(tp_datasets):
            means, stds = class_mean_std(df, COL)
            cnts        = class_counts(df)
            for c in CLASSES:
                if means[c] > 0:
                    ax.errorbar(c + offsets[i], means[c], yerr=stds[c],
                                fmt=MARKERS[i], color=CLASS_COLORS[c],
                                markeredgecolor="#333", markeredgewidth=0.5,
                                markersize=5, capsize=3, capthick=0.8, linewidth=0.8)
                    if cnts[c] < MIN_N:
                        overlay_hollow_marker(ax, c + offsets[i], means[c],
                                              CLASS_COLORS[c], markersize=7)

        day_handles = [
            mlines.Line2D([], [], marker=MARKERS[i], color="#555", linestyle="none",
                          markersize=7, label="{} (N={:,})".format(day_lbl, len(df)))
            for i, (day_lbl, df) in enumerate(tp_datasets)
        ]
        ax.legend(handles=day_handles + [_hollow_handle()], fontsize=7, framealpha=0.9)
        ax.set_title(gname, fontsize=9)
        ax.set_xticks(range(16))
        ax.set_xticklabels([str(c) for c in CLASSES], fontsize=7)
        if idx % 2 == 0:
            ax.set_ylabel(YLABEL)
        if idx >= 2:
            ax.set_xlabel("Class")

    fig.tight_layout()
    save_figure(fig, "fig3B_timecourse.png")
    plt.close(fig)


# -- Figure 3B (x = day) ------------------------------------------------------

def fig3B_byday():
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharey=True)
    fig.suptitle("Figure 3B (by day) -- Mean radius vs time (per group)", fontsize=11)

    for idx, (gname, timepoints) in enumerate(FIG3_GROUPS.items()):
        ax          = axes[idx // 2][idx % 2]
        days_x      = [DAY_MAP[day_lbl] for day_lbl, _ in timepoints]
        tp_datasets = [load_condition(folder) for _, folder in timepoints]

        for c in CLASSES:
            means_c, stds_c, days_v, counts_v = [], [], [], []
            for d, df in zip(days_x, tp_datasets):
                means, stds = class_mean_std(df, COL)
                cnts        = class_counts(df)
                if means[c] > 0:
                    means_c.append(means[c])
                    stds_c.append(stds[c])
                    days_v.append(d)
                    counts_v.append(cnts[c])
            if means_c:
                ax.errorbar(days_v, means_c, yerr=stds_c,
                            fmt="o-", color=CLASS_COLORS[c],
                            markersize=5, linewidth=1.2,
                            capsize=3, capthick=0.8)
                for d, m, n in zip(days_v, means_c, counts_v):
                    if n < MIN_N:
                        overlay_hollow_marker(ax, d, m, CLASS_COLORS[c], markersize=7)

        ax.set_title(gname, fontsize=9)
        ax.set_xticks(days_x)
        ax.set_xticklabels([str(d) for d in days_x])
        ax.set_xlabel("Day")
        if idx % 2 == 0:
            ax.set_ylabel(YLABEL)

    class_handles = [
        mlines.Line2D([], [], color=CLASS_COLORS[c], marker="o",
                      linestyle="-", markersize=5, label=str(c))
        for c in CLASSES
    ]
    fig.legend(handles=class_handles + [_hollow_handle()], title="Class", ncol=9,
               loc="lower center", bbox_to_anchor=(0.5, -0.03), fontsize=7)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    save_figure(fig, "fig3B_byday.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1B(); fig2B(); fig3B(); fig3B_byday()
