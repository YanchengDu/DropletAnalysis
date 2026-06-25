# -*- coding: utf-8 -*-
"""
metric_a_class_composition.py
Metric A: class composition (fraction of each class by droplet count).
Classes with N < MIN_N are flagged with '* N=X' above the bar (bar plots)
or hollow markers (line plots).

Outputs:
  fig1A_page_comparison.png
  fig2A_crossgroup_day4.png
  fig3A_timecourse.png
  fig3A_byday.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from utils import (
    CLASSES, CLASS_COLORS, FIG1_CONDITIONS, FIG2_LAYOUT, FIG3_GROUPS,
    MIN_N, load_condition, class_fractions, class_counts,
    mark_low_n_bar, overlay_hollow_marker, apply_style, save_figure,
)

DAY_MAP = {"Day 0": 0, "Day 1": 1, "Day 4": 4}
ALPHAS  = [1.0, 0.55, 0.25]


# -- Figure 1A ----------------------------------------------------------------

def fig1A():
    apply_style()
    labels   = list(FIG1_CONDITIONS.keys())
    folders  = list(FIG1_CONDITIONS.values())
    datasets = [load_condition(f) for f in folders]
    fracs    = [class_fractions(df) for df in datasets]
    cnts     = [class_counts(df) for df in datasets]
    totals   = [len(df) for df in datasets]

    n_cond  = len(labels)
    x       = np.arange(16)
    width   = 0.25
    offsets = np.linspace(-(n_cond - 1) / 2, (n_cond - 1) / 2, n_cond) * width

    fig, ax = plt.subplots(figsize=(13, 4.5))

    for i, frac in enumerate(fracs):
        for c in CLASSES:
            ax.bar(c + offsets[i], frac[c], width=width * 0.92,
                   color=CLASS_COLORS[c], alpha=ALPHAS[i],
                   edgecolor="#333", linewidth=0.5)
            n = cnts[i][c]
            if 0 < n < MIN_N:
                mark_low_n_bar(ax, c + offsets[i], frac[c], n, rotate=True)

    cond_patches = [
        mpatches.Patch(facecolor="#888", alpha=ALPHAS[i],
                       edgecolor="#333", label="{} (N={:,})".format(lbl, tot))
        for i, (lbl, tot) in enumerate(zip(labels, totals))
    ]
    low_n_patch = mpatches.Patch(facecolor="none", edgecolor="#cc0000",
                                  label="* N<{}: unreliable".format(MIN_N))
    ax.legend(handles=cond_patches + [low_n_patch], fontsize=8, framealpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in CLASSES])
    ax.set_xlabel("Class")
    ax.set_ylabel("Fraction of droplets")
    ax.set_title("Figure 1A -- Class composition: PAGE (day 3) vs heated conditions (day 4)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(-0.6, 15.6)
    fig.tight_layout()
    save_figure(fig, "fig1A_page_comparison.png")
    plt.close(fig)


# -- Figure 2A ----------------------------------------------------------------

def fig2A():
    apply_style()
    x     = np.arange(16)
    width = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey=True)
    fig.suptitle("Figure 2A -- Class composition: cross-group comparison (day 4)", fontsize=11)

    for row in range(2):
        for col in range(2):
            lbl, folder = FIG2_LAYOUT[row][col]
            df   = load_condition(folder)
            frac = class_fractions(df)
            cnts = class_counts(df)
            ax   = axes[row][col]

            ax.bar(x, frac, width=width,
                   color=[CLASS_COLORS[c] for c in CLASSES],
                   edgecolor="#333", linewidth=0.4)
            for c in CLASSES:
                n = cnts[c]
                if 0 < n < MIN_N:
                    mark_low_n_bar(ax, c, frac[c], n, rotate=False)
            ax.set_title("{}\n(N={:,})".format(lbl.replace("\n", ", "), len(df)), fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in CLASSES], fontsize=7)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            if col == 0:
                ax.set_ylabel("Fraction of droplets")
            if row == 1:
                ax.set_xlabel("Class")

    fig.text(0.01, 0.01, "* N<{}: unreliable".format(MIN_N),
             fontsize=7, color="#cc0000", va="bottom")
    fig.tight_layout()
    save_figure(fig, "fig2A_crossgroup_day4.png")
    plt.close(fig)


# -- Figure 3A (x = class) ----------------------------------------------------

def fig3A():
    apply_style()
    x     = np.arange(16)
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle("Figure 3A -- Class composition over time (per group)", fontsize=11)

    for idx, (gname, timepoints) in enumerate(FIG3_GROUPS.items()):
        ax          = axes[idx // 2][idx % 2]
        n_t         = len(timepoints)
        offsets     = np.linspace(-(n_t - 1) / 2, (n_t - 1) / 2, n_t) * width
        tp_datasets = [(day_lbl, load_condition(folder)) for day_lbl, folder in timepoints]

        for i, (day_lbl, df) in enumerate(tp_datasets):
            frac = class_fractions(df)
            cnts = class_counts(df)
            for c in CLASSES:
                ax.bar(c + offsets[i], frac[c], width=width * 0.88,
                       color=CLASS_COLORS[c], alpha=ALPHAS[i],
                       edgecolor="#333", linewidth=0.5)
                n = cnts[c]
                if 0 < n < MIN_N:
                    mark_low_n_bar(ax, c + offsets[i], frac[c], n, rotate=True)

        day_patches = [
            mpatches.Patch(facecolor="#888", alpha=ALPHAS[i],
                           edgecolor="#333", label="{} (N={:,})".format(day_lbl, len(df)))
            for i, (day_lbl, df) in enumerate(tp_datasets)
        ]
        low_n_patch = mpatches.Patch(facecolor="none", edgecolor="#cc0000",
                                      label="* N<{}".format(MIN_N))
        ax.legend(handles=day_patches + [low_n_patch], fontsize=7, framealpha=0.9)
        ax.set_title(gname, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in CLASSES], fontsize=7)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        if idx % 2 == 0:
            ax.set_ylabel("Fraction of droplets")
        if idx >= 2:
            ax.set_xlabel("Class")

    fig.tight_layout()
    save_figure(fig, "fig3A_timecourse.png")
    plt.close(fig)


# -- Figure 3A (x = day) ------------------------------------------------------

def fig3A_byday():
    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharey=True)
    fig.suptitle("Figure 3A (by day) -- Class composition vs time (per group)", fontsize=11)

    for idx, (gname, timepoints) in enumerate(FIG3_GROUPS.items()):
        ax          = axes[idx // 2][idx % 2]
        days_x      = [DAY_MAP[day_lbl] for day_lbl, _ in timepoints]
        tp_datasets = [load_condition(folder) for _, folder in timepoints]

        for c in CLASSES:
            fracs_c  = [class_fractions(df)[c] for df in tp_datasets]
            counts_c = [class_counts(df)[c] for df in tp_datasets]
            if any(f > 0 for f in fracs_c):
                ax.plot(days_x, fracs_c, "o-", color=CLASS_COLORS[c],
                        markersize=5, linewidth=1.2)
                for d, f, n in zip(days_x, fracs_c, counts_c):
                    if 0 < n < MIN_N:
                        overlay_hollow_marker(ax, d, f, CLASS_COLORS[c], markersize=7)

        ax.set_title(gname, fontsize=9)
        ax.set_xticks(days_x)
        ax.set_xticklabels([str(d) for d in days_x])
        ax.set_xlabel("Day")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        if idx % 2 == 0:
            ax.set_ylabel("Fraction of droplets")

    class_handles = [
        mlines.Line2D([], [], color=CLASS_COLORS[c], marker="o",
                      linestyle="-", markersize=5, label=str(c))
        for c in CLASSES
    ]
    hollow_handle = mlines.Line2D([], [], marker="o", markerfacecolor="none",
                                   markeredgecolor="#555", linestyle="none",
                                   markersize=7, label="N<{}: unreliable".format(MIN_N))
    fig.legend(handles=class_handles + [hollow_handle], title="Class", ncol=9,
               loc="lower center", bbox_to_anchor=(0.5, -0.03), fontsize=7)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    save_figure(fig, "fig3A_byday.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1A(); fig2A(); fig3A(); fig3A_byday()
