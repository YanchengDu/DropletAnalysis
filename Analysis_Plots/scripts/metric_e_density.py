# -*- coding: utf-8 -*-
"""
metric_e_density.py
Metric E: droplet density per class (droplets / um^2).
20x and 40x FOVs are pooled: each FOV is normalised by its own physical area
  20x FOV ~ 102073 um^2, 40x FOV ~ 25518 um^2
then all FOVs contribute equally to mean +/- std.
PAGE condition has 40x only -- handled gracefully (only 40x FOVs used).

Outputs (PNG + SVG):
  fig1E_page_comparison.png/svg
  fig2E_crossgroup_day4.png/svg
  fig3E_timecourse.png/svg
  fig3E_byday.png/svg
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from utils import (
    CLASSES, CLASS_COLORS, FIG1_CONDITIONS, FIG2_LAYOUT, FIG3_GROUPS,
    MIN_N, load_condition, class_density_combined, class_counts,
    overlay_hollow_marker, apply_style, save_figure,
)

DAY_MAP = {"Day 0": 0, "Day 1": 1, "Day 4": 4}
MARKERS = ["o", "s", "^"]
YLABEL  = "Density (droplets / um^2)"


def _hollow_handle():
    return mlines.Line2D([], [], marker="o", markerfacecolor="none",
                         markeredgecolor="#555", linestyle="none",
                         markersize=6, label="N<{}: unreliable".format(MIN_N))


# -- Figure 1E ----------------------------------------------------------------

def fig1E():
    apply_style()
    labels   = list(FIG1_CONDITIONS.keys())
    folders  = list(FIG1_CONDITIONS.values())
    datasets = [load_condition(f) for f in folders]

    offsets = np.linspace(-0.2, 0.2, len(labels))
    fig, ax = plt.subplots(figsize=(13, 4.5))

    for i, (lbl, df) in enumerate(zip(labels, datasets)):
        means, stds, n_fovs = class_density_combined(df)
        cnts                = class_counts(df)
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
                      markersize=8,
                      label="{} (n={} FOVs)".format(
                          lbl, class_density_combined(datasets[i])[2]))
        for i, lbl in enumerate(labels)
    ]
    ax.legend(handles=cond_handles + [_hollow_handle()], fontsize=8, framealpha=0.9)
    ax.set_xticks(range(16))
    ax.set_xticklabels([str(c) for c in CLASSES])
    ax.set_xlabel("Class")
    ax.set_ylabel(YLABEL)
    ax.set_title("Figure 1E -- Droplet density: PAGE (day 3) vs heated conditions (day 4)\n"
                 "(20x+40x FOVs pooled, each normalised by its own FOV area)")
    ax.set_xlim(-0.6, 15.6)
    fig.tight_layout()
    save_figure(fig, "fig1E_page_comparison.png")
    plt.close(fig)


# -- Figure 2E ----------------------------------------------------------------

def fig2E():
    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey=True)
    fig.suptitle("Figure 2E -- Droplet density per class: cross-group comparison (day 4)\n"
                 "(20x+40x FOVs pooled)", fontsize=11)

    for row in range(2):
        for col in range(2):
            lbl, folder = FIG2_LAYOUT[row][col]
            df          = load_condition(folder)
            means, stds, n_fovs = class_density_combined(df)
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

            ax.set_title("{}\n(n={} FOVs)".format(
                lbl.replace("\n", ", "), n_fovs), fontsize=9)
            ax.set_xticks(range(16))
            ax.set_xticklabels([str(c) for c in CLASSES], fontsize=7)
            if col == 0:
                ax.set_ylabel(YLABEL)
            if row == 1:
                ax.set_xlabel("Class")

    fig.text(0.01, 0.01, "open circle = N<{}: unreliable".format(MIN_N),
             fontsize=7, color="#cc0000", va="bottom")
    fig.tight_layout()
    save_figure(fig, "fig2E_crossgroup_day4.png")
    plt.close(fig)


# -- Figure 3E (x = class) ----------------------------------------------------

def fig3E():
    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle("Figure 3E -- Droplet density per class over time (per group)\n"
                 "(20x+40x FOVs pooled)", fontsize=11)

    for idx, (gname, timepoints) in enumerate(FIG3_GROUPS.items()):
        ax          = axes[idx // 2][idx % 2]
        n_t         = len(timepoints)
        offsets     = np.linspace(-0.2, 0.2, n_t)
        tp_datasets = [(day_lbl, load_condition(folder)) for day_lbl, folder in timepoints]

        day_handles = []
        for i, (day_lbl, df) in enumerate(tp_datasets):
            means, stds, n_fovs = class_density_combined(df)
            cnts                = class_counts(df)
            for c in CLASSES:
                if means[c] > 0:
                    ax.errorbar(c + offsets[i], means[c], yerr=stds[c],
                                fmt=MARKERS[i % len(MARKERS)], color=CLASS_COLORS[c],
                                markeredgecolor="#333", markeredgewidth=0.5,
                                markersize=5, capsize=3, capthick=0.8, linewidth=0.8)
                    if cnts[c] < MIN_N:
                        overlay_hollow_marker(ax, c + offsets[i], means[c],
                                              CLASS_COLORS[c], markersize=7)
            day_handles.append(
                mlines.Line2D([], [], marker=MARKERS[i % len(MARKERS)],
                              color="#555", linestyle="none", markersize=7,
                              label="{} (n={} FOVs)".format(day_lbl, n_fovs)))

        ax.legend(handles=day_handles + [_hollow_handle()], fontsize=7, framealpha=0.9)
        ax.set_title(gname, fontsize=9)
        ax.set_xticks(range(16))
        ax.set_xticklabels([str(c) for c in CLASSES], fontsize=7)
        if idx % 2 == 0:
            ax.set_ylabel(YLABEL)
        if idx >= 2:
            ax.set_xlabel("Class")

    fig.tight_layout()
    save_figure(fig, "fig3E_timecourse.png")
    plt.close(fig)


# -- Figure 3E (x = day) ------------------------------------------------------

def fig3E_byday():
    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharey=True)
    fig.suptitle("Figure 3E (by day) -- Droplet density vs time (per group)\n"
                 "(20x+40x FOVs pooled)", fontsize=11)

    for idx, (gname, timepoints) in enumerate(FIG3_GROUPS.items()):
        ax          = axes[idx // 2][idx % 2]
        days_x      = [DAY_MAP[day_lbl] for day_lbl, _ in timepoints]
        tp_datasets = [load_condition(folder) for _, folder in timepoints]

        for c in CLASSES:
            means_c, stds_c, days_v, counts_v = [], [], [], []
            for d, df in zip(days_x, tp_datasets):
                means, stds, _ = class_density_combined(df)
                cnts           = class_counts(df)
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
    save_figure(fig, "fig3E_byday.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1E(); fig2E(); fig3E(); fig3E_byday()
