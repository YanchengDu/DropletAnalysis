"""
droplet_ui.py
Interactive analysis UI for DNA nanostar condensate microscopy images.

Usage:
    from droplet_ui import launch_ui, show_analysis
    S = launch_ui()
    show_analysis(S['df'], S['pixel_size'])
"""

import glob
import os
import uuid
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, HTML
import matplotlib.pyplot as plt

from droplet_pipeline import (
    load_image, normalize_image, detect_droplets,
    measure_droplet_intensities, estimate_background,
    classify_channels, detect_bright_red, build_dataframe,
    make_rgb, HEX_COLORS,
)

HEX = HEX_COLORS


# ─────────────────────────────────────────────────────────────────────────────
def show_analysis(df, pixel_size_um=0.312, img_shape=None, save_prefix=None):
    """Display a full analysis dashboard:
    counts, radius/volume distributions, intensity heatmap, unreliable-channel
    fraction, size by class, spatial map, per-channel intensity by class, and
    nearest-neighbour distances.

    Parameters
    ----------
    df           : DataFrame returned by launch_ui() → S['df']
    pixel_size_um: µm per pixel (S['pixel_size'])
    img_shape    : optional (n_ch, H, W) from S['norm'].shape — used to set
                   spatial-map axes to true image extent
    """
    if df is None or len(df) == 0:
        print("No data — run the pipeline first.")
        return
    df = df.copy()  # avoid mutating caller's dataframe

    from scipy.spatial import KDTree

    # ── Save helper ───────────────────────────────────────────────────────────
    _save_n = [0]
    def _maybe_save(fig):
        if save_prefix:
            _save_n[0] += 1
            fig.savefig(f"{save_prefix}_{_save_n[0]:02d}.png", dpi=150,
                        bbox_inches="tight")

    counts  = df.groupby("code_int").size().reindex(range(16), fill_value=0)
    # Exclude -2 (undecided) from class-based plots; track separately
    n_undecided = int((df.code_int == -2).sum())
    present = sorted(c for c in df.code_int.unique() if c >= 0)
    ch_cols = ["intensity_ch1", "intensity_ch2", "intensity_ch3", "intensity_ch4"]
    bin_cols = ["binary_ch1", "binary_ch2", "binary_ch3", "binary_ch4"]

    # ── 1. Overview: counts / total volume / radius / volume by class ────────
    total_vol = df.groupby("code_int")["volume_um3"].sum().reindex(range(16), fill_value=0)
    _bp_kw = dict(patch_artist=True,
                  medianprops=dict(color="white", linewidth=1.5),
                  whiskerprops=dict(color="#888"), capprops=dict(color="#888"),
                  flierprops=dict(marker=".", color="#888", markersize=3))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=150)

    # [0,0] counts per class
    ax = axes[0, 0]
    ax.bar(counts.index, counts.values,
           color=[HEX_COLORS[i] for i in range(16)],
           edgecolor="#555", linewidth=0.6)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Droplets per class")
    ax.set_xticks(range(16))

    # [0,1] total volume per class
    ax = axes[0, 1]
    ax.bar(total_vol.index, total_vol.values,
           color=[HEX_COLORS[i] for i in range(16)],
           edgecolor="#555", linewidth=0.6)
    for cls in present:
        v = total_vol[cls]
        if v > 0:
            ax.text(cls, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Class")
    ax.set_ylabel("Total volume (µm³)")
    ax.set_title("Total condensate volume per class")
    ax.set_xticks(range(16))

    # [1,0] radius by class
    ax = axes[1, 0]
    data_r = [df[df.code_int == c].radius_um.values for c in present]
    bp = ax.boxplot(data_r, positions=present, **_bp_kw)
    for patch, cls in zip(bp["boxes"], present):
        patch.set_facecolor(HEX_COLORS[cls]); patch.set_alpha(0.75)
    ax.set_xticks(range(16))
    ax.set_xticklabels([str(i) for i in range(16)], fontsize=8)
    ax.set_xlim(-0.5, 15.5)
    ax.set_xlabel("Class")
    ax.set_ylabel("Radius (µm)")
    ax.set_title("Radius by class")

    # [1,1] volume by class
    ax = axes[1, 1]
    data_v = [df[df.code_int == c].volume_um3.values for c in present]
    bp = ax.boxplot(data_v, positions=present, **_bp_kw)
    for patch, cls in zip(bp["boxes"], present):
        patch.set_facecolor(HEX_COLORS[cls]); patch.set_alpha(0.75)
    ax.set_xticks(range(16))
    ax.set_xticklabels([str(i) for i in range(16)], fontsize=8)
    ax.set_xlim(-0.5, 15.5)
    ax.set_xlabel("Class")
    ax.set_ylabel("Volume (µm³)")
    ax.set_title("Volume by class")

    plt.tight_layout()
    _maybe_save(fig)
    plt.show()

    # ── 2. Intensity heatmap + unreliable-channel fraction ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Heatmap: mean raw intensity per class × channel, column-normalised
    heat = df.groupby("code_int")[ch_cols].mean().reindex(range(16))
    col_min, col_max = heat.min(), heat.max()
    heat_norm = (heat - col_min) / (col_max - col_min + 1e-9)

    ax = axes[0]
    im = ax.imshow(heat_norm.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["ch1\n(red)", "ch2\n(yellow)", "ch3\n(cyan)", "ch4\n(blue)"])
    ax.set_yticks(range(16))
    ax.set_yticklabels([str(i) for i in range(16)])
    ax.set_ylabel("Class")
    ax.set_title("Mean intensity per class × channel\n(column-normalised)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Grey out empty classes
    for cls in range(16):
        if counts[cls] == 0:
            ax.axhspan(cls - 0.5, cls + 0.5, color="grey", alpha=0.4)
            ax.text(1.5, cls, "n=0", ha="center", va="center",
                    color="white", fontsize=7)

    # Unreliable fraction
    n_unrel = (df[bin_cols] == -1).sum(axis=1)
    unrel_counts = n_unrel.value_counts().reindex([0, 1, 2, 3, 4], fill_value=0)
    ax = axes[1]
    bar_colors = ["#4a90d9", "#e8a020", "#d04040", "#a020d0", "#20a040"]
    bars = ax.bar(unrel_counts.index, unrel_counts.values / len(df) * 100,
                  color=bar_colors, edgecolor="#555", linewidth=0.6)
    for bar, val in zip(bars, unrel_counts.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Unreliable channels per droplet")
    ax.set_ylabel("Fraction (%)")
    ax.set_title("Unreliable channel fraction")
    ax.set_xticks([0, 1, 2, 3, 4])

    plt.tight_layout()
    _maybe_save(fig)
    plt.show()

    # ── 3. Size by class + spatial map ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    ax = axes[0]
    vp_present = [c for c in present if (df.code_int == c).sum() >= 2]
    data_by_class = [df[df.code_int == c].radius_um.values for c in vp_present]
    if data_by_class:
        parts = ax.violinplot(data_by_class, positions=vp_present,
                              showmedians=True, showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(HEX_COLORS[vp_present[i]])
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("white")
    ax.set_xticks(range(16))
    ax.set_xticklabels([str(i) for i in range(16)], fontsize=8)
    ax.set_xlim(-0.5, 15.5)
    ax.set_xlabel("Class")
    ax.set_ylabel("Radius (µm)")
    ax.set_title("Size distribution by class")

    ax = axes[1]
    if img_shape is not None:
        _, H, W = img_shape
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)   # y-axis flipped to match image coords
    else:
        m = 20
        ax.set_xlim(df.centroid_x.min() - m, df.centroid_x.max() + m)
        ax.set_ylim(df.centroid_y.max() + m, df.centroid_y.min() - m)
    for cls in present:
        sub = df[df.code_int == cls]
        ax.scatter(sub.centroid_x, sub.centroid_y,
                   c=HEX_COLORS[cls], s=35, label=str(cls),
                   edgecolors="white", linewidths=0.3, alpha=0.85)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Spatial map of droplet classes")
    ax.legend(title="class", markerscale=0.8, fontsize=7,
              loc="upper right", ncol=2, framealpha=0.7)

    plt.tight_layout()
    _maybe_save(fig)
    plt.show()

    # ── 4. Per-channel intensity by class (violin) ────────────────────────────
    ch_labels = ["ch1 (red)", "ch2 (yellow)", "ch3 (cyan)", "ch4 (blue)"]
    ch_colors = ["#ff4444", "#cccc22", "#22cccc", "#4444ff"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=150)
    for i, (col, lbl, clr) in enumerate(zip(ch_cols, ch_labels, ch_colors)):
        ax = axes[i // 2][i % 2]
        data_ch = [df[df.code_int == c][col].values for c in vp_present]
        if data_ch:
            parts = ax.violinplot(data_ch, positions=vp_present,
                                  showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(clr)
                pc.set_alpha(0.65)
            parts["cmedians"].set_color("black")
        ax.set_xticks(range(16))
        ax.set_xticklabels([str(i) for i in range(16)], fontsize=8)
        ax.set_xlim(-0.5, 15.5)
        ax.set_xlabel("Class")
        ax.set_ylabel("Raw intensity")
        ax.set_title(f"Intensity — {lbl}")

    plt.tight_layout()
    _maybe_save(fig)
    plt.show()

    # ── 5. Pairwise surface-to-surface distance analysis (analytical null) ────
    # For each class-i droplet, find the nearest class-j droplet and record the
    # surface-to-surface distance (d_centre - r_i - r_j).
    #
    # Null model (no permutations needed): for each target class j, compute the
    # same distance from ALL non-class-j droplets.  global_mean[j] and
    # global_std[j] capture the baseline under random class assignment,
    # accounting for class-j abundance and its spatial distribution.
    #
    # z_ij = (global_mean[j] - obs_d[i,j]) / (global_std[j] / sqrt(n_i))
    # Positive z = class i is CLOSER to class j than the average droplet
    #              = ATTRACTION.
    # Significance naturally shrinks when n_i is small (sqrt(n_i) denominator),
    # suppressing unreliable estimates without a hard threshold.
    coords  = df[["centroid_y", "centroid_x"]].values.astype(float)
    classes = df.code_int.values
    radii   = df.radius_um.values
    n_cls   = len(present)
    labels  = [str(c) for c in present]

    if n_cls >= 2 and len(coords) >= 4:
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize

        # -- Build one KDTree per class ----------------------------------------
        trees_j = {}
        radii_j = {}
        for jj, cj in enumerate(present):
            m_j = (classes == cj)
            if m_j.sum() > 0:
                trees_j[jj] = KDTree(coords[m_j])
                radii_j[jj] = radii[m_j]

        # -- Observed mean distance matrix obs_d[i, j] ------------------------
        # obs_d[i, j] = mean surface-to-surface distance from each class-i
        # droplet to its nearest class-j droplet.
        obs_d = np.full((n_cls, n_cls), np.nan)
        for ii, ci in enumerate(present):
            m_i = (classes == ci)
            n_i = m_i.sum()
            if n_i == 0:
                continue
            c_i = coords[m_i]
            r_i = radii[m_i]
            for jj in range(n_cls):
                if jj not in trees_j:
                    continue
                same = (ii == jj)
                if same and n_i < 2:
                    continue
                k = 2 if same else 1
                d_c, nn = trees_j[jj].query(c_i, k=k)
                if k == 1:
                    d_s = d_c - r_i - radii_j[jj][nn]
                else:
                    d_s = d_c[:, 1] - r_i - radii_j[jj][nn[:, 1]]
                obs_d[ii, jj] = d_s.mean()

        # -- Analytical null: all non-class-j droplets to nearest class-j -----
        # This sets the expected baseline for each target class j, accounting
        # for both j's abundance and its spatial distribution in the FOV.
        global_mean = np.full(n_cls, np.nan)
        global_std  = np.full(n_cls, np.nan)
        for jj, cj in enumerate(present):
            if jj not in trees_j:
                continue
            mask_other = (classes != cj)
            if mask_other.sum() < 2:
                continue
            c_o = coords[mask_other]
            r_o = radii[mask_other]
            d_c, nn = trees_j[jj].query(c_o, k=1)
            d_s = d_c - r_o - radii_j[jj][nn]
            global_mean[jj] = d_s.mean()
            global_std[jj]  = d_s.std()

        # -- Z-score and distance ratio ----------------------------------------
        n_src  = np.array([counts[c] for c in present], dtype=float)
        se_mat = global_std[np.newaxis, :] / np.sqrt(n_src[:, np.newaxis])
        zscore_mat = np.divide(
            global_mean[np.newaxis, :] - obs_d, se_mat,
            out=np.full_like(obs_d, np.nan),
            where=(se_mat > 0) & (~np.isnan(obs_d))
        )
        ratio_mat = np.divide(
            obs_d, global_mean[np.newaxis, :],
            out=np.full_like(obs_d, np.nan),
            where=(~np.isnan(global_mean[np.newaxis, :])) & (global_mean[np.newaxis, :] > 0)
        )
        # Diagonal uses a different null (intra-class); suppress z/ratio there
        np.fill_diagonal(zscore_mat, np.nan)
        np.fill_diagonal(ratio_mat,  np.nan)

        # -- Attraction-emphasis colormaps ------------------------------------
        # Avoidance: pale steel blue  |  zero / ratio=1: white
        # Attraction: pale orange -> vivid orange -> deep crimson  (eye-catching)
        # The vivid warm colours draw the eye to attraction, which is what we
        # care about; avoidance fades to a quiet pale blue.
        _cmap_z = LinearSegmentedColormap.from_list(
            '_attract',
            [(0.00, '#9ecae1'),   # avoidance: pale blue
             (0.38, '#deebf7'),   # mild avoidance: very pale blue
             (0.50, '#ffffff'),   # zero: white
             (0.64, '#fdd0a2'),   # mild attraction: pale orange
             (0.80, '#f16913'),   # moderate: vivid orange
             (0.91, '#cb181d'),   # strong: red
             (1.00, '#67000d')]   # very strong: dark crimson
        )
        _cmap_r = _cmap_z.reversed()  # ratio < 1 = attraction = low value = red

        # -- Annotated heatmap helper -----------------------------------------
        row_labels = ["{0}  (n={1:d})".format(lbl, int(n))
                      for lbl, n in zip(labels, n_src)]

        def _annotated_heatmap(mat, cmap, norm, title, fmt, cb_label):
            sz = max(6, n_cls * 0.55)
            fig, ax = plt.subplots(figsize=(sz + 2.5, sz), dpi=150)
            im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
            ax.set_xticks(range(n_cls)); ax.set_xticklabels(labels, fontsize=9)
            ax.set_yticks(range(n_cls)); ax.set_yticklabels(row_labels, fontsize=8)
            ax.set_xlabel("Target class", fontsize=10)
            ax.set_ylabel("Source class  (n = droplets in that class)", fontsize=10)
            ax.set_title(title, fontsize=11)
            plt.colorbar(im, ax=ax, label=cb_label, fraction=0.046, pad=0.04)
            cmap_fn = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
            for i in range(n_cls):
                for j in range(n_cls):
                    v = mat[i, j]
                    if np.isnan(v):
                        ax.text(j, i, "-", ha="center", va="center",
                                fontsize=8, color="gray")
                        continue
                    r2, g2, b2, _ = cmap_fn(norm(v))
                    lum = 0.2126 * r2 + 0.7152 * g2 + 0.0722 * b2
                    ax.text(j, i, fmt(v), ha="center", va="center",
                            fontsize=8,
                            color="white" if lum < 0.45 else "black")
            _maybe_save(fig)
            plt.tight_layout()
            plt.show()

        # Heatmap 1: raw mean surface-to-surface distance (um)
        _annotated_heatmap(
            obs_d, "viridis_r",
            Normalize(vmin=np.nanmin(obs_d), vmax=np.nanmax(obs_d)),
            "Mean surface-to-surface distance: class i -> nearest class j  (um)\n"
            "Diagonal = nearest same-class neighbour",
            lambda v: "{0:.2f}".format(v), "distance (um)",
        )

        # Heatmap 2: z-score with asymmetric norm (attraction gets full range,
        # avoidance compressed to half) so vivid reds stand out visually
        z_lim = max(2.0, np.nanpercentile(
            np.abs(zscore_mat[~np.isnan(zscore_mat)]), 97))
        norm_z = TwoSlopeNorm(vcenter=0.0, vmin=-z_lim * 0.5, vmax=z_lim)
        _annotated_heatmap(
            zscore_mat, _cmap_z, norm_z,
            "Interaction z-score  |  red = ATTRACTION   blue = avoidance\n"
            "null = all non-j droplets to nearest class j;  "
            "significance scales with sqrt(n_i)",
            lambda v: "{0:.1f}".format(v), "z-score",
        )

        # Heatmap 3: Benjamini-Hochberg FDR-corrected significance
        # One-tailed p-values for attraction: p_ij = P(Z > z_ij) = 1 - Phi(z_ij)
        # Only off-diagonal pairs are tested (diagonal null is different).
        # BH correction controls FDR at q=0.05 across all m off-diagonal pairs.
        # Display as -log10(BH-adjusted p): higher = more significant.
        # Cells with adj_p <= 0.05 are marked with * (FDR-significant attraction).
        from scipy.stats import norm as _norm
        FDR_Q = 0.05

        # Compute one-tailed p-values (attraction direction only)
        p_mat = np.full_like(zscore_mat, np.nan)
        valid_mask = ~np.isnan(zscore_mat)
        p_mat[valid_mask] = 1.0 - _norm.cdf(zscore_mat[valid_mask])

        # Collect off-diagonal valid pairs
        off_diag = valid_mask & ~np.eye(n_cls, dtype=bool)
        ii_idx, jj_idx = np.where(off_diag)
        p_vals = p_mat[ii_idx, jj_idx]
        m = len(p_vals)

        adj_p_mat = np.full_like(p_mat, np.nan)
        if m > 0:
            # BH step-up: sort, compute k/m * q threshold, take running min from right
            sort_order = np.argsort(p_vals)
            sorted_p   = p_vals[sort_order]
            ranks      = np.arange(1, m + 1, dtype=float)
            bh_adj     = np.clip(sorted_p * m / ranks, 0.0, 1.0)
            # Enforce monotonicity (adjusted p cannot decrease as rank increases)
            bh_adj     = np.minimum.accumulate(bh_adj[::-1])[::-1]
            # Put back into matrix (unsorted)
            adj_p_unsorted              = np.empty(m)
            adj_p_unsorted[sort_order]  = bh_adj
            adj_p_mat[ii_idx, jj_idx]   = adj_p_unsorted

        # -log10 transform: 0 = no signal, >1.3 = adj_p < 0.05
        logp_mat  = -np.log10(np.clip(adj_p_mat, 1e-10, 1.0))
        sig_mask  = (adj_p_mat <= FDR_Q)   # cells to annotate with *

        # Sequential colormap: white (not significant) -> vivid red (significant)
        _cmap_bh = LinearSegmentedColormap.from_list(
            '_bh',
            [(0.00, '#ffffff'),   # p=1: white
             (0.50, '#fdd0a2'),   # borderline: pale orange
             (0.70, '#f16913'),   # moderate: vivid orange
             (0.85, '#cb181d'),   # strong: red
             (1.00, '#67000d')]   # very strong: dark crimson
        )
        lp_max = max(1.5, np.nanpercentile(logp_mat[~np.isnan(logp_mat)], 99))
        norm_bh = Normalize(vmin=0, vmax=lp_max)

        # Custom heatmap for BH (annotates significant cells with *)
        sz = max(6, n_cls * 0.55)
        fig, ax = plt.subplots(figsize=(sz + 2.5, sz), dpi=150)
        im = ax.imshow(logp_mat, cmap=_cmap_bh, norm=norm_bh, aspect="auto")
        ax.set_xticks(range(n_cls)); ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks(range(n_cls)); ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xlabel("Target class", fontsize=10)
        ax.set_ylabel("Source class  (n = droplets in that class)", fontsize=10)
        ax.set_title(
            "Attraction significance  (-log10 BH-adjusted p,  FDR q=0.05)\n"
            "* = FDR-significant attraction;  one-tailed test (attraction only)",
            fontsize=11)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("-log10(BH adj. p)")
        cb.ax.axhline(-np.log10(FDR_Q), color='black', lw=1.2, linestyle='--')
        cb.ax.text(2.6, -np.log10(FDR_Q), "q=0.05", va='center', fontsize=7)
        for i in range(n_cls):
            for j in range(n_cls):
                v = logp_mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, "-", ha="center", va="center",
                            fontsize=8, color="gray")
                    continue
                r2, g2, b2, _ = _cmap_bh(norm_bh(v))
                lum = 0.2126 * r2 + 0.7152 * g2 + 0.0722 * b2
                txt_col = "white" if lum < 0.45 else "black"
                label_txt = "* {0:.2f}".format(v) if sig_mask[i, j] else "{0:.2f}".format(v)
                ax.text(j, i, label_txt, ha="center", va="center",
                        fontsize=7, color=txt_col,
                        fontweight="bold" if sig_mask[i, j] else "normal")
        _maybe_save(fig)
        plt.tight_layout()
        plt.show()
    else:
        print("Need >=2 classes and >=4 droplets for distance analysis.")

    # ── 6. System quality metrics ─────────────────────────────────────────────
    # Category A — pipeline / imaging quality
    #   Fraction undecided, fraction with ≥1 unreliable channel, mean classifier
    #   confidence, and intra-class intensity CV (how consistent are raw
    #   intensities within the same barcode class?).
    # Category B — physical / experimental quality
    #   Global radius CV (monodispersity), per-class radius CV, class coverage
    #   (n_present / 16), and class composition entropy normalised to log₂16
    #   (1.0 = all 16 classes equally abundant = ideal mixing).
    n_total      = len(df)
    n_any_unrel  = int((n_unrel > 0).sum())
    n_mult_unrel = int((n_unrel > 1).sum())

    # ── A: pipeline metrics ───────────────────────────────────────────────────
    frac_undecided = n_undecided / n_total
    frac_any_unrel = n_any_unrel / n_total

    has_conf  = "confidence" in df.columns
    mean_conf = float(df["confidence"].mean()) if has_conf else np.nan

    # Intra-class intensity CV: σ/μ of raw intensity per class × channel
    cv_mat = np.full((n_cls, 4), np.nan)
    for ii, ci in enumerate(present):
        sub = df[df.code_int == ci]
        if len(sub) >= 2:
            for jj, col in enumerate(ch_cols):
                mu = sub[col].mean()
                if mu > 0:
                    cv_mat[ii, jj] = sub[col].std() / mu
    mean_intensity_cv = float(np.nanmean(cv_mat)) if not np.all(np.isnan(cv_mat)) else np.nan

    # ── B: experimental metrics ───────────────────────────────────────────────
    r_mean    = df.radius_um.mean()
    radius_cv = df.radius_um.std() / r_mean if r_mean > 0 else np.nan

    per_cls_cv = {}
    for ci in present:
        sub = df[df.code_int == ci]
        if len(sub) >= 2 and sub.radius_um.mean() > 0:
            per_cls_cv[ci] = sub.radius_um.std() / sub.radius_um.mean()

    # Class composition entropy vs. ideal uniform across all 16 classes.
    # Normalise over classified droplets only (excludes undecided -2) so that
    # the 16 probabilities sum to 1 and entropy_norm is independent of the
    # undecided fraction (already captured by frac_undecided separately).
    n_classified = int(counts.sum())
    if n_classified > 0:
        p_all = np.array([counts[c] / n_classified for c in range(16)], dtype=float)
        h_raw = -np.sum(p_all * np.log2(np.where(p_all > 0, p_all, 1.0)))
        entropy_norm = h_raw / np.log2(16)   # 1.0 = all 16 equally abundant
    else:
        entropy_norm = np.nan
    class_coverage = len(present) / 16

    # ── Figure Q1: scalar metric gauge bars ───────────────────────────────────
    _gauge_rows = [
        # (label, value, zones[(lo, hi, color)], hint)
        ("Fraction undecided", frac_undecided,
         [(0.00, 0.05, "#2ecc71"), (0.05, 0.15, "#f39c12"), (0.15, 1.0, "#e74c3c")],
         "< 5% good"),
        ("Fraction ≥1 unrel. channel", frac_any_unrel,
         [(0.00, 0.10, "#2ecc71"), (0.10, 0.25, "#f39c12"), (0.25, 1.0, "#e74c3c")],
         "< 10% good"),
        ("Mean classifier confidence", mean_conf,
         [(0.00, 0.60, "#e74c3c"), (0.60, 0.80, "#f39c12"), (0.80, 1.0, "#2ecc71")],
         "> 0.8 good"),
        ("Global radius CV  (σ/μ)", radius_cv,
         [(0.00, 0.30, "#2ecc71"), (0.30, 0.50, "#f39c12"), (0.50, 1.0, "#e74c3c")],
         "< 0.3 excellent"),
        ("Class entropy  H / log₂16", entropy_norm,
         [(0.00, 0.50, "#e74c3c"), (0.50, 0.75, "#f39c12"), (0.75, 1.0, "#2ecc71")],
         "1.0 = all 16 equal"),
        ("Class coverage  (n/16)", class_coverage,
         [(0.00, 0.50, "#e74c3c"), (0.50, 0.75, "#f39c12"), (0.75, 1.0, "#2ecc71")],
         "1.0 = all 16 present"),
    ]
    if not has_conf:
        _gauge_rows = [r for r in _gauge_rows if "confidence" not in r[0].lower()]

    n_g = len(_gauge_rows)
    fig, ax = plt.subplots(figsize=(11, 0.75 * n_g + 1.2), dpi=150)
    ax.set_xlim(0, 1.22)
    ax.set_ylim(-0.7, n_g - 0.3)
    ax.set_axis_off()
    ax.set_title("System Quality Metrics", fontsize=12, fontweight="bold", pad=8)

    BAR_LO, BAR_HI = 0.40, 0.90
    BAR_W = BAR_HI - BAR_LO
    BH    = 0.38

    for row_i, (label, val, zones, desc) in enumerate(_gauge_rows):
        y = n_g - 1 - row_i
        for z_lo, z_hi, z_col in zones:
            x0 = BAR_LO + z_lo * BAR_W
            x1 = BAR_LO + z_hi * BAR_W
            ax.barh(y, x1 - x0, left=x0, height=BH,
                    color=z_col, alpha=0.20, edgecolor="none")
        ax.barh(y, BAR_W, left=BAR_LO, height=BH,
                color="none", edgecolor="#666", linewidth=0.8)
        if not np.isnan(val):
            val_c = float(np.clip(val, 0.0, 1.0))
            x_val = BAR_LO + val_c * BAR_W
            fill_col = "#888"
            for z_lo, z_hi, z_col in zones:
                if z_lo <= val_c < z_hi or (val_c == z_hi == 1.0):
                    fill_col = z_col; break
            ax.barh(y, x_val - BAR_LO, left=BAR_LO, height=BH,
                    color=fill_col, alpha=0.75, edgecolor="none")
            ax.plot([x_val, x_val], [y - BH / 2, y + BH / 2],
                    color="black", lw=1.5)
        ax.text(BAR_LO - 0.01, y, label, ha="right", va="center", fontsize=9)
        val_str = f"{val:.3f}" if not np.isnan(val) else "N/A"
        ax.text(BAR_HI + 0.012, y, val_str,
                ha="left", va="center", fontsize=9, fontweight="bold")
        ax.text(BAR_HI + 0.095, y, desc,
                ha="left", va="center", fontsize=7, color="#666")

    # dashed divider between category A and B rows
    n_a = 3 if has_conf else 2
    div_y = n_g - n_a - 0.5
    ax.axhline(div_y, xmin=0.01, xmax=0.99,
               color="#aaa", lw=0.8, linestyle="--")
    ax.text(0.01, div_y + 0.15, "A — pipeline quality",
            fontsize=7, color="#555", va="bottom")
    ax.text(0.01, div_y - 0.15, "B — experimental quality",
            fontsize=7, color="#555", va="top")

    plt.tight_layout()
    _maybe_save(fig)
    plt.show()

    # ── Figure Q2: intra-class raw intensity CV heatmap ───────────────────────
    if not np.all(np.isnan(cv_mat)):
        fig, ax = plt.subplots(
            figsize=(6, max(3.0, n_cls * 0.45 + 1.2)), dpi=150)
        im = ax.imshow(cv_mat, cmap="RdYlGn_r", vmin=0, vmax=0.8, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels(
            ["ch1\n(red)", "ch2\n(yellow)", "ch3\n(cyan)", "ch4\n(blue)"])
        ax.set_yticks(range(n_cls))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_ylabel("Class")
        ax.set_title(
            "Intra-class raw intensity CV  (σ/μ per class × channel)\n"
            "green = consistent,  red = high within-class spread",
            fontsize=10)
        plt.colorbar(im, ax=ax, label="CV (σ/μ)", fraction=0.046, pad=0.04)
        for i in range(n_cls):
            for j in range(4):
                v = cv_mat[i, j]
                txt = f"{v:.2f}" if not np.isnan(v) else "—"
                lum_v = 0.0 if np.isnan(v) else v / 0.8
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if lum_v > 0.6 else "black")
        plt.tight_layout()
        _maybe_save(fig)
        plt.show()

    # ── Figure Q3: per-class radius CV bar chart ──────────────────────────────
    if per_cls_cv:
        cls_list = sorted(per_cls_cv.keys())
        cv_vals  = [per_cls_cv[c] for c in cls_list]
        fig, ax  = plt.subplots(
            figsize=(max(6, len(cls_list) * 0.6 + 2), 3.5), dpi=150)
        bars = ax.bar(range(len(cls_list)), cv_vals,
                      color=[HEX_COLORS[c] for c in cls_list],
                      edgecolor="#555", linewidth=0.6)
        ax.axhline(0.30, color="#2ecc71", lw=1.0, linestyle="--",
                   label="CV = 0.30  (excellent)")
        ax.axhline(0.50, color="#e74c3c", lw=1.0, linestyle="--",
                   label="CV = 0.50  (poor)")
        for bar, val in zip(bars, cv_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(cls_list)))
        ax.set_xticklabels([str(c) for c in cls_list], fontsize=9)
        ax.set_xlabel("Class")
        ax.set_ylabel("Radius CV  (σ/μ)")
        ax.set_title("Per-class radius coefficient of variation")
        ax.legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        _maybe_save(fig)
        plt.show()

    # ── Summary text ──────────────────────────────────────────────────────────
    print(f"Total droplets : {n_total}")
    print(f"Undecided (−2) : {n_undecided}  ({100*frac_undecided:.1f}%)")
    print(f"Classes found  : {len(present)}/16  (coverage {100*class_coverage:.0f}%)")
    print(f"Bright red (0) : {df.bright_red.sum()}")
    print(f"Radius         : {df.radius_um.mean():.2f} ± {df.radius_um.std():.2f} µm  "
          f"(CV {radius_cv:.3f})")
    print(f"Volume         : {df.volume_um3.mean():.2f} ± {df.volume_um3.std():.2f} µm³")
    print(f"Any unrel. ch  : {n_any_unrel} ({100*frac_any_unrel:.1f}%)")
    print(f">1 unrel. ch   : {n_mult_unrel} ({100*n_mult_unrel/n_total:.1f}%)")
    print(f"Class entropy  : {entropy_norm:.3f}  (H/log₂16;  1.0 = all 16 equal)")
    if has_conf:
        print(f"Mean confidence: {mean_conf:.3f}")
    if not np.isnan(mean_intensity_cv):
        print(f"Intensity CV   : {mean_intensity_cv:.3f}  "
              f"(mean across classes × channels)")
    print()
    print("Counts per class:")
    for cls, cnt in counts[counts > 0].items():
        print(f"  class {cls:2d} : {cnt:4d}  {'█' * min(cnt, 40)}")


# ─────────────────────────────────────────────────────────────────────────────
def launch_ui():
    """Build and display the interactive droplet-analysis UI.
    Returns the shared state dict S (access S['df'], S['pixel_size'], etc.).
    """

    # ── Unique CSS scope ──────────────────────────────────────
    _uid = "dui-" + uuid.uuid4().hex[:6]
    display(HTML(f"""
    <style>
    .{_uid} {{
        background-color: #1a1a1a !important;
    }}
    .{_uid} .p-Widget,
    .{_uid} .jupyter-widgets,
    .{_uid} .widget-box,
    .{_uid} .widget-hbox,
    .{_uid} .widget-vbox {{
        background: transparent !important;
        background-color: transparent !important;
    }}
    .{_uid} .widget-label,
    .{_uid} .widget-label-basic,
    .{_uid} label {{
        background: transparent !important;
        background-color: transparent !important;
        color: #cccccc !important;
    }}
    .{_uid} .widget-readout {{
        background: #2a2a2a !important;
        color: #cccccc !important;
        border: 1px solid #555 !important;
        min-width: 38px !important;
    }}
    .{_uid} input[type=text],
    .{_uid} input[type=number] {{
        background: #2a2a2a !important;
        color: #cccccc !important;
        border: 1px solid #555 !important;
    }}
    .{_uid} select {{
        background: #2a2a2a !important;
        color: #cccccc !important;
        border: 1px solid #555 !important;
    }}
    .{_uid} .noUi-target {{
        background: #3a3a3a !important;
        border-color: #555 !important;
        box-shadow: none !important;
    }}
    .{_uid} .noUi-connect {{ background: #4a90d9 !important; }}
    .{_uid} .noUi-handle {{
        background: #4a90d9 !important;
        border-color: #2a6db5 !important;
        box-shadow: none !important;
    }}
    .{_uid} .widget-hslider,
    .{_uid} .widget-slider {{ background: transparent !important; }}
    .{_uid} hr {{
        border: none !important;
        border-top: 1px solid #444 !important;
        margin: 3px 0 !important;
    }}
    /* ── All buttons ── */
    .{_uid} button,
    .{_uid} .widget-button,
    .{_uid} .fc-button,
    .{_uid} .jupyter-button {{
        color: #cccccc !important;
        background-color: #2a2a2a !important;
        border-color: #555 !important;
    }}
    .{_uid} button:hover,
    .{_uid} .widget-button:hover,
    .{_uid} .fc-button:hover {{
        background-color: #3a3a3a !important;
        color: #ffffff !important;
    }}
    /* ── ipyfilechooser selects & inputs ── */
    .{_uid} .fc-select,
    .{_uid} .fc-select option,
    .{_uid} .fc-input,
    .{_uid} .fc-name {{
        background: #2a2a2a !important;
        color: #cccccc !important;
        border: 1px solid #555 !important;
    }}
    /* ── FileChooser title / path / status text ── */
    .{_uid} .fc-title,
    .{_uid} .fc-current-label,
    .{_uid} .fc-current-value,
    .{_uid} .fc-label,
    .{_uid} .fc-desc,
    .{_uid} .widget-output p,
    .{_uid} .widget-output span,
    .{_uid} .widget-output div {{
        color: #aaaaaa !important;
    }}
    /* ── HTML widget content (ipyfilechooser title, status lines) ── */
    .{_uid} .widget-html,
    .{_uid} .widget-html-content,
    .{_uid} .widget-html-content * {{
        color: #cccccc !important;
        background: transparent !important;
    }}
    /* ── Catch-all: any plain text inside the scoped panel ── */
    .{_uid} p,
    .{_uid} span:not(.noUi-handle):not(.noUi-origin),
    .{_uid} div > div > span {{
        color: #cccccc !important;
    }}
    /* ── Placeholder text ("No file selected", "e.g. …") ── */
    .{_uid} input::placeholder,
    .{_uid} textarea::placeholder {{
        color: #666666 !important;
        opacity: 1 !important;
    }}
    /* ── Dropdown option text (select > option) ── */
    .{_uid} select option {{
        background: #2a2a2a !important;
        color: #cccccc !important;
    }}
    /* ── Tab styling — covers legacy (p-) and modern (lm-/widget-tab) names ── */
    .{_uid} .p-TabBar,
    .{_uid} .lm-TabBar,
    .{_uid} .jupyter-widgets-tab-bar
        {{ background: #1a1a1a !important; border-bottom: 1px solid #444 !important; }}
    .{_uid} .p-TabBar-tab,
    .{_uid} .lm-TabBar-tab
        {{ background: #2a2a2a !important; color: #aaa !important;
           border-color: #444 !important; border-bottom: none !important; }}
    .{_uid} .p-TabBar-tab.p-mod-current,
    .{_uid} .lm-TabBar-tab.lm-mod-current,
    .{_uid} .p-TabBar-tab[aria-selected="true"]
        {{ background: #1a1a1a !important; color: #fff !important;
           border-top: 2px solid #4a90d9 !important; }}
    .{_uid} .p-TabPanel,
    .{_uid} .lm-TabPanel,
    .{_uid} .p-StackedPanel,
    .{_uid} .lm-StackedPanel
        {{ background: #1a1a1a !important; border: 1px solid #444 !important;
           border-top: none !important; }}
    .{_uid} .widget-tab-bar
        {{ background: #1a1a1a !important; }}
    .{_uid} .widget-tab-bar .widget-tab
        {{ background: #2a2a2a !important; color: #aaa !important;
           border: 1px solid #444 !important; border-bottom: none !important; }}
    .{_uid} .widget-tab-bar .widget-tab.mod-active,
    .{_uid} .widget-tab-bar .widget-tab[aria-selected="true"]
        {{ background: #1a1a1a !important; color: #fff !important;
           border-top: 2px solid #4a90d9 !important; }}
    .{_uid} .widget-tab-contents
        {{ background: #1a1a1a !important; border: 1px solid #444 !important; }}
    </style>
    """))

    S = {
        "df": None, "norm": None, "img": None,
        "bg_means": None, "bg_stds": None,
        "reliable": None, "filename": None, "pixel_size": 0.312,
    }

    # ── File picker (FileChooser if available, dropdown fallback) ────────────
    path_input = widgets.Text(
        placeholder="Paste full file path here (overrides picker above)",
        description="",
        layout=widgets.Layout(width="288px"),
    )

    try:
        from ipyfilechooser import FileChooser
        _fc = FileChooser(
            path=os.getcwd(),
            filename="",
            filter_pattern="*.czi",
            title="Select a .czi file",
            show_hidden=False,
        )
        _fc.layout = widgets.Layout(width="288px")
        file_picker_widget = _fc

        def _active_filename():
            txt = path_input.value.strip().strip('"\'')
            if txt: return txt
            return _fc.selected or ""

    except ImportError:
        # Graceful fallback: dropdown + text entry
        def _find_czi():
            seen, out = set(), []
            for p in glob.glob("**/*.czi", recursive=True):
                n = os.path.normpath(p)
                if n not in seen:
                    seen.add(n); out.append(n)
            return sorted(out)

        _files = _find_czi()
        file_dd = widgets.Dropdown(
            options=_files if _files else ["(no .czi found)"],
            description="File:",
            style={"description_width": "35px"},
            layout=widgets.Layout(width="252px"),
        )
        refresh_btn = widgets.Button(description="⟳", tooltip="Refresh list",
                                      layout=widgets.Layout(width="32px"))
        def _refresh(_):
            found = _find_czi()
            file_dd.options = found if found else ["(no .czi found)"]
        refresh_btn.on_click(_refresh)

        file_text = widgets.Text(
            placeholder="or type full path",
            description="",
            layout=widgets.Layout(width="288px"),
        )
        file_picker_widget = widgets.VBox([
            widgets.HBox([file_dd, refresh_btn]),
            file_text,
        ])

        def _active_filename():
            txt = file_text.value.strip()
            if txt: return txt
            v = file_dd.value or ""
            return v if v != "(no .czi found)" else ""

    # ── Pixel-size selector ───────────────────────────────────
    mag_dd = widgets.Dropdown(
        options=[("20×  0.312 µm/px", 0.312),
                 ("40×  0.156 µm/px", 0.156),
                 ("custom", None)],
        value=0.312, description="Obj:",
        style={"description_width": "28px"},
        layout=widgets.Layout(width="175px"),
    )
    pxsz_box = widgets.FloatText(
        value=0.312, description="µm/px:",
        style={"description_width": "42px"},
        layout=widgets.Layout(width="108px"), disabled=True,
    )
    def _mag_changed(change):
        if change["new"] is None: pxsz_box.disabled = False
        else: pxsz_box.value = change["new"]; pxsz_box.disabled = True
    mag_dd.observe(_mag_changed, names="value")

    # ── Figure widget (dark) ──────────────────────────────────
    _BG = "#111111"

    def _make_fig_layout(W=512, H=512, title_text="Select a file and click ▶ Run",
                         shapes=None, annotations=None):
        """Return a go.Layout suitable for the microscopy figure widget."""
        return go.Layout(
            width=640, height=640,
            margin=dict(l=0, r=0, t=24, b=0),
            paper_bgcolor=_BG, plot_bgcolor=_BG,
            font=dict(color="#cccccc"),
            showlegend=False,
            shapes=shapes or [],
            annotations=annotations or [],
            xaxis=dict(range=[0, W], autorange=False, constrain="domain",
                       showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[H, 0], autorange=False, scaleanchor="x",
                       scaleratio=1, showgrid=False, zeroline=False,
                       showticklabels=False),
            title=dict(text=title_text,
                       font=dict(size=12, color="#888"), x=0.01, xanchor="left"),
            dragmode="zoom",
        )

    fig = go.FigureWidget(layout=_make_fig_layout())
    _outer_ref = [None]   # filled after outer is created; lets draw_droplets swap the widget

    # ── draw_droplets ─────────────────────────────────────────
    def draw_droplets():
        nonlocal fig   # allow rebinding when we swap the widget for a new file
        df_cur, norm_cur = S["df"], S["norm"]
        if df_cur is None or norm_cur is None: return
        _, H, W = norm_cur.shape
        # Fluorescence composite: ch1=red, ch2=yellow, ch3=cyan, ch4=blue
        from skimage import filters as _flt
        _ch_colors = np.array([[1,0,0],[1,1,0],[0,1,1],[0,0,1]], dtype=float)
        _rgb = np.zeros((H, W, 3), dtype=float)
        for _c in range(min(norm_cur.shape[0], 4)):
            _rgb += norm_cur[_c, :, :, np.newaxis] * _ch_colors[_c]
        _rgb = _flt.gaussian(_rgb, sigma=1.0, channel_axis=-1)
        new_rgb = (np.clip(_rgb, 0, 1) * 255).astype(np.uint8)
        # Pre-compute grid for brightness lookup
        yy, xx = np.mgrid[0:H, 0:W]

        shapes, scatters, annotations = [], [], []
        for _, row in df_cur.iterrows():
            ci      = int(row.code_int)
            is_und  = (ci == -2)
            is_br   = bool(row.get("bright_red", False))
            if is_und:
                col = "#aaaaaa"
            elif is_br:
                col = "#ff2020"   # bright red for bright-red class
            else:
                col = HEX[ci % 16]
            cx, cy, r = row.centroid_x, row.centroid_y, row.radius

            # ── brightness-adaptive text colour ───────────────
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
            mean_brightness = new_rgb[mask].mean() / 255.0 if mask.any() else 0.0
            txt_color = "black" if mean_brightness > 0.45 else "white"

            # Circle outline — dashed for undecided
            shapes.append(dict(
                type="circle",
                x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
                line=dict(color=col, width=2,
                          dash="dot" if is_und else "solid"),
            ))

            # Hover text — show SNR and posterior if available
            snr_txt  = ""
            post_txt = ""
            if hasattr(row, "min_snr") and not np.isnan(row.min_snr):
                snr_txt = f"<br>SNR: {row.snr_ch1:.1f}/{row.snr_ch2:.1f}/{row.snr_ch3:.1f}/{row.snr_ch4:.1f}"
            if hasattr(row, "min_confidence") and not np.isnan(row.min_confidence):
                post_txt = f"<br>Confidence: {row.min_confidence:.2f}"
            cls_txt = "undecided" if is_und else str(ci)
            scatters.append(go.Scatter(
                x=[cx], y=[cy], mode="markers",
                marker=dict(size=max(8, int(r * 0.8)),
                            color="rgba(0,0,0,0)", opacity=0),
                hovertext=(f"ID:{int(row.label)}  class:{cls_txt}<br>"
                           f"r={r:.1f}px  vol={row.volume_um3:.2f} µm³<br>"
                           f"bright_red:{row.bright_red}"
                           + snr_txt + post_txt),
                hoverinfo="text", showlegend=False,
            ))

            # Annotation — "?" for undecided
            annotations.append(dict(
                x=cx, y=cy,
                text="?" if is_und else str(ci),
                showarrow=False,
                font=dict(color=txt_color, size=9, family="monospace"),
                xref="x", yref="y",
            ))

        rel = S.get("reliable")
        unrel = (" ⚠ " + ",".join(f"ch{i+1}" for i, r in enumerate(rel) if not r)
                 + " unreliable") if rel is not None and not all(rel) else ""
        n_undet = int((df_cur.code_int == -2).sum())
        n_cls   = int(df_cur[df_cur.code_int >= 0].code_int.nunique())
        undet_txt = f" · {n_undet} undecided" if n_undet else ""
        title = (f"{os.path.basename(S['filename'] or '')}  "
                 f"{len(df_cur)} droplets · "
                 f"{n_cls}/16 classes" + undet_txt + unrel)

        # ── Decide: new file → fresh widget; same file → in-place update ──
        is_new_file = S.get("filename") != S.get("_drawn_file")

        if is_new_file or len(fig.data) == 0:
            # Create a brand-new FigureWidget with correct layout from the start.
            # This avoids stale axis / zoom state from the previous image.
            new_fig = go.FigureWidget(
                layout=_make_fig_layout(W=W, H=H, title_text=title,
                                        shapes=shapes, annotations=annotations)
            )
            new_fig.add_trace(go.Image(z=new_rgb))
            if scatters:
                new_fig.add_traces(scatters)
            # Swap the widget in the outer HBox so the user sees it
            if _outer_ref[0] is not None:
                _outer_ref[0].children = (new_fig,) + tuple(_outer_ref[0].children[1:])
            fig = new_fig          # rebind closure variable (nonlocal)
            S["_drawn_file"] = S["filename"]
        else:
            # Same file (Reclassify, manual edit, etc.) — update in-place.
            # Remove old scatter traces, add new ones, then atomically
            # update image z + layout in one batch message.
            if len(fig.data) > 1:
                fig.data = (fig.data[0],)   # valid subset: keep only image trace
            if scatters:
                fig.add_traces(scatters)
            with fig.batch_update():
                fig.data[0].z = new_rgb
                fig.update_layout(
                    shapes=shapes,
                    annotations=annotations,
                    xaxis=dict(range=[0, W], autorange=False, showgrid=False,
                               zeroline=False, showticklabels=False,
                               constrain="domain"),
                    yaxis=dict(range=[H, 0], autorange=False, showgrid=False,
                               zeroline=False, showticklabels=False,
                               scaleanchor="x", scaleratio=1),
                    title_text=title,
                )

        # Attach click handlers to all scatter traces
        for t in fig.data[1:]:
            t.on_click(_on_click)

    # ── Click handler ─────────────────────────────────────────
    def _on_click(trace, points, selector):
        if not points.point_inds: return
        df_cur = S["df"]
        if df_cur is None: return
        tidx = list(fig.data).index(trace) - 1
        if 0 <= tidx < len(df_cur):
            row = df_cur.iloc[tidx]
            sel_id_box.value  = int(row.label)
            class_input.value = int(row.code_int)
            # Also append this droplet's ID to the merge box
            new_id = str(int(row.label))
            cur = merge_box.value.strip()
            if cur:
                existing = [x.strip() for x in cur.split(",")]
                if new_id not in existing:
                    merge_box.value = cur + ", " + new_id
            else:
                merge_box.value = new_id

    # ── Sliders ───────────────────────────────────────────────
    _sl = dict(style={"description_width": "100px"},
               layout=widgets.Layout(width="265px"))
    sigma_sl   = widgets.FloatSlider(value=1.0, min=0.3, max=5.0, step=0.1,
                                      description="σ blur",         **_sl)
    otsu_sl    = widgets.FloatSlider(value=1.0, min=0.3, max=1.5, step=0.05,
                                      description="Otsu scale",     **_sl)
    minsize_sl = widgets.IntSlider(  value=20,  min=5,   max=300, step=5,
                                      description="min area (px²)", **_sl)
    mind_sl    = widgets.IntSlider(  value=5,   min=2,   max=60,  step=1,
                                      description="min dist (px)",  **_sl)
    excl_sl    = widgets.FloatSlider(value=2.0, min=1.0, max=5.0, step=0.1,
                                      description="BG excl ×r",     **_sl)
    relth_sl   = widgets.FloatSlider(value=1.0, min=0.0, max=5.0, step=0.1,
                                      description="Reliability σ",  **_sl)
    margin_sl  = widgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.01,
                                      description="Undecided zone", **_sl)
    throff_sl  = widgets.FloatSlider(value=1.0, min=0.0, max=5.0, step=0.1,
                                      description="Thr offset σ",
                                      style={"description_width": "100px"},
                                      layout=widgets.Layout(width="220px"))
    snrth_sl   = widgets.FloatSlider(value=2.0, min=0.5, max=10.0, step=0.5,
                                      description="SNR threshold",  **_sl)
    brcomp_sl  = widgets.FloatSlider(value=2.0, min=1.1, max=5.0, step=0.1,
                                      description="Bright-red ×",   **_sl)
    brpct_sl   = widgets.FloatSlider(value=98.0, min=80.0, max=99.9, step=0.5,
                                      description="Bright-red pct", **_sl)

    # ── Per-channel threshold text boxes (disabled until after Run) ───────
    _ch_names = ["Ch1 red", "Ch2 yellow", "Ch3 cyan", "Ch4 blue"]
    _th_sl = dict(style={"description_width": "90px"},
                  layout=widgets.Layout(width="215px"))
    thresh_sls = [
        widgets.FloatText(value=0.0, step=10.0,
                          description=f"Thr {_ch_names[c]}", disabled=True,
                          **_th_sl)
        for c in range(4)
    ]

    # ── Buttons ───────────────────────────────────────────────
    run_btn   = widgets.Button(description="▶ Run",        button_style="primary",
                                layout=widgets.Layout(width="72px"))
    simg_btn  = widgets.Button(description="📷 Image",    button_style="success",
                                layout=widgets.Layout(width="100px"))
    sana_btn  = widgets.Button(description="📊 Save Analysis", button_style="success",
                                layout=widgets.Layout(width="215px"))
    reclf_btn = widgets.Button(description="↺ Reclassify", button_style="info",
                                layout=widgets.Layout(width="105px"))
    save_btn  = widgets.Button(description="💾 CSV",       button_style="success",
                                layout=widgets.Layout(width="72px"))
    diag_btn  = widgets.Button(description="📊 Diagnostics", button_style="warning",
                                layout=widgets.Layout(width="115px"))
    diag_out  = widgets.Output()
    status_lbl = widgets.Label(value="Ready.",
                                layout=widgets.Layout(width="560px"))

    # ── Pipeline callbacks ────────────────────────────────────
    def run_pipeline_cb(_):
        fname = _active_filename()
        if not fname: status_lbl.value = "No file selected."; return
        if not os.path.isfile(fname):
            status_lbl.value = f"Not found: {fname}"; return
        status_lbl.value = "Loading …"
        try:
            ps = pxsz_box.value
            S.update(filename=fname, pixel_size=ps)
            img = load_image(fname); norm = normalize_image(img)
            S.update(img=img, norm=norm)
            status_lbl.value = "Detecting …"
            centers, radii, _ = detect_droplets(
                norm, ps,
                sigma_override=sigma_sl.value,
                minsize_override=minsize_sl.value,
                mind_override=mind_sl.value,
                otsu_scale=otsu_sl.value,
            )
            # Pass 1: full circle → fit GMM → thresholds
            inten_full  = measure_droplet_intensities(img, centers, radii, meas_frac=1.0)
            # Pass 2: shrunk circle → classify using thresholds from pass 1
            inten_small = measure_droplet_intensities(img, centers, radii, meas_frac=0.65)
            bgm, bgs = estimate_background(img, centers, radii, ps,
                                            excl_factor=excl_sl.value)
            S.update(bg_means=bgm, bg_stds=bgs)
            status_lbl.value = "Classifying …"
            # GMM fit on full-circle intensities (no margin — just for thresholds)
            _, threshs, rel, _ = classify_channels(inten_full, bgm, bgs,
                                                    reliability_sigma=relth_sl.value,
                                                    margin_sigma=0.0)
            # Apply offset: effective_thresh = GMM_thresh + offset × bg_std
            effective_threshs = threshs + throff_sl.value * bgs
            # Classify shrunk-circle intensities using effective thresholds
            bn, _, _, post = classify_channels(inten_small, bgm, bgs,
                                               reliability_sigma=relth_sl.value,
                                               margin_sigma=margin_sl.value,
                                               manual_thresholds=effective_threshs)
            S["reliable"]       = rel
            S["thresholds"]     = effective_threshs
            S["gmm_thresholds"] = threshs          # raw GMM (pre-offset) for reclassify
            S["bg_sub"]         = inten_full - bgm   # diagnostics histogram uses full-circle data
            # ── Set threshold boxes to effective (offset-adjusted) values ─
            for c, sl in enumerate(thresh_sls):
                sl.value    = float(effective_threshs[c])
                sl.disabled = False
            br = detect_bright_red(inten_small, bgm, bgs,
                                    bright_factor=brcomp_sl.value,
                                    bright_pct=brpct_sl.value)
            S["df"] = build_dataframe(centers, radii, inten_small, bn, br, rel, ps,
                                      bg_means=bgm, bg_stds=bgs,
                                      posteriors=post,
                                      snr_threshold=snrth_sl.value)
            draw_droplets()
            df_d = S['df']
            n_undet = int((df_d.code_int == -2).sum())
            n_cls   = int(df_d[df_d.code_int >= 0].code_int.nunique())
            n_conf  = int((df_d['min_confidence'] > 0.05).sum()) if 'min_confidence' in df_d.columns else len(df_d)
            frac    = f"{n_conf/len(df_d)*100:.0f}%" if len(df_d) else '\u2014'
            status_lbl.value = (f"Done — {len(df_d)} droplets, "
                                f"{n_cls}/16 classes"
                                + (f", {n_undet} undecided" if n_undet else "")
                                + f" · confident: {n_conf}/{len(df_d)} ({frac}).")
        except Exception as e:
            status_lbl.value = f"Error: {e}"; raise

    def reclassify_cb(_):
        df_old = S["df"]
        if df_old is None: status_lbl.value = "Run pipeline first."; return
        status_lbl.value = "Re-classifying …"
        try:
            ps = S["pixel_size"]
            inten = np.column_stack([df_old[f"intensity_ch{i+1}"] for i in range(4)])
            bgm, bgs = S["bg_means"], S["bg_stds"]
            # Use whatever is currently shown in the threshold boxes.
            # (throff_sl.observe keeps them in sync automatically; user can also edit directly.)
            man_thresh = [sl.value if not sl.disabled else None for sl in thresh_sls]
            bn, threshs, rel, post = classify_channels(inten, bgm, bgs,
                                                        reliability_sigma=relth_sl.value,
                                                        margin_sigma=margin_sl.value,
                                                        manual_thresholds=man_thresh)
            S["reliable"]   = rel
            S["thresholds"] = threshs
            S["bg_sub"]     = inten - bgm
            br = detect_bright_red(inten, bgm, bgs,
                                    bright_factor=brcomp_sl.value,
                                    bright_pct=brpct_sl.value)
            S["df"] = build_dataframe(
                df_old[["centroid_y", "centroid_x"]].values,
                df_old["radius"].values, inten, bn, br, rel, ps,
                bg_means=bgm, bg_stds=bgs,
                posteriors=post, snr_threshold=snrth_sl.value)
            draw_droplets()
            df_d = S['df']
            n_undet = int((df_d.code_int == -2).sum())
            n_cls   = int(df_d[df_d.code_int >= 0].code_int.nunique())
            n_conf  = int((df_d['min_confidence'] > 0.05).sum()) if 'min_confidence' in df_d.columns else len(df_d)
            frac    = f"{n_conf/len(df_d)*100:.0f}%" if len(df_d) else '\u2014'
            status_lbl.value = (f"Re-classified — {n_cls}/16 classes"
                                + (f", {n_undet} undecided" if n_undet else "")
                                + f" · confident: {n_conf}/{len(df_d)} ({frac}).")
        except Exception as e:
            status_lbl.value = f"Error: {e}"; raise

    def save_cb(_):
        df_cur = S["df"]
        if df_cur is None: status_lbl.value = "Nothing to save."; return
        base = os.path.splitext(S["filename"] or "output")[0]
        out  = base + "_analysis.csv"
        if 'min_confidence' in df_cur.columns:
            df_save = df_cur[df_cur['min_confidence'] > 0.05]
        else:
            df_save = df_cur
        df_save.to_csv(out, index=False)
        status_lbl.value = f"Saved {len(df_save)}/{len(df_cur)} confident droplets → {os.path.basename(out)}"

    def diag_cb(_):
        diag_out.clear_output(wait=True)
        with diag_out:
            bg_sub = S.get("bg_sub")
            threshs = S.get("thresholds")
            if bg_sub is None or threshs is None:
                print("Run pipeline first."); return
            import matplotlib
            matplotlib.use("module://matplotlib_inline.backend_inline")
            import matplotlib.pyplot as plt
            bgs = S["bg_stds"]
            fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), dpi=150)
            ch_labels = ["Ch1 red", "Ch2 yellow", "Ch3 cyan", "Ch4 blue"]
            ch_colors = ["#e57373", "#fff176", "#80deea", "#82b1ff"]
            for c, ax in enumerate(axes):
                vals = bg_sub[:, c]
                ax.hist(vals, bins=40, color=ch_colors[c], edgecolor="#555",
                        alpha=0.85, density=True)
                t = threshs[c]
                ax.axvline(t, color="white", linewidth=1.5, linestyle="--",
                           label=f"thr={t:.0f}")
                # shade dead zone if margin_sl > 0
                dz = margin_sl.value * bgs[c]
                if dz > 0:
                    ax.axvspan(t - dz, t + dz, color="gray", alpha=0.3,
                               label=f"±{dz:.0f}")
                ax.set_title(ch_labels[c], fontsize=9)
                ax.set_xlabel("bg-sub intensity", fontsize=8)
                if c == 0:
                    ax.set_ylabel("density", fontsize=8)
                ax.legend(fontsize=7)
                ax.tick_params(labelsize=7)
            fig.tight_layout()
            plt.show()

    def _save_image_cb(_):
        """Render classified image (composite + circles) to PNG via matplotlib."""
        df_cur = S["df"]
        norm_cur = S["norm"]
        if df_cur is None or norm_cur is None:
            status_lbl.value = "Run pipeline first."; return
        import matplotlib.patches as _mp
        from skimage import filters as _flt
        _, H, W = norm_cur.shape
        _ch_colors = np.array([[1,0,0],[1,1,0],[0,1,1],[0,0,1]], dtype=float)
        _rgb = np.zeros((H, W, 3), dtype=float)
        for _c in range(min(norm_cur.shape[0], 4)):
            _rgb += norm_cur[_c, :, :, np.newaxis] * _ch_colors[_c]
        _rgb = _flt.gaussian(_rgb, sigma=1.0, channel_axis=-1)
        rgb_img = (np.clip(_rgb, 0, 1) * 255).astype(np.uint8)
        yy, xx = np.mgrid[0:H, 0:W]
        fig_s, ax_s = plt.subplots(figsize=(8, 8), dpi=150)
        ax_s.imshow(rgb_img, origin="upper")
        ax_s.set_xlim(0, W); ax_s.set_ylim(H, 0)
        ax_s.axis("off")
        for _, row in df_cur.iterrows():
            ci = int(row.code_int)
            is_br = bool(row.get("bright_red", False))
            col = "#aaaaaa" if ci == -2 else ("#ff2020" if is_br else HEX[ci % 16])
            cx, cy, r = row.centroid_x, row.centroid_y, row.radius
            mask = (yy - cy)**2 + (xx - cx)**2 <= r**2
            mean_br = rgb_img[mask].mean() / 255.0 if mask.any() else 0.0
            tc = "black" if mean_br > 0.45 else "white"
            circ = _mp.Circle((cx, cy), r, fill=False, edgecolor=col, linewidth=1.5,
                               linestyle=":" if ci == -2 else "-")
            ax_s.add_patch(circ)
            ax_s.text(cx, cy, "?" if ci == -2 else str(ci),
                      ha="center", va="center", fontsize=7, color=tc, fontweight="bold")
        base = os.path.splitext(S["filename"] or "output")[0]
        out = base + "_classified.png"
        fig_s.savefig(out, bbox_inches="tight", pad_inches=0.02, dpi=150)
        plt.close(fig_s)
        # Save matching parameters JSON so the result is reproducible
        import json, datetime
        params_out = base + "_classified_params.json"
        params_dict = dict(
            source_file       = S["filename"],
            saved_at          = datetime.datetime.now().isoformat(timespec="seconds"),
            pixel_size_um     = float(pxsz_box.value),
            sigma             = float(sigma_sl.value),
            min_size          = int(minsize_sl.value),
            min_dist          = int(mind_sl.value),
            excl_factor       = float(excl_sl.value),
            reliability_sigma = float(relth_sl.value),
            margin_sigma      = float(margin_sl.value),
            thresh_offset_sigma = float(throff_sl.value),
            snr_threshold     = float(snrth_sl.value),
            bright_factor     = float(brcomp_sl.value),
            bright_pct        = float(brpct_sl.value),
            channel_thresholds = [float(sl.value) for sl in thresh_sls],
        )
        with open(params_out, "w") as _f:
            json.dump(params_dict, _f, indent=2)
        status_lbl.value = (f"Saved image → {os.path.basename(out)}  "
                            f"+ {os.path.basename(params_out)}")

    def _save_analysis_cb(_):
        """Save all show_analysis plots as numbered PNGs."""
        try:
            df_cur = S["df"]
            if df_cur is None:
                status_lbl.value = "Run pipeline first."; return
            base = os.path.splitext(S["filename"] or "output")[0]
            prefix = base + "_analysis"
            show_analysis(df_cur, pixel_size_um=S["pixel_size"],
                          img_shape=(S["norm"].shape if S["norm"] is not None else None),
                          save_prefix=prefix)
            import glob as _gl
            saved = sorted(_gl.glob(prefix + "_??.png"))
            status_lbl.value = f"Saved {len(saved)} analysis plot(s) → {os.path.basename(prefix)}_*.png"
        except Exception as e:
            import traceback; traceback.print_exc()
            status_lbl.value = f"Analysis error: {e}"

    run_btn.on_click(run_pipeline_cb)
    simg_btn.on_click(_save_image_cb)
    sana_btn.on_click(_save_analysis_cb)
    reclf_btn.on_click(reclassify_cb)
    save_btn.on_click(save_cb)
    diag_btn.on_click(diag_cb)

    # ── Edit controls ─────────────────────────────────────────
    sel_id_box  = widgets.BoundedIntText(
        value=1, min=1, max=9999, description="ID:",
        style={"description_width": "22px"},
        layout=widgets.Layout(width="90px"),
    )
    class_input = widgets.BoundedIntText(
        value=0, min=0, max=15, description="cls:",
        style={"description_width": "25px"},
        layout=widgets.Layout(width="75px"),
    )
    upd_btn = widgets.Button(description="Update",
                              button_style="warning",
                              layout=widgets.Layout(width="68px"))

    def _update_class(_):
        df_cur = S["df"]
        if df_cur is None: return
        did, cls = sel_id_box.value, class_input.value
        if did not in df_cur.label.values:
            status_lbl.value = f"ID {did} not found."; return
        idx = df_cur.index[df_cur.label == did][0]
        S["df"].at[idx, "code_int"]   = cls
        S["df"].at[idx, "bright_red"] = (cls == 0)
        draw_droplets()
        status_lbl.value = f"ID {did} → class {cls}"
    upd_btn.on_click(_update_class)

    merge_box = widgets.Text(placeholder="Click droplets to add IDs",
                              description="IDs:",
                              style={"description_width": "25px"},
                              layout=widgets.Layout(width="148px"))
    merge_btn = widgets.Button(description="Merge",
                                button_style="warning",
                                layout=widgets.Layout(width="60px"))
    merge_clr_btn = widgets.Button(description="✕",
                                    button_style="",
                                    layout=widgets.Layout(width="32px"))
    merge_clr_btn.on_click(lambda _: setattr(merge_box, "value", ""))

    def _merge_cb(_):
        try:
            df_cur = S["df"]
            if df_cur is None: return
            try:
                ids = [int(x.strip()) for x in merge_box.value.split(",") if x.strip()]
            except ValueError:
                status_lbl.value = "Merge: enter comma-separated IDs."; return
            if len(ids) < 2:
                status_lbl.value = "Merge: need at least 2 IDs."; return
            valid_labels = set(df_cur.label.astype(int).tolist())
            bad = [i for i in ids if i not in valid_labels]
            if bad:
                status_lbl.value = f"Merge: IDs not found: {bad}"; return
            rows = df_cur[df_cur.label.astype(int).isin(ids)]
            if rows.empty:
                status_lbl.value = "Merge: no matching rows found."; return
            ps = S["pixel_size"]
            # New centroid = mean of all merged droplet centres
            new_cy = int(round(rows.centroid_y.mean()))
            new_cx = int(round(rows.centroid_x.mean()))
            # Radius = smallest circle centred at new centroid that covers all original droplets
            new_r = float(max(
                np.sqrt((new_cy - ry)**2 + (new_cx - rx)**2) + rr
                for ry, rx, rr in zip(rows.centroid_y, rows.centroid_x, rows["radius"])
            ))
            new_r_um   = new_r * ps
            new_vol    = (4/3) * np.pi * new_r_um ** 3
            kr = rows.iloc[0].copy()
            kr["centroid_y"] = new_cy
            kr["centroid_x"] = new_cx
            kr["radius"]     = new_r
            kr["radius_um"]  = new_r_um
            kr["volume_um3"] = new_vol
            df2 = df_cur[~df_cur.label.astype(int).isin(ids[1:])].copy()
            df2 = df2.reset_index(drop=True)
            ki  = df2.index[df2.label.astype(int) == ids[0]][0]
            for col in kr.index:
                if col in df2.columns:
                    try:
                        df2.at[ki, col] = df2[col].dtype.type(kr[col])
                    except (ValueError, TypeError):
                        df2.at[ki, col] = kr[col]
            S["df"] = df2
            S["_drawn_file"] = None   # force fresh FigureWidget so shapes re-render
            merge_box.value = ""
            draw_droplets()
            status_lbl.value = f"Merged {ids} → ID {ids[0]}  r={new_r:.1f}px"
        except Exception as e:
            status_lbl.value = f"Merge error: {type(e).__name__}: {e}"
    merge_btn.on_click(_merge_cb)

    _ai = dict(style={"description_width": "14px"},
               layout=widgets.Layout(width="78px"))
    ay  = widgets.IntText(  value=0,  description="y:",   **_ai)
    ax  = widgets.IntText(  value=0,  description="x:",   **_ai)
    ar  = widgets.FloatText(value=10, description="r:",   **_ai)
    ac  = widgets.BoundedIntText(value=1, min=0, max=15,
                                  description="cls:",
                                  style={"description_width": "18px"},
                                  layout=widgets.Layout(width="72px"))
    add_btn = widgets.Button(description="Add",
                              button_style="success",
                              layout=widgets.Layout(width="52px"))

    def _add_cb(_):
        df_cur = S["df"]; img = S["img"]
        if df_cur is None or img is None:
            status_lbl.value = "Run pipeline first."; return
        try:
            y, x, r, cls = int(ay.value), int(ax.value), float(ar.value), int(ac.value)
        except ValueError:
            status_lbl.value = "Invalid y/x/r/cls."; return
        ps = S["pixel_size"]
        n_ch = img.shape[0]; H2, W2 = img.shape[1], img.shape[2]
        yy, xx = np.mgrid[0:H2, 0:W2]
        mask  = (yy-y)**2 + (xx-x)**2 <= r**2
        inten = np.array([img[c][mask].mean() for c in range(n_ch)])
        new_id = int(df_cur.label.max()) + 1
        S["df"] = pd.concat([df_cur, pd.DataFrame([dict(
            label=new_id, centroid_y=y, centroid_x=x,
            radius=r, radius_um=r*ps,
            volume_um3=(4/3)*np.pi*(r*ps)**3,
            bright_red=(cls==0), code_int=cls,
            intensity_ch1=float(inten[0]), intensity_ch2=float(inten[1]),
            intensity_ch3=float(inten[2]), intensity_ch4=float(inten[3]),
            binary_ch1=0, binary_ch2=0, binary_ch3=0, binary_ch4=0,
        )])], ignore_index=True)
        draw_droplets()
        status_lbl.value = f"Added ID {new_id} class {cls} at ({x},{y})"
    add_btn.on_click(_add_cb)

    del_id_box = widgets.BoundedIntText(
        value=1, min=1, max=9999, description="ID:",
        style={"description_width": "22px"},
        layout=widgets.Layout(width="90px"),
    )
    del_btn = widgets.Button(description="Delete",
                              button_style="danger",
                              layout=widgets.Layout(width="60px"))

    def _delete_cb(_):
        df_cur = S["df"]
        if df_cur is None: return
        did = del_id_box.value
        if did not in df_cur.label.values:
            status_lbl.value = f"ID {did} not found."; return
        S["df"] = df_cur[df_cur.label != did].reset_index(drop=True)
        draw_droplets()
        status_lbl.value = f"Deleted ID {did}"
    del_btn.on_click(_delete_cb)

    # ── Layout ────────────────────────────────────────────────
    def _h(t):
        return widgets.HTML(
            f"<b style='font-size:0.78em;letter-spacing:0.05em;"
            f"color:#888'>{t}</b>"
        )
    def _sep():
        return widgets.HTML("<hr style='margin:3px 0'>")

    _tab_pad = widgets.Layout(padding="6px")
    tab_detect   = widgets.VBox([sigma_sl, otsu_sl, minsize_sl, mind_sl],
                                 layout=_tab_pad)
    tab_classify = widgets.VBox([excl_sl, relth_sl, margin_sl, snrth_sl, throff_sl],
                                 layout=_tab_pad)
    tab_bright   = widgets.VBox([brcomp_sl, brpct_sl],
                                 layout=_tab_pad)
    param_tabs = widgets.Tab(children=[tab_detect, tab_classify, tab_bright])
    for _i, _t in enumerate(["Detect", "Classify", "Bright-red"]):
        param_tabs.set_title(_i, _t)

    left_col = widgets.VBox([
        file_picker_widget,
        path_input,
        widgets.HBox([mag_dd, pxsz_box]),
        _sep(),
        param_tabs,
        _sep(),
        widgets.HBox([run_btn, reclf_btn, save_btn],
                     layout=widgets.Layout(gap="4px")),
    ], layout=widgets.Layout(width="300px", padding="8px"))

    right_col = widgets.VBox([
        _h("EDIT DROPLETS"),
        _sep(),
        _h("Update class"),
        widgets.HBox([sel_id_box, class_input, upd_btn],
                     layout=widgets.Layout(gap="4px")),
        _sep(),
        _h("Merge"),
        widgets.HBox([merge_box, merge_btn, merge_clr_btn], layout=widgets.Layout(gap="4px")),
        _sep(),
        _h("Add new  (y, x, r, cls)"),
        widgets.HBox([ay, ax], layout=widgets.Layout(gap="3px")),
        widgets.HBox([ar, ac, add_btn], layout=widgets.Layout(gap="3px")),
        _sep(),
        _h("Delete"),
        widgets.HBox([del_id_box, del_btn], layout=widgets.Layout(gap="4px")),
        _sep(),
        _h("CHANNEL THRESHOLDS (auto; edit to override)"),
        *thresh_sls,
        widgets.HBox([diag_btn, simg_btn], layout=widgets.Layout(gap="4px", margin="4px 0")),
        sana_btn,
    ], layout=widgets.Layout(
        width="240px", padding="8px",
        border_left="1px solid #444",
    ))

    ctrl_panel = widgets.HBox(
        [left_col, right_col],
        layout=widgets.Layout(border="1px solid #444", max_height="645px",
                              overflow_y="auto"),
    )
    ctrl_panel.add_class(_uid)

    status_row = widgets.HBox(
        [status_lbl],
        layout=widgets.Layout(padding="3px 8px", border="1px solid #444",
                              border_top="none"),
    )
    status_row.add_class(_uid)
    # ── throff live-update: update thresh_sls whenever offset slider moves ──
    def _throff_changed(change):
        gmm_th = S.get("gmm_thresholds")
        bgs_   = S.get("bg_stds")
        if gmm_th is not None and bgs_ is not None:
            effective = gmm_th + change["new"] * bgs_
            for c, sl in enumerate(thresh_sls):
                sl.value = float(effective[c])
    throff_sl.observe(_throff_changed, names="value")

    outer = widgets.HBox([fig, widgets.VBox([ctrl_panel, status_row])])
    _outer_ref[0] = outer   # allow draw_droplets to swap the figure widget
    display(outer)
    display(diag_out)   # diagnostics plot appears below main UI, doesn't affect layout
    return S
