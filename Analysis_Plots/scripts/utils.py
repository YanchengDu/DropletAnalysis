"""
utils.py
Shared data loading, constants, and helper functions for all analysis plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -- Paths --------------------------------------------------------------------
ANALYSIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(ANALYSIS_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Class colours (matches pipeline HEX_COLORS) ------------------------------
CLASS_COLORS = [
    "#ff2020", "#1d1b7d", "#086564", "#15a2ff",
    "#b8b910", "#6a6a69", "#81f66c", "#95d9ab",
    "#530409", "#720351", "#7db8c0", "#798ced",
    "#ffd425", "#faa34b", "#de963e", "#f7e5b6",
]
CLASSES = list(range(16))

# -- Reliability threshold ----------------------------------------------------
MIN_N = 10   # classes with fewer droplets than this are flagged as unreliable

# -- Density calculation parameters ------------------------------------------
# Image assumed 1024x1024 px (centroid_max ~1018 in all files confirms this)
# Pixel sizes provided by user
IMAGE_SIZE_PX  = 1024
PIXEL_SIZE_UM  = {"20x": 0.312, "40x": 0.156}
# FOV areas in um^2 (identical for both mags since same camera chip)
FOV_AREA_UM2   = {mag: (IMAGE_SIZE_PX * px) ** 2 for mag, px in PIXEL_SIZE_UM.items()}
MAGS           = ["20x", "40x"]

# -- Condition definitions ----------------------------------------------------

# Figure 1: PAGE vs heated day4
FIG1_CONDITIONS = {
    "PAGE (day 3)":                "PAGEPurified_5uMsalt_heated_day3",
    "+2 uM salt, heated (day 4)":  "layeringpurified_extra2uMsalt_heated_day4",
    "No added salt, heated (day 4)":"layeringpurified_noaddedsalt_heated_day4",
}

# Figure 2: cross-group at day 4 (2x2 layout: rows=salt, cols=mix)
FIG2_LAYOUT = [
    [
        ("+2 uM salt\ndirect mix",  "layeringpurified_extra2uMsalt_directmix_day4"),
        ("+2 uM salt\nheated",      "layeringpurified_extra2uMsalt_heated_day4"),
    ],
    [
        ("No added salt\ndirect mix","layeringpurified_noaddedsalt_directmix_day4"),
        ("No added salt\nheated",    "layeringpurified_noaddedsalt_heated_day4"),
    ],
]

# Figure 3: time course -- 4 groups x available days
FIG3_GROUPS = {
    "+2 uM salt, direct mix": [
        ("Day 0", "layeringpurified_extra2uMsalt_directmix_day0"),
        ("Day 1", "layeringpurified_extra2uMsalt_directmix_day1"),
        ("Day 4", "layeringpurified_extra2uMsalt_directmix_day4"),
    ],
    "+2 uM salt, heated": [
        ("Day 1", "layeringpurified_extra2uMsalt_heated_day1"),
        ("Day 4", "layeringpurified_extra2uMsalt_heated_day4"),
    ],
    "No added salt, direct mix": [
        ("Day 0", "layeringpurified_noaddedsalt_directmix_day0"),
        ("Day 1", "layeringpurified_noaddedsalt_directmix_day1"),
        ("Day 4", "layeringpurified_noaddedsalt_directmix_day4"),
    ],
    "No added salt, heated": [
        ("Day 1", "layeringpurified_noaddedsalt_heated_day1"),
        ("Day 4", "layeringpurified_noaddedsalt_heated_day4"),
    ],
}

DAY_COLORS  = {"Day 0": "#4e9af1", "Day 1": "#f4a020", "Day 4": "#e03030"}
FIG1_COLORS = ["#888888", "#e03030", "#4e9af1"]
FIG2_COLOR  = "#4e9af1"


# -- Data loading -------------------------------------------------------------

def load_condition(folder_name):
    """Pool all CSVs in a condition folder (20x + 40x), filter min_confidence > 0.05."""
    folder = os.path.join(ANALYSIS_DIR, folder_name)
    frames = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, fname))
            df["magnification"] = fname.replace(".csv", "")
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined[combined["min_confidence"] > 0.05].copy()


def load_condition_by_mag(folder_name):
    """
    Load a condition folder keeping magnifications separate.
    Returns dict {mag_str: df}, e.g. {"20x": df20, "40x": df40}.
    Only includes magnifications that have a CSV file.
    """
    folder = os.path.join(ANALYSIS_DIR, folder_name)
    result = {}
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".csv"):
            mag = fname.replace(".csv", "")
            df  = pd.read_csv(os.path.join(folder, fname))
            result[mag] = df[df["min_confidence"] > 0.05].copy()
    return result


# -- Per-class statistics -----------------------------------------------------

def class_counts(df):
    """Return length-16 integer array of droplet counts per class."""
    counts = np.zeros(16, dtype=int)
    if len(df) == 0:
        return counts
    vc = df["code_int"].value_counts()
    for c in CLASSES:
        counts[c] = int(vc.get(c, 0))
    return counts


def class_fractions(df):
    """Return length-16 array of class fractions by droplet count."""
    fracs = np.zeros(16)
    if len(df) == 0:
        return fracs
    vc = df["code_int"].value_counts()
    for c in CLASSES:
        fracs[c] = vc.get(c, 0) / len(df)
    return fracs


def class_volume_fractions(df):
    """
    Return length-16 array: vol_frac[c] = sum(vol in class c) / sum(vol total).
    Reflects spatial occupancy.
    """
    vol_fracs = np.zeros(16)
    if len(df) == 0:
        return vol_fracs
    total_vol = df["volume_um3"].sum()
    if total_vol == 0:
        return vol_fracs
    for c in CLASSES:
        sub = df.loc[df["code_int"] == c, "volume_um3"]
        vol_fracs[c] = sub.sum() / total_vol
    return vol_fracs


def class_mean_std(df, col):
    """Return (means, stds) arrays of length 16 for the given column."""
    means = np.zeros(16)
    stds  = np.zeros(16)
    for c in CLASSES:
        sub = df.loc[df["code_int"] == c, col]
        if len(sub) > 0:
            means[c] = sub.mean()
            stds[c]  = sub.std(ddof=1) if len(sub) > 1 else 0.0
    return means, stds


def class_density(df, mag):
    """
    Compute per-class droplet density (droplets / um^2).
    Each FOV (identified by filename+label) is treated as one replicate.
    Returns (means, stds, n_fovs) each of length 16.
    - means[c] = mean density of class c across FOVs
    - stds[c]  = std across FOVs (0 if only 1 FOV)
    """
    fov_area = FOV_AREA_UM2[mag]
    means = np.zeros(16)
    stds  = np.zeros(16)
    if len(df) == 0:
        return means, stds, 0

    fov_groups = list(df.groupby(["filename", "label"]))
    n_fovs     = len(fov_groups)
    per_fov    = np.zeros((n_fovs, 16))

    for i, (_, fov_df) in enumerate(fov_groups):
        vc = fov_df["code_int"].value_counts()
        for c in CLASSES:
            per_fov[i, c] = vc.get(c, 0) / fov_area

    means = per_fov.mean(axis=0)
    stds  = per_fov.std(axis=0, ddof=1) if n_fovs > 1 else np.zeros(16)
    return means, stds, n_fovs


def total_density(df, mag):
    """
    Total droplet density (all classes) per FOV.
    Returns (mean, std, n_fovs).
    """
    fov_area   = FOV_AREA_UM2[mag]
    fov_groups = list(df.groupby(["filename", "label"]))
    n_fovs     = len(fov_groups)
    if n_fovs == 0:
        return 0.0, 0.0, 0
    per_fov = [len(fov_df) / fov_area for _, fov_df in fov_groups]
    mean = np.mean(per_fov)
    std  = np.std(per_fov, ddof=1) if n_fovs > 1 else 0.0
    return mean, std, n_fovs


def class_density_combined(df):
    """
    Compute per-class droplet density (droplets / um^2) pooling 20x and 40x FOVs.
    Each FOV is normalised by its own FOV area (looked up from the
    'magnification' column set by load_condition).
    Returns (means, stds, n_fovs) each of length 16.
    """
    means = np.zeros(16)
    stds  = np.zeros(16)
    if len(df) == 0 or "magnification" not in df.columns:
        return means, stds, 0

    fov_groups = list(df.groupby(["filename", "label", "magnification"]))
    n_fovs     = len(fov_groups)
    per_fov    = np.zeros((n_fovs, 16))

    for i, ((_, _, mag), fov_df) in enumerate(fov_groups):
        fov_area = FOV_AREA_UM2.get(mag, FOV_AREA_UM2["40x"])
        vc       = fov_df["code_int"].value_counts()
        for c in CLASSES:
            per_fov[i, c] = vc.get(c, 0) / fov_area

    means = per_fov.mean(axis=0)
    stds  = per_fov.std(axis=0, ddof=1) if n_fovs > 1 else np.zeros(16)
    return means, stds, n_fovs


# -- Reliability annotation helpers ------------------------------------------

def mark_low_n_bar(ax, x, bar_top, n, rotate=True):
    """Annotate a bar where class count < MIN_N with '* N=X' in red."""
    txt = "* N={}".format(n)
    if rotate:
        ax.text(x, bar_top, txt, ha="center", va="bottom",
                fontsize=4.5, color="#cc0000", rotation=90)
    else:
        ax.text(x, bar_top, txt, ha="center", va="bottom",
                fontsize=5, color="#cc0000")


def overlay_hollow_marker(ax, x, y, color, markersize=6):
    """Draw an open circle to flag a low-N data point."""
    ax.plot(x, y, "o", markerfacecolor="none",
            markeredgecolor=color, markeredgewidth=1.5,
            markersize=markersize, zorder=5)


# -- Plot helpers -------------------------------------------------------------

def class_legend(ax, fontsize=7):
    patches = [mpatches.Patch(color=CLASS_COLORS[c], label=str(c)) for c in CLASSES]
    ax.legend(handles=patches, title="Class", ncol=4,
              fontsize=fontsize, title_fontsize=fontsize,
              loc="upper right", framealpha=0.8)


def save_figure(fig, filename, dpi=300):
    """Save PNG (300 DPI) and SVG (vector) versions."""
    png_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print("Saved -> {}".format(png_path))
    svg_path = os.path.join(OUTPUT_DIR, filename.replace(".png", ".svg"))
    fig.savefig(svg_path, bbox_inches="tight")
    print("Saved -> {}".format(svg_path))


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#333",
        "axes.grid":        True,
        "grid.color":       "#e0e0e0",
        "grid.linewidth":   0.5,
        "font.size":        9,
    })
