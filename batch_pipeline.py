"""
batch_pipeline.py
Batch-process multiple CZI images through the droplet analysis pipeline.

Usage:
    from batch_pipeline import find_czi_files, run_batch, summarise_batch
"""

import glob
import os
import traceback

import numpy as np
import pandas as pd

from droplet_pipeline import (
    load_image, normalize_image, detect_droplets,
    measure_droplet_intensities, estimate_background,
    classify_channels, detect_bright_red, build_dataframe,
)


# ── Default parameters (mirror the UI slider defaults) ───────────────────────

DEFAULT_PARAMS = dict(
    pixel_size_um       = 0.312,  # µm per pixel  (20× objective)
    sigma               = 1.0,   # Gaussian blur for detection
    min_size            = 20,    # minimum droplet area (px²)
    min_dist            = 5,     # minimum centre-to-centre distance (px)
    excl_factor         = 2.0,   # background exclusion zone (× radius)
    reliability_sigma   = 1.0,   # channel reliability threshold (σ)
    thresh_offset_sigma = 1.0,   # shift GMM threshold by N × bg_std (tighter)
    margin_sigma        = 0.0,   # dead-zone half-width around threshold (× bg_std)
    snr_threshold       = 2.0,   # min per-channel SNR to keep a droplet
    meas_frac_classify  = 0.65,  # circle fraction used for classification (pass 2)
    bright_factor       = 2.0,   # bright-red GMM ratio threshold
    bright_pct          = 98.0,  # bright-red fallback percentile
    manual_thresholds   = None,  # [ch1, ch2, ch3, ch4] floats — overrides GMM if set
)


# ── File discovery ────────────────────────────────────────────────────────────

def find_czi_files(root="."):
    """Return a sorted, de-duplicated list of all .czi files under *root*."""
    seen, out = set(), []
    for p in glob.glob(os.path.join(root, "**", "*.czi"), recursive=True):
        n = os.path.normpath(p)
        if n not in seen:
            seen.add(n); out.append(n)
    return sorted(out)


# ── Single-image processing ───────────────────────────────────────────────────

def process_image(filename, params=None):
    """Run the full pipeline on one CZI file.

    Parameters
    ----------
    filename : str
        Path to the .czi file.
    params : dict, optional
        Override any key in DEFAULT_PARAMS.

    Returns
    -------
    df : pd.DataFrame or None
        Per-droplet results with an added 'filename' column.
        None if processing failed.
    meta : dict
        Metadata: filename, n_droplets, n_classes, reliable, error.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    meta = dict(filename=filename, n_droplets=0, n_classes=0,
                reliable=None, error=None)
    try:
        img  = load_image(filename)
        norm = normalize_image(img)
        ps   = p["pixel_size_um"]

        centers, radii, _ = detect_droplets(
            norm, ps,
            sigma_override   = p["sigma"],
            minsize_override = p["min_size"],
            mind_override    = p["min_dist"],
        )

        if len(centers) == 0:
            meta["error"] = "No droplets detected"
            return None, meta

        bg_means, bg_stds = estimate_background(img, centers, radii, ps,
                                                excl_factor=p["excl_factor"])

        # Pass 1: full circle → fit GMM → raw thresholds
        inten_full = measure_droplet_intensities(img, centers, radii, meas_frac=1.0)
        _, threshs, rel, _ = classify_channels(
            inten_full, bg_means, bg_stds,
            reliability_sigma=p["reliability_sigma"],
            margin_sigma=0.0)

        # Apply threshold offset (same logic as single-file UI)
        effective_threshs = threshs + p["thresh_offset_sigma"] * bg_stds

        # Manual override: replace per-channel thresholds if provided
        if p["manual_thresholds"] is not None:
            mt = p["manual_thresholds"]
            if len(mt) != 4:
                raise ValueError(f"manual_thresholds must have 4 values, got {len(mt)}")
            effective_threshs = np.array([float(v) for v in mt])

        # Pass 2: shrunk circle → classify using effective thresholds
        inten_small = measure_droplet_intensities(
            img, centers, radii, meas_frac=p["meas_frac_classify"])
        binary, _, rel, posteriors = classify_channels(
            inten_small, bg_means, bg_stds,
            reliability_sigma=p["reliability_sigma"],
            margin_sigma=p["margin_sigma"],
            manual_thresholds=list(effective_threshs))

        bright_red = detect_bright_red(inten_small, bg_means, bg_stds,
                                       bright_factor=p["bright_factor"],
                                       bright_pct=p["bright_pct"])
        df = build_dataframe(centers, radii, inten_small, binary, bright_red, rel, ps,
                             bg_means=bg_means, bg_stds=bg_stds,
                             posteriors=posteriors,
                             snr_threshold=p["snr_threshold"])

        # Tag with source file
        df.insert(0, "filename", os.path.basename(filename))
        df.insert(1, "filepath", filename)

        n_conf = int((df["min_confidence"] > 0.05).sum()) if "min_confidence" in df.columns else len(df)
        meta.update(n_droplets=len(df), n_classes=int(df.code_int.nunique()),
                    n_confident=n_conf, reliable=rel)
        return df, meta

    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        return None, meta


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(files, params=None, verbose=True):
    """Process a list of CZI files and return combined results.

    Parameters
    ----------
    files   : list of str
    params  : dict, optional  — overrides for DEFAULT_PARAMS
    verbose : bool            — print progress

    Returns
    -------
    combined_df : pd.DataFrame
        All droplets from all images with 'filename' and 'filepath' columns.
    summary_df  : pd.DataFrame
        One row per image with aggregate statistics.
    errors      : list of dict
        Files that failed, with error messages.
    """
    all_dfs, summary_rows, errors = [], [], []

    for i, fpath in enumerate(files, 1):
        name = os.path.basename(fpath)
        if verbose:
            print(f"[{i}/{len(files)}] {name} … ", end="", flush=True)

        df, meta = process_image(fpath, params)

        if df is None:
            if verbose:
                print(f"FAILED — {meta['error']}")
            errors.append(meta)
            continue

        all_dfs.append(df)

        # Per-class count fractions for summary
        cls_counts = df.groupby("code_int").size().reindex(range(16), fill_value=0)
        cls_fracs  = cls_counts / len(df)

        rel    = meta["reliable"]
        n_conf = meta.get("n_confident", len(df))
        row = dict(
            filename        = name,
            filepath        = fpath,
            n_droplets      = len(df),
            n_confident     = n_conf,
            confident_frac  = round(n_conf / len(df), 3) if len(df) else 0.0,
            n_classes       = meta["n_classes"],
            bright_red_n    = int(df.bright_red.sum()),
            bright_red_frac = df.bright_red.mean(),
            mean_radius_um  = df.radius_um.mean(),
            std_radius_um   = df.radius_um.std(),
            mean_volume_um3 = df.volume_um3.mean(),
            std_volume_um3  = df.volume_um3.std(),
            total_volume_um3= df.volume_um3.sum(),
            unreliable_ch   = (", ".join(f"ch{j+1}" for j, r in enumerate(rel) if not r)
                               if rel is not None else ""),
        )
        # Append per-class fraction columns
        for cls in range(16):
            row[f"frac_class{cls:02d}"] = float(cls_fracs[cls])

        summary_rows.append(row)

        if verbose:
            frac = f"{n_conf/len(df)*100:.0f}%" if len(df) else "—"
            print(f"{len(df)} droplets, {meta['n_classes']}/16 classes, "
                  f"confident: {n_conf} ({frac})")

    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    summary_df  = pd.DataFrame(summary_rows)
    return combined_df, summary_df, errors
