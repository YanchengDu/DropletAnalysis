"""
zstack_pipeline.py
Z-stack analysis for DNA nanostar condensate fluorescence microscopy.

Three analysis modes:
  Option A (mip)     — Max Intensity Projection: collapse z first, run standard 2D pipeline.
  Option B (optionb) — Per-plane detect + cross-z NMS + best-focus classify.
  Option C (optionc) — MIP-guided detect + per-droplet axial profiling (recommended).
                        Detects once on MIP (no weak-plane artifacts), then for each
                        droplet extracts a z-intensity profile to find best focus and
                        axial extent.  Robust to low-signal z-planes.

Usage:
    from zstack_pipeline import process_zstack
    df, meta = process_zstack("sample.czi", method="optionc")
"""

import os
import traceback
import numpy as np
import pandas as pd
import czifile

from droplet_pipeline import (
    normalize_image, detect_droplets,
    measure_droplet_intensities, estimate_background,
    classify_channels, detect_bright_red, build_dataframe,
)


# ── Default parameters ────────────────────────────────────────────────────────

DEFAULT_PARAMS = dict(
    pixel_size_um        = 0.312,
    z_spacing_um         = None,    # None = read from CZI metadata
    sigma                = 1.0,
    min_size             = 100,
    min_dist             = 5,
    excl_factor          = 2.0,
    reliability_sigma    = 1.0,
    thresh_offset_sigma  = 0.0,   # use raw GMM threshold; increase to tighten
    margin_sigma         = 0.0,
    snr_threshold        = 2.0,
    meas_frac_classify   = 0.65,
    bright_factor        = 2.0,
    bright_pct           = 98.0,
    manual_thresholds    = None,
    # Option B only
    cross_z_overlap      = 0.5,
    min_z_planes         = 1,
    # Option C only
    axial_threshold_frac = 0.5,     # fraction of peak intensity for axial FWHM
    axial_meas_frac      = 0.8,     # circle fraction used for z-profiling
    # Option E only — per-plane detect + z-linking
    link_alpha           = 0.7,     # XY link dist = link_alpha × (r_i+r_j)/2
    min_z_planes_e       = 2,       # min planes for a valid 3D droplet
    nms_overlap_e        = 0.0,     # post-link NMS: suppress j when d_xy < r_i + r_j*nms_overlap_e; 0=strict containment
    nms_r_cap_px         = 20,      # cap on EDT radius used in post-link NMS (prevents over-suppression when otsu_scale is low)
    otsu_scale           = 0.85,    # multiply reference-plane Otsu threshold; 0.85 captures outer ring without merging droplets
    link_r_cap_px        = 40,      # max EDT radius used when computing link distance (prevents inflation at low otsu_scale)
    link_max_dist_px     = 10,      # fixed XY link tolerance (px); overrides radius-based criterion when > 0
    link_class_filter    = True,    # only link detections with compatible per-channel class
    link_snr_high        = 3.0,     # SNR >= this → channel ON (+1)
    link_snr_low         = 0.5,     # SNR <= this → channel OFF (-1), else UNCERTAIN (0)
    # Option D only — 3D LoG blob detection
    sigma3d_min_um       = 0.5,     # smallest blob sigma in µm  (~0.7 µm radius)
    sigma3d_max_um       = 6.0,     # largest  blob sigma in µm  (~8.5 µm radius)
    sigma3d_n_scales     = 5,       # number of log-spaced sigma levels
    d3_threshold         = 0.01,    # normalised LoG response threshold (0–1)
    d3_overlap           = 0.5,     # max allowed overlap fraction for 3D NMS
    # Shape filters (zstack only)
    min_circularity      = 0.0,     # 0 = disabled; 0.5 removes rectangular artifacts
    border_margin        = 0,       # pixels to exclude at image edges (scan-run boundaries)
)


# ── 1. Load z-stack ───────────────────────────────────────────────────────────

def _read_z_spacing(czi):
    try:
        import xml.etree.ElementTree as ET
        meta = czi.metadata()
        if isinstance(meta, bytes):
            meta = meta.decode()
        root = ET.fromstring(meta)
        for dist in root.iter("Distance"):
            if dist.get("Id") == "Z":
                val = dist.find("Value")
                if val is not None and val.text:
                    return float(val.text) * 1e6
    except Exception:
        pass
    return 1.0


def load_zstack(filename):
    """Load a z-stack CZI file.

    Returns
    -------
    img_z        : float32 ndarray, shape (n_z, n_ch, H, W)
    z_spacing_um : float
    """
    with czifile.CziFile(filename) as czi:
        axes_str     = czi.axes
        img          = czi.asarray().astype(np.float32)
        z_spacing_um = _read_z_spacing(czi)

    i = 0
    while i < len(axes_str):
        if img.shape[i] == 1 and axes_str[i] not in "ZCYX":
            img      = np.squeeze(img, axis=i)
            axes_str = axes_str[:i] + axes_str[i + 1:]
        else:
            i += 1

    if "Z" not in axes_str:
        target = [a for a in "CYX" if a in axes_str]
        perm   = [axes_str.index(a) for a in target]
        img    = np.transpose(img, perm)[np.newaxis]
    else:
        target = [a for a in "ZCYX" if a in axes_str]
        perm   = [axes_str.index(a) for a in target]
        img    = np.transpose(img, perm)

    return img, z_spacing_um


# ── 2. Option A — Max Intensity Projection ────────────────────────────────────

def run_zstack_mip(filename, params=None):
    p    = {**DEFAULT_PARAMS, **(params or {})}
    meta = dict(filename=filename, n_droplets=0, n_classes=0,
                reliable=None, error=None, method="mip")
    try:
        img_z, z_spacing = load_zstack(filename)
        if p["z_spacing_um"] is not None:
            z_spacing = p["z_spacing_um"]
        n_z = img_z.shape[0]
        ps  = p["pixel_size_um"]

        img_mip = np.max(img_z, axis=0)
        norm    = normalize_image(img_mip)

        centers, radii, _ = detect_droplets(
            norm, ps,
            sigma_override   = p["sigma"],
            minsize_override = p["min_size"],
            mind_override    = p["min_dist"],
            min_circularity  = p["min_circularity"],
            border_margin    = p["border_margin"],
        )
        if len(centers) < 2:
            meta["error"] = (f"Too few droplets detected ({len(centers)}) — "
                             "try lowering Min circularity or Border margin")
            return None, meta

        bg_means, bg_stds = estimate_background(
            img_mip, centers, radii, ps, excl_factor=p["excl_factor"])

        inten_full = measure_droplet_intensities(img_mip, centers, radii, meas_frac=1.0)
        _, threshs, rel, _ = classify_channels(
            inten_full, bg_means, bg_stds,
            reliability_sigma=p["reliability_sigma"], margin_sigma=0.0)

        effective_threshs = threshs + p["thresh_offset_sigma"] * bg_stds
        if p["manual_thresholds"] is not None:
            mt = p["manual_thresholds"]
            if len(mt) != 4:
                raise ValueError(f"manual_thresholds must have 4 values, got {len(mt)}")
            effective_threshs = np.array([float(v) for v in mt])

        inten_small = measure_droplet_intensities(
            img_mip, centers, radii, meas_frac=p["meas_frac_classify"])
        binary, _, rel, posteriors = classify_channels(
            inten_small, bg_means, bg_stds,
            reliability_sigma=p["reliability_sigma"],
            margin_sigma     =p["margin_sigma"],
            manual_thresholds=list(effective_threshs))

        # detect_bright_red fits a 3-component GMM — needs at least 3 droplets
        if len(centers) >= 3:
            bright_red = detect_bright_red(
                inten_small, bg_means, bg_stds,
                bright_factor=p["bright_factor"], bright_pct=p["bright_pct"])
        else:
            bright_red = np.zeros(len(centers), dtype=bool)

        df = build_dataframe(centers, radii, inten_small, binary, bright_red, rel, ps,
                             bg_means=bg_means, bg_stds=bg_stds,
                             posteriors=posteriors, snr_threshold=p["snr_threshold"])
        df.insert(0, "filename", os.path.basename(filename))
        df.insert(1, "filepath", filename)
        df["n_z_planes_total"] = n_z
        df["z_spacing_um"]     = z_spacing

        n_conf = int((df["min_confidence"] > 0.05).sum()) if "min_confidence" in df.columns else len(df)
        meta.update(n_droplets=len(df), n_classes=int(df.code_int.nunique()),
                    n_confident=n_conf, reliable=rel,
                    n_z_planes_total=n_z, z_spacing_um=z_spacing,
                    thresholds=list(effective_threshs))
        return df, meta

    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        return None, meta


# ── 3. Option B helpers ───────────────────────────────────────────────────────

def _detect_all_planes(img_z, ps, p):
    results = []
    for z in range(img_z.shape[0]):
        norm = normalize_image(img_z[z])
        centers, radii, _ = detect_droplets(
            norm, ps,
            sigma_override   = p["sigma"],
            minsize_override = p["min_size"],
            mind_override    = p["min_dist"],
            min_circularity  = p["min_circularity"],
            border_margin    = p["border_margin"],
        )
        results.append((centers, radii))
    return results


def _cross_z_nms(detections_per_z, overlap_frac=0.5, min_z_planes=1):
    all_dets = []
    for z, (centers, radii) in enumerate(detections_per_z):
        for (cy, cx), r in zip(centers, radii):
            all_dets.append((z, int(cy), int(cx), float(r)))

    if not all_dets:
        return (np.empty((0, 2), int), np.empty(0, float), [], [])

    all_dets.sort(key=lambda d: -d[3])
    assigned = [False] * len(all_dets)
    groups   = []

    for i, (zi, cyi, cxi, ri) in enumerate(all_dets):
        if assigned[i]: continue
        g_zs  = [zi]; g_cys = [cyi]; g_cxs = [cxi]; g_rs = [ri]
        assigned[i] = True

        for j, (zj, cyj, cxj, rj) in enumerate(all_dets):
            if assigned[j]: continue
            d_xy = np.hypot(cyi - cyj, cxi - cxj)
            r_lo = min(ri, rj); r_hi = max(ri, rj)
            if d_xy < r_hi + r_lo * overlap_frac:
                g_zs.append(zj); g_cys.append(cyj)
                g_cxs.append(cxj); g_rs.append(rj)
                assigned[j] = True

        if len(g_zs) < min_z_planes: continue
        groups.append(dict(
            cy=int(round(np.mean(g_cys))), cx=int(round(np.mean(g_cxs))),
            r_xy=float(np.max(g_rs)), z_planes=g_zs, z_radii=g_rs,
        ))

    if not groups:
        return (np.empty((0, 2), int), np.empty(0, float), [], [])

    unique_centers = np.array([(g["cy"], g["cx"]) for g in groups], dtype=int)
    unique_radii   = np.array([g["r_xy"]           for g in groups], dtype=float)
    z_planes       = [g["z_planes"]                 for g in groups]
    z_radii        = [g["z_radii"]                  for g in groups]
    return unique_centers, unique_radii, z_planes, z_radii


def _best_focus_plane(img_z, centers, radii, z_planes_per_droplet):
    n_z, n_ch, H, W = img_z.shape
    yy, xx   = np.mgrid[0:H, 0:W]
    z_best   = np.zeros(len(centers), dtype=int)
    for i, ((cy, cx), r, zplanes) in enumerate(zip(centers, radii, z_planes_per_droplet)):
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        if not mask.any():
            z_best[i] = zplanes[0]; continue
        best_score = -np.inf; best_z = zplanes[0]
        for z in zplanes:
            score = img_z[z][:, mask].mean()
            if score > best_score:
                best_score = score; best_z = z
        z_best[i] = best_z
    return z_best


def _extract_intensities_bestfocus(img_z, centers, radii, z_best, meas_frac=1.0):
    n_ch   = img_z.shape[1]
    H, W   = img_z.shape[2], img_z.shape[3]
    yy, xx = np.mgrid[0:H, 0:W]
    intensities = np.zeros((len(centers), n_ch), dtype=float)
    for i, ((cy, cx), r, z) in enumerate(zip(centers, radii, z_best)):
        r_meas = r * meas_frac
        mask   = (yy - cy) ** 2 + (xx - cx) ** 2 <= r_meas ** 2
        plane  = img_z[z]
        if mask.any():
            for c in range(n_ch):
                intensities[i, c] = plane[c][mask].mean()
        else:
            for c in range(n_ch):
                intensities[i, c] = plane[c, cy, cx]
    return intensities


# ── 4. Option B ───────────────────────────────────────────────────────────────

def run_zstack_optionb(filename, params=None):
    p    = {**DEFAULT_PARAMS, **(params or {})}
    meta = dict(filename=filename, n_droplets=0, n_classes=0,
                reliable=None, error=None, method="optionb")
    try:
        img_z, z_spacing = load_zstack(filename)
        if p["z_spacing_um"] is not None:
            z_spacing = p["z_spacing_um"]
        n_z, n_ch, H, W = img_z.shape
        ps = p["pixel_size_um"]

        detections = _detect_all_planes(img_z, ps, p)
        total_raw  = sum(len(c) for c, _ in detections)
        if total_raw == 0:
            meta["error"] = "No droplets detected in any z-plane"; return None, meta

        centers, radii, z_planes, z_radii = _cross_z_nms(
            detections, overlap_frac=p["cross_z_overlap"], min_z_planes=p["min_z_planes"])
        if len(centers) < 2:
            meta["error"] = (f"Too few droplets survived cross-z NMS ({len(centers)}) — "
                             "try lowering Min circularity or Border margin")
            return None, meta

        z_best   = _best_focus_plane(img_z, centers, radii, z_planes)
        img_mip  = np.max(img_z, axis=0)
        bg_means, bg_stds = estimate_background(
            img_mip, centers, radii, ps, excl_factor=p["excl_factor"])

        # Use MIP for classification (consistent scale + Voronoi clipping)
        inten_full = measure_droplet_intensities(img_mip, centers, radii, meas_frac=1.0)
        _, threshs, rel, _ = classify_channels(
            inten_full, bg_means, bg_stds,
            reliability_sigma=p["reliability_sigma"], margin_sigma=0.0)

        effective_threshs = threshs + p["thresh_offset_sigma"] * bg_stds
        if p["manual_thresholds"] is not None:
            mt = p["manual_thresholds"]
            if len(mt) != 4:
                raise ValueError(f"manual_thresholds must have 4 values, got {len(mt)}")
            effective_threshs = np.array([float(v) for v in mt])

        inten_small = measure_droplet_intensities(
            img_mip, centers, radii, meas_frac=p["meas_frac_classify"])
        binary, _, rel, posteriors = classify_channels(
            inten_small, bg_means, bg_stds,
            reliability_sigma=p["reliability_sigma"],
            margin_sigma     =p["margin_sigma"],
            manual_thresholds=list(effective_threshs))

        # detect_bright_red fits a 3-component GMM — needs at least 3 droplets
        if len(centers) >= 3:
            bright_red = detect_bright_red(
                inten_small, bg_means, bg_stds,
                bright_factor=p["bright_factor"], bright_pct=p["bright_pct"])
        else:
            bright_red = np.zeros(len(centers), dtype=bool)

        df = build_dataframe(centers, radii, inten_small, binary, bright_red, rel, ps,
                             bg_means=bg_means, bg_stds=bg_stds,
                             posteriors=posteriors, snr_threshold=p["snr_threshold"])

        n_z_per_drop = np.array([len(zp) for zp in z_planes])
        r_z_um  = n_z_per_drop * z_spacing / 2.0
        r_xy_um = df["radius_um"].values
        vol_3d  = (4.0 / 3.0) * np.pi * r_xy_um ** 2 * r_z_um

        df["centroid_z_um"] = z_best * z_spacing
        df["z_best_plane"]  = z_best
        df["n_z_planes"]    = n_z_per_drop
        df["radius_z_um"]   = r_z_um
        df["volume_3d_um3"] = vol_3d
        df.insert(0, "filename", os.path.basename(filename))
        df.insert(1, "filepath", filename)

        n_conf = int((df["min_confidence"] > 0.05).sum()) if "min_confidence" in df.columns else len(df)
        meta.update(n_droplets=len(df), n_classes=int(df.code_int.nunique()),
                    n_confident=n_conf, reliable=rel,
                    n_z_planes_total=n_z, z_spacing_um=z_spacing,
                    raw_detections=total_raw,
                    thresholds=list(effective_threshs))
        return df, meta

    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        return None, meta


# ── 5. Option C helpers ───────────────────────────────────────────────────────

def _axial_profiles(img_z, centers, radii, meas_frac=0.8):

    """Compute z-intensity profile for each droplet.

    Uses raw (un-normalized) pixel values so weak planes stay weak.
    Averages over all channels and all pixels inside the circle.

    Returns
    -------
    profiles : (N, n_z) float array — mean intensity at each z-plane
    """
    n_z, n_ch, H, W = img_z.shape
    yy, xx   = np.mgrid[0:H, 0:W]
    profiles = np.zeros((len(centers), n_z), dtype=float)

    for i, ((cy, cx), r) in enumerate(zip(centers, radii)):
        r_meas = max(1.0, r * meas_frac)
        mask   = (yy - cy) ** 2 + (xx - cx) ** 2 <= r_meas ** 2
        if not mask.any():
            mask[cy, cx] = True
        for z in range(n_z):
            profiles[i, z] = img_z[z][:, mask].mean()

    return profiles



def _find_axial_peaks(profile, z_spacing_um, threshold_frac=0.5, min_sep=2):
    """Find ALL local maxima in a 1-D z-intensity profile.

    Two condensates stacked along the optical axis produce a multi-peaked
    axial profile.  This function returns one entry per peak so the caller
    can create independent droplet entries for each.

    Parameters
    ----------
    profile       : 1-D float array, length n_z
    z_spacing_um  : µm per z-step
    threshold_frac: fraction of global peak used as minimum height
    min_sep       : minimum distance (in z-planes) between two peaks

    Returns
    -------
    list of (z_best, radius_z_um, n_planes) — one per detected peak.
    Always contains at least one entry.
    """
    n       = len(profile)
    peak_v  = float(profile.max())
    if peak_v <= 0:
        return [(int(np.argmax(profile)), z_spacing_um * 0.5, 1)]

    # Light smoothing to suppress single-pixel noise spikes
    if n >= 5:
        kernel = np.array([0.25, 0.5, 0.25])
        smooth = np.convolve(profile.astype(float), kernel, mode="same")
    else:
        smooth = profile.astype(float)

    min_h = threshold_frac * smooth.max()

    # Collect local maxima
    raw = []
    for z in range(n):
        if smooth[z] < min_h:
            continue
        left_ok  = (z == 0)     or (smooth[z] >= smooth[z - 1])
        right_ok = (z == n - 1) or (smooth[z] >= smooth[z + 1])
        if left_ok and right_ok:
            raw.append(z)

    if not raw:
        raw = [int(np.argmax(profile))]

    # Merge peaks closer than min_sep (keep the taller one)
    merged = [raw[0]]
    for z in raw[1:]:
        if z - merged[-1] < min_sep:
            if smooth[z] > smooth[merged[-1]]:
                merged[-1] = z
        else:
            merged.append(z)

    # Build result list
    results = []
    for pk in merged:
        thresh = threshold_frac * profile[pk]
        lo = pk
        while lo > 0 and profile[lo - 1] >= thresh:
            lo -= 1
        hi = pk
        while hi < n - 1 and profile[hi + 1] >= thresh:
            hi += 1
        np_ = hi - lo + 1
        rz  = max(np_ * z_spacing_um / 2.0, z_spacing_um * 0.5)
        results.append((int(pk), float(rz), int(np_)))

    return results


def _expand_by_z_peaks(centers, radii, profiles, z_spacing_um,
                        threshold_frac=0.5, min_sep=2):
    """Expand detected droplets by splitting multi-peak axial profiles.

    For each MIP-detected position, find all peaks in the axial profile.
    If there are N peaks, create N independent droplet entries at the same
    XY coordinates but at their own best-focus z-plane.

    Returns
    -------
    exp_centers   : list of (cy, cx)
    exp_radii     : list of float
    z_best_arr    : np.ndarray int
    radius_z_arr  : np.ndarray float
    n_planes_arr  : np.ndarray int
    n_split       : int — extra droplets created (total new - total original)
    """
    exp_centers, exp_radii = [], []
    z_best_list, radius_z_list, n_planes_list = [], [], []
    n_split = 0

    for i in range(len(centers)):
        peaks = _find_axial_peaks(profiles[i], z_spacing_um,
                                   threshold_frac=threshold_frac,
                                   min_sep=min_sep)
        for zb, rz, np_ in peaks:
            exp_centers.append(centers[i])
            exp_radii.append(radii[i])
            z_best_list.append(zb)
            radius_z_list.append(rz)
            n_planes_list.append(np_)
        if len(peaks) > 1:
            n_split += len(peaks) - 1

    return (exp_centers, exp_radii,
            np.array(z_best_list, dtype=int),
            np.array(radius_z_list, dtype=float),
            np.array(n_planes_list, dtype=int),
            n_split)


def _estimate_background_per_plane(img_z, centers, radii, z_best_arr,
                                    pixel_size_um, excl_factor=2.0):
    """Estimate per-channel background from each droplet's best-focus z-plane.

    Using single z-planes rather than MIP avoids the noise-max inflation
    (MIP background = max of n_z noise values > any individual plane).

    Collects local annular background pixels across all droplets and returns
    global per-channel median and std.

    Returns
    -------
    bg_means : (n_ch,) float
    bg_stds  : (n_ch,) float
    """
    n_ch = img_z.shape[1]
    H, W = img_z.shape[2], img_z.shape[3]
    yy, xx = np.mgrid[0:H, 0:W]

    all_bg = [[] for _ in range(n_ch)]

    for i, ((cy, cx), r, zb) in enumerate(zip(centers, radii, z_best_arr)):
        plane  = img_z[zb]
        r_in   = r * excl_factor
        r_out  = r_in * 2.0
        dist2  = (yy - cy) ** 2 + (xx - cx) ** 2
        mask   = (dist2 > r_in ** 2) & (dist2 <= r_out ** 2)
        if not mask.any():
            mask = dist2 > r_in ** 2
        for c in range(n_ch):
            vals = plane[c][mask]
            if len(vals) > 5:
                all_bg[c].extend(vals.flat)

    bg_means = np.zeros(n_ch, dtype=float)
    bg_stds  = np.zeros(n_ch, dtype=float)
    for c in range(n_ch):
        if all_bg[c]:
            arr        = np.array(all_bg[c], dtype=float)
            bg_means[c] = float(np.median(arr))
            bg_stds[c]  = float(arr.std()) or 1.0
        else:
            # fallback: first plane global stats
            bg_means[c] = float(img_z[0, c].mean())
            bg_stds[c]  = float(img_z[0, c].std()) or 1.0

    return bg_means, bg_stds



def _axial_extent(profile, z_spacing_um, threshold_frac=0.5):
    """Estimate best-focus plane and axial radius from a z-profile.

    Parameters
    ----------
    profile        : 1-D array of length n_z
    z_spacing_um   : um per z-step
    threshold_frac : fraction of peak used to define the axial extent

    Returns
    -------
    z_best      : int   -- index of the plane with peak intensity
    radius_z_um : float -- half-width at threshold_frac of peak (um)
    n_planes    : int   -- number of planes above threshold
    """
    z_best = int(np.argmax(profile))
    peak   = float(profile[z_best])
    if peak <= 0:
        return z_best, z_spacing_um * 0.5, 1

    thresh = threshold_frac * peak
    above  = profile >= thresh
    idx    = np.where(above)[0]
    n_planes    = int(len(idx))
    radius_z_um = max(n_planes * z_spacing_um / 2.0, z_spacing_um * 0.5)
    return z_best, radius_z_um, n_planes


# -- 6. Option C -- MIP-guided + axial profiling + z-peak splitting -----------

def run_zstack_optionc(filename, params=None):
    """Option C: detect on MIP, split multi-peak axial profiles, classify per plane.

    Each MIP-detected XY position whose axial profile has multiple peaks
    (two condensates stacked along the optical axis) is split into independent
    droplet entries, each classified on its own best-focus z-plane.

    Extra DataFrame columns:
      centroid_z_um, z_best_plane, n_z_planes, radius_z_um, volume_3d_um3
    """
    p    = {**DEFAULT_PARAMS, **(params or {})}
    meta = dict(filename=filename, n_droplets=0, n_classes=0,
                reliable=None, error=None, method="optionc")
    try:
        img_z, z_spacing = load_zstack(filename)
        if p["z_spacing_um"] is not None:
            z_spacing = p["z_spacing_um"]

        n_z, n_ch, H, W = img_z.shape
        ps = p["pixel_size_um"]

        # -- Detect on MIP ----------------------------------------------------
        img_mip = np.max(img_z, axis=0)
        norm    = normalize_image(img_mip)

        centers, radii, _ = detect_droplets(
            norm, ps,
            sigma_override   = p["sigma"],
            minsize_override = p["min_size"],
            mind_override    = p["min_dist"],
            min_circularity  = p["min_circularity"],
            border_margin    = p["border_margin"],
        )
        if len(centers) < 2:
            meta["error"] = (f"Too few droplets detected ({len(centers)}) -- "
                             "try lowering Min circularity or Border margin")
            return None, meta

        # -- Z-profile per detected position ----------------------------------
        profiles = _axial_profiles(img_z, centers, radii,
                                   meas_frac=p["axial_meas_frac"])

        # -- Split multi-peak profiles -> independent droplet per z-instance --
        (exp_centers, exp_radii,
         z_best_arr, radius_z_arr, n_planes_arr, n_split) = _expand_by_z_peaks(
            centers, radii, profiles, z_spacing,
            threshold_frac=p["axial_threshold_frac"])

        if len(exp_centers) < 2:
            meta["error"] = (f"Too few droplets after z-expansion ({len(exp_centers)}) -- "
                             "try lowering Axial threshold or Min circularity")
            return None, meta

        # -- Per-plane background (fixes MIP noise inflation) -----------------
        bg_means, bg_stds = _estimate_background_per_plane(
            img_z, exp_centers, exp_radii, z_best_arr, ps,
            excl_factor=p["excl_factor"])

        # -- Intensities from best-focus plane per droplet --------------------
        inten_full = _extract_intensities_bestfocus(
            img_z, exp_centers, exp_radii, z_best_arr, meas_frac=1.0)
        _, threshs, rel, _ = classify_channels(
            inten_full, bg_means, bg_stds,
            reliability_sigma=p["reliability_sigma"], margin_sigma=0.0)

        effective_threshs = threshs + p["thresh_offset_sigma"] * bg_stds
        if p["manual_thresholds"] is not None:
            mt = p["manual_thresholds"]
            if len(mt) != 4:
                raise ValueError(f"manual_thresholds must have 4 values, got {len(mt)}")
            effective_threshs = np.array([float(v) for v in mt])

        inten_small = _extract_intensities_bestfocus(
            img_z, exp_centers, exp_radii, z_best_arr,
            meas_frac=p["meas_frac_classify"])
        binary, _, rel, posteriors = classify_channels(
            inten_small, bg_means, bg_stds,
            reliability_sigma=p["reliability_sigma"],
            margin_sigma     =p["margin_sigma"],
            manual_thresholds=list(effective_threshs))

        if len(exp_centers) >= 3:
            bright_red = detect_bright_red(
                inten_small, bg_means, bg_stds,
                bright_factor=p["bright_factor"], bright_pct=p["bright_pct"])
        else:
            bright_red = np.zeros(len(exp_centers), dtype=bool)

        # -- Build DataFrame --------------------------------------------------
        df = build_dataframe(exp_centers, exp_radii, inten_small, binary,
                             bright_red, rel, ps,
                             bg_means=bg_means, bg_stds=bg_stds,
                             posteriors=posteriors, snr_threshold=p["snr_threshold"])

        r_xy_um = df["radius_um"].values
        vol_3d  = (4.0 / 3.0) * np.pi * r_xy_um ** 2 * radius_z_arr

        df["centroid_z_um"] = z_best_arr * z_spacing
        df["z_best_plane"]  = z_best_arr
        df["n_z_planes"]    = n_planes_arr
        df["radius_z_um"]   = radius_z_arr
        df["volume_3d_um3"] = vol_3d

        df.insert(0, "filename", os.path.basename(filename))
        df.insert(1, "filepath", filename)

        n_conf = int((df["min_confidence"] > 0.05).sum()) if "min_confidence" in df.columns else len(df)
        meta.update(n_droplets=len(df), n_classes=int(df.code_int.nunique()),
                    n_confident=n_conf, reliable=rel,
                    n_z_planes_total=n_z, z_spacing_um=z_spacing,
                    n_split=n_split,
                    thresholds=list(effective_threshs))
        return df, meta

    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        return None, meta


# -- 7. Option D -- anisotropic 3D LoG blob detection -------------------------

def detect_droplets_3d_log(img_z, xy_pixel_um, z_spacing_um,
                            sigma_min_um=0.5, sigma_max_um=3.0, n_scales=5,
                            threshold=0.01, overlap=0.5,
                            min_size_px=10, border_margin=0):
    """Detect 3D blob centres using anisotropic Laplacian-of-Gaussian.

    Operates directly on native anisotropic voxels -- no upsampling.
    For each physical sigma level, applies scipy gaussian_laplace with
    separate sigma_z = sigma_um / z_spacing  and  sigma_xy = sigma_um / xy_pixel
    (in pixel units), so the same physical blob size is described correctly on
    both axes.

    Scale-space responses are sigma^2-normalised so magnitudes are comparable
    across scales; local maxima are found per scale then merged with 3D NMS.

    Parameters
    ----------
    img_z        : (n_z, n_ch, H, W) float32
    xy_pixel_um  : um per xy pixel
    z_spacing_um : um per z-step
    sigma_min_um : smallest blob sigma in um  (~= r_min / sqrt(2))
    sigma_max_um : largest  blob sigma in um  (~= r_max / sqrt(2))
    n_scales     : number of log-spaced sigma levels
    threshold    : minimum normalised LoG response to keep a peak
    overlap      : max allowed blob overlap for 3D NMS (0-1)
    min_size_px  : min xy footprint area (px^2) -- removes sub-pixel noise peaks
    border_margin: pixels to exclude at xy image edges

    Returns
    -------
    centers  : list of (cy, cx) int tuples
    radii    : list of float -- xy radius in pixels (= sigma_um * sqrt(2) / xy_pixel_um)
    z_best   : np.ndarray int -- best-focus z-plane index per blob
    """
    from scipy.ndimage import gaussian_laplace
    try:
        from skimage.feature import peak_local_max
    except ImportError:
        raise ImportError("scikit-image is required for 3D LoG detection")

    n_z, n_ch, H, W = img_z.shape

    # Composite: max across channels, global normalise
    composite = img_z.max(axis=1).astype(np.float64)
    p1, p99   = np.percentile(composite, [1, 99])
    composite = np.clip((composite - p1) / (max(p99 - p1, 1e-9)), 0.0, 1.0)

    # Scale-space LoG
    # Skip scales whose peak radius would always be filtered by min_size_px,
    # saving LoG computation time when min_size_px is large.
    r_min_px = np.sqrt(min_size_px / np.pi)
    sigmas_um_all = np.geomspace(sigma_min_um, sigma_max_um, n_scales)
    sigmas_um = [s for s in sigmas_um_all
                 if s * np.sqrt(2) / xy_pixel_um >= r_min_px]
    if not sigmas_um:
        sigmas_um = [sigmas_um_all[-1]]   # always keep at least the largest scale
    all_blobs = []   # (z, y, x, sigma_um, response)

    for s_um in sigmas_um:
        s_xy = float(s_um / xy_pixel_um)
        s_z  = float(s_um / z_spacing_um)

        # Negative LoG so bright blobs give positive response; sigma^2-normalised
        resp = -gaussian_laplace(composite, sigma=(s_z, s_xy, s_xy)) * (s_um ** 2)

        min_d  = max(1, int(np.ceil(s_xy)))
        coords = peak_local_max(resp, min_distance=min_d,
                                threshold_abs=threshold, exclude_border=False)

        for z, y, x in coords:
            r_px = s_um * np.sqrt(2) / xy_pixel_um
            if np.pi * r_px ** 2 < min_size_px:
                continue
            if (border_margin > 0 and
                    (y < border_margin or y >= H - border_margin or
                     x < border_margin or x >= W - border_margin)):
                continue
            all_blobs.append((int(z), int(y), int(x),
                               float(s_um), float(resp[z, y, x])))

    if not all_blobs:
        return [], [], np.array([], dtype=int)

    blobs = np.array(all_blobs, dtype=float)

    # 3D NMS: stronger response wins; ellipsoidal overlap criterion.
    # Fast KDTree-based O(N log N) implementation -- the naive O(N^2) loop
    # becomes unusably slow (hours) when low thresholds produce >10k candidates.
    from scipy.spatial import KDTree

    # Sort by sigma DESCENDING (large scale first), then by response descending.
    # This ensures large-scale detections are processed before small-scale ones,
    # so they can always suppress ring-artifact sub-detections that appear inside
    # large droplets at smaller sigma.  Sorting by response alone fails because
    # ring artifacts can have stronger normalised response than the large-blob
    # detection, causing them to win the sort order and escape suppression.
    order = np.lexsort((-blobs[:, 4], -blobs[:, 3]))  # primary: -sigma, secondary: -response
    blobs = blobs[order]

    # Scale z-coords so Euclidean distance in scaled space == xy distance.
    # This lets one KDTree query approximate the ellipsoidal overlap check.
    # A blob at sigma s_um has r_xy = s*sqrt2/xy_px and r_z = s*sqrt2/z_sp.
    # Max search radius (when both blobs are at sigma_max_um):
    r_xy_max = sigma_max_um * np.sqrt(2) / xy_pixel_um
    r_z_max  = sigma_max_um * np.sqrt(2) / z_spacing_um
    z_scale  = r_xy_max / max(r_z_max, 1e-9)   # scale z so it matches xy units
    search_r = 2.0 * r_xy_max                   # generous upper bound

    coords_scaled = np.column_stack([
        blobs[:, 1],               # y (px)
        blobs[:, 2],               # x (px)
        blobs[:, 0] * z_scale,     # z scaled to xy units
    ])

    suppressed = np.zeros(len(blobs), dtype=bool)
    tree = KDTree(coords_scaled)

    for i in range(len(blobs)):
        if suppressed[i]:
            continue
        # Find all blobs within max-possible conflict distance
        neighbors = tree.query_ball_point(coords_scaled[i], r=search_r)
        z_i, y_i, x_i, s_i = blobs[i, :4]
        r_xy_i = s_i * np.sqrt(2) / xy_pixel_um
        r_z_i  = s_i * np.sqrt(2) / z_spacing_um
        for j in neighbors:
            if j <= i or suppressed[j]:
                continue
            z_j, y_j, x_j, s_j = blobs[j, :4]
            r_xy_j = s_j * np.sqrt(2) / xy_pixel_um
            r_z_j  = s_j * np.sqrt(2) / z_spacing_um
            dxy = np.sqrt((y_i - y_j) ** 2 + (x_i - x_j) ** 2)
            dz  = abs(z_i - z_j)
            # Standard overlap criterion
            overlap_suppress = (dxy < (1.0 - overlap) * (r_xy_i + r_xy_j) and
                                 dz  < (1.0 - overlap) * (r_z_i  + r_z_j))
            # Containment: j's centre is inside i's ellipsoid -- always suppress
            # regardless of overlap parameter.  Catches small blobs that are
            # sub-detections of a larger droplet at a smaller sigma scale.
            contained = (dxy < r_xy_i and dz < r_z_i)
            if overlap_suppress or contained:
                suppressed[j] = True

    kept_mask = ~suppressed
    if not kept_mask.any():
        return [], [], np.array([], dtype=int)

    kept   = blobs[kept_mask]
    n_z_mx = n_z - 1

    centers = [(int(np.clip(round(b[1]), 0, H - 1)),
                int(np.clip(round(b[2]), 0, W - 1))) for b in kept]
    radii   = [float(b[3] * np.sqrt(2) / xy_pixel_um) for b in kept]
    z_best  = np.array([int(np.clip(round(b[0]), 0, n_z_mx)) for b in kept],
                        dtype=int)
    return centers, radii, z_best


def run_zstack_optiond(filename, params=None):
    """Option D: anisotropic 3D LoG blob detection + per-plane classification.

    Advantages over Option C (MIP + axial profiling):
    - Detects true 3D centres (x, y, z) in a single pass -- no MIP artefacts,
      no z-peak splitting heuristics.
    - Handles dense fields where droplets share the same XY but differ in z.
    - Anisotropic LoG respects the actual voxel geometry without upsampling.
    - Multi-scale (log-spaced sigma_min to sigma_max) catches the full size range.

    Classification: intensities and background both from each droplet's
    best-focus z-plane (same as updated Option C).

    Extra DataFrame columns:
      centroid_z_um, z_best_plane, n_z_planes, radius_z_um, volume_3d_um3
    """
    p    = {**DEFAULT_PARAMS, **(params or {})}
    meta = dict(filename=filename, n_droplets=0, n_classes=0,
                reliable=None, error=None, method="optiond")
    try:
        img_z, z_spacing = load_zstack(filename)
        if p["z_spacing_um"] is not None:
            z_spacing = p["z_spacing_um"]

        n_z, n_ch, H, W = img_z.shape
        ps = p["pixel_size_um"]

        # -- 3D LoG detection -------------------------------------------------
        centers, radii, z_best_arr = detect_droplets_3d_log(
            img_z, ps, z_spacing,
            sigma_min_um  = p["sigma3d_min_um"],
            sigma_max_um  = p["sigma3d_max_um"],
            n_scales      = p["sigma3d_n_scales"],
            threshold     = p["d3_threshold"],
            overlap       = p["d3_overlap"],
            min_size_px   = p["min_size"],
            border_margin = p["border_margin"],
        )

        if len(centers) < 2:
            meta["error"] = (f"Too few blobs detected ({len(centers)}) -- "
                             "try lowering '3D threshold' or widening the sigma range")
            return None, meta

        # -- Per-plane background ---------------------------------------------
        bg_means, bg_stds = _estimate_background_per_plane(
            img_z, centers, radii, z_best_arr, ps,
            excl_factor=p["excl_factor"])

        # -- Intensities from best-focus plane --------------------------------
        inten_full = _extract_intensities_bestfocus(
            img_z, centers, radii, z_best_arr, meas_frac=1.0)
        _, threshs, rel, _ = classify_channels(
            inten_full, bg_means, bg_stds,
            reliability_sigma=p["reliability_sigma"], margin_sigma=0.0)

        effective_threshs = threshs + p["thresh_offset_sigma"] * bg_stds
        if p["manual_thresholds"] is not None:
            mt = p["manual_thresholds"]
            if len(mt) != 4:
                raise ValueError(f"manual_thresholds must have 4 values, got {len(mt)}")
            effective_threshs = np.array([float(v) for v in mt])

        inten_small = _extract_intensities_bestfocus(
            img_z, centers, radii, z_best_arr,
            meas_frac=p["meas_frac_classify"])
        binary, _, rel, posteriors = classify_channels(
            inten_small, bg_means, bg_stds,
            reliability_sigma=p["reliability_sigma"],
            margin_sigma     =p["margin_sigma"],
            manual_thresholds=list(effective_threshs))

        if len(centers) >= 3:
            bright_red = detect_bright_red(
                inten_small, bg_means, bg_stds,
                bright_factor=p["bright_factor"], bright_pct=p["bright_pct"])
        else:
            bright_red = np.zeros(len(centers), dtype=bool)

        # -- Build DataFrame --------------------------------------------------
        df = build_dataframe(centers, radii, inten_small, binary,
                             bright_red, rel, ps,
                             bg_means=bg_means, bg_stds=bg_stds,
                             posteriors=posteriors,
                             snr_threshold=p["snr_threshold"])

        # Axial extent for z-display and 3D volume
        profiles     = _axial_profiles(img_z, centers, radii,
                                        meas_frac=p["axial_meas_frac"])
        radius_z_arr = np.zeros(len(centers), dtype=float)
        n_planes_arr = np.zeros(len(centers), dtype=int)
        for i in range(len(centers)):
            _, rz, np_ = _axial_extent(profiles[i], z_spacing,
                                        threshold_frac=p["axial_threshold_frac"])
            radius_z_arr[i] = rz
            n_planes_arr[i] = np_

        r_xy_um = df["radius_um"].values
        vol_3d  = (4.0 / 3.0) * np.pi * r_xy_um ** 2 * radius_z_arr

        df["centroid_z_um"] = z_best_arr * z_spacing
        df["z_best_plane"]  = z_best_arr
        df["n_z_planes"]    = n_planes_arr
        df["radius_z_um"]   = radius_z_arr
        df["volume_3d_um3"] = vol_3d

        df.insert(0, "filename", os.path.basename(filename))
        df.insert(1, "filepath", filename)


        n_conf = int((df["min_confidence"] > 0.05).sum()) if "min_confidence" in df.columns else len(df)
        meta.update(n_droplets=len(df), n_classes=int(df.code_int.nunique()),
                    n_confident=n_conf, reliable=rel,
                    n_z_planes_total=n_z, z_spacing_um=z_spacing,
                    thresholds=list(effective_threshs))
        return df, meta

    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        return None, meta


# ── 9. Option E helpers ───────────────────────────────────────────────────────

def _global_normalize_zstack(img_z):
    """Percentile-stretch each channel using statistics across ALL z-planes.

    Unlike normalize_image (which normalises each plane independently), this
    gives a consistent intensity scale so the reference-plane Otsu threshold
    means the same thing on every z-plane.

    Returns float32 array, same shape as img_z (n_z, n_ch, H, W).
    """
    n_z, n_ch, H, W = img_z.shape
    out = np.zeros_like(img_z, dtype=np.float32)
    for c in range(n_ch):
        lo = np.percentile(img_z[:, c, :, :], 1)
        hi = np.percentile(img_z[:, c, :, :], 99)
        out[:, c, :, :] = np.clip(
            (img_z[:, c, :, :] - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return out


def _reference_plane_thresholds(norm_z, sigma, otsu_scale=0.85):
    """Find the brightest z-plane and compute per-channel Otsu thresholds on it,
    scaled by otsu_scale.

    Why reference-plane Otsu (not pooled, not median-of-planes)
    ------------------------------------------------------------
    The brightest plane is the equatorial cross-section of the densest
    condensates — it always has a true bimodal background/droplet distribution,
    so Otsu reliably finds a meaningful threshold.

    Pooled-all-planes and median-of-planes both fail because many z-planes are
    empty (no droplets).  Otsu on an empty plane arbitrarily splits a unimodal
    background distribution.  Including these empty-plane values inflates the
    pooled threshold (pooled) or the median, giving a threshold that can be
    *higher* than the reference plane's.  Scaling down from an already-inflated
    value then undershoots the background level → entire image becomes mask →
    distance transform has one peak → only 1–2 detections.

    otsu_scale : float (default 0.85)
        Multiply the reference-plane Otsu threshold before applying to all planes.
        - 1.0  : original equatorial threshold (may clip outer ring of droplets)
        - 0.85 : slightly lenient — captures outer ring, avoids merging
        - 0.7  : more lenient — detects dimmer droplets at risk of merging
        Tuning range: 0.6–1.0.  Go below 0.7 only if droplets are well-separated.

    Returns
    -------
    z_ref      : int         — index of the reference (brightest) z-plane
    thresholds : list[float] — per-channel Otsu threshold × otsu_scale
    """
    from skimage import filters

    mean_per_z = norm_z.mean(axis=(1, 2, 3))   # (n_z,)
    z_ref      = int(np.argmax(mean_per_z))

    thresholds = []
    for c in range(norm_z.shape[1]):
        sm = filters.gaussian(norm_z[z_ref, c], sigma=sigma)
        t  = float(filters.threshold_otsu(sm)) * otsu_scale
        thresholds.append(t)

    return z_ref, thresholds


def _detect_plane_fixed_threshold(norm_plane, ref_thresholds, sigma,
                                   min_size, min_d, min_circularity, border_margin,
                                   nms_r_cap=None):
    """Detect droplets in one plane using pre-computed reference thresholds.

    Identical logic to droplet_pipeline.detect_droplets but replaces per-plane
    Otsu with the supplied global thresholds so detection sensitivity is
    consistent across all z-planes.

    Parameters
    ----------
    norm_plane     : (n_ch, H, W) float32 — globally-normalised plane
    ref_thresholds : list[float] — per-channel Otsu from reference plane

    Returns
    -------
    centers : (N, 2) int
    radii   : (N,)  float
    """
    from skimage import filters, morphology
    from skimage.measure import label as _lbl, regionprops as _rp
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max

    n_ch, H, W = norm_plane.shape

    ch_masks = []
    for c in range(n_ch):
        sm = filters.gaussian(norm_plane[c], sigma=sigma)
        # Use min_size (correct skimage API). max_size is not a valid kwarg.
        m  = morphology.remove_small_objects(sm > ref_thresholds[c], min_size=min_size)
        ch_masks.append(m)
    union_mask = np.logical_or.reduce(ch_masks)

    if not union_mask.any():
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=float)

    _labeled = _lbl(union_mask) if min_circularity > 0 else None

    dist   = ndi.distance_transform_edt(union_mask)
    coords = peak_local_max(dist, min_distance=int(min_d), labels=union_mask)

    if len(coords) == 0:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=float)

    # Peak-level NMS: keep highest-DT peak within min_d of each kept peak
    order  = np.argsort(-dist[coords[:, 0], coords[:, 1]])
    coords = coords[order]
    kept   = []
    for r, c in coords:
        if all(np.hypot(r - r0, c - c0) > min_d for r0, c0 in kept):
            kept.append((r, c))

    centers = np.array(kept, dtype=int)
    radii   = dist[centers[:, 0], centers[:, 1]]

    # Circle-level NMS: suppress j only when its centre lies INSIDE i's radius.
    # Criterion: d < r_i  (strict containment).
    # We do NOT use d < r_i + r_j*0.5 here because in the z-stack fixed-threshold
    # pipeline, lower otsu_scale inflates EDT radii (masks grow into background).
    # The looser criterion would then suppress adjacent condensates that are
    # legitimately separate — causing fewer detections at lower scale.
    # Strict containment correctly handles the only intra-plane duplicate case:
    # two peaks from the same condensate cross-section, where the weaker peak
    # is always within the stronger one's radius.
    if len(centers) > 1:
        order    = np.argsort(-radii)
        centers  = centers[order]
        radii    = radii[order]
        suppress = np.zeros(len(centers), dtype=bool)
        for i in range(len(centers)):
            if suppress[i]:
                continue
            for j in range(i + 1, len(centers)):
                if suppress[j]:
                    continue
                d = np.hypot(centers[i, 0] - centers[j, 0],
                             centers[i, 1] - centers[j, 1])
                r_eff = min(radii[i], nms_r_cap) if nms_r_cap else radii[i]
                if d < r_eff:              # j's centre is inside i — same condensate
                    suppress[j] = True
        centers = centers[~suppress]
        radii   = radii[~suppress]

    # Min-size radius filter
    if len(centers) > 0:
        min_r = np.sqrt(min_size / np.pi)
        keep  = radii >= min_r
        centers = centers[keep]
        radii   = radii[keep]

    # Border margin
    if len(centers) > 0 and border_margin > 0:
        bm   = int(border_margin)
        keep = ((centers[:, 0] >= bm) & (centers[:, 0] < H - bm) &
                (centers[:, 1] >= bm) & (centers[:, 1] < W - bm))
        centers = centers[keep]
        radii   = radii[keep]

    # Circularity filter
    if len(centers) > 0 and min_circularity > 0 and _labeled is not None:
        from skimage.measure import regionprops as _rp2
        _rp_map = {rp.label: rp for rp in _rp2(_labeled)}
        circ_ok = np.ones(len(centers), dtype=bool)
        for k, (r, c) in enumerate(centers):
            lbl = _labeled[r, c]
            if lbl == 0:
                circ_ok[k] = False
                continue
            rp    = _rp_map.get(lbl)
            if rp is None:
                continue
            perim = rp.perimeter
            if perim <= 0:
                continue
            if 4.0 * np.pi * rp.area / (perim ** 2) < min_circularity:
                circ_ok[k] = False
        centers = centers[circ_ok]
        radii   = radii[circ_ok]

    return centers, radii



def _per_plane_classify(norm_plane, centers, radii, meas_frac, snr_high, snr_low):
    """Quick per-plane channel classification using background SNR.

    Returns classes array of shape (N, n_ch) int8:
        +1 = channel ON  (SNR >= snr_high)
        -1 = channel OFF (SNR <= snr_low)
         0 = UNCERTAIN
    """
    if len(centers) == 0:
        n_ch = norm_plane.shape[0]
        return np.empty((0, n_ch), dtype=np.int8)
    n_ch, H, W = norm_plane.shape
    N = len(centers)
    classes = np.zeros((N, n_ch), dtype=np.int8)

    # Build union mask of all condensate discs for background estimation
    bg_mask = np.ones((H, W), dtype=bool)
    cy_arr = np.array([c[0] for c in centers])
    cx_arr = np.array([c[1] for c in centers])
    r_arr  = np.array(radii, dtype=float)
    yy, xx = np.ogrid[:H, :W]
    for k in range(N):
        r_k = max(1.0, r_arr[k] * meas_frac)
        disc = (yy - cy_arr[k])**2 + (xx - cx_arr[k])**2 <= r_k**2
        bg_mask &= ~disc

    for c in range(n_ch):
        plane_c = norm_plane[c]
        bg_pixels = plane_c[bg_mask]
        if len(bg_pixels) < 10:
            continue
        bg_med = float(np.median(bg_pixels))
        bg_std = float(np.std(bg_pixels))
        if bg_std < 1e-9:
            continue
        for k in range(N):
            r_k = max(1.0, r_arr[k] * meas_frac)
            disc = (yy - cy_arr[k])**2 + (xx - cx_arr[k])**2 <= r_k**2
            inten = float(plane_c[disc].mean())
            snr = (inten - bg_med) / bg_std
            if snr >= snr_high:
                classes[k, c] = np.int8(1)
            elif snr <= snr_low:
                classes[k, c] = np.int8(-1)
    return classes


def _link_z_detections(detections_per_z, link_alpha=0.7, min_z_planes=2, link_r_cap=40, link_max_dist=10,
                        classes_per_z=None):
    """Group per-plane detections into 3D droplets via adjacent-z linking.

    Two detections in adjacent z-planes are linked when:
        dist_xy(center_i, center_j) <= link_alpha * (r_i + r_j) / 2

    Connected components of the resulting graph = individual 3D droplets.
    z_center is the plane with the largest detected radius (equatorial plane
    of a filled sphere has the largest cross-section).

    Parameters
    ----------
    detections_per_z : list of (centers (N,2), radii (N,)) tuples
    link_alpha       : float — XY link tolerance as fraction of mean radius
    min_z_planes     : int  — discard droplets seen in fewer planes than this

    Returns
    -------
    list of dicts with keys:
        cy, cx       : centroid at z_center (int)
        r_center     : radius at z_center   (float)
        z_center     : equatorial plane index (int)
        z_min, z_max : z extent (int)
        n_planes     : unique z-planes detected (int)
    """
    from collections import defaultdict

    nodes      = []       # list of (z, cy, cx, r, class_vec_or_None)
    z_to_nodes = {}       # z -> [(global_idx, cy, cx, r)]

    for z, (centers, radii) in enumerate(detections_per_z):
        cls_z = classes_per_z[z] if classes_per_z is not None else None
        for k, ((cy, cx), r) in enumerate(zip(centers, radii)):
            idx = len(nodes)
            cls_k = cls_z[k] if cls_z is not None else None
            nodes.append((z, int(cy), int(cx), float(r), cls_k))
            z_to_nodes.setdefault(z, []).append((idx, int(cy), int(cx), float(r)))

    if not nodes:
        return []

    parent = list(range(len(nodes)))

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(a, b):
        pa, pb = _find(a), _find(b)
        if pa != pb:
            parent[pb] = pa

    # Build edges between adjacent planes only.
    # Two modes:
    #   link_max_dist > 0 : fixed absolute pixel tolerance (preferred — stable
    #                        across all otsu_scale values because it does not
    #                        depend on EDT radius, which inflates at lower thresholds)
    #   link_max_dist == 0: radius-scaled criterion link_alpha*(ri+rj)/2 with
    #                        cap at link_r_cap (legacy behaviour)
    for z in range(len(detections_per_z) - 1):
        for idx_j, cyj, cxj, rj in z_to_nodes.get(z + 1, []):
            # Collect all plane-z nodes close enough to link with this plane-(z+1) node
            tol = link_max_dist if link_max_dist > 0 else 0
            candidates = []
            for idx_i, cyi, cxi, ri in z_to_nodes.get(z, []):
                d_xy = np.hypot(cyi - cyj, cxi - cxj)
                if link_max_dist == 0:
                    tol = link_alpha * (min(ri, link_r_cap) + min(rj, link_r_cap)) / 2.0
                if d_xy <= tol:
                    candidates.append(idx_i)
            if len(candidates) == 0:
                continue
            # Skip if this detection would bridge two DIFFERENT groups in plane z.
            # That indicates a merged mask bridging two separate 3D tracks.
            roots = {_find(c) for c in candidates}
            if len(roots) > 1:
                continue   # ambiguous — do not fuse separate tracks
            for idx_i in candidates:
                # Class-compatibility: only block definitive contradictions (ON vs OFF).
                # UNCERTAIN (0) is compatible with both ON and OFF.
                if classes_per_z is not None:
                    cls_i = nodes[idx_i][4]
                    cls_j = nodes[idx_j][4]
                    if cls_i is not None and cls_j is not None:
                        contradiction = (np.any((cls_i == 1) & (cls_j == -1)) or
                                         np.any((cls_i == -1) & (cls_j == 1)))
                        if contradiction:
                            continue  # different condensate class — do not link
                _union(idx_i, idx_j)

    # Collect connected components
    components = defaultdict(list)
    for idx, node in enumerate(nodes):
        components[_find(idx)].append(node)

    results = []
    for comp_nodes in components.values():
        unique_z = list({n[0] for n in comp_nodes})
        if len(unique_z) < min_z_planes:
            continue
        # Equatorial plane = largest radius
        best     = max(comp_nodes, key=lambda n: n[3])
        results.append(dict(
            cy=best[1], cx=best[2], r_center=best[3],
            z_center=best[0],
            z_min=min(unique_z), z_max=max(unique_z),
            n_planes=len(unique_z),
        ))

    return results


# ── 10. Option E ──────────────────────────────────────────────────────────────

def run_zstack_optione(filename, params=None):
    """Option E: per-plane 2D detection with global reference threshold + z-linking.

    Design
    ------
    1. Global normalisation  — percentile-stretch per channel across all z-planes,
       so intensity values are comparable between planes.
    2. Reference threshold   — find z_ref (highest mean intensity = equatorial
       plane of the densest condensates), compute per-channel Otsu there.
       Apply the same threshold to every plane: dim off-equator planes yield
       smaller circles or no detection, which is physically correct.
    3. Per-plane detection   — same pipeline as single-image analysis
       (Gaussian blur → union Otsu mask → distance transform → peak_local_max
       → circle NMS), but driven by the global threshold.
    4. Z-linking             — adjacent-plane greedy union-find: two detections
       are linked when their XY centres are within link_alpha × mean_radius.
    5. Z-centre from max r   — filled spheres are largest at their equator.
       z_center = plane with maximum detected radius.  No heuristics needed.
    6. Classify at z_center  — intensities, background, and GMM all from each
       droplet's own equatorial plane.

    Extra DataFrame columns
    -----------------------
    centroid_z_um, z_center_plane, z_min_plane, z_max_plane,
    n_z_planes, z_extent_um, radius_z_um, volume_3d_um3, z_ref_plane
    """
    p    = {**DEFAULT_PARAMS, **(params or {})}
    meta = dict(filename=filename, n_droplets=0, n_classes=0,
                reliable=None, error=None, method="optione")
    try:
        img_z, z_spacing = load_zstack(filename)
        if p["z_spacing_um"] is not None:
            z_spacing = p["z_spacing_um"]
        n_z, n_ch, H, W = img_z.shape
        ps = p["pixel_size_um"]

        # Step 1 — global normalisation
        norm_z = _global_normalize_zstack(img_z)

        # Step 2 — reference plane threshold (reference-plane Otsu × otsu_scale)
        otsu_sc = p.get("otsu_scale", 0.85)
        z_ref, ref_thresholds = _reference_plane_thresholds(
            norm_z, p["sigma"], otsu_scale=otsu_sc)

        # Step 3 — per-plane detection with fixed thresholds
        detections_per_z = []
        for z in range(n_z):
            cen, rad = _detect_plane_fixed_threshold(
                norm_z[z], ref_thresholds,
                sigma           = p["sigma"],
                min_size        = p["min_size"],
                min_d           = p["min_dist"],
                min_circularity = p["min_circularity"],
                border_margin   = p["border_margin"],
                nms_r_cap       = p.get("nms_r_cap_px", 20),
            )
            detections_per_z.append((cen, rad))

        planes_with_dets = sum(1 for c, _ in detections_per_z if len(c) > 0)
        total_raw = sum(len(c) for c, _ in detections_per_z)

        if total_raw == 0:
            meta["error"] = "No droplets detected in any z-plane"
            return None, meta

        # Step 3b — per-plane pre-classification for class-constrained linking
        classes_per_z = None
        if p.get("link_class_filter", True):
            classes_per_z = []
            for z, (cen, rad) in enumerate(detections_per_z):
                if len(cen) == 0:
                    n_ch = norm_z[z].shape[0]
                    classes_per_z.append(np.empty((0, n_ch), dtype=np.int8))
                else:
                    cls = _per_plane_classify(
                        norm_z[z], cen, rad,
                        meas_frac=p["meas_frac_classify"],
                        snr_high=p.get("link_snr_high", 3.0),
                        snr_low =p.get("link_snr_low",  0.5),
                    )
                    classes_per_z.append(cls)

        # Step 4 — z-linking
        droplets_3d = _link_z_detections(
            detections_per_z,
            link_alpha    = p.get("link_alpha",       0.7),
            min_z_planes  = p.get("min_z_planes_e",   2),
            link_r_cap    = p.get("link_r_cap_px",    40),
            link_max_dist = p.get("link_max_dist_px", 10),
            classes_per_z = classes_per_z,
        )
        meta["diag_raw_dets"]    = total_raw
        meta["diag_active_planes"] = planes_with_dets

        if len(droplets_3d) < 2:
            meta["error"] = (
                f"Too few 3D droplets found ({len(droplets_3d)}) after z-linking — "
                "try lowering 'Min z-planes (E)' or 'Border margin'")
            return None, meta

        # Post-linking 2D NMS: remove duplicate 3D detections of the same condensate.
        # Criterion: suppress j when d_xy < r_i + r_j * nms_overlap_e.
        # Default nms_overlap_e=0.0 → strict containment (d_xy < r_i).
        # This avoids falsely suppressing adjacent condensates when EDT radii are
        # inflated by a lower otsu_scale.  Increase nms_overlap_e (e.g. to 0.3)
        # only if you observe duplicate 3D detections for the same condensate.
        # Sort largest-first so the dominant detection wins.
        nms_ov   = p.get("nms_overlap_e", 0.0)
        nms_rcap = p.get("nms_r_cap_px",  20)
        n_before_nms = len(droplets_3d)
        if len(droplets_3d) > 1 and nms_ov >= 0:
            droplets_3d.sort(key=lambda d: -d["r_center"])
            suppress = [False] * len(droplets_3d)
            for _i in range(len(droplets_3d)):
                if suppress[_i]:
                    continue
                for _j in range(_i + 1, len(droplets_3d)):
                    if suppress[_j]:
                        continue
                    d_xy   = np.hypot(droplets_3d[_i]["cy"] - droplets_3d[_j]["cy"],
                                      droplets_3d[_i]["cx"] - droplets_3d[_j]["cx"])
                    ri_eff = min(droplets_3d[_i]["r_center"], nms_rcap)
                    rj_eff = min(droplets_3d[_j]["r_center"], nms_rcap)
                    if d_xy < ri_eff + rj_eff * nms_ov:
                        suppress[_j] = True
            droplets_3d = [d for d, s in zip(droplets_3d, suppress) if not s]
        meta["diag_nms_suppressed"] = n_before_nms - len(droplets_3d)

        if len(droplets_3d) < 2:
            meta["error"] = (
                f"Too few 3D droplets remaining ({len(droplets_3d)}) after NMS "
                "-- try increasing 'NMS overlap E' or lowering 'Min z-planes (E)'")
            return None, meta

        centers   = np.array([(d["cy"], d["cx"]) for d in droplets_3d], dtype=int)
        radii     = np.array([d["r_center"]        for d in droplets_3d], dtype=float)
        z_centers = np.array([d["z_center"]         for d in droplets_3d], dtype=int)
        z_mins    = np.array([d["z_min"]            for d in droplets_3d], dtype=int)
        z_maxs    = np.array([d["z_max"]            for d in droplets_3d], dtype=int)
        n_planes  = np.array([d["n_planes"]         for d in droplets_3d], dtype=int)

        # Step 5 -- classify at each droplet's z_center (equatorial) plane
        bg_means, bg_stds = _estimate_background_per_plane(
            img_z, centers, radii, z_centers, ps,
            excl_factor=p["excl_factor"])

        inten_full = _extract_intensities_bestfocus(
            img_z, centers, radii, z_centers, meas_frac=1.0)
        _, threshs, rel, _ = classify_channels(
            inten_full, bg_means, bg_stds,
            reliability_sigma=p["reliability_sigma"], margin_sigma=0.0)

        effective_threshs = threshs + p["thresh_offset_sigma"] * bg_stds
        if p["manual_thresholds"] is not None:
            mt = p["manual_thresholds"]
            if len(mt) != 4:
                raise ValueError(
                    f"manual_thresholds must have 4 values, got {len(mt)}")
            effective_threshs = np.array([float(v) for v in mt])

        inten_small = _extract_intensities_bestfocus(
            img_z, centers, radii, z_centers, p["meas_frac_classify"])
        binary, _, rel, posteriors = classify_channels(
            inten_small, bg_means, bg_stds,
            reliability_sigma=p["reliability_sigma"],
            margin_sigma     =p["margin_sigma"],
            manual_thresholds=list(effective_threshs))

        bright_red = detect_bright_red(
            inten_small, bg_means, bg_stds,
            bright_factor=p["bright_factor"], bright_pct=p["bright_pct"])

        df = build_dataframe(centers, radii, inten_small, binary, bright_red, rel, ps,
                             bg_means=bg_means, bg_stds=bg_stds,
                             posteriors=posteriors, snr_threshold=p["snr_threshold"])

        z_extent_um = (z_maxs - z_mins) * z_spacing
        r_z_um      = np.maximum(z_extent_um / 2.0, z_spacing * 0.5)
        r_xy_um     = df["radius_um"].values
        vol_3d      = (4.0 / 3.0) * np.pi * r_xy_um ** 2 * r_z_um

        df["centroid_z_um"]  = z_centers * z_spacing
        df["z_center_plane"] = z_centers
        df["z_min_plane"]    = z_mins
        df["z_max_plane"]    = z_maxs
        df["n_z_planes"]     = n_planes
        df["z_extent_um"]    = z_extent_um
        df["radius_z_um"]    = r_z_um
        df["volume_3d_um3"]  = vol_3d
        df["z_ref_plane"]    = z_ref

        df.insert(0, "filename", os.path.basename(filename))
        df.insert(1, "filepath", filename)

        n_conf = (int((df["min_confidence"] > 0.05).sum())
                  if "min_confidence" in df.columns else len(df))
        meta.update(
            n_droplets        = len(df),
            n_classes         = int(df.code_int.nunique()),
            n_confident       = n_conf,
            reliable          = rel,
            n_z_planes_total  = n_z,
            z_spacing_um      = z_spacing,
            z_ref_plane       = z_ref,
            ref_thresholds    = ref_thresholds,
            raw_detections    = total_raw,
            diag_active_planes= planes_with_dets,
            thresholds        = list(effective_threshs),
        )
        return df, meta

    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        return None, meta


# -- 8. Top-level dispatcher --------------------------------------------------

def process_zstack(filename, params=None, method="optione"):
    """Process a z-stack CZI file.

    Parameters
    ----------
    method : str
        'optione' (default) -- per-plane detect + z-linking (recommended)
        'optionc'           -- MIP-guided + axial profiling
        'optiond'           -- 3-D LoG blob detection
        'optionb'           -- per-plane detect + cross-z NMS
        'mip'               -- max-intensity projection only
    """
    if method == "mip":
        return run_zstack_mip(filename, params)
    elif method == "optionb":
        return run_zstack_optionb(filename, params)
    elif method == "optionc":
        return run_zstack_optionc(filename, params)
    elif method == "optiond":
        return run_zstack_optiond(filename, params)
    elif method == "optione":
        return run_zstack_optione(filename, params)
    else:
        raise ValueError(
            f"Unknown method {method!r}. "
            "Choose from: 'optione', 'optionc', 'optiond', 'optionb', 'mip'.")
