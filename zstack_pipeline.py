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
    min_size             = 20,
    min_dist             = 5,
    excl_factor          = 2.0,
    reliability_sigma    = 1.0,
    thresh_offset_sigma  = 1.0,
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
        if len(centers) == 0:
            meta["error"] = "No droplets detected"; return None, meta

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

        bright_red = detect_bright_red(
            inten_small, bg_means, bg_stds,
            bright_factor=p["bright_factor"], bright_pct=p["bright_pct"])

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
                    n_z_planes_total=n_z, z_spacing_um=z_spacing)
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
        if len(centers) == 0:
            meta["error"] = "No droplets survived cross-z NMS"; return None, meta

        z_best   = _best_focus_plane(img_z, centers, radii, z_planes)
        img_mip  = np.max(img_z, axis=0)
        bg_means, bg_stds = estimate_background(
            img_mip, centers, radii, ps, excl_factor=p["excl_factor"])

        inten_full = _extract_intensities_bestfocus(img_z, centers, radii, z_best, 1.0)
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
            img_z, centers, radii, z_best, p["meas_frac_classify"])
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
                    raw_detections=total_raw)
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


def _axial_extent(profile, z_spacing_um, threshold_frac=0.5):
    """Estimate best-focus plane and axial radius from a z-profile.

    Parameters
    ----------
    profile        : 1-D array of length n_z
    z_spacing_um   : µm per z-step
    threshold_frac : fraction of peak used to define the axial extent

    Returns
    -------
    z_best      : int   — index of the plane with peak intensity
    radius_z_um : float — half-width at threshold_frac of peak (µm)
    n_planes    : int   — number of planes above threshold
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


# ── 6. Option C — MIP-guided + axial profiling ────────────────────────────────

def run_zstack_optionc(filename, params=None):
    """Option C: detect on MIP, then extract per-droplet z-profiles.

    Advantages over Option B:
    - Detection uses MIP (max signal from any plane) → no weak-plane false positives
    - Axial extent estimated from raw intensity profile, not plane-count heuristic
    - Much faster: detect once instead of n_z times

    Extra DataFrame columns:
      centroid_z_um   — z-position of best-focus plane (µm)
      z_best_plane    — integer z-plane index
      n_z_planes      — planes above axial_threshold_frac × peak
      radius_z_um     — axial half-width (µm)
      volume_3d_um3   — oblate-spheroid volume (4/3)π r_xy² r_z
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

        # ── Detect on MIP (best-possible 2D signal) ───────────────────────────
        img_mip = np.max(img_z, axis=0)             # (n_ch, H, W)
        norm    = normalize_image(img_mip)

        centers, radii, _ = detect_droplets(
            norm, ps,
            sigma_override   = p["sigma"],
            minsize_override = p["min_size"],
            mind_override    = p["min_dist"],
            min_circularity  = p["min_circularity"],
            border_margin    = p["border_margin"],
        )
        if len(centers) == 0:
            meta["error"] = "No droplets detected"; return None, meta

        # ── Z-profile per droplet ─────────────────────────────────────────────
        profiles = _axial_profiles(img_z, centers, radii,
                                   meas_frac=p["axial_meas_frac"])

        z_best_arr   = np.zeros(len(centers), dtype=int)
        radius_z_arr = np.zeros(len(centers), dtype=float)
        n_planes_arr = np.zeros(len(centers), dtype=int)

        for i in range(len(centers)):
            zb, rz, np_ = _axial_extent(profiles[i], z_spacing,
                                         threshold_frac=p["axial_threshold_frac"])
            z_best_arr[i]   = zb
            radius_z_arr[i] = rz
            n_planes_arr[i] = np_

        # ── Background from MIP ───────────────────────────────────────────────
        bg_means, bg_stds = estimate_background(
            img_mip, centers, radii, ps, excl_factor=p["excl_factor"])

        # ── GMM thresholds from best-focus intensities ────────────────────────
        inten_full = _extract_intensities_bestfocus(img_z, centers, radii, z_best_arr, 1.0)
        _, threshs, rel, _ = classify_channels(
            inten_full, bg_means, bg_stds,
            reliability_sigma=p["reliability_sigma"], margin_sigma=0.0)

        effective_threshs = threshs + p["thresh_offset_sigma"] * bg_stds
        if p["manual_thresholds"] is not None:
            mt = p["manual_thresholds"]
            if len(mt) != 4:
                raise ValueError(f"manual_thresholds must have 4 values, got {len(mt)}")
            effective_threshs = np.array([float(v) for v in mt])

        # ── Classify at best-focus planes ─────────────────────────────────────
        inten_small = _extract_intensities_bestfocus(
            img_z, centers, radii, z_best_arr, p["meas_frac_classify"])
        binary, _, rel, posteriors = classify_channels(
            inten_small, bg_means, bg_stds,
            reliability_sigma=p["reliability_sigma"],
            margin_sigma     =p["margin_sigma"],
            manual_thresholds=list(effective_threshs))

        bright_red = detect_bright_red(
            inten_small, bg_means, bg_stds,
            bright_factor=p["bright_factor"], bright_pct=p["bright_pct"])

        # ── Build DataFrame ───────────────────────────────────────────────────
        df = build_dataframe(centers, radii, inten_small, binary, bright_red, rel, ps,
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
                    n_z_planes_total=n_z, z_spacing_um=z_spacing)
        return df, meta

    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        return None, meta


# ── 7. Top-level dispatcher ───────────────────────────────────────────────────

def process_zstack(filename, params=None, method="optionc"):
    """Process a z-stack CZI file.

    Parameters
    ----------
    method : "optionc" (default) — MIP-guided + axial profiling (recommended)
             "mip"               — Option A: max-intensity projection only
             "optionb"           — Option B: per-plane detect + cross-z NMS
    """
    if method == "mip":
        return run_zstack_mip(filename, params)
    elif method == "optionb":
        return run_zstack_optionb(filename, params)
    elif method == "optionc":
        return run_zstack_optionc(filename, params)
    else:
        raise ValueError(f"method must be 'mip', 'optionb', or 'optionc', got {method!r}")
