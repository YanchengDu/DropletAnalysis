"""
droplet_pipeline.py
All image-processing and classification functions for DNA nanostar condensate analysis.
Import with:  from droplet_pipeline import *
"""

import numpy as np
import pandas as pd
import czifile
from scipy import ndimage as ndi
from scipy.stats import norm as sp_norm
from scipy.optimize import brentq
from skimage import filters, morphology
from skimage.feature import peak_local_max
from skimage.measure import label as _sk_label, regionprops as _sk_regionprops
from sklearn.mixture import GaussianMixture


# -- 1. Load & normalise -------------------------------------------------------

def load_image(filename):
    """Load a CZI file and return a float32 array of shape (n_ch, H, W)."""
    img = np.squeeze(czifile.imread(filename)).astype(np.float32)
    assert img.ndim == 3, f"Expected (C,H,W), got {img.shape}"
    return img


def normalize_image(img):
    """Percentile-stretch each channel independently to [0, 1]."""
    out = np.zeros_like(img)
    for c in range(img.shape[0]):
        lo, hi = np.percentile(img[c], [1, 99])
        out[c] = np.clip((img[c] - lo) / (hi - lo + 1e-9), 0, 1)
    return out


# -- 2. Detect droplets -- union of per-channel masks -------------------------

def detect_droplets(norm, pixel_size_um,
                    sigma_override=None,
                    minsize_override=None,
                    mind_override=None,
                    min_circularity=0.0,
                    border_margin=0,
                    otsu_scale=1.0):
    """Return centers (N,2), radii (N,), distance map (H,W).

    Uses a union of per-channel Otsu masks so droplets visible in only one
    dim channel are still detected.

    sigma_override, minsize_override, mind_override replace the pixel-size-
    derived defaults when provided (used by the interactive sliders).

    After peak detection, circle-level NMS is applied: when one detected
    centre falls inside another circle's radius, the smaller circle is
    suppressed.  This prevents one large droplet from being split into many
    small detections.
    """
    n_ch, H, W = norm.shape
    sigma    = (sigma_override   if sigma_override    is not None
                else max(1.0, 2.0 * 0.312 / pixel_size_um))
    min_size = (int(minsize_override) if minsize_override is not None
                else max(20, int(20 * (0.312 / pixel_size_um) ** 2)))
    min_d    = (int(mind_override)    if mind_override    is not None
                else max(5, int(5.0 / pixel_size_um)))

    ch_masks = []
    for c in range(n_ch):
        sm = filters.gaussian(norm[c], sigma=sigma)
        t  = filters.threshold_otsu(sm) * otsu_scale
        m  = morphology.remove_small_objects(sm > t, max_size=min_size - 1)
        ch_masks.append(m)
    union_mask = np.logical_or.reduce(ch_masks)

    # Pre-compute connected-component labels for circularity filter
    _labeled = _sk_label(union_mask) if min_circularity > 0 else None

    dist   = ndi.distance_transform_edt(union_mask)
    coords = peak_local_max(dist, min_distance=min_d, labels=union_mask)

    if len(coords) == 0:
        return np.empty((0, 2), int), np.empty(0), dist

    # Keep highest-distance peak when two centres are within min_d
    order  = np.argsort(-dist[coords[:, 0], coords[:, 1]])
    coords = coords[order]
    kept = []
    for r, c in coords:
        if all(np.hypot(r - r0, c - c0) > min_d for r0, c0 in kept):
            kept.append((r, c))

    centers = np.array(kept, dtype=int)
    radii   = dist[centers[:, 0], centers[:, 1]]

    # -- Circle-level NMS: suppress smaller circle when it lies inside (or
    #    substantially overlaps) a larger circle.  The threshold is
    #    radii[i] + 0.5*radii[j]: this handles the common case where the
    #    distance-transform radius of the large droplet slightly underestimates
    #    its visual size (dim outer halo below Otsu threshold), allowing a
    #    small spurious peak near the edge to slip through the strict
    #    centre-inside-circle test.
    if len(centers) > 1:
        order    = np.argsort(-radii)           # largest first
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
                # Suppress j if its centre is within i's radius plus half
                # of j's own radius (small droplet near the edge of a big one)
                if d < radii[i] + radii[j] * 0.5:
                    suppress[j] = True
        centers = centers[~suppress]
        radii   = radii[~suppress]

    # -- Radius filter derived from min_size: r_min = sqrt(min_size / pi) --
    # Same threshold as remove_small_objects, expressed as an equivalent
    # circle radius, so one slider controls both mask and detection cutoffs.
    if len(centers) > 0:
        min_r = np.sqrt(min_size / np.pi)
        keep  = radii >= min_r
        centers = centers[keep]
        radii   = radii[keep]

    # -- Border-margin filter: drop detections near image edges ---------------
    if len(centers) > 0 and border_margin > 0:
        bm = int(border_margin)
        keep = ((centers[:, 0] >= bm) & (centers[:, 0] < H - bm) &
                (centers[:, 1] >= bm) & (centers[:, 1] < W - bm))
        centers = centers[keep]
        radii   = radii[keep]

    # -- Circularity filter: 4*pi*area/perimeter^2 (circle=1, rectangle~0.79) -
    # Removes scan-boundary lines and tile-edge rectangles that pass the size
    # and NMS filters but whose connected component is clearly non-circular.
    if len(centers) > 0 and min_circularity > 0 and _labeled is not None:
        _rp_map = {rp.label: rp for rp in _sk_regionprops(_labeled)}
        circ_ok = np.ones(len(centers), dtype=bool)
        for k, (r, c) in enumerate(centers):
            lbl = _labeled[r, c]
            if lbl == 0:
                circ_ok[k] = False
                continue
            rp = _rp_map.get(lbl)
            if rp is None:
                continue
            perim = rp.perimeter
            if perim <= 0:
                continue
            circ = 4.0 * np.pi * rp.area / (perim ** 2)
            if circ < min_circularity:
                circ_ok[k] = False
        centers = centers[circ_ok]
        radii   = radii[circ_ok]

    return centers, radii, dist


# -- 3. Per-droplet intensity --------------------------------------------------

def measure_droplet_intensities(img, centers, radii, meas_frac=1.0):
    """Median raw intensity inside each droplet's Voronoi-clipped circular ROI.

    Two contamination sources are removed simultaneously:
    1. Radius shrinkage (meas_frac): the sampling circle is scaled to
       meas_frac x radius, keeping away from the droplet boundary.
    2. Voronoi masking: for every pixel inside the circle, only include it
       if this droplet's centre is the nearest centre (KDTree lookup).
       This eliminates any overlap between adjacent droplets -- pixels in
       the overlap region are assigned exclusively to whichever droplet is
       closer.
    The median (rather than mean) then handles any residual contamination
    from pixels < 50 % of the masked area.

    meas_frac : fraction of detected radius used for the initial circle (0-1).
                The stored radius and display circle are unaffected.
    """
    from scipy.spatial import KDTree

    n_ch, H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    intensities = np.zeros((len(centers), n_ch))

    if len(centers) == 0:
        return intensities

    # Build KDTree over all centres for Voronoi assignment
    tree = KDTree(centers)           # centers are (row, col) = (y, x)

    for i, ((r, c), rad) in enumerate(zip(centers, radii)):
        meas_rad = rad * float(meas_frac)

        # 1. Candidate pixels within the measurement circle
        rows = np.arange(max(0, int(r - meas_rad)), min(H, int(r + meas_rad) + 1))
        cols = np.arange(max(0, int(c - meas_rad)), min(W, int(c + meas_rad) + 1))
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        circ = (rr - r) ** 2 + (cc - c) ** 2 <= meas_rad ** 2
        pts  = np.stack([rr[circ], cc[circ]], axis=1)  # (M, 2)

        if len(pts) == 0:
            pts = np.array([[r, c]])

        # 2. Voronoi clip: keep only pixels closest to this centre
        _, nn_idx = tree.query(pts)
        voronoi   = nn_idx == i
        pts_v     = pts[voronoi]

        if len(pts_v) == 0:
            pts_v = np.array([[r, c]])

        # 3. Mean intensity over the Voronoi-clipped, circle-bounded pixels
        intensities[i] = [img[ch][pts_v[:, 0], pts_v[:, 1]].mean()
                          for ch in range(n_ch)]
    return intensities


# -- 4. Background estimation --------------------------------------------------

def estimate_background(img, centers, radii, pixel_size_um, excl_factor=2.0):
    """Background pixels are at least excl_factor x droplet-radius away from
    every centre, then further dilated to avoid halo contamination."""
    n_ch, H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    excl = np.zeros((H, W), bool)
    for (r, c), rad in zip(centers, radii):
        excl |= (yy - r) ** 2 + (xx - c) ** 2 <= (rad * excl_factor) ** 2
    erode_px = max(3, int(1.0 / pixel_size_um))
    excl     = morphology.dilation(excl, morphology.disk(erode_px))
    bg = ~excl
    if bg.sum() < 500:
        excl2 = np.zeros((H, W), bool)
        for (r, c), rad in zip(centers, radii):
            excl2 |= (yy - r) ** 2 + (xx - c) ** 2 <= (rad * 1.5) ** 2
        bg = ~excl2
    bg_means = np.array([img[ch][bg].mean() for ch in range(n_ch)])
    bg_stds  = np.array([img[ch][bg].std()  for ch in range(n_ch)])
    return bg_means, bg_stds


# -- 5. GMM classification per channel ----------------------------------------

def classify_channels(intensities, bg_means, bg_stds,
                      reliability_sigma=1.0, margin_sigma=0.0,
                      manual_thresholds=None, thresh_offset_sigma=0.0):
    """Fit a 2-component GMM on bg-subtracted per-droplet intensities for each
    channel.  Threshold = intersection of the two Gaussians.

    Parameters
    ----------
    reliability_sigma : float
        Channel is flagged image-unreliable when the bright-component mean
        < reliability_sigma x bg_std.
    margin_sigma : float
        Per-droplet dead zone.  Droplets whose bg-subtracted intensity is
        within margin_sigma x bg_std of the threshold are marked -2 (undecided)
        instead of 0/1.  Set to 0 (default) to disable.
    thresh_offset_sigma : float
        Shift the GMM threshold upward by this many bg_std units before
        classifying.  Positive values make the classifier stricter (harder
        to be labelled bright).  Does not affect manual threshold overrides.
    manual_thresholds : list/array of 4 floats or None, optional
        Per-channel threshold overrides (in bg-subtracted intensity units).
        If manual_thresholds[c] is not None, skip GMM for that channel and
        use the supplied value directly as the threshold.

    Returns
    -------
    binary     : (N, n_ch) int   -- values: 1=bright, 0=dark,
                                           -2=undecided (near threshold)
    thresholds : (n_ch,) float
    reliable   : (n_ch,) bool    -- False when GMM fails for whole image
    posteriors : (N, n_ch) float -- P(bright | intensity); NaN for unreliable ch
    """
    n_ch   = intensities.shape[1]
    bg_sub = intensities - bg_means
    binary     = np.zeros_like(bg_sub, dtype=int)
    thresholds = np.zeros(n_ch)
    reliable   = np.ones(n_ch, dtype=bool)
    posteriors = np.full((len(intensities), n_ch), np.nan)

    for c in range(n_ch):
        vals = bg_sub[:, c]
        if vals.max() <= vals.min():
            reliable[c] = False
            continue

        # -- Manual threshold override: skip GMM ------------------------------
        man = None if manual_thresholds is None else manual_thresholds[c]
        if man is not None:
            thresh = float(man)
            thresholds[c] = thresh
            binary[:, c]  = (vals > thresh).astype(int)
            # Approximate posterior using sigmoid around threshold
            scale = bg_stds[c] if bg_stds[c] > 0 else 1.0
            posteriors[:, c] = 1.0 / (1.0 + np.exp(-(vals - thresh) / (scale * 0.5)))
        else:
            gmm = GaussianMixture(n_components=2, random_state=0,
                                  n_init=10, max_iter=200).fit(vals.reshape(-1, 1))
            means = gmm.means_.flatten()
            order = np.argsort(means)
            m0, m1 = means[order]
            w0, w1 = gmm.weights_[order]
            s0 = np.sqrt(gmm.covariances_[order[0], 0, 0])
            s1 = np.sqrt(gmm.covariances_[order[1], 0, 0])

            try:
                thresh = brentq(
                    lambda x: w0 * sp_norm.pdf(x, m0, s0) - w1 * sp_norm.pdf(x, m1, s1),
                    m0, m1,
                )
            except Exception:
                thresh = (m0 + m1) / 2.0

            thresholds[c] = thresh
            binary[:, c]  = (vals > thresh).astype(int)

            # -- GMM posteriors (B metric) -------------------------------------
            proba = gmm.predict_proba(vals.reshape(-1, 1))   # (N, 2)
            posteriors[:, c] = proba[:, order[1]]             # P(bright component)

            if m1 < reliability_sigma * bg_stds[c]:
                reliable[c] = False

        # -- Dead-zone / undecided (decision-boundary sensitivity) ------------
        if margin_sigma > 0:
            dead = np.abs(vals - thresh) < margin_sigma * bg_stds[c]
            binary[dead, c] = -2

    return binary, thresholds, reliable, posteriors


# -- 6. Bright-red detection ---------------------------------------------------

def detect_bright_red(intensities, bg_means, bg_stds, red_ch=0,
                      bright_factor=2.0, bright_pct=98.0, dom_factor=1.5):
    """3-component GMM on the red channel to separate negative / dim-red /
    bright-red.  A droplet is labelled bright-red only when:
      1. Its red bg-subtracted intensity is in the GMM bright-red component
         (bright-component mean > bright_factor x dim-component mean), AND
      2. Red is DOMINANT: red_bgsub > dom_factor x max(other channels bgsub).
         This prevents bright droplets with signal in all channels from being
         mis-labelled as bright-red.

    Parameters
    ----------
    dom_factor : float
        Dominance threshold.  Red must exceed dom_factor x the largest of the
        other three channels (bg-subtracted).  Default 1.5.
    """
    n_ch = intensities.shape[1]
    bg_sub = intensities - bg_means          # (N, n_ch)
    red_vals = bg_sub[:, red_ch]
    other_chs = [c for c in range(n_ch) if c != red_ch]

    # -- GMM classification on red channel ------------------------------------
    vals = red_vals.reshape(-1, 1)
    vals = red_vals.reshape(-1, 1)
    gmm3    = GaussianMixture(n_components=3, random_state=0,
                               n_init=10, max_iter=200).fit(vals)
    means3  = gmm3.means_.flatten()
    order3  = np.argsort(means3)
    labels3 = gmm3.predict(vals)

    dim_mean    = means3[order3[1]]
    bright_mean = means3[order3[2]]

    if bright_mean > 0 and dim_mean > 0 and bright_mean > bright_factor * dim_mean:
        gmm_bright = (labels3 == order3[2])
    else:
        threshold  = np.percentile(vals, bright_pct)
        gmm_bright = ((red_vals > threshold) &
                      (red_vals > bright_factor * dim_mean))

    # -- Dominance check: red must exceed all other channels ------------------
    if len(other_chs) > 0 and dom_factor > 0:
        max_other  = bg_sub[:, other_chs].max(axis=1)
        red_dominant = red_vals > dom_factor * max_other
    else:
        red_dominant = np.ones(len(intensities), dtype=bool)

    return gmm_bright & red_dominant


# -- 7. Assemble dataframe -----------------------------------------------------

def build_dataframe(centers, radii, intensities, binary, bright_red,
                    reliable, pixel_size_um,
                    bg_means=None, bg_stds=None,
                    posteriors=None, snr_threshold=2.0):
    """Build a tidy DataFrame of one row per droplet.

    New optional columns (when bg_means/bg_stds/posteriors are supplied):
      snr_ch1..4   -- (intensity - bg_mean) / bg_std  per channel  [metric A]
      min_snr      -- minimum SNR across all channels
      snr_reliable -- True when min_snr >= snr_threshold
      posterior_ch1..4 -- P(bright) from GMM               [metric B]
      min_confidence   -- min |posterior - 0.5| (0=most ambiguous, 0.5=certain)
      size_reliable    -- True when radius >= 3 px           [metric C]
      undecided        -- True when any channel is coded -2
    """
    # -- code_int: treat undecided (-2) bits as 0 for bitmask, then override --
    b_clip = np.clip(binary, 0, 1)
    code_ints = (b_clip[:, 0] * 8 + b_clip[:, 1] * 4 +
                 b_clip[:, 2] * 2 + b_clip[:, 3] * 1).astype(int)
    code_ints[bright_red] = 0
    undecided_mask = np.any(binary == -2, axis=1)
    code_ints[undecided_mask] = -2

    # -- per-droplet SNR (metric A) -------------------------------------------
    snr = None
    if bg_means is not None and bg_stds is not None:
        snr = (intensities - bg_means) / (bg_stds + 1e-9)

    rows = []
    for i, ((r, c), rad) in enumerate(zip(centers, radii)):
        row = dict(
            label      = i + 1,
            centroid_y = int(r),
            centroid_x = int(c),
            radius     = float(rad),
            radius_um  = float(rad * pixel_size_um),
            volume_um3 = float((4 / 3) * np.pi * (rad * pixel_size_um) ** 3),
            bright_red = bool(bright_red[i]),
            code_int   = int(code_ints[i]),
            undecided  = bool(undecided_mask[i]),
            intensity_ch1 = float(intensities[i, 0]),
            intensity_ch2 = float(intensities[i, 1]),
            intensity_ch3 = float(intensities[i, 2]),
            intensity_ch4 = float(intensities[i, 3]),
        )
        for ci, ch in enumerate(['binary_ch1', 'binary_ch2', 'binary_ch3', 'binary_ch4']):
            if not reliable[ci]:
                row[ch] = -1
            else:
                row[ch] = int(binary[i, ci])

        if snr is not None:
            for ci, ch in enumerate(['snr_ch1','snr_ch2','snr_ch3','snr_ch4']):
                row[ch] = float(snr[i, ci])
            row['min_snr']      = float(snr[i].min())
            row['snr_reliable'] = bool(snr[i].min() >= snr_threshold)

        if posteriors is not None:
            for ci, ch in enumerate(['posterior_ch1','posterior_ch2',
                                     'posterior_ch3','posterior_ch4']):
                row[ch] = float(posteriors[i, ci])
            conf = np.abs(posteriors[i] - 0.5)
            row['min_confidence'] = float(conf.min())

        row['size_reliable'] = bool(rad >= 3.0)

        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df['reliable_ch1'] = df['binary_ch1'] != -1
        df['reliable_ch2'] = df['binary_ch2'] != -1
        df['reliable_ch3'] = df['binary_ch3'] != -1
        df['reliable_ch4'] = df['binary_ch4'] != -1
    return df


# -- 8. RGB composite helper ---------------------------------------------------

HEX_COLORS = [
    "#ff2020", "#1d1b7d", "#086564", "#15a2ff",
    "#b8b910", "#6a6a69", "#81f66c", "#95d9ab",
    "#530409", "#720351", "#7db8c0", "#798ced",
    "#ffd425", "#faa34b", "#de963e", "#f7e5b6",
]


def make_rgb(norm, centers, radii, code_ints, bright_red,
             alpha_bg=0.35, overlay_alpha=0.55):
    """Return uint8 (H,W,3) composite for the UI."""
    import skimage.filters as _flt

    n_ch, H, W = norm.shape
    # Fluorescence composite: ch1=red, ch2=yellow, ch3=cyan, ch4=blue
    _ch_colors = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=float)
    rgb = np.zeros((H, W, 3), dtype=float)
    for c in range(min(n_ch, 4)):
        rgb += norm[c, :, :, np.newaxis] * _ch_colors[c]
    rgb = _flt.gaussian(rgb, sigma=1.0, channel_axis=-1)
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)
