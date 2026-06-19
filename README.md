# DropletAnalysis

Image analysis pipeline for fluorescence microscopy of **DNA nanostar condensates**. Detects, classifies, and quantifies droplets encoded with up to 4 fluorescent channels, producing 4-bit barcodes (classes 0–15) for each droplet. Supports single-image interactive analysis, batch processing, and full 3D z-stack analysis.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements & Installation](#requirements--installation)
- [Workflows](#workflows)
  - [Single-Image Analysis](#1-single-image-analysis)
  - [Batch Analysis](#2-batch-analysis)
  - [Z-Stack Analysis](#3-z-stack-analysis)
- [Launching the UIs](#launching-the-uis)
- [Parameters Reference](#parameters-reference)
  - [Imaging Setup](#imaging-setup)
  - [Detection](#detection-parameters)
  - [Classification](#classification-parameters)
  - [Two-Pass Measurement](#two-pass-intensity-measurement)
  - [Confidence Filtering](#confidence-filtering)
  - [Z-Stack–Specific Parameters](#z-stack-specific-parameters)
- [Run vs Re-classify](#when-to-use--run-vs--re-classify)
- [Saving Results](#saving-results)
- [Barcode Classes](#barcode-classes)

---

## Overview

Each CZI image contains droplets labelled with combinations of fluorescent dyes across 4 channels (red, yellow, cyan, blue). The pipeline:

1. **Loads** the CZI file and normalises each channel independently (1–99th percentile stretch).
2. **Detects** droplets via a union of per-channel Otsu-thresholded masks, followed by distance-transform peak finding and circle-level non-maximum suppression.
3. **Measures** per-droplet intensities in two passes (full radius for GMM fitting, shrunk core for classification).
4. **Estimates background** fluorescence using pixels outside all droplet regions.
5. **Classifies** each channel as ON/OFF per droplet using a Gaussian Mixture Model (GMM) threshold with optional offset and dead-zone.
6. **Assigns a barcode class** (0–15) based on the 4-bit channel combination.
7. **Reports** counts, volumes, SNR, confidence scores, and publication-quality plots.

---

## Repository Structure

```
DropletAnalysis/
├── droplet_pipeline.py       # Core detection, measurement, and classification functions
├── droplet_ui.py             # Interactive single-image Voilà UI
├── run_analysis.ipynb        # Notebook that launches the single-image UI
│
├── batch_pipeline.py         # Batch processing over folders of CZI files
├── run_batch.ipynb           # Notebook that launches the batch UI
│
├── zstack_pipeline.py        # 3D z-stack analysis (Option E: per-plane + z-linking)
├── zstack_ui.py              # Interactive z-stack Voilà UI
├── run_zstack.ipynb          # Notebook that launches the z-stack UI
│
├── Launch Analysis UI.bat    # One-click launcher for single-image UI (port 8866)
├── Launch Batch Analysis.bat # One-click launcher for batch UI
├── Launch Z-stack UI.bat     # One-click launcher for z-stack UI (port 8867)
│
├── parameters_guide.md       # Original parameters reference
└── README.md                 # This file
```

---

## Requirements & Installation

**Python environment** — a conda or virtualenv named `droplet_env` is expected in the repository root.

```bash
pip install numpy pandas scipy scikit-image scikit-learn matplotlib plotly \
            ipywidgets voila czifile
```

Key dependencies:

| Package | Purpose |
|---|---|
| `czifile` | Read Zeiss CZI microscopy files |
| `scikit-image` | Gaussian blur, Otsu threshold, distance transform, peak detection |
| `scikit-learn` | Gaussian Mixture Model for channel classification |
| `ipywidgets` | Interactive slider UI |
| `voila` | Serve notebooks as standalone web apps |
| `plotly` | Interactive overlays and diagnostic plots |

---

## Workflows

### 1. Single-Image Analysis

**Entry point:** `Launch Analysis UI.bat` → opens browser at `http://localhost:8866`

**Steps:**

1. Paste or browse to a `.czi` file path.
2. Set the objective (20×, 40×, or custom µm/px).
3. Adjust **Detection** sliders (σ blur, Otsu scale, min area, min dist) and click **▶ Run**.
4. Inspect the overlay — circles coloured by class, dashed for undecided.
5. Optionally adjust **Classify** sliders and click **↺ Re-classify** (fast — no re-detection).
6. Check the Diagnostics tab for per-channel intensity histograms and threshold positions.
7. Save results with **💾 CSV**, **📷 Image**, or **📊 Save Analysis**.

### 2. Batch Analysis

**Entry point:** `Launch Batch Analysis.bat`

1. Paste or browse to a folder containing `.czi` files (scanned recursively).
2. Set parameters via sliders (same as single-image).
3. Click **▶ Run Batch** — processes all files sequentially, shows per-file progress.
4. Review the combined summary plots (class distribution, droplet counts, volumes).
5. Save with **💾 Save CSVs** (combined + per-image summary) or **📊 Save Plots**.

### 3. Z-Stack Analysis

**Entry point:** `Launch Z-stack UI.bat` → opens browser at `http://localhost:8867`

Uses **Option E** — per-plane detection with class-constrained z-linking:

1. Detect droplets independently on each z-plane using the Otsu-scaled threshold.
2. **Per-plane classify** each detection (SNR-based: ON if SNR ≥ 3.0, OFF if SNR ≤ 0.5, UNCERTAIN otherwise).
3. **Z-link** detections across planes: two detections on adjacent planes are linked if their XY centres are within `link_alpha × (r_i + r_j) / 2`. Links are **blocked** if the class labels contradict (one channel ON in one plane and OFF in the other).
4. Post-link NMS removes duplicate 2D traces of the same 3D condensate.
5. Best-focus plane is selected per condensate; intensities are measured there for 4-bit barcode classification.
6. 3D volume is estimated from the axial extent (z_min to z_max planes × z-spacing × cross-sectional area).

**After running**, the channel threshold boxes are auto-populated from the Otsu-derived values. Edit any box and click **↺ Re-classify** to override thresholds without re-running detection.

---

## Launching the UIs

Double-click the `.bat` files from Windows Explorer, or run from a terminal:

```batch
:: Single-image UI
voila run_analysis.ipynb --port=8866

:: Batch UI
voila run_batch.ipynb --port=8866

:: Z-stack UI
voila run_zstack.ipynb --port=8867
```

---

## Parameters Reference

### Imaging Setup

| Parameter | Default | Description |
|---|---|---|
| **Objective** | 20× | Sets pixel size (µm/px). Scales all physical-unit calculations (radius, volume). |
| **µm/px** | 0.312 | Pixel size in microns. Use "custom" to enter manually. 20× ≈ 0.312, 40× ≈ 0.156. |

---

### Detection Parameters

Control how droplets are found. Adjust first if droplets are missed or if there are too many false positives.

#### σ blur (default 1.0)
Gaussian blur radius (pixels) applied before thresholding.
- **Increase** if noise or internal granularity is detected as separate objects.
- **Decrease** if small dim droplets are being smeared together or missed.

#### Otsu scale (default 1.0; z-stack default 0.85)
Multiplier on the automatically computed Otsu threshold.
- **< 1.0** lowers the threshold → detects more and larger droplets (captures outer fluorescent rings).
- **> 1.0** raises the threshold → more conservative detection, fewer false positives.
- In z-stack mode, 0.85 is recommended to capture the full droplet boundary without merging neighbours.

#### min area (px², default 20)
Minimum pixel count for a connected bright region to be kept.
- **Increase** if tiny noise spots appear as droplets.
- **Decrease** if small droplets are being discarded.

#### min dist (px, default 5)
Minimum centre-to-centre distance between detections.
- **Increase** if clustered droplets are merged into one.
- **Decrease** if closely packed droplets are skipped.

---

### Classification Parameters

Control how each detected droplet is assigned a channel ON/OFF label. Adjust after detection looks good.

#### BG excl ×r (default 2.0)
Exclusion zone radius (multiples of droplet radius) used when estimating background.
- **Increase** if bright halos around droplets contaminate the background estimate.
- **Decrease** in very dense images where background pixels are scarce.

#### Reliability σ (default 1.0)
A channel is flagged unreliable (⚠ in title, reported as −1) when its positive-population mean is below `Reliability σ × bg_std`.
- **Decrease** to treat more channels as classifiable.
- **Increase** to be stricter — flag more dim channels as untrustworthy.

#### Thr offset σ (default 1.0)
Shifts the GMM threshold upward by `Thr offset σ × bg_std`. Makes classification stricter.
- **Increase** to reduce false positives.
- **Decrease toward 0** to use the raw GMM threshold without offset.

#### Undecided zone (margin σ, default 0.0)
Half-width of a dead band around the threshold (in units of `bg_std`). Droplets within `±margin_σ × bg_std` of the threshold are marked "undecided" (dashed circles, "?" label) rather than forced to 0/1.
- **Increase** to flag borderline droplets as uncertain instead of guessing.
- **Keep at 0** for hard 0/1 decisions on all droplets.

#### SNR threshold (default 2.0)
Minimum per-channel signal-to-noise ratio. Droplets with any channel below this are filtered from results. `SNR = (intensity − bg_mean) / bg_std`.
- **Increase** to retain only droplets with clearly detectable signal.
- **Decrease toward 0** to keep low-signal droplets.

#### Bright-red × (default 2.0)
A droplet is labelled bright red (class 0) when the GMM highest-intensity component mean is ≥ `Bright-red ×` times the dim-component mean.
- **Increase** if dim red droplets are incorrectly assigned to class 0.
- **Decrease** if bright red droplets are not being separated.

#### Bright-red pct (default 98.0)
Fallback percentile threshold when the GMM ratio test fails. Droplets above this intensity percentile in the red channel are labelled bright red.
- **Increase toward 99.9** to label only the very brightest.
- **Decrease toward 90** to be more inclusive.

---

### Two-Pass Intensity Measurement

Classification uses two measurement passes:

**Pass 1 — full circle (100% radius):** Intensities are measured across the entire droplet area and used to fit the GMM and determine thresholds. The effective threshold is `GMM_threshold + Thr offset σ × bg_std`.

**Pass 2 — shrunk circle (65% radius):** Intensities are re-measured from the inner core of each droplet, avoiding the boundary region where signal bleeds in from neighbours. These core intensities are used for the actual 0/1 classification against the thresholds from Pass 1.

---

### Confidence Filtering

After classification, each droplet receives a `min_confidence` score based on how far its GMM posterior probability is from 0.5 (the decision boundary). Droplets with `min_confidence > 0.05` are considered **confident**.

- The status bar shows the fraction of confident droplets after each Run or Re-classify.
- **💾 CSV** saves only the confident droplets.
- The Diagnostics histogram shows each channel's intensity distribution relative to the threshold and dead zone.

---

### Z-Stack–Specific Parameters

| Parameter | Default | Description |
|---|---|---|
| **link_alpha** | 0.7 | XY link distance = `link_alpha × (r_i + r_j) / 2`. Controls how far apart two planes' centres can be and still be linked. |
| **min_z_planes** | 2 | Minimum number of z-planes a condensate must span to be kept as a valid 3D object. |
| **nms_overlap_e** | 0.0 | Post-link NMS aggressiveness. 0 = suppress only fully contained circles; increase to suppress more overlapping circles. |
| **link_r_cap_px** | 40 | Cap on the EDT radius used for link-distance computation (prevents inflated radii at low `otsu_scale` from over-linking). |
| **nms_r_cap_px** | 20 | Cap on the EDT radius used in post-link NMS. |
| **link_snr_high** | 3.0 | Per-plane SNR above which a channel is classified ON for z-linking compatibility check. |
| **link_snr_low** | 0.5 | Per-plane SNR below which a channel is classified OFF. Between the two values = UNCERTAIN (no constraint on linking). |

**Channel thresholds (z-stack UI):** After running, the threshold boxes are auto-filled from the Otsu-derived values. Edit any box to override, then click **↺ Re-classify** — no re-detection needed.

---

## When to Use ▶ Run vs ↺ Re-classify

| Action | When to use |
|---|---|
| **▶ Run** | When changing detection parameters (σ blur, Otsu scale, min area, min dist) or loading a new file. Re-runs the full pipeline: detection → measurement → background → GMM → classification. |
| **↺ Re-classify** | When only changing classification parameters (BG excl, Reliability σ, Thr offset σ, Undecided zone, SNR threshold, Bright-red, or threshold overrides). Skips re-detection and reuses existing droplet positions — much faster. |

---

## Saving Results

### Single-Image UI

| Button | Output |
|---|---|
| **💾 CSV** | `<filename>_analysis.csv` — confident droplets only (`min_confidence > 0.05`) |
| **📷 Image** | `<filename>_classified.png` — composite fluorescence image with coloured circles and class labels |
| **📊 Save Analysis** | `<filename>_analysis_01.png` … `_06.png` — all analysis dashboard plots |

### Batch UI

| Button | Output |
|---|---|
| **💾 Save CSVs** | `batch_combined.csv` (all droplets) and `batch_summary.csv` (per-image statistics) saved to the scanned folder |
| **📊 Save Plots** | `batch_plots_01.png` … `_04.png` — batch summary plots |

### Z-Stack UI

| Button | Output |
|---|---|
| **💾 CSV** | `<filename>_zstack_analysis.csv` — 3D droplet table with `x`, `y`, `z_center_plane`, `z_min_plane`, `z_max_plane`, `volume_3d_um3`, class, and per-channel intensities |
| **📷 Save Image** | Overlay image at the current z-slice with circle annotations |
| **📊 Save Analysis** | Analysis dashboard plots |

---

## Barcode Classes

Each droplet is assigned a 4-bit binary code based on which of the 4 fluorescent channels tests positive:

```
ch1 (red) · ch2 (yellow) · ch3 (cyan) · ch4 (blue)

Class  Code    Meaning
  0    ────    Bright red (high red intensity — overrides binary code)
  1    0001    ch4 only
  2    0010    ch3 only
  3    0011    ch3 + ch4
  4    0100    ch2 only
  5    0101    ch2 + ch4
  6    0110    ch2 + ch3
  7    0111    ch2 + ch3 + ch4
  8    1000    ch1 (dim red) only
  9    1001    ch1 + ch4
 10    1010    ch1 + ch3
 11    1011    ch1 + ch3 + ch4
 12    1100    ch1 + ch2
 13    1101    ch1 + ch2 + ch4
 14    1110    ch1 + ch2 + ch3
 15    1111    all four channels
```

Channels flagged as unreliable (low SNR, marked ⚠ in the figure title) are excluded from the binary code — their bit defaults to 0, which may cause misclassification. Check the diagnostics panel when ⚠ appears.

---

*Pipeline developed for DNA nanostar condensate fluorescence microscopy analysis.*
