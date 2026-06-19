# Droplet Analysis — Tunable Parameters Guide

## Imaging Setup

| Parameter | Default | What it does |
|---|---|---|
| **Objective** | 20× | Sets pixel size (µm/px). Affects all physical-unit calculations (radius, volume) and the scale of detection filters. |
| **µm/px** | 0.312 | Pixel size in microns. Use "custom" objective to enter manually. 20× ≈ 0.312, 40× ≈ 0.156. |

---

## Detection Parameters

These control how droplets are found in the image. Adjust these first if droplets are missed or if too many false positives appear.

### σ blur (default 1.0)
Gaussian blur radius (in pixels) applied to each channel before thresholding.
- **Increase** if the pipeline picks up noise or granularity inside droplets as separate objects.
- **Decrease** if small dim droplets are being smeared together or missed.

### min area (px²) (default 20)
Minimum number of pixels for a connected bright region to be kept. Smaller regions are discarded as noise.
- **Increase** if tiny noise spots are being detected as droplets.
- **Decrease** if small or dim droplets are being removed.

### min dist (px) (default 5)
Minimum centre-to-centre distance between two detected droplets (in pixels).
- **Increase** if clustered droplets are being merged into one.
- **Decrease** if closely packed droplets are being skipped.

---

## Classification Parameters

These control how each detected droplet is assigned a barcode class (0–15). Adjust after detection looks good.

### BG excl ×r (default 2.0)
When estimating the background fluorescence level, pixels within this multiple of each droplet's radius are excluded.
- **Increase** if droplets have bright halos that contaminate the background estimate.
- **Decrease** if the image is densely packed and background pixels are scarce.

### Reliability σ (default 1.0)
A channel is flagged "unreliable" (shown as ⚠ in the title) when its positive-population mean is below `Reliability σ × background std`. Unreliable channels are reported as −1 (unknown) instead of 0/1.
- **Decrease** to be more permissive (more channels treated as classifiable).
- **Increase** to be stricter (more channels flagged as too dim to trust).

### Thr offset σ (default 1.0)
Shifts the GMM-derived threshold upward by `Thr offset σ × background std`. This makes classification stricter: a droplet must be further above background to be called positive.
- **Increase** to reduce false positives (more droplets called negative).
- **Decrease** (toward 0) to use the raw GMM threshold with no offset.
- Works together with Undecided zone: offset moves the decision boundary; the zone widens it into a dead band.

### Undecided zone (margin σ, default 0.0)
Half-width of a dead band around the threshold, in units of `background std`. Droplets whose background-subtracted intensity falls within `±margin_σ × bg_std` of the threshold are marked as "undecided" (shown with dashed circles and "?") rather than forced into 0 or 1.
- **Increase** to flag more borderline droplets as uncertain instead of guessing.
- **Keep at 0** to always make a hard 0/1 call.

### SNR threshold (default 2.0)
Minimum signal-to-noise ratio per channel. Droplets with any channel's SNR below this value are filtered out of the final results. SNR = (intensity − background mean) / background std.
- **Increase** to keep only droplets with clearly detectable signal in every channel.
- **Decrease** (toward 0) to retain low-signal droplets.

### Bright-red × (default 2.0)
A droplet is labelled "bright red" (class 0) when the 3-component GMM finds its highest-intensity component has a mean at least `Bright-red ×` times the dim-red component mean.
- **Increase** if dim red droplets are being incorrectly assigned to class 0.
- **Decrease** if bright red droplets are not being separated from dim red.

### Bright-red pct (default 98.0)
Fallback percentile threshold used when the GMM ratio test fails. Droplets above this intensity percentile in the red channel are labelled bright red.
- **Increase** toward 99.9 to label only the very brightest as class 0.
- **Decrease** toward 90 to be more inclusive.

---

## Two-Pass Intensity Measurement

Classification uses two measurement passes to improve accuracy:

**Pass 1** — full circle (100% radius): intensities are measured across the entire droplet area and used solely to fit the GMM and determine thresholds. The threshold offset is then applied: `effective_threshold = GMM_threshold + Thr offset σ × bg_std`.

**Pass 2** — shrunk circle (65% radius): intensities are re-measured from the inner core of each droplet, avoiding the boundary region where signal bleeds in from neighbours. These intensities are used for the actual 0/1 classification against the effective thresholds from Pass 1.

---

## Confidence Filtering

After classification, each droplet receives a `min_confidence` score based on how far its GMM posterior probability is from 0.5 (the decision boundary). Droplets with `min_confidence > 0.05` are considered **confident**.

- The status bar shows the fraction of confident droplets after each Run or Re-classify.
- **💾 CSV** saves only the confident droplets.
- The Diagnostics histogram shows where each channel's intensities fall relative to the threshold and dead zone.

---

## When to use ▶ Run vs ↺ Re-classify

| Action | When to use |
|---|---|
| **▶ Run** | Any time you change detection parameters (σ blur, min area, min dist) or load a new file. Re-runs the full pipeline: detection → intensity measurement → background estimation → GMM → classification. |
| **↺ Re-classify** | When you only change classification parameters (BG excl, Reliability σ, Thr offset σ, Undecided zone, SNR threshold, Bright-red). Skips re-detection and re-uses existing droplet positions and intensities — much faster. |

---

## File Input

Both the single-image and batch UIs accept file/folder paths by direct paste:

- **Single-image UI**: paste a `.czi` file path in the text box above the file picker. Surrounding quotes (from Windows Explorer) are stripped automatically.
- **Batch UI**: paste a folder path in the text box above the folder picker. The pipeline will scan that folder recursively for all `.czi` files.

---

## Saving Results

### Single-image UI
| Button | Output |
|---|---|
| **💾 CSV** | `<filename>_analysis.csv` — confident droplets only (min_confidence > 0.05) |
| **📷 Image** | `<filename>_classified.png` — composite fluorescence image with coloured circles and class labels |
| **📊 Save Analysis** | `<filename>_analysis_01.png` … `_06.png` — all analysis dashboard plots |

### Batch UI
| Button | Output |
|---|---|
| **💾 Save CSVs** | `batch_combined.csv` (all droplets) and `batch_summary.csv` (per-image stats) saved to the current working folder |
| **📊 Save Plots** | `batch_plots_01.png` … `_04.png` — all batch summary plots saved to the scanned data folder |

---

## Barcode Classes

Each droplet is assigned a 4-bit binary code based on which of the 4 fluorescent channels it tests positive for:

```
ch1 (red) · ch2 (yellow) · ch3 (cyan) · ch4 (blue)

Class  Code    Meaning
  0    ────    Bright red (special — high red intensity, overrides binary code)
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

Channels flagged as unreliable (low SNR) are excluded from the code — their bit is set to 0 by default, which may cause misclassification. Check the ⚠ warning in the figure title.
