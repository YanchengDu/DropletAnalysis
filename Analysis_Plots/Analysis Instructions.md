# Analysis Instructions

## Data Loading

- Pool all CSV files within a condition folder (20x and 40x magnifications combined)
- Filter: keep only rows where `min_confidence > 0.05`
- Always display all 16 classes (0-15), even if a class has zero droplets
- Error bars represent standard deviation (ddof=1)

## Condition Groups

**Figure 1 -- PAGE comparison (day 3 vs day 4)**
Compare PAGE-purified sample (day 3) against layering-purified heated conditions (day 4).
Note: PAGE condition has 40x data only.

**Figure 2 -- Cross-group comparison (day 4 only)**
2x2 grid: rows = salt condition (+2 uM salt / no added salt), columns = mixing method (direct mix / heated).

**Figure 3 -- Time course (day 0 / 1 / 4)**
Four groups (2 salt x 2 mix conditions), each showing evolution over available days.

**Figure 3 (by-day variant)**
Same data as Figure 3 but x-axis = day number; one coloured line per class.

---

## Metrics

### Metric A -- Class composition (count fraction)
**Definition:** fraction of total droplets belonging to each class.
```
frac[c] = count(class c) / count(all droplets)
```
**Plot type:** bar chart; y-axis formatted as percentage.

### Metric B -- Mean radius
**Definition:** mean +/- std of `radius_um` per class.
**Plot type:** error-bar scatter; y-axis in micrometres.

### Metric C -- Mean volume
**Definition:** mean +/- std of `volume_um3` per class.
**Plot type:** error-bar scatter; y-axis in um^3 with log scale (volumes span many orders of magnitude).

### Metric D -- Volume fraction (spatial occupancy)
**Definition:** fraction of total condensate volume occupied by each class.
```
vol_frac[c] = sum(volume_um3 for class c) / sum(volume_um3 for all droplets)
```
This differs from Metric A: a rare class with large droplets can have a disproportionately large volume fraction.
**Plot type:** bar chart; y-axis formatted as percentage.

### Metric E -- Droplet density
**Definition:** number of droplets per unit imaged area (um^2), per class.
```
density[c] = count(class c in FOV) / FOV_area_um2
```
Each field of view (FOV, identified by filename + label) is one replicate.
Mean +/- std is computed across FOVs within a condition.

**Handling 20x and 40x:**
20x and 40x FOVs are pooled. Each FOV is normalised by its own physical area
before pooling, so both magnifications contribute on equal footing.
```
20x pixel size : 0.312 um/px  ->  FOV area = (1024 x 0.312)^2 ~ 102073 um^2
40x pixel size : 0.156 um/px  ->  FOV area = (1024 x 0.156)^2 ~  25518 um^2
```
Image size assumed 1024x1024 px (confirmed by centroid_max ~1018 across all files).
To change, update IMAGE_SIZE_PX in utils.py.

**Plot type:** error-bar scatter; y-axis in droplets/um^2.
Legend shows total FOV count (20x + 40x combined) per condition.

---

## Reliability Flagging

Classes with fewer than `MIN_N = 10` droplets in a given condition are flagged as unreliable.
This threshold is set in `utils.py` and applies to all metrics.

- **Bar charts (A, D):** low-N bars annotated with `* N=X` in red above the bar.
  - Narrow/grouped bars: annotation rotated 90 degrees.
  - Single-condition subplots (fig2): annotation horizontal.
- **Error-bar and line plots (B, C, E):** low-N data points shown with an open (hollow) circle overlaid on the marker.
- **By-day line plots (A, D byday):** hollow circle overlaid at low-N time points.
- Legend always includes a `* N<10: unreliable` or `open circle = N<10: unreliable` entry.

---

## Plot Conventions

### Distinguishing conditions / days in grouped bar charts (Metrics A, D)
- Each bar is coloured by **class** (CLASS_COLORS, 16 fixed hex colours matching the pipeline).
- Different conditions or days are distinguished by **alpha transparency**: [1.0, 0.55, 0.25] (most to least opaque).
- Legend uses grey alpha-varied patches labelled with condition name and droplet count N.

### Distinguishing conditions / days in error-bar plots (Metrics B, C, E)
- Points are coloured by **class**.
- Different conditions or days are distinguished by **marker shape**: ["o", "s", "^"] (circle / square / triangle).
- Legend uses grey markers labelled with condition name and N.

### Figure 2 subplot titles
- No floating column/row header text; each subplot title is self-contained:
  "{condition}, {mix_type}\n(N={:,})" for A/B/C/D
  "{condition}, {mix_type}\n(n={} FOVs)" for E.

### Log scale
- Applied to all C-metric figures (mean volume).

---

## Output Files

All figures saved to `Analysis_Plots/output/` at 300 DPI (PNG) and vector (SVG).

| Metric | Fig 1 | Fig 2 | Fig 3 (by class) | Fig 3 (by day) |
|--------|-------|-------|------------------|----------------|
| A | fig1A_page_comparison | fig2A_crossgroup_day4 | fig3A_timecourse | fig3A_byday |
| B | fig1B_page_comparison | fig2B_crossgroup_day4 | fig3B_timecourse | fig3B_byday |
| C | fig1C_page_comparison | fig2C_crossgroup_day4 | fig3C_timecourse | fig3C_byday |
| D | fig1D_page_comparison | fig2D_crossgroup_day4 | fig3D_timecourse | fig3D_byday |
| E | fig1E_page_comparison | fig2E_crossgroup_day4 | fig3E_timecourse | fig3E_byday |

Each entry saves as both `.png` and `.svg`.

## Running

```bash
cd Analysis_Plots/scripts
python run_all.py              # regenerates all 20 figures (A/B/C/D/E x 4)
python metric_a_class_composition.py   # single metric
python metric_e_density.py             # density only
```

## Configurable Parameters (utils.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| MIN_N | 10 | Minimum droplets per class to be considered reliable |
| IMAGE_SIZE_PX | 1024 | Camera image size in pixels (square) |
| PIXEL_SIZE_UM | 20x: 0.312, 40x: 0.156 | Pixel size in um per pixel |
