"""
zstack_ui.py
Interactive analysis UI for z-stack CZI images.

Usage:
    from zstack_ui import launch_zstack_ui
    S = launch_zstack_ui()
"""

import os
import uuid
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, HTML
import matplotlib.pyplot as plt

from droplet_pipeline import HEX_COLORS
from droplet_ui import show_analysis
from zstack_pipeline import process_zstack, load_zstack, DEFAULT_PARAMS

HEX = HEX_COLORS


def launch_zstack_ui():
    """Build and display the interactive z-stack analysis UI.
    Returns the shared state dict S.
    """
    _uid = "zui-" + uuid.uuid4().hex[:6]
    display(HTML(f"""
    <style>
    .{_uid} {{ background-color: #1a1a1a !important; }}
    .{_uid} .p-Widget, .{_uid} .jupyter-widgets,
    .{_uid} .widget-box, .{_uid} .widget-hbox, .{_uid} .widget-vbox
        {{ background: transparent !important; background-color: transparent !important; }}
    .{_uid} .widget-label, .{_uid} .widget-label-basic, .{_uid} label
        {{ background: transparent !important; color: #cccccc !important; }}
    .{_uid} .widget-readout
        {{ background: #2a2a2a !important; color: #cccccc !important;
           border: 1px solid #555 !important; min-width: 38px !important; }}
    .{_uid} input[type=text], .{_uid} input[type=number]
        {{ background: #2a2a2a !important; color: #cccccc !important;
           border: 1px solid #555 !important; }}
    .{_uid} select
        {{ background: #2a2a2a !important; color: #cccccc !important;
           border: 1px solid #555 !important; }}
    .{_uid} .noUi-target
        {{ background: #3a3a3a !important; border-color: #555 !important;
           box-shadow: none !important; }}
    .{_uid} .noUi-connect {{ background: #4a90d9 !important; }}
    .{_uid} .noUi-handle
        {{ background: #4a90d9 !important; border-color: #2a6db5 !important;
           box-shadow: none !important; }}
    .{_uid} .widget-hslider, .{_uid} .widget-slider {{ background: transparent !important; }}
    .{_uid} hr {{ border: none !important; border-top: 1px solid #444 !important;
                  margin: 3px 0 !important; }}
    .{_uid} button, .{_uid} .widget-button, .{_uid} .jupyter-button
        {{ color: #cccccc !important; background-color: #2a2a2a !important;
           border-color: #555 !important; }}
    .{_uid} button:hover, .{_uid} .widget-button:hover
        {{ background-color: #3a3a3a !important; color: #ffffff !important; }}
    .{_uid} .fc-select, .{_uid} .fc-select option, .{_uid} .fc-input, .{_uid} .fc-name
        {{ background: #2a2a2a !important; color: #cccccc !important;
           border: 1px solid #555 !important; }}
    .{_uid} .fc-title, .{_uid} .fc-current-label, .{_uid} .fc-current-value,
    .{_uid} .widget-html, .{_uid} .widget-html-content, .{_uid} .widget-html-content *
        {{ color: #cccccc !important; background: transparent !important; }}
    .{_uid} p, .{_uid} span:not(.noUi-handle):not(.noUi-origin)
        {{ color: #cccccc !important; }}
    .{_uid} input::placeholder {{ color: #666666 !important; opacity: 1 !important; }}
    .{_uid} select option {{ background: #2a2a2a !important; color: #cccccc !important; }}
    /* Tab styling — covers both legacy (p-) and modern (lm-/jp-) class names */
    .{_uid} .p-TabBar,
    .{_uid} .lm-TabBar,
    .{_uid} .jupyter-widgets-tab-bar
        {{ background: #1a1a1a !important; border-bottom: 1px solid #444 !important; }}
    .{_uid} .p-TabBar-tab,
    .{_uid} .lm-TabBar-tab,
    .{_uid} .jupyter-widgets-tab-bar .p-TabBar-tab
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
        "df": None, "img_z": None, "filename": None,
        "pixel_size": 0.312, "z_spacing": 1.0,
        "method": "optione", "n_z": 0, "norm_range": None,
    }

    # ── File picker ───────────────────────────────────────────────────────────
    path_input = widgets.Text(
        placeholder="Paste full file path here",
        description="", layout=widgets.Layout(width="280px"),
    )
    try:
        from ipyfilechooser import FileChooser
        _fc = FileChooser(path=os.getcwd(), filename="", filter_pattern="*.czi",
                          title="Select a z-stack .czi file", show_hidden=False)
        _fc.layout = widgets.Layout(width="280px")
        file_picker_widget = _fc
        def _active_filename():
            txt = path_input.value.strip().strip("\"'")
            return txt if txt else (_fc.selected or "")
    except ImportError:
        import glob as _gl
        _files = sorted(set(os.path.normpath(p)
                            for p in _gl.glob("**/*.czi", recursive=True)))
        file_dd = widgets.Dropdown(
            options=_files or ["(no .czi found)"],
            description="File:", style={"description_width": "35px"},
            layout=widgets.Layout(width="245px"),
        )
        file_picker_widget = file_dd
        def _active_filename():
            txt = path_input.value.strip()
            if txt: return txt
            v = file_dd.value or ""
            return v if v != "(no .czi found)" else ""

    # ── Top controls (always visible) ─────────────────────────────────────────
    mag_dd = widgets.Dropdown(
        options=[("20x  0.312 um/px", 0.312), ("40x  0.156 um/px", 0.156), ("custom", None)],
        value=0.312, description="Obj:",
        style={"description_width": "28px"}, layout=widgets.Layout(width="170px"),
    )
    pxsz_box = widgets.FloatText(
        value=0.312, description="um/px:",
        style={"description_width": "42px"},
        layout=widgets.Layout(width="108px"), disabled=True,
    )
    def _mag_changed(change):
        if change["new"] is None: pxsz_box.disabled = False
        else: pxsz_box.value = change["new"]; pxsz_box.disabled = True
    mag_dd.observe(_mag_changed, names="value")

    zstep_box = widgets.FloatText(
        value=1.0, description="Z step:",
        style={"description_width": "45px"}, layout=widgets.Layout(width="120px"),
    )
    method_dd = widgets.HTML(
        "<b style='font-size:0.8em;color:#aaa'>Method: Option E — per-plane detect + z-linking</b>",
        layout=widgets.Layout(width="280px"),
    )

    # ── Sliders organised into tabs ───────────────────────────────────────────
    _sw = dict(style={"description_width": "115px"}, layout=widgets.Layout(width="270px"))

    # Tab 0: Detection
    sigma_sl   = widgets.FloatSlider(value=1.0, min=0.3, max=5.0,  step=0.1,
                                      description="Sigma blur",      **_sw)
    minsize_sl = widgets.IntSlider(  value=20,  min=5,   max=300,  step=5,
                                      description="Min area (px^2)", **_sw)
    mind_sl    = widgets.IntSlider(  value=5,   min=2,   max=60,   step=1,
                                      description="Min dist (px)",   **_sw)
    circ_sl    = widgets.FloatSlider(value=0.0, min=0.0, max=1.0,  step=0.05,
                                      description="Min circularity", **_sw)
    bmargin_sl = widgets.IntSlider(  value=0,   min=0,   max=150,  step=5,
                                      description="Border margin px", **_sw)
    tab_detect = widgets.VBox([sigma_sl, minsize_sl, mind_sl, circ_sl, bmargin_sl],
                               layout=widgets.Layout(padding="6px"))

    # Tab 1: Classification
    excl_sl   = widgets.FloatSlider(value=2.0, min=1.0, max=5.0,  step=0.1,
                                     description="BG excl x radius", **_sw)
    relth_sl  = widgets.FloatSlider(value=1.0, min=0.0, max=5.0,  step=0.1,
                                     description="Reliability sig",   **_sw)
    margin_sl = widgets.FloatSlider(value=0.0, min=0.0, max=1.0,  step=0.01,
                                     description="Undecided zone",    **_sw)
    snrth_sl  = widgets.FloatSlider(value=2.0, min=0.5, max=10.0, step=0.5,
                                     description="SNR threshold",     **_sw)
    throff_sl = widgets.FloatSlider(value=1.0, min=0.0, max=5.0,  step=0.1,
                                     description="Thr offset sig",    **_sw)
    tab_classify = widgets.VBox([excl_sl, relth_sl, margin_sl, snrth_sl, throff_sl],
                                 layout=widgets.Layout(padding="6px"))

    # Tab 2: 3D / Z
    otsu_sl   = widgets.FloatSlider(value=0.85, min=0.3, max=1.5, step=0.05,
                                     description="Otsu scale",
                                     style={"description_width": "115px"},
                                     layout=widgets.Layout(width="270px"))
    nms_ov_sl = widgets.FloatSlider(value=0.0,  min=0.0, max=1.0, step=0.05,
                                     description="NMS overlap",
                                     style={"description_width": "115px"},
                                     layout=widgets.Layout(width="270px"))
    lkdist_sl = widgets.IntSlider(   value=10,  min=0,   max=40,  step=1,
                                     description="Link max dist px",
                                     style={"description_width": "115px"},
                                     layout=widgets.Layout(width="270px"))
    minzp_sl  = widgets.IntSlider(   value=2,   min=1,   max=10,  step=1,
                                     description="Min z-planes",
                                     style={"description_width": "115px"},
                                     layout=widgets.Layout(width="270px"))
    tab_3d = widgets.VBox([otsu_sl, nms_ov_sl, lkdist_sl, minzp_sl],
                           layout=widgets.Layout(padding="6px"))

    # Tab 3: Bright-red
    brcomp_sl = widgets.FloatSlider(value=2.0, min=1.1, max=5.0,  step=0.1,
                                     description="Bright-red x",    **_sw)
    brpct_sl  = widgets.FloatSlider(value=98.0, min=80.0, max=99.9, step=0.5,
                                     description="Bright-red pct",  **_sw)
    tab_bright = widgets.VBox([brcomp_sl, brpct_sl],
                               layout=widgets.Layout(padding="6px"))

    param_tabs = widgets.Tab(children=[tab_detect, tab_classify, tab_3d, tab_bright])
    for i, title in enumerate(["Detect", "Classify", "3D / Z", "Bright-red"]):
        param_tabs.set_title(i, title)

    # ── Per-channel threshold boxes (right panel) ─────────────────────────────
    _ch_names = ["Ch1 red", "Ch2 yellow", "Ch3 cyan", "Ch4 blue"]
    thresh_sls = [
        widgets.FloatText(value=0.0, step=10.0, description=f"{_ch_names[c]}:",
                          disabled=False,
                          style={"description_width": "80px"},
                          layout=widgets.Layout(width="200px"))
        for c in range(4)
    ]

    # ── Figure widget ─────────────────────────────────────────────────────────
    _BG = "#111111"
    _outer_ref = [None]
    fig = go.FigureWidget(layout=go.Layout(
        width=640, height=640,
        margin=dict(l=0, r=0, t=24, b=0),
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(color="#cccccc"), showlegend=False,
        shapes=[], annotations=[],
        xaxis=dict(range=[0, 512], autorange=False, constrain="domain",
                   showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[512, 0], autorange=False, scaleanchor="x", scaleratio=1,
                   showgrid=False, zeroline=False, showticklabels=False),
        title=dict(text="Select a file and click Run",
                   font=dict(size=12, color="#888"), x=0.01, xanchor="left"),
        dragmode="zoom",
    ))

    def _make_layout(W=512, H=512, title_text="", shapes=None, annotations=None):
        return go.Layout(
            width=640, height=640,
            margin=dict(l=0, r=0, t=24, b=0),
            paper_bgcolor=_BG, plot_bgcolor=_BG,
            font=dict(color="#cccccc"), showlegend=False,
            shapes=shapes or [], annotations=annotations or [],
            xaxis=dict(range=[0, W], autorange=False, constrain="domain",
                       showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[H, 0], autorange=False, scaleanchor="x", scaleratio=1,
                       showgrid=False, zeroline=False, showticklabels=False),
            title=dict(text=title_text, font=dict(size=12, color="#888"),
                       x=0.01, xanchor="left"),
            dragmode="zoom",
        )

    # ── Z-plane slider ────────────────────────────────────────────────────────
    z_sl    = widgets.IntSlider(value=0, min=0, max=0, step=1,
                                 description="Z plane:",
                                 style={"description_width": "58px"},
                                 layout=widgets.Layout(width="460px"))
    z_label = widgets.Label("(run pipeline first)")

    def _build_rgb(z_idx):
        img_z = S.get("img_z")
        if img_z is None: return None, 1, 1
        n_z, n_ch, H, W = img_z.shape
        plane = img_z[z_idx].astype(float)
        rng   = S.get("norm_range")
        if rng is None: return None, H, W
        ch_min, ch_max = rng
        _ch_col = np.array([[1,0,0],[1,1,0],[0,1,1],[0,0,1]], dtype=float)
        rgb = np.zeros((H, W, 3), dtype=float)
        for c in range(min(n_ch, 4)):
            denom = float(ch_max[c] - ch_min[c]) + 1e-9
            norm_c = np.clip((plane[c] - ch_min[c]) / denom, 0, 1)
            rgb += norm_c[:, :, np.newaxis] * _ch_col[c]
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), H, W

    def _overlays_for_z(z_idx):
        df_cur = S.get("df")
        if df_cur is None: return [], [], []
        method = S.get("method", "optione")
        shapes, scatters, annotations = [], [], []
        for _, row in df_cur.iterrows():
            ci    = int(row.code_int)
            is_und = (ci == -2)
            is_br  = bool(row.get("bright_red", False))
            col    = "#aaaaaa" if is_und else ("#ff2020" if is_br else HEX[ci % 16])
            cx, cy, r = row.centroid_x, row.centroid_y, row.radius
            # Determine visibility at this z-plane
            # Option E: show condensate only in its z-range; solid at z_center
            if "z_center_plane" in row.index:
                z_lo    = int(row.get("z_min_plane", row.z_center_plane))
                z_hi    = int(row.get("z_max_plane", row.z_center_plane))
                if not (z_lo <= z_idx <= z_hi): continue
                at_best = (z_idx == int(row.z_center_plane))
                dash    = "dot" if is_und else ("solid" if at_best else "dash")
                lw      = 2 if at_best else 1
                opacity = 1.0 if at_best else 0.35
            else:
                at_best = True; dash = "dot" if is_und else "solid"
                lw = 2; opacity = 1.0
            shapes.append(dict(
                type="circle",
                x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
                line=dict(color=col, width=lw, dash=dash),
                opacity=opacity,
            ))
            hover = (f"ID:{int(row.label)}  class:{'?' if is_und else ci}<br>"
                     f"r={r:.1f}px")
            if "centroid_z_um" in row.index:
                hover += f"  z={row.centroid_z_um:.1f}um"
            if "min_confidence" in row.index and not np.isnan(row.min_confidence):
                hover += f"<br>Conf:{row.min_confidence:.2f}"
            scatters.append(go.Scatter(
                x=[cx], y=[cy], mode="markers",
                marker=dict(size=max(8, int(r * 0.8)), color="rgba(0,0,0,0)", opacity=0),
                hovertext=hover, hoverinfo="text", showlegend=False,
            ))
            if at_best:
                annotations.append(dict(
                    x=cx, y=cy, text="?" if is_und else str(ci),
                    showarrow=False,
                    font=dict(color="white", size=9, family="monospace"),
                    xref="x", yref="y",
                ))
        return shapes, scatters, annotations

    def draw_zplane(z_idx=None):
        nonlocal fig
        if z_idx is None: z_idx = z_sl.value
        rgb, H, W = _build_rgb(z_idx)
        if rgb is None: return
        n_z  = S.get("n_z", 0)
        zsp  = S.get("z_spacing", 1.0)
        z_um = z_idx * zsp
        shapes, scatters, annotations = _overlays_for_z(z_idx)
        df_cur = S.get("df")
        n_vis  = len(shapes)
        n_tot  = len(df_cur) if df_cur is not None else 0
        title  = (f"{os.path.basename(S.get('filename') or '')}  "
                  f"z={z_idx}/{n_z-1 if n_z>0 else 0} ({z_um:.2f}um)  "
                  f"{n_vis}/{n_tot} droplets")

        is_new = S.get("_drawn_file") != S.get("filename")
        if is_new or len(fig.data) == 0:
            new_fig = go.FigureWidget(
                layout=_make_layout(W=W, H=H, title_text=title,
                                    shapes=shapes, annotations=annotations))
            new_fig.add_trace(go.Image(z=rgb))
            if scatters: new_fig.add_traces(scatters)
            if _outer_ref[0] is not None:
                _outer_ref[0].children = (
                    widgets.VBox([new_fig, z_sl, z_label]),
                ) + tuple(_outer_ref[0].children[1:])
            fig = new_fig
            S["_drawn_file"] = S.get("filename")
            for t in fig.data[1:]: t.on_click(_on_click)
        else:
            if len(fig.data) > 1: fig.data = (fig.data[0],)
            if scatters: fig.add_traces(scatters)
            with fig.batch_update():
                fig.data[0].z = rgb
                fig.update_layout(
                    shapes=shapes, annotations=annotations,
                    xaxis=dict(range=[0, W], autorange=False, showgrid=False,
                               zeroline=False, showticklabels=False, constrain="domain"),
                    yaxis=dict(range=[H, 0], autorange=False, showgrid=False,
                               zeroline=False, showticklabels=False,
                               scaleanchor="x", scaleratio=1),
                    title_text=title,
                )
            for t in fig.data[1:]: t.on_click(_on_click)
        z_label.value = f"z = {z_idx} / {n_z-1 if n_z>0 else 0}  ({z_um:.2f} um)"

    z_sl.observe(lambda c: draw_zplane(c["new"]), names="value")

    def _on_click(trace, points, selector):
        if not points.point_inds: return
        df_cur = S["df"]
        if df_cur is None: return
        tidx = list(fig.data).index(trace) - 1
        if 0 <= tidx < len(df_cur):
            row = df_cur.iloc[tidx]
            sel_id_box.value  = int(row.label)
            class_input.value = int(row.code_int) if row.code_int >= 0 else 0

    # ── Buttons ───────────────────────────────────────────────────────────────
    run_btn    = widgets.Button(description="Run",         button_style="primary",
                                 layout=widgets.Layout(width="72px"))
    reclf_btn  = widgets.Button(description="Reclassify",  button_style="info",
                                 layout=widgets.Layout(width="100px"))
    save_btn   = widgets.Button(description="CSV",         button_style="success",
                                 layout=widgets.Layout(width="60px"))
    simg_btn   = widgets.Button(description="Image",       button_style="success",
                                 layout=widgets.Layout(width="72px"))
    sana_btn   = widgets.Button(description="Save Analysis", button_style="success",
                                 layout=widgets.Layout(width="210px"))
    diag_btn   = widgets.Button(description="Diagnostics", button_style="warning",
                                 layout=widgets.Layout(width="105px"))
    view3d_btn = widgets.Button(description="3D View",     button_style="",
                                 layout=widgets.Layout(width="80px"))
    diag_out   = widgets.Output()
    view3d_out = widgets.Output()
    status_lbl = widgets.Label("Ready.", layout=widgets.Layout(width="560px"))

    # ── Run pipeline ──────────────────────────────────────────────────────────
    def run_pipeline_cb(_):
        fname = _active_filename()
        if not fname: status_lbl.value = "No file selected."; return
        if not os.path.isfile(fname):
            status_lbl.value = f"Not found: {fname}"; return
        status_lbl.value = "Loading z-stack ..."
        try:
            ps  = pxsz_box.value
            zsp = zstep_box.value
            mth = "optione"
            S.update(filename=fname, pixel_size=ps, z_spacing=zsp, method=mth)

            img_z, zsp_auto = load_zstack(fname)
            if zsp_auto and zsp_auto > 0 and abs(zstep_box.value - 1.0) < 0.001:
                zstep_box.value = round(zsp_auto, 3)
                zsp = zstep_box.value
            n_z, n_ch, H, W = img_z.shape
            S["img_z"]    = img_z
            S["n_z"]      = n_z
            S["z_spacing"]= zsp

            ch_min = np.array([np.percentile(img_z[:, c], 1) for c in range(n_ch)])
            ch_max = np.array([np.percentile(img_z[:, c], 99) for c in range(n_ch)])
            S["norm_range"] = (ch_min, ch_max)
            z_sl.max = n_z - 1; z_sl.value = 0

            params = {
                "pixel_size_um":       ps,
                "z_spacing_um":        zsp,
                "sigma":               sigma_sl.value,
                "min_size":            minsize_sl.value,
                "min_dist":            mind_sl.value,
                "excl_factor":         excl_sl.value,
                "reliability_sigma":   relth_sl.value,
                "thresh_offset_sigma": throff_sl.value,
                "margin_sigma":        margin_sl.value,
                "snr_threshold":       snrth_sl.value,
                "bright_factor":       brcomp_sl.value,
                "bright_pct":          brpct_sl.value,

                "min_circularity":     circ_sl.value,
                "border_margin":       bmargin_sl.value,
                # Option E
                "otsu_scale":          otsu_sl.value,
                "nms_overlap_e":       nms_ov_sl.value,
                "link_max_dist_px":    lkdist_sl.value,
                "min_z_planes_e":      minzp_sl.value,
                "manual_thresholds":   None,
            }
            status_lbl.value = f"Running {mth} ..."
            df, meta = process_zstack(fname, params=params, method=mth)
            if df is None:
                status_lbl.value = f"Failed: {meta.get('error', '?')}"; return
            S["df"] = df; S["meta"] = meta; S["params"] = params
            # Populate threshold boxes with auto-computed values (user can override)
            computed_threshs = meta.get("thresholds", [])
            for c, sl in enumerate(thresh_sls):
                if c < len(computed_threshs):
                    sl.value = float(computed_threshs[c])
            S["_computed_threshs"] = list(computed_threshs)
            S["_drawn_file"] = None
            draw_zplane(0)
            n_tot = meta["n_droplets"]; n_cls = meta["n_classes"]
            zinfo = (f"  z-planes: {meta.get('n_z_planes_total','?')}"
                     if "n_z_planes_total" in meta else "")
            raw   = meta.get("raw_detections", 0)
            nact  = meta.get("diag_active_planes", "?")
            nsup  = meta.get("diag_nms_suppressed", "?")
            dinfo = (f"  [raw={raw} planes={nact} nms_sup={nsup}]"
                     if meta.get("method") == "optione" else "")
            status_lbl.value = (f"Done - {n_tot} droplets, {n_cls}/16 classes"
                                + zinfo + dinfo)
        except Exception as e:
            import traceback; traceback.print_exc()
            status_lbl.value = f"Error: {e}"

    def reclassify_cb(_):
        if S["df"] is None: status_lbl.value = "Run pipeline first."; return
        status_lbl.value = "Re-classifying ..."
        try:
            man_thresh = [float(sl.value) for sl in thresh_sls]
            params = dict(S.get("params") or {})
            params["manual_thresholds"]   = man_thresh
            params["thresh_offset_sigma"] = throff_sl.value
            params["margin_sigma"]        = margin_sl.value
            params["reliability_sigma"]   = relth_sl.value
            params["snr_threshold"]       = snrth_sl.value
            params["bright_factor"]       = brcomp_sl.value
            params["bright_pct"]          = brpct_sl.value
            df, meta = process_zstack(S["filename"], params=params, method="optione")
            if df is None:
                status_lbl.value = f"Failed: {meta.get('error', '?')}"; return
            S["df"] = df; S["meta"] = meta
            draw_zplane()
            status_lbl.value = (f"Re-classified - {meta['n_droplets']} droplets, "
                                f"{meta['n_classes']}/16 classes")
        except Exception as e:
            import traceback; traceback.print_exc()
            status_lbl.value = f"Error: {e}"

    def save_cb(_):
        df_cur = S["df"]
        if df_cur is None: status_lbl.value = "Nothing to save."; return
        base = os.path.splitext(S["filename"] or "output")[0]
        out  = base + "_zstack.csv"
        df_cur.to_csv(out, index=False)
        import json, datetime
        params_out = base + "_zstack_params.json"
        p_save = dict(S.get("params") or {})
        p_save["source_file"] = S["filename"]
        p_save["method"]      = S.get("method", "")
        p_save["saved_at"]    = datetime.datetime.now().isoformat(timespec="seconds")
        with open(params_out, "w") as _f: json.dump(p_save, _f, indent=2)
        status_lbl.value = (f"Saved {len(df_cur)} rows -> {os.path.basename(out)}  "
                            f"+ {os.path.basename(params_out)}")

    def _save_image_cb(_):
        try:
            df_cur = S["df"]; img_z = S["img_z"]
            if df_cur is None or img_z is None:
                status_lbl.value = "Run pipeline first."; return
            import matplotlib.patches as _mp
            n_z, n_ch, H, W = img_z.shape
            mip = np.max(img_z, axis=0)
            _ch_c = np.array([[1,0,0],[1,1,0],[0,1,1],[0,0,1]], dtype=float)
            rgb   = np.zeros((H, W, 3), dtype=float)
            rng   = S.get("norm_range")
            ch_min, ch_max = rng if rng else (np.zeros(n_ch), np.ones(n_ch) * 65535)
            for c in range(min(n_ch, 4)):
                denom  = float(ch_max[c] - ch_min[c]) + 1e-9
                norm_c = np.clip((mip[c].astype(float) - ch_min[c]) / denom, 0, 1)
                rgb   += norm_c[:, :, np.newaxis] * _ch_c[c]
            rgb_img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            yy, xx  = np.mgrid[0:H, 0:W]
            fig_s, ax_s = plt.subplots(figsize=(8, 8), dpi=150)
            ax_s.imshow(rgb_img, origin="upper"); ax_s.set_xlim(0, W)
            ax_s.set_ylim(H, 0); ax_s.axis("off")
            for _, row in df_cur.iterrows():
                ci   = int(row.code_int)
                is_br = bool(row.get("bright_red", False))
                col  = "#aaaaaa" if ci == -2 else ("#ff2020" if is_br else HEX[ci % 16])
                cx, cy, r = row.centroid_x, row.centroid_y, row.radius
                mask = (yy - cy)**2 + (xx - cx)**2 <= r**2
                mean_br = rgb_img[mask].mean() / 255.0 if mask.any() else 0.0
                tc = "black" if mean_br > 0.45 else "white"
                ax_s.add_patch(_mp.Circle((cx, cy), r, fill=False, edgecolor=col,
                                           linewidth=1.5, linestyle=":" if ci == -2 else "-"))
                ax_s.text(cx, cy, "?" if ci == -2 else str(ci),
                          ha="center", va="center", fontsize=7, color=tc, fontweight="bold")
            base = os.path.splitext(S["filename"] or "output")[0]
            out  = base + "_zstack_classified.png"
            fig_s.savefig(out, bbox_inches="tight", pad_inches=0.02, dpi=150)
            plt.close(fig_s)
            status_lbl.value = f"Saved -> {os.path.basename(out)}"
        except Exception as e:
            import traceback; traceback.print_exc()
            status_lbl.value = f"Save image error: {e}"

    def _save_analysis_cb(_):
        try:
            df_cur = S["df"]
            if df_cur is None: status_lbl.value = "Run pipeline first."; return
            # Ensure volume_um3 column exists (show_analysis expects it)
            df_cur = df_cur.copy()
            if "volume_um3" not in df_cur.columns:
                if "volume_3d_um3" in df_cur.columns:
                    df_cur["volume_um3"] = df_cur["volume_3d_um3"]
                else:
                    df_cur["volume_um3"] = (4/3) * np.pi * (df_cur["radius_um"] ** 3)
            base   = os.path.splitext(S["filename"] or "output")[0]
            prefix = base + "_zstack_analysis"
            show_analysis(df_cur, pixel_size_um=S["pixel_size"], save_prefix=prefix)
            import glob as _gl
            saved = sorted(_gl.glob(prefix + "_??.png"))
            status_lbl.value = f"Saved {len(saved)} plot(s) -> {os.path.basename(prefix)}_*.png"
        except Exception as e:
            import traceback; traceback.print_exc()
            status_lbl.value = f"Analysis error: {e}"

    def diag_cb(_):
        diag_out.clear_output(wait=True)
        with diag_out:
            df_cur = S.get("df")
            if df_cur is None: print("Run pipeline first."); return
            ch_cols = ["intensity_ch1","intensity_ch2","intensity_ch3","intensity_ch4"]
            if not all(c in df_cur.columns for c in ch_cols):
                print("Intensity data not available."); return
            ch_labels = ["Ch1 red", "Ch2 yellow", "Ch3 cyan", "Ch4 blue"]
            ch_colors = ["#e57373", "#fff176", "#80deea", "#82b1ff"]
            fig_d, axes = plt.subplots(1, 4, figsize=(14, 3.5), dpi=150)
            for c, ax in enumerate(axes):
                vals = df_cur[ch_cols[c]].dropna().values
                ax.hist(vals, bins=40, color=ch_colors[c], edgecolor="#555",
                        alpha=0.85, density=True)
                if not thresh_sls[c].disabled:
                    t = thresh_sls[c].value
                    ax.axvline(t, color="white", linewidth=1.5, linestyle="--",
                               label=f"thr={t:.0f}")
                    ax.legend(fontsize=7)
                ax.set_title(ch_labels[c], fontsize=9)
                ax.set_xlabel("raw intensity", fontsize=8)
                if c == 0: ax.set_ylabel("density", fontsize=8)
                ax.tick_params(labelsize=7)
            fig_d.tight_layout(); plt.show()

    def view3d_cb(_):
        view3d_out.clear_output(wait=True)
        with view3d_out:
            df_cur = S.get("df")
            if df_cur is None: print("Run pipeline first."); return
            ps  = S.get("pixel_size", 0.312)
            zsp = S.get("z_spacing", 1.0)
            x_um = df_cur.centroid_x.values * ps
            y_um = df_cur.centroid_y.values * ps
            if "centroid_z_um" in df_cur.columns:
                z_um  = df_cur.centroid_z_um.values; z_lbl = "Z (um)"
            elif "z_best_plane" in df_cur.columns:
                z_um  = df_cur.z_best_plane.values * zsp; z_lbl = "Z best-focus (um)"
            else:
                z_um  = np.zeros(len(df_cur)); z_lbl = "Z (n/a)"
            codes  = df_cur.code_int.values
            colors = [HEX[max(ci, 0) % 16] if ci >= 0 else "#aaaaaa" for ci in codes]
            sizes  = np.clip(df_cur.radius.values * ps * 3, 3, 20)
            hover  = [
                f"ID:{int(df_cur.iloc[i].label)}  class:{codes[i] if codes[i]>=0 else '?'}<br>"
                f"x={x_um[i]:.1f}  y={y_um[i]:.1f}  z={z_um[i]:.1f} um"
                + (f"  conf={df_cur.iloc[i].min_confidence:.2f}"
                   if "min_confidence" in df_cur.columns else "")
                for i in range(len(df_cur))
            ]
            fig3d = go.FigureWidget(go.Scatter3d(
                x=x_um, y=y_um, z=z_um, mode="markers",
                marker=dict(size=sizes, color=colors, opacity=0.8, line=dict(width=0)),
                text=hover, hoverinfo="text",
            ))
            fig3d.update_layout(
                width=700, height=600, paper_bgcolor="#1a1a1a",
                scene=dict(
                    xaxis=dict(title="X (um)", backgroundcolor="#1a1a1a",
                               gridcolor="#444", zerolinecolor="#444", color="#ccc"),
                    yaxis=dict(title="Y (um)", backgroundcolor="#1a1a1a",
                               gridcolor="#444", zerolinecolor="#444", color="#ccc"),
                    zaxis=dict(title=z_lbl, backgroundcolor="#1a1a1a",
                               gridcolor="#444", zerolinecolor="#444", color="#ccc"),
                    bgcolor="#1a1a1a", aspectmode="data",
                ),
                font=dict(color="#cccccc"),
                margin=dict(l=0, r=0, t=30, b=0),
                title=dict(text=f"3D droplet map  ({len(df_cur)} droplets)",
                           font=dict(size=12, color="#888"), x=0.01),
                showlegend=False,
            )
            display(fig3d)

    run_btn.on_click(run_pipeline_cb)
    reclf_btn.on_click(reclassify_cb)
    save_btn.on_click(save_cb)
    simg_btn.on_click(_save_image_cb)
    sana_btn.on_click(_save_analysis_cb)
    diag_btn.on_click(diag_cb)
    view3d_btn.on_click(view3d_cb)

    # ── Edit controls ─────────────────────────────────────────────────────────
    sel_id_box = widgets.BoundedIntText(
        value=1, min=1, max=9999, description="ID:",
        style={"description_width": "22px"}, layout=widgets.Layout(width="90px"))
    class_input = widgets.BoundedIntText(
        value=0, min=0, max=15, description="cls:",
        style={"description_width": "25px"}, layout=widgets.Layout(width="75px"))
    upd_btn = widgets.Button(description="Update", button_style="warning",
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
        draw_zplane(); status_lbl.value = f"ID {did} -> class {cls}"
    upd_btn.on_click(_update_class)

    del_id_box = widgets.BoundedIntText(
        value=1, min=1, max=9999, description="ID:",
        style={"description_width": "22px"}, layout=widgets.Layout(width="90px"))
    del_btn = widgets.Button(description="Delete", button_style="danger",
                              layout=widgets.Layout(width="60px"))

    def _delete_cb(_):
        df_cur = S["df"]
        if df_cur is None: return
        did = del_id_box.value
        if did not in df_cur.label.values:
            status_lbl.value = f"ID {did} not found."; return
        S["df"] = df_cur[df_cur.label != did].reset_index(drop=True)
        draw_zplane(); status_lbl.value = f"Deleted ID {did}"
    del_btn.on_click(_delete_cb)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _h(t):
        return widgets.HTML(
            f"<b style='font-size:0.78em;letter-spacing:0.05em;color:#888'>{t}</b>")
    def _sep():
        return widgets.HTML("<hr style='margin:3px 0'>")

    left_col = widgets.VBox([
        file_picker_widget,
        path_input,
        widgets.HBox([mag_dd, pxsz_box]),
        method_dd,
        widgets.HBox([zstep_box]),
        _sep(),
        param_tabs,
        _sep(),
        widgets.HBox([run_btn, reclf_btn, save_btn],
                     layout=widgets.Layout(gap="4px")),
    ], layout=widgets.Layout(width="300px", padding="8px"))

    right_col = widgets.VBox([
        _h("EDIT"),
        _sep(),
        widgets.HBox([sel_id_box, class_input, upd_btn],
                     layout=widgets.Layout(gap="4px")),
        widgets.HBox([del_id_box, del_btn],
                     layout=widgets.Layout(gap="4px", margin="4px 0")),
        _sep(),
        _h("CHANNEL THRESHOLDS (auto-filled; edit to override)"),
        *thresh_sls,
        _sep(),
        _h("OUTPUT"),
        widgets.HBox([diag_btn, simg_btn], layout=widgets.Layout(gap="4px", margin="4px 0")),
        widgets.HBox([view3d_btn],         layout=widgets.Layout(margin="0 0 4px 0")),
        sana_btn,
    ], layout=widgets.Layout(
        width="230px", padding="8px", border_left="1px solid #444"))

    ctrl_panel = widgets.HBox(
        [left_col, right_col],
        layout=widgets.Layout(border="1px solid #444", max_height="680px",
                              overflow_y="auto"),
    )
    ctrl_panel.add_class(_uid)
    status_row = widgets.HBox(
        [status_lbl],
        layout=widgets.Layout(padding="3px 8px", border="1px solid #444",
                              border_top="none"),
    )
    status_row.add_class(_uid)

    fig_box = widgets.VBox([fig, z_sl, z_label])
    outer   = widgets.HBox([fig_box, widgets.VBox([ctrl_panel, status_row])])
    _outer_ref[0] = outer
    display(outer)
    display(diag_out)
    display(view3d_out)
    return S
