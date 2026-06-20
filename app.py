"""SandAnalyze Streamlit application.

Usage:
    streamlit run app.py
    sandanalyze             # via CLI entry point
"""

import base64
import csv
import io
import time

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.detector import detect_grains, FlocculationConfig
from core.morphology import (
    GrainMorphology,
    GrainStatistics,
    get_classification_color,
    CLASSIFICATION_COLORS,
)
from core.pipeline import run_detection_pipeline
from core.preprocessor import (
    PreprocessConfig,
    auto_tune_params,
    auto_tune_for_microscope,
    auto_detect_preset,
)
from core.report import generate_pdf_report
from core.exporter import export_csv, export_annotated_image

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SandAnalyze - Sand Grain Morphology Analysis",
    page_icon="🔬",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Compact sidebar */
[data-testid="stSidebar"] .stMarkdown h2 { font-size: 1rem; padding-top: 0.5rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.3rem; }

/* Full-height main area */
[data-testid="stAppViewContainer"] > .main { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state initialisation ─────────────────────────────────────────────

DEFAULTS = {
    "original_image": None,
    "grains": [],
    "morphologies": [],
    "statistics": None,
    "config": PreprocessConfig(),
    "detection_method": "traditional",
    "last_processing_time": 0.0,
    "hull_expansion_ratio": 1.5,
    "border_margin": 5,
    "use_flocculation": True,
    "floc_config": FlocculationConfig(),
    "use_auto_tune": True,
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helper: image overlay ────────────────────────────────────────────────────


def _overlay_grains(
    image: np.ndarray, grains: list, morphologies: list
) -> np.ndarray:
    """Draw grain contours and labels on a copy of the image."""
    display = image.copy()
    if len(display.shape) == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    for idx, grain in enumerate(grains):
        contour = getattr(grain, "contour", None)
        if contour is None or len(contour) == 0:
            continue

        color = (0, 255, 0)
        if idx < len(morphologies):
            color = get_classification_color(morphologies[idx].shape_class)

        cv2.drawContours(display, [contour], -1, color, 2)

        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.circle(display, (cx, cy), 3, (255, 255, 255), -1)
            cv2.putText(
                display, str(idx + 1), (cx - 8, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
            )

    return display


def _draw_legend(image: np.ndarray) -> np.ndarray:
    """Draw classification legend on the image."""
    h, w = image.shape[:2]
    lx, ly = w - 160, h - 100
    ih = 20
    items = list(CLASSIFICATION_COLORS.items())

    cv2.rectangle(image, (lx - 10, ly - 25),
                  (lx + 150, ly + len(items) * ih + 5), (40, 40, 40), -1)
    cv2.rectangle(image, (lx - 10, ly - 25),
                  (lx + 150, ly + len(items) * ih + 5), (200, 200, 200), 1)
    cv2.putText(image, "Classification", (lx, ly - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for i, (label, color) in enumerate(items):
        y = ly + i * ih + 10
        cv2.rectangle(image, (lx, y - 10), (lx + 15, y + 5), color, -1)
        cv2.putText(image, label, (lx + 20, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return image


def _make_zoomable_image(image: np.ndarray) -> go.Figure:
    """Render image as a Plotly figure with zoom/pan support.

    Converts BGR to RGB, embeds in a Plotly layout with the modebar
    providing zoom, pan, and reset tools.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    fig = go.Figure(
        data=go.Image(z=rgb),
    )
    fig.update_layout(
        xaxis=dict(
            range=[0, w], showgrid=False, zeroline=False,
            showticklabels=False, constrain="domain",
        ),
        yaxis=dict(
            range=[h, 0], showgrid=False, zeroline=False,
            showticklabels=False, scaleanchor="x", scaleratio=1,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white",
        dragmode="pan",
    )
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    return fig


# ── Helper: charts ───────────────────────────────────────────────────────────


def _make_size_histogram(stats: GrainStatistics) -> go.Figure:
    fig = go.Figure(
        data=[go.Histogram(
            x=stats.d_eq_values, nbinsx=20,
            marker=dict(line=dict(width=1, color="black")),
        )]
    )
    fig.update_layout(
        title="Size Distribution", xaxis_title="Equivalent Diameter (d_eq)",
        yaxis_title="Frequency",
        showlegend=False, margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white",
    )
    return fig


def _make_scatter(stats: GrainStatistics) -> go.Figure:
    fig = go.Figure(
        data=[go.Scatter(
            x=stats.circularity_values, y=stats.sphericity_values,
            mode="markers",
            marker=dict(opacity=0.6, line=dict(width=0.5, color="black")),
        )]
    )
    fig.update_layout(
        title="Circularity vs Sphericity", xaxis_title="Circularity",
        yaxis_title="Sphericity",
        margin=dict(l=40, r=20, t=40, b=40), template="plotly_white",
    )
    return fig


def _make_classification_pie(stats: GrainStatistics) -> go.Figure:
    labels = list(stats.zingg_counts.keys())
    values = list(stats.zingg_counts.values())
    color_map = {
        "spherical": "#99ff99",
        "rod-like": "#66b3ff",
        "discoidal": "#ff9999",
        "flocculation": "#ffcc99",
    }
    colors = [color_map.get(l, "#cccccc") for l in labels]
    fig = go.Figure(
        data=[go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors),
            textposition="inside", textinfo="percent+label",
        )]
    )
    fig.update_layout(
        title="Classification Distribution",
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔬 SandAnalyze")

    # ── Image upload ───────────────────────────────────────────────────────
    with st.expander("📷 Image Upload", expanded=True):
        uploaded_file = st.file_uploader(
            "Select sand grain image",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is not None:
                st.session_state.original_image = image
                st.session_state.grains = []
                st.session_state.morphologies = []
                st.session_state.statistics = None
                st.session_state.detection_method = "traditional"
                st.success(f"✓ {uploaded_file.name}")
            else:
                st.error("Failed to read image file")

    # ── Preprocessing params ───────────────────────────────────────────────
    with st.expander("⚙️ Preprocessing Parameters", expanded=False):
        config = st.session_state.config

        col_a, col_b = st.columns(2)
        with col_a:
            blur_kernel = st.number_input(
                "Blur Kernel", min_value=1, max_value=31,
                value=config.blur_kernel, step=2,
            )
            adaptive_block = st.number_input(
                "Adaptive Block Size", min_value=3, max_value=99,
                value=config.adaptive_block_size, step=2,
            )
            adaptive_c = st.number_input(
                "Adaptive Constant C", min_value=-10, max_value=20,
                value=config.adaptive_c, step=1,
            )
        with col_b:
            morph_kernel = st.number_input(
                "Morphology Kernel", min_value=1, max_value=21,
                value=config.morph_kernel_size, step=2,
            )
            min_area = st.number_input(
                "Min Area", min_value=1, max_value=10000,
                value=config.min_area, step=1,
            )

        use_clahe = st.checkbox("CLAHE Enhancement", value=config.use_clahe)
        use_auto_tune = st.checkbox("Auto-Tune Parameters", value=st.session_state.use_auto_tune)

        st.session_state.config = PreprocessConfig(
            blur_kernel=blur_kernel,
            adaptive_block_size=adaptive_block,
            adaptive_c=adaptive_c,
            morph_kernel_size=morph_kernel,
            min_area=min_area,
            use_clahe=use_clahe,
        )
        st.session_state.use_auto_tune = use_auto_tune

    # ── Detection options ──────────────────────────────────────────────────
    with st.expander("🔍 Detection Options", expanded=False):
        border_margin = st.number_input(
            "Border Margin (px)", min_value=0, max_value=50,
            value=st.session_state.border_margin, step=1,
            help="Distance from ROI boundary for edge filtering"
        )
        hull_expansion_ratio = st.slider(
            "Hull Expansion Ratio", min_value=1.0, max_value=3.0,
            value=st.session_state.hull_expansion_ratio, step=0.1,
            help="Threshold for using convex hull vs mask filling"
        )
        use_flocculation = st.checkbox(
            "Flocculation Detection", value=st.session_state.use_flocculation,
            help="Detect and classify grain clusters"
        )

        st.session_state.border_margin = border_margin
        st.session_state.hull_expansion_ratio = hull_expansion_ratio
        st.session_state.use_flocculation = use_flocculation

    # ── Run button ─────────────────────────────────────────────────────────
    if st.button("🔍 Run Detection", type="primary", width="stretch"):
        if st.session_state.original_image is None:
            st.warning("Please upload an image first")
        else:
            start = time.time()
            try:
                image = st.session_state.original_image
                config = st.session_state.config

                # Auto-tune if enabled
                detection_params = {}
                if st.session_state.use_auto_tune:
                    # Auto-detect best preset based on image characteristics
                    detected_preset = auto_detect_preset(image)
                    config = PreprocessConfig.from_preset(detected_preset)

                    # Further fine-tune based on image analysis
                    config, detection_params = auto_tune_for_microscope(image)
                    st.session_state.config = config
                    st.info(
                        f"Auto-detected preset: {detected_preset} | "
                        f"blur={config.blur_kernel}, "
                        f"block={config.adaptive_block_size}, "
                        f"min_area={detection_params['min_area']}"
                    )

                # Run the shared detection pipeline
                grains, morphologies, statistics = run_detection_pipeline(
                    image=image,
                    config=config,
                    min_area=detection_params.get("min_area", config.min_area),
                    max_area=detection_params.get("max_area", 15000),
                    border_margin=st.session_state.border_margin,
                    hull_expansion_ratio=st.session_state.hull_expansion_ratio,
                    floc_config=st.session_state.floc_config if st.session_state.use_flocculation else None,
                    crop_black_background=True,
                )

                st.session_state.detection_method = "traditional"
                st.session_state.grains = grains
                st.session_state.morphologies = morphologies
                st.session_state.statistics = statistics
                st.session_state.last_processing_time = time.time() - start
            except Exception as exc:
                st.error(f"Detection error: {exc}")

    # ── Export ─────────────────────────────────────────────────────────────
    morphs = st.session_state.morphologies

    if morphs:
        st.divider()

        # CSV export
        fieldnames = [
            "grain_id", "area", "perimeter", "circularity", "d_eq",
            "major_axis", "minor_axis", "aspect_ratio", "sphericity",
            "convexity", "feret_max", "feret_min",
            "shape_class", "is_flocculation", "confidence",
        ]
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
        writer.writeheader()
        for idx, m in enumerate(morphs, start=1):
            writer.writerow({
                "grain_id": idx,
                "area": round(m.area, 4),
                "perimeter": round(m.perimeter, 4),
                "circularity": round(m.circularity, 6),
                "d_eq": round(m.d_eq, 4),
                "major_axis": round(m.major_axis, 4),
                "minor_axis": round(m.minor_axis, 4),
                "aspect_ratio": round(m.aspect_ratio, 4),
                "sphericity": round(m.sphericity, 6),
                "convexity": round(m.convexity, 6),
                "feret_max": round(m.feret_max, 4),
                "feret_min": round(m.feret_min, 4),
                "shape_class": m.shape_class,
                "is_flocculation": m.is_flocculation,
                "confidence": round(m.confidence, 4),
            })
        st.download_button(
            "📊 Export CSV", data=csv_buf.getvalue(),
            file_name="sand_analysis.csv", mime="text/csv",
            width="stretch",
        )

        # Image export
        if st.session_state.original_image is not None and st.session_state.grains:
            annotated = st.session_state.original_image.copy()
            for idx, grain in enumerate(st.session_state.grains, start=1):
                contour = getattr(grain, "contour", None)
                if contour is None:
                    continue
                color = (0, 255, 0)
                if idx - 1 < len(morphs):
                    color = get_classification_color(morphs[idx - 1].shape_class)
                cv2.drawContours(annotated, [contour], -1, color, 2)
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    cx, cy = x + w // 2, y + h // 2
                cv2.putText(annotated, str(idx), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            _, png_buf = cv2.imencode(".png", annotated)
            st.download_button(
                "🖼️ Export Annotated Image", data=png_buf.tobytes(),
                file_name="sand_annotated.png", mime="image/png",
                width="stretch",
            )

        # PDF export
        if st.session_state.original_image is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                annotated_path = tmp.name
            cv2.imwrite(annotated_path, annotated)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                pdf_path = tmp.name
            try:
                generate_pdf_report(
                    annotated_path, morphs, st.session_state.statistics, pdf_path,
                    annotated_image_path=annotated_path,
                )
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    "📄 Export PDF Report", data=pdf_bytes,
                    file_name="sand_analysis_report.pdf", mime="application/pdf",
                    width="stretch",
                )
            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")
            finally:
                import os
                for p in [annotated_path, pdf_path]:
                    if os.path.exists(p):
                        os.unlink(p)

# ── Main area ────────────────────────────────────────────────────────────────

# Status bar
method = st.session_state.detection_method
stats = st.session_state.statistics
morphs = st.session_state.morphologies

if stats is not None:
    count = stats.count
    elapsed = st.session_state.last_processing_time
    status_text = f"Method: {method} | Grains: {count} | Time: {elapsed:.2f}s"
else:
    status_text = f"Method: {method} | Grains: 0 | Please upload image and run detection"

st.caption(status_text)

# ── Main layout: image (left) + tabs (right) ────────────────────────────────

col_img, col_res = st.columns([5, 4], gap="small")

with col_img:
    if st.session_state.original_image is not None:
        display = _overlay_grains(
            st.session_state.original_image,
            st.session_state.grains,
            st.session_state.morphologies,
        )
        if st.session_state.morphologies:
            display = _draw_legend(display)

        # Zoomable Plotly image
        fig = _make_zoomable_image(display)
        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "modeBarButtonsToAdd": ["drawrect", "eraseshape"],
                "modeBarButtonsToRemove": [
                    "zoom2d", "pan2d", "select2d", "lasso2d",
                    "zoomIn2d", "zoomOut2d", "autoScale2d",
                    "resetScale2d", "toImage",
                ],
                "displaylogo": False,
            },
        )
    else:
        st.info("👈 Please upload a sand grain image in the sidebar, then click 'Run Detection'")

with col_res:
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "📋 Grain Data", "📈 Charts"])

    # ── Summary tab ────────────────────────────────────────────────────────
    with tab1:
        if stats is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Grain Count", stats.count)
                st.metric("Mean Circularity", f"{stats.circularity_mean:.4f}")
                st.metric("Mean Sphericity", f"{stats.sphericity_mean:.4f}")
            with col_b:
                st.metric("Mean Equivalent Diameter", f"{stats.d_eq_mean:.4f}")
                st.metric("Mean Aspect Ratio", f"{stats.aspect_ratio_mean:.4f}")
                st.metric("Mean Convexity", f"{stats.convexity_mean:.4f}")

            st.divider()
            st.caption("Classification Summary")
            if stats.zingg_counts:
                parts = []
                for key, cnt in stats.zingg_counts.items():
                    pct = cnt / stats.count * 100 if stats.count > 0 else 0
                    parts.append(f"**{key}**: {cnt} ({pct:.1f}%)")
                st.markdown(" | ".join(parts))

            if stats.flocculation_count > 0:
                st.divider()
                st.caption("Flocculation")
                st.metric("Flocculation Count", stats.flocculation_count)
                st.metric("Flocculation Ratio", f"{stats.flocculation_ratio:.2%}")
        else:
            st.info("Please run detection first")

    # ── Table tab ──────────────────────────────────────────────────────────
    with tab2:
        if morphs:
            rows = []
            for i, g in enumerate(morphs):
                rows.append({
                    "ID": i + 1,
                    "Area": f"{g.area:.2f}",
                    "Perimeter": f"{g.perimeter:.2f}",
                    "Circularity": f"{g.circularity:.4f}",
                    "d_eq": f"{g.d_eq:.4f}",
                    "Aspect Ratio": f"{g.aspect_ratio:.4f}",
                    "Sphericity": f"{g.sphericity:.4f}",
                    "Convexity": f"{g.convexity:.4f}",
                    "Class": g.shape_class,
                    "Floc": "Yes" if g.is_flocculation else "No",
                })
            st.dataframe(rows, width="stretch", hide_index=True,
                         height=420)
        else:
            st.info("Please run detection first")

    # ── Charts tab ─────────────────────────────────────────────────────────
    with tab3:
        if stats is not None and morphs:
            st.plotly_chart(
                _make_size_histogram(stats), width="stretch",
            )
            st.plotly_chart(
                _make_scatter(stats), width="stretch",
            )
            st.plotly_chart(
                _make_classification_pie(stats), width="stretch",
            )
        else:
            st.info("Please run detection first")
