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

from core.morphology import (
    GrainMorphology,
    GrainStatistics,
    compute_morphology,
    compute_statistics,
    get_zingg_color,
    ZINGG_COLORS,
)
from core.preprocessor import PreprocessConfig, preprocess
from core.traditional import GrainContour, detect_grains
from core.yolo_detector import YOLODetector, refine_with_yolo

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SandAnalyze - 沙粒形态分析系统",
    page_icon="🔬",
    layout="wide",
)

# ── Session state initialisation ─────────────────────────────────────────────

DEFAULTS = {
    "original_image": None,
    "grains": [],
    "morphologies": [],
    "statistics": None,
    "config": PreprocessConfig(),
    "detection_method": "传统方法",
    "last_processing_time": 0.0,
    "yolo_detector": None,
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.yolo_detector is None:
    st.session_state.yolo_detector = YOLODetector()

# ── Helper functions ─────────────────────────────────────────────────────────


def _fig_to_html(fig: go.Figure) -> str:
    """Convert a Plotly figure to an HTML string for st.plotly_chart."""
    return fig


def _make_size_histogram(stats: GrainStatistics) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Histogram(
                x=stats.d_eq_values,
                nbinsx=20,
                marker=dict(line=dict(width=1, color="black")),
            )
        ]
    )
    fig.update_layout(
        title="粒径分布",
        xaxis_title="等效粒径 (d_eq)",
        yaxis_title="频数",
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white",
    )
    return fig


def _make_scatter(stats: GrainStatistics) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scatter(
                x=stats.circularity_values,
                y=stats.sphericity_values,
                mode="markers",
                marker=dict(opacity=0.6, line=dict(width=0.5, color="black")),
            )
        ]
    )
    fig.update_layout(
        title="圆度 vs 球度",
        xaxis_title="圆度",
        yaxis_title="球度",
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white",
    )
    return fig


def _make_zingg_pie(stats: GrainStatistics) -> go.Figure:
    labels = list(stats.zingg_counts.keys())
    values = list(stats.zingg_counts.values())
    colors = ["#99ff99", "#66b3ff", "#ff9999"]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors[: len(labels)]),
                textposition="inside",
                textinfo="percent+label",
            )
        ]
    )
    fig.update_layout(
        title="Zingg分类",
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
    )
    return fig


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

        color = (0, 255, 0)  # default green
        if idx < len(morphologies):
            color = get_zingg_color(morphologies[idx].aspect_ratio)

        cv2.drawContours(display, [contour], -1, color, 2)

        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.circle(display, (cx, cy), 3, (255, 255, 255), -1)
            cv2.putText(
                display,
                str(idx + 1),
                (cx - 8, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    return display


def _draw_legend(image: np.ndarray) -> np.ndarray:
    """Draw Zingg classification legend on the image."""
    h, w = image.shape[:2]
    lx, ly = w - 140, h - 80
    ih = 20
    items = list(ZINGG_COLORS.items())

    cv2.rectangle(image, (lx - 10, ly - 25),
                  (lx + 130, ly + len(items) * ih + 5), (40, 40, 40), -1)
    cv2.rectangle(image, (lx - 10, ly - 25),
                  (lx + 130, ly + len(items) * ih + 5), (200, 200, 200), 1)
    cv2.putText(image, "Zingg分类", (lx, ly - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for i, (label, color) in enumerate(items):
        y = ly + i * ih + 10
        cv2.rectangle(image, (lx, y - 10), (lx + 15, y + 5), color, -1)
        cv2.putText(image, label, (lx + 20, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return image


def _np_to_png_base64(image: np.ndarray) -> str:
    """Encode a numpy image as base64 PNG for st.image."""
    if len(image.shape) == 2:
        display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        display = image
    _, buf = cv2.imencode(".png", display)
    return base64.b64encode(buf).decode("utf-8")


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📷 图像加载")

    uploaded_file = st.file_uploader(
        "上传沙粒图像",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    )
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            st.session_state.original_image = image
            st.session_state.grains = []
            st.session_state.morphologies = []
            st.session_state.statistics = None
            st.session_state.detection_method = "传统方法"
            st.success(f"已加载: {uploaded_file.name}")
        else:
            st.error("无法读取图像文件")

    st.divider()
    st.header("⚙️ 预处理参数")

    config = st.session_state.config

    blur_kernel = st.number_input(
        "模糊核大小", min_value=1, max_value=31, value=config.blur_kernel, step=2
    )
    adaptive_block = st.number_input(
        "自适应块大小", min_value=3, max_value=99, value=config.adaptive_block_size, step=2
    )
    adaptive_c = st.number_input(
        "自适应常数 C", min_value=-10, max_value=20, value=config.adaptive_c, step=1
    )
    morph_kernel = st.number_input(
        "形态学核大小", min_value=1, max_value=21, value=config.morph_kernel_size, step=2
    )
    min_area = st.number_input(
        "最小面积", min_value=1, max_value=10000, value=config.min_area, step=1
    )

    use_clahe = st.checkbox("CLAHE 增强", value=config.use_clahe)
    use_watershed = st.checkbox("分水岭分割", value=config.use_watershed)

    st.session_state.config = PreprocessConfig(
        blur_kernel=blur_kernel,
        adaptive_block_size=adaptive_block,
        adaptive_c=adaptive_c,
        morph_kernel_size=morph_kernel,
        min_area=min_area,
        use_clahe=use_clahe,
        use_watershed=use_watershed,
    )

    st.divider()
    st.header("🤖 YOLO 设置")

    yolo_available = st.session_state.yolo_detector.is_available
    use_yolo = st.checkbox(
        "YOLO精细分割",
        value=yolo_available,
        disabled=not yolo_available,
        help="使用YOLOv8-seg进行精细分割" if yolo_available else "YOLO模型不可用",
    )

    st.divider()

    if st.button("🔍 运行检测", type="primary", use_container_width=True):
        if st.session_state.original_image is None:
            st.warning("请先加载图像")
        else:
            start = time.time()
            try:
                mask = preprocess(
                    st.session_state.original_image,
                    st.session_state.config,
                )
                traditional = detect_grains(
                    mask, min_area=st.session_state.config.min_area
                )
                st.session_state.detection_method = "传统方法"

                if yolo_available and use_yolo:
                    refined = refine_with_yolo(
                        st.session_state.original_image,
                        traditional,
                        st.session_state.yolo_detector,
                        min_area=st.session_state.config.min_area,
                    )
                    if refined is not traditional:
                        st.session_state.grains = refined
                        st.session_state.detection_method = "混合方法(传统+YOLO)"
                    else:
                        st.session_state.grains = traditional
                else:
                    st.session_state.grains = traditional

                st.session_state.morphologies = [
                    compute_morphology(g.contour, g.mask)
                    for g in st.session_state.grains
                ]
                st.session_state.statistics = compute_statistics(
                    st.session_state.morphologies
                )
                st.session_state.last_processing_time = time.time() - start
            except Exception as exc:
                st.error(f"检测错误: {exc}")

    st.divider()
    st.header("📤 导出")

    stats = st.session_state.statistics
    morphs = st.session_state.morphologies

    if morphs:
        # Build CSV in-memory
        fieldnames = [
            "grain_id", "area", "perimeter", "circularity", "d_eq",
            "major_axis", "minor_axis", "aspect_ratio", "sphericity",
            "convexity", "feret_max", "feret_min",
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
            })
        st.download_button(
            "📊 导出 CSV",
            data=csv_buf.getvalue(),
            file_name="sand_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if st.session_state.original_image is not None and st.session_state.grains:
        # Build annotated PNG in-memory
        annotated = st.session_state.original_image.copy()
        for idx, grain in enumerate(st.session_state.grains, start=1):
            contour = getattr(grain, "contour", None)
            if contour is None:
                continue
            color = (0, 255, 0)
            if idx - 1 < len(morphs) if morphs else False:
                color = get_zingg_color(morphs[idx - 1].aspect_ratio)
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
            "🖼️ 导出标注图",
            data=png_buf.tobytes(),
            file_name="sand_annotated.png",
            mime="image/png",
            use_container_width=True,
        )

# ── Main area ────────────────────────────────────────────────────────────────

st.title("🔬 SandAnalyze - 沙粒形态分析系统")

# Status bar
method = st.session_state.detection_method
if st.session_state.statistics is not None:
    count = st.session_state.statistics.count
    elapsed = st.session_state.last_processing_time
    st.caption(
        f"方法: {method} | 颗粒数: {count} | 处理时间: {elapsed:.2f}s"
    )
else:
    st.caption(f"方法: {method} | 颗粒数: 0 | 请加载图像并运行检测")

st.divider()

# ── Two-column layout: image | results ──────────────────────────────────────

col_img, col_res = st.columns([3, 2], gap="medium")

# ── Left: Image display ─────────────────────────────────────────────────────

with col_img:
    st.subheader("📷 沙粒图像")

    if st.session_state.original_image is not None:
        display = _overlay_grains(
            st.session_state.original_image,
            st.session_state.grains,
            st.session_state.morphologies,
        )
        if st.session_state.morphologies:
            display = _draw_legend(display)
        st.image(display, use_container_width=True, channels="BGR")
    else:
        st.info("请在侧边栏上传沙粒图像")

# ── Right: Results tabs ─────────────────────────────────────────────────────

with col_res:
    tab1, tab2, tab3 = st.tabs(["📊 统计摘要", "📋 颗粒数据", "📈 图表"])

    stats = st.session_state.statistics
    morphs = st.session_state.morphologies

    # ── Summary tab ──────────────────────────────────────────────────────
    with tab1:
        if stats is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("颗粒数量", stats.count)
                st.metric("平均圆度", f"{stats.circularity_mean:.4f}")
                st.metric("平均球度", f"{stats.sphericity_mean:.4f}")
            with col_b:
                st.metric("平均等效粒径", f"{stats.d_eq_mean:.4f}")
                st.metric("平均长短轴比", f"{stats.aspect_ratio_mean:.4f}")
                st.metric("平均凸度", f"{stats.convexity_mean:.4f}")

            st.divider()
            st.subheader("Zingg 分类")
            if stats.zingg_counts:
                parts = []
                for key, cnt in stats.zingg_counts.items():
                    pct = cnt / stats.count * 100 if stats.count > 0 else 0
                    parts.append(f"**{key}**: {cnt} ({pct:.1f}%)")
                st.markdown(" | ".join(parts))
        else:
            st.info("请先运行检测")

    # ── Table tab ────────────────────────────────────────────────────────
    with tab2:
        if morphs:
            rows = []
            for i, g in enumerate(morphs):
                rows.append(
                    {
                        "ID": i + 1,
                        "面积": f"{g.area:.2f}",
                        "周长": f"{g.perimeter:.2f}",
                        "圆度": f"{g.circularity:.4f}",
                        "等效粒径": f"{g.d_eq:.4f}",
                        "长短轴比": f"{g.aspect_ratio:.4f}",
                        "球度": f"{g.sphericity:.4f}",
                        "凸度": f"{g.convexity:.4f}",
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("请先运行检测")

    # ── Charts tab ───────────────────────────────────────────────────────
    with tab3:
        if stats is not None and morphs:
            st.plotly_chart(
                _make_size_histogram(stats),
                use_container_width=True,
            )
            st.plotly_chart(
                _make_scatter(stats),
                use_container_width=True,
            )
            st.plotly_chart(
                _make_zingg_pie(stats),
                use_container_width=True,
            )
        else:
            st.info("请先运行检测")
