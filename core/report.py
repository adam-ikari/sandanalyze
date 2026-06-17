"""PDF report generation for sand grain analysis results."""

import os
import tempfile
from datetime import datetime

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from core.morphology import GrainStatistics


def _create_classification_chart(zingg_counts: dict, width: int = 400, height: int = 300) -> str:
    """Create a classification pie chart using OpenCV.

    Returns:
        Path to temporary PNG file (caller must delete).
    """
    labels = list(zingg_counts.keys())
    values = list(zingg_counts.values())
    total = sum(values)

    if total == 0:
        # Return empty chart placeholder
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.putText(img, "No data", (width // 2 - 40, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        path = tempfile.mktemp(suffix=".png")
        cv2.imwrite(path, img)
        return path

    # Color mapping
    color_map = {
        "spherical": (0, 255, 0),
        "rod-like": (0, 0, 255),
        "discoidal": (255, 0, 0),
        "flocculation": (0, 255, 255),
    }

    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    center = (width // 2, height // 2)
    radius = min(width, height) // 3

    # Draw pie chart
    start_angle = 0.0
    for label, value in zip(labels, values):
        if value == 0:
            continue
        angle = (value / total) * 360.0
        color = color_map.get(label, (128, 128, 128))
        # Convert BGR to RGB for display
        rgb = (color[2], color[1], color[0])
        end_angle = start_angle + angle
        cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle,
                    (rgb[2], rgb[1], rgb[0]), -1)
        start_angle = end_angle

    # Add legend
    y_offset = 20
    for label, value in zip(labels, values):
        if value == 0:
            continue
        color = color_map.get(label, (128, 128, 128))
        text = f"{label}: {value} ({value/total*100:.1f}%)"
        cv2.putText(img, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 20

    path = tempfile.mktemp(suffix=".png")
    cv2.imwrite(path, img)
    return path


def generate_pdf_report(
    image_path: str,
    morphologies: list,
    stats: GrainStatistics,
    output_path: str,
    annotated_image_path: str | None = None,
) -> None:
    """Generate a PDF report for sand grain analysis.

    Parameters
    ----------
    image_path : str
        Path to the original input image.
    morphologies : list
        List of GrainMorphology objects.
    stats : GrainStatistics
        Computed statistics for the analysis.
    output_path : str
        Path for the output PDF file.
    annotated_image_path : str | None, optional
        Path to an annotated image to include in the report.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=20,
        spaceAfter=20,
        alignment=1,  # Center
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=10,
    )

    story = []

    # Title
    story.append(Paragraph("Sand Grain Analysis Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    # Summary statistics
    story.append(Paragraph("Summary Statistics", heading_style))
    summary_data = [
        ["Metric", "Value"],
        ["Total Grains", str(stats.count)],
        ["Mean Area (px²)", f"{stats.area_mean:.2f}"],
        ["Mean Circularity", f"{stats.circularity_mean:.4f}"],
        ["Mean Equivalent Diameter", f"{stats.d_eq_mean:.2f}"],
        ["Mean Aspect Ratio", f"{stats.aspect_ratio_mean:.2f}"],
        ["Mean Sphericity", f"{stats.sphericity_mean:.4f}"],
        ["Mean Convexity", f"{stats.convexity_mean:.4f}"],
        ["Flocculation Count", str(stats.flocculation_count)],
        ["Flocculation Ratio", f"{stats.flocculation_ratio:.2%}"],
    ]

    summary_table = Table(summary_data, colWidths=[3 * inch, 2 * inch])
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#D9E2F3")),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ])
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.3 * inch))

    # Classification distribution
    story.append(Paragraph("Classification Distribution", heading_style))
    class_data = [["Classification", "Count", "Percentage"]]
    total = stats.count
    for label, count in stats.zingg_counts.items():
        pct = f"{count / total * 100:.1f}%" if total > 0 else "0.0%"
        class_data.append([label, str(count), pct])

    class_table = Table(class_data, colWidths=[2.5 * inch, 1.5 * inch, 1.5 * inch])
    class_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#D9E2F3")),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ])
    )
    story.append(class_table)
    story.append(Spacer(1, 0.3 * inch))

    # Classification chart
    chart_path = _create_classification_chart(stats.zingg_counts)
    chart_loaded = False
    try:
        # Verify the chart file exists before adding to PDF
        if os.path.exists(chart_path):
            story.append(Paragraph("Classification Chart", heading_style))
            # Use absolute path for reportlab
            abs_chart_path = os.path.abspath(chart_path)
            img = RLImage(abs_chart_path, width=400, height=300)
            story.append(img)
            chart_loaded = True
            story.append(Spacer(1, 0.2 * inch))
    except Exception:
        # If chart image fails to load, skip it
        pass

    # Annotated image
    if annotated_image_path and os.path.exists(annotated_image_path):
        story.append(Paragraph("Annotated Image", heading_style))
        abs_annotated_path = os.path.abspath(annotated_image_path)
        img = RLImage(abs_annotated_path, width=400, height=300)
        story.append(img)
        story.append(Spacer(1, 0.2 * inch))

    # Detailed grain data (first 20)
    if morphologies:
        story.append(Paragraph("Grain Details (First 20)", heading_style))
        detail_data = [
            ["#", "Area", "Circularity", "Aspect Ratio", "Class", "Floc"]
        ]
        for i, m in enumerate(morphologies[:20], start=1):
            detail_data.append([
                str(i),
                f"{m.area:.1f}",
                f"{m.circularity:.3f}",
                f"{m.aspect_ratio:.2f}",
                m.shape_class,
                "Yes" if m.is_flocculation else "No",
            ])

        detail_table = Table(detail_data, colWidths=[0.5 * inch, 1.2 * inch, 1.2 * inch,
                                                     1.2 * inch, 1.5 * inch, 0.8 * inch])
        detail_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#D9E2F3")),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ])
        )
        story.append(detail_table)

    doc.build(story)

    # Clean up chart file after build completes
    if os.path.exists(chart_path):
        os.unlink(chart_path)
