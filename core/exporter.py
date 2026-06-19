"""Export utilities for sand grain analysis results."""

import csv

import cv2
import numpy as np

from core.morphology import GrainContour, GrainMorphology


def export_csv(morphologies: list[GrainMorphology], path: str) -> None:
    """Export grain morphologies to a CSV file.

    Parameters
    ----------
    morphologies : list[GrainMorphology]
        List of grain morphology objects to export.
    path : str
        Output file path.
    """
    fieldnames = [
        "grain_id",
        "area",
        "perimeter",
        "circularity",
        "d_eq",
        "major_axis",
        "minor_axis",
        "aspect_ratio",
        "sphericity",
        "convexity",
        "feret_max",
        "feret_min",
        "shape_class",
        "is_flocculation",
        "confidence",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, morph in enumerate(morphologies, start=1):
            writer.writerow(
                {
                    "grain_id": idx,
                    "area": round(morph.area, 4),
                    "perimeter": round(morph.perimeter, 4),
                    "circularity": round(morph.circularity, 6),
                    "d_eq": round(morph.d_eq, 4),
                    "major_axis": round(morph.major_axis, 4),
                    "minor_axis": round(morph.minor_axis, 4),
                    "aspect_ratio": round(morph.aspect_ratio, 4),
                    "sphericity": round(morph.sphericity, 6),
                    "convexity": round(morph.convexity, 6),
                    "feret_max": round(morph.feret_max, 4),
                    "feret_min": round(morph.feret_min, 4),
                    "shape_class": morph.shape_class,
                    "is_flocculation": morph.is_flocculation,
                    "confidence": round(morph.confidence, 4),
                }
            )


def export_annotated_image(
    image: np.ndarray,
    grains: list[GrainContour],
    path: str,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
    morphologies: list[GrainMorphology] | None = None,
) -> None:
    """Export an annotated image with grain contours and labels.

    Parameters
    ----------
    image : np.ndarray
        Original image to annotate.
    grains : list[GrainContour]
        List of detected grains with contours.
    path : str
        Output file path for the annotated image.
    color : tuple[int, int, int], optional
        BGR color for contours, default is green.
    thickness : int, optional
        Thickness of contour lines.
    morphologies : list[GrainMorphology] | None, optional
        Optional list of morphologies for classification coloring.
    """
    annotated = image.copy()

    for idx, grain in enumerate(grains, start=1):
        # Determine color based on classification if morphologies provided
        if morphologies and idx - 1 < len(morphologies):
            from core.morphology import get_classification_color
            morph = morphologies[idx - 1]
            contour_color = get_classification_color(morph.shape_class)
        else:
            contour_color = color

        cv2.drawContours(annotated, [grain.contour], -1, contour_color, thickness)

        # Compute centroid for label placement
        moments = cv2.moments(grain.contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            # Fallback to bounding box center
            x, y, w, h = cv2.boundingRect(grain.contour)
            cx = x + w // 2
            cy = y + h // 2

        cv2.putText(
            annotated,
            str(idx),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            thickness,
        )

    cv2.imwrite(path, annotated)
