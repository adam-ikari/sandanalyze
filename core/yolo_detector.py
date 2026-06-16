"""YOLOv8-seg grain detector with hybrid detection support.

This module provides YOLOv8 segmentation-based grain detection that can be
used standalone or in combination with traditional contour detection for
refined segmentation of touching grains.
"""

from __future__ import annotations

import logging
import warnings

import cv2
import numpy as np

from core.model_manager import ensure_model_available, get_model_path
from core.traditional import GrainContour

logger = logging.getLogger(__name__)

# Default model to use
DEFAULT_MODEL = "yolov8n-seg.pt"

# Threshold for detecting touching grains (aspect ratio of bounding box)
TOUCHING_THRESHOLD = 2.0


class YOLODetector:
    """Grain detector using YOLOv8-seg with graceful fallback.

    Parameters
    ----------
    model_name : str, optional
        Name or path of the YOLO segmentation model to load.
        Default is ``"yolov8n-seg.pt"``.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load the YOLO model, catching exceptions gracefully."""
        try:
            from ultralytics import YOLO

            # First, try to find the model in our managed locations
            model_path = get_model_path(self._model_name)

            if model_path is not None:
                # Use the bundled/cached model
                self._model = YOLO(str(model_path))
                logger.info("Loaded YOLO model from: %s", model_path)
            else:
                # Try to download or use default path
                model_path = ensure_model_available(self._model_name)
                if model_path is not None:
                    self._model = YOLO(str(model_path))
                    logger.info("Loaded YOLO model: %s", model_path)
                else:
                    # Last resort: let ultralytics handle it
                    self._model = YOLO(self._model_name)
                    logger.info("Loaded YOLO model (default): %s", self._model_name)

        except Exception as exc:  # noqa: BLE001
            self._model = None
            logger.warning(
                "Failed to load YOLO model '%s': %s", self._model_name, exc
            )

    @property
    def is_available(self) -> bool:
        """Return True if the YOLO model is loaded and ready."""
        return self._model is not None

    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        min_area: int = 50,
    ) -> list[GrainContour]:
        """Detect grains in *image* using YOLOv8-seg.

        Parameters
        ----------
        image : np.ndarray
            Input image (H x W or H x W x C).
        conf : float, optional
            Confidence threshold for detections. Default is 0.25.
        min_area : int, optional
            Minimum mask area (in pixels) to keep a grain. Default is 50.

        Returns
        -------
        list[GrainContour]
            Detected grains sorted by area (descending).  If the model is
            unavailable an empty list is returned and a warning is issued.
        """
        if not self.is_available:
            warnings.warn(
                "YOLO model is not available; returning empty grain list.",
                stacklevel=2,
            )
            return []

        # Ensure image is in uint8 RGB format expected by ultralytics
        if image.ndim == 2:
            # Grayscale -> RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # Assume already color; ultralytics handles BGR/RGB internally
            image_rgb = image

        results = self._model(image_rgb, conf=conf, verbose=False)

        grains: list[GrainContour] = []
        h, w = image.shape[:2]

        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()  # shape: (N, H_model, W_model)

            for mask in masks:
                # Resize mask to original image size
                if mask.shape != (h, w):
                    mask_resized = cv2.resize(
                        mask.astype(np.float32),
                        (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    mask_resized = mask.astype(np.float32)

                # Binarize mask
                binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

                # Filter by area
                area = np.count_nonzero(binary_mask)
                if area < min_area:
                    continue

                # Extract contour from binary mask
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    continue

                # Use the largest contour
                contour = max(contours, key=cv2.contourArea)

                # Create filled mask from contour
                grain_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(
                    grain_mask, [contour], -1, 255, thickness=cv2.FILLED
                )

                grains.append(GrainContour(contour=contour, mask=grain_mask))

        # Sort by area descending
        grains.sort(key=lambda g: cv2.contourArea(g.contour), reverse=True)
        return grains

    def detect_in_region(
        self,
        image: np.ndarray,
        region_mask: np.ndarray,
        conf: float = 0.25,
        min_area: int = 50,
    ) -> list[GrainContour]:
        """Detect grains in a specific region of the image using YOLO.

        This is useful for refining detection in regions where traditional
        methods struggle (e.g., touching grains).

        Parameters
        ----------
        image : np.ndarray
            Full input image.
        region_mask : np.ndarray
            Binary mask defining the region of interest.
        conf : float, optional
            Confidence threshold. Default is 0.25.
        min_area : int, optional
            Minimum area threshold. Default is 50.

        Returns
        -------
        list[GrainContour]
            Detected grains in the region.
        """
        if not self.is_available:
            return []

        # Extract bounding box of the region
        ys, xs = np.where(region_mask > 0)
        if len(xs) == 0:
            return []

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # Add padding
        pad = 20
        h_img, w_img = image.shape[:2]
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w_img, x_max + pad)
        y_max = min(h_img, y_max + pad)

        # Crop region
        region = image[y_min:y_max, x_min:x_max]

        # Run YOLO on region
        region_grains = self.detect(region, conf=conf, min_area=min_area)

        # Adjust contours to full image coordinates
        full_grains: list[GrainContour] = []
        for grain in region_grains:
            # Shift contour
            shifted_contour = grain.contour.copy()
            shifted_contour[:, :, 0] += x_min
            shifted_contour[:, :, 1] += y_min

            # Create mask in full image
            full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            # Draw the shifted contour
            cv2.drawContours(full_mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)

            full_grains.append(GrainContour(contour=shifted_contour, mask=full_mask))

        return full_grains


def _is_touching_grain(grain: GrainContour, threshold: float = TOUCHING_THRESHOLD) -> bool:
    """Check if a grain is likely touching/overlapping with others.

    Uses bounding box aspect ratio and convexity as heuristics.

    Parameters
    ----------
    grain : GrainContour
        The grain to check.
    threshold : float
        Aspect ratio threshold. Higher means more likely touching.

    Returns
    -------
    bool
        True if the grain appears to be touching others.
    """
    # Check bounding box aspect ratio
    x, y, w, h = cv2.boundingRect(grain.contour)
    aspect_ratio = max(w, h) / max(min(w, h), 1)

    # Check convexity (touching grains often have lower convexity)
    area = cv2.contourArea(grain.contour)
    hull = cv2.convexHull(grain.contour)
    hull_area = cv2.contourArea(hull)
    convexity = area / max(hull_area, 1)

    # Heuristic: high aspect ratio or low convexity suggests touching
    return aspect_ratio > threshold or convexity < 0.85


def refine_with_yolo(
    image: np.ndarray,
    traditional_grains: list[GrainContour],
    yolo_detector: YOLODetector,
    min_area: int = 50,
) -> list[GrainContour]:
    """Refine traditional detection results with YOLO segmentation.

    This hybrid approach:
    1. Identifies grains that are likely touching/overlapping
    2. Uses YOLO to re-segment those specific regions
    3. Combines YOLO-refined regions with unaffected traditional grains

    Parameters
    ----------
    image : np.ndarray
        Original input image.
    traditional_grains : list[GrainContour]
        Grains detected by traditional method.
    yolo_detector : YOLODetector
        Initialized YOLO detector.
    min_area : int, optional
        Minimum area threshold. Default is 50.

    Returns
    -------
    list[GrainContour]
        Refined grain list. If YOLO is unavailable, returns the original
        traditional grains.
    """
    if not yolo_detector.is_available:
        logger.warning("YOLO not available; returning traditional results.")
        return traditional_grains

    if not traditional_grains:
        return traditional_grains

    # Identify touching grains that need YOLO refinement
    touching_indices = []
    for i, grain in enumerate(traditional_grains):
        if _is_touching_grain(grain):
            touching_indices.append(i)

    if not touching_indices:
        logger.info("No touching grains detected; using traditional results.")
        return traditional_grains

    logger.info(
        "Refining %d/%d grains with YOLO",
        len(touching_indices),
        len(traditional_grains),
    )

    # Collect regions that need refinement
    refined_grains: list[GrainContour] = []
    used_yolo = False

    for i, grain in enumerate(traditional_grains):
        if i not in touching_indices:
            # Non-touching grain: keep traditional result
            refined_grains.append(grain)
            continue

        # Touching grain: try YOLO refinement
        # Create a mask for this grain's region
        region_mask = grain.mask.copy()

        # Run YOLO on this region
        yolo_grains = yolo_detector.detect_in_region(
            image, region_mask, conf=0.25, min_area=min_area
        )

        if len(yolo_grains) > 1:
            # YOLO successfully split this region into multiple grains
            logger.info(
                "YOLO split grain %d into %d grains",
                i,
                len(yolo_grains),
            )
            refined_grains.extend(yolo_grains)
            used_yolo = True
        else:
            # YOLO couldn't split; keep traditional result
            refined_grains.append(grain)

    if used_yolo:
        logger.info(
            "Hybrid detection: %d grains (was %d with traditional)",
            len(refined_grains),
            len(traditional_grains),
        )

    return refined_grains
