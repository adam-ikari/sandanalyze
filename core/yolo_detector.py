"""YOLOv8-seg grain detector.

This module provides YOLOv8 segmentation-based grain detection with
graceful fallback when the model is unavailable.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from core.traditional import GrainContour

logger = logging.getLogger(__name__)


class YOLODetector:
    """Grain detector using YOLOv8-seg.

    Parameters
    ----------
    model_name : str, optional
        Name or path of the YOLO segmentation model to load.
        Default is ``"yolov8n-seg.pt"``.
    """

    def __init__(self, model_name: str = "yolov8n-seg.pt") -> None:
        self._model_name = model_name
        self._model = None
        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load the YOLO model, catching exceptions gracefully."""
        try:
            from ultralytics import YOLO

            self._model = YOLO(self._model_name)
            logger.info("Loaded YOLO model: %s", self._model_name)
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
