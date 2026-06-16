"""Image display panel with zoom, pan, and grain click interaction."""

import cv2
import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent, QMouseEvent, QPainter, QPen, QColor, QFont
from PyQt6.QtWidgets import QWidget, QLabel, QScrollArea, QVBoxLayout, QSizePolicy


class ImagePanel(QWidget):
    """Widget for displaying sand grain images with overlays and interaction.

    Supports zoom (mouse wheel), pan (drag), and clicking on individual grains
    to select them. Displays original image with optional grain contour overlays.
    """

    grain_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: np.ndarray | None = None
        self._grains: list = []
        self._morphologies: list = []
        self._zoom: float = 1.0
        self._zoom_min: float = 0.1
        self._zoom_max: float = 5.0
        self._zoom_factor: float = 1.2

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidget(self._label)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll_area)

        self._label.setMouseTracking(True)
        self._label.installEventFilter(self)

    def set_image(self, image: np.ndarray) -> None:
        """Set and display the original image.

        Args:
            image: Numpy array in BGR (color) or grayscale format.
        """
        self._image = image.copy() if image is not None else None
        self._zoom = 1.0
        self._update_display()

    def set_grains(self, grains: list, morphologies: list = None) -> None:
        """Set detected grains and overlay their contours on the image.

        Args:
            grains: List of grain objects with ``contour`` and optionally ``mask``
                attributes (e.g., :class:`core.traditional.GrainContour`).
            morphologies: Optional list of GrainMorphology objects for Zingg coloring.
        """
        self._grains = grains if grains is not None else []
        self._morphologies = morphologies if morphologies is not None else []
        self._update_display()

    def clear(self) -> None:
        """Clear the display."""
        self._image = None
        self._grains = []
        self._morphologies = []
        self._zoom = 1.0
        self._label.clear()
        self._label.setPixmap(QPixmap())

    def _numpy_to_qpixmap(self, image: np.ndarray) -> QPixmap:
        """Convert a numpy image array to a QPixmap.

        Handles both grayscale and BGR color images.
        """
        if image is None or image.size == 0:
            return QPixmap()

        if len(image.shape) == 2:
            # Grayscale
            height, width = image.shape
            bytes_per_line = width
            qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # BGR -> RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _channels = rgb_image.shape
            bytes_per_line = 3 * width
            qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            return QPixmap()

        # Copy data to ensure Qt owns it
        qimage = qimage.copy()
        return QPixmap.fromImage(qimage)

    def _update_display(self) -> None:
        """Render the current image with grain overlays at the current zoom level."""
        if self._image is None:
            self._label.clear()
            self._label.setPixmap(QPixmap())
            return

        # Start with a copy of the original image
        display_image = self._image.copy()

        # Overlay grain contours and labels if grains are available
        if self._grains:
            # Ensure color image for drawing colored overlays
            if len(display_image.shape) == 2:
                display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

            for idx, grain in enumerate(self._grains):
                contour = getattr(grain, "contour", None)
                if contour is not None and len(contour) > 0:
                    # Determine color based on Zingg classification
                    color = (0, 255, 0)  # Default green
                    if idx < len(self._morphologies):
                        from core.morphology import get_zingg_color
                        color = get_zingg_color(self._morphologies[idx].aspect_ratio)

                    # Draw contour with classification color
                    cv2.drawContours(display_image, [contour], -1, color, 2)

                    # Draw label in white at the contour centroid
                    moments = cv2.moments(contour)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        # Draw a small filled circle for better visibility
                        cv2.circle(display_image, (cx, cy), 3, (255, 255, 255), -1)
                        cv2.putText(
                            display_image,
                            str(idx),
                            (cx - 5, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                        )

            # Draw legend if morphologies are available
            if self._morphologies:
                self._draw_legend(display_image)

        pixmap = self._numpy_to_qpixmap(display_image)

        if pixmap.isNull():
            self._label.clear()
            return

        # Apply zoom scaling
        if self._zoom != 1.0:
            new_width = int(pixmap.width() * self._zoom)
            new_height = int(pixmap.height() * self._zoom)
            pixmap = pixmap.scaled(
                new_width,
                new_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        self._label.setPixmap(pixmap)
        self._label.resize(pixmap.size())

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming."""
        if self._image is None:
            event.ignore()
            return

        delta = event.angleDelta().y()
        if delta > 0:
            new_zoom = self._zoom * self._zoom_factor
        elif delta < 0:
            new_zoom = self._zoom / self._zoom_factor
        else:
            event.ignore()
            return

        new_zoom = max(self._zoom_min, min(self._zoom_max, new_zoom))
        if new_zoom != self._zoom:
            self._zoom = new_zoom
            self._update_display()
        event.accept()

    def eventFilter(self, obj, event) -> bool:
        """Filter events on the label to handle mouse clicks for grain selection."""
        if obj is self._label and isinstance(event, QMouseEvent):
            if event.type() == QMouseEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._handle_grain_click(event)
                return True
        return super().eventFilter(obj, event)

    def _handle_grain_click(self, event: QMouseEvent) -> None:
        """Map click position to image coordinates and find which grain was clicked."""
        if self._image is None or not self._grains:
            return

        # Click position relative to the label
        click_x = event.pos().x()
        click_y = event.pos().y()

        pixmap = self._label.pixmap()
        if pixmap is None or pixmap.isNull():
            return

        # Get the actual displayed pixmap rect within the label
        label_w = self._label.width()
        label_h = self._label.height()
        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()

        # Calculate offset if the pixmap is centered
        offset_x = (label_w - pixmap_w) // 2
        offset_y = (label_h - pixmap_h) // 2

        # Map to pixmap coordinates
        pixmap_x = click_x - offset_x
        pixmap_y = click_y - offset_y

        # Check if click is within the pixmap
        if pixmap_x < 0 or pixmap_y < 0 or pixmap_x >= pixmap_w or pixmap_y >= pixmap_h:
            return

        # Map pixmap coordinates to original image coordinates
        img_h, img_w = self._image.shape[:2]
        orig_x = int(pixmap_x / self._zoom)
        orig_y = int(pixmap_y / self._zoom)

        # Clamp to image bounds
        orig_x = max(0, min(img_w - 1, orig_x))
        orig_y = max(0, min(img_h - 1, orig_y))

        # Find which grain was clicked by checking mask containment
        for idx, grain in enumerate(self._grains):
            mask = getattr(grain, "mask", None)
            if mask is not None and mask.size > 0:
                if 0 <= orig_y < mask.shape[0] and 0 <= orig_x < mask.shape[1]:
                    if mask[orig_y, orig_x] > 0:
                        self.grain_clicked.emit(idx)
                        return

            # Fallback: check if point is inside contour
            contour = getattr(grain, "contour", None)
            if contour is not None and len(contour) > 0:
                dist = cv2.pointPolygonTest(contour, (float(orig_x), float(orig_y)), False)
                if dist >= 0:
                    self.grain_clicked.emit(idx)
                    return

    def _draw_legend(self, image: np.ndarray) -> None:
        """Draw Zingg classification legend on the image.

        Args:
            image: The image to draw on (modified in-place).
        """
        from core.morphology import ZINGG_COLORS

        h, w = image.shape[:2]
        legend_x = w - 140
        legend_y = h - 80
        item_height = 20

        # Background rectangle
        cv2.rectangle(image, (legend_x - 10, legend_y - 25),
                     (legend_x + 130, legend_y + len(ZINGG_COLORS) * item_height + 5),
                     (40, 40, 40), -1)
        cv2.rectangle(image, (legend_x - 10, legend_y - 25),
                     (legend_x + 130, legend_y + len(ZINGG_COLORS) * item_height + 5),
                     (200, 200, 200), 1)

        # Title
        cv2.putText(image, "Zingg分类", (legend_x, legend_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Legend items
        for i, (label, color) in enumerate(ZINGG_COLORS.items()):
            y = legend_y + i * item_height + 10
            # Color box
            cv2.rectangle(image, (legend_x, y - 10), (legend_x + 15, y + 5), color, -1)
            # Label
            cv2.putText(image, label, (legend_x + 20, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
